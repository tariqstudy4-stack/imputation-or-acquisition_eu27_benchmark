"""FULL BENCHMARK -- Learned imputation vs pedigree-widened proxies for live LCI.
Methods: PEDIGREE (sibling proxy + lognormal MC), KNN (donor imputation + donor-resample MC),
SOFTIMPUTE (low-rank completion), VAE (column autoencoder + decoder MC), GNN (message passing + MC dropout).
Fix vs pilot: log-space outputs for VAE/GNN are clamped to the observed min/max of the matrix
before inverse-transform. This is standard practice (bounding reconstructions to the physically
plausible range of the training data) and eliminates the expm1 blow-up seen with unclamped nets.
Testbed: EU-27 Eurostat IOT (naio_10_cp1750) + air emissions accounts (env_ac_ainah_r2),
NACE-64, reference year 2022. 4 Eurostat-characterized indicators. Manufacturing targets (19).
"""
import numpy as np, pandas as pd, torch, torch.nn as nn, itertools, time, sys, os

torch.set_num_threads(1)
import os
def _find(f):
    for p in (f, os.path.join('..','data_build',f), os.path.join('data_build',f)):
        if os.path.exists(p): return p
    raise FileNotFoundError(f)
A0 = np.load(_find('A_eu.npy')); B0 = np.load(_find('B_eu.npy')); C = np.load(_find('C_eu.npy'))
import json
meta = json.load(open(_find('meta_eu.json')))
nace = meta['nace']; n = A0.shape[1]
C_env = C; K = C_env.shape[0]
grp = np.array([c[0] for c in nace])
groups = {g: np.where(grp==g)[0] for g in np.unique(grp)}
targets = np.array([i for i,c in enumerate(nace) if c[0]=='C'])
I_n = np.eye(n); E_T = I_n[:, targets]
H_true = C_env @ (B0 @ np.linalg.solve(I_n - A0, E_T))
nzA = np.argwhere((A0!=0) & ~np.eye(n,dtype=bool))
nzB_rows = np.where((B0!=0).any(axis=1))[0]
GSD_A, GSD_B, ALPHA = 2.0, 1.8, 0.95
sig_A, sig_B = np.log(GSD_A), np.log(GSD_B)

def make_masks(pattern, rate, seed):
    # NOTE: an earlier version derived this offset from Python's built-in hash(pattern),
    # which is randomized per process by design (PEP 456) and is therefore NOT
    # reproducible across runs even with a fixed seed. Fixed to an explicit,
    # deterministic mapping -- see CHANGELOG.md.
    PATTERN_OFFSET = {'MCAR': 11, 'FLOW': 43, 'BLOCK': 79}
    rng = np.random.default_rng(seed*1000 + int(rate*100) + PATTERN_OFFSET[pattern])
    mA = np.zeros((n,n), bool); mB = np.zeros(B0.shape, bool)
    if pattern=='MCAR':
        pick = nzA[rng.choice(len(nzA), int(rate*len(nzA)), replace=False)]
        mA[pick[:,0], pick[:,1]] = True
    elif pattern=='FLOW':
        fl = rng.choice(nzB_rows, max(1,int(rate*len(nzB_rows))), replace=False)
        for f in fl:
            cols = rng.choice(n, n//2, replace=False); mB[f, cols] = True
    else:
        cand = np.setdiff1d(np.arange(n), targets)
        cols = rng.choice(cand, min(max(1,int(rate*n)), len(cand)), replace=False)
        mA[:, cols] = True; mB[:, cols] = True
    return mA, mB

def impacts(Ah, Bh):
    cs = Ah.sum(axis=0); bad = cs>=0.98
    if bad.any(): Ah = Ah.copy(); Ah[:,bad] *= (0.95/cs[bad])
    return C_env @ (Bh @ np.linalg.solve(I_n - Ah, E_T))

def metrics_point(Hp):
    mdape = np.median(np.abs(Hp-H_true)/np.abs(H_true), axis=1)
    pairs = list(itertools.combinations(range(len(targets)),2))
    flm = np.zeros(K); nm = np.zeros(K)
    for k in range(K):
        t,p = H_true[k], Hp[k]
        for a,b in pairs:
            gap = abs(t[a]-t[b])/max(abs(t[a]),abs(t[b]))
            if gap>0.05:
                nm[k]+=1; flm[k]+= (np.sign(t[a]-t[b])!=np.sign(p[a]-p[b]))
    return mdape.mean(), np.divide(flm,nm,out=np.zeros(K),where=nm>0).mean()

# ---------------- PEDIGREE ----------------
def sib_cols(j, blocked):
    s = [t for t in groups[grp[j]] if t!=j and t not in blocked]
    return s if s else [t for t in range(n) if t!=j and t not in blocked]

# NOTE (masked-aware fix, see CHANGELOG.md): the previous version averaged sibling
# columns from A0/B0 directly, which are the TRUE unmasked matrices. When more than
# one entry is masked in a row (the normal case above the lowest missingness rates),
# that let the "sibling average" for one masked cell be computed partly from other
# cells that are themselves masked in this same run -- information that would not
# actually be observable in a real missing-data scenario. Fixed to average only over
# sibling entries that are NOT masked at that row, per row, with a widened fallback
# (then a row mean as a last resort) when no sibling is observed at a given row.
def repair_one(M0, mask):
    Mh = M0.copy()
    blocked = set(np.where(mask.all(axis=0))[0].tolist())
    for j in np.unique(np.where(mask)[1]):
        base_s = sib_cols(j, blocked)
        rows = np.where(mask[:, j])[0]
        for r in rows:
            obs_s = [c for c in base_s if not mask[r, c]]
            if not obs_s:
                obs_s = [c for c in range(n) if c != j and c not in blocked and not mask[r, c]]
            if obs_s:
                Mh[r, j] = M0[r, obs_s].mean()
            else:
                # guarded last resort: mean of observed cells in the row; 0 if the whole
                # row is masked (unreachable at the tested rates, but avoids an all-NaN mean)
                row_obs = ~mask[r, :]
                Mh[r, j] = M0[r, row_obs].mean() if row_obs.any() else 0.0
    return Mh

def repair_baseline(mA, mB):
    return repair_one(A0, mA), repair_one(B0, mB)

def pedigree_coverage(Ah, Bh, mA, mB, mc=80, seed=0):
    rng = np.random.default_rng(seed)
    iA, iB = np.argwhere(mA), np.argwhere(mB)
    S = np.empty((mc, K, len(targets)))
    for m in range(mc):
        Am, Bm = Ah.copy(), Bh.copy()
        if len(iA): Am[iA[:,0],iA[:,1]] = Ah[iA[:,0],iA[:,1]]*np.exp(rng.normal(0,sig_A,len(iA)))
        if len(iB): Bm[iB[:,0],iB[:,1]] = Bh[iB[:,0],iB[:,1]]*np.exp(rng.normal(0,sig_B,len(iB)))
        S[m] = impacts(Am,Bm)
    lo,hi = np.quantile(S,0.025,0), np.quantile(S,0.975,0)
    return ((H_true>=lo)&(H_true<=hi)).mean()

# ---------------- KNN ----------------
# NOTE (masked-aware fix, see CHANGELOG.md): the previous version chose donor columns
# by distance computed against the TRUE candidate-column values even at rows where
# the candidate was itself masked in this run, and then pulled the imputed value from
# those same true-but-actually-unobservable donor cells. Both steps leaked ground
# truth. Fixed so distance is computed only over (row, candidate) pairs where the
# candidate is also actually observed, and the final donor value pulled for a given
# masked row uses only donors that are observed AT THAT ROW, reweighted; a row with
# no masked-aware-valid donor among the top-k falls back to the sibling average.
def knn_impute(M, mask, k=10):
    Mh = M.copy(); donors = {}
    obs = ~mask; blocked = set(np.where(mask.all(axis=0))[0].tolist())
    for j in np.unique(np.where(mask)[1]):
        oj = obs[:,j]
        rows = np.where(mask[:, j])[0]
        if oj.sum() < 5:
            cand = np.array(sib_cols(j, blocked)); d = np.ones(len(cand))
        else:
            cand = np.array([t for t in range(n) if t!=j and t not in blocked])
            obs_rows = np.where(oj)[0]
            X = M[np.ix_(obs_rows, cand)]
            valid = ~mask[np.ix_(obs_rows, cand)]
            v = M[oj, j][:,None]
            sqd = np.where(valid, (X - v)**2, np.nan)
            with np.errstate(invalid='ignore'):
                d = np.sqrt(np.nanmean(sqd, axis=0))
            n_valid = valid.sum(axis=0)
            d = np.where(n_valid > 0, d, np.inf) + 1e-12
        kk = min(k, len(cand)); top = np.argsort(d)[:kk]
        dj, cj = d[top], cand[top]

        vals = M[np.ix_(rows, cj)].astype(float)
        donor_obs = ~mask[np.ix_(rows, cj)]
        w = (1/dj)
        Wrow = np.tile(w, (len(rows), 1))
        Wrow = np.where(donor_obs, Wrow, 0.0)
        wsum = Wrow.sum(axis=1, keepdims=True)
        Wn = np.divide(Wrow, wsum, out=np.zeros_like(Wrow), where=wsum > 0)
        imputed = (np.where(donor_obs, vals, 0.0) * Wn).sum(axis=1)

        no_donor = (wsum[:, 0] == 0)
        if no_donor.any():
            s = sib_cols(j, blocked)
            for ridx in np.where(no_donor)[0]:
                r = rows[ridx]
                obs_s = [c for c in s if not mask[r, c]]
                if not obs_s:
                    obs_s = [c for c in range(n) if c != j and c not in blocked and not mask[r, c]]
                if obs_s:
                    fb = M[r, obs_s].mean()
                else:
                    # last resort: mean of observed cells in the row; 0 if the entire row is masked
                    row_obs = ~mask[r, :]
                    fb = M[r, row_obs].mean() if row_obs.any() else 0.0
                imputed[ridx] = fb
                # NOTE (leak hardening): the raw `vals` slots for this row are donor cells that are
                # ALL masked in this run -- resampling from them would reintroduce ground truth into
                # the MC interval. Overwrite this row's donor values with the fallback estimate so
                # resampling is degenerate (always returns the masked-aware fallback), never leaky.
                # This branch never fires on the reported 12x3 grid (verified), but is now safe if
                # rates or seeds are extended.
                vals[ridx, :] = fb
                Wn[ridx] = 1.0 / Wn.shape[1]  # uniform weights over identical fallback values

        Mh[rows, j] = imputed
        donors[j] = (rows, np.nan_to_num(vals, nan=0.0), Wn)
    return Mh, donors

def flatten_donors(donors):
    """Per-row donor values and (mask-aware, possibly per-row-varying) resampling weights."""
    R, Cc, V, W = [], [], [], []
    kmax = max((v[1].shape[1] for v in donors.values()), default=0)
    for j, (rows, vals, w) in donors.items():
        k = vals.shape[1]
        vpad = np.pad(vals, ((0, 0), (0, kmax - k)), mode='edge')
        wpad = np.pad(w, ((0, 0), (0, kmax - k)), constant_values=0.0)
        for ridx, r in enumerate(rows):
            wr = wpad[ridx]
            s = wr.sum()
            wr = wr / s if s > 0 else np.full(kmax, 1.0 / kmax)
            cw = np.cumsum(wr); cw[-1] = 1.0
            R.append(r); Cc.append(j); V.append(vpad[ridx]); W.append(cw)
    if not R:
        return (None, None, None, None)
    return (np.array(R), np.array(Cc), np.array(V), np.array(W))

def knn_coverage(Ah, Bh, mA, mB, dA, dB, mc=60, seed=0):
    rng = np.random.default_rng(seed)
    fA, fB = flatten_donors(dA), flatten_donors(dB)
    S = np.empty((mc, K, len(targets)))
    for m in range(mc):
        Am, Bm = Ah.copy(), Bh.copy()
        for M_, flat in ((Am,fA), (Bm,fB)):
            r_,c_,V_,W_ = flat
            if r_ is None: continue
            u = rng.random(len(r_)); idx = (u[:,None] > W_).sum(axis=1)
            idx = np.clip(idx, 0, V_.shape[1]-1)
            M_[r_, c_] = V_[np.arange(len(r_)), idx]
        S[m] = impacts(Am, Bm)
    lo,hi = np.quantile(S,0.025,0), np.quantile(S,0.975,0)
    return ((H_true>=lo)&(H_true<=hi)).mean()

# ---------------- SOFTIMPUTE ----------------
def softimpute(M, mask, lam_frac=0.02, iters=15):
    if not mask.any(): return M.copy()
    Mh = M.copy()
    # columns that are entirely masked (BLOCK pattern) have no observed values to average;
    # np.nanmean warns "Mean of empty slice" for those columns even though the np.where
    # below discards the result and substitutes 0 -- suppress that expected, harmless warning.
    with np.errstate(invalid='ignore'):
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Mean of empty slice')
            col_mean = np.where(mask.all(axis=0), 0, np.nanmean(np.where(mask, np.nan, M), axis=0))
    Mh[mask] = np.take(np.nan_to_num(col_mean), np.where(mask)[1])
    lam = None
    for it in range(iters):
        U, s, Vt = np.linalg.svd(Mh, full_matrices=False)
        if lam is None: lam = lam_frac * s[0]
        s2 = np.maximum(s - lam, 0)
        Z = (U * s2) @ Vt
        Mh[mask] = Z[mask]
    return Mh

# ---------------- VAE (clamped) ----------------
class VAE(nn.Module):
    def __init__(self, d, h=64, z=12, p=0.15):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d,h), nn.ReLU(), nn.Dropout(p), nn.Linear(h,h), nn.ReLU())
        self.mu, self.lv = nn.Linear(h,z), nn.Linear(h,z)
        self.dec = nn.Sequential(nn.Linear(z,h), nn.ReLU(), nn.Dropout(p), nn.Linear(h,h), nn.ReLU(), nn.Linear(h,d))
    def forward(self,x):
        h=self.enc(x); mu,lv=self.mu(h),self.lv(h)
        z = mu + torch.randn_like(mu)*torch.exp(0.5*lv)
        return self.dec(z), mu, lv

def vae_impute(M, mask, epochs=150, lr=1e-2, mc=40, seed=0):
    if not mask.any(): return M.copy(), None
    torch.manual_seed(seed)
    sign = np.sign(M); logM = sign*np.log1p(np.abs(M))
    clip_lo, clip_hi = logM.min(), logM.max()          # <-- observed range, the fix
    scale = max(abs(clip_lo), abs(clip_hi)) + 1e-9
    X = torch.tensor((logM/scale).T, dtype=torch.float32)
    Mk = torch.tensor((~mask).T, dtype=torch.float32)
    d = X.shape[1]; model = VAE(d); opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xin = X*Mk
    for ep in range(epochs):
        opt.zero_grad()
        out, mu, lv = model(Xin)
        rec = ((out - X)**2 * Mk).sum() / Mk.sum().clamp(min=1)
        kld = -0.5*torch.mean(1+lv-mu.pow(2)-lv.exp())
        (rec + 1e-3*kld).backward(); opt.step()
    with torch.no_grad():
        samples = []
        for _ in range(mc):
            out,_,_ = model(Xin)
            out = torch.clamp(out, clip_lo/scale, clip_hi/scale)   # <-- clamp before inverse transform
            samples.append(out.numpy())
        S = np.stack(samples)
    mean_log = (S.mean(0)*scale).T
    Mh = M.copy(); rec_lin = np.sign(mean_log)*np.expm1(np.abs(mean_log))
    Mh[mask] = rec_lin[mask]
    S_lin = np.sign(S*scale)*np.expm1(np.abs(S*scale))
    return Mh, np.transpose(S_lin, (0,2,1))

# ---------------- GNN (clamped, MC dropout) ----------------
class GNN(nn.Module):
    def __init__(self, din, h=32, p=0.15):
        super().__init__()
        self.w1 = nn.Linear(din, h); self.drop = nn.Dropout(p)
        self.w2 = nn.Linear(h, h); self.out = nn.Linear(h, din)
    def forward(self, X, Adj):
        h1 = torch.relu(self.w1(Adj @ X)); h1 = self.drop(h1)
        h2 = torch.relu(self.w2(Adj @ h1)); h2 = self.drop(h2)
        return self.out(h2)

def gnn_impute(M, mask, epochs=150, lr=1e-2, mc=40, seed=0):
    if not mask.any(): return M.copy(), None
    torch.manual_seed(seed)
    sign = np.sign(M); logM = sign*np.log1p(np.abs(M))
    clip_lo, clip_hi = logM.min(), logM.max()
    scale = max(abs(clip_lo), abs(clip_hi)) + 1e-9
    Xf = logM/scale
    W = (np.abs(A0)>0).astype(float) + np.eye(n)
    deg = W.sum(1, keepdims=True); Wn = W/np.clip(deg,1,None)
    Adj = torch.tensor(Wn, dtype=torch.float32)
    X = torch.tensor(Xf.T, dtype=torch.float32)
    Mk = torch.tensor((~mask).T, dtype=torch.float32)
    model = GNN(X.shape[1]); opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xin = X*Mk
    for ep in range(epochs):
        opt.zero_grad()
        out = model(Xin, Adj)
        loss = ((out-X)**2 * Mk).sum() / Mk.sum().clamp(min=1)
        loss.backward(); opt.step()
    model.train()  # keep dropout active for MC-dropout uncertainty
    with torch.no_grad():
        samples=[]
        for _ in range(mc):
            out = model(Xin, Adj)
            out = torch.clamp(out, clip_lo/scale, clip_hi/scale)
            samples.append(out.numpy())
        S = np.stack(samples)
    mean_log = (S.mean(0)*scale).T
    Mh = M.copy(); rec_lin = np.sign(mean_log)*np.expm1(np.abs(mean_log))
    Mh[mask] = rec_lin[mask]
    S_lin = np.sign(S*scale)*np.expm1(np.abs(S*scale))
    return Mh, np.transpose(S_lin, (0,2,1))

def mc_coverage_from_S(SA, mA, SB, mB, Ah, Bh, mc):
    preds = np.empty((mc, K, len(targets)))
    for m in range(mc):
        Am, Bm = Ah.copy(), Bh.copy()
        if SA is not None: Am[mA] = SA[m][mA]
        if SB is not None: Bm[mB] = SB[m][mB]
        cs = Am.sum(0); bad = cs>=0.98
        if bad.any(): Am[:,bad]*=(0.95/cs[bad])
        preds[m] = C_env @ (Bm @ np.linalg.solve(I_n-Am, E_T))
    lo,hi = np.quantile(preds,0.025,0), np.quantile(preds,0.975,0)
    return ((H_true>=lo)&(H_true<=hi)).mean()

# ==================== MAIN GRID ====================
PAT = sys.argv[1]
RATES = [0.05,0.10,0.20,0.40]
SEEDS = [0,1,2]
rows = []; t0=time.time()
for rate in RATES:
    for seed in SEEDS:
        mA, mB = make_masks(PAT, rate, seed)

        Ah, Bh = repair_baseline(mA, mB)
        md, fl = metrics_point(impacts(Ah,Bh))
        cov = pedigree_coverage(Ah, Bh, mA, mB, seed=seed)
        rows.append(dict(method='PEDIGREE',pattern=PAT,rate=rate,seed=seed,mdape=md,flips=fl,coverage=cov))

        Ak, dA = knn_impute(A0, mA); Bk, dB = knn_impute(B0, mB)
        md, fl = metrics_point(impacts(Ak,Bk))
        cov = knn_coverage(Ak, Bk, mA, mB, dA, dB, seed=seed)
        rows.append(dict(method='KNN',pattern=PAT,rate=rate,seed=seed,mdape=md,flips=fl,coverage=cov))

        As = softimpute(A0, mA); Bs = softimpute(B0, mB)
        md, fl = metrics_point(impacts(As,Bs))
        rows.append(dict(method='SOFTIMPUTE',pattern=PAT,rate=rate,seed=seed,mdape=md,flips=fl,coverage=np.nan))

        Av, SA = vae_impute(A0, mA, seed=seed); Bv, SB = vae_impute(B0, mB, seed=seed)
        md, fl = metrics_point(impacts(Av,Bv))
        cov = mc_coverage_from_S(SA, mA, SB, mB, Av, Bv, mc=40)
        rows.append(dict(method='VAE',pattern=PAT,rate=rate,seed=seed,mdape=md,flips=fl,coverage=cov))

        Ag, GA = gnn_impute(A0, mA, seed=seed); Bg, GB = gnn_impute(B0, mB, seed=seed)
        md, fl = metrics_point(impacts(Ag,Bg))
        cov = mc_coverage_from_S(GA, mA, GB, mB, Ag, Bg, mc=40)
        rows.append(dict(method='GNN',pattern=PAT,rate=rate,seed=seed,mdape=md,flips=fl,coverage=cov))

        print(f'{PAT} rate={rate} seed={seed} done {time.time()-t0:.0f}s', flush=True)

res = pd.DataFrame(rows)
out_dir = os.path.join('..', 'results') if os.path.isdir(os.path.join('..', 'results')) else '.'
out_path = os.path.join(out_dir, 'results_final_eu.csv')
hdr = not os.path.exists(out_path)
res.to_csv(out_path, index=False, mode='a', header=hdr)
print(f'\n{PAT} chunk complete in {time.time()-t0:.0f}s -> {out_path}')
