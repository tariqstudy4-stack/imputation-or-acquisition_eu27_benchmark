"""Sensitivity sweep: does the pedigree calibration failure depend on the specific
GSD assumption (2.0 / 1.8), or does it hold across the plausible pedigree range?
Sweeps GSD_A = GSD_B over {1.2, 1.5, 2.0, 2.5, 3.0} across the full grid of
missingness patterns (MCAR, FLOW, BLOCK) and rates (5/10/20/40%) on the EU-27
model, baseline method only.
"""
import numpy as np, pandas as pd, json, time

import os
def _find(f):
    for p in (f, os.path.join('..','data_build',f), os.path.join('data_build',f)):
        if os.path.exists(p): return p
    raise FileNotFoundError(f)
A0 = np.load(_find('A_eu.npy')); B0 = np.load(_find('B_eu.npy')); C = np.load(_find('C_eu.npy'))
meta = json.load(open(_find('meta_eu.json'))); nace = meta['nace']; n = A0.shape[1]
K = C.shape[0]
grp = np.array([c[0] for c in nace]); groups = {g: np.where(grp==g)[0] for g in np.unique(grp)}
targets = np.array([i for i,c in enumerate(nace) if c[0]=='C'])
I_n = np.eye(n); E_T = I_n[:, targets]
H_true = C @ (B0 @ np.linalg.solve(I_n - A0, E_T))
nzA = np.argwhere((A0!=0) & ~np.eye(n,dtype=bool))
nzB_rows = np.where((B0!=0).any(axis=1))[0]

def sib_cols(j, blocked):
    s = [t for t in groups[grp[j]] if t!=j and t not in blocked]
    return s if s else [t for t in range(n) if t!=j and t not in blocked]

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
        # NOTE: this script never had a real FLOW branch before -- 'FLOW' silently fell
        # into the `else` (BLOCK-style column masking) below, since the only distinction
        # made was MCAR vs. "everything else". This was latent and harmless while the
        # sweep never included FLOW; it was caught only once FLOW was added to the sweep
        # (see CHANGELOG.md) and produced masks nothing like the FLOW pattern used in
        # experiment_final_eu.py. Mirrors that script's FLOW branch exactly.
        fl = rng.choice(nzB_rows, max(1,int(rate*len(nzB_rows))), replace=False)
        for f in fl:
            cols = rng.choice(n, n//2, replace=False); mB[f, cols] = True
    else:
        cand = np.setdiff1d(np.arange(n), targets)
        cols = rng.choice(cand, min(max(1,int(rate*n)), len(cand)), replace=False)
        mA[:, cols] = True; mB[:, cols] = True
    return mA, mB

# NOTE (masked-aware fix, see CHANGELOG.md): identical fix to experiment_final_eu.py --
# average only over sibling entries not themselves masked at that row, rather than
# reading the true A0/B0 values of co-masked siblings.
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

def impacts(Ah, Bh):
    cs = Ah.sum(0); bad = cs>=0.98
    if bad.any(): Ah=Ah.copy(); Ah[:,bad]*=(0.95/cs[bad])
    return C @ (Bh @ np.linalg.solve(I_n-Ah, E_T))

rows=[]
# NOTE: earlier version swept only MCAR/BLOCK at 20%/40% (4 of 12 pattern-rate
# conditions), omitting FLOW entirely -- including FLOW@10%, the single condition
# where the pedigree baseline's coverage comes closest to nominal anywhere in the
# main grid (see README/manuscript Section 4.3). The "not a GSD artifact" claim
# needs the full grid to be evidenced rather than assumed -- see CHANGELOG.md.
for pattern in ('MCAR', 'FLOW', 'BLOCK'):
  for rate in (0.05, 0.10, 0.20, 0.40):
    for seed in [0,1,2]:
        mA, mB = make_masks(pattern, rate, seed)
        Ah, Bh = repair_baseline(mA, mB)
        iA, iB = np.argwhere(mA), np.argwhere(mB)
        for gsd in [1.2, 1.5, 2.0, 2.5, 3.0]:
            sig = np.log(gsd)
            rng = np.random.default_rng(seed*7919+int(rate*100)+int(gsd*10))
            S = np.empty((150, K, len(targets)))
            for m in range(150):
                Am, Bm = Ah.copy(), Bh.copy()
                if len(iA): Am[iA[:,0],iA[:,1]] = Ah[iA[:,0],iA[:,1]]*np.exp(rng.normal(0,sig,len(iA)))
                if len(iB): Bm[iB[:,0],iB[:,1]] = Bh[iB[:,0],iB[:,1]]*np.exp(rng.normal(0,sig,len(iB)))
                S[m] = impacts(Am,Bm)
            lo,hi = np.quantile(S,0.025,0), np.quantile(S,0.975,0)
            cov = ((H_true>=lo)&(H_true<=hi)).mean()
            rows.append(dict(pattern=pattern, rate=rate, seed=seed, gsd=gsd, coverage=cov))

res = pd.DataFrame(rows)
summ = res.groupby(['pattern','rate','gsd'])['coverage'].mean().reset_index()
out_dir = os.path.join('..', 'results') if os.path.isdir(os.path.join('..', 'results')) else '.'
out_path = os.path.join(out_dir, 'gsd_sensitivity_eu.csv')
summ.to_csv(out_path, index=False)
piv = summ.pivot_table(index=['pattern','rate'], columns='gsd', values='coverage')*100
print(piv.round(0).to_string())
print(f'\nwritten -> {out_path}')
