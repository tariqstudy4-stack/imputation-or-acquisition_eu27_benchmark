# Lognormal pedigree factors (GSD 2.0 / 1.8, independent, substituted entries only, 300 draws) applied around
# different point estimates: zero-error centre (true value), row mean, SoftImpute, VAE, GNN.
# usage: python center_test.py <pattern> <seed0> <seed1>   (run from code/)
import sys,time,numpy as np,pandas as pd
src=open('experiment_final_eu.py').read().split('# ==================== MAIN GRID')[0]
g={'__name__':'x'}; exec(compile(src,'e','exec'),g)
A0,B0,K,targets,H_true,impacts,make_masks,metrics_point=[g[k] for k in ('A0','B0','K','targets','H_true','impacts','make_masks','metrics_point')]
sA,sB=np.log(2.0),np.log(1.8); MC=300
def ln_S(Ah,Bh,mA,mB,seed):
    rng=np.random.default_rng(seed); iA,iB=np.argwhere(mA),np.argwhere(mB); S=np.empty((MC,K,len(targets)))
    for m in range(MC):
        Am,Bm=Ah.copy(),Bh.copy()
        if len(iA): Am[iA[:,0],iA[:,1]]=Ah[iA[:,0],iA[:,1]]*np.exp(rng.normal(0,sA,len(iA)))
        if len(iB): Bm[iB[:,0],iB[:,1]]=Bh[iB[:,0],iB[:,1]]*np.exp(rng.normal(0,sB,len(iB)))
        S[m]=impacts(Am,Bm)
    return S
def rowmean(M0,mask):
    Mh=M0.copy()
    for r in range(M0.shape[0]):
        c=np.where(mask[r])[0]
        if len(c):
            o=M0[r,~mask[r]]; Mh[r,c]=o.mean() if len(o) else 0.0
    return Mh
def summ(S):
    lo,hi=np.quantile(S,0.025,0),np.quantile(S,0.975,0)
    return ((H_true>=lo)&(H_true<=hi)).mean(), float(np.median((hi-lo)/np.abs(H_true)))
pat=sys.argv[1]; s0,s1=int(sys.argv[2]),int(sys.argv[3]); out=f'center_{pat}_{s0}_{s1}.csv'
rows=[]; t0=time.time()
for seed in range(s0,s1):
    for rate in (0.05,0.10,0.20,0.40):
        mA,mB=make_masks(pat,rate,seed)
        C={'TRUE':(A0.copy(),B0.copy()),'ROWMEAN':(rowmean(A0,mA),rowmean(B0,mB)),
           'SOFTIMPUTE':(g['softimpute'](A0,mA),g['softimpute'](B0,mB))}
        C['VAE']=(g['vae_impute'](A0,mA,mc=150,seed=seed)[0],g['vae_impute'](B0,mB,mc=150,seed=seed)[0])
        C['GNN']=(g['gnn_impute'](A0,mA,mc=150,seed=seed)[0],g['gnn_impute'](B0,mB,mc=150,seed=seed)[0])
        for nm,(Ah,Bh) in C.items():
            md,fl=metrics_point(impacts(Ah,Bh)); cov,w=summ(ln_S(Ah,Bh,mA,mB,seed))
            rows.append(dict(centre=nm,pattern=pat,rate=rate,seed=seed,mdape=md,flips=fl,coverage=cov,width=w))
    if (seed+1)%5==0 or seed==s1-1:
        pd.DataFrame(rows).to_csv(out,index=False); print(pat,seed,f'{time.time()-t0:.0f}s',flush=True)
