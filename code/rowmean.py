import sys,types,numpy as np,pandas as pd
src=open('experiment_final_eu.py').read().split('# ==================== MAIN GRID')[0]
src=src.replace("os.path.join('..','data_build',f)","os.path.join('..','data_build',f)")
t=types.ModuleType('torch'); t.set_num_threads=lambda *a:None; nn=types.ModuleType('torch.nn'); nn.Module=object; t.nn=nn
sys.modules['torch']=t; sys.modules['torch.nn']=nn
g={'__name__':'x'}; exec(compile(src,'e','exec'),g)
A0,B0,make_masks,impacts,metrics_point,n=[g[k] for k in ('A0','B0','make_masks','impacts','metrics_point','n')]
cand=np.setdiff1d(np.arange(n),g['targets'])
def rowmean(M0,mask):
    Mh=M0.copy()
    for r in range(M0.shape[0]):
        c=np.where(mask[r])[0]
        if len(c):
            o=M0[r,~mask[r]]; Mh[r,c]=o.mean() if len(o) else 0.0
    return Mh
rows=[]
for pat in ('MCAR','FLOW','BLOCK'):
    for rate in (0.05,0.10,0.20,0.40):
        for seed in range(500):
            mA,mB=make_masks(pat,rate,seed)
            Ah,Bh=rowmean(A0,mA),rowmean(B0,mB); md,fl=metrics_point(impacts(Ah,Bh))
            cs=Ah.sum(0); rows.append(dict(method='ROWMEAN',pattern=pat,rate=rate,seed=seed,mdape=md,flips=fl,coverage=np.nan,rescaled=int((cs>=0.98).any())))
    print(pat,flush=True)
pd.DataFrame(rows).to_csv('rowmean_500.csv',index=False)
# dose: row mean
rows=[]
for seed in range(200):
    for k in (1,2,3,6,12,25):
        rng=np.random.default_rng(seed*1000+k+5000); cols=rng.choice(cand,k,replace=False)
        mA=np.zeros((n,n),bool); mB=np.zeros(B0.shape,bool); mA[:,cols]=True; mB[:,cols]=True
        Ah,Bh=rowmean(A0,mA),rowmean(B0,mB); md,fl=metrics_point(impacts(Ah,Bh))
        rows.append(dict(k=k,seed=seed,method='ROWMEAN',mdape=md,flips=fl,coverage=np.nan))
pd.DataFrame(rows).to_csv('dose_ROWMEAN_0_200.csv',index=False); print('dose done',flush=True)
