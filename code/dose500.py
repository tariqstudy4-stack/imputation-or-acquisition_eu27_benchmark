import sys,os,time,numpy as np,pandas as pd
src=open('experiment_final_eu.py').read().split('# ==================== MAIN GRID')[0]
src=src.replace("os.path.join('..','data_build',f)","os.path.join('..','data_build',f)")
g={'__name__':'x'}; exec(compile(src,'e','exec'),g)
import numpy as np
n=g['n'];targets=g['targets'];A0=g['A0'];B0=g['B0']
meth=sys.argv[1].split(','); s0,s1=int(sys.argv[2]),int(sys.argv[3]); out=f'dose_{"-".join(meth)}_{s0}_{s1}.csv'
cand=np.setdiff1d(np.arange(n),targets); imp=g['impacts']; mp=g['metrics_point']
hdr=not os.path.exists(out); t0=time.time()
for seed in range(s0,s1):
    rows=[]
    for k in (1,2,3,6,12,25):
        rng=np.random.default_rng(seed*1000+k+5000); cols=rng.choice(cand,k,replace=False)
        mA=np.zeros((n,n),bool); mB=np.zeros(B0.shape,bool); mA[:,cols]=True; mB[:,cols]=True
        base=dict(k=k,seed=seed)
        if 'PEDIGREE' in meth:
            Ah,Bh=g['repair_baseline'](mA,mB); md,fl=mp(imp(Ah,Bh)); rows.append(dict(base,method='PEDIGREE',mdape=md,flips=fl,coverage=g['pedigree_coverage'](Ah,Bh,mA,mB,mc=300,seed=seed)))
        if 'KNN' in meth:
            Ak,dA=g['knn_impute'](A0,mA); Bk,dB=g['knn_impute'](B0,mB); md,fl=mp(imp(Ak,Bk)); rows.append(dict(base,method='KNN',mdape=md,flips=fl,coverage=g['knn_coverage'](Ak,Bk,mA,mB,dA,dB,mc=200,seed=seed)))
        if 'SOFTIMPUTE' in meth:
            As=g['softimpute'](A0,mA); Bs=g['softimpute'](B0,mB); md,fl=mp(imp(As,Bs)); rows.append(dict(base,method='SOFTIMPUTE',mdape=md,flips=fl,coverage=np.nan))
        if 'VAE' in meth:
            Av,SA=g['vae_impute'](A0,mA,mc=150,seed=seed); Bv,SB=g['vae_impute'](B0,mB,mc=150,seed=seed); md,fl=mp(imp(Av,Bv)); rows.append(dict(base,method='VAE',mdape=md,flips=fl,coverage=g['mc_coverage_from_S'](SA,mA,SB,mB,Av,Bv,mc=150)))
        if 'GNN' in meth:
            Ag,GA=g['gnn_impute'](A0,mA,mc=150,seed=seed); Bg,GB=g['gnn_impute'](B0,mB,mc=150,seed=seed); md,fl=mp(imp(Ag,Bg)); rows.append(dict(base,method='GNN',mdape=md,flips=fl,coverage=g['mc_coverage_from_S'](GA,GB if False else GA,mA,GB,Ag,Bg,mc=150) if False else g['mc_coverage_from_S'](GA,mA,GB,mB,Ag,Bg,mc=150)))
    pd.DataFrame(rows).to_csv(out,mode='a',header=hdr,index=False); hdr=False
    print(seed,f'{time.time()-t0:.0f}s',flush=True)
