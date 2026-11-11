import sys,types,os,time
src=open('experiment_final_eu.py').read().split('# ==================== MAIN GRID')[0]
src=src.replace("os.path.join('..','data_build',f)","os.path.join('..','data_build',f)")
g={'__name__':'x'}; exec(compile(src,'e','exec'),g)
import numpy as np, pandas as pd
pat=sys.argv[1]; methods=sys.argv[2].split(','); s0,s1=int(sys.argv[3]),int(sys.argv[4])
out=f'res_{pat}_{"-".join(methods)}_{s0}_{s1}.csv'
RATES=[0.05,0.10,0.20,0.40]
mk=g['make_masks']; imp=g['impacts']; mp=g['metrics_point']
hdr=not os.path.exists(out); t0=time.time()
for seed in range(s0,s1):
    rows=[]
    for rate in RATES:
        mA,mB=mk(pat,rate,seed)
        if 'PEDIGREE' in methods:
            Ah,Bh=g['repair_baseline'](mA,mB); md,fl=mp(imp(Ah,Bh))
            rows.append(dict(method='PEDIGREE',pattern=pat,rate=rate,seed=seed,mdape=md,flips=fl,coverage=g['pedigree_coverage'](Ah,Bh,mA,mB,mc=300,seed=seed)))
        if 'KNN' in methods:
            Ak,dA=g['knn_impute'](g['A0'],mA); Bk,dB=g['knn_impute'](g['B0'],mB); md,fl=mp(imp(Ak,Bk))
            rows.append(dict(method='KNN',pattern=pat,rate=rate,seed=seed,mdape=md,flips=fl,coverage=g['knn_coverage'](Ak,Bk,mA,mB,dA,dB,mc=200,seed=seed)))
        if 'SOFTIMPUTE' in methods:
            As=g['softimpute'](g['A0'],mA); Bs=g['softimpute'](g['B0'],mB); md,fl=mp(imp(As,Bs))
            rows.append(dict(method='SOFTIMPUTE',pattern=pat,rate=rate,seed=seed,mdape=md,flips=fl,coverage=np.nan))
        if 'VAE' in methods:
            Av,SA=g['vae_impute'](g['A0'],mA,mc=150,seed=seed); Bv,SB=g['vae_impute'](g['B0'],mB,mc=150,seed=seed); md,fl=mp(imp(Av,Bv))
            rows.append(dict(method='VAE',pattern=pat,rate=rate,seed=seed,mdape=md,flips=fl,coverage=g['mc_coverage_from_S'](SA,mA,SB,mB,Av,Bv,mc=150)))
        if 'GNN' in methods:
            Ag,GA=g['gnn_impute'](g['A0'],mA,mc=150,seed=seed); Bg,GB=g['gnn_impute'](g['B0'],mB,mc=150,seed=seed); md,fl=mp(imp(Ag,Bg))
            rows.append(dict(method='GNN',pattern=pat,rate=rate,seed=seed,mdape=md,flips=fl,coverage=g['mc_coverage_from_S'](GA,mA,GB,mB,Ag,Bg,mc=150)))
    pd.DataFrame(rows).to_csv(out,mode='a',header=hdr,index=False); hdr=False
    print(pat,seed,f'{time.time()-t0:.0f}s',flush=True)
