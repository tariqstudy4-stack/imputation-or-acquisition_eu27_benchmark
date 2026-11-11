# Sensitivity of VAE and GNN interval coverage to the number of Monte Carlo draws (75, 150, 300) on the same trained models.
# usage: python draws_check.py <seed0> <seed1>   (run from code/)
import sys,numpy as np,pandas as pd
src=open('experiment_final_eu.py').read().split('# ==================== MAIN GRID')[0]
g={'__name__':'x'}; exec(compile(src,'e','exec'),g)
A0,B0,make_masks,impacts=[g[k] for k in ('A0','B0','make_masks','impacts')]
s0,s1=int(sys.argv[1]),int(sys.argv[2]); rows=[]
for pat in ('MCAR','FLOW','BLOCK'):
    for seed in range(s0,s1):
        for rate in (0.20,0.40):
            mA,mB=make_masks(pat,rate,seed)
            for nm,fn in (('VAE',g['vae_impute']),('GNN',g['gnn_impute'])):
                Ah,SA=fn(A0,mA,mc=300,seed=seed); Bh,SB=fn(B0,mB,mc=300,seed=seed)
                r=dict(method=nm,pattern=pat,rate=rate,seed=seed)
                for m in (75,150,300): r[f'cov{m}']=g['mc_coverage_from_S'](SA,mA,SB,mB,Ah,Bh,mc=m)
                rows.append(r)
    print(pat,flush=True)
pd.DataFrame(rows).to_csv(f'draws_check_{s0}_{s1}.csv',index=False)
