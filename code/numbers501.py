import glob,json,numpy as np,pandas as pd
R='results/'
M=pd.concat([pd.read_csv(f) for f in glob.glob(R+'res_*_*_*.csv')],ignore_index=True).drop_duplicates(['method','pattern','rate','seed'])
D=pd.concat([pd.read_csv(f) for f in glob.glob(R+'dose_*.csv')],ignore_index=True)
F={p:pd.read_csv(R+f'fact_{p}_0_500.csv') for p in ('MCAR','FLOW','BLOCK')}
N={}
def g(m,p,r): return M[(M.method==m)&(M.pattern==p)&np.isclose(M.rate,r)]
for m in ('PEDIGREE','KNN','SOFTIMPUTE','VAE','GNN','ROWMEAN'):
    for p in ('MCAR','FLOW','BLOCK'):
        for r in (.05,.1,.2,.4):
            x=g(m,p,r); N[f'{m}|{p}|{r}']=dict(n=len(x),mdape=float(x.mdape.median()*100),cov=(None if x.coverage.isna().all() else float(x.coverage.mean()*100)),flips=float(x.flips.mean()*100))
N['rowmean_rescaled_MCAR40']=float(g('ROWMEAN','MCAR',.4).rescaled.mean()*100)
# factorial + width
for p,f in F.items():
    for r in (.05,.1,.2,.4):
        x=f[np.isclose(f.rate,r)]
        N[f'fact|{p}|{r}']={c:dict(cov=float(x[c+'_cov'].mean()*100),w=float(x[c+'_w'].median()*100),w_mean=float(x[c+'_w'].mean()*100)) for c in ('SIB_LN','SIB_EMP','KNN_LN','KNN_EMP')}
        N[f'fact|{p}|{r}']['below_LN']=float(x.SIB_LN_below.sum()/(x.SIB_LN_below.sum()+x.SIB_LN_above.sum())*100)
# dose
for m in D.method.unique():
    for k in (1,2,3,6,12,25):
        x=D[(D.method==m)&(D.k==k)]; v=x.mdape*100
        N[f'dose|{m}|{k}']=dict(n=len(x),med=float(v.median()),p25=float(np.percentile(v,25)),p75=float(np.percentile(v,75)),lt2=float((v<2).mean()*100),gt10=float((v>10).mean()*100),cov=(None if x.coverage.isna().all() else float(x.coverage.mean()*100)),flips=float(x.flips.mean()*100))
json.dump(N,open('results/numbers.json','w'),indent=1)
# print key
for key in ['PEDIGREE|BLOCK|0.4','KNN|BLOCK|0.4','ROWMEAN|BLOCK|0.4','ROWMEAN|MCAR|0.4','ROWMEAN|FLOW|0.4','KNN|BLOCK|0.2']: print(key,{a:(round(b,1) if b is not None else None) for a,b in N[key].items()})
for p,r in (('MCAR',.05),('MCAR',.4),('FLOW',.4),('BLOCK',.4)):
    print(p,r,{c:(round(v['cov'],1),round(v['w'],0)) for c,v in N[f'fact|{p}|{r}'].items() if isinstance(v,dict)})
for k in (1,2,3,6,12,25): print('dose',k,{m:(round(N[f'dose|{m}|{k}']['med'],2)) for m in ('PEDIGREE','KNN','ROWMEAN')},'PED q',round(N[f'dose|PEDIGREE|{k}']['p25'],2),round(N[f'dose|PEDIGREE|{k}']['p75'],2),'lt2',round(N[f'dose|PEDIGREE|{k}']['lt2'],1),'gt10',round(N[f'dose|PEDIGREE|{k}']['gt10'],1))
print('KNN dose cov',[round(N[f'dose|KNN|{k}']['cov'],1) for k in (1,2,3,6,12,25)])
