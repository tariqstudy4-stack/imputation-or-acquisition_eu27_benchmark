import glob,json,numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':10.5,'axes.titlesize':11,'axes.labelsize':10.5,'legend.fontsize':8.5,'font.family':'DejaVu Sans'})
M=pd.concat([pd.read_csv(f) for f in glob.glob('results/res_*_*_*.csv') if 'PEDIGREE' in f or 'VAE' in f or 'ROWMEAN' in f],ignore_index=True)
M=M.drop_duplicates(['method','pattern','rate','seed'])
STY={'ROWMEAN':('#9a9a9a','Row mean (naive fill)','x'),'PEDIGREE':('#555555','Pedigree (current practice)','o'),'KNN':('#1b9e56','KNN donor','s'),'SOFTIMPUTE':('#8c56a8','SoftImpute','^'),'VAE':('#e08214','VAE','D'),'GNN':('#d7301f','GNN','v')}
PATS=[('MCAR','Random loss'),('FLOW','Sensor failure'),('BLOCK','Supplier non-disclosure')]
R=[5,10,20,40]
def agg(df,method,pat,col,fn):
    d=df[(df.method==method)&(df.pattern==pat)].groupby('rate')[col].agg(fn).reindex([.05,.1,.2,.4]); return d.values*100
S={}
for m in STY:
    for p,_ in PATS:
        d=M[(M.method==m)&(M.pattern==p)]
        for r in (.05,.1,.2,.4):
            x=d[d.rate==r]; S[f'{m}|{p}|{r}']=dict(n=len(x),mdape_med=float(x.mdape.median()*100) if len(x) else None,mdape_mean=float(x.mdape.mean()*100) if len(x) else None,cov=float(x.coverage.mean()*100) if x.coverage.notna().any() else None,flips=float(x.flips.mean()*100) if len(x) else None)
json.dump(S,open('results/summary501.json','w'),indent=1)
sw=pd.concat([pd.read_csv(f) for f in glob.glob('results/sweep_*.csv')]) if glob.glob('results/sweep_*.csv') else None
def fig1():
    fig,ax=plt.subplots(2,4,figsize=(15.2,7.0))
    for j,(p,t) in enumerate(PATS):
        a=ax[0,j]
        for m,(c,l,mk) in STY.items():
            y=agg(M,m,p,'mdape','median')
            if np.isfinite(y).any(): a.plot(R,y,mk+('--' if m=='ROWMEAN' else '-'),color=c,label=l,lw=(1.4 if m=='ROWMEAN' else 1.8),ms=5)
        a.set_yscale('log'); a.set_title(f'({"abc"[j]}) {t}: error',loc='left'); a.set_xlabel('Missing data (%)'); a.set_ylabel('Median impact error, MdAPE (%)'); a.grid(alpha=.3,which='both'); a.axhline(100,color='gray',ls=':',lw=1)
        if j==0: a.legend(loc='lower right',fontsize=7.5)
        b=ax[1,j]
        for m,(c,l,mk) in STY.items():
            if m in ('SOFTIMPUTE','ROWMEAN'): continue
            y=agg(M,m,p,'coverage','mean')
            if np.isfinite(y).any(): b.plot(R,y,mk+'-',color=c,label=l,lw=1.8,ms=5)
        b.axhline(95,color='gray',ls='--',lw=1.3); b.text(5,96.5,'nominal 95%',color='gray',fontsize=8.5); b.set_ylim(0,100); b.set_xlabel('Missing data (%)'); b.set_ylabel('Coverage of 95% intervals (%)'); b.set_title(f'({"def"[j]}) {t}: calibration',loc='left'); b.grid(alpha=.3)
    # g: GSD sweep
    g=ax[0,3]
    if sw is not None:
        d=sw.groupby(['pattern','rate','gsd']).coverage.mean().reset_index()
        cs={('MCAR',.2):('tab:green','Random 20%'),('MCAR',.4):('tab:red','Random 40%'),('FLOW',.1):('tab:purple','Sensor 10%'),('FLOW',.4):('tab:brown','Sensor 40%'),('BLOCK',.2):('tab:blue','Supplier 20%'),('BLOCK',.4):('tab:orange','Supplier 40%')}
        for (pp,rr),(c,l) in cs.items():
            x=d[(d.pattern==pp)&np.isclose(d.rate,rr)].sort_values('gsd')
            if len(x): g.plot(x.gsd,x.coverage*100,'o-',color=c,label=l,lw=1.6,ms=4)
    g.axhline(95,color='gray',ls='--',lw=1.3); g.set_ylim(0,100); g.set_xlabel('Pedigree GSD of the technosphere factor'); g.set_ylabel('Coverage of 95% intervals (%)'); g.set_title('(g) Widening the pedigree factor',loc='left'); g.legend(loc='upper left',fontsize=7.5,ncol=2); g.grid(alpha=.3)
    # h: flips at 40%
    h=ax[1,3]; w=0.13
    for i,(m,(c,l,mk)) in enumerate(STY.items()):
        v=[S[f'{m}|{p}|0.4']['flips'] or 0 for p,_ in PATS]
        h.bar(np.arange(3)+(i-2.5)*w,v,w,color=c,label=l)
    h.set_xticks(range(3)); h.set_xticklabels(['Random','Sensor','Supplier']); h.set_ylabel('Ranking flips at 40% missing (%)'); h.set_title('(h) Decision stability, all methods',loc='left'); h.grid(alpha=.3,axis='y')
    fig.tight_layout(); fig.savefig('figs/fig1.png',dpi=200); plt.close(fig)
fig1()
dose=pd.concat([pd.read_csv(f) for f in glob.glob('results/dose_*.csv')]) if glob.glob('results/dose_*.csv') else None
def fig2():
    if dose is None: return
    K=[1,2,3,6,12,25]; fig,ax=plt.subplots(1,3,figsize=(15.2,3.6))
    for m,(c,l,mk) in STY.items():
        d=dose[dose.method==m]
        if not len(d): continue
        e=d.groupby('k').mdape.median().reindex(K)*100; f=d.groupby('k').flips.mean().reindex(K)*100; cv=d.groupby('k').coverage.mean().reindex(K)*100
        ax[0].plot(K,e,mk+('--' if m=='ROWMEAN' else '-'),color=c,label=l,lw=(1.4 if m=='ROWMEAN' else 1.8),ms=5); ax[2].plot(K,f,mk+('--' if m=='ROWMEAN' else '-'),color=c,lw=(1.4 if m=='ROWMEAN' else 1.8),ms=5)
        if m not in ('SOFTIMPUTE','ROWMEAN'): ax[1].plot(K,cv,mk+'-',color=c,lw=1.8,ms=5,label=l)
    dd=dose[dose.method=='PEDIGREE']; q1=dd.groupby('k').mdape.quantile(.25).reindex(K)*100; q3=dd.groupby('k').mdape.quantile(.75).reindex(K)*100
    ax[0].fill_between(K,q1,q3,color='#555555',alpha=.16,lw=0,label='Pedigree, 25th to 75th percentile')
    ax[0].set_yscale('log'); ax[0].axhline(10,color='gray',ls=':',lw=1); ax[0].text(1,11,'10% error',color='gray',fontsize=8.5)
    ax[1].axhline(95,color='gray',ls='--',lw=1.3); ax[1].set_ylim(0,100)
    for a,t,y in zip(ax,['(a) Error','(b) Calibration','(c) Decision stability'],['Median impact error, MdAPE (%)','Coverage of 95% intervals (%)','Ranking flips (%)']):
        a.set_xscale('log'); a.set_xticks(K); a.set_xticklabels([str(k) for k in K]); a.set_xlabel('Supplier industries with no data (of 45 candidates)'); a.set_ylabel(y); a.set_title(t,loc='left'); a.grid(alpha=.3,which='both')
    ax[0].legend(loc='lower right',fontsize=7.5)
    fig.tight_layout(); fig.savefig('figs/fig2.png',dpi=200); plt.close(fig)
fig2()
print('ok',M.groupby(['method','pattern']).seed.nunique().to_dict())
