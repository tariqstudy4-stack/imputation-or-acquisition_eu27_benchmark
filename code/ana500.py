# 2x2 (point estimate x uncertainty model) and GSD sweep, 500 seeds, equal 300 MC draws everywhere
import sys,os,time,numpy as np,pandas as pd
src=open('experiment_final_eu.py').read().split('# ==================== MAIN GRID')[0]
src=src.replace("os.path.join('..','data_build',f)","os.path.join('..','data_build',f)")
import types
t=types.ModuleType('torch'); t.set_num_threads=lambda *a:None; nn=types.ModuleType('torch.nn'); nn.Module=object; t.nn=nn
sys.modules['torch']=t; sys.modules['torch.nn']=nn
g={'__name__':'x'}; exec(compile(src,'e','exec'),g)
globals().update({k:g[k] for k in ('A0','B0','K','targets','H_true','impacts','make_masks','repair_baseline','knn_impute','flatten_donors','sib_cols','n','metrics_point')})
sA,sB=np.log(2.0),np.log(1.8)
MC=300
def sib_donors(M0,mask):
    blocked=set(np.where(mask.all(axis=0))[0].tolist()); R=[];C=[];V=[]
    for j in np.unique(np.where(mask)[1]):
        base=sib_cols(j,blocked)
        for r in np.where(mask[:,j])[0]:
            o=[c for c in base if not mask[r,c]]
            if not o: o=[c for c in range(n) if c!=j and c not in blocked and not mask[r,c]]
            if not o: o=[c for c in range(n) if not mask[r,c]]
            R.append(r);C.append(j);V.append(M0[r,o])
    if not R: return None
    kmax=max(len(v) for v in V); P=np.zeros((len(V),kmax)); cnt=np.array([len(v) for v in V])
    for i,v in enumerate(V): P[i,:len(v)]=v
    return np.array(R),np.array(C),P,cnt
def ln_S(Ah,Bh,mA,mB,k,seed):
    rng=np.random.default_rng(seed); iA,iB=np.argwhere(mA),np.argwhere(mB); S=np.empty((MC,K,len(targets)))
    for m in range(MC):
        Am,Bm=Ah.copy(),Bh.copy()
        if len(iA): Am[iA[:,0],iA[:,1]]=Ah[iA[:,0],iA[:,1]]*np.exp(rng.normal(0,k*sA,len(iA)))
        if len(iB): Bm[iB[:,0],iB[:,1]]=Bh[iB[:,0],iB[:,1]]*np.exp(rng.normal(0,k*sB,len(iB)))
        S[m]=impacts(Am,Bm)
    return S
def emp_S(Ah,Bh,dA,dB,seed):
    rng=np.random.default_rng(seed+99); S=np.empty((MC,K,len(targets)))
    for m in range(MC):
        Am,Bm=Ah.copy(),Bh.copy()
        for M_,d in ((Am,dA),(Bm,dB)):
            if d is None: continue
            r,c,P,cnt=d; idx=(rng.random(len(r))*cnt).astype(int); M_[r,c]=P[np.arange(len(r)),idx]
        S[m]=impacts(Am,Bm)
    return S
def knn_S(Ah,Bh,fA,fB,seed):
    rng=np.random.default_rng(seed+199); S=np.empty((MC,K,len(targets)))
    for m in range(MC):
        Am,Bm=Ah.copy(),Bh.copy()
        for M_,fl in ((Am,fA),(Bm,fB)):
            r_,c_,V_,W_=fl
            if r_ is None: continue
            u=rng.random(len(r_)); idx=np.clip((u[:,None]>W_).sum(axis=1),0,V_.shape[1]-1); M_[r_,c_]=V_[np.arange(len(r_)),idx]
        S[m]=impacts(Am,Bm)
    return S
def sc(S):
    lo,hi=np.quantile(S,0.025,0),np.quantile(S,0.975,0)
    return ((H_true>=lo)&(H_true<=hi)).mean(),(H_true<lo).mean(),(H_true>hi).mean()
pat=sys.argv[1]; s0,s1=int(sys.argv[2]),int(sys.argv[3]); out2=f'fact_{pat}_{s0}_{s1}.csv'; outg=f'sweep_{pat}_{s0}_{s1}.csv'
F=[];G=[]; t0=time.time()
for seed in range(s0,s1):
    for rate in (0.05,0.10,0.20,0.40):
        mA,mB=make_masks(pat,rate,seed)
        Ah,Bh=repair_baseline(mA,mB); Ak,dA=knn_impute(A0,mA); Bk,dB=knn_impute(B0,mB)
        row=dict(pattern=pat,rate=rate,seed=seed)
        for nm,(a,b,S) in {'SIB_LN':(Ah,Bh,lambda:ln_S(Ah,Bh,mA,mB,1.0,seed)),
                           'SIB_EMP':(Ah,Bh,lambda:emp_S(Ah,Bh,sib_donors(A0,mA),sib_donors(B0,mB),seed)),
                           'KNN_LN':(Ak,Bk,lambda:ln_S(Ak,Bk,mA,mB,1.0,seed)),
                           'KNN_EMP':(Ak,Bk,lambda:knn_S(Ak,Bk,flatten_donors(dA),flatten_donors(dB),seed))}.items():
            c,lo_,hi_=sc(S()); row[nm+'_cov']=c; row[nm+'_below']=lo_; row[nm+'_above']=hi_
        F.append(row)
        for gsd in (1.2,1.5,2.0,2.5,3.0):
            c,_,_=sc(ln_S(Ah,Bh,mA,mB,np.log(gsd)/np.log(2.0),seed)); G.append(dict(pattern=pat,rate=rate,seed=seed,gsd=gsd,coverage=c))
    if (seed+1)%25==0 or seed==s1-1:
        pd.DataFrame(F).to_csv(out2,index=False); pd.DataFrame(G).to_csv(outg,index=False); print(pat,seed,f'{time.time()-t0:.0f}s',flush=True)
