"""Symmetric two-state forward/backward visibility smoothing."""
import numpy as np
from scipy.special import logit,expit

def posterior(prob,keys,threshold,penalty):
    out=np.zeros(len(prob));sites={}
    for i,(date,sid) in enumerate(keys):sites.setdefault(sid,[]).append((date,i))
    for pairs in sites.values():
        ids=[i for _,i in sorted(pairs)];v=logit(np.clip(np.asarray(prob)[ids],1e-5,1-1e-5))-logit(threshold);fw=np.zeros((len(v),2));bw=np.zeros_like(fw);fw[0,1]=v[0]
        for i in range(1,len(v)):
            fw[i,0]=np.logaddexp(fw[i-1,0],fw[i-1,1]-penalty)
            fw[i,1]=v[i]+np.logaddexp(fw[i-1,0]-penalty,fw[i-1,1])
        for i in range(len(v)-2,-1,-1):
            bw[i,0]=np.logaddexp(bw[i+1,0],bw[i+1,1]+v[i+1]-penalty)
            bw[i,1]=np.logaddexp(bw[i+1,0]-penalty,bw[i+1,1]+v[i+1])
        post=fw+bw;out[ids]=expit(post[:,1]-post[:,0])
    return out
