"""Radiometrically normalized pixel features; no target labels used."""
import numpy as np

def enrich(X):
    c=X[:,:8];bg=X[:,32]<-2
    if bg.sum()<20:bg=np.ones(len(X),bool)
    q10,med,q90=np.quantile(c[bg],[.1,.5,.9],axis=0);scale=np.maximum(q90-q10,np.array([.02]*4+[.1]*3+[.02]))
    fg=X[:,32]>1
    fgmed=np.median(c[fg],axis=0) if fg.sum()>=5 else med
    rgb=c[:,:3]/np.maximum(c[:,:3].sum(1,keepdims=True),.01)
    local1=X[:,16:24]-c;local2=X[:,24:32]-c
    relative=np.clip((c-med)/scale,-10,10);fg_relative=np.clip((c-fgmed)/scale,-10,10)
    ratios=c[:,:4]/np.maximum(c[:,:4].sum(1,keepdims=True),.01)
    return np.concatenate([X,relative,fg_relative,rgb,ratios,local1,local2],1).astype(np.float32)
