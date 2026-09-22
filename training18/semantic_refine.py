"""Frozen SAM encoder features augment the spectral boundary learner."""
import numpy as np,cv2
from refine_features import enrich

def semantic_features(X,roi,neck):
    r0,r1,c0,c1=roi
    feature=cv2.resize(neck.astype(np.float32).transpose(1,2,0),(512,512),interpolation=cv2.INTER_LINEAR)[r0:r1,c0:c1].reshape(len(X),-1)
    bg=X[:,32]<-2
    if bg.sum()<20:bg=np.ones(len(X),bool)
    med=np.median(feature[bg],axis=0);std=np.maximum(np.std(feature[bg],axis=0),.1)
    return np.concatenate([enrich(X),feature,np.clip((feature-med)/std,-10,10)],1).astype(np.float32)
