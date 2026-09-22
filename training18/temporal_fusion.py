import numpy as np

def neighbours(desc,kind='spectral'):
    n=len(desc)
    if kind=='spectral':
        # NDVI and relative contrast receive equal weight after within-site scaling.
        cols=[0,3,4,6,8,11,12,14]
        x=desc[:,cols];scale=np.maximum(np.std(x,axis=0),.025);x=x/scale
        dist=((x[:,None]-x[None,:])**2).mean(-1)
    else:dist=np.abs(np.arange(n)[:,None]-np.arange(n)[None,:]).astype(float)
    np.fill_diagonal(dist,np.inf)
    return np.argsort(dist,axis=1)[:,:min(3,n-1)]

def fuse(maps,desc,alpha,kind='spectral'):
    ids=neighbours(desc,kind);out=np.asarray(maps,dtype=np.float32).copy()
    if not ids.shape[1]:return out
    for i,js in enumerate(ids):out[i]=(1-alpha)*maps[i].astype(np.float32)+alpha*np.mean(maps[js].astype(np.float32),axis=0)
    return out
