"""Label-free multispectral and temporal features shared by train and inference."""
import numpy as np
from scipy.ndimage import zoom
from shapely.geometry import Polygon
from shapely.ops import unary_union
import shapely

def geom(polys):
    parts=[Polygon(p[0],p[1:]).buffer(0) for p in polys or [] if p and len(p[0])>=3]
    return unary_union(parts) if parts else None

def mask(g):
    out=np.zeros((256,256),bool)
    if g is None or g.is_empty:return out
    x0,y0,x1,y1=g.bounds;c0=max(0,int(x0)-1);c1=min(256,int(x1)+2);r0=max(0,int(y0)-1);r1=min(256,int(y1)+2)
    if c1>c0 and r1>r0:
        xx,yy=np.meshgrid(np.arange(c0,c1)+.5,np.arange(r0,r1)+.5);out[r0:r1,c0:c1]=shapely.contains_xy(g,xx,yy)
    return out

def channels(b):
    x=np.nan_to_num(b.astype(np.float32)/10000);r,g,bl,n=x.transpose(2,0,1)
    ndvi=(n-r)/np.maximum(n+r,1e-5);ndwi=(g-n)/np.maximum(g+n,1e-5);bsi=(r+bl-n-g)/np.maximum(r+bl+n+g,1e-5)
    return np.stack([r,g,bl,n,ndvi,ndwi,bsi,(r+g+bl)/3],-1)

def stats(stack,m):
    z=stack[m];z=z[np.isfinite(z).all(1)]
    if not len(z):return np.zeros(stack.shape[-1]*5,np.float32)
    return np.concatenate([z.mean(0),z.std(0),*np.quantile(z,[.1,.5,.9],axis=0)]).astype(np.float32)

def temporal(raw):
    raw=np.asarray(raw,np.float32);med=np.median(raw,axis=0);q25,q75=np.quantile(raw,[.25,.75],axis=0)
    ranks=np.argsort(np.argsort(raw,axis=0),axis=0)/max(len(raw)-1,1)
    local=np.array([raw[max(0,i-1):min(len(raw),i+2)].mean(0) for i in range(len(raw))])
    return np.concatenate([raw,np.clip((raw-med)/np.maximum(q75-q25,.001),-20,20),ranks,local-raw],1).astype(np.float32)

def entry_features(sites,dates,arrays,sam=None):
    geoms=[geom(s.get('prior_polygon')) for s in sites]
    stacks=[channels(b) for b in arrays];ref=np.median(np.stack(stacks),axis=0)
    keys=[];result=[]
    for s,g in zip(sites,geoms):
        if g is None:continue
        inside=mask(g);others=unary_union([q.buffer(2) for q in geoms if q is not None and q is not g])
        expanded=mask(g.buffer(3).difference(others));ring=mask(g.buffer(10).difference(g.buffer(4)).difference(others))
        raw=[]
        for i,(d,b,c) in enumerate(zip(dates,arrays,stacks)):
            valid=(b[...,0]>0)&(b[...,3]>0)&np.isfinite(b).all(-1)
            regions=[inside&valid,expanded&valid,ring&valid]
            ss=[stats(c,m) for m in regions]
            f=[*ss[0],*ss[1],*ss[2],*(ss[0]-ss[2]),*(ss[1]-ss[2])]
            # Co-registered change, particularly scars outside the old prior boundary.
            for m in regions[:2]:f.extend(stats((c-ref)[...,[0,3,4,6]],m))
            for m in regions:
                f.extend([np.mean(c[...,4][m]<t) if m.any() else 0. for t in [0,.2,.4,.6]])
            month=int(d[4:6]);f.extend([np.log1p(g.area),inside.sum(),valid[inside].mean() if inside.any() else 0.,np.sin(2*np.pi*month/12),np.cos(2*np.pi*month/12)])
            if sam is not None:
                logits,conf=sam[d][s['site_id']]
                l=zoom(logits.astype(np.float32),256/logits.shape[-1],order=1)
                sm=(l>1)&valid
                sx=stats(c,sm);f.extend(sx);f.extend(sx-ss[2]);f.extend(stats((c-ref)[...,[0,3,4,6]],sm))
                f.extend([float(conf),np.log1p(sm.sum()),sm.sum()/max(inside.sum(),1),(sm&inside).sum()/max((sm|inside).sum(),1)])
                f.extend([np.mean(l[m]) if m.any() else -10. for m in regions]);f.extend([np.mean(l[m]>1) if m.any() else 0. for m in regions])
            raw.append(f)
        ft=temporal(raw);keys.extend([(d,s['site_id']) for d in dates]);result.extend(ft)
    return np.asarray(result,np.float32),keys
