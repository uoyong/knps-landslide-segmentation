"""A spectral boundary refiner, evaluated with whole windows held out."""
import json,time,argparse
from pathlib import Path
import cv2,joblib,numpy as np,shapely
from scipy.ndimage import gaussian_filter,distance_transform_edt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from features import geom,channels

RES=512

def patch_features(bands,reference):
    c=channels(bands);ref=channels(reference)
    out=np.concatenate([c,c-ref,gaussian_filter(c,(1,1,0)),gaussian_filter(c,(2,2,0))],-1)
    return cv2.resize(out,(RES,RES),interpolation=cv2.INTER_LINEAR)

def site_features(patch,logits,prior):
    l=cv2.resize(logits.astype(np.float32),(RES,RES),interpolation=cv2.INTER_LINEAR)
    # Region is determined only by the input prior and model prediction.
    x0,y0,x1,y1=prior.bounds;pad=8
    lmask=l>0
    rr,cc=np.where(lmask)
    if len(rr):x0=min(x0,cc.min()/2);x1=max(x1,(cc.max()+1)/2);y0=min(y0,rr.min()/2);y1=max(y1,(rr.max()+1)/2)
    c0=max(0,int((x0-pad)*2));c1=min(RES,int((x1+pad)*2)+1);r0=max(0,int((y0-pad)*2));r1=min(RES,int((y1+pad)*2)+1)
    yy,xx=np.mgrid[r0:r1,c0:c1];xx=(xx+.5)/2;yy=(yy+.5)/2
    pm=shapely.contains_xy(prior,xx,yy)
    prior_dist=(distance_transform_edt(pm)-distance_transform_edt(~pm))/2
    pred_dist=(distance_transform_edt(l[r0:r1,c0:c1]>1)-distance_transform_edt(l[r0:r1,c0:c1]<=1))/2
    bx0,by0,bx1,by1=prior.bounds
    sx=(xx-(bx0+bx1)/2)/max(bx1-bx0,1);sy=(yy-(by0+by1)/2)/max(by1-by0,1)
    extra=np.stack([l[r0:r1,c0:c1],gaussian_filter(l,2)[r0:r1,c0:c1],np.clip(prior_dist,-10,10),np.clip(pred_dist,-10,10),sx,sy,np.full_like(sx,np.log1p(prior.area))],-1)
    f=np.concatenate([patch[r0:r1,c0:c1],extra],-1).reshape(-1,patch.shape[-1]+7).astype(np.float32)
    return f,(r0,r1,c0,c1),xx,yy

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--train',action='store_true');ap.add_argument('--features-only',action='store_true');ap.add_argument('--group',choices=['window','scene'],default='window');a=ap.parse_args()
    root=Path('training17');dest=root/'pixel_data';dest.mkdir(exist_ok=True);rng=np.random.default_rng(1702);t=time.time()
    X=[];y=[];groups=[];items=[]
    gy,gx=np.mgrid[:RES,:RES];gx=(gx+.5)/2;gy=(gy+.5)/2
    for loc in sorted(Path('train').iterdir()):
        dates=sorted(p.stem for p in (loc/'bands').glob('*.npy'));arrays=[np.load(loc/'bands'/f'{d}.npy') for d in dates]
        ref=np.median(np.stack(arrays),0);priors={s['site_id']:geom(s['prior_polygon']) for s in json.loads((loc/'sites.json').read_text())}
        for date,b in zip(dates,arrays):
            cp=root/'masks'/f'{loc.name}_{date}.npz'
            if not cp.exists():continue
            d=np.load(cp);idx={sid:i for i,sid in enumerate(d['sites'])};patch=None
            for row in json.loads((loc/'labels'/f'{date}.json').read_text()):
                if row.get('ignore') or not row.get('visible'):continue
                sid=row['site_id'];gt=geom(row['polygon']);key=[loc.name,date,sid];path=dest/f'{loc.name}_{date}_{sid}.npz'
                if not path.exists():
                    if patch is None:patch=patch_features(b,ref)
                    f,roi,xx,yy=site_features(patch,d['tuned'][idx[sid]],priors[sid]);target=shapely.contains_xy(gt,xx,yy).ravel()
                    gt_area=gt.area
                    np.savez_compressed(path,X=f,y=target,roi=roi,gt_area=gt_area)
                z=np.load(path);target=z['y'];f=z['X'];pos=np.flatnonzero(target);neg=np.flatnonzero(~target)
                # Uniform sampling keeps class prior representative inside each region.
                take=rng.choice(len(target),min(300,len(target)),replace=False)
                gt_count=int(shapely.contains_xy(gt,gx,gy).sum())
                X.append(f[take]);y.append(target[take]);groups.extend([loc.name]*len(take));items.append((key,path,gt_count))
        print('pixel features',loc.name,len(items),'elapsed',round(time.time()-t,1),flush=True)
    if a.features_only:return
    X=np.concatenate(X);y=np.concatenate(y);groups=np.array(groups);print('pixel training',X.shape,y.mean(),flush=True)
    cv=GroupKFold(5);records=[];pred_dir=root/('pixel_oof' if a.group=='window' else 'pixel_oof_scene');pred_dir.mkdir(exist_ok=True)
    cvgroups=groups if a.group=='window' else np.array([g.split('_w')[0] for g in groups])
    for fold,(tr,va) in enumerate(cv.split(X,y,cvgroups)):
        model=HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=15,learning_rate=.065,l2_regularization=15,min_samples_leaf=80,max_bins=127,random_state=1703)
        model.fit(X[tr],y[tr]);valgroups=set(groups[va])
        joblib.dump({'model':model,'held_out_windows':sorted(valgroups)},root/f'pixel_{a.group}_fold{fold}.joblib',compress=3)
        for key,path,gt_count in items:
            if key[0] not in valgroups:continue
            z=np.load(path);prob=model.predict_proba(z['X'])[:,1].astype(np.float32);r0,r1,c0,c1=z['roi'];p=prob.reshape(r1-r0,c1-c0)
            np.savez_compressed(pred_dir/path.name,prob=p.astype(np.float16),roi=z['roi'])
            gt=z['y'];base=1/(1+np.exp(-np.clip(z['X'][:,-7]+0.,-20,20)))
            r={'key':key,'area':float(z['gt_area']),'ious':{}}
            # Raster only for candidate ranking; chosen output gets exact polygon verification.
            for weight in [.25,.5,.75,1.]:
                mix=weight*prob+(1-weight)*base
                for th in [.4,.5,.6]:
                    m=mix>th;inter=(m & gt).sum();r['ious'][f'{weight}_{th}']=float(inter/max(m.sum()+gt_count-inter,1))
            bm=base>1/(1+np.exp(-1));inter=(bm & gt).sum();r['base']=float(inter/max(bm.sum()+gt_count-inter,1));records.append(r)
        print('pixel fold',fold+1,'elapsed',round(time.time()-t,1),flush=True)
    (root/('pixel_scores.json' if a.group=='window' else 'pixel_scores_scene.json')).write_text(json.dumps(records))
    for variant in ['base',*records[0]['ious']]:
        ss={}
        for r in records:
            if r['area']>=10:ss.setdefault((r['key'][0],r['key'][2]),[]).append(r['base'] if variant=='base' else r['ious'][variant])
        print(variant,np.mean([np.mean(v) for v in ss.values()]),flush=True)
    if a.train:
        model.fit(X,y);joblib.dump(model,'17_submission/assets/pixel_refiner.joblib',compress=3)

if __name__=='__main__':main()
