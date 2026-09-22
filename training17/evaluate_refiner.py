import argparse,json,time
from pathlib import Path
import cv2,joblib,numpy as np
from shapely.ops import unary_union
from features import geom
from pixel_refine import patch_features,site_features
from baseline import px_polys_from_mask

ap=argparse.ArgumentParser();ap.add_argument('--group',choices=['window','scene'],default='window');args=ap.parse_args()
cv2.setNumThreads(1);root=Path('training17');models={}
for p in root.glob(f'pixel_{args.group}_fold*.joblib'):
    bundle=joblib.load(p)
    for loc in bundle['held_out_windows']:models[loc]=bundle['model']
assert len(models)==14 # one public window has no positive labels
fallback=joblib.load('17_submission/assets/pixel_refiner.joblib')
records=[];t=time.time()
for loc in sorted(Path('train').iterdir()):
    dates=sorted(p.stem for p in (loc/'bands').glob('*.npy'));arrays=[np.load(loc/'bands'/f'{d}.npy') for d in dates];ref=np.median(np.stack(arrays),0)
    priors={s['site_id']:geom(s['prior_polygon']) for s in json.loads((loc/'sites.json').read_text())};model=models.get(loc.name,fallback)
    for date,b in zip(dates,arrays):
        z0=np.load(root/'masks'/f'{loc.name}_{date}.npz');z={k:z0[k] for k in ['sites','tuned']};index={s:i for i,s in enumerate(z['sites'])};patch=None
        for row in json.loads((loc/'labels'/f'{date}.json').read_text()):
            if row.get('ignore'):continue
            sid=row['site_id'];prior=priors[sid];logits=z['tuned'][index[sid]].astype(np.float32);gt=geom(row.get('polygon')) if row['visible'] else None
            path=root/'pixel_data'/f'{loc.name}_{date}_{sid}.npz'
            if path.exists():d=np.load(path);X=d['X'];roi=d['roi']
            else:
                if patch is None:patch=patch_features(b,ref)
                X,roi,_,_=site_features(patch,logits,prior)
            r0,r1,c0,c1=roi;prob=model.predict_proba(X)[:,1].reshape(r1-r0,c1-c0);base=1/(1+np.exp(-np.clip(cv2.resize(logits,(512,512)),-20,20)))
            r={'key':[loc.name,date,sid],'y':int(row['visible']),'area':float(gt.area) if gt else 0.,'iou':{},'nonempty':{}}
            for w,th in [(.5,.5),(.75,.4),(1.,.4)]:
                mix=np.zeros((512,512),np.float32);mix[r0:r1,c0:c1]=w*prob+(1-w)*base[r0:r1,c0:c1];m=cv2.resize(mix,(1024,1024))>th;name=f'pixel_{w}_{th}'
                if gt is not None and gt.area>=10:
                    parts=px_polys_from_mask(m,max(1.,min(2.,.3*prior.area)));g=unary_union(parts).simplify(.05) if parts else None
                    r['iou'][name]=float(g.intersection(gt).area/g.union(gt).area) if g is not None else 0.;r['nonempty'][name]=g is not None
                else:
                    _,_,stats,_=cv2.connectedComponentsWithStats(m.astype(np.uint8),connectivity=4);r['nonempty'][name]=bool(np.any(stats[1:,cv2.CC_STAT_AREA]>=16*max(1.,min(2.,.3*prior.area))));r['iou'][name]=0.
            records.append(r)
    print(loc.name,len(records),'elapsed',round(time.time()-t,1),flush=True)
    (root/f'refined_scores_{args.group}.json').write_text(json.dumps(records))
assert len(records)==1601
