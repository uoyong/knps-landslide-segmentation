"""Exact polygon score and empty-mask behavior for new boundary learners."""
import sys,json,argparse,time
from pathlib import Path
import cv2,numpy as np,joblib
from shapely.ops import unary_union
from refine_features import enrich
sys.path.insert(0,'training17')
from features import geom
from pixel_refine import patch_features,site_features
from baseline import px_polys_from_mask

ap=argparse.ArgumentParser();ap.add_argument('--group',choices=['scene','window'],default='scene');a=ap.parse_args();cv2.setNumThreads(1);models={};oldmodels={};records=[];t=time.time()
for p in Path('training18').glob(f'refiner_{a.group}_fold*.joblib'):
    b=joblib.load(p)
    for loc in b['held_out_windows']:models[loc]=b['model']
for p in Path('training17').glob(f'pixel_{a.group}_fold*.joblib'):
    b=joblib.load(p)
    for loc in b['held_out_windows']:oldmodels[loc]=b['model']
assert len(models)==14
fallback=joblib.load('17_submission/assets/pixel_refiner.joblib')
for loc in sorted(Path('train').iterdir()):
    dates=sorted(p.stem for p in (loc/'bands').glob('*.npy'));arrays=[np.load(loc/'bands'/f'{d}.npy') for d in dates];ref=np.median(np.stack(arrays),0);sites={s['site_id']:geom(s['prior_polygon']) for s in json.loads((loc/'sites.json').read_text())}
    old=oldmodels.get(loc.name,fallback);new=models.get(loc.name,next(iter(models.values())))
    # The all-negative region has no examples in any refiner training fold.
    for date,bands in zip(dates,arrays):
        z0=np.load(Path('training17/masks')/f'{loc.name}_{date}.npz');idx={s:i for i,s in enumerate(z0['sites'])};logits=z0['tuned'];patch=None
        for row in json.loads((loc/'labels'/f'{date}.json').read_text()):
            if row.get('ignore'):continue
            sid=row['site_id'];prior=sites[sid];gt=geom(row['polygon']) if row['visible'] else None;cp=Path('training17/pixel_data')/f'{loc.name}_{date}_{sid}.npz'
            if cp.exists():d=np.load(cp);X=d['X'];roi=d['roi']
            else:
                if patch is None:patch=patch_features(bands,ref)
                X,roi,_,_=site_features(patch,logits[idx[sid]],prior)
            r0,r1,c0,c1=roi;pn=new.predict_proba(enrich(X))[:,1].reshape(r1-r0,c1-c0);po=old.predict_proba(X)[:,1].reshape(r1-r0,c1-c0)
            r={'key':[loc.name,date,sid],'y':int(row['visible']),'area':gt.area if gt else 0.,'iou':{},'nonempty':{}}
            for weight in [0.,.5,1.]:
                p=np.zeros((512,512),np.float32);p[r0:r1,c0:c1]=weight*pn+(1-weight)*po;big=cv2.resize(p,(1024,1024))
                for th in [.25,.35,.4]:
                    name=f'refine_{weight}_{th}';m=big>th
                    if gt is not None and gt.area>=10:
                        parts=px_polys_from_mask(m,max(1.,min(2.,.3*prior.area)));g=unary_union(parts).simplify(.05) if parts else None;r['nonempty'][name]=g is not None;r['iou'][name]=g.intersection(gt).area/g.union(gt).area if g is not None else 0.
                    else:
                        _,_,stats,_=cv2.connectedComponentsWithStats(m.astype(np.uint8),connectivity=4);r['nonempty'][name]=bool(np.any(stats[1:,cv2.CC_STAT_AREA]>=16*max(1.,min(2.,.3*prior.area))));r['iou'][name]=0.
            records.append(r)
    print(a.group,loc.name,len(records),round(time.time()-t,1),flush=True);Path(f'training18/shape_scores_{a.group}.json').write_text(json.dumps(records))
assert len(records)==1601
