"""Exact held-out geometry and empty-mask behavior for SAM feature refinement."""
import sys,json,argparse,time
from pathlib import Path
sys.path.insert(0,str(Path('training18/vendor').resolve()));sys.path.insert(0,'training17')
import cv2,numpy as np,joblib,torch
from catboost import CatBoostClassifier
from shapely.ops import unary_union
from features import geom
from pixel_refine import patch_features,site_features
from baseline import px_polys_from_mask
from semantic_refine import semantic_features
from refine_features import enrich

ap=argparse.ArgumentParser();ap.add_argument('--group',choices=['scene','window'],required=True);a=ap.parse_args()
cv2.setNumThreads(1);torch.set_num_threads(2);root=Path('training18');records=[];t=time.time();models={};oldmodels={};normmodels={}
for p in root.glob(f'semantic_{a.group}_fold*.cbm'):
    m=CatBoostClassifier();m.load_model(str(p))
    for loc in json.loads(p.with_suffix('.json').read_text()):models[loc]=m
for folder,prefix,out in [('training17','pixel',oldmodels),('training18','refiner',normmodels)]:
    for p in Path(folder).glob(f'{prefix}_{a.group}_fold*.joblib'):
        b=joblib.load(p)
        for loc in b['held_out_windows']:out[loc]=b['model']
assert len(models)==len(oldmodels)==len(normmodels)==14
for loc in sorted(Path('train').iterdir()):
    dates=sorted(p.stem for p in (loc/'bands').glob('*.npy'));arrays=[np.load(loc/'bands'/f'{d}.npy') for d in dates];ref=np.median(np.stack(arrays),0);sites={s['site_id']:geom(s['prior_polygon']) for s in json.loads((loc/'sites.json').read_text())}
    # Region 31 has no positive pixels in any training fold.
    model=models.get(loc.name,next(iter(models.values())));old=oldmodels.get(loc.name,next(iter(oldmodels.values())));norm=normmodels.get(loc.name,next(iter(normmodels.values())))
    for date,bands in zip(dates,arrays):
        z0=np.load(Path('training17/masks')/f'{loc.name}_{date}.npz');idx={s:i for i,s in enumerate(z0['sites'])};patch=None;neck=None
        for row in json.loads((loc/'labels'/f'{date}.json').read_text()):
            if row.get('ignore'):continue
            sid=row['site_id'];prior=sites[sid];gt=geom(row['polygon']) if row['visible'] else None;name=f'{loc.name}_{date}_{sid}.npz';cp=Path('training17/pixel_data')/name
            if cp.exists():
                d=np.load(cp);X=d['X'];roi=d['roi'];ps=np.load(root/f'semantic_{a.group}'/name)['prob'].astype(np.float32);pn=np.load(root/f'refined_{a.group}'/name)['prob'].astype(np.float32)
                po=np.load(Path('training17')/('pixel_oof' if a.group=='window' else 'pixel_oof_scene')/name)['prob'].astype(np.float32)
            else:
                if patch is None:patch=patch_features(bands,ref)
                X,roi,_,_=site_features(patch,z0['tuned'][idx[sid]],prior)
                if neck is None:
                    emb=torch.load(Path('cache_embs')/f'{loc.name}_{date}.pt',map_location='cpu',weights_only=True);neck=((emb['orig'][0][0].float()+emb['hflip'][0][0].float().flip(-1))/2).numpy()
                r0,r1,c0,c1=roi;shape=(r1-r0,c1-c0)
                ps=model.predict_proba(semantic_features(X,roi,neck),thread_count=2)[:,1].astype(np.float16).astype(np.float32).reshape(shape);po=old.predict_proba(X)[:,1].reshape(shape);pn=norm.predict_proba(enrich(X))[:,1].reshape(shape)
            r0,r1,c0,c1=roi;r={'key':[loc.name,date,sid],'y':int(row['visible']),'area':gt.area if gt else 0.,'iou':{},'nonempty':{}}
            for nw,sw in [(0.,.5),(.5,.5),(1.,.5),(0.,1.)]:
                pp=(1-sw)*((1-nw)*po+nw*pn)+sw*ps;p=np.zeros((512,512),np.float32);p[r0:r1,c0:c1]=pp;big=cv2.resize(p,(1024,1024))
                for th in [.35,.4,.5,.6]:
                    v=f'semantic_{nw}_{sw}_{th}';mask=big>th
                    if gt is not None and gt.area>=10:
                        parts=px_polys_from_mask(mask,max(1.,min(2.,.3*prior.area)));g=unary_union(parts).simplify(.05) if parts else None;r['nonempty'][v]=g is not None;r['iou'][v]=g.intersection(gt).area/g.union(gt).area if g is not None else 0.
                    else:
                        _,_,stats,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8),connectivity=4);r['nonempty'][v]=bool(np.any(stats[1:,cv2.CC_STAT_AREA]>=16*max(1.,min(2.,.3*prior.area))));r['iou'][v]=0.
            records.append(r)
    print(a.group,loc.name,len(records),round(time.time()-t,1),flush=True);(root/f'semantic_shapes_{a.group}.json').write_text(json.dumps(records))
assert len(records)==1601
