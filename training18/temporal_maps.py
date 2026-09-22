"""Cache label-free per-date maps for spectral-neighbour temporal fusion."""
import sys,json,time,argparse
from pathlib import Path
import cv2,numpy as np,joblib
sys.path.insert(0,'training17')
from features import geom,mask,channels,stats
from pixel_refine import patch_features,site_features

ap=argparse.ArgumentParser();ap.add_argument('--group',choices=['scene','window'],default='scene');a=ap.parse_args();root=Path('training18')/f'temporal_{a.group}';root.mkdir(exist_ok=True);cv2.setNumThreads(1);models={};t=time.time()
for p in Path('training17').glob(f'pixel_{a.group}_fold*.joblib'):
    b=joblib.load(p)
    for loc in b['held_out_windows']:models[loc]=b['model']
fallback=joblib.load('17_submission/assets/pixel_refiner.joblib')
for loc in sorted(Path('train').iterdir()):
    dest=root/f'{loc.name}.npz'
    if dest.exists():continue
    dates=sorted(p.stem for p in (loc/'bands').glob('*.npy'));arrays=[np.load(loc/'bands'/f'{d}.npy') for d in dates];ref=np.median(np.stack(arrays),0);sites=json.loads((loc/'sites.json').read_text());gs=[geom(s['prior_polygon']) for s in sites]
    regions=[(mask(g.buffer(3)),mask(g.buffer(10).difference(g.buffer(4)))) for g in gs]
    maps=np.zeros((len(sites),len(dates),512,512),np.float16);desc=np.zeros((len(sites),len(dates),16),np.float32);model=models.get(loc.name,fallback)
    for di,(date,bands) in enumerate(zip(dates,arrays)):
        patch=patch_features(bands,ref);c=channels(bands);z0=np.load(Path('training17/masks')/f'{loc.name}_{date}.npz');ids={s:i for i,s in enumerate(z0['sites'])};logits=z0['tuned']
        for si,(s,g,(inside,ring)) in enumerate(zip(sites,gs,regions)):
            old=Path('training17/pixel_data')/f'{loc.name}_{date}_{s["site_id"]}.npz'
            if old.exists():d=np.load(old);X=d['X'];roi=d['roi']
            else:X,roi,_,_=site_features(patch,logits[ids[s['site_id']]],g)
            r0,r1,c0,c1=roi;maps[si,di,r0:r1,c0:c1]=model.predict_proba(X)[:,1].reshape(r1-r0,c1-c0).astype(np.float16)
            ii=stats(c,inside)[:8];rr=stats(c,ring)[:8];desc[si,di]=np.r_[ii,ii-rr]
    np.savez_compressed(dest,maps=maps,desc=desc,dates=np.array(dates),sites=np.array([s['site_id'] for s in sites]));print(a.group,loc.name,maps.shape[:2],'elapsed',round(time.time()-t,1),flush=True)
