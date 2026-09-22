"""Stratified boundary learning with radiometric normalization."""
import json,time,argparse,sys
from pathlib import Path
import numpy as np,joblib,shapely
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from refine_features import enrich
sys.path.insert(0,'training17')
from features import geom

ap=argparse.ArgumentParser();ap.add_argument('--group',choices=['scene','window'],default='scene');args=ap.parse_args()
root=Path('training18');rng=np.random.default_rng(1802);t=time.time();items=[];XX=[];yy=[];ww=[];groups=[]
gy,gx=np.mgrid[:512,:512];gx=(gx+.5)/2;gy=(gy+.5)/2
for loc in sorted(Path('train').iterdir()):
    for lp in sorted((loc/'labels').glob('*.json')):
        for row in json.loads(lp.read_text()):
            if row.get('ignore') or not row.get('visible'):continue
            sid=row['site_id'];p=Path('training17/pixel_data')/f'{loc.name}_{lp.stem}_{sid}.npz';d=np.load(p);target=d['y'];f=enrich(d['X']);pos=np.flatnonzero(target);neg=np.flatnonzero(~target)
            takep=rng.choice(pos,min(128,len(pos)),replace=False);taken=rng.choice(neg,min(256,len(neg)),replace=False);take=np.r_[takep,taken]
            # Importance weights restore the per-image foreground prior after oversampling.
            w=np.r_[np.full(len(takep),len(pos)/max(len(takep),1)),np.full(len(taken),len(neg)/max(len(taken),1))]*300/len(target)
            XX.append(f[take]);yy.append(target[take]);ww.append(w);groups.extend([loc.name]*len(take))
            gt=geom(row['polygon']);items.append(([loc.name,lp.stem,sid],p,float(gt.area),int(shapely.contains_xy(gt,gx,gy).sum())))
    print('features',loc.name,len(items),round(time.time()-t,1),flush=True)
X=np.concatenate(XX);y=np.concatenate(yy);weights=np.concatenate(ww);groups=np.array(groups)
cvgroups=groups if args.group=='window' else np.array([g.split('_w')[0] for g in groups]);print('train',X.shape,'sample foreground',y.mean(),flush=True)
records=[];pred_dir=root/f'refined_{args.group}';pred_dir.mkdir(exist_ok=True)
for fold,(tr,va) in enumerate(GroupKFold(5).split(X,y,cvgroups)):
    model=HistGradientBoostingClassifier(max_iter=300,max_leaf_nodes=24,learning_rate=.055,l2_regularization=10,min_samples_leaf=50,max_bins=127,random_state=1803)
    model.fit(X[tr],y[tr],sample_weight=weights[tr]);valgroups=set(groups[va]);joblib.dump({'model':model,'held_out_windows':sorted(valgroups)},root/f'refiner_{args.group}_fold{fold}.joblib',compress=3)
    for key,path,area,count in items:
        if key[0] not in valgroups:continue
        z=np.load(path);prob=model.predict_proba(enrich(z['X']))[:,1].astype(np.float32);r0,r1,c0,c1=z['roi'];np.savez_compressed(pred_dir/path.name,prob=prob.reshape(r1-r0,c1-c0).astype(np.float16),roi=z['roi'])
        old=np.load(Path('training17')/('pixel_oof' if args.group=='window' else 'pixel_oof_scene')/path.name)['prob'].astype(np.float32).ravel();gt=z['y'];r={'key':key,'area':area,'ious':{}}
        for w in [.5,1.]:
            mix=w*prob+(1-w)*old
            for th in [.25,.35,.4,.45,.5]:
                m=mix>th;inter=(m & gt).sum();r['ious'][f'{w}_{th}']=float(inter/max(m.sum()+count-inter,1))
        records.append(r)
    print(args.group,'fold',fold+1,'elapsed',round(time.time()-t,1),flush=True)
(root/f'refiner_scores_{args.group}.json').write_text(json.dumps(records))
for v in records[0]['ious']:
    sites={}
    for r in records:
        if r['area']>=10:sites.setdefault((r['key'][0],r['key'][2]),[]).append(r['ious'][v])
    print(v,np.mean([np.mean(z) for z in sites.values()]),flush=True)
if args.group=='window':
    model.fit(X,y,sample_weight=weights);joblib.dump(model,root/'refiner.joblib',compress=3)
