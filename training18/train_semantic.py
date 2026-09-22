import sys,json,time,argparse
from pathlib import Path
sys.path.insert(0,str(Path('training18/vendor').resolve()));sys.path.insert(0,'training17')
import numpy as np,torch,joblib,shapely
from catboost import CatBoostClassifier
from sklearn.model_selection import GroupKFold
from features import geom
from semantic_refine import semantic_features

ap=argparse.ArgumentParser();ap.add_argument('--group',choices=['scene','window'],default='scene');ap.add_argument('--device',default='GPU');a=ap.parse_args();root=Path('training18');dest=root/'semantic_data';dest.mkdir(exist_ok=True);rng=np.random.default_rng(1804);t=time.time();XX=[];yy=[];ww=[];groups=[];items=[]
torch.set_num_threads(2);gy,gx=np.mgrid[:512,:512];gx=(gx+.5)/2;gy=(gy+.5)/2
for loc in sorted(Path('train').iterdir()):
    for lp in sorted((loc/'labels').glob('*.json')):
        neck=None
        for row in json.loads(lp.read_text()):
            if row.get('ignore') or not row.get('visible'):continue
            sid=row['site_id'];old=Path('training17/pixel_data')/f'{loc.name}_{lp.stem}_{sid}.npz';z=np.load(old);cp=dest/old.name
            if not cp.exists():
                if neck is None:
                    emb=torch.load(Path('cache_embs')/f'{loc.name}_{lp.stem}.pt',map_location='cpu',weights_only=True)
                    neck=((emb['orig'][0][0].float()+emb['hflip'][0][0].float().flip(-1))/2).numpy()
                X=semantic_features(z['X'],z['roi'],neck);np.savez_compressed(cp,X=X)
            else:X=np.load(cp)['X']
            target=z['y'];pos=np.flatnonzero(target);neg=np.flatnonzero(~target);tp=rng.choice(pos,min(128,len(pos)),replace=False);tn=rng.choice(neg,min(256,len(neg)),replace=False);take=np.r_[tp,tn]
            weights=np.r_[np.full(len(tp),len(pos)/max(len(tp),1)),np.full(len(tn),len(neg)/max(len(tn),1))]*300/len(target)
            XX.append(X[take]);yy.append(target[take]);ww.append(weights);groups.extend([loc.name]*len(take));g=geom(row['polygon']);items.append(([loc.name,lp.stem,sid],old,cp,g.area,int(shapely.contains_xy(g,gx,gy).sum())))
    print('semantic features',loc.name,len(items),round(time.time()-t,1),flush=True)
X=np.concatenate(XX);y=np.concatenate(yy);weights=np.concatenate(ww);groups=np.array(groups);cvgroups=groups if a.group=='window' else np.array([g.split('_w')[0] for g in groups]);print('training',a.group,a.device,X.shape,flush=True)
records=[];pred_dir=root/f'semantic_{a.group}';pred_dir.mkdir(exist_ok=True)
for fold,(tr,va) in enumerate(GroupKFold(5).split(X,y,cvgroups)):
    model=CatBoostClassifier(iterations=450,depth=6,learning_rate=.06,l2_leaf_reg=15,random_seed=1805,thread_count=3,verbose=False,allow_writing_files=False,border_count=64,loss_function='Logloss',class_weights=[1.,2.],task_type=a.device)
    model.fit(X[tr],y[tr],sample_weight=weights[tr]);vg=sorted(set(groups[va]));model.save_model(str(root/f'semantic_{a.group}_fold{fold}.cbm'));(root/f'semantic_{a.group}_fold{fold}.json').write_text(json.dumps(vg))
    for key,old,cp,area,count in items:
        if key[0] not in vg:continue
        z=np.load(old);roi=z['roi'];r0,r1,c0,c1=roi;p=model.predict_proba(np.load(cp)['X'],thread_count=2)[:,1].astype(np.float32);np.savez_compressed(pred_dir/old.name,prob=p.reshape(r1-r0,c1-c0).astype(np.float16),roi=roi)
        oldp=np.load(Path('training17')/('pixel_oof' if a.group=='window' else 'pixel_oof_scene')/old.name)['prob'].astype(np.float32).ravel();gt=z['y'];r={'key':key,'area':area,'iou':{}}
        for w in [.5,1.]:
            mix=w*p+(1-w)*oldp
            for th in [.3,.4,.5,.6]:
                m=mix>th;inter=(m & gt).sum();r['iou'][f'{w}_{th}']=float(inter/max(m.sum()+count-inter,1))
        records.append(r)
    print('semantic',a.group,'fold',fold+1,round(time.time()-t,1),flush=True)
(root/f'semantic_scores_{a.group}.json').write_text(json.dumps(records))
for name in records[0]['iou']:
    sites={}
    for r in records:
        if r['area']>=10:sites.setdefault((r['key'][0],r['key'][2]),[]).append(r['iou'][name])
    print(name,np.mean([np.mean(v) for v in sites.values()]),flush=True)
if a.group=='window':model.fit(X,y,sample_weight=weights);model.save_model(str(root/'semantic_refiner.cbm'))
