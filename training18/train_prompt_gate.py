"""Visibility features from an ensemble of historical and expanded SAM prompts."""
import sys,json,time
from pathlib import Path
import numpy as np,joblib
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import ExtraTreesClassifier,HistGradientBoostingClassifier
from sklearn.metrics import f1_score
sys.path.insert(0,'training17');from features import entry_features

root=Path('training18');t=time.time();cache=root/'features_prompts.npz'
if cache.exists():
    z=np.load(cache);X,y,keys=z['X'],z['y'],z['keys']
else:
    xx=[];yy=[];kk=[]
    for loc in sorted(Path('train').iterdir()):
        dates=sorted(p.stem for p in (loc/'bands').glob('*.npy'));arrays=[np.load(loc/'bands'/f'{d}.npy') for d in dates];sites=json.loads((loc/'sites.json').read_text());sam={}
        labels={(d,r['site_id']):r for d in dates for r in json.loads((loc/'labels'/f'{d}.json').read_text())}
        for d in dates:
            z0=np.load(Path('training17/masks')/f'{loc.name}_{d}.npz');z1=np.load(root/'prompts'/f'{loc.name}_{d}.npz');ids0={s:i for i,s in enumerate(z0['sites'])};ids1={s:i for i,s in enumerate(z1['sites'])}
            old=z0['tuned'].astype(np.float32);wp=z1['widepoint'].astype(np.float32);wb=z1['widebox'].astype(np.float32);c0=z0['tuned_conf'];c1=z1['widepoint_conf'];c2=z1['widebox_conf'];sam[d]={}
            for s in sites:
                sid=s['site_id'];i=ids0[sid];j=ids1[sid];sam[d][sid]=(((old[i]+wp[j]+wb[j])/3).astype(np.float16),float((c0[i]+c1[j]+c2[j])/3))
        f,ks=entry_features(sites,dates,arrays,sam)
        for row,(d,sid) in zip(f,ks):
            label=labels.get((d,sid))
            if not label or label.get('ignore'):continue
            xx.append(row);yy.append(int(label['visible']));kk.append((loc.name,d,sid))
        print('features',loc.name,len(xx),round(time.time()-t,1),flush=True)
    X=np.asarray(xx);y=np.asarray(yy);keys=np.asarray(kk);np.savez_compressed(cache,X=X,y=y,keys=keys)
for name in ['prompt_et','prompt_hg']:
    for split in ['scene','window']:
        path=root/f'oof_{name}_{split}.npz'
        if path.exists():continue
        groups=np.array([k[0] if split=='window' else k[0].split('_w')[0] for k in keys]);p=np.zeros(len(y))
        for fold,(tr,va) in enumerate(GroupKFold(5).split(X,y,groups)):
            model=ExtraTreesClassifier(n_estimators=500,min_samples_leaf=3,max_features=.6,class_weight='balanced',n_jobs=3,random_state=1701) if name.endswith('et') else HistGradientBoostingClassifier(max_iter=300,max_leaf_nodes=12,l2_regularization=5.,min_samples_leaf=20,learning_rate=.04,random_state=1701)
            model.fit(X[tr],y[tr]);p[va]=model.predict_proba(X[va])[:,1]
            print(name,split,fold+1,round(time.time()-t,1),flush=True)
        np.savez_compressed(path,p=p,y=y,keys=keys);print('F1',name,split,[(th,round(f1_score(y,p>=th,average='macro'),4)) for th in [.25,.35,.45]],flush=True)
    model.fit(X,y);joblib.dump(model,root/f'{name}.joblib',compress=3)
