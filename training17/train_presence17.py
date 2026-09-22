import argparse,json,time,sys
from pathlib import Path
import joblib,numpy as np
from sklearn.ensemble import ExtraTreesClassifier,HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
from features import entry_features

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--sam',action='store_true');args=ap.parse_args();tag='sam' if args.sam else 'spectral'
    out=Path('training17');cache=out/f'features_{tag}.npz';t=time.time()
    if cache.exists():
        data=np.load(cache);X,y,keys=data['X'],data['y'],data['keys']
    else:
        XX=[];yy=[];kk=[]
        for loc in sorted(Path('train').iterdir()):
            dates=sorted(p.stem for p in (loc/'bands').glob('*.npy'));sites=json.loads((loc/'sites.json').read_text())
            arrays=[np.load(loc/'bands'/f'{d}.npy') for d in dates]
            labels={(d,r['site_id']):r for d in dates for r in json.loads((loc/'labels'/f'{d}.json').read_text())}
            sam=None
            if args.sam:
                sam={}
                for d in dates:
                    p=out/'masks'/f'{loc.name}_{d}.npz'
                    if not p.exists():raise RuntimeError(f'Missing {p}')
                    z0=np.load(p);z={k:z0[k] for k in ['sites','tuned','tuned_conf']};sam[d]={sid:(z['tuned'][i],z['tuned_conf'][i]) for i,sid in enumerate(z['sites'])}
                    for s in sites:
                        if s['site_id'] not in sam[d]:raise RuntimeError(f'Incomplete label-free cache: {p} {s["site_id"]}')
            x,k=entry_features(sites,dates,arrays,sam)
            for row,(d,sid) in zip(x,k):
                lab=labels.get((d,sid))
                if lab is None or lab.get('ignore'):continue
                XX.append(row);yy.append(int(lab['visible']));kk.append((loc.name,d,sid))
            print('features',loc.name,len(XX),round(time.time()-t,1),flush=True)
        X=np.asarray(XX);y=np.asarray(yy);keys=np.asarray(kk);np.savez_compressed(cache,X=X,y=y,keys=keys)
    print('data',X.shape,flush=True)
    for grouping in ['window','scene']:
        groups=np.array([k[0] if grouping=='window' else k[0].split('_w')[0] for k in keys]);cv=GroupKFold(5)
        for typ in ['et','hg']:
            p=np.zeros(len(y));name=f'{tag}_{typ}_{grouping}';path=out/f'oof_{name}.npz'
            if path.exists():continue
            for fold,(tr,va) in enumerate(cv.split(X,y,groups)):
                model=(ExtraTreesClassifier(n_estimators=500,min_samples_leaf=3,max_features=.6,class_weight='balanced',n_jobs=4,random_state=1701) if typ=='et' else HistGradientBoostingClassifier(max_iter=300,max_leaf_nodes=12,l2_regularization=5.,min_samples_leaf=20,learning_rate=.04,random_state=1701))
                model.fit(X[tr],y[tr]);p[va]=model.predict_proba(X[va])[:,1]
                print(name,'fold',fold+1,'elapsed',round(time.time()-t,1),flush=True)
            np.savez_compressed(path,p=p,y=y,keys=keys)
            print(name,'F1',[(th,round(f1_score(y,p>=th,average='macro'),4)) for th in [.25,.35,.45,.55]],flush=True)
            if grouping=='window':
                model.fit(X,y);joblib.dump(model,Path('17_submission/assets')/f'{tag}_{typ}.joblib',compress=3)

if __name__=='__main__':main()
