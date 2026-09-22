"""Alternative visibility learners, with window and region holdouts."""
import os,sys,json,time
from pathlib import Path
sys.path.insert(0,str(Path('training18/vendor').resolve()))
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score

root=Path('training18');t=time.time();d=np.load('training17/features_sam.npz');X,y,keys=d['X'],d['y'],d['keys']
windows=keys[:,0];scenes=np.array([s.split('_w')[0] for s in windows])
for name,depth,balance in [('cat5',5,False),('cat4_balanced',4,True)]:
    for split,groups in [('scene',scenes),('window',windows)]:
        dest=root/f'oof_{name}_{split}.npz'
        if dest.exists():continue
        p=np.zeros(len(y))
        for fold,(tr,va) in enumerate(GroupKFold(5).split(X,y,groups)):
            model=CatBoostClassifier(iterations=600,depth=depth,learning_rate=.04,l2_leaf_reg=10,random_seed=1801,thread_count=4,verbose=False,allow_writing_files=False,rsm=.6,border_count=64,loss_function='Logloss')
            weights=None
            if balance:
                unique,counts=np.unique(scenes[tr],return_counts=True);lookup=dict(zip(unique,counts));weights=np.array([1/np.sqrt(lookup[s]) for s in scenes[tr]]);weights/=weights.mean()
            model.fit(X[tr],y[tr],sample_weight=weights);p[va]=model.predict_proba(X[va])[:,1]
            print(name,split,'fold',fold+1,'elapsed',round(time.time()-t,1),flush=True)
        np.savez_compressed(dest,p=p,y=y,keys=keys)
        print(name,split,[(th,round(f1_score(y,p>=th,average='macro'),4)) for th in [.25,.35,.45,.55]],flush=True)
    weights=None
    if balance:
        unique,counts=np.unique(scenes,return_counts=True);lookup=dict(zip(unique,counts));weights=np.array([1/np.sqrt(lookup[s]) for s in scenes]);weights/=weights.mean()
    model.fit(X,y,sample_weight=weights);model.save_model(str(root/f'{name}.cbm'))
