import sys,time
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupKFold
sys.path.insert(0,'12_submission')
from train_presence import load_samples,make_model

t=time.time();dest=Path('training17/features_baseline.npz')
if dest.exists():
    d=np.load(dest);X,y,groups,keys=d['X'],d['y'],d['groups'],d['keys']
else:
    X,y,groups,keys=load_samples('train');keys=np.asarray(keys);np.savez_compressed(dest,X=X,y=y,groups=groups,keys=keys)
print('baseline features',X.shape,flush=True);p=np.zeros(len(y))
for i,(tr,va) in enumerate(GroupKFold(5).split(X,y,groups)):
    model=make_model();model.set_params(et__n_jobs=2);model.fit(X[tr],y[tr]);p[va]=model.predict_proba(X[va])[:,1]
    print('baseline fold',i+1,round(time.time()-t,1),flush=True)
np.savez_compressed('training17/oof_baseline_window.npz',p=p,y=y,keys=keys)
