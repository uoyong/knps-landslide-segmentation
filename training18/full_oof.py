"""Reproduce OOF visibility predictions on every date, including ignored labels.

Ignored labels are excluded from fitting and scoring, but their images must remain
in the temporal inference sequence exactly as they do in the submitted notebook.
"""
import sys,json,time
from pathlib import Path
import numpy as np,joblib
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
sys.path.insert(0,'training17');from features import entry_features

root=Path('training18');cache=root/'all_features.npz';t=time.time()
if cache.exists():
    z=np.load(cache);Xall,allkeys=z['X'],z['keys']
else:
    XX=[];kk=[]
    for loc in sorted(Path('train').iterdir()):
        dates=sorted(p.stem for p in (loc/'bands').glob('*.npy'));sites=json.loads((loc/'sites.json').read_text());arrays=[np.load(loc/'bands'/f'{d}.npy') for d in dates];sam={}
        for d in dates:
            z0=np.load(Path('training17/masks')/f'{loc.name}_{d}.npz');log=z0['tuned'];conf=z0['tuned_conf'];sam[d]={s:(log[i],conf[i]) for i,s in enumerate(z0['sites'])}
        f,k=entry_features(sites,dates,arrays,sam);XX.extend(f);kk.extend([(loc.name,d,sid) for d,sid in k]);print('full features',loc.name,len(XX),flush=True)
    Xall=np.asarray(XX);allkeys=np.asarray(kk);np.savez_compressed(cache,X=Xall,keys=allkeys)
base=np.load('training17/features_sam.npz');keys=base['keys'];y=base['y'];lookup={tuple(k):i for i,k in enumerate(allkeys)};index=np.array([lookup[tuple(k)] for k in keys]);assert np.array_equal(Xall[index],base['X'])
for split in ['scene','window']:
    groups=np.array([k[0] if split=='window' else k[0].split('_w')[0] for k in keys]);allgroups=np.array([k[0] if split=='window' else k[0].split('_w')[0] for k in allkeys])
    for name in ['sam_et','sam_hg','spectral_et','spectral_hg']:
        dest=root/f'oof_all_{name}_{split}.npz'
        if dest.exists():continue
        xa=Xall if name.startswith('sam') else Xall[:,np.concatenate([np.arange(257)+367*j for j in range(4)])]
        xx=xa[index];old=joblib.load(Path('17_submission/assets')/f'{name}.joblib');p=np.zeros(len(allkeys))
        for fold,(tr,va) in enumerate(GroupKFold(5).split(xx,y,groups)):
            model=clone(old)
            if name.endswith('et'):model.set_params(n_jobs=2)
            model.fit(xx[tr],y[tr]);test=np.flatnonzero(np.isin(allgroups,np.unique(groups[va])));p[test]=model.predict_proba(xa[test])[:,1]
            print('full oof',name,split,fold+1,round(time.time()-t,1),flush=True)
        oldp=np.load(Path('training17')/f'oof_{name}_{split}.npz')['p'];error=float(np.max(np.abs(p[index]-oldp)));assert error<1e-8,(name,split,error)
        np.savez_compressed(dest,p=p,keys=allkeys,valid_index=index,y=y);print('matched original OOF',name,split,error,flush=True)
