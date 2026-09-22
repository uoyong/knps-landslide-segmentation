import sys,json,time
from pathlib import Path
import cv2,numpy as np,shapely
sys.path.insert(0,'training17')
from features import geom

cv2.setNumThreads(1);records=[];t=time.time();yy,xx=np.mgrid[:512,:512];xx=(xx+.5)/2;yy=(yy+.5)/2
for loc in sorted(Path('train').iterdir()):
    for lp in sorted((loc/'labels').glob('*.json')):
        cp=Path('training18/prompts')/f'{loc.name}_{lp.stem}.npz'
        if not cp.exists():continue
        old0=np.load(Path('training17/masks')/cp.name);new0=np.load(cp);d={k:new0[k] for k in new0.files};d['tuned']=old0['tuned'];idx={s:i for i,s in enumerate(d['sites'])}
        for row in json.loads(lp.read_text()):
            if row.get('ignore') or not row.get('visible'):continue
            gt=geom(row['polygon']);truth=shapely.contains_xy(gt,xx,yy);i=idx[row['site_id']];r={'key':[loc.name,lp.stem,row['site_id']],'area':gt.area,'iou':{}}
            for name in ['tuned','widepoint','widebox','pointmean','boxmean','allmean']:
                logits={'tuned':d['tuned'][i],'widepoint':d['widepoint'][i],'widebox':d['widebox'][i],
                        'pointmean':(d['tuned'][i].astype(np.float32)+d['widepoint'][i].astype(np.float32))/2,
                        'boxmean':(d['tuned'][i].astype(np.float32)+d['widebox'][i].astype(np.float32))/2,
                        'allmean':(d['tuned'][i].astype(np.float32)+d['widepoint'][i].astype(np.float32)+d['widebox'][i].astype(np.float32))/3}[name]
                big=cv2.resize(logits.astype(np.float32),(512,512))
                for th in [0.,1.,2.]:
                    m=big>th;r['iou'][f'{name}_{th}']=float((m & truth).sum()/max((m|truth).sum(),1))
            records.append(r)
    print(loc.name,len(records),round(time.time()-t,1),flush=True)
Path('training18/prompt_scores.json').write_text(json.dumps(records))
for variant in records[0]['iou']:
    sites={}
    for r in records:
        if r['area']>=10:sites.setdefault((r['key'][0],r['key'][2]),[]).append(r['iou'][variant])
    print(variant,np.mean([np.mean(v) for v in sites.values()]),flush=True)
