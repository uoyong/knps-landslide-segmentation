import json,time,sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from baseline import polygons_to_geom,px_polys_from_mask
from shapely.ops import unary_union

def poly(logit,prior,threshold=1.):
    l=F.interpolate(torch.from_numpy(logit.astype(np.float32))[None,None],(1024,1024),mode='bilinear',align_corners=False)[0,0].numpy()
    polys=px_polys_from_mask(l>threshold,max(1.,min(2.,.3*prior.area)))
    return unary_union(polys).simplify(.05) if polys else None

def main():
    torch.set_num_threads(2);t=time.time();out=[]
    dest=Path('training17/shape_scores.json')
    if dest.exists():out=json.loads(dest.read_text())
    seen={tuple(r['key']) for r in out}
    for loc in sorted(Path('train').iterdir()):
        sites={s['site_id']:polygons_to_geom(s['prior_polygon']) for s in json.loads((loc/'sites.json').read_text())}
        for lp in sorted((loc/'labels').glob('*.json')):
            p=Path('training17/masks')/f'{loc.name}_{lp.stem}.npz'
            if not p.exists():continue
            d0=np.load(p);d={k:d0[k] for k in d0.files};index={s:i for i,s in enumerate(d['sites'])}
            for row in json.loads(lp.read_text()):
                if row.get('ignore') or row['site_id'] not in index:continue
                if (loc.name,lp.stem,row['site_id']) in seen:continue
                sid=row['site_id'];i=index[sid];prior=sites[sid];gt=polygons_to_geom(row.get('polygon')) if row['visible'] else None
                r={'key':[loc.name,lp.stem,sid],'y':int(row['visible']),'area':gt.area if gt else 0.,'iou':{},'nonempty':{}}
                for name in ['tuned','pretrained','blend']:
                    logit=(d['tuned'][i].astype(np.float32)+d['pretrained'][i].astype(np.float32))/2 if name=='blend' else d[name][i]
                    for th in ([.0,1.,2.] if name=='tuned' else [1.]):
                        key=f'{name}_{th:g}'
                        if gt is not None and gt.area>=10:
                            g=poly(logit,prior,th);r['nonempty'][key]=bool(g is not None)
                            r['iou'][key]=g.intersection(gt).area/g.union(gt).area if g is not None else 0.
                        else:
                            big=cv2.resize(logit.astype(np.float32),(1024,1024),interpolation=cv2.INTER_LINEAR)>th
                            _,_,stats,_=cv2.connectedComponentsWithStats(big.astype(np.uint8),connectivity=4)
                            r['nonempty'][key]=bool(np.any(stats[1:,cv2.CC_STAT_AREA]>=16*max(1.,min(2.,.3*prior.area))))
                            r['iou'][key]=0.
                out.append(r)
        print(loc.name,'rows',len(out),'elapsed',round(time.time()-t,1),flush=True)
        dest.write_text(json.dumps(out))
    dest.write_text(json.dumps(out))
    names=list(out[0]['iou'])
    for name in names:
        groups={}
        for r in out:
            if r['y'] and r['area']>=10:groups.setdefault((r['key'][0],r['key'][2]),[]).append(r['iou'][name])
        print(name,'oracle presence site mIoU',np.mean([np.mean(v) for v in groups.values()]),flush=True)

if __name__=='__main__':main()
