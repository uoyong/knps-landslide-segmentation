import sys,json,argparse,time
from pathlib import Path
import numpy as np,shapely
from temporal_fusion import fuse
sys.path.insert(0,'training17');from features import geom
ap=argparse.ArgumentParser();ap.add_argument('--group',default='scene');a=ap.parse_args();records=[];yy,xx=np.mgrid[:512,:512];xx=(xx+.5)/2;yy=(yy+.5)/2
for cp in sorted((Path('training18')/f'temporal_{a.group}').glob('*.npz')):
    z=np.load(cp);maps=z['maps'];desc=z['desc'];dates=z['dates'];sites=z['sites'];loc=Path('train')/cp.stem
    labels={d:{r['site_id']:r for r in json.loads((loc/'labels'/f'{d}.json').read_text())} for d in dates}
    for si,sid in enumerate(sites):
        variants={'base':maps[si].astype(np.float32)}
        for kind in ['spectral','time']:
            for alpha in [.25,.5,.75]:variants[f'{kind}_{alpha}']=fuse(maps[si],desc[si],alpha,kind)
        for di,d in enumerate(dates):
            row=labels[d].get(sid)
            if not row or row.get('ignore') or not row.get('visible'):continue
            g=geom(row['polygon']);gt=shapely.contains_xy(g,xx,yy);r={'key':[cp.stem,str(d),str(sid)],'area':g.area,'iou':{}}
            for name,ps in variants.items():
                for th in [.3,.4,.5]:
                    m=ps[di]>th;r['iou'][f'{name}_{th}']=float((m & gt).sum()/max((m|gt).sum(),1))
            records.append(r)
    print(cp.stem,len(records),flush=True)
Path(f'training18/temporal_scores_{a.group}.json').write_text(json.dumps(records))
for v in records[0]['iou']:
    sites={}
    for r in records:
        if r['area']>=10:sites.setdefault((r['key'][0],r['key'][2]),[]).append(r['iou'][v])
    print(v,np.mean([np.mean(x) for x in sites.values()]),flush=True)
