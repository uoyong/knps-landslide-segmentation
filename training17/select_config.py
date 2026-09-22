"""Compare candidates using the published macro-F1 / per-site polygon IoU metric."""
import json,sys
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score,confusion_matrix
sys.path.insert(0,'12_submission')
from train_presence import smooth_site_probabilities

root=Path('training17');shapes=json.loads((root/'shape_scores.json').read_text());shape_by={tuple(r['key']):r for r in shapes}
base=np.load(root/'oof_baseline_window.npz');keys=base['keys'];y=base['y'];rows=[shape_by[tuple(k)] for k in keys]
assert len(rows)==1601
site_groups={}
for i,r in enumerate(rows):
    if r['y'] and r['area']>=10:site_groups.setdefault((r['key'][0],r['key'][2]),[]).append(i)
area_weights=np.zeros(len(y))
for ids in site_groups.values():area_weights[ids]=1/len(ids)/len(site_groups)

rows_by_split={s:[dict(r,iou=dict(r['iou']),nonempty=dict(r['nonempty'])) for r in rows] for s in ['window','scene']}
shape_names=['tuned_1','tuned_2']
if all((root/f'refined_scores_{s}.json').exists() and len(json.loads((root/f'refined_scores_{s}.json').read_text()))==1601 for s in ['window','scene']):
    for s in ['window','scene']:
        data={tuple(r['key']):r for r in json.loads((root/f'refined_scores_{s}.json').read_text())}
        for k,r in zip(keys,rows_by_split[s]):
            r['iou'].update(data[tuple(k)]['iou']);r['nonempty'].update(data[tuple(k)]['nonempty'])
    shape_names += ['pixel_0.5_0.5','pixel_0.75_0.4','pixel_1.0_0.4']

def metrics(p,th,shape='tuned_1',split='window'):
    rs=rows_by_split[split]
    pred=(p>=th)&np.array([r['nonempty'][shape] for r in rs]);ious=np.array([r['iou'][shape] for r in rs])
    f1=f1_score(y,pred,average='macro');miou=float(np.sum(area_weights*ious*pred));return {'score':.6*f1+.4*miou,'macro_f1':f1,'site_miou':miou,'confusion':confusion_matrix(y,pred).tolist()}

probs={}
for split in ['window','scene']:
    probs[split]={}
    for path in root.glob(f'oof_*_{split}.npz'):
        d=np.load(path);lookup={tuple(k):float(p) for k,p in zip(d['keys'],d['p'])};name=path.stem.removeprefix('oof_').removesuffix('_'+split)
        probs[split][name]=np.array([lookup[tuple(k)] for k in keys])
    if split=='scene':
        d=np.load('/tmp/baseline_scene_oof.npz',allow_pickle=True);lookup={tuple(k):float(p) for k,p in zip(d['keys'],d['p'])};probs[split]['baseline']=np.array([lookup[tuple(k)] for k in keys])

candidates=[{'baseline':1.},{'spectral_et':.5,'spectral_hg':.5},{'spectral_et':.667,'spectral_hg':.333},{'spectral_et':.333,'spectral_hg':.667},{'sam_et':.5,'sam_hg':.5},{'sam_et':.667,'sam_hg':.333},{'spectral_et':.25,'spectral_hg':.25,'sam_et':.25,'sam_hg':.25},{'spectral_et':.125,'spectral_hg':.125,'sam_et':.375,'sam_hg':.375},{'baseline':.2,'spectral_et':.2,'spectral_hg':.2,'sam_et':.2,'sam_hg':.2}]
results=[]
for weights in candidates:
    if any(n not in probs[s] for n in weights for s in ['window','scene']):continue
    for alpha in [.0,.45,.75]:
        ps={s:smooth_site_probabilities(sum(w*probs[s][n] for n,w in weights.items()),keys,alpha) for s in probs}
        for th in [.25,.30,.35,.40,.45,.50,.55]:
            for shape in shape_names:
                scores={s:metrics(p,th,shape,s) for s,p in ps.items()}
                pixel_weight=float(shape.split('_')[1]) if shape.startswith('pixel') else 0.
                pixel_threshold=float(shape.split('_')[2]) if shape.startswith('pixel') else .5
                results.append({'weights':weights,'smooth_alpha':alpha,'threshold':th,'mask_threshold':float(shape[-1]) if shape.startswith('tuned') else 1.,'pixel_weight':pixel_weight,'pixel_threshold':pixel_threshold,'scores':scores,'selection_score':.65*scores['window']['score']+.35*scores['scene']['score']})
results.sort(key=lambda r:r['selection_score'],reverse=True)
baseline={s:metrics(smooth_site_probabilities(probs[s]['baseline'],keys,.45),.35,split=s) for s in probs}
print('BASELINE',json.dumps(baseline),flush=True)
for r in results[:8]:print(json.dumps(r),flush=True)
(root/'selection_report.json').write_text(json.dumps({'metric_source':'https://aifactory.space/en/task/9305/data','baseline':baseline,'candidates':results},indent=2))
if results:
    best=results[0];config={k:best[k] for k in ['weights','smooth_alpha','threshold','mask_threshold','pixel_weight','pixel_threshold']}
    (root/'candidate_config.json').write_text(json.dumps(config,indent=2))
