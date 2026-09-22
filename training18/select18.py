"""Joint evaluation of visibility and exact refined geometry."""
import sys,json
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score,confusion_matrix
from gate_study import markov
sys.path.insert(0,'12_submission');from train_presence import smooth_site_probabilities

root=Path('training18');base=np.load('training17/features_sam.npz');keys=base['keys'];y=base['y'];rows={};probs={}
for split in ['window','scene']:
    rr=json.load(open(root/f'shape_scores_{split}.json'));assert len(rr)==1601
    mapping={tuple(r['key']):r for r in rr};rows[split]=[mapping[tuple(k)] for k in keys];probs[split]={}
    for folder in ['training17','training18']:
        for path in Path(folder).glob(f'oof_*_{split}.npz'):
            d=np.load(path);lookup={tuple(k):float(p) for k,p in zip(d['keys'],d['p'])};probs[split][path.stem.removeprefix('oof_').removesuffix('_'+split)]=np.array([lookup[tuple(k)] for k in keys])
site_weights=np.zeros(len(y));sites={}
for i,r in enumerate(rows['window']):
    if r['area']>=10:sites.setdefault((r['key'][0],r['key'][2]),[]).append(i)
for ids in sites.values():site_weights[ids]=1/len(ids)/len(sites)
weights17={'spectral_et':.125,'spectral_hg':.125,'sam_et':.375,'sam_hg':.375};ensembles={'v17':weights17}
for name in ['cat5','cat4_balanced']:
    if all(name in probs[s] for s in probs):
        for w in [.25,.5]:ensembles[f'{name}_{w}']={**{k:v*(1-w) for k,v in weights17.items()},name:w}
if all('prompt_et' in probs[s] and 'prompt_hg' in probs[s] for s in probs):
    for w in [.25,.5,.75,1.]:ensembles[f'prompts_{w}']={**{k:v*(1-w) for k,v in weights17.items()},'prompt_et':w/2,'prompt_hg':w/2}
variants=list(rows['window'][0]['iou']);arrays={s:{v:(np.array([r['nonempty'][v] for r in rows[s]]),np.array([r['iou'][v] for r in rows[s]])) for v in variants} for s in rows}
def evaluate(p,split,variant):
    nonempty,iou=arrays[split][variant];pred=p & nonempty;f=f1_score(y,pred,average='macro');miou=float(np.sum(site_weights*iou*pred));return {'score':.6*f+.4*miou,'macro_f1':f,'site_miou':miou,'confusion':confusion_matrix(y,pred).tolist()}
records=[]
for name,weights in ensembles.items():
    ps={s:sum(w*probs[s][n] for n,w in weights.items() if w) for s in rows}
    for method,strengths in [('local',[.45,.75]),('markov',[.5,1.,2.])]:
        for strength in strengths:
            for th in [.2,.25,.3,.35,.4]:
                binary={s:(markov(p,keys,th,strength)>=.5 if method=='markov' else smooth_site_probabilities(p,keys,strength)>=th) for s,p in ps.items()}
                for variant in variants:
                    scores={s:evaluate(binary[s],s,variant) for s in rows};_,w,pth=variant.split('_');record={'weights':{k:v for k,v in weights.items() if v},'gate_method':method,'gate_strength':strength,'threshold':th,'refiner_mix':float(w),'pixel_threshold':float(pth),'scores':scores,'selection':.4*scores['window']['score']+.6*scores['scene']['score']};records.append(record)
records.sort(key=lambda r:r['selection'],reverse=True)
baseline=next(r for r in records if r['weights']==weights17 and r['gate_method']=='local' and r['gate_strength']==.45 and r['threshold']==.3 and r['refiner_mix']==0. and r['pixel_threshold']==.4)
print('BASELINE17',json.dumps(baseline),flush=True)
for r in records[:8]:print(json.dumps(r),flush=True)
(root/'selection_report.json').write_text(json.dumps({'baseline17':baseline,'candidates':records},indent=2))
best=records[0];config={k:best[k] for k in ['weights','gate_method','gate_strength','threshold','refiner_mix','pixel_threshold']};(root/'candidate_config.json').write_text(json.dumps(config,indent=2))
