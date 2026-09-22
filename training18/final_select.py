"""Match deployment temporal context, score only valid labels, compare to v17."""
import sys,json
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score,confusion_matrix
sys.path.insert(0,'17_submission/assets')
from inference import smooth
from temporal_gate import posterior

root=Path('training18');base=np.load('training17/features_sam.npz');keys=base['keys'];y=base['y'];weights={'spectral_et':.125,'spectral_hg':.125,'sam_et':.375,'sam_hg':.375};rows={};ps={}
for split in ['window','scene']:
    mapping={tuple(r['key']):r for r in json.loads((root/f'shape_scores_{split}.json').read_text())}
    for r in json.loads((root/f'semantic_shapes_{split}.json').read_text()):
        for field in ['iou','nonempty']:mapping[tuple(r['key'])][field].update(r[field])
    rows[split]=[mapping[tuple(k)] for k in keys];p=0
    for name,w in weights.items():
        d=np.load(root/f'oof_all_{name}_{split}.npz');p=p+w*d['p'];allkeys=d['keys'];valid=d['valid_index'];assert np.array_equal(allkeys[valid],keys)
    ps[split]=p
groups={}
for i,(loc,date,sid) in enumerate(allkeys):groups.setdefault(loc,[]).append(i)
site_weights=np.zeros(len(y));sites={}
for i,r in enumerate(rows['window']):
    if r['area']>=10:sites.setdefault((r['key'][0],r['key'][2]),[]).append(i)
for ids in sites.values():site_weights[ids]=1/len(ids)/len(sites)
variants=list(rows['window'][0]['iou']);arrays={s:{v:(np.array([r['nonempty'][v] for r in rows[s]]),np.array([r['iou'][v] for r in rows[s]])) for v in variants} for s in rows}
def evaluate(binary,split,variant):
    nonempty,iou=arrays[split][variant];pred=binary & nonempty;f=f1_score(y,pred,average='macro');miou=float(np.sum(site_weights*iou*pred));return {'score':.6*f+.4*miou,'macro_f1':f,'site_miou':miou,'confusion':confusion_matrix(y,pred).tolist()}
records=[]
for method,strengths in [('local',[.45,.75]),('markov',[.5,1.,2.])]:
    for strength in strengths:
        for th in [.2,.25,.3,.35,.4]:
            binary={}
            for split,p in ps.items():
                q=np.zeros(len(p))
                for ids in groups.values():
                    kk=[tuple(k[1:]) for k in allkeys[ids]];q[ids]=posterior(p[ids],kk,th,strength) if method=='markov' else smooth(p[ids],kk,strength)
                binary[split]=(q>=(.5 if method=='markov' else th))[valid]
            for variant in variants:
                tokens=variant.split('_');nw,sw,pth=(float(tokens[1]),float(tokens[2]),float(tokens[3])) if tokens[0]=='semantic' else (float(tokens[1]),0.,float(tokens[2]));scores={s:evaluate(binary[s],s,variant) for s in rows}
                records.append({'weights':weights,'gate_method':method,'gate_strength':strength,'threshold':th,'refiner_mix':nw,'semantic_weight':sw,'pixel_threshold':pth,'scores':scores,'selection':.4*scores['window']['score']+.6*scores['scene']['score']})
records.sort(key=lambda r:r['selection'],reverse=True)
baseline=next(r for r in records if r['gate_method']=='local' and r['gate_strength']==.45 and r['threshold']==.3 and r['refiner_mix']==0 and r['semantic_weight']==0 and r['pixel_threshold']==.4)
print('BASELINE17',json.dumps(baseline),flush=True)
for r in records[:10]:print(json.dumps(r),flush=True)
(root/'final_selection_report.json').write_text(json.dumps({'protocol':'OOF fits exclude held-out groups; temporal inference uses all 1691 input rows; only 1601 valid labels scored. SAM weights shared with v17.','baseline17':baseline,'candidates':records},indent=2))
best=records[0];(root/'final_config.json').write_text(json.dumps({k:v for k,v in best.items() if k not in ['scores','selection']},indent=2))
