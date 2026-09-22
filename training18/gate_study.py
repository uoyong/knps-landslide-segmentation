"""Ablate learners and probabilistic temporal persistence on saved OOF predictions."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.special import logsumexp,expit,logit
from sklearn.metrics import f1_score
sys.path.insert(0,'12_submission');from train_presence import smooth_site_probabilities

def markov(p,keys,threshold,penalty):
    out=np.zeros(len(p));sites={}
    for i,(loc,date,sid) in enumerate(keys):sites.setdefault((loc,sid),[]).append((date,i))
    transition=np.array([[0.,-penalty],[-penalty,0.]])
    for pairs in sites.values():
        ids=[i for _,i in sorted(pairs)];v=logit(np.clip(p[ids],1e-5,1-1e-5))-logit(threshold);em=np.stack([np.zeros(len(v)),v],1)
        fw=np.zeros_like(em);bw=np.zeros_like(em);fw[0]=em[0]
        for i in range(1,len(v)):
            fw[i,0]=np.logaddexp(fw[i-1,0],fw[i-1,1]-penalty)
            fw[i,1]=v[i]+np.logaddexp(fw[i-1,0]-penalty,fw[i-1,1])
        for i in range(len(v)-2,-1,-1):
            bw[i,0]=np.logaddexp(bw[i+1,0],bw[i+1,1]+v[i+1]-penalty)
            bw[i,1]=np.logaddexp(bw[i+1,0]-penalty,bw[i+1,1]+v[i+1])
        post=fw+bw;out[ids]=expit(post[:,1]-post[:,0])
    return out

def main():
    base=np.load('training17/features_sam.npz');keys=base['keys'];y=base['y'];probs={};rows={}
    for split in ['window','scene']:
        probs[split]={}
        for folder in ['training17','training18']:
            for path in Path(folder).glob(f'oof_*_{split}.npz'):
                d=np.load(path);m={tuple(k):float(p) for k,p in zip(d['keys'],d['p'])};name=path.stem.removeprefix('oof_').removesuffix('_'+split);probs[split][name]=np.array([m[tuple(k)] for k in keys])
        rs={tuple(r['key']):r for r in json.load(open(f'training17/refined_scores_{split}.json'))};rows[split]=[rs[tuple(k)] for k in keys]
    weights17={'spectral_et':.125,'spectral_hg':.125,'sam_et':.375,'sam_hg':.375}
    ensembles={'v17':weights17}
    for n in ['cat5','cat4_balanced']:
        if all(n in probs[s] for s in probs):
            for w in [.25,.5,1.]:ensembles[f'{n}_{w}']={**{k:v*(1-w) for k,v in weights17.items()},n:w}
    if all('prompt_et' in probs[s] and 'prompt_hg' in probs[s] for s in probs):
        for w in [.25,.5,.75,1.]:ensembles[f'prompts_{w}']={**{k:v*(1-w) for k,v in weights17.items()},'prompt_et':w/2,'prompt_hg':w/2}
    site_weights=np.zeros(len(y));sites={}
    for i,r in enumerate(rows['window']):
        if r['area']>=10:sites.setdefault((r['key'][0],r['key'][2]),[]).append(i)
    for ids in sites.values():site_weights[ids]=1/len(ids)/len(sites)
    def score(p,s):
        pred=(p>=.5)&np.array([r['nonempty']['pixel_1.0_0.4'] for r in rows[s]]);f=f1_score(y,pred,average='macro');iou=float(np.sum(site_weights*pred*np.array([r['iou']['pixel_1.0_0.4'] for r in rows[s]])))
        return {'score':.6*f+.4*iou,'f1':f,'iou':iou}
    records=[]
    for name,weights in ensembles.items():
        ps={s:sum(w*probs[s][k] for k,w in weights.items()) for s in probs}
        for method in ['local','markov']:
            for strength in ([.0,.45,.75] if method=='local' else [.5,1.,2.,3.]):
                for th in [.2,.25,.3,.35,.4,.45]:
                    scores={}
                    for s,p in ps.items():
                        q=markov(p,keys,th,strength) if method=='markov' else smooth_site_probabilities(p,keys,strength)-th+.5
                        scores[s]=score(q,s)
                    records.append({'ensemble':name,'weights':weights,'method':method,'strength':strength,'threshold':th,'scores':scores,'selection':.5*(scores['window']['score']+scores['scene']['score'])})
    records.sort(key=lambda r:r['selection'],reverse=True);Path('training18/gate_scores.json').write_text(json.dumps(records,indent=2))
    for r in records[:8]:print(json.dumps(r),flush=True)

if __name__=='__main__':main()
