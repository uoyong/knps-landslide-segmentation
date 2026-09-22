"""Submission 17: multispectral visibility ensemble and optional learned contours."""
import csv,json,os,time
from pathlib import Path
import cv2,joblib,numpy as np,torch
import torch.nn.functional as F
from PIL import Image
from shapely.ops import unary_union
from features import entry_features
from common import (load_tree,polygons_to_geom,geom_to_polys,render_rgb,px_polys_from_mask,
                    rasterize,_raw_features,_temporalize)

def smooth(p,keys,alpha):
    q=np.asarray(p).copy();groups={}
    for i,(d,sid) in enumerate(keys):groups.setdefault(sid,[]).append((d,i))
    for group in groups.values():
        ids=[i for _,i in sorted(group)];v=np.asarray(p)[ids]
        local=np.array([v[max(0,j-1):min(len(v),j+2)].mean() for j in range(len(v))])
        q[ids]=(1-alpha)*v+alpha*local
    return q

def baseline_probabilities(sites,dates,arrays,bundle):
    gs=[polygons_to_geom(s.get('prior_polygon')) for s in sites];X=[];keys=[]
    for site,g in zip(sites,gs):
        if g is None:continue
        inside=rasterize(g,(256,256));others=unary_union([q.buffer(3) for q in gs if q is not None and q is not g])
        ring=rasterize(g.buffer(8).difference(g.buffer(2)).difference(others),(256,256))
        raw=[_raw_features(b,inside,ring,int(d[4:6])) for b,d in zip(arrays,dates)]
        X.extend(_temporalize(raw));keys.extend([(d,site['site_id']) for d in dates])
    return bundle['model'].predict_proba(np.asarray(X))[:,1],keys

@torch.inference_mode()
def infer_masks(model,device,bands,sites):
    sites=[s for s in sites if polygons_to_geom(s.get('prior_polygon')) is not None]
    if not sites:return {}
    image=render_rgb(bands);x=torch.from_numpy(np.array(image.resize((1008,1008),Image.BILINEAR),dtype=np.float32)/255).permute(2,0,1)[None].to(device)*2-1
    boxes=[];points=[]
    for s in sites:
        g=polygons_to_geom(s['prior_polygon']);x0,y0,x1,y1=g.bounds;w=x1-x0;h=y1-y0;p=g.representative_point()
        boxes.append([(x0-.15*w)*1008/256,(y0-.15*h)*1008/256,(x1+.15*w)*1008/256,(y1+.15*h)*1008/256]);points.append([[p.x*1008/256,p.y*1008/256]])
    box=torch.tensor([boxes],device=device,dtype=torch.float32);points=torch.tensor([points],device=device,dtype=torch.float32);log=[];conf=[]
    for flip in [False,True]:
        emb=model(pixel_values=x.flip(-1) if flip else x,multimask_output=False).image_embeddings
        # Match the cached feature precision used in cross-validation.
        emb=[f.half().float() for f in emb]
        b=box.clone();p=points.clone()
        if flip:b[...,0]=1008-box[...,2];b[...,2]=1008-box[...,0];p[...,0]=1008-points[...,0]
        ll=[];ss=[]
        for start in range(0,len(sites),4):
            o=model(image_embeddings=emb,input_boxes=b[:,start:start+4],input_points=p[:,start:start+4],input_labels=torch.ones_like(p[:,start:start+4,...,0],dtype=torch.int32),multimask_output=True)
            scores=o.iou_scores[0];idx=scores.argmax(-1);bi=torch.arange(len(idx),device=device)
            ll.append(o.pred_masks[0,bi,idx].float());ss.append(scores[bi,idx].float())
        l=torch.cat(ll);log.append(l.flip(-1) if flip else l);conf.append(torch.cat(ss))
    logits=((log[0]+log[1])/2).cpu().numpy().astype(np.float16);scores=torch.stack(conf).mean(0).cpu().numpy()
    return {s['site_id']:(logits[i],float(scores[i])) for i,s in enumerate(sites)}

def predict_entry(model,device,entry,models,config,log=print):
    dates=sorted(entry['dates']);sites=entry['sites'];arrays=[np.load(entry['dates'][d]) for d in dates]
    sam={}
    for j,(d,b) in enumerate(zip(dates,arrays)):
        sam[d]=infer_masks(model,device,b,sites)
        if (j+1)%5==0:log(f'  SAM {j+1}/{len(dates)}',flush=True)
    predictions=[];keys=None
    for family in ['baseline','spectral','sam']:
        selected={k:w for k,w in config['weights'].items() if k.startswith(family) and w>0}
        if not selected:continue
        if family=='baseline':
            p,k=baseline_probabilities(sites,dates,arrays,models['baseline']);predictions.append((config['weights']['baseline'],p));keys=k
        else:
            X,k=entry_features(sites,dates,arrays,sam if family=='sam' else None)
            if keys is not None:assert k==keys
            keys=k
            for name,w in selected.items():predictions.append((w,models[name].predict_proba(X)[:,1]))
    prob=sum(w*p for w,p in predictions);prob=smooth(prob,keys,config['smooth_alpha']);presence=dict(zip(keys,prob))
    ref=np.median(np.stack(arrays),0) if config.get('pixel_weight',0)>0 else None
    rows=[]
    for d,b in zip(dates,arrays):
        patch=None
        for s in sites:
            sid=s['site_id'];prior=polygons_to_geom(s.get('prior_polygon'));polys=[]
            if prior is not None and presence.get((d,sid),0)>=config['threshold']:
                logits=sam[d][sid][0].astype(np.float32)
                if config.get('pixel_weight',0)>0:
                    from pixel_refine import patch_features,site_features
                    if patch is None:patch=patch_features(b,ref)
                    X,roi,_,_=site_features(patch,logits,prior);r0,r1,c0,c1=roi
                    pp=models['pixel_refiner'].predict_proba(X)[:,1].reshape(r1-r0,c1-c0)
                    baseline=1/(1+np.exp(-np.clip(cv2.resize(logits,(512,512)),-20,20)))
                    mix=np.zeros((512,512),np.float32);w=config['pixel_weight'];mix[r0:r1,c0:c1]=w*pp+(1-w)*baseline[r0:r1,c0:c1]
                    mask=cv2.resize(mix,(1024,1024))>config['pixel_threshold']
                else:
                    big=F.interpolate(torch.from_numpy(logits)[None,None],(1024,1024),mode='bilinear',align_corners=False)[0,0].numpy();mask=big>config['mask_threshold']
                parts=px_polys_from_mask(mask,max(1.,min(2.,.3*prior.area)));g=unary_union(parts).simplify(.05) if parts else None;polys=geom_to_polys(g)
            rows.append((d,sid,polys,float(presence.get((d,sid),0))))
    return rows

def main():
    torch.set_num_threads(4);cv2.setNumThreads(1)
    config=json.loads(Path('assets/config.json').read_text());models={}
    for name,w in config['weights'].items():
        if w>0:models[name]=joblib.load(f'assets/{name}.joblib')
    if config.get('pixel_weight',0)>0:models['pixel_refiner']=joblib.load('assets/pixel_refiner.joblib')
    from transformers import Sam3TrackerModel
    device='cuda' if torch.cuda.is_available() else 'cpu';model=Sam3TrackerModel.from_pretrained('assets/sam3',dtype=torch.float32,local_files_only=True).to(device).eval()
    locations=load_tree(Path(os.environ['AIF_INPUT_DIR']));out=Path(os.environ['AIF_PREDICTION_PATH']);out.parent.mkdir(parents=True,exist_ok=True);t=time.time();count=0
    with out.open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['location_id','date','site_id','polygons'])
        for loc,entry in sorted(locations.items()):
            print('location',loc,flush=True)
            for d,sid,polys,_ in predict_entry(model,device,entry,models,config):
                writer.writerow([loc,d,sid,json.dumps(polys,separators=(',',':')) if polys else '']);count+=1
            print('rows',count,'elapsed',round(time.time()-t,1),flush=True)
    print('complete',count,str(out),flush=True)

if __name__=='__main__':main()
