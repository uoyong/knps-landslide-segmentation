"""Submission 18 inference: all temporal inputs remain label-free."""
import csv,json,os,time
from pathlib import Path
import cv2,joblib,numpy as np,torch
from PIL import Image
from shapely.ops import unary_union
from features import entry_features
from common import load_tree,polygons_to_geom,geom_to_polys,render_rgb,px_polys_from_mask
from inference17 import smooth
from pixel_refine import patch_features,site_features
from refine_features import enrich
from temporal_gate import posterior

@torch.inference_mode()
def infer_masks(model,device,bands,sites,expanded=False,semantic=False):
    valid=[s for s in sites if polygons_to_geom(s.get('prior_polygon')) is not None]
    if not valid:return {},{},None
    image=render_rgb(bands);x=torch.from_numpy(np.array(image.resize((1008,1008),Image.BILINEAR),dtype=np.float32)/255).permute(2,0,1)[None].to(device)*2-1
    boxes=[];wide=[];points=[]
    for s in valid:
        g=polygons_to_geom(s['prior_polygon']);x0,y0,x1,y1=g.bounds;w=x1-x0;h=y1-y0;p=g.representative_point()
        boxes.append(np.array([x0-.15*w,y0-.15*h,x1+.15*w,y1+.15*h])*1008/256)
        dx=max(.25*w,3);dy=max(.25*h,3);wide.append(np.array([max(0,x0-dx),max(0,y0-dy),min(256,x1+dx),min(256,y1+dy)])*1008/256)
        points.append([[p.x*1008/256,p.y*1008/256]])
    box=torch.tensor(np.array([boxes]),dtype=torch.float32,device=device);wide=torch.tensor(np.array([wide]),dtype=torch.float32,device=device);pts=torch.tensor([points],dtype=torch.float32,device=device)
    variants=['tuned']+(['widepoint','widebox'] if expanded else []);logs={k:[] for k in variants};confs={k:[] for k in variants};necks=[]
    for flip in [False,True]:
        emb=model(pixel_values=x.flip(-1) if flip else x,multimask_output=False).image_embeddings;emb=[f.half().float() for f in emb]
        if semantic:necks.append((emb[0][0].flip(-1) if flip else emb[0][0]).cpu().numpy())
        for name in variants:
            original=box if name=='tuned' else wide;b=original.clone();p=pts.clone()
            if flip:b[...,0]=1008-original[...,2];b[...,2]=1008-original[...,0];p[...,0]=1008-pts[...,0]
            ll=[];ss=[]
            for start in range(0,len(valid),4):
                extra={'input_points':p[:,start:start+4],'input_labels':torch.ones_like(p[:,start:start+4,...,0],dtype=torch.int32)} if name!='widebox' else {}
                o=model(image_embeddings=emb,input_boxes=b[:,start:start+4],multimask_output=True,**extra);scores=o.iou_scores[0];idx=scores.argmax(-1);bi=torch.arange(len(idx),device=device);ll.append(o.pred_masks[0,bi,idx].float());ss.append(scores[bi,idx])
            l=torch.cat(ll);logs[name].append(l.flip(-1) if flip else l);confs[name].append(torch.cat(ss))
    values={k:((logs[k][0]+logs[k][1])/2).cpu().numpy().astype(np.float16) for k in variants};scores={k:torch.stack(confs[k]).mean(0).cpu().numpy() for k in variants}
    old={s['site_id']:(values['tuned'][i],float(scores['tuned'][i])) for i,s in enumerate(valid)};alternative={}
    if expanded:
        mean=sum(values[k].astype(np.float32) for k in variants)/3;conf=sum(scores.values())/3
        alternative={s['site_id']:(mean[i].astype(np.float16),float(conf[i])) for i,s in enumerate(valid)}
    neck=(necks[0]+necks[1])/2 if semantic else None
    return old,alternative,neck

def predict_entry(model,device,entry,models,config):
    dates=sorted(entry['dates']);sites=entry['sites'];arrays=[np.load(entry['dates'][d]) for d in dates];sam={};prompt={};necks={};expanded=any(n.startswith('prompt') and w for n,w in config['weights'].items());semantic=config.get('semantic_weight',0)>0
    for i,(d,b) in enumerate(zip(dates,arrays)):
        sam[d],prompt[d],necks[d]=infer_masks(model,device,b,sites,expanded,semantic)
        if (i+1)%5==0:print('  SAM',i+1,'/',len(dates),flush=True)
    X,keys=entry_features(sites,dates,arrays,sam);spectral=X[:,np.concatenate([np.arange(257)+367*j for j in range(4)])]
    Xprompt=None
    if expanded:Xprompt,k=entry_features(sites,dates,arrays,prompt);assert k==keys
    prob=np.zeros(len(keys))
    for name,w in config['weights'].items():
        if not w:continue
        features=spectral if name.startswith('spectral') else Xprompt if name.startswith('prompt') else X
        prob+=w*models[name].predict_proba(features)[:,1]
    if config['gate_method']=='markov':post=posterior(prob,keys,config['threshold'],config['gate_strength']);visible=post>=.5
    else:post=smooth(prob,keys,config['gate_strength']);visible=post>=config['threshold']
    decisions=dict(zip(keys,visible));ref=np.median(np.stack(arrays),0);rows=[]
    for d,b in zip(dates,arrays):
        patch=None
        for s in sites:
            sid=s['site_id'];prior=polygons_to_geom(s.get('prior_polygon'));polys=[]
            if prior is not None and decisions.get((d,sid),False):
                if patch is None:patch=patch_features(b,ref)
                xx,roi,_,_=site_features(patch,sam[d][sid][0],prior);r0,r1,c0,c1=roi
                old=models['pixel_refiner'].predict_proba(xx)[:,1];w=config['refiner_mix']
                pp=(1-w)*old+w*models['refiner18'].predict_proba(enrich(xx))[:,1] if w else old
                if semantic:
                    from semantic_refine import semantic_features
                    sp=models['semantic_refiner'].predict_proba(semantic_features(xx,roi,necks[d]),thread_count=4)[:,1];sw=config['semantic_weight'];pp=(1-sw)*pp+sw*sp
                full=np.zeros((512,512),np.float32);full[r0:r1,c0:c1]=pp.reshape(r1-r0,c1-c0);m=cv2.resize(full,(1024,1024))>config['pixel_threshold'];parts=px_polys_from_mask(m,max(1.,min(2.,.3*prior.area)));g=unary_union(parts).simplify(.05) if parts else None;polys=geom_to_polys(g)
            rows.append((d,sid,polys))
    return rows

def main():
    torch.set_num_threads(4);cv2.setNumThreads(1);config=json.loads(Path('assets/config.json').read_text());models={}
    for name,w in config['weights'].items():
        if not w:continue
        if name.startswith('cat'):
            from catboost import CatBoostClassifier
            models[name]=CatBoostClassifier();models[name].load_model(f'assets/{name}.cbm')
        else:models[name]=joblib.load(f'assets/{name}.joblib')
    models['pixel_refiner']=joblib.load('assets/pixel_refiner.joblib')
    if config['refiner_mix']>0:models['refiner18']=joblib.load('assets/refiner18.joblib')
    if config.get('semantic_weight',0)>0:
        from catboost import CatBoostClassifier
        models['semantic_refiner']=CatBoostClassifier();models['semantic_refiner'].load_model('assets/semantic_refiner.cbm')
    from transformers import Sam3TrackerModel
    device='cuda' if torch.cuda.is_available() else 'cpu';model=Sam3TrackerModel.from_pretrained('assets/sam3',dtype=torch.float32,local_files_only=True).to(device).eval();locations=load_tree(Path(os.environ['AIF_INPUT_DIR']));out=Path(os.environ['AIF_PREDICTION_PATH']);out.parent.mkdir(parents=True,exist_ok=True);t=time.time();count=0
    with out.open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['location_id','date','site_id','polygons'])
        for loc,entry in sorted(locations.items()):
            print('location',loc,flush=True)
            for d,sid,polys in predict_entry(model,device,entry,models,config):writer.writerow([loc,d,sid,json.dumps(polys,separators=(',',':')) if polys else '']);count+=1
            print('rows',count,'elapsed',round(time.time()-t,1),flush=True)
    print('complete',count,str(out),flush=True)

if __name__=='__main__':main()
