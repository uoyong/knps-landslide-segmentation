"""Alternative prompts compensate for displaced, narrow historical priors."""
import json,time,sys
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,'training17')
from features import geom

torch.set_num_threads(3)
from transformers import Sam3TrackerModel
device='cuda' if torch.cuda.is_available() else 'cpu';print('device',device,flush=True)
model=Sam3TrackerModel.from_pretrained('17_submission/assets/sam3',dtype=torch.float32,local_files_only=True).to(device).eval()
root=Path('training18/prompts');root.mkdir(exist_ok=True);t=time.time();n=0
with torch.inference_mode():
    for loc in sorted(Path('train').iterdir()):
        sites=json.loads((loc/'sites.json').read_text());gs=[geom(s['prior_polygon']) for s in sites]
        boxes=[];points=[]
        for g in gs:
            x0,y0,x1,y1=g.bounds;dx=max(.25*(x1-x0),3);dy=max(.25*(y1-y0),3);p=g.representative_point()
            boxes.append([max(0,x0-dx)*1008/256,max(0,y0-dy)*1008/256,min(256,x1+dx)*1008/256,min(256,y1+dy)*1008/256]);points.append([[p.x*1008/256,p.y*1008/256]])
        box=torch.tensor([boxes],dtype=torch.float32,device=device);point=torch.tensor([points],dtype=torch.float32,device=device)
        for p in sorted((loc/'bands').glob('*.npy')):
            dest=root/f'{loc.name}_{p.stem}.npz'
            if dest.exists():continue
            emb0=torch.load(Path('cache_embs')/f'{loc.name}_{p.stem}.pt',map_location='cpu',weights_only=True);emb={k:[f.float().to(device) for f in v] for k,v in emb0.items()}
            results={'sites':np.array([s['site_id'] for s in sites])}
            for name,use_point in [('widepoint',True),('widebox',False)]:
                logs=[];scores=[]
                for flip,key in [(False,'orig'),(True,'hflip')]:
                    b=box.clone();pt=point.clone()
                    if flip:b[...,0]=1008-box[...,2];b[...,2]=1008-box[...,0];pt[...,0]=1008-point[...,0]
                    ll=[];ss=[]
                    for start in range(0,len(sites),4):
                        extra={'input_points':pt[:,start:start+4],'input_labels':torch.ones_like(pt[:,start:start+4,...,0],dtype=torch.int32)} if use_point else {}
                        o=model(image_embeddings=emb[key],input_boxes=b[:,start:start+4],multimask_output=True,**extra);sc=o.iou_scores[0];idx=sc.argmax(-1);bi=torch.arange(len(idx),device=device);ll.append(o.pred_masks[0,bi,idx].float());ss.append(sc[bi,idx])
                    l=torch.cat(ll);logs.append(l.flip(-1) if flip else l);scores.append(torch.cat(ss))
                results[name]=((logs[0]+logs[1])/2).cpu().numpy().astype(np.float16);results[name+'_conf']=torch.stack(scores).mean(0).cpu().numpy()
            np.savez_compressed(dest,**results);n+=1
            if n%25==0:print('patches',n,'elapsed',round(time.time()-t,1),flush=True)
print('complete',n,round(time.time()-t,1),flush=True)
