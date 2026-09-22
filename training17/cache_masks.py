"""Cache SAM candidates on public training images, without a visibility gate."""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from baseline import load_model, polygons_to_geom, render_rgb, MODEL_IN, CKPT

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--limit',type=int,default=0);ap.add_argument('--missing',action='store_true');a=ap.parse_args()
    torch.set_num_threads(4)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    print('device',device,flush=True)
    from transformers import Sam3TrackerModel
    model=Sam3TrackerModel.from_pretrained('12_submission/assets/model/sam3',dtype=torch.float32)
    original={k:{n:v.detach().cpu().clone() for n,v in getattr(model,k).state_dict().items()} for k in ['prompt_encoder','mask_decoder']}
    tuned=torch.load(CKPT,map_location='cpu',weights_only=True)
    model=model.to(device).eval()
    outdir=Path('training17/masks');outdir.mkdir(exist_ok=True)
    tasks=[]
    for loc in sorted(Path('train').iterdir()):
        sites=json.loads((loc/'sites.json').read_text())
        for lp in sorted((loc/'labels').glob('*.json')):
            out=outdir/f'{loc.name}_{lp.stem}.npz'
            cache=Path('cache_embs')/f'{loc.name}_{lp.stem}.pt'
            selected=sites
            complete=False
            if out.exists():
                with np.load(out) as old:complete=set(old['sites'])=={s['site_id'] for s in sites}
            if selected and not complete and (cache.exists() or a.missing):tasks.append((loc,lp.stem,selected,cache,out))
    if a.limit:tasks=tasks[:a.limit]
    t=time.time()
    with torch.inference_mode():
        for j,(loc,date,sites,cache,outpath) in enumerate(tasks):
            if cache.exists():
                embs=torch.load(cache,map_location='cpu',weights_only=True)
            else:
                image=render_rgb(np.load(loc/'bands'/f'{date}.npy'))
                x=torch.from_numpy(np.array(image.resize((MODEL_IN,MODEL_IN),Image.BILINEAR),dtype=np.float32)/255).permute(2,0,1)[None].to(device)*2-1
                embs={}
                for key,pix in [('orig',x),('hflip',x.flip(-1))]:
                    enc=model(pixel_values=pix,multimask_output=False)
                    embs[key]=[f.half().cpu() for f in enc.image_embeddings]
                torch.save(embs,cache)
            embs={k:[f.float().to(device) for f in v] for k,v in embs.items()}
            geoms=[polygons_to_geom(s['prior_polygon']) for s in sites]
            boxes=[];points=[]
            for g in geoms:
                x0,y0,x1,y1=g.bounds;w=x1-x0;h=y1-y0;r=g.representative_point()
                boxes.append([(x0-.15*w)*MODEL_IN/256,(y0-.15*h)*MODEL_IN/256,(x1+.15*w)*MODEL_IN/256,(y1+.15*h)*MODEL_IN/256])
                points.append([[r.x*MODEL_IN/256,r.y*MODEL_IN/256]])
            box=torch.tensor([boxes],device=device,dtype=torch.float32);pts=torch.tensor([points],device=device,dtype=torch.float32)
            result={'sites':np.array([s['site_id'] for s in sites])}
            for name,sd in [('tuned',tuned),('pretrained',original)]:
                for k in ['prompt_encoder','mask_decoder']:getattr(model,k).load_state_dict(sd[k])
                log=[];conf=[]
                for flip,key in [(False,'orig'),(True,'hflip')]:
                    b=box.clone();p=pts.clone()
                    if flip:b[...,0]=MODEL_IN-box[...,2];b[...,2]=MODEL_IN-box[...,0];p[...,0]=MODEL_IN-pts[...,0]
                    ll=[];ss=[]
                    for start in range(0,len(sites),4):
                        o=model(image_embeddings=embs[key],input_boxes=b[:,start:start+4],input_points=p[:,start:start+4],input_labels=torch.ones_like(p[:,start:start+4,...,0],dtype=torch.int32),multimask_output=True)
                        scores=o.iou_scores[0];idx=scores.argmax(-1);batch=torch.arange(len(idx),device=device)
                        ll.append(o.pred_masks[0,batch,idx].float());ss.append(scores[batch,idx].float())
                    l=torch.cat(ll);log.append(l.flip(-1) if flip else l);conf.append(torch.cat(ss))
                result[name]=((log[0]+log[1])/2).cpu().numpy().astype(np.float16)
                result[name+'_conf']=torch.stack(conf).mean(0).cpu().numpy()
            np.savez_compressed(outpath,**result)
            if (j+1)%10==0 or j==0 or j+1==len(tasks):print(f'{j+1}/{len(tasks)} {loc.name}/{date} elapsed={time.time()-t:.1f}s',flush=True)

if __name__=='__main__':main()
