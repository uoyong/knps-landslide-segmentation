"""Run the packaged notebook offline and check SAM feature/cache parity."""
import os,json,sys,csv,time,hashlib,resource
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np,torch
root=Path(__file__).resolve().parents[1];os.chdir(root/'18_submission');sys.path.insert(0,str(root/'training18/vendor'));sys.path.insert(0,'assets')
os.environ['AIF_INPUT_DIR']=str(root/'train'/os.environ.get('SMOKE_LOCATION','ls_scene_11_w01'))
os.environ['AIF_PREDICTION_PATH']=str(root/'training18/smoke_prediction.csv');os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import inference18
entry=Path(os.environ['AIF_INPUT_DIR']);date_by_hash={hashlib.sha256(np.load(p).tobytes()).hexdigest():p.stem for p in (entry/'bands').glob('*.npy')};original=inference18.infer_masks;parity=[];neck_parity=[]
def checked(model,device,bands,sites,expanded=False,semantic=False):
    result=original(model,device,bands,sites,expanded,semantic);date=date_by_hash[hashlib.sha256(bands.tobytes()).hexdigest()]
    cached=np.load(root/'training17/masks'/f'{entry.name}_{date}.npz');ids={s:i for i,s in enumerate(cached['sites'])}
    for sid,(logits,_) in result[0].items():
        diff=np.abs(logits.astype(np.float32)-cached['tuned'][ids[sid]].astype(np.float32));parity.append(float(diff.max()));assert diff.mean()<.005 and diff.max()<.15
    if semantic:
        emb=torch.load(root/'cache_embs'/f'{entry.name}_{date}.pt',map_location='cpu',weights_only=True);expected=((emb['orig'][0][0].float()+emb['hflip'][0][0].float().flip(-1))/2).numpy();diff=float(np.abs(result[2]-expected).max());neck_parity.append(diff);assert diff<1e-4,diff
    return result
inference18.infer_masks=checked;t=time.time();nb=json.loads(Path('predict.ipynb').read_text());ns={'__name__':'__main__'}
for cell in nb['cells']:
    if cell['cell_type']=='code':exec(compile(''.join(cell['source']),'predict.ipynb','exec'),ns)
from shapely.geometry import Polygon
rows=list(csv.DictReader(open(os.environ['AIF_PREDICTION_PATH'])));seen=set();npoly=0;sites=json.loads((entry/'sites.json').read_text());dates=list((entry/'bands').glob('*.npy'));assert len(rows)==len(sites)*len(dates)
for row in rows:
    key=(row['location_id'],row['date'],row['site_id']);assert key not in seen;seen.add(key)
    if row['polygons']:
        for poly in json.loads(row['polygons']):
            g=Polygon(poly[0],poly[1:]);assert g.is_valid and g.area>0;assert all(0<=v<=256 for ring in poly for point in ring for v in point)
        npoly+=1
report={'location':entry.name,'rows':len(rows),'visible':npoly,'elapsed_seconds':time.time()-t,'max_logit_difference':max(parity),'max_neck_difference':max(neck_parity,default=0),'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'max_gpu_allocated_bytes':torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0}
(root/'18_submission/smoke.json').write_text(json.dumps(report,indent=2));print('SMOKE VERIFIED',json.dumps(report),flush=True)
