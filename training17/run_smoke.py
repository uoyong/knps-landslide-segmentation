"""Execute the actual notebook against one complete public input window."""
import os,json,sys,csv,time,hashlib
import numpy as np
from pathlib import Path
root=Path(__file__).resolve().parents[1];os.chdir(root/'17_submission')
os.environ['AIF_INPUT_DIR']=str(root/'train'/os.environ.get('SMOKE_LOCATION','ls_scene_11_w01'))
os.environ['AIF_PREDICTION_PATH']=str(root/'training17/smoke_prediction.csv')
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
sys.path.insert(0,'assets')
import inference
entry=Path(os.environ['AIF_INPUT_DIR'])
date_by_hash={hashlib.sha256(np.load(p).tobytes()).hexdigest():p.stem for p in (entry/'bands').glob('*.npy')}
original_infer=inference.infer_masks;parity=[]
def checked_infer(model,device,bands,sites):
    result=original_infer(model,device,bands,sites)
    date=date_by_hash[hashlib.sha256(bands.tobytes()).hexdigest()]
    cached=np.load(root/'training17/masks'/f'{entry.name}_{date}.npz');expected=cached['tuned'];ids={s:i for i,s in enumerate(cached['sites'])}
    for sid,(logits,_) in result.items():
        diff=np.abs(logits.astype(np.float32)-expected[ids[sid]].astype(np.float32));parity.append(float(diff.max()))
        assert diff.mean()<.005 and diff.max()<.15,(date,sid,float(diff.mean()),float(diff.max()))
    return result
inference.infer_masks=checked_infer
nb=json.loads(Path('predict.ipynb').read_text());ns={'__name__':'__main__'}
for cell in nb['cells']:
    if cell['cell_type']=='code':exec(compile(''.join(cell['source']),'predict.ipynb','exec'),ns)
from shapely.geometry import Polygon
rows=list(csv.DictReader(open(os.environ['AIF_PREDICTION_PATH'])));seen=set();npoly=0
entry=Path(os.environ['AIF_INPUT_DIR']);sites=json.loads((entry/'sites.json').read_text());dates=list((entry/'bands').glob('*.npy'))
assert len(rows)==len(sites)*len(dates)
for row in rows:
    key=(row['location_id'],row['date'],row['site_id']);assert key not in seen;seen.add(key)
    if row['polygons']:
        p=json.loads(row['polygons']);assert isinstance(p,list)
        for poly in p:
            g=Polygon(poly[0],poly[1:]);assert g.is_valid and g.area>0
            assert all(0<=v<=256 for ring in poly for point in ring for v in point)
        npoly+=1
print('SMOKE VERIFIED',len(rows),'unique rows;',npoly,'visible predictions',flush=True)
print('CACHE PARITY maximum logit difference',max(parity),flush=True)
