import json,shutil
from pathlib import Path

root=Path(__file__).resolve().parents[1];dest=root/'17_submission';assets=dest/'assets'
for name in ['features.py','pixel_refine.py','inference.py']:
    shutil.copy2(root/'training17'/name,assets/name)
common=(root/'training17/baseline.py').read_text().replace('14_submission/assets/','assets/')
(assets/'common.py').write_text(common)
shutil.copy2(root/'14_submission/assets/presence.joblib',assets/'baseline.joblib')
shutil.copy2(root/'training17/candidate_config.json',assets/'config.json')
cells=[{'cell_type':'markdown','metadata':{},'source':['# KNPS 산사태 변화추적 — 17차\n','다중분광·시계열 존재 판정과 SAM3 경계 보정.']},
       {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],
        'source':['import sys\n','sys.dont_write_bytecode = True\n','sys.path.insert(0, "assets")\n','from inference import main\n','main()\n']}]
nb={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.11'}},'nbformat':4,'nbformat_minor':5}
(dest/'predict.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=2))
(dest/'requirements.txt').write_text('torch==2.5.1\ntransformers==5.16.1\nscikit-learn==1.9.0\nscipy==1.17.1\nnumpy==2.4.6\nshapely==2.1.2\njoblib==1.6.0\npillow==12.3.0\nopencv-python-headless==5.0.0.93\n')
print('Built',dest)
