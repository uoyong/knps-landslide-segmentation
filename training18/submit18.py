"""Submit exactly the reviewed 18th notebook; reuse the local participation key."""
import os,json,re
from pathlib import Path
from aifactory import cli,client

root=Path(__file__).resolve().parents[1]
key=os.environ.get('AIF_API_KEY')
if not key:
    for version in ['14_submission','12_submission']:
        nb=json.loads((root/version/'predict.ipynb').read_text())
        source='\n'.join(''.join(c.get('source',[])) for c in nb['cells'])
        match=re.search(r"--api-key\s+['\"]([^'\"]+)['\"]",source)
        if match:key=match.group(1);break
if not key:raise RuntimeError('No existing participation key found')
os.environ['AIF_API_KEY']=key
original=client.submit
def tracked_submit(*args,**kwargs):
    response=original(*args,**kwargs)
    (root/'18_submission/submission_receipt.json').write_text(json.dumps(response,ensure_ascii=False,indent=2))
    return response
client.submit=tracked_submit
os.chdir(root/'18_submission')
raise SystemExit(cli.main(['submit','--notebook','predict.ipynb','--model-name','sam3-context-refine-v18']))
