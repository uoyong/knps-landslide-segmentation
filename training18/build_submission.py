"""Package the selected, validated configuration without training data."""
import json,os,shutil,hashlib
from pathlib import Path

root=Path(__file__).resolve().parents[1];dest=root/'18_submission';assets=dest/'assets';assets.mkdir(parents=True,exist_ok=True)
config=json.loads((root/'training18/final_config.json').read_text())
for name in ['common.py','features.py','pixel_refine.py','pixel_refiner.joblib']:
    shutil.copy2(root/'17_submission/assets'/name,assets/name)
shutil.copy2(root/'17_submission/assets/inference.py',assets/'inference17.py')
for name in ['inference18.py','refine_features.py','semantic_refine.py','temporal_gate.py']:
    shutil.copy2(root/'training18'/name,assets/name)
for name,w in config['weights'].items():
    if w:shutil.copy2(root/'17_submission/assets'/f'{name}.joblib',assets/f'{name}.joblib')
if config['refiner_mix']>0:shutil.copy2(root/'training18/refiner.joblib',assets/'refiner18.joblib')
if config['semantic_weight']>0:shutil.copy2(root/'training18/semantic_refiner.cbm',assets/'semantic_refiner.cbm')
(assets/'sam3').mkdir(exist_ok=True)
for p in (root/'17_submission/assets/sam3').iterdir():
    target=assets/'sam3'/p.name
    if not target.exists():os.link(p,target)
(assets/'config.json').write_text(json.dumps(config,indent=2))
cells=[{'cell_type':'markdown','metadata':{},'source':['# KNPS 산사태 변화추적 — 18차\n','SAM3 영상 특징·다중분광 경계 보정과 시계열 존재 판정.']},
       {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':['import sys\n','sys.dont_write_bytecode = True\n','sys.path.insert(0, "assets")\n','from inference18 import main\n','main()\n']}]
nb={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.11'}},'nbformat':4,'nbformat_minor':5}
(dest/'predict.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=2))
(dest/'requirements.txt').write_text((root/'17_submission/requirements.txt').read_text()+('catboost==1.2.10\n' if config['semantic_weight'] else ''))
report=json.loads((root/'training18/final_selection_report.json').read_text())
(dest/'validation.json').write_text(json.dumps({'protocol':report['protocol'],'baseline17':report['baseline17'],'selected':report['candidates'][0]},indent=2))
baseline=report['baseline17'];selected=report['candidates'][0]
lines=['# 18차 — SAM 영상 특징을 이용한 경계 보정', '', '모델명: `sam3-context-refine-v18`', '', '**상태: 실행 검사 및 제출 준비 중. 실제 리더보드 점수는 아직 없다.**', '',
       '17차 실측 점수 0.7226459에서 0.75 이상을 목표로 추가 실험했다. SAM 인코더의 고해상도 영상 특징과 다중분광 정보를 함께 학습한 CatBoost 픽셀 모델을 추가했다. 존재 판정은 기존 4개 모델을 사용하며, 시계열 후처리와 경계 모델 조합을 공동 검증해 선택했다.', '',
       '## 최종 설정', '', '```json', json.dumps(config,ensure_ascii=False,indent=2), '```', '',
       '## 검증 결과', '', '| 검증 | 17차 재평가 | 18차 | 차이 |', '|---|---:|---:|---:|']
for split,label in [('window','창 전체 제외'),('scene','상위 지역 전체 제외')]:
    b=baseline['scores'][split]['score'];s=selected['scores'][split]['score'];lines.append(f'| {label} | {b:.6f} | {s:.6f} | {s-b:+.6f} |')
lines+=['', '공식 비율인 존재 macro-F1 60% + 개소별 형상 mIoU 40%를 사용한다. 학습·평가에는 유효 정답 1,601건만 쓰되, 시계열 추론에는 ignore 영상을 포함한 입력 1,691건 전체를 사용한다. 이 차이 때문에 위 17차 재평가 값은 과거에 유효 정답만 시계열 후처리했던 기록과 조금 다를 수 있다.', '',
        '새 학습 모델을 창 또는 지역으로 분리한 5-fold 교차검증이다. SAM 가중치는 17차와 공통이며, SAM 전체를 fold마다 재학습한 결과는 아니다. 후보 선택에는 창 분리 40%, 지역 분리 60%를 사용했다. 모델 선택에 사용한 로컬 검증이며 비공개 점수나 목표 0.75 달성을 보장하지 않는다.', '',
        '세부 F1·IoU·혼동행렬은 `validation.json`, 전체 후보는 `../training18/final_selection_report.json`, 비교 실험은 `../training18/README.md`에 기록했다.', '',
        '## 재현', '', '프로젝트 최상위에서 `training18/full_oof.py`, `train_semantic.py --group scene/window`, `evaluate_semantic.py --group scene/window`, `final_select.py`, `build_submission.py`, `run_smoke.py`, `submit18.py` 순서로 실행한다. 기존 17차 캐시와 18차 정규화 모델 학습 결과를 사용한다.', '',
        '제출 대상은 `predict.ipynb`, `requirements.txt`, `assets/`이다. 데이터·정답·API 키는 포함하지 않는다. 추론 시 저장된 모델을 로컬에서 읽고 외부 다운로드를 하지 않는다. 모델 파일 해시는 `preflight.json`에 저장한다.', '']
(dest/'README.md').write_text('\n'.join(lines))
manifest={}
for p in sorted(assets.rglob('*')):
    if p.is_file():
        assert not p.is_symlink();h=hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
        manifest[str(p.relative_to(dest))]={'bytes':p.stat().st_size,'sha256':h.hexdigest()}
(dest/'preflight.json').write_text(json.dumps(manifest,indent=2));print('Built',dest,'asset files',len(manifest),'bytes',sum(m['bytes'] for m in manifest.values()))
