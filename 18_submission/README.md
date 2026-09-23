# 18차 제출 모델 (sam3-context-refine-v18)

- 모델명: `sam3-context-refine-v18`
- Private 점수: `0.714650707` (최종 5위 달성)

17차 베이스라인(0.7226점) 대비 미학습 지역 일반화 성능 향상을 위해 SAM3 인코더의 고해상도 영상 특징과 다중분광 밴드를 결합한 CatBoost 픽셀 보정기를 추가한 모델입니다. 존재 판정은 4개 모델 앙상블 및 마르코프 시계열 후처리를 적용했습니다.

## 최종 설정

```json
{
  "weights": {
    "spectral_et": 0.125,
    "spectral_hg": 0.125,
    "sam_et": 0.375,
    "sam_hg": 0.375
  },
  "gate_method": "markov",
  "gate_strength": 1.0,
  "threshold": 0.2,
  "refiner_mix": 0.5,
  "semantic_weight": 0.5,
  "pixel_threshold": 0.5
}
```

## 검증 결과

| 검증 | 17차 재평가 | 18차 | 차이 |
|---|---:|---:|---:|
| 창 전체 제외 | 0.770312 | 0.766684 | -0.003628 |
| 상위 지역 전체 제외 | 0.660811 | 0.673204 | +0.012394 |

공식 비율인 존재 macro-F1 60% + 개소별 형상 mIoU 40%를 사용한다. 학습·평가에는 유효 정답 1,601건만 쓰되, 시계열 추론에는 ignore 영상을 포함한 입력 1,691건 전체를 사용한다. 이 차이 때문에 위 17차 재평가 값은 과거에 유효 정답만 시계열 후처리했던 기록과 조금 다를 수 있다.

새 학습 모델을 창 또는 지역으로 분리한 5-fold 교차검증이다. SAM 가중치는 17차와 공통이며, SAM 전체를 fold마다 재학습한 결과는 아니다. 후보 선택에는 창 분리 40%, 지역 분리 60%를 사용했다. 모델 선택에 사용한 로컬 검증이며 비공개 점수나 목표 0.75 달성을 보장하지 않는다.

세부 F1·IoU·혼동행렬은 `validation.json`, 전체 후보는 `../training18/final_selection_report.json`, 비교 실험은 `../training18/README.md`에 기록했다.

## 재현

프로젝트 최상위에서 `training18/full_oof.py`, `train_semantic.py --group scene/window`, `evaluate_semantic.py --group scene/window`, `final_select.py`, `build_submission.py`, `run_smoke.py`, `submit18.py` 순서로 실행한다. 기존 17차 캐시와 18차 정규화 모델 학습 결과를 사용한다.

제출 대상은 `predict.ipynb`, `requirements.txt`, `assets/`이다. 데이터·정답·API 키는 포함하지 않는다. 추론 시 저장된 모델을 로컬에서 읽고 외부 다운로드를 하지 않는다. 모델 파일 해시는 `preflight.json`에 저장한다.

## 실제 실행 검사

오프라인 상태에서 17개 시점·3개 개소의 실제 제출 노트북을 실행했다. 추론 89.3초, 모델 로딩과 검사를 포함해 101.7초에 51행을 생성했고, 26행을 가시로 예측했다. 중복 키, 무효 폴리곤, 좌표 범위 오류가 없었다. SAM logit 및 추가 영상 특징은 검증 캐시와 최대 차이 0.0으로 일치했다. 최대 프로세스 RSS는 약 2.56 GiB, PyTorch 최대 GPU 할당은 약 2.24 GiB였다. 자세한 기록은 `smoke.json`에 있다.

업로드 로그는 `../training18/submit.log`에 기록되며, 서버 응답을 받으면 제출 스크립트가 `submission_receipt.json`을 자동 저장한다.
