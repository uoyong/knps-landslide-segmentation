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

공식 평가 산식인 존재 macro-F1 60% + 개소별 형상 mIoU 40%를 기준으로 평가했습니다. 학습 및 평가에는 유효 정답 1,601건을 사용했으며, 시계열 추론에는 ignore 영상을 포함한 입력 1,691건 전체를 활용했습니다. 이러한 조건 차이로 인해 위 17차 재평가 수치는 기존 기록과 소폭 차이가 있을 수 있습니다.

본 결과는 모델을 관측 창(Window) 또는 지역(Scene) 단위로 분리한 5-fold 교차검증 결과입니다. SAM 가중치는 17차와 동일한 가중치를 공유합니다. 최종 후보 선택 시에는 창 분리 40%, 지역 분리 60% 가중치를 반영하여 종합 점수를 산출했습니다.

세부 F1, IoU 및 혼동행렬 지표는 `validation.json`에 정리되어 있으며, 전체 앙상블 후보군 결과는 `../training18/final_selection_report.json`, 비교 실험 내용은 `../training18/README.md`에서 확인하실 수 있습니다.

## 재현 방법

프로젝트 루트 디렉토리에서 `training18/full_oof.py`, `train_semantic.py --group scene/window`, `evaluate_semantic.py --group scene/window`, `final_select.py`, `build_submission.py`, `run_smoke.py`, `submit18.py` 순서로 실행합니다. 사전 생성된 캐시와 18차 모델 학습 결과를 바탕으로 동작합니다.

제출 대상 파일은 `predict.ipynb`, `requirements.txt`, `assets/`입니다. 보안 및 규정에 따라 데이터, 정답 라벨, API 키 등은 일체 포함하지 않습니다. 추론 실행 시 오프라인 환경에서 로컬에 저장된 모델 가중치만 참조하며, 모델 파일 해시값은 `preflight.json`에 기록되어 있습니다.

## 오프라인 검증 테스트

완전한 오프라인 환경에서 17개 시점, 3개 개소를 대상으로 실제 제출 노트북을 테스트했습니다. 모델 로딩 및 무결성 검증을 포함해 총 101.7초(순수 추론 89.3초)가 소요되었으며, 51개 샘플 중 26개 샘플을 산사태 발생으로 판정했습니다. 중복 키, 무효 폴리곤, 좌표계 범위 이탈 등의 오류가 전혀 발생하지 않음을 확인했습니다. 최대 메모리 사용량은 시스템 메모리 RSS 약 2.56 GiB, GPU 메모리 약 2.24 GiB로 측정되었으며, 세부 로그는 `smoke.json`에 기록되어 있습니다.

제출 실행 로그는 `../training18/submit.log`에 기록되며, 채점 서버 응답 수신 시 `submission_receipt.json`에 제출 영수증이 자동으로 저장됩니다.

