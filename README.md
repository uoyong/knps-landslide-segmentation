# KNPS Landslide Segmentation (2026)

시계열 위성 영상 기반 산사태 변화 탐지 및 형상 분할 모델  
18차 Private 점수: `0.714650707` 최종 5위 달성 (상위 6.2%, 5/81)

---

## 1. Overview

국립공원공단 산사태 시계열 위성 영상(Sentinel-2 다중분광 밴드)을 활용하여 산사태 발생 여부를 판정하고, 발생 지점의 정확한 경계를 분할(Segmentation)하는 파이프라인입니다.

본 repository는 최종 5위 달성 모델인 **18차 솔루션(`sam3-context-refine-v18`)**의 추론 패키지(`18_submission/`) 및 학습 파이프라인(`training18/`)을 통합하여 구성했습니다.

### 주요 파이프라인 특징
1. **시계열 다중분광 특징 추출**: Sentinel-2 밴드(RGB, NIR, RedEdge 등) 및 dNDVI 변화량 추출
2. **산사태 존재 판정기 (Presence Gate)**:
   - ExtraTrees + HistGradientBoosting 기반 4모델 앙상블
   - 2-state Forward/Backward 마르코프 체인 기반 시계열 평활화 및 연속성 보정
3. **형상 분할 (SAM3 Mask Generation)**:
   - Segment Anything Model (SAM3) 기반 Box + Point 프롬프트 추론
   - 수평 반전(H-flip) Test-Time Augmentation (TTA)
4. **문맥 경계 보정기 (Context-Aware Boundary Refiner)**:
   - SAM3 고해상도 인코더 특징(32채널) + 분광 밴드를 결합한 142차원 픽셀 특징 구성
   - CatBoost 기반 픽셀 단위 경계 정밀 보정

---

## 2. Directory Structure

```text
├── 18_submission/        # 최종 제출용 독립 추론 패키지 (AIFactory 제출 규격)
│   ├── assets/           # 추론 모듈 및 학습된 모델 가중치
│   ├── predict.ipynb     # 최종 채점용 추론 노트북
│   ├── requirements.txt  # 실행 의존성
│   └── preflight.json    # 모델 가중치 무결성 검증 메타데이터
├── training18/           # 18차 모델 학습, 최적화 및 앙상블 탐색 파이프라인
│   ├── train_semantic.py # 문맥 경계 보정기 학습
│   ├── final_select.py   # 교차검증 기반 최적 하이퍼파라미터 탐색
│   ├── full_oof.py       # OOF 예측값 생성
│   └── build_submission.py # 제출물 자동 패키징
├── training17/           # 17차 베이스라인 모델 학습 
├── README.md             # 프로젝트 종합 문서
└── .gitignore
```

---

## 3. Pipeline Details

### (1) Presence Gate
- **입력**: 다중 시점 분광 차분 및 SAM 영역 통계량 (1,468차원)
- **모델**: ExtraTrees (Spectral/SAM) + HistGradientBoosting (Spectral/SAM) 4종 앙상블
- **시계열 후처리**: 2-state 마르코프 전이 확률을 적용하여 구름·그림자 등으로 인한 단발성 오탐(False Positive)을 제거하고 발생 전후 시계열 일관성을 확보했습니다.

### (2) SAM3 Segmentation & TTA
- 사전 라벨 영역에 패딩을 적용한 Box 프롬프트와 중심 Representative Point 프롬프트를 함께 입력합니다.
- 위성 궤도 및 일조 방향(지형 그림자) 특성을 고려하여, 성능 향상이 입증된 **좌우 반전(H-flip) TTA**만 선별 적용했습니다 (상하 반전은 지형 음영 왜곡으로 제외).
- 마스크 Logit 임계값 `1.0` 및 허용 오차 `0.05`의 세밀한 폴리곤 근사(`simplify`)를 적용했습니다.

### (3) Context-Aware Boundary Refinement
- SAM3 인코더의 다중 스케일 특징 맵에서 추출한 32채널 고해상도 문맥 특징과 배경 정규화된 분광 피처를 결합해 142차원 픽셀 특징을 구성했습니다.
- CatBoost 픽셀 분류기를 통해 기존 SAM 분할 마스크와 앙상블하여 미세 경계 오탐을 교정했습니다.

### (4) 18차 최종 하이퍼파라미터 설정
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

---

## 4. Validation & Environment

### 검증 프로토콜
- **공식 평가 산식**: `0.6 * Presence Macro-F1 + 0.4 * Shape mIoU`
- **교차검증**: 관측 창(Window) 및 지역(Scene) 단위 5-fold 교차검증을 적용하여 미학습 지역에 대한 일반화 성능을 집중 검증했습니다.
- 유효 정답 1,601건을 대상으로 점수를 산출했으며, 시계열 추론 시에는 무효(ignore) 영상을 포함한 1,691건 전체 시퀀스를 활용했습니다.

### 오프라인 실행 검증
- 외부 인터넷 접속이 차단된 완전한 오프라인 환경에서 17개 시점·3개 개소 테스트를 통과했습니다.
- 모델 로딩 및 무결성 검증을 포함해 총 101.7초(순수 추론 89.3초)가 소요되었으며, 메모리 피크는 RSS 약 2.56 GiB, GPU 약 2.24 GiB로 안정적인 자원 사용량을 확인했습니다.

---

## 5. How to Run

### 추론 (Inference)
`18_submission/` 디렉토리는 단독 실행이 가능한 형태로 패키징되어 있습니다.
```bash
cd 18_submission
jupyter notebook predict.ipynb
```

### 전체 파이프라인 재현 (Reproduce Training)
```bash
# 1. 시계열 OOF 특징 생성
python training18/full_oof.py

# 2. 문맥 픽셀 보정 모델 학습
python training18/train_semantic.py --group scene
python training18/train_semantic.py --group window

# 3. 최적 파라미터 탐색 및 제출 패키지 빌드
python training18/final_select.py
python training18/build_submission.py

# 4. 제출 무결성 검증 (Smoke Test)
python training18/run_smoke.py
```
