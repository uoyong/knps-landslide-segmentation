# KNPS Landslide Segmentation (2026)

Sentinel-2 시계열 위성 영상 기반 산사태 변화 탐지 및 형상 분할 솔루션  
18차 Private 점수: `0.714650707` (최종 5위 달성)

---

## 1. Overview

국립공원공단 산사태 시계열 위성 영상(Sentinel-2 다중분광 밴드)을 활용하여 산사태 발생 여부를 판정하고, 발생 지점의 정확한 경계를 분할(Segmentation)하는 파이프라인입니다.

본 저장소는 최종 5위 모델인 **18차 솔루션(`sam3-context-refine-v18`)**을 기준으로 구성되어 있습니다.

### 핵심 구성요소
1. **시계열 다중분광 특징 추출**: Sentinel-2 밴드(RGB, NIR, RedEdge 등) 및 dNDVI 변화량 추출
2. **산사태 존재 판정기 (Presence Gate)**:
   - ExtraTrees + HistGradientBoosting 기반 4모델 앙상블
   - 2-state 마르코프 체인(Forward/Backward) 기반 시계열 평활화 및 연속성 보정
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
│   ├── assets/           # 추론 모듈 및 학습된 경량 모델 가중치
│   ├── predict.ipynb     # 최종 채점용 추론 노트북
│   ├── requirements.txt  # 실행 의존성
│   └── README.md         # 18차 제출 상세 스펙
├── training18/           # 18차 모델 학습, 최적화 및 앙상블 탐색 파이프라인
│   ├── train_catboost.py # 문맥 경계 보정기 학습
│   ├── final_select.py   # 교차검증 기반 최적 하이퍼파라미터 탐색
│   └── build_submission.py # 제출물 자동 패키징
├── training17/           # 17차 베이스라인 및 분광 정제 파이프라인
└── .gitignore
```

---

## 3. Pipeline Details

### (1) Presence Gate
- **Input**: 다중 시점 분광 차분 및 SAM 영역 통계량 (1,468차원)
- **Model**: ExtraTrees (Spectral/SAM) + HistGradientBoosting (Spectral/SAM) 4종 앙상블
- **Temporal Post-processing**: 마르코프 전이 확률을 이용해 단발성 오탐(구름, 그림자 등)을 필터링하고 발생 전후 일관성을 확보

### (2) SAM3 Segmentation & TTA
- 사전 라벨 영역을 기반으로 여유 버퍼(Padding)를 부여한 Box 프롬프트와 중심 Representative Point 프롬프트를 함께 입력
- 위성 궤도 및 지형 그림자 특성을 고려하여 성능 향상이 입증된 **좌우 반전(H-flip) TTA**만 선별 적용 (상하 반전은 지형 음영 왜곡으로 제외)
- 마스크 Logit 임계값 `1.0` 및 세밀한 폴리곤 근사(`simplify(0.05)`) 적용

### (3) Context-Aware Boundary Refinement
- SAM3 인코더의 다중 스케일 특징 맵에서 고해상도 컨텍스트(32ch)를 추출
- 배경 정규화된 분광 피처와 결합하여 CatBoost 픽셀 분류기를 학습
- 기존 SAM 분할 결과와 앙상블하여 경계 뭉개짐(Erosion/Dilation) 및 미세 경계 오탐을 교정

---

## 4. Results

- **최종 모델**: 18차 (`sam3-context-refine-v18`)
- **Private Score**: `0.714650707` (최종 5위 달성)

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
