# X-ray Measurement Preprocessing Module

X-ray 제품 이미지에서 `Plate`, `Mesh`, `3-pin`, `JIG`, `V-hole`을 검출하는 Python 알고리즘 정리본입니다.

이 폴더는 공개 저장소 기준으로, 현재 사용하는 최종 추론 코드와 V-hole V7 ONNX 추론/학습 스크립트를 정리한 버전입니다. 원본 Raw Data, 실행 결과물, HTML POC 화면, 실험용 중간 산출물, 모델 weight 파일은 포함하지 않습니다.

## 현재 기준

- V-hole: 메인 추론은 V7 ONNX 딥러닝 모델을 사용합니다.
- V-hole legacy: 기존 FFT 잔차 + polar 템플릿 매칭 코드는 비교/백업용으로 `vhole.py`에 보존했습니다.
- 모델 weight: `*.onnx`, `*.pt`, `*.pth`는 저장소에 올리지 않고 별도 보관합니다.
- JIG: 파일명/제품명 전용 분기 없이 12시, 3시, 6시, 9시 ROI에서 실제 선분을 찾습니다.
- TYPE_E 계열: V-hole이 3/9시 방향에 있어도 파일명 분기가 아니라 회전 추론과 pair geometry로 처리합니다.
- TYPE_A 회색 배경: 전처리 중 이미지가 하얗게 날아가는 케이스를 막는 배경 polarity 보정이 들어 있습니다.

## 폴더 구성

```text
preprocessing_module/
├── README.md
├── requirements.txt
├── pyproject.toml
├── src/
│   └── xray_3pin_detector/
│       ├── __init__.py
│       ├── preprocess.py
│       ├── circles.py
│       ├── detection.py
│       ├── products.py
│       ├── vhole.py
│       ├── vhole_v7.py
│       ├── vhole_unet86.py      # 실험: YOLO + Small U-Net V-hole 추론
│       ├── pipeline.py
│       ├── visualization.py
│       └── cli.py
├── models/
│   └── v7_hv_onnx/
│       ├── model_manifest.json
│       ├── README.md
│       └── 로컬 배치: yolo_vhole.onnx, tip_model.onnx
├── training/
│   ├── README.md
│   ├── run_coarse_fine_v7.py
│   ├── export_v7_to_onnx.py
│   ├── verify_v7_onnx.py
│   ├── make_v7_export_visualizations.py
│   └── requirements-training.txt
├── tools/
│   └── run_xray_inference.py
├── experiments/
│   └── vhole_unet86/            # 실험: Raw 이미지 대상 U-Net86 / V7 export 러너, legacy V-hole 정확도 평가
└── docs/
    └── V7_MODEL_EXPORT_README.md
```

## 처리 흐름

```text
입력 이미지
  ↓
preprocess.py
  - 알파 채널 / 배경 polarity 처리
  - 밝기, 음영 보정
  - morph 이미지 생성
  - 수평/수직 구조 강조 HV 이미지 생성
  ↓
circles.py
  - Plate 외곽 원 검출
  - Mesh 외곽 원 검출
  - Plate-Mesh 중심 편차 계산
  ↓
detection.py
  - Mesh 내부 3-pin 검출
  - Plate-Mesh gap 영역에서 JIG 4방향 선분 검출
  - JIG 조립/미조립 판정
  ↓
vhole_v7.py
  - HV crop 생성
  - YOLO ONNX로 V-hole 후보 검출
  - tip_model ONNX로 tip/corner keypoint refinement
  - pair geometry로 최종 V-hole 2점 선택
  - 현재 메인 추론 경로
  ↓
vhole.py
  - 기존 FFT 잔차 + polar 템플릿 매칭 V-hole 로직
  - 현재 메인 추론에서는 직접 호출하지 않는 보존/비교용 코드
  ↓
pipeline.py / cli.py
  - 결과 JSON 정리
  - summary.csv 생성
  - review image와 index.html 생성
```

## 주요 모듈

### `preprocess.py`

입력 이미지를 추론용 grayscale, morph, HV 이미지로 변환합니다.

핵심 로직:

- RGBA 이미지의 알파 채널 처리
- 이미지 테두리 통계를 이용한 light/dark background 판단
- TYPE_A 회색 배경처럼 완전 흰색은 아니지만 제품보다 밝은 배경 케이스 보정
- 음영 보정 후 구조 강조
- 수평/수직 morphology open을 결합한 HV 이미지 생성

### `circles.py`

Plate와 Mesh의 원 정보를 찾습니다.

핵심 로직:

- Plate: Canny edge → contour → fitEllipse 후보 생성, 실패 시 최대 면적 contour fallback
- Mesh: HV 이미지 warpPolar 방사형 프로파일로 경계 검출 후 최소제곱 원 피팅
- Plate와 Mesh 반지름, 중심 좌표 산출
- Plate-Mesh 편심을 px 및 mm 단위로 계산

### `detection.py`

3-pin과 JIG를 검출합니다.

3-pin 검출:

- Mesh 내부만 search mask로 제한
- HV 이미지를 blur한 뒤 여러 threshold 비율을 순회
- connected component 중 면적, circularity 조건을 만족하는 후보만 유지
- 3개 후보의 각도 간격이 가장 안정적인 조합을 선택

JIG 검출:

- Plate 내부에서 Mesh 바깥쪽만 `outside_mask`로 사용
- 12시/6시는 세로 선분, 3시/9시는 가로 선분만 남도록 ROI별 방향 필터 적용
- 방향 필터 이미지와 raw normalized 이미지 둘 다에서 LSD 후보를 생성
- LSD가 놓치는 짧은 직선 조각은 component 기반 후보로 보강
- 같은 축에 놓인 조각들을 grouped 후보로 병합
- top-bottom, left-right 후보 pair를 각각 선택
- pair 좌표 차이, 중심축 거리, 선분 점수, 전체 점수로 JIG 조립/미조립을 판정

JIG 후보 점수는 아래 요소를 합산/감점합니다.

- 선분 길이
- Plate 외곽에 붙어 있는 정도
- 선분 위 intensity와 support ratio
- 수평/수직 각도 오차
- Plate 외곽에서 안쪽으로 들어온 거리
- 중심축에서 벗어난 거리

현재 로직은 예전처럼 교차점 후보를 먼저 찾는 방식이 아니라, 4방향 ROI에서 실제 선분을 고른 뒤 두 축을 연결해 JIG 중심을 계산합니다.

### `products.py`

이미지 파일명 stem을 제품군으로 분류합니다. 현재 V-hole V7 crop margin 선택에 사용됩니다.

매칭 방식:

```text
완전 일치 → 접두사 일치 → unknown
```

### `vhole_v7.py`

V-hole V7 ONNX 추론 모듈입니다.

모델 구조:

```text
HV 이미지
  ↓
Stage 1: YOLO ONNX coarse detector
  ↓
Stage 2: 3-keypoint heatmap ONNX model
  - tip
  - corner1
  - corner2
  ↓
refinement / pair validation
  ↓
최종 V-hole pair, 축 방향, 회전 보정각
```

일반 제품의 12/6시 V-hole과 TYPE_E처럼 3/9시 방향에 있는 V-hole을 모두 pair geometry 기준으로 처리합니다.

### `vhole.py`

기존 V-hole 템플릿 매칭 로직입니다. 현재 `pipeline.py`의 메인 추론은 `vhole_v7.py`를 사용하지만, 이전 방식과 결과를 비교하거나 장애 시 참고할 수 있도록 보존했습니다.

기존 방식의 처리 흐름:

```text
HV 이미지
  ↓
Mesh edge 도넛 영역을 polar 좌표로 변환
  ↓
각 반경 row에 FFT 적용
  ↓
주기적 mesh 패턴 복원 후 원본에서 제거
  ↓
잔차 이미지에서 V-hole 이상치 강조
  ↓
제품군별 template matching
  ↓
NMS와 각도 제약으로 V-hole 2점 선택
```

주의:

- 이 파일은 현재 메인 추론 결과에 영향을 주지 않습니다.
- 템플릿 매칭 방식은 제품군별 대표 템플릿과 GT 각도 정보가 필요합니다.
- 신규 POC 기준 검출은 `vhole_v7.py`의 ONNX 모델을 기준으로 봅니다.

### `pipeline.py`

이미지 1장에 대한 전체 추론 진입점입니다. 처음 코드를 볼 때는 이 파일을 먼저 보면 흐름을 가장 빠르게 파악할 수 있습니다.

### `visualization.py`

분석용 review image를 생성합니다.

기본 패널:

- Raw 이미지
- Geometry overlay
- HV + V-hole overlay
- V-hole zoom crop

## 설치

Python 3.10 이상을 권장합니다.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

개발 모드로 설치하려면:

```bash
pip install -e .
```

## 단일 이미지 실행

설치 없이 source tree 기준으로 실행:

```bash
python tools/run_xray_inference.py \
  --image "/path/to/image.png" \
  --output-dir outputs/single_test
```

패키지 설치 후 실행:

```bash
xray-3pin-infer \
  --image "/path/to/image.png" \
  --output-dir outputs/single_test
```

## 배치 실행

Raw Data 폴더 전체를 실행하려면:

```bash
python tools/run_xray_inference.py \
  --data-root "/path/to/Raw Data" \
  --output-dir outputs/all_products
```

빠른 검증용으로 개수를 제한할 수 있습니다.

```bash
python tools/run_xray_inference.py \
  --data-root "/path/to/Raw Data" \
  --output-dir outputs/quick_check \
  --limit 10
```

폴더별 샘플 수를 제한할 수도 있습니다.

```bash
python tools/run_xray_inference.py \
  --data-root "/path/to/Raw Data" \
  --output-dir outputs/folder_sample \
  --max-per-folder 3
```

중간 전처리 이미지를 같이 저장하려면 `--write-intermediates` 옵션을 사용합니다.

```bash
python tools/run_xray_inference.py \
  --image "/path/to/image.png" \
  --output-dir outputs/debug_one \
  --write-intermediates
```

## 출력물

```text
outputs/single_test/
├── summary.csv
├── full_results.json
├── index.html
└── review_images/
```

주요 파일:

- `summary.csv`: 이미지별 핵심 수치 요약
- `full_results.json`: 이미지별 상세 추론 결과
- `index.html`: review image를 한 번에 보기 위한 로컬 HTML
- `review_images/`: 분석용 overlay 이미지
- `intermediates/`: `--write-intermediates` 사용 시 raw/processed/morph/HV 이미지

## V-hole 모델 파일

현재 추론에 필요한 ONNX 모델 weight는 저장소에 포함하지 않습니다. 추론 실행 전 별도 모델 보관 위치에서 아래 파일을 받아 `models/v7_hv_onnx/`에 로컬로 배치해야 합니다.

```text
models/v7_hv_onnx/
├── yolo_vhole.onnx   # Git 제외, 로컬 배치
├── tip_model.onnx    # Git 제외, 로컬 배치
├── model_manifest.json
└── README.md
```

`.gitignore`에서 `*.onnx`, `*.pt`, `*.pth`, `*.ckpt`를 제외하므로, 모델 파일을 로컬에 넣어도 실수로 커밋되지 않습니다.

모델 파일이 없으면 `tools/run_xray_inference.py` 실행 시 필요한 파일명을 안내하는 에러 메시지가 출력됩니다.

## 학습 / export

학습 관련 코드는 `training/`에 정리되어 있습니다.

전체 흐름:

```text
training/run_coarse_fine_v7.py
  ↓
yolo_best.pt + tip_model.pth
  ↓
training/export_v7_to_onnx.py
  ↓
yolo_vhole.onnx + tip_model.onnx
  ↓
models/v7_hv_onnx/에 로컬 반영
  ↓
training/verify_v7_onnx.py
```

학습 의존성은 추론 의존성과 분리되어 있습니다.

```bash
pip install -r training/requirements-training.txt
```

자세한 학습 방법론과 실행 옵션은 [training/README.md](./training/README.md)를 확인합니다.

## 커밋 전 체크리스트

포함해야 하는 것:

- `src/xray_3pin_detector/`
- `models/v7_hv_onnx/model_manifest.json`
- `models/v7_hv_onnx/README.md`
- `training/`
- `tools/`
- `docs/`
- `README.md`, `requirements.txt`, `pyproject.toml`

포함하지 않는 것:

- 원본 `Raw Data`
- `outputs/`, `review_images/`, `intermediates/`
- 학습 산출물 폴더
- `*.onnx`, `*.onnx.data`
- `*.pt`, `*.pth`, `*.ckpt`
- `__pycache__/`, `.DS_Store`
- 실험용 notebook, scratch 파일

## 추천 코드 확인 순서

1. `src/xray_3pin_detector/pipeline.py`
2. `src/xray_3pin_detector/preprocess.py`
3. `src/xray_3pin_detector/circles.py`
4. `src/xray_3pin_detector/detection.py`
5. `src/xray_3pin_detector/vhole_v7.py`
6. `src/xray_3pin_detector/vhole.py`
7. `tools/run_xray_inference.py`
8. `training/README.md`

## 메모

- 이 폴더는 공개용 Python 알고리즘 패키지입니다.
- 별도 배포용 standalone HTML 데모는 이 폴더에 포함하지 않았습니다 (브라우저 POC는 루트의 `poc_sample_page/` 참고).
- 기존 템플릿 기반 V-hole 코드는 `src/xray_3pin_detector/vhole.py`에 보존했습니다.
- 현재 메인 V-hole 추론은 `src/xray_3pin_detector/vhole_v7.py`의 ONNX 딥러닝 로직입니다.
- 커밋하기 전에는 `git status`에서 의도하지 않은 Raw Data나 output 파일이 잡히지 않는지 반드시 확인합니다.
