# 반도체 부품 X-ray 기하 계측 (X-ray Geometry Measurement)

반도체 부품 X-ray 이미지에서 Plate·Mesh·3핀·JIG·V홀 구조를 자동으로 찾아 위치·편심·각도를 계산하는 Python 계측 파이프라인과 브라우저 POC입니다.

## 한눈에 보기

- **문제**: 작업자가 X-ray 이미지를 눈으로 보며 구조 위치와 편심을 재는 반복 작업은 부담이 크고, 사람·이미지 상태마다 기준이 흔들림
- **접근**: 규칙 기반 영상처리(전처리 → 원 검출 → 3핀 → JIG 선분 검색)에 V홀만 딥러닝(YOLOv8 + ResNet18 keypoint heatmap, ONNX)을 결합
- **결과물**: 설치형 패키지 `xray_3pin_detector` + CLI `xray-3pin-infer`(단일/폴더 배치 → `summary.csv`, `full_results.json`, overlay 리뷰 이미지, `index.html`), 서버 없이 여는 PyScript POC 페이지 2종
- **기술 스택**: Python 3.10+, OpenCV, NumPy, ONNX Runtime · 학습: PyTorch, timm, Ultralytics YOLOv8 · POC: PyScript
- **포함**: 추론 코드, V7 학습/ONNX export/검증 스크립트, U-Net 후속 실험 코드, 브라우저 POC
- **미포함**: X-ray 원본 이미지, 라벨 데이터, ONNX/PyTorch 가중치, V홀 템플릿 자산 (제품군은 `TYPE_A`~`TYPE_E`로 익명화)

## 왜 필요한가

부품 X-ray 검사에서는 원형 구조끼리의 중심 편차, 핀 배치, JIG 조립 여부, V홀 방향 같은 기하 정보를 확인해야 합니다.
이 작업을 사람이 이미지마다 직접 확인하면 시간이 많이 들고, 같은 이미지라도 작업자나 밝기·대비 상태에 따라 측정 기준이 달라질 수 있습니다.
이 프로젝트는 같은 입력에 같은 규칙을 적용해 계측값과 검토용 overlay를 자동으로 만들어, 사람은 결과 확인과 예외 케이스 판단에 집중할 수 있게 하는 것을 목표로 합니다.

## 동작 방식

```text
X-ray 이미지
  → 전처리      배경 polarity 통일, 음영·대비 보정, morph / HV(수평·수직 강조) 이미지 생성
  → 원 검출     Plate 외곽 원 + Mesh 원 → 중심·반지름, Plate-Mesh 편심(px, mm)
  → 3핀         Mesh 내부에서 threshold 순회 + 형상 필터 → 각도 간격이 가장 안정적인 3점
  → JIG         Plate-Mesh 갭의 12/3/6/9시 ROI에서 실제 선분 검색 → 조립 여부·JIG 중심
  → V홀         V7: YOLOv8 coarse 후보 → ResNet18 3-keypoint heatmap → pair geometry로 2점 선택
  → 출력        summary.csv / full_results.json / review 이미지 / index.html
```

- **제품명 분기 최소화**: JIG는 파일명이 아니라 4방향 ROI에서 찾은 선분으로 판정하고, V홀이 3/9시에 있는 TYPE_E 계열도 회전 추론과 pair geometry로 처리합니다.
- **전처리 강건성**: 이미지 테두리 통계로 밝은/어두운 배경을 판단해, 회색 배경 이미지가 하얗게 날아가는 케이스를 보정합니다.
- **V홀만 학습 모델**: 나머지는 규칙 기반으로 두고, 템플릿 튜닝 부담이 컸던 V홀 단계만 ONNX 모델로 교체했습니다. 기존 규칙 기반 코드(`vhole.py`)는 비교용으로 보존합니다.
- **학습/추론 분리**: 추론은 `numpy`·`opencv-python`·`onnxruntime`·`Pillow`만 필요하고, PyTorch는 학습/실험에서만 씁니다.

## 시도와 개선 과정

V홀 검출 방식의 변화입니다.

| 단계 | 방식 | 위치 | 한계 / 개선 |
|---|---|---|---|
| 1. 규칙 기반 (legacy) | Mesh 경계 도넛을 polar 변환 → row별 FFT로 주기 패턴 제거 → 잔차에서 제품군별 template matching → NMS·각도 제약 | `src/.../vhole.py`, POC 페이지 | 제품군별 템플릿·GT 각도·밴드 튜닝 필요, 3/9시 방향 V홀에 취약 |
| 2. V7 (현재 메인) | YOLOv8 coarse 검출 → ResNet18 3-keypoint(tip, corner×2) heatmap → 삼각형 기하 refinement → ONNX export | `src/.../vhole_v7.py`, `training/` | 템플릿 불필요. 학습 데이터 49장 / V홀 98개. 학습셋 평균 tip 오차 약 1.22px (**학습셋 기준, 일반화 성능 아님**) |
| 3. U-Net 실험 | YOLO + Small U-Net 마스크 → tip 디코딩 + shape prior, 12/6·3/9시 radial candidate 비교 | `src/.../vhole_unet86.py`, `experiments/vhole_unet86/` | 측면 V홀 미검 보완 목적의 실험. 메인 파이프라인에는 연결되지 않음 |

## 빠른 시작

```bash
cd preprocessing_module
python -m venv .venv && source .venv/bin/activate
pip install -e .                 # 추론용 (학습까지: pip install -e ".[training]")
```

**1) 모델 가중치 배치** — 저장소에 포함되지 않습니다. 직접 학습·export한 파일을 아래 위치에 둡니다 (없으면 CLI가 필요한 파일명을 안내하고 종료).

```text
preprocessing_module/models/v7_hv_onnx/
├── yolo_vhole.onnx     # 로컬 배치 (Git 제외)
├── tip_model.onnx      # 로컬 배치 (Git 제외)
└── model_manifest.json # 포함
```

**2) 추론 실행**

```bash
# 단일 이미지 (--image는 여러 번 지정 가능)
xray-3pin-infer --image /path/to/image.png --output-dir outputs/single_test

# 폴더 배치 (빠른 확인용 --limit / --max-per-folder)
xray-3pin-infer --data-root /path/to/images --output-dir outputs/batch --limit 10

# 설치 없이 source tree에서 실행
python tools/run_xray_inference.py --image /path/to/image.png --write-intermediates
```

주요 옵션: `--model-dir`(기본 `models/v7_hv_onnx`), `--pixel-size-mm`(mm 환산, 기본 `0.030769`), `--include-rotated-type-e`, `--write-intermediates`(raw/processed/morph/HV 중간 이미지 저장).

**3) 브라우저 POC** — `poc_sample_page/xray_inspection_demo.html`을 Chrome/Edge에서 직접 엽니다. 첫 실행 시 PyScript 런타임과 OpenCV·NumPy·Pillow를 네트워크로 받아오므로 로딩 시간이 걸립니다. V홀 템플릿은 `vhole_template_builder.html`로 만들 수 있습니다.

## 폴더 구조

```text
industrial-xray-geometry-measurement/
├── poc_sample_page/                 # 브라우저 단독 실행 POC (HTML + PyScript)
│   ├── xray_inspection_demo.html    #   단일/배치 계측, overlay, V홀 수동 보정, CSV·ZIP 저장
│   └── vhole_template_builder.html  #   대표 이미지 1장으로 V홀 상/하단 템플릿 생성
└── preprocessing_module/            # Python 패키지 (pyproject.toml, requirements.txt)
    ├── src/xray_3pin_detector/      #   전처리·원·3핀·JIG·V홀 추론, pipeline, CLI
    ├── models/v7_hv_onnx/           #   ONNX 배치 위치 (manifest·README만 포함)
    ├── training/                    #   V7 학습 / ONNX export / 검증 / 시각화 스크립트
    ├── experiments/vhole_unet86/    #   U-Net V홀 실험 러너, legacy V홀 정확도 평가
    ├── tools/run_xray_inference.py  #   설치 없이 CLI를 실행하는 래퍼
    └── docs/                        #   V7 export 메타데이터 설명
```

## 기술 상세

<details>
<summary>모듈별 역할 (코드 읽는 순서)</summary>

1. `pipeline.py` — 이미지 1장 → 전체 결과(JSON/summary)를 묶는 진입점. 흐름 파악은 여기서 시작
2. `preprocess.py` — 알파 채널·배경 polarity 처리, 음영 보정, 수평/수직 morphology open을 결합한 HV 이미지
3. `circles.py` — Plate: Canny → contour → fitEllipse(실패 시 최대 면적 contour), Mesh: HV 이미지 warpPolar 방사형 프로파일 → 각도별 경계점 최소제곱 원 피팅, 편심(px/mm) 계산
4. `detection.py` — 3핀 검출, JIG 4방향 선분 검출 및 조립/미조립 판정
5. `vhole_v7.py` — V7 ONNX 추론 (`V7ONNXDetector`, 기본 `CPUExecutionProvider`)
6. `vhole.py` — legacy 규칙 기반 V홀 (메인 추론에서는 호출하지 않음)
7. `products.py` — 파일명 stem → 제품군 분류(완전 일치 → 접두사 일치 → unknown). V7 crop margin 선택에 사용
8. `visualization.py` — review 이미지(Raw / Geometry overlay / HV + V홀 overlay / V홀 zoom)
9. `cli.py` — 이미지 수집, 배치 실행, `summary.csv`·`full_results.json`·`index.html` 생성

</details>

<details>
<summary>JIG 검출 상세</summary>

- Plate 내부이면서 Mesh 바깥인 영역만 `outside_mask`로 사용
- 12/6시는 세로 선분, 3/9시는 가로 선분만 남도록 ROI별 방향 필터 적용
- 방향 필터 이미지와 raw normalized 이미지 양쪽에서 OpenCV LSD 후보 생성, LSD가 놓친 짧은 조각은 component 기반 후보로 보강
- 같은 축의 조각을 병합한 뒤 top-bottom, left-right pair를 선택
- 점수 요소: 선분 길이, Plate 외곽 부착 정도, intensity·support ratio, 수평/수직 각도 오차, 외곽에서 안쪽까지 거리, 중심축 이탈 거리
- 교차점을 먼저 찾던 이전 방식 대신, 4방향 실제 선분을 고른 뒤 두 축을 연결해 JIG 중심을 계산

</details>

<details>
<summary>V7 모델 학습 · export 흐름</summary>

```text
training/run_coarse_fine_v7.py   (--data-dir, --groups, --out-prefix)
  → yolo_best.pt + tip_model.pth
training/export_v7_to_onnx.py    (--export-dir, --opset 17)
  → yolo_vhole.onnx + tip_model.onnx + model_manifest.json
  → models/v7_hv_onnx/ 로 로컬 복사
training/verify_v7_onnx.py       (--data-dir, --export-dir, --max-images)
  → 이미지별 tip 오차, 평균/중앙값/최대 오차 출력
```

- 입력 데이터: HV crop 이미지(`*_hv.png`) + LabelMe JSON(label `v`). 중심에 가장 가까운 점을 tip, 나머지 두 점을 corner로 사용
- `--groups`로 지정한 제품군 prefix 기준 leave-one-group-out fold를 만들고, test group의 5px 이내 tip 성공률을 출력
- 주요 설정(`model_manifest.json`): patch 192, heatmap 96×96, keypoint 3, YOLO imgsz 1280, ONNX opset 17
- 자세한 내용: [training/README.md](./preprocessing_module/training/README.md), [docs/V7_MODEL_EXPORT_README.md](./preprocessing_module/docs/V7_MODEL_EXPORT_README.md)

</details>

<details>
<summary>출력물</summary>

| 파일 | 내용 |
|---|---|
| `summary.csv` | 이미지별 핵심 수치: V홀 방법·축·회전 보정각·pair 거리, 핀 수, JIG 상태, Plate-Mesh dx/dy/거리, Mesh 반지름, 처리 시간 |
| `full_results.json` | 이미지별 상세 추론 결과 |
| `review_images/` | 분석용 overlay 이미지 |
| `index.html` | review 이미지를 한 번에 보는 로컬 페이지 |
| `intermediates/` | `--write-intermediates` 사용 시 raw/processed/morph/HV 이미지 |

</details>

<details>
<summary>브라우저 POC 상세</summary>

- 단일 HTML 안에 UI(HTML/CSS/JS)와 처리 로직(PyScript 위 Python + OpenCV + NumPy)이 함께 있고, 일부 배치 처리는 Web Worker로 병렬 실행
- 입력: 단일 이미지 / 디렉토리, 선택적 템플릿 디렉토리, JIG 분기(자동/있음/없음), `mm/px` 환산값
- 출력: 결합 overlay PNG + `measurement_summary.csv` (`showDirectoryPicker` 미지원 브라우저는 ZIP/개별 다운로드로 대체)
- 템플릿 디렉토리 형식과 필드 설명: [poc_sample_page/README.md](./poc_sample_page/README.md)

</details>

## 한계와 다음 단계

- **바로 재현 불가**: X-ray 데이터, 라벨, ONNX 가중치, 템플릿이 모두 미포함이라, CLI 실행에는 별도로 학습·export한 모델과 이미지가 필요합니다.
- **검증 범위**: 저장소에 남아 있는 V7 수치는 학습셋 49장 기준입니다. leave-one-group-out 검증 스크립트는 있으나 그 결과는 포함되어 있지 않아, 일반화 성능은 별도 데이터로 확인해야 합니다.
- **mm 환산**: 고정 `--pixel-size-mm` 값으로 환산하며, 장비별 calibration 절차는 저장소에 없습니다.
- **실험 코드 정리 필요**: U-Net 실험은 메인 파이프라인과 분리되어 있고, 학습 코드·가중치가 없으며 기본 경로가 원 작업 환경 기준이라 인자 지정 없이 그대로 실행되지 않습니다. 브라우저 POC의 V홀 로직도 V7이 아닌 템플릿 기반입니다.
