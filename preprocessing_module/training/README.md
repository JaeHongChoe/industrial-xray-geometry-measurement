# V7 V-hole 학습 코드

이 폴더는 V-hole 딥러닝 모델 V7을 학습하고, Python 추론 파이프라인에서 사용할 ONNX 모델로 변환하기 위한 코드 정리본입니다.

현재 추론 파이프라인의 V-hole 검출은 `src/xray_3pin_detector/vhole_v7.py`에서 ONNX 모델을 사용합니다. 이 `training/` 폴더는 그 ONNX 모델을 만들기 전 단계인 PyTorch/YOLO 학습 흐름을 보존하기 위한 영역입니다.

## 폴더 구성

```text
training/
├── README.md
├── requirements-training.txt
├── run_coarse_fine_v7.py
├── export_v7_to_onnx.py
├── verify_v7_onnx.py
└── make_v7_export_visualizations.py
```

### `run_coarse_fine_v7.py`

V7 학습 메인 스크립트입니다.

구조:

- Stage 1: YOLOv8 기반 coarse V-hole 후보 검출
- Stage 2: ResNet18 기반 3-keypoint heatmap 모델
- 후처리: tip/corner 삼각형 기반 safety-guard refinement

입력 데이터는 HV crop 이미지와 LabelMe JSON 쌍을 전제로 합니다.

```text
vhole_hv_crops/
├── TYPE_C-14-G125_hv.png
├── TYPE_C-14-G125_hv.json
├── TYPE_A-1-G003_hv.png
└── TYPE_A-1-G003_hv.json
```

JSON에는 label이 `v`인 polygon/points가 들어 있어야 합니다. 스크립트는 각 V-hole에서 중심 쪽에 가장 가까운 점을 tip으로 보고, 나머지 두 점을 corner로 사용합니다.

### `export_v7_to_onnx.py`

학습된 PyTorch/YOLO weight를 ONNX로 변환합니다.

필요 파일:

- `tip_model.pth`
- `yolo_best.pt`
- `model_manifest.json`

생성 파일:

- `tip_model.onnx`
- `yolo_vhole.onnx`
- 갱신된 `model_manifest.json`

### `verify_v7_onnx.py`

Export된 ONNX 모델이 학습 코드와 동일한 입력/후처리 흐름으로 동작하는지 확인합니다.

### `make_v7_export_visualizations.py`

학습 또는 export 결과를 사람이 검토하기 쉽게 overlay, zoom review, contact sheet 형태로 시각화합니다.

## 설치

추론만 할 때는 루트 `requirements.txt`만 설치하면 됩니다. 학습까지 하려면 별도 의존성을 설치합니다.

```bash
pip install -r requirements.txt
pip install -r training/requirements-training.txt
```

CUDA 환경에서는 사용하는 PyTorch/CUDA 버전에 맞게 `torch`를 먼저 설치한 뒤 나머지를 설치하는 편이 안전합니다.

## 학습 실행

기본 실행:

```bash
python training/run_coarse_fine_v7.py \
  --data-dir vhole_hv_crops \
  --groups TYPE_C TYPE_B TYPE_A TYPE_D \
  --out-prefix training/vhole_cv_v7_hybrid
```

동작 방식:

1. `--data-dir` 아래의 `*_hv.png`, `*_hv.json` 쌍을 읽습니다.
2. `--groups`에 지정된 prefix를 기준으로 leave-one-group-out 방식의 fold를 만듭니다.
3. 각 fold마다 YOLO coarse detector를 학습합니다.
4. 각 fold마다 tip/corner heatmap 모델을 학습합니다.
5. test group에 대해 5px 이내 tip localization 성공률을 출력합니다.

출력 예:

```text
training/vhole_cv_v7_hybrid_TYPE_C/
├── training_log.txt
├── tip_model.pth
├── stage1_yolo/
├── *_result.png
└── final_results.json
```

## ONNX Export

Export 스크립트는 하나의 export 폴더 안에 아래 파일들이 있다고 가정합니다.

```text
export_dir/
├── tip_model.pth
├── yolo_best.pt
└── model_manifest.json
```

실행:

```bash
python training/export_v7_to_onnx.py \
  --export-dir path/to/export_dir \
  --opset 17
```

성공하면 아래 파일이 만들어집니다.

```text
export_dir/
├── tip_model.onnx
├── yolo_vhole.onnx
└── model_manifest.json
```

최종 추론 환경에 반영하려면 생성된 ONNX 파일을 로컬 모델 폴더로 복사합니다. ONNX weight는 저장소에 커밋하지 않습니다.

```bash
cp path/to/export_dir/yolo_vhole.onnx models/v7_hv_onnx/yolo_vhole.onnx
cp path/to/export_dir/tip_model.onnx models/v7_hv_onnx/tip_model.onnx
cp path/to/export_dir/model_manifest.json models/v7_hv_onnx/model_manifest.json
```

## ONNX 검증

```bash
python training/verify_v7_onnx.py \
  --data-dir vhole_hv_crops \
  --export-dir models/v7_hv_onnx \
  --max-images 12
```

검증 결과는 이미지별 tip 오차와 전체 평균/중앙값/최대 오차로 출력됩니다.

## 커밋 시 주의

학습 산출물은 커질 수 있으므로 기본적으로 `.gitignore`에 제외했습니다.

제외 대상:

- `*.pt`
- `*.pth`
- `*.ckpt`
- `stage1_yolo/`
- `vhole_cv_v7_hybrid*/`
- `training/visualizations/`

ONNX 모델은 현재 추론 실행에 필요하지만 저장소에는 포함하지 않습니다. `*.onnx`, `*.pt`, `*.pth`, `*.ckpt`는 `.gitignore`로 제외하고, 모델 weight는 별도 보관소나 배포 패키지로 관리합니다.

## 학습 코드와 추론 코드의 관계

```text
training/run_coarse_fine_v7.py
  ↓ 학습
tip_model.pth + yolo_best.pt
  ↓ export_v7_to_onnx.py
tip_model.onnx + yolo_vhole.onnx
  ↓ 로컬 복사
models/v7_hv_onnx/
  ↓ 사용
src/xray_3pin_detector/vhole_v7.py
```

즉, `training/`은 모델을 새로 만들거나 개선할 때 보는 영역이고, 일반 추론 실행은 `src/`, `tools/`를 보면 됩니다.
