# V7 HV ONNX Model Directory

이 폴더는 V-hole V7 ONNX 모델을 로컬에서 배치하는 위치입니다.

보안/용량 규칙상 실제 모델 weight는 저장소에 커밋하지 않습니다. 추론을 실행하려면 별도 모델 저장소나 배포 패키지에서 아래 파일을 받아 이 폴더에 직접 넣어야 합니다.

필수 로컬 파일:

- `yolo_vhole.onnx`: coarse V-hole candidate detector
- `tip_model.onnx`: fine V-tip keypoint model
- `model_manifest.json`: export metadata

현재 Git에는 `README.md`와 `model_manifest.json`만 보존합니다. `.gitignore`에서 `*.onnx`를 제외하도록 설정되어 있으므로, 로컬에 ONNX 파일을 넣어도 실수로 커밋되지 않습니다.

기본 추론 경로는 이 폴더를 모델 위치로 사용합니다.

```bash
python tools/run_xray_inference.py \
  --image "/path/to/image.png" \
  --model-dir models/v7_hv_onnx
```
