# Semiconductor Parts X-ray Geometry Measurement

반도체 부품 X-ray 이미지에서 기하 구조를 자동 검출하고 픽셀 단위 계측값을 계산하는 **브라우저 POC 페이지**와 **Python 전처리·계측 파이프라인**입니다.

`Plate / Mesh / 3핀 / JIG / V홀` 같은 구조를 규칙 기반 영상처리 알고리즘으로 검출하고, 각 구조의 위치·각도·편심을 일관되게 산출하는 것이 전체 흐름의 핵심입니다.

하나의 완결된 상용 애플리케이션은 아니고, **브라우저 단독 실행형 데모**(POC)와 **Python 알고리즘 모듈**(preprocessing_module)을 같은 맥락에서 볼 수 있도록 정리한 코드 아카이브입니다.

## Project Snapshot

- Domain: 반도체 부품 X-ray 계측 / 검사 자동화
- Problem: 사람이 X-ray 이미지를 보며 구조 위치와 편심을 확인하면 반복 검토 부담이 크고, 계측 기준이 작업자·이미지 상태에 따라 흔들릴 수 있음
- Approach: 배경/음영/대비 보정 후 plate, mesh, pin, JIG, V-hole 구조를 단계적으로 검출하고, V-hole은 polar transform + FFT residual + template matching으로 보강
- Output: 중심 좌표, 편심, 거리, V-hole 각도, overlay 이미지, CSV/ZIP 저장 결과
- Scope: 공개 가능한 계측 알고리즘과 브라우저 POC 흐름 정리. 실제 X-ray 이미지, 고객 데이터, 템플릿 파일은 포함하지 않음

## What to Review First

평가자가 빠르게 구조를 보려면 아래 파일부터 보면 됩니다.

1. `preprocessing_module/src/xray_3pin_detector/preprocess.py`  
   X-ray 이미지의 배경 polarity·음영·대비 보정과 구조 강조를 담당하는 전처리 흐름입니다.

2. `preprocessing_module/src/xray_3pin_detector/circles.py`  
   Plate / Mesh 원형 구조를 검출하고 중심점과 반지름, 편심을 산출합니다.

3. `preprocessing_module/src/xray_3pin_detector/detection.py`  
   3-pin 검출과 12/3/6/9시 ROI 기반 JIG 선분 검출 등 주요 계측 로직이 모여 있습니다.

4. `preprocessing_module/src/xray_3pin_detector/vhole_v7.py`  
   V-hole 메인 추론. YOLOv8 coarse 검출 + ResNet18 3-keypoint heatmap(ONNX)으로 V-hole 꼭짓점을 찾습니다.
   기존 규칙 기반(polar + FFT 잔차 + 템플릿 매칭) 방식은 `vhole.py`에 비교용으로 보존했습니다.

5. `preprocessing_module/src/xray_3pin_detector/pipeline.py`, `cli.py`  
   이미지 1장 → 전체 계측 결과를 묶는 파이프라인과 `xray-3pin-infer` CLI 엔트리입니다.

6. `poc_sample_page/xray_inspection_demo.html`, `poc_sample_page/vhole_template_builder.html`  
   브라우저에서 계측 결과 overlay, CSV 저장, 수동 보정, V-hole 템플릿 생성을 확인할 수 있는 POC 화면입니다.

## 구성

```text
industrial-xray-geometry-measurement/
├── README.md
├── .gitignore
├── poc_sample_page/          # 브라우저 POC (HTML + PyScript)
└── preprocessing_module/     # Python 전처리·계측 파이프라인
```

## 하위 폴더 요약

### `poc_sample_page/`

서버 없이 브라우저에서 바로 실행되는 X-ray 계측 데모 페이지입니다.

- `xray_inspection_demo.html` — 단일/배치 이미지 계측 + 오버레이 + CSV 저장
- `vhole_template_builder.html` — 대표 이미지 1장으로 V홀 상/하단 템플릿 생성

PyScript + OpenCV + NumPy + Pillow를 브라우저 안에서 돌리는 구조라, 정적 파일만으로 계측 데모가 가능합니다.

자세한 사용법은 [poc_sample_page/README.md](./poc_sample_page/README.md) 참고.

### `preprocessing_module/`

X-ray 이미지 한 장을 받아 다음을 계산하는 Python 모듈입니다.

- 배경 통일·음영 보정·구조 강조 전처리
- Plate / Mesh 외곽 원 검출 및 편심 계산
- Mesh 내부 3핀 검출
- Plate-Mesh 갭 영역의 JIG 십자 검출
- V홀 2점 검출: V7 딥러닝(YOLOv8 + keypoint heatmap, ONNX) 메인 + 규칙 기반(polar + FFT 잔차) legacy

`src/xray_3pin_detector/` 패키지(`pip install -e .` 후 `xray-3pin-infer`)로 단일 이미지·폴더 배치 추론을 지원하고,
`training/`에 V7 학습·ONNX export·검증 스크립트, `experiments/`에 U-Net 기반 V-hole 후속 실험이 있습니다.

### V-hole 검출 방식의 변화

| 단계 | 방식 | 한계 / 개선 |
|---|---|---|
| 1 | polar 변환 + FFT 잔차 + 템플릿 매칭 (규칙 기반) | 제품군별 템플릿·밴드 튜닝이 필요하고, 방향이 다른 제품(3/9시 V-hole)에 취약 |
| 2 | V7: YOLOv8 coarse 검출 → ResNet18 3-keypoint heatmap → 삼각형 기하 refinement | 템플릿 없이 학습 이미지 49장으로 꼭짓점 평균 오차 약 1.2px (학습셋 기준) |
| 3 | 실험: YOLO + Small U-Net 마스크 → tip 디코딩 + shape prior | 측면 V-hole 미검 보완을 위한 radial candidate crop 비교 실험 |

자세한 알고리즘 설명은 [preprocessing_module/README.md](./preprocessing_module/README.md) 참고.

## 읽는 순서

1. 전체 맥락은 이 루트 README
2. 화면·UI 흐름을 보고 싶으면 `poc_sample_page/`
3. 실제 계측 알고리즘 코드를 보고 싶으면 `preprocessing_module/`

## 참고

- 입력 이미지·템플릿·테스트 산출물 같은 데이터는 이 저장소에 포함되지 않습니다(`.gitignore` 참조).
- 코드 구조와 알고리즘 흐름을 이해하는 것이 목적인 정리본이고, 특정 데이터셋에 종속된 실행을 위한 저장소는 아닙니다.
