# Browser-Based Semiconductor X-ray Measurement POC

반도체 부품 X-ray 계측 결과를 서버 없이 **정적 HTML 하나만 열어** 확인할 수 있는 POC 페이지 모음입니다.
내부적으로는 PyScript 런타임 위에서 OpenCV·NumPy·Pillow를 돌려 계측 로직을 실행합니다.

## 개요

역할이 다른 두 개의 데모 페이지가 있습니다.

1. `xray_inspection_demo.html`
   - X-ray 이미지를 받아 Plate / Mesh / 3핀 / JIG / V홀까지 한 번에 계측
   - 단일 이미지 모드 + 디렉토리 배치 모드
   - 검사 오버레이, V홀 crop, 계측값 테이블, CSV 저장까지 포함

2. `vhole_template_builder.html`
   - 대표 이미지 1장에서 V홀 상단/하단 위치를 직접 지정해 템플릿을 생성
   - `top` / `bottom` PNG + `vhole_angles.json` 저장

## 주요 특징

- 서버 없이 브라우저 단독 실행
- `PyScript + OpenCV + NumPy + Pillow` 조합으로 이미지 처리
- 단일 이미지 / 폴더 배치 실행 모두 지원
- 템플릿 디렉토리 미선택 시 내장 기본 템플릿으로 동작
- V홀 crop 클릭 보정과 편집 캔버스 제공
- 결과 이미지 ZIP 저장 + CSV 저장 지원
- V홀 템플릿 생성과 각도 메타데이터 저장 지원

## 폴더 구성

```text
poc_sample_page/
├── README.md
├── xray_inspection_demo.html
└── vhole_template_builder.html
```

## 저장소에서 제외한 것

브라우저 POC는 템플릿 디렉토리와 예시 이미지가 있으면 더 풍부하게 동작하지만,
이 저장소에는 아래 항목을 포함하지 않았습니다.

- `templates/` 예시 자산
- 제품별 상단/하단 PNG
- `vhole_angles.json` 예시 메타데이터

즉, 이 폴더는 UI와 로직을 보기 위한 코드 중심 정리본입니다.

## 실행 환경

정적 HTML 파일이지만, 내부적으로 PyScript 런타임과 Python 패키지를 로드하므로 첫 실행 시 네트워크가 필요합니다.

- `https://pyscript.net`에서 PyScript 런타임 로드
- `opencv-python`, `numpy`, `Pillow` 패키지를 브라우저 안에서 로드
- Chromium 계열 브라우저(Chrome / Edge) 권장
  - `showDirectoryPicker` API를 지원하면 결과를 폴더 단위로 저장 가능
  - 미지원 브라우저에서는 ZIP 또는 개별 파일 다운로드로 자동 대체

## 빠른 시작

### 1. 검사 데모 페이지

`xray_inspection_demo.html`을 브라우저에서 엽니다.

권장 사용 순서:

1. 단일 이미지 또는 이미지 디렉토리를 선택
2. 필요 시 외부 템플릿 디렉토리를 추가로 선택
3. `지그 분기`와 `픽셀 환산(mm/px)` 값 설정
4. `현재 이미지 실행` 또는 `디렉토리 전체 실행` 클릭
5. 필요 시 V홀 crop를 클릭해 위치 보정
6. `처리결과 저장` 또는 `CSV 저장`으로 산출물 내보내기

### 2. V홀 템플릿 생성 페이지

`vhole_template_builder.html`을 브라우저에서 엽니다.

권장 사용 순서:

1. 대표 X-ray 이미지 1장 선택
2. 저장할 제품명을 입력
3. 필요 시 `가이드 원 분석 크기` 조정
4. 가이드 오버레이 위에서 상단/하단 위치 지정
5. 미리보기 확인 후 `현재 템플릿 저장`

## 검사 데모 페이지 상세

### 입력 항목

- 단일 이미지
- 이미지 디렉토리
- 디렉토리 내 이미지 선택
- 템플릿 디렉토리
- 지그 분기: `자동 / 있음 / 없음`
- `mm/px` 픽셀 환산값

### 화면 구성

- 원본 이미지 / 검사 오버레이
- V홀 상단 crop / 하단 crop
- 계측값 테이블
- 실행 로그
- 확대 보기 모달
- V홀 위치 수정용 편집 캔버스

### 계측 결과 필드

CSV와 결과 요약에 포함되는 항목:

- `image_name`, `mode`, `template_source`
- `mm_per_px`
- `plate_mesh_x_mm`, `plate_mesh_y_mm`
- `three_pin_left_mm`, `three_pin_right_mm`, `three_pin_bottom_mm`
- `jig_center_x_mm`, `jig_center_y_mm`
- `vhole_angle_deg`
- `three_pin_status`, `jig_center_status`

`vhole_angle_deg`는 상단/하단 V홀 쌍의 축을 `12시-6시`에 맞추기 위한 signed rotation 값입니다.

### 저장 결과

브라우저가 폴더 저장 API를 지원할 때:

```text
inspection_results_YYYYMMDD_HHMMSS/
├── {image_stem}_{job_id}_combined_overlay.png
├── ...
└── measurement_summary.csv
```

지원하지 않을 때:

- `inspection_results_YYYYMMDD_HHMMSS.zip`
- `measurement_summary_YYYYMMDD_HHMMSS.csv`

로 각각 다운로드됩니다.

## V홀 템플릿 생성 페이지 상세

### 입력 항목

- 단일 이미지
- 저장 제품명
- 가이드 원 분석 크기

### 화면 구성

- 원본 이미지 / 가이드 오버레이
- 상단·하단 위치 확인 crop
- 선택 정보 테이블
- 실행 로그
- 확대 보기 모달

### 저장 결과

템플릿 저장 시 제품명 기준 폴더가 생성되며 다음 구조로 저장됩니다.

```text
{product_key}/
├── {shot_key}_top.png
├── {shot_key}_bottom.png
└── vhole_angles.json
```

`vhole_angles.json`에 포함되는 필드:

- `source_key`, `image_name`
- `top_angle_deg`, `bottom_angle_deg`
- `guide_circle`

이 메타데이터는 이후 검사 페이지가 템플릿 각도 제약 조건으로 활용합니다.

## 템플릿 폴더 형식

저장소에는 `templates/` 예시를 포함하지 않지만, 페이지가 저장하거나 외부에서 불러오는 템플릿 디렉토리는 아래 구조를 따릅니다.

```text
{template_name}/
├── {image_id}_top.png
├── {image_id}_bottom.png
└── vhole_angles.json
```

## 구현 방식 요약

두 페이지 모두 단일 HTML 안에 UI와 처리 로직이 함께 들어 있는 구조입니다.

- UI: HTML + CSS + JavaScript
- 연산 브리지: PyScript
- 이미지 처리: Python + OpenCV + NumPy
- 일부 배치 처리: Web Worker 기반 병렬 실행

즉, 이 폴더는 프론트/백엔드가 분리된 서비스 저장소가 아니라, 브라우저 안에서 바로 데모가 돌아가는 **독립 실행형 POC**입니다.

## 사용 시 참고 사항

- 최초 실행 시 PyScript와 Python 패키지 로딩 때문에 시작에 시간이 걸립니다.
- 로컬 파일 선택·폴더 저장 동작은 브라우저별 차이가 있습니다.
- 운영 서버나 배포용 웹 서비스 구조와는 다른 POC 형태입니다.
- 알고리즘 흐름 시연·화면 검증·기능 확인용으로 적합합니다.
