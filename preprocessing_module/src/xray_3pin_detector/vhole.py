#!/usr/bin/env python3
"""V홀 검출 — FFT 잔차 + Polar 템플릿 매칭

Mesh 가장자리에 위치한 V홀(V-notch)을 검출한다.
V홀은 mesh edge에 있는 작은 V자 노치로, 제품 정렬 기준점이다.

전체 파이프라인:
  1. HV 이미지에서 mesh 원 검출 (HoughCircles / contour fallback)
  2. mesh 가장자리 도넛 영역을 polar 좌표로 변환 (1440각도 × 120반경)
  3. 각 행(반경)에 FFT → 상위 K개 주파수만 남겨 주기적 배경(격자 패턴) 복원
  4. 잔차 = |원본 - 복원| → 반복 격자 패턴 제거, V홀 이상치만 남음
  5. 잔차에 z-score 정규화 적용
  6. 대표 이미지 GT 위치에서 추출한 템플릿으로 TM_CCOEFF_NORMED 매칭
  7. NMS로 후보 추출 후, GT angle ±45° 제약으로 최종 2개 검출 (180° 플립 방지)

핵심 설계 결정:
  - FFT 잔차: mesh 격자는 규칙적 주기 패턴 → FFT 상위 주파수에 에너지 집중
    이를 복원 후 빼면 비주기적 이상치(V홀)만 남음
  - GT angle 제약: 템플릿 매칭만으로는 180° 반대편에서도 높은 score가 나올 수 있음
    대표 이미지의 GT 각도 ±45° 범위로 제한하면 플립 오류 방지 (72% → 95% 개선)
  - z-score PNG 인코딩: z-score [-5, +5] → uint8 [0, 255] 고정 범위 매핑으로
    PNG 저장 시 float32 원본과 동일한 매칭 성능 보존 (min-max 스트레칭은 10% 하락)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# 상수: Polar 변환 파라미터
# ---------------------------------------------------------------------------

# 각도 해상도 1440 = 0.25°/샘플 (V홀 폭 약 5~8° → 20~32 샘플로 충분히 표현)
NUM_ANGLE_SAMPLES = 1440

# 반경 샘플 120개 (V홀 높이 약 10~20px → 도넛 전체에서 약 15행 차지)
NUM_RADIUS_SAMPLES = 120

# 기본 도넛 밴드 비율 (Mesh 반지름 기준)
# V홀은 mesh edge에 위치하므로 0.92~1.00 범위
# compute_adaptive_band()로 GT 기반 자동 조정 가능
INNER_RADIUS_RATIO = 0.92
OUTER_RADIUS_RATIO = 1.00


# ---------------------------------------------------------------------------
# 상수: FFT 파라미터
# ---------------------------------------------------------------------------

# 유지할 상위 주파수 개수 (DC 제외)
# mesh 격자 패턴은 보통 2~3개 주파수로 구성됨
# 4개면 격자 복원에 충분하면서 V홀 이상치를 남길 수 있음
TOP_K_FREQUENCIES = 4


# ---------------------------------------------------------------------------
# 상수: 템플릿 매칭 파라미터
# ---------------------------------------------------------------------------

# 템플릿 크기 (polar 좌표 기준)
# HEIGHT 15 = 반경 방향 ≈ 도넛 120행 중 12.5%
# WIDTH 33 = 각도 방향 ≈ 33/1440 × 360° ≈ 8.25° (V홀 폭에 맞춤)
TEMPLATE_HEIGHT = 15
TEMPLATE_WIDTH = 33

# TM_CCOEFF_NORMED 최소 매칭 score (노이즈 후보 제거)
MATCH_THRESHOLD = 0.20

# NMS 최대 후보 수 (충분히 크게 설정하여 놓치는 후보 방지)
MAX_CANDIDATES = 150

# NMS 억제 반경 (같은 V홀에서 중복 검출 방지)
# 행 5 = 반경 방향 약 5행 이내 중복 억제
# 열 15 = 각도 방향 약 15열 ≈ 3.75° 이내 중복 억제
NMS_RADIUS_ROWS = 5
NMS_RADIUS_COLS = 15

# GT angle 제약 허용 오차 (도)
# 대표 이미지의 V홀 각도 대비 ±45° 범위에서만 후보 선택
# 45°이면 상/하 V홀의 90° 영역이 겹치지 않으면서 충분한 탐색 범위 확보
ANGLE_TOLERANCE = 45.0


# ---------------------------------------------------------------------------
# 상수: Z-score PNG 인코딩 (고정 범위 매핑)
# ---------------------------------------------------------------------------

# 크롭 도구가 z-score 잔차를 uint8 PNG로 저장할 때 사용하는 고정 범위
# z-score [-5, +5] → uint8 [0, 255] 선형 매핑
# 10σ 범위는 실측 잔차 분포의 99.9%+ 를 커버
ZSCORE_ENCODE_MIN = -5.0
ZSCORE_ENCODE_MAX = 5.0
ZSCORE_ENCODE_RANGE = ZSCORE_ENCODE_MAX - ZSCORE_ENCODE_MIN  # 10.0


# ---------------------------------------------------------------------------
# 상수: 제품군 정의
# ---------------------------------------------------------------------------

# 이미지 파일명 접두사로 제품군을 분류
# 같은 제품군 내에서는 동일한 대표 이미지 + 템플릿을 공유
PRODUCT_GROUPS = {
    "JIG미조립": ["TYPE_B-1", "TYPE_A-U", "TYPE_A-1-G002"],
    "TYPE_C": ["TYPE_C"],
    "TYPE_A-1": ["TYPE_A-1"],
    "TYPE_A_미조립": ["TYPE_A-G033", "TYPE_A-G034", "TYPE_A-G035",
                     "TYPE_A-G038", "TYPE_A-G039", "TYPE_A-G041",
                     "TYPE_A-G042", "TYPE_A-G043", "TYPE_A-G044",
                     "TYPE_A-G045", "TYPE_A-G046", "TYPE_A-G053",
                     "TYPE_A-G054", "TYPE_A-G055"],
    "TYPE_A_조립": ["TYPE_A-G062", "TYPE_A-G063", "TYPE_A-G064",
                   "TYPE_A-G065", "TYPE_A-G066", "TYPE_A-G067",
                   "TYPE_A-G068", "TYPE_A-G069", "TYPE_A-G070",
                   "TYPE_A-G071", "TYPE_A-G072", "TYPE_A-G076",
                   "TYPE_A-G078", "TYPE_A-G079"],
    "TYPE_D": ["TYPE_D"],
}

# 제품군 → 템플릿 저장 폴더명 매핑
# 크롭 도구(index_pyscript_vhole_template_crop_standalone.html)에서
# 생성한 템플릿이 이 폴더 구조로 저장됨
GROUP_TO_TEMPLATE_DIR = {
    "JIG미조립": "JIG미조립",
    "TYPE_C": "TYPE_C-14",
    "TYPE_A-1": "TYPE_A-1_JIG조립_흑색 배경",
    "TYPE_A_미조립": "TYPE_A_JIG 미조립_흑색 배경",
    "TYPE_A_조립": "TYPE_A_JIG 조립_흑색 배경",
    "TYPE_D": "TYPE_D",
}

# 제품군별 대표 이미지 (이 이미지의 GT 위치에서 템플릿 자동 추출)
GROUP_REF_IMAGE = {
    "JIG미조립": "TYPE_B-1-G001",
    "TYPE_C": "TYPE_C-14-G125",
    "TYPE_A-1": "TYPE_A-1-G008",
    "TYPE_A_미조립": "TYPE_A-G033",
    "TYPE_A_조립": "TYPE_A-G062",
    "TYPE_D": "TYPE_D-G001",
}


# ---------------------------------------------------------------------------
# 결과 타입
# ---------------------------------------------------------------------------

class VholeDetection(NamedTuple):
    """단일 V홀 검출 결과

    Attributes:
        x: 원본 이미지 x 좌표 (px)
        y: 원본 이미지 y 좌표 (px)
        angle_deg: polar 각도 (0~360°, 3시 방향=0°, 반시계)
        score: 템플릿 매칭 score (TM_CCOEFF_NORMED, -1.0~1.0)
    """
    x: float
    y: float
    angle_deg: float
    score: float


class VholeResult(NamedTuple):
    """V홀 검출 전체 결과

    Attributes:
        detections: 검출된 V홀 리스트 (보통 2개: 상/하)
        gt_angles: 대표 이미지의 GT 각도 리스트 (제약 조건에 사용)
        inner_ratio: 사용된 도넛 내경 비율
        outer_ratio: 사용된 도넛 외경 비율
        polar: polar 변환 이미지 (시각화용, shape: 120×1440)
        residual: z-score 잔차 이미지 (시각화용, shape: 120×1440)
    """
    detections: list[VholeDetection]
    gt_angles: list[float]
    inner_ratio: float
    outer_ratio: float
    polar: np.ndarray | None
    residual: np.ndarray | None


# ===========================================================================
# 1. Mesh 원 검출 (HoughCircles + contour fallback)
# ===========================================================================

def get_circle_from_hv(hv_img):
    """
    HV 이미지에서 mesh 외곽 원 검출

    0.25× 축소 후 HoughCircles로 원 검출하여 속도를 확보한다.
    원본 3000px 이미지에서 직접 HoughCircles를 수행하면 20분 이상 소요되지만
    750px로 축소하면 수 ms 이내에 완료된다.

    검출 실패 시 contour 기반 fallback → 이미지 중심 fallback 순으로 시도한다.

    Args:
        hv_img: HV 필터 그레이스케일 이미지 (uint8)

    Returns:
        (cx, cy, radius): 원 중심 좌표와 반지름 (원본 스케일, px)
    """
    h, w = hv_img.shape[:2]

    # 1차: HoughCircles (0.25× 축소 → 속도 최적화)
    # INTER_AREA: 축소 시 안티앨리어싱으로 HoughCircles 정확도 유지
    scale = 0.25
    small = cv2.resize(hv_img, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_AREA)
    blurred = cv2.GaussianBlur(small, (9, 9), 2)
    sh, sw = small.shape[:2]

    # mesh 반지름은 이미지 최소 변의 35~55% 범위
    min_r = int(min(sh, sw) * 0.35)
    max_r = int(min(sh, sw) * 0.55)

    # param2를 점진적으로 낮춰가며 시도 (높을수록 엄격)
    # 50부터 시작하여 검출 실패 시 20까지 완화
    for param2 in [50, 40, 30, 20]:
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1,
            minDist=max(sh, sw),   # 원 1개만 검출
            param1=80, param2=param2,
            minRadius=min_r, maxRadius=max_r)
        if circles is not None:
            c = circles[0][0]
            # 축소 좌표를 원본 스케일로 복원
            return float(c[0] / scale), float(c[1] / scale), float(c[2] / scale)

    # 2차 fallback: 이진화 → 최대 contour → 최소 외접원
    # 흑색 배경 이미지에서 threshold=10으로 밝은 영역 추출
    # 주의: 백색 배경(TYPE_C 등)에서는 부정확할 수 있음
    _, binary = cv2.threshold(hv_img, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        (cx, cy), radius = cv2.minEnclosingCircle(largest)
        return cx, cy, radius

    # 최종 fallback: 이미지 중심
    return w / 2, h / 2, min(w, h) / 2


# ===========================================================================
# 2. Polar 변환 + FFT 잔차 생성
# ===========================================================================

def build_polar_annulus(hv_img, cx, cy, radius, inner_ratio, outer_ratio):
    """
    mesh 가장자리 도넛 영역을 polar 좌표(직교 → 극좌표)로 변환

    도넛 영역을 직사각형 (NUM_RADIUS_SAMPLES × NUM_ANGLE_SAMPLES) 이미지로
    펼친다. 행 = 반지름 방향, 열 = 각도 방향.

    cv2.remap()으로 서브픽셀 보간하여 변환 정밀도를 확보한다.

        원본 좌표:  x = cx + r·cos(θ),  y = cy + r·sin(θ)
        polar 좌표: row = r 인덱스,      col = θ 인덱스

    Args:
        hv_img: HV 필터 그레이스케일 이미지 (uint8)
        cx, cy: mesh 원 중심 좌표 (px)
        radius: mesh 원 반지름 (px)
        inner_ratio: 도넛 내경 비율 (예: 0.95 = 반지름의 95% 지점부터)
        outer_ratio: 도넛 외경 비율 (예: 1.00 = 반지름 경계까지)

    Returns:
        polar: polar 변환 이미지 (uint8, shape: R×A)
        angle_values_deg: 각도 배열 (0~360°, 길이 A)
        radius_values: 반지름 배열 (px 단위, 길이 R)
    """
    # 0~360°를 1440등분 (0.25° 간격)
    angle_values_deg = np.linspace(0.0, 360.0, NUM_ANGLE_SAMPLES, endpoint=False)

    # 도넛 내경~외경을 120등분
    radius_values = np.linspace(
        radius * inner_ratio,
        radius * outer_ratio,
        NUM_RADIUS_SAMPLES,
    )

    # remap 좌표 맵 생성: (R, A) 크기의 x, y 좌표 배열
    # theta: (1, A) — 브로드캐스트로 (R, A)에 적용
    # rr:    (R, 1) — 브로드캐스트로 (R, A)에 적용
    theta = np.deg2rad(angle_values_deg)[None, :]  # (1, A)
    rr = radius_values[:, None]                     # (R, 1)
    map_x = (cx + rr * np.cos(theta)).astype(np.float32)
    map_y = (cy + rr * np.sin(theta)).astype(np.float32)

    # BORDER_REFLECT: 이미지 경계 바깥은 반사하여 아티팩트 방지
    polar = cv2.remap(hv_img, map_x, map_y,
                      interpolation=cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_REFLECT)
    return polar, angle_values_deg, radius_values


def reconstruct_periodic_rows(polar, top_k=TOP_K_FREQUENCIES):
    """
    각 행(반경)별 FFT로 주기적 배경(격자 패턴) 복원 (벡터화)

    mesh 격자는 각도 방향으로 규칙적인 주기 패턴을 형성한다.
    각 행에 FFT를 적용하면 격자 주파수에 에너지가 집중되므로,
    DC + 상위 K개 주파수만 남기고 역변환하면 격자만 복원된다.

    벡터화 구현:
      - np.fft.rfft()로 전체 행을 한번에 FFT (행별 Python 루프 대비 10배 빠름)
      - np.argpartition()으로 O(n) 복잡도로 상위 K개 선택

    Args:
        polar: polar 변환 이미지 (uint8 또는 float32, shape: R×A)
        top_k: 유지할 상위 주파수 개수 (DC 제외)

    Returns:
        recon: 복원된 주기 배경 이미지 (float32, shape: R×A)
    """
    polar_f = polar.astype(np.float32)

    # 각 행의 평균을 빼서 DC 성분 분리 (DC는 항상 유지)
    means = polar_f.mean(axis=1, keepdims=True)
    centered = polar_f - means

    # 전체 행 일괄 FFT (실수 입력 → rfft: 양의 주파수만 반환)
    fft_all = np.fft.rfft(centered, axis=1)       # shape: (R, F)
    magnitudes = np.abs(fft_all)

    # DC(0번 주파수)는 이미 빼놓았으므로 선택 대상에서 제외
    magnitudes[:, 0] = -1

    # 각 행별 상위 K개 주파수 인덱스 선택 (DC 제외 = 1번 이후)
    # argpartition: O(n) — argsort의 O(n log n) 대비 효율적
    top_indices = np.argpartition(-magnitudes[:, 1:], top_k, axis=1)[:, :top_k]

    # 마스크 생성: DC + 상위 K개만 True
    mask = np.zeros_like(fft_all, dtype=bool)
    mask[:, 0] = True  # DC 항 유지
    rows = np.arange(fft_all.shape[0])[:, None]
    mask[rows, top_indices + 1] = True  # +1: DC 제외 인덱스 보정

    # 선택되지 않은 주파수를 0으로 → 역변환 = 주기 배경
    filtered = np.where(mask, fft_all, 0)
    recon = np.fft.irfft(filtered, n=polar_f.shape[1], axis=1) + means
    return recon


def build_residual(hv_img, cx, cy, radius, inner_ratio, outer_ratio):
    """
    HV 이미지에서 z-score 정규화 잔차 생성

    파이프라인: HV → polar 변환 → FFT 주기 배경 복원 → 잔차 → z-score

    잔차 = |원본 - 복원|이므로 격자 패턴이 제거되고
    V홀 같은 비주기적 이상치만 높은 값을 갖는다.

    z-score 정규화: (잔차 - 평균) / 표준편차
    → 이미지 간 밝기/대비 차이를 정규화하여 템플릿 매칭 안정성 확보

    Args:
        hv_img: HV 필터 그레이스케일 이미지 (uint8)
        cx, cy: mesh 원 중심 좌표 (px)
        radius: mesh 원 반지름 (px)
        inner_ratio: 도넛 내경 비율
        outer_ratio: 도넛 외경 비율

    Returns:
        polar: polar 변환 이미지 (uint8, R×A)
        residual: z-score 잔차 (float32, R×A)
        angle_deg: 각도 배열 (0~360°, 길이 A)
        radius_vals: 반지름 배열 (px 단위, 길이 R)
    """
    polar, angle_deg, radius_vals = build_polar_annulus(
        hv_img, cx, cy, radius, inner_ratio, outer_ratio)
    periodic = reconstruct_periodic_rows(polar)

    # 잔차: |원본 - 주기 배경|
    residual = np.abs(polar.astype(np.float32) - periodic)

    # z-score 정규화 (ε=1e-6으로 0 나눗셈 방지)
    mean_r = float(residual.mean())
    std_r = float(residual.std()) + 1e-6
    residual = (residual - mean_r) / std_r

    return polar, residual, angle_deg, radius_vals


# ===========================================================================
# 3. 템플릿 추출 + 매칭
# ===========================================================================

def pad_wrap_columns(image, pad_cols):
    """
    polar 이미지의 0°/360° 경계를 wrap-around 패딩

    polar 이미지에서 0°와 360°는 물리적으로 같은 위치이므로
    양 끝을 서로의 데이터로 패딩한다.
    이렇게 하면 0° 근처의 V홀도 템플릿 매칭이 가능하다.

    Args:
        image: polar 이미지 (R × A)
        pad_cols: 좌우 패딩할 열 수

    Returns:
        padded: 패딩된 이미지 (R × (A + 2·pad_cols))
    """
    return np.hstack([image[:, -pad_cols:], image, image[:, :pad_cols]])


def angle_distance_deg(a, b):
    """
    두 각도 사이의 최소 거리 (0~180° 범위)

    360° 순환을 고려하여 실제 차이를 계산한다.
    예: angle_distance_deg(350, 10) = 20  (350과 370 사이)

    Args:
        a: 첫번째 각도 (도)
        b: 두번째 각도 (도)

    Returns:
        float: 최소 각도 차이 (0~180°)
    """
    return abs(((a - b + 180.0) % 360.0) - 180.0)


def extract_polar_template(wrapped_residual, row_idx, angle_idx, wrap_cols):
    """
    잔차 이미지에서 템플릿 패치(TEMPLATE_HEIGHT × TEMPLATE_WIDTH) 추출

    지정된 (row_idx, angle_idx) 중심으로 패치를 잘라내고
    z-score 정규화한다.

    z-score 정규화된 패치를 사용해야 TM_CCOEFF_NORMED 매칭에서
    일관된 score를 얻을 수 있다.

    Args:
        wrapped_residual: wrap-around 패딩된 잔차 (float32)
        row_idx: 반경 방향 중심 행 인덱스
        angle_idx: 각도 방향 중심 열 인덱스 (패딩 전 기준)
        wrap_cols: 패딩에 사용된 열 수

    Returns:
        patch: z-score 정규화된 패치 (float32, H×W) 또는 None (경계 초과)
    """
    # 패딩 보정: 원본 angle_idx → 패딩된 이미지의 열 인덱스
    center_col = angle_idx + wrap_cols

    r1 = row_idx - TEMPLATE_HEIGHT // 2
    r2 = row_idx + TEMPLATE_HEIGHT // 2 + 1
    c1 = center_col - TEMPLATE_WIDTH // 2
    c2 = center_col + TEMPLATE_WIDTH // 2 + 1

    # 반경 방향 경계 초과 체크 (각도 방향은 wrap으로 해결됨)
    if r1 < 0 or r2 > wrapped_residual.shape[0]:
        return None

    patch = wrapped_residual[r1:r2, c1:c2]
    if patch.shape != (TEMPLATE_HEIGHT, TEMPLATE_WIDTH):
        return None

    # 패치 자체도 z-score 정규화 (ε=1e-6)
    patch = patch.astype(np.float32)
    return (patch - float(patch.mean())) / (float(patch.std()) + 1e-6)


def extract_candidates(match_stack):
    """
    템플릿 매칭 score map에서 NMS(Non-Maximum Suppression)로 후보 추출

    여러 템플릿의 score map을 합친 3D 배열에서
    최고 score 위치를 반복적으로 찾고, 주변을 억제한다.
    YOLO의 NMS와 동일한 개념: score 순서대로 선택하고 겹치는 영역 억제.

    Args:
        match_stack: 템플릿별 매칭 score 3D 배열 (H' × W' × T)
                     H' = R - TEMPLATE_HEIGHT + 1
                     W' = (A + 2·wrap) - TEMPLATE_WIDTH + 1
                     T = 템플릿 수

    Returns:
        candidates: 후보 리스트 [{score, row, angle_idx, tmpl_idx}, ...]
    """
    # 각 위치에서 가장 높은 score의 템플릿 선택
    score_map = np.max(match_stack, axis=2)        # (H', W')
    tmpl_idx_map = np.argmax(match_stack, axis=2)  # (H', W')
    work = score_map.copy()
    candidates = []

    for _ in range(MAX_CANDIDATES):
        flat_idx = int(np.argmax(work))
        val = float(work.flat[flat_idx])
        if val < MATCH_THRESHOLD:
            break

        row, col = np.unravel_index(flat_idx, work.shape)

        # score map 좌표 → 원본 polar 좌표 변환
        # matchTemplate 결과에서 패딩/템플릿 오프셋 보정
        angle_idx = col + TEMPLATE_WIDTH // 2 - TEMPLATE_WIDTH
        if 0 <= angle_idx < NUM_ANGLE_SAMPLES:
            candidates.append({
                "score": val,
                "row": int(row + TEMPLATE_HEIGHT // 2),  # 중심 행
                "angle_idx": int(angle_idx),
                "tmpl_idx": int(tmpl_idx_map[row, col]),
            })

        # NMS 억제: 현재 위치 주변의 score를 -1로 설정
        r1 = max(0, row - NMS_RADIUS_ROWS)
        r2 = min(work.shape[0], row + NMS_RADIUS_ROWS + 1)
        c1 = max(0, col - NMS_RADIUS_COLS)
        c2 = min(work.shape[1], col + NMS_RADIUS_COLS + 1)
        work[r1:r2, c1:c2] = -1.0

    return candidates


def candidate_to_xy(cand, cx, cy, angle_deg, radius_vals):
    """
    polar 좌표의 후보를 원본 이미지 좌표(x, y)로 역변환

    극좌표 → 직교좌표:
        x = cx + r · cos(θ)
        y = cy + r · sin(θ)

    Args:
        cand: 후보 dict (row, angle_idx 포함)
        cx, cy: mesh 원 중심 (px)
        angle_deg: polar 각도 배열
        radius_vals: polar 반지름 배열

    Returns:
        (x, y, angle, radius): 원본 좌표, 각도, 반지름
    """
    a = float(angle_deg[cand["angle_idx"]])
    r = float(radius_vals[min(cand["row"], len(radius_vals) - 1)])
    x = cx + r * math.cos(math.radians(a))
    y = cy + r * math.sin(math.radians(a))
    return x, y, a, r


# ===========================================================================
# 4. 템플릿 로드 (디렉토리 기반 — 크롭 도구 출력)
# ===========================================================================

def _decode_zscore_png(image):
    """고정 범위 uint8 PNG → float32 z-score 복원.

    크롭 도구가 저장한 PNG는 z-score [-5, +5] → [0, 255] 고정 범위 매핑을
    사용한다. 이 함수는 그 역변환을 수행한다:
        float32_zscore = uint8 / 255.0 × 10.0 - 5.0

    min-max 스트레칭(normalize_u8)과 달리, 고정 범위 매핑은 저장/로드 시
    z-score 패턴이 정확히 보존되어 float32 원본과 동일한 매칭 성능을 얻는다.
    """
    return image.astype(np.float32) / 255.0 * ZSCORE_ENCODE_RANGE + ZSCORE_ENCODE_MIN


def load_template_from_directory(template_dir):
    """
    크롭 도구가 저장한 템플릿 디렉토리에서 템플릿 + 각도 + adaptive band 로드.

    크롭 도구는 z-score 잔차 패치를 고정 범위 매핑 [-5, +5] → [0, 255]으로
    uint8 PNG에 저장한다. 로드 시 _decode_zscore_png()로 float32 z-score를
    복원하므로 float32 원본과 동일한 매칭 성능을 얻는다.

    vhole_angles.json에 adaptive_band가 있으면 함께 반환한다.
    adaptive_band는 대표 이미지의 V홀 좌표에서 계산된 최적 밴드 비율로,
    compute_adaptive_band()와 동일한 로직 (mean_ratio ± 0.03).

    디렉토리 구조 (크롭 도구 출력):
        TYPE_A_조립/
        ├── G062_top.png          # z-score 패치 (고정 범위 uint8 인코딩)
        ├── G062_bottom.png       # z-score 패치 (고정 범위 uint8 인코딩)
        └── vhole_angles.json     # 각도 + adaptive band 메타데이터

    Args:
        template_dir: 템플릿 디렉토리 경로

    Returns:
        templates: z-score 정규화된 float32 패치 리스트
        gt_angles: [top_angle_deg, bottom_angle_deg]
        adaptive_band: (inner_ratio, outer_ratio) or None
    """
    template_dir = Path(template_dir)
    gt_angles = []
    templates = []
    adaptive_band = None

    # 각도 + adaptive band 메타데이터 로드
    angles_json = template_dir / "vhole_angles.json"
    if angles_json.exists():
        with open(angles_json) as f:
            data = json.load(f)
        top_angle = data.get("top_angle_deg")
        bottom_angle = data.get("bottom_angle_deg")
        if top_angle is not None:
            gt_angles.append(float(top_angle))
        if bottom_angle is not None:
            gt_angles.append(float(bottom_angle))
        # adaptive band (크롭 도구가 V홀 좌표에서 계산한 최적 밴드)
        band = data.get("adaptive_band")
        if band and "inner_ratio" in band and "outer_ratio" in band:
            adaptive_band = (float(band["inner_ratio"]), float(band["outer_ratio"]))

    # PNG 로드: 고정 범위 uint8 → float32 z-score 역변환
    for png_path in sorted(template_dir.glob("*.png")):
        img = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if img.shape != (TEMPLATE_HEIGHT, TEMPLATE_WIDTH):
            img = cv2.resize(img, (TEMPLATE_WIDTH, TEMPLATE_HEIGHT),
                             interpolation=cv2.INTER_AREA)
        # 고정 범위 디코딩으로 z-score 복원 (min-max 재정규화 아님)
        patch = _decode_zscore_png(img)
        templates.append(patch)

    return templates, gt_angles, adaptive_band


# ===========================================================================
# 5. GT 기반 템플릿 자동 추출 (대표 이미지에서)
# ===========================================================================

def load_gt_coords(json_path):
    """
    LabelMe JSON에서 V홀 꼭지점(tip) 좌표 추출

    V홀은 V자형 노치로, LabelMe에서 label="v"인 다각형으로 표시된다.
    V자의 꼭지점(tip)은 이미지 상반부 V홀은 가장 아래 점,
    하반부 V홀은 가장 위 점이다.

    Args:
        json_path: LabelMe JSON 파일 경로

    Returns:
        coords: V홀 tip 좌표 리스트 [(x, y), ...]
    """
    with open(json_path) as f:
        data = json.load(f)
    img_h = data.get("imageHeight", 0)
    coords = []
    for shape in data["shapes"]:
        if shape["label"] != "v":
            continue
        pts = shape["points"]
        # V자 노치의 꼭지점 = 중심 방향으로 가장 가까운 점
        mean_y = sum(p[1] for p in pts) / len(pts)
        if mean_y < img_h / 2:
            # 상반부 V홀: 가장 아래 점이 tip (중심을 향함)
            tip = max(pts, key=lambda p: p[1])
        else:
            # 하반부 V홀: 가장 위 점이 tip (중심을 향함)
            tip = min(pts, key=lambda p: p[1])
        coords.append((tip[0], tip[1]))
    return coords


def extract_templates_from_reference(
    hv_img, cx, cy, radius, gt_coords, inner_ratio, outer_ratio,
):
    """
    대표 이미지의 GT 위치에서 z-score 잔차 기반 템플릿 추출

    대표 이미지(제품군당 1장)의 V홀 GT 좌표를 polar 좌표로 변환하여
    잔차 이미지에서 직접 패치를 추출한다.

    이렇게 추출한 float32 z-score 템플릿은 같은 파이프라인의 잔차와
    동일한 분포를 가지므로 TM_CCOEFF_NORMED 매칭 성능이 높다 (95%).
    (반면 uint8 PNG 템플릿은 양자화 손실로 10% 수준)

    Args:
        hv_img: 대표 이미지 HV (uint8)
        cx, cy: mesh 원 중심 (px)
        radius: mesh 원 반지름 (px)
        gt_coords: GT V홀 tip 좌표 리스트 [(x, y), ...]
        inner_ratio: 도넛 내경 비율
        outer_ratio: 도넛 외경 비율

    Returns:
        templates: z-score float32 패치 리스트
        gt_angles: 각 V홀의 polar 각도 (도) 리스트
    """
    polar, residual, angle_deg, radius_vals = build_residual(
        hv_img, cx, cy, radius, inner_ratio, outer_ratio)
    wrapped = pad_wrap_columns(residual, TEMPLATE_WIDTH)

    templates = []
    gt_angles = []
    for gx, gy in gt_coords:
        # GT 좌표를 polar 각도/반지름으로 변환
        ga = math.degrees(math.atan2(gy - cy, gx - cx)) % 360
        gr = math.hypot(gx - cx, gy - cy)
        gt_angles.append(ga)

        # polar 좌표에서 가장 가까운 인덱스 찾기
        angle_idx = int(np.argmin(
            [angle_distance_deg(a, ga) for a in angle_deg]))
        row_idx = int(np.argmin(np.abs(radius_vals - gr)))

        # 잔차에서 패치 추출
        patch = extract_polar_template(wrapped, row_idx, angle_idx,
                                       TEMPLATE_WIDTH)
        if patch is not None:
            templates.append(patch)

    return templates, gt_angles


def compute_adaptive_band(gt_coords, cx, cy, radius):
    """
    GT V홀 좌표로부터 도넛 밴드(inner/outer ratio)를 자동 계산

    V홀의 실제 반지름 비율(= mesh 반지름 대비 위치)의 평균에서
    ±0.03 범위를 도넛 밴드로 설정한다.

    예: GT 비율이 0.976이면 → 밴드: 0.946 ~ 1.006

    제품별로 V홀 위치가 약간 다르므로 (예: 0.97 vs 0.98)
    고정 밴드 대신 GT 기반 적응형 밴드를 사용하면 정확도가 향상된다.

    Args:
        gt_coords: GT V홀 tip 좌표 리스트 [(x, y), ...]
        cx, cy: mesh 원 중심 (px)
        radius: mesh 원 반지름 (px)

    Returns:
        (inner_ratio, outer_ratio): 도넛 밴드 비율
    """
    if not gt_coords:
        return INNER_RADIUS_RATIO, OUTER_RADIUS_RATIO

    # 각 V홀의 반지름 비율 계산
    gt_ratios = [math.hypot(gx - cx, gy - cy) / radius
                 for gx, gy in gt_coords]
    mean_ratio = float(np.mean(gt_ratios))

    # ±0.03 마진 (V홀 크기 + 제품 간 편차 고려)
    inner = max(0.85, mean_ratio - 0.03)
    outer = min(1.02, mean_ratio + 0.03)
    return inner, outer


# ===========================================================================
# 6. 제품군 분류
# ===========================================================================

def classify_group(name):
    """
    이미지 파일명으로 제품군 분류

    파일명 접두사를 PRODUCT_GROUPS 사전의 키와 매칭하여
    제품군을 결정한다. 같은 제품군은 동일한 대표 이미지와 템플릿을 공유.

    매칭 순서: 완전 일치(==) → 접두사 일치(startswith)
    매칭되지 않으면 "unknown" 반환.

    Args:
        name: 이미지 파일명 (확장자 제외, 예: "TYPE_A-G062")

    Returns:
        str: 제품군명 (예: "TYPE_A_조립") 또는 "unknown"
    """
    for group, prefixes in PRODUCT_GROUPS.items():
        for prefix in prefixes:
            if name == prefix or name.startswith(prefix):
                return group
    return "unknown"


# ===========================================================================
# 7. V홀 검출 메인 함수
# ===========================================================================

def detect_vholes(
    hv_img,
    templates,
    gt_angles,
    cx=None,
    cy=None,
    radius=None,
    inner_ratio=INNER_RADIUS_RATIO,
    outer_ratio=OUTER_RADIUS_RATIO,
):
    """
    V홀 검출 메인 함수

    전체 파이프라인을 실행하여 HV 이미지에서 V홀 2개를 검출한다.

    파이프라인:
      1. mesh 원 검출 (cx, cy, radius가 None이면)
      2. polar 변환 → FFT 잔차 → z-score 정규화
      3. wrap-around 패딩 → 템플릿 매칭 (TM_CCOEFF_NORMED)
      4. NMS로 후보 추출 (최대 150개)
      5. GT angle ±45° 제약으로 최종 선택 (V홀당 1개)

    GT angle 제약 핵심:
      템플릿 매칭만으로는 V홀 위치와 180° 반대편에서도 높은 score가 나올 수 있다.
      (mesh 격자 패턴이 반주기 대칭성을 가지기 때문)
      대표 이미지의 GT 각도 ±45° 범위로 탐색을 제한하면
      180° 플립 오류를 원천 차단한다.

      예: GT 각도 = [270°, 90°]이면
          → 상단 V홀은 225~315° 범위에서만 선택
          → 하단 V홀은 45~135° 범위에서만 선택

    Args:
        hv_img: HV 필터 그레이스케일 이미지 (uint8)
        templates: z-score float32 템플릿 패치 리스트 (보통 2개: 상/하)
        gt_angles: GT 각도 리스트 (보통 [~270°, ~90°])
        cx, cy, radius: mesh 원 중심/반지름 (None이면 자동 검출)
        inner_ratio, outer_ratio: 도넛 밴드 비율

    Returns:
        VholeResult: 검출 결과 (detections, polar, residual 등)
    """
    if not templates:
        return VholeResult([], gt_angles, inner_ratio, outer_ratio, None, None)

    # 1. 원 검출 (제공되지 않은 경우)
    if cx is None or cy is None or radius is None:
        cx, cy, radius = get_circle_from_hv(hv_img)

    # 2. polar 변환 → FFT 잔차
    polar, residual, angle_deg, radius_vals = build_residual(
        hv_img, cx, cy, radius, inner_ratio, outer_ratio)

    # 3. wrap-around 패딩 + 멀티 템플릿 매칭
    wrapped = pad_wrap_columns(residual, TEMPLATE_WIDTH)
    match_stack = np.stack([
        cv2.matchTemplate(wrapped.astype(np.float32),
                          t.astype(np.float32),
                          cv2.TM_CCOEFF_NORMED)
        for t in templates], axis=2)

    # 4. NMS로 후보 추출
    candidates = extract_candidates(match_stack)

    # 5. GT angle ±45° 제약으로 최종 선택
    #    각 GT 각도마다 가장 높은 score의 후보 1개씩 선택
    detections = []
    if candidates and gt_angles:
        # 모든 후보를 원본 좌표로 역변환 (한번만 계산)
        cand_xy = []
        for c in candidates:
            x, y, a, rv = candidate_to_xy(c, cx, cy, angle_deg, radius_vals)
            cand_xy.append((c, x, y, a, rv))

        for ga in gt_angles:
            # GT 각도 ±45° 범위의 후보만 필터링
            near = [(c, x, y, a, rv) for c, x, y, a, rv in cand_xy
                    if angle_distance_deg(a, ga) <= ANGLE_TOLERANCE]
            if near:
                # 범위 내 최고 score 후보 선택
                best = max(near, key=lambda t: t[0]["score"])
                c, x, y, a, rv = best
                detections.append(VholeDetection(
                    x=x, y=y, angle_deg=a, score=c["score"]))

    return VholeResult(
        detections=detections,
        gt_angles=gt_angles,
        inner_ratio=inner_ratio,
        outer_ratio=outer_ratio,
        polar=polar,
        residual=residual,
    )


# ===========================================================================
# 8. 시각화
# ===========================================================================

def draw_vhole_overlay(hv_img, result, gt_coords=None):
    """
    HV 이미지에 V홀 검출 결과 오버레이

    오버레이 요소:
      - 빨간 원(●): 검출된 V홀 위치 + score 텍스트
      - 노란 별(★): GT 위치 (gt_coords가 제공된 경우)
      - 노란 선(─): GT↔검출 최근접 매칭 + 오차(px) 텍스트

    Args:
        hv_img: HV 필터 그레이스케일 이미지 (uint8)
        result: VholeResult (detect_vholes의 반환값)
        gt_coords: GT V홀 tip 좌표 리스트 (평가용, 없으면 None)

    Returns:
        vis: BGR 오버레이 이미지 (uint8)
    """
    vis = cv2.cvtColor(hv_img, cv2.COLOR_GRAY2BGR)

    # GT 위치 표시 (노란 별)
    if gt_coords:
        for gx, gy in gt_coords:
            cv2.drawMarker(vis, (int(gx), int(gy)), (0, 255, 255),
                           cv2.MARKER_STAR, 18, 2)

    # 검출 결과 표시 (빨간 원 + score)
    for det in result.detections:
        cv2.circle(vis, (int(det.x), int(det.y)), 8,
                   (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(vis, (int(det.x), int(det.y)), 10,
                   (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, f"{det.score:.3f}",
                    (int(det.x) + 12, int(det.y) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

    # GT↔검출 매칭선 (최근접 페어 + 오차 표시)
    if gt_coords and result.detections:
        for gx, gy in gt_coords:
            dists = [math.hypot(gx - d.x, gy - d.y)
                     for d in result.detections]
            bi = int(np.argmin(dists))
            bp = result.detections[bi]
            # GT → 검출 연결선 (노란색)
            cv2.line(vis, (int(gx), int(gy)),
                     (int(bp.x), int(bp.y)), (0, 255, 255), 1)
            # 중간 지점에 오차 텍스트
            mx = (int(gx) + int(bp.x)) // 2
            my = (int(gy) + int(bp.y)) // 2
            cv2.putText(vis, f"{dists[bi]:.0f}px", (mx + 5, my),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    return vis


def draw_residual_colormap(residual):
    """
    z-score 잔차를 TURBO 컬러맵으로 시각화

    높은 잔차(= V홀 등 이상치)가 빨간색,
    낮은 잔차(= 격자 패턴 제거 후 배경)가 파란색으로 표시된다.

    Args:
        residual: z-score 잔차 (float32, R×A)

    Returns:
        colormap: TURBO 컬러맵 적용된 BGR 이미지 (uint8)
    """
    resid_u8 = cv2.normalize(np.abs(residual), None, 0, 255,
                             cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.applyColorMap(resid_u8, cv2.COLORMAP_TURBO)


def draw_polar_image(polar):
    """
    polar 변환 이미지를 8비트 그레이스케일로 정규화

    시각화용으로 원본 polar 이미지의 값 범위를 0~255로 스케일링한다.

    Args:
        polar: polar 변환 이미지 (uint8 또는 float32, R×A)

    Returns:
        normalized: 0~255 정규화된 이미지 (uint8)
    """
    return cv2.normalize(polar, None, 0, 255,
                         cv2.NORM_MINMAX).astype(np.uint8)
