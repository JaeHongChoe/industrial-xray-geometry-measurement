#!/usr/bin/env python3
"""X-ray 이미지 전처리 파이프라인

그레이스케일 변환, 배경 정규화, 음영 보정, 밝기/대비 정규화,
Morphology 디테일 강조, HV 필터 적용까지의 전처리 단계를 수행한다.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# 1. 이미지 로드 및 그레이스케일 변환
# ---------------------------------------------------------------------------

def load_grayscale(image_path):
    """
    이미지를 8비트 그레이스케일로 로드

    알파 채널이 있는 경우 흰색 배경 위에 합성 후 변환한다.

    Args:
        image_path: 이미지 파일 경로

    Returns:
        gray: 그레이스케일 이미지 (uint8)
    """
    # PNG 등 알파 채널 포함 이미지 → 흰 배경 합성 후 변환
    # (투명 영역이 검정으로 변환되는 것을 방지)
    rgba = Image.open(image_path).convert("RGBA")
    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    composited = Image.alpha_composite(background, rgba)
    return np.array(composited.convert("L"))


def _border_intensity_stats(gray):
    """
    이미지 외곽 테두리 픽셀의 통계값 계산

    Args:
        gray: 그레이스케일 이미지

    Returns:
        (mean, p90): 테두리 픽셀의 평균, 90 퍼센타일
    """
    height, width = gray.shape
    # 이미지 외곽 4% 폭의 테두리 픽셀을 샘플링 (배경 영역)
    border_width = max(12, int(min(height, width) * 0.04))
    border_pixels = np.concatenate([
        gray[:border_width, :].ravel(),       # 상단
        gray[-border_width:, :].ravel(),      # 하단
        gray[:, :border_width].ravel(),       # 좌측
        gray[:, -border_width:].ravel(),      # 우측
    ]).astype(np.float32)
    return float(np.mean(border_pixels)), float(np.percentile(border_pixels, 90))


def is_white_background(gray):
    """
    배경이 흰색인지 판별

    테두리 픽셀의 평균/p90으로 판단한다.

    Args:
        gray: 그레이스케일 이미지

    Returns:
        bool: 흰색 배경이면 True
    """
    border_mean, border_p90 = _border_intensity_stats(gray)
    return border_mean > 185.0 or border_p90 > 215.0


def _normalize_to_white_background(gray):
    """
    배경이 제품보다 어두운 경우에만 반전하여 흰색 배경으로 통일

    Args:
        gray: 그레이스케일 이미지

    Returns:
        gray: 흰색 배경으로 정규화된 이미지
    """
    if is_white_background(gray):
        return gray
    border_mean, border_p90 = _border_intensity_stats(gray)
    image_median = float(np.median(gray))
    image_p75 = float(np.percentile(gray, 75))

    # TYPE_A PNG처럼 배경이 흰색은 아니어도 제품보다 충분히 밝은 회색이면
    # 이미 light-background 이미지다. 여기서 반전하면 후속 배경 스케일링에서
    # 전체가 255로 클리핑되어 디테일/HV가 사라진다.
    if border_mean >= image_median + 12.0 or border_p90 >= image_p75 + 8.0:
        return gray
    return cv2.bitwise_not(gray)


# ---------------------------------------------------------------------------
# 2. 배경값 추정 및 밝기 스케일링
# ---------------------------------------------------------------------------

def estimate_bg_mode(gray):
    """
    테두리 히스토그램의 최빈값으로 배경 밝기 추정

    Args:
        gray: 그레이스케일 이미지

    Returns:
        bg_mode: 배경 픽셀값 (0~255)
    """
    height, width = gray.shape
    bw = max(12, int(min(height, width) * 0.04))
    border = np.concatenate([
        gray[:bw, :].ravel(), gray[-bw:, :].ravel(),
        gray[:, :bw].ravel(), gray[:, -bw:].ravel(),
    ])
    hist, _ = np.histogram(border, bins=256, range=(0, 256))
    return int(np.argmax(hist))


def _scale_to_255(gray, bg_mode=None):
    """
    배경 밝기를 255로 스케일링

    일부 X-ray 장비는 출력 범위가 0~228 등으로 제한되어 있어
    배경값 기준으로 비례 확장한다.

    Args:
        gray: 그레이스케일 이미지
        bg_mode: 배경 밝기값 (None이면 자동 추정)

    Returns:
        gray: 스케일링된 이미지
    """
    if bg_mode is None:
        bg_mode = estimate_bg_mode(gray)
    # 이미 255에 가까우면 스케일링 불필요
    if bg_mode >= 253:
        return gray
    # 배경값을 255로 맞추는 비례 스케일링
    # (예: 배경이 228이면 × 1.118 → 0~255 전체 범위 활용)
    ratio = 255.0 / max(bg_mode, 1)
    scaled = gray.astype(np.float32) * ratio
    return np.clip(scaled, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 3. 음영 보정 (Weighted Gaussian flat-field)
# ---------------------------------------------------------------------------

def correct_shading(gray, ksize=301, bg_mode=None):
    """
    가중 가우시안 플랫필드로 2D 음영 비균일성 제거

    X-ray 점광원에 의한 중앙 밝음 / 가장자리 어두움을 보정한다.
    4배 축소 → 블러 → 확대 방식으로 속도를 최적화한다.

    Args:
        gray: 그레이스케일 이미지
        ksize: 가우시안 커널 크기
        bg_mode: 배경 밝기값 (None이면 자동 추정)

    Returns:
        corrected: 음영 보정된 이미지
    """
    if bg_mode is None:
        bg_mode = estimate_bg_mode(gray)

    # 배경(bg_mode) 보다 약간 어두운 픽셀 = 제품 영역
    threshold = bg_mode - 5
    product_mask = (gray < threshold).astype(np.float32)
    if product_mask.sum() < 100:
        return gray

    # 가중 평균 블러: 제품 영역만 참여하는 illumination field 추정
    # 배경 픽셀은 마스크=0이므로 블러에 기여하지 않음
    gray_f = gray.astype(np.float32)
    weighted = gray_f * product_mask

    # 4× 축소 → 블러 → 복원 (속도 15배 향상, 결과 동일)
    # INTER_AREA: 축소 시 안티앨리어싱으로 모아레 방지
    scale = 4
    h, w = gray.shape
    sh, sw = h // scale, w // scale
    small_weighted = cv2.resize(weighted, (sw, sh), interpolation=cv2.INTER_AREA)
    small_mask = cv2.resize(product_mask, (sw, sh), interpolation=cv2.INTER_AREA)
    small_ksize = max(3, ksize // scale) | 1
    blur_w_small = cv2.GaussianBlur(small_weighted, (small_ksize, small_ksize), 0)
    blur_m_small = cv2.GaussianBlur(small_mask, (small_ksize, small_ksize), 0)
    blur_weighted = cv2.resize(blur_w_small, (w, h), interpolation=cv2.INTER_LINEAR)
    blur_mask = cv2.resize(blur_m_small, (w, h), interpolation=cv2.INTER_LINEAR)

    # 가중 블러 / 마스크 블러 = 제품 영역의 로컬 평균 밝기 (illumination field)
    blur = blur_weighted / np.maximum(blur_mask, 1e-5)

    # flat-field 보정: 원본 / illumination × 전체 평균
    # → 밝기 비균일성 제거, 전체 밝기 수준은 유지
    blur_mean = float(gray_f[gray < threshold].mean())
    corrected = gray_f / np.maximum(blur, 1.0) * blur_mean

    # 배경 픽셀은 보정하지 않고 원본 유지
    bg_mask = gray >= threshold
    result = np.clip(corrected, 0, 255).astype(np.uint8)
    result[bg_mask] = gray[bg_mask]
    return result


# ---------------------------------------------------------------------------
# 4. 퍼센타일 스트레치 (밝기/대비 정규화)
# ---------------------------------------------------------------------------

def _percentile_stretch(gray, target_p5=50.0, target_p95=200.0, bg_mode=None):
    """
    제품 영역의 밝기/대비를 퍼센타일 매핑으로 정규화

    5퍼센타일 → target_p5, 95퍼센타일 → target_p95로 선형 변환한다.
    배경 픽셀은 255로 고정한다.

    Args:
        gray: 그레이스케일 이미지
        target_p5: 5퍼센타일 목표값
        target_p95: 95퍼센타일 목표값
        bg_mode: 배경 밝기값 (None이면 자동 추정)

    Returns:
        gray: 정규화된 이미지
    """
    if bg_mode is None:
        bg_mode = estimate_bg_mode(gray)
    product_mask = gray < (bg_mode - 5)
    if product_mask.sum() < 100:
        return gray

    # 제품 영역의 5%ile / 95%ile 기준으로 선형 매핑
    # → 장비별 밝기/대비 차이를 일정한 범위로 정규화
    product_pixels = gray[product_mask].astype(np.float32)
    p5 = float(np.percentile(product_pixels, 5))
    p95 = float(np.percentile(product_pixels, 95))
    if p95 - p5 < 1:
        return gray

    scale = (target_p95 - target_p5) / (p95 - p5)
    offset = target_p5 - p5 * scale

    result = gray.astype(np.float32) * scale + offset
    result[~product_mask] = 255  # 배경은 흰색 고정
    return np.clip(result, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 5. Morphology 디테일 이미지 생성
# ---------------------------------------------------------------------------

def _build_mesh_morph_image(gray, roi_mask=None, kernel_size=31):
    """
    형태학적 연산으로 메시 디테일 이미지 생성

    Morphological closing으로 로컬 배경을 추정한 뒤
    원본과의 차이로 메시 격자 등 어두운 미세 구조를 강조한다.

    Args:
        gray: 전처리된 그레이스케일 이미지
        roi_mask: ROI 마스크 (None이면 전체 영역)
        kernel_size: 커널 크기

    Returns:
        morph: 디테일 이미지 (uint8, 0~255 정규화)
    """
    kernel_size = max(3, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Closing = 로컬 배경 추정 (메시 격자보다 큰 커널로 격자를 매꿔버림)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    # 로컬 배경 - 원본 = 어두운 미세 구조 (메시 격자, 핀 등) 강조
    mesh_detail = cv2.subtract(background, gray)

    if roi_mask is not None:
        mesh_detail = mesh_detail.copy()
        mesh_detail[roi_mask == 0] = 0

    if not np.any(mesh_detail):
        return np.zeros_like(gray, dtype=np.uint8)
    return cv2.normalize(mesh_detail, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


# ---------------------------------------------------------------------------
# 6. Plate ROI 마스크 (제품 영역 검출)
# ---------------------------------------------------------------------------

def _build_plate_roi_mask(gray):
    """
    Plate(제품) 영역의 이진 마스크 생성

    여러 임계값으로 이진화 후 가장 큰 contour를 선택한다.

    Args:
        gray: 그레이스케일 이미지

    Returns:
        mask: 이진 마스크 (255=제품, 0=배경)
    """
    height, width = gray.shape
    total_area = height * width
    best_mask = None
    best_area = 0.0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))

    if is_white_background(gray):
        threshold_jobs = [
            (cv2.THRESH_BINARY_INV, 180),
            (cv2.THRESH_BINARY_INV, 190),
            (cv2.THRESH_BINARY_INV, 200),
            (cv2.THRESH_BINARY_INV, 210),
            (cv2.THRESH_BINARY_INV, 220),
            (cv2.THRESH_BINARY_INV, 230),
        ]
    else:
        threshold_jobs = [
            (cv2.THRESH_BINARY, 20),
            (cv2.THRESH_BINARY, 40),
            (cv2.THRESH_BINARY + cv2.THRESH_OTSU, 0),
        ]

    # 여러 임계값을 순차 시도하여 가장 큰 제품 영역을 찾음
    # (X-ray 장비마다 배경/제품 경계 밝기가 다를 수 있으므로)
    for threshold_type, threshold_value in threshold_jobs:
        if threshold_type & cv2.THRESH_OTSU:
            _, binary = cv2.threshold(gray, 0, 255, threshold_type)
        else:
            _, binary = cv2.threshold(gray, threshold_value, 255, threshold_type)

        # Closing으로 제품 내부 구멍 메우기
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        largest_contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(largest_contour))
        # 면적이 10~95% 범위 밖이면 노이즈 또는 전체 이미지이므로 무시
        if area <= total_area * 0.10 or area >= total_area * 0.95 or area <= best_area:
            continue

        best_area = area
        best_mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(best_mask, [largest_contour], -1, 255, -1)
        # 60% 이상이면 충분히 큰 제품 영역 → 더 이상 시도 불필요
        if area > total_area * 0.60:
            break

    if best_mask is None:
        return np.full(gray.shape, 255, dtype=np.uint8)
    return best_mask


# ---------------------------------------------------------------------------
# 7. HV 필터 (수평/수직 메시 격자 추출)
# ---------------------------------------------------------------------------

def _build_hv_image(morph, hv_kernel_len=25):
    """
    수평/수직 방향 Morphological open 필터 적용

    메시 격자 라인만 남기고 곡선형 열선 패턴을 제거한다.

    Args:
        morph: Morphology 디테일 이미지
        hv_kernel_len: 커널 길이

    Returns:
        hv: HV 필터 적용 이미지 (uint8)
    """
    # 수평/수직 선형 커널로 Open → 해당 방향의 직선 구조만 통과
    # 열선 등 곡선형 패턴은 직선 커널에 맞지 않아 제거됨
    kh = cv2.getStructuringElement(cv2.MORPH_RECT, (hv_kernel_len, 1))  # 수평
    kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, hv_kernel_len))  # 수직
    # 두 방향 결과의 max → 수평 + 수직 격자 모두 보존
    return np.maximum(
        cv2.morphologyEx(morph, cv2.MORPH_OPEN, kh),
        cv2.morphologyEx(morph, cv2.MORPH_OPEN, kv),
    )


# ---------------------------------------------------------------------------
# 8. 원형 ROI 마스크 헬퍼
# ---------------------------------------------------------------------------

def _build_circle_roi_mask(shape, circle):
    """
    검출된 원으로부터 원형 ROI 마스크 생성

    Args:
        shape: (height, width) 튜플
        circle: Circle 객체

    Returns:
        mask: 원형 이진 마스크 (uint8)
    """
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.circle(
        mask,
        (int(round(circle.center_x_px)), int(round(circle.center_y_px))),
        int(round(circle.radius_px)),
        255, -1,
    )
    return mask


# ---------------------------------------------------------------------------
# 9. 전처리 파이프라인
# ---------------------------------------------------------------------------

def preprocess_image(image_path):
    """
    검출 전 이미지 전처리 파이프라인

    1. 그레이스케일 로드
    2. 흰색 배경 정규화 (흑색이면 반전)
    3. 배경 밝기 255 스케일링
    4. 음영 보정 (가중 가우시안 플랫필드)
    5. 퍼센타일 스트레치 (밝기/대비 정규화)
    6. Morphology 디테일 이미지 생성
    7. HV 필터 적용

    Args:
        image_path: 이미지 파일 경로

    Returns:
        gray: 전처리된 그레이스케일 이미지 (uint8)
        morph: Morphology 디테일 이미지 (uint8)
        hv: HV 필터 이미지 (uint8)
    """
    raw_gray = load_grayscale(image_path)
    gray = _normalize_to_white_background(raw_gray)
    bg_mode = estimate_bg_mode(gray)
    gray = _scale_to_255(gray, bg_mode=bg_mode)
    bg_mode = estimate_bg_mode(gray)
    gray = correct_shading(gray, bg_mode=bg_mode)
    gray = _percentile_stretch(gray, bg_mode=estimate_bg_mode(gray))
    roi_mask = _build_plate_roi_mask(gray)
    morph = _build_mesh_morph_image(gray, roi_mask)
    hv = _build_hv_image(morph)
    return gray, morph, hv
