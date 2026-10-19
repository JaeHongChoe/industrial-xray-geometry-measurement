#!/usr/bin/env python3
"""원 검출 및 편심 측정

Plate 외곽 원, Mesh 외곽 원을 검출하고
두 원의 중심 간 편심(eccentricity)을 측정한다.
"""

from __future__ import annotations

import math

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Circle 클래스
# ---------------------------------------------------------------------------

class Circle:
    """원 표현 (중심 좌표 + 반지름, 단위: px)"""

    def __init__(self, center_x, center_y, radius):
        self.center_x_px = center_x
        self.center_y_px = center_y
        self.radius_px = radius

    def __repr__(self):
        return (
            f"Circle(cx={self.center_x_px:.1f}, cy={self.center_y_px:.1f}, "
            f"r={self.radius_px:.1f})"
        )


def fit_circle_least_squares(points):
    """
    2D 점들에 대한 대수적 최소제곱 원 피팅

    Args:
        points: (N, 2) 좌표 배열

    Returns:
        Circle: 피팅된 원
    """
    # (x - cx)² + (y - cy)² = r² 을 전개하면
    # 2·cx·x + 2·cy·y + (r² - cx² - cy²) = x² + y²
    # → Ax = b 형태의 선형 시스템으로 풀 수 있음
    x = points[:, 0]
    y = points[:, 1]
    a = np.column_stack([2.0 * x, 2.0 * y, np.ones(len(points))])
    b = x * x + y * y
    solution, *_ = np.linalg.lstsq(a, b, rcond=None)
    cx = float(solution[0])
    cy = float(solution[1])
    r = math.sqrt(float(solution[2] + cx * cx + cy * cy))
    return Circle(cx, cy, r)


# ---------------------------------------------------------------------------
# Plate 외곽 원 검출
# ---------------------------------------------------------------------------

def detect_plate_outer_circle(gray):
    """
    Canny edge + fitEllipse로 Plate 외곽 원 검출

    이미지 크기의 35~55% 범위에 해당하는 타원을 찾고
    r * perimeter 점수가 가장 높은 것을 선택한다.

    Args:
        gray: 전처리된 그레이스케일 이미지

    Returns:
        Circle: Plate 외곽 원
    """
    h, w = gray.shape
    # Plate 반지름은 이미지 최소 변의 35~55% 범위에 있다고 가정
    min_r = min(h, w) * 0.35
    max_r = min(h, w) * 0.55

    # Canny edge → contour → fitEllipse
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
    edges = cv2.Canny(blurred, 30, 100)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return Circle(w / 2.0, h / 2.0, min(h, w) / 2.0)

    # 후보 타원 중 r × 둘레길이 점수가 최대인 것을 선택
    # (큰 원 + 긴 외곽선 = 가장 확실한 Plate 경계)
    best_ellipse = None
    best_score = -1.0
    for cnt in contours:
        if len(cnt) < 5:
            continue
        e = cv2.fitEllipse(cnt)
        r = (e[1][0] + e[1][1]) / 4.0  # 장축+단축 평균 / 2 = 평균 반지름
        if r < min_r or r > max_r:
            continue
        peri = cv2.arcLength(cnt, True)
        score = r * peri
        if score > best_score:
            best_score = score
            best_ellipse = e

    if best_ellipse is not None:
        cx, cy = best_ellipse[0]
        r = (best_ellipse[1][0] + best_ellipse[1][1]) / 4.0
        return Circle(float(cx), float(cy), float(r))

    # 폴백: 범위 필터 통과 못 하면 면적 최대 contour 사용
    largest = max(contours, key=cv2.contourArea)
    if len(largest) >= 5:
        e = cv2.fitEllipse(largest)
        cx, cy = e[0]
        r = (e[1][0] + e[1][1]) / 4.0
        return Circle(float(cx), float(cy), float(r))
    return Circle(w / 2.0, h / 2.0, min(h, w) / 2.0)


# ---------------------------------------------------------------------------
# Mesh 외곽 원 검출 (warpPolar 방사형 프로파일)
# ---------------------------------------------------------------------------

def _smooth_1d(profile, sigma=5.0):
    """
    1D 가우시안 스무딩 (scipy 의존성 없음)

    Args:
        profile: 1D 배열
        sigma: 가우시안 시그마

    Returns:
        smoothed: 스무딩된 1D 배열
    """
    radius = int(4 * sigma + 0.5)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    return np.convolve(profile, kernel, mode="same")


def detect_mesh_circle(hv, plate_circle, smooth_sigma=5.0, num_angles=360):
    """
    HV 이미지의 warpPolar 방사형 프로파일로 Mesh 외곽 원 검출

    1. warpPolar로 극좌표 변환
    2. 각도 평균 → 1D 방사형 프로파일 → 최대 음의 기울기 = Mesh 경계
    3. 각도별 경계점 수집 → 최소제곱 원 피팅으로 독립적인 Mesh 중심 추정

    Args:
        hv: HV 필터 이미지
        plate_circle: Plate 원 (중심 기준점)
        smooth_sigma: 프로파일 스무딩 시그마
        num_angles: 극좌표 각도 해상도

    Returns:
        Circle: Mesh 외곽 원
    """
    pcx = plate_circle.center_x_px
    pcy = plate_circle.center_y_px
    plate_r = plate_circle.radius_px
    h, w = hv.shape
    max_r = float(min(plate_r, pcx, pcy, w - pcx, h - pcy) * 0.98)
    radius_res = int(max_r)

    # 극좌표 변환: (x,y) → (θ, r) → 2D 배열 [각도 × 반지름]
    polar = cv2.warpPolar(
        hv,
        dsize=(radius_res, num_angles),
        center=(pcx, pcy),
        maxRadius=max_r,
        flags=cv2.WARP_POLAR_LINEAR + cv2.INTER_LINEAR,
    )

    # 1D 방사형 프로파일: 각도 방향 평균 → 반지름에 따른 밝기 변화
    profile = polar.astype(np.float64).mean(axis=0)
    smoothed = _smooth_1d(profile, sigma=smooth_sigma)
    # 기울기가 가장 급격히 감소하는 지점 = Mesh 외곽 경계
    # (메시 격자 영역 → 빈 갭 영역으로 넘어가는 순간)
    grad = np.diff(smoothed)
    start = max(1, len(grad) // 10)  # 중심부 노이즈 영역 건너뜀
    approx_mesh_r = float(start + np.argmin(grad[start:]))

    mesh_radius = approx_mesh_r * max_r / radius_res

    # 각도별 경계점 수집 (Mesh 원의 중심을 독립적으로 추정하기 위해)
    angles = np.linspace(0, 2.0 * np.pi, num_angles, endpoint=False)
    search_band = max(30, int(approx_mesh_r * 0.05))
    r_lo = max(0, int(approx_mesh_r - search_band))
    r_hi = min(radius_res, int(approx_mesh_r + search_band))

    # 2D 가우시안 블러 → 각도별 최대 음의 기울기 지점 = 경계점
    band = polar[:, r_lo:r_hi].astype(np.float64)
    band_w = band.shape[1]
    boundary_points = []
    if band_w >= 10:
        sigma_px = 3.0
        ksize_1d = int(8 * sigma_px + 1) | 1
        band_smooth = cv2.GaussianBlur(
            band, (ksize_1d, 1), sigmaX=sigma_px, sigmaY=0,
        )
        band_grad = np.diff(band_smooth, axis=1)
        drop_indices = np.argmin(band_grad, axis=1)
        drop_values = band_grad[np.arange(num_angles), drop_indices]

        valid = drop_values < -0.5
        valid_idx = np.where(valid)[0]
        if len(valid_idx) > 0:
            # 극좌표 경계점 → 직교좌표 변환
            r_px_arr = (r_lo + drop_indices[valid_idx].astype(np.float64)) * max_r / radius_res
            cos_a = np.cos(angles[valid_idx])
            sin_a = np.sin(angles[valid_idx])
            bx_arr = pcx + r_px_arr * cos_a
            by_arr = pcy + r_px_arr * sin_a
            boundary_points = list(zip(bx_arr.tolist(), by_arr.tolist()))

    if len(boundary_points) >= 24:
        pts = np.array(boundary_points, dtype=np.float64)

        # 이상치 제거: 중앙값 반지름 ±5% 벗어나는 점 제외
        dists = np.sqrt((pts[:, 0] - pcx) ** 2 + (pts[:, 1] - pcy) ** 2)
        median_r = float(np.median(dists))
        keep = np.abs(dists - median_r) < median_r * 0.05
        pts_clean = pts[keep] if keep.sum() >= 24 else pts

        # 최소제곱 원 피팅으로 Mesh 중심 추정 (Plate 중심과 독립)
        fitted = fit_circle_least_squares(pts_clean)
        return Circle(fitted.center_x_px, fitted.center_y_px, mesh_radius)

    return Circle(pcx, pcy, mesh_radius)


# ---------------------------------------------------------------------------
# 편심 측정 (Plate vs Mesh 중심 오프셋)
# ---------------------------------------------------------------------------

def compute_eccentricity(plate_circle, mesh_circle, pixel_size_mm=None):
    """
    Plate 중심과 Mesh 중심 간 오프셋 계산

    Args:
        plate_circle: Plate 원
        mesh_circle: Mesh 원
        pixel_size_mm: 픽셀당 mm (None이면 px 단위만 반환)

    Returns:
        dict: delta_x_px, delta_y_px, distance_px (+ mm 단위)
    """
    delta_x_px = mesh_circle.center_x_px - plate_circle.center_x_px
    delta_y_px = mesh_circle.center_y_px - plate_circle.center_y_px
    distance_px = math.hypot(delta_x_px, delta_y_px)

    result = {
        "delta_x_px": round(delta_x_px, 3),
        "delta_y_px": round(delta_y_px, 3),
        "distance_px": round(distance_px, 3),
    }
    if pixel_size_mm is not None:
        result["delta_x_mm"] = round(delta_x_px * pixel_size_mm, 4)
        result["delta_y_mm"] = round(delta_y_px * pixel_size_mm, 4)
        result["distance_mm"] = round(distance_px * pixel_size_mm, 4)
    return result


# ---------------------------------------------------------------------------
# 오버레이 시각화 (CLI용)
# ---------------------------------------------------------------------------

def render_overlay(gray, plate_circle, mesh_circle, eccentricity):
    """
    Plate/Mesh 원, 중심 마커, X/Y 편차를 오버레이한 시각화 이미지 생성

    Args:
        gray: 그레이스케일 이미지
        plate_circle: Plate 원
        mesh_circle: Mesh 원
        eccentricity: compute_eccentricity() 결과 dict

    Returns:
        overlay: BGR 오버레이 이미지
    """
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    plate_color = (0, 220, 0)
    mesh_color = (255, 100, 0)
    shift_color = (0, 0, 255)
    text_color = (30, 30, 255)

    pcx = int(round(plate_circle.center_x_px))
    pcy = int(round(plate_circle.center_y_px))
    mcx = int(round(mesh_circle.center_x_px))
    mcy = int(round(mesh_circle.center_y_px))

    cv2.circle(overlay, (pcx, pcy), int(round(plate_circle.radius_px)), plate_color, 4)
    cv2.circle(overlay, (mcx, mcy), int(round(mesh_circle.radius_px)), mesh_color, 4)

    cv2.drawMarker(overlay, (pcx, pcy), plate_color, cv2.MARKER_CROSS, 42, 4)
    cv2.drawMarker(overlay, (mcx, mcy), mesh_color, cv2.MARKER_CROSS, 42, 4)

    cv2.arrowedLine(overlay, (pcx, pcy), (mcx, mcy), shift_color, 3, cv2.LINE_AA, tipLength=0.18)

    mid_x = int(round((pcx + mcx) * 0.5)) + 24
    mid_y = int(round((pcy + mcy) * 0.5)) - 8

    has_mm = "delta_x_mm" in eccentricity
    x_val = eccentricity.get("delta_x_mm", eccentricity["delta_x_px"])
    y_val = eccentricity.get("delta_y_mm", eccentricity["delta_y_px"])
    suffix = " mm" if has_mm else " px"
    x_label = f"dX={x_val:.3f}{suffix}"
    y_label = f"dY={y_val:.3f}{suffix}"

    for idx, line in enumerate((x_label, y_label)):
        y = mid_y + idx * 34
        cv2.putText(overlay, line, (mid_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (20, 20, 20), 4, cv2.LINE_AA)
        cv2.putText(overlay, line, (mid_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.82, shift_color, 2, cv2.LINE_AA)

    font_scale = max(gray.shape[0], gray.shape[1]) / 2200.0
    font_scale = max(font_scale, 0.8)
    line_height = int(round(38 * font_scale))

    info_lines = [
        "Plate vs Mesh center-axis X/Y shift",
        f"Plate center = ({plate_circle.center_x_px:.1f}, {plate_circle.center_y_px:.1f}) px  r={plate_circle.radius_px:.1f}",
        f"Mesh  center = ({mesh_circle.center_x_px:.1f}, {mesh_circle.center_y_px:.1f}) px  r={mesh_circle.radius_px:.1f}",
        f"dX={eccentricity['delta_x_px']} px,  dY={eccentricity['delta_y_px']} px,  dist={eccentricity['distance_px']} px",
    ]
    if has_mm:
        info_lines.append(
            f"dX={eccentricity['delta_x_mm']} mm,  dY={eccentricity['delta_y_mm']} mm,  dist={eccentricity['distance_mm']} mm"
        )

    for i, line in enumerate(info_lines):
        baseline_y = 70 + i * line_height
        cv2.putText(overlay, line, (40, baseline_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 2, cv2.LINE_AA)

    return overlay


# ---------------------------------------------------------------------------
# 원 검출 진입점
# ---------------------------------------------------------------------------

def detect_circles(gray, hv):
    """
    Plate 외곽 원 + Mesh 외곽 원 검출

    Args:
        gray: 전처리된 그레이스케일 이미지
        hv: HV 필터 이미지

    Returns:
        plate_circle: Plate 외곽 원
        mesh_circle: Mesh 외곽 원
    """
    plate_circle = detect_plate_outer_circle(gray)
    mesh_circle = detect_mesh_circle(hv, plate_circle)
    return plate_circle, mesh_circle
