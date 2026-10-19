#!/usr/bin/env python3
"""3-pin and JIG line detection for the X-ray inspection pipeline."""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from .circles import fit_circle_least_squares


def _circle_value(circle: Any, key: str) -> float:
    if isinstance(circle, dict):
        return float(circle[key])
    return float(getattr(circle, key))


# ===========================================================================
# 3-pin detection
# ===========================================================================

def detect_3pins(hv, mesh_circle):
    """Detect the three dark pin markers inside the mesh circle."""
    h, w = hv.shape
    mcx = mesh_circle.center_x_px
    mcy = mesh_circle.center_y_px
    mr = int(round(mesh_circle.radius_px))

    search_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(search_mask, (int(round(mcx)), int(round(mcy))), mr, 255, -1)

    blur_k = max(15, int(mr * 0.02) | 1)
    if blur_k % 2 == 0:
        blur_k += 1

    hv_f = hv.astype(np.float32)
    hv_masked = hv_f.copy()
    hv_masked[search_mask == 0] = 0
    mask_f = (search_mask / 255.0).astype(np.float32)

    blur_val = cv2.GaussianBlur(hv_masked, (blur_k, blur_k), 0)
    blur_mask = cv2.GaussianBlur(mask_f, (blur_k, blur_k), 0)
    blur_mask[blur_mask < 0.01] = 1.0
    hv_blurred = blur_val / blur_mask

    mean_val = float(np.mean(hv_blurred[search_mask > 0]))
    best = None
    best_gap = 0

    for pct in [0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80]:
        thresh = mean_val * pct
        binary = np.zeros((h, w), dtype=np.uint8)
        binary[(hv_blurred < thresh) & (search_mask > 0)] = 255

        open_k = max(3, int(mr * 0.004) | 1)
        if open_k % 2 == 0:
            open_k += 1
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kern)

        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
        mesh_area = np.count_nonzero(search_mask)
        min_area = max(50, int(mesh_area * 0.0001))
        max_area = int(mesh_area * 0.02)

        candidates = []
        for i in range(1, n_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_area or area > max_area:
                continue
            comp = (labels == i).astype(np.uint8) * 255
            cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            cnt = max(cnts, key=cv2.contourArea)
            peri = cv2.arcLength(cnt, True)
            if peri < 1:
                continue
            circ = 4 * math.pi * area / (peri * peri)
            if circ < 0.4:
                continue
            candidates.append({
                "cx": float(centroids[i][0]),
                "cy": float(centroids[i][1]),
                "area": int(area),
                "radius": math.sqrt(float(area) / math.pi),
                "circularity": float(circ),
            })

        if len(candidates) >= 3:
            candidates.sort(key=lambda c: c["area"], reverse=True)
            top3 = candidates[:3]
            angles = sorted(
                math.atan2(c["cy"] - mcy, c["cx"] - mcx) for c in top3
            )
            gaps = [
                math.degrees((angles[(j + 1) % 3] - angles[j]) % (2 * math.pi))
                for j in range(3)
            ]
            min_gap = min(gaps)
            if min_gap > best_gap:
                best_gap = min_gap
                pts = np.array([(c["cx"], c["cy"]) for c in top3], dtype=np.float64)
                best = {
                    "top3": top3,
                    "circumscribed": fit_circle_least_squares(pts),
                    "thresh_pct": pct,
                    "min_gap": float(min_gap),
                    "blur_k": blur_k,
                    "n_candidates": len(candidates),
                }

    return best


# ===========================================================================
# JIG four-axis line detection
# ===========================================================================

_JIG_SIDES = {
    "top": {"angle": 270.0, "orientation": "vertical", "color": (255, 0, 255)},
    "right": {"angle": 0.0, "orientation": "horizontal", "color": (0, 190, 255)},
    "bottom": {"angle": 90.0, "orientation": "vertical", "color": (255, 0, 255)},
    "left": {"angle": 180.0, "orientation": "horizontal", "color": (0, 190, 255)},
}


def _normalize_u8(image: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    arr = image.astype(np.float32)
    if mask is not None and np.any(mask > 0):
        sample = arr[mask > 0]
    else:
        sample = arr[arr > 0]
    if sample.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    lo = float(np.percentile(sample, 1.0))
    hi = float(np.percentile(sample, 99.7))
    if hi <= lo + 1e-6:
        lo = 0.0
        hi = float(sample.max())
    if hi <= lo + 1e-6:
        return np.zeros(arr.shape, dtype=np.uint8)
    return np.clip((arr - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)


def _build_plate_outside_mesh_masks(
    shape: tuple[int, int],
    plate_circle: Any,
    mesh_circle: Any,
    mesh_margin_px: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    plate_cx = _circle_value(plate_circle, "center_x_px")
    plate_cy = _circle_value(plate_circle, "center_y_px")
    plate_r = _circle_value(plate_circle, "radius_px")
    mesh_cx = _circle_value(mesh_circle, "center_x_px")
    mesh_cy = _circle_value(mesh_circle, "center_y_px")
    mesh_r = _circle_value(mesh_circle, "radius_px")

    plate_mask = np.zeros(shape, dtype=np.uint8)
    cv2.circle(
        plate_mask,
        (int(round(plate_cx)), int(round(plate_cy))),
        max(1, int(round(plate_r - 2.0))),
        255,
        -1,
    )
    outside_mask = plate_mask.copy()
    cv2.circle(
        outside_mask,
        (int(round(mesh_cx)), int(round(mesh_cy))),
        max(1, int(round(mesh_r + mesh_margin_px))),
        0,
        -1,
    )
    return plate_mask, outside_mask


def _side_roi_masks(
    shape: tuple[int, int],
    plate_circle: Any,
    outside_mask: np.ndarray,
    half_width_deg: float = 16.0,
    radial_dilate_px: int = 2,
) -> dict[str, np.ndarray]:
    cx = _circle_value(plate_circle, "center_x_px")
    cy = _circle_value(plate_circle, "center_y_px")
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    angles = (np.degrees(np.arctan2(yy - cy, xx - cx)) + 360.0) % 360.0

    roi_base = outside_mask.copy()
    if radial_dilate_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (radial_dilate_px * 2 + 1, radial_dilate_px * 2 + 1),
        )
        roi_base = cv2.dilate(roi_base, kernel, iterations=1)

    masks: dict[str, np.ndarray] = {}
    for side, cfg in _JIG_SIDES.items():
        delta = ((angles - float(cfg["angle"]) + 180.0) % 360.0) - 180.0
        mask = np.zeros(shape, dtype=np.uint8)
        mask[(np.abs(delta) <= half_width_deg) & (roi_base > 0)] = 255
        masks[side] = mask
    return masks


def _directional_gray_filter(seed_norm: np.ndarray, active_mask: np.ndarray, orientation: str, kernel_len_px: int) -> np.ndarray:
    masked = seed_norm.copy()
    masked[active_mask == 0] = 0
    if kernel_len_px <= 1:
        filtered = masked
    else:
        k = max(2, int(kernel_len_px))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (1, k) if orientation == "vertical" else (k, 1),
        )
        filtered = cv2.morphologyEx(masked, cv2.MORPH_OPEN, kernel)
    filtered[active_mask == 0] = 0
    return filtered


def _line_angle_deg(line: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = line
    angle = math.degrees(math.atan2(float(y2 - y1), float(x2 - x1)))
    if angle < 0:
        angle += 180.0
    if angle >= 180.0:
        angle -= 180.0
    return angle


def _angle_error_deg(angle: float, orientation: str) -> float:
    if orientation == "vertical":
        return abs(float(angle) - 90.0)
    return min(abs(float(angle)), abs(180.0 - float(angle)))


def _line_length(line: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = line
    return float(math.hypot(float(x2 - x1), float(y2 - y1)))


def _line_coord(line: tuple[int, int, int, int], orientation: str) -> float:
    x1, y1, x2, y2 = line
    if orientation == "vertical":
        return 0.5 * (float(x1) + float(x2))
    return 0.5 * (float(y1) + float(y2))


def _line_radial_stats(line: tuple[int, int, int, int], plate_circle: Any) -> dict[str, float]:
    x1, y1, x2, y2 = [float(v) for v in line]
    length = max(_line_length(line), 1.0)
    n = max(8, int(length))
    xs = np.linspace(x1, x2, n, dtype=np.float32)
    ys = np.linspace(y1, y2, n, dtype=np.float32)
    cx = _circle_value(plate_circle, "center_x_px")
    cy = _circle_value(plate_circle, "center_y_px")
    plate_r = _circle_value(plate_circle, "radius_px")
    radial = np.hypot(xs - cx, ys - cy)
    return {
        "p95_radius_px": float(np.percentile(radial, 95.0)),
        "max_radius_px": float(radial.max()),
        "closest_plate_gap_px": float(np.min(np.abs(radial - plate_r))),
        "outer_gap_px": float(max(0.0, plate_r - float(np.percentile(radial, 95.0)))),
        "attach_ratio": float(np.mean(radial >= plate_r - 28.0)),
    }


def _line_intensity_stats(line: tuple[int, int, int, int], image: np.ndarray) -> dict[str, float]:
    x1, y1, x2, y2 = [float(v) for v in line]
    length = max(_line_length(line), 1.0)
    n = max(8, int(length))
    xs = np.clip(np.rint(np.linspace(x1, x2, n)).astype(np.int32), 0, image.shape[1] - 1)
    ys = np.clip(np.rint(np.linspace(y1, y2, n)).astype(np.int32), 0, image.shape[0] - 1)
    values = image[ys, xs].astype(np.float32)
    return {
        "mean": float(values.mean()),
        "p90": float(np.percentile(values, 90.0)),
        "support_ratio": float(np.mean(values > 0)),
    }


def _candidate_from_line(
    line: tuple[int, int, int, int],
    side: str,
    plate_circle: Any,
    seed_norm: np.ndarray,
    source: str,
    allow_inner_axis_line: bool = False,
) -> dict[str, Any] | None:
    orientation = str(_JIG_SIDES[side]["orientation"])
    expected_coord = _circle_value(plate_circle, "center_x_px") if orientation == "vertical" else _circle_value(plate_circle, "center_y_px")
    length = _line_length(line)
    angle = _line_angle_deg(line)
    angle_error = _angle_error_deg(angle, orientation)
    if angle_error > 12.0:
        return None

    intensity = _line_intensity_stats(line, seed_norm)
    if intensity["support_ratio"] < 0.04:
        return None

    coord = _line_coord(line, orientation)
    center_distance = abs(coord - expected_coord)
    radial = _line_radial_stats(line, plate_circle)
    if radial["attach_ratio"] < 0.08 and radial["closest_plate_gap_px"] > 22.0:
        radius = _circle_value(plate_circle, "radius_px")
        center_near = center_distance <= max(45.0, radius * 0.030)
        long_enough = length >= max(45.0, radius * 0.028)
        still_in_outer_band = radial["outer_gap_px"] <= max(55.0, radius * 0.040)
        if not (allow_inner_axis_line and center_near and long_enough and still_in_outer_band):
            return None

    plate_attach_score = 190.0 * radial["attach_ratio"] + (260.0 / (radial["closest_plate_gap_px"] + 3.0))
    line_support_score = (0.70 * intensity["p90"]) + (0.28 * intensity["mean"]) + (90.0 * intensity["support_ratio"])
    score = (
        (1.70 * length)
        + plate_attach_score
        + line_support_score
        - (4.0 * angle_error)
        - (0.85 * radial["outer_gap_px"])
        - (0.08 * center_distance)
    )
    return {
        "line": [int(v) for v in line],
        "length_px": round(length, 3),
        "angle_deg": round(angle, 3),
        "angle_error_deg": round(angle_error, 3),
        "coord": round(float(coord), 3),
        "center_distance_px": round(float(center_distance), 3),
        "max_radius_px": round(radial["max_radius_px"], 3),
        "p95_radius_px": round(radial["p95_radius_px"], 3),
        "closest_plate_gap_px": round(radial["closest_plate_gap_px"], 3),
        "outer_gap_px": round(radial["outer_gap_px"], 3),
        "attach_ratio": round(radial["attach_ratio"], 3),
        "line_mean_u8": round(intensity["mean"], 3),
        "line_p90_u8": round(intensity["p90"], 3),
        "support_ratio": round(intensity["support_ratio"], 3),
        "score": round(float(score), 3),
        "orientation": orientation,
        "side": side,
        "source": source,
    }


def _lsd_candidates(
    seed_norm: np.ndarray,
    roi_mask: np.ndarray,
    side: str,
    plate_circle: Any,
    source: str,
) -> list[dict[str, Any]]:
    points = cv2.findNonZero(roi_mask)
    if points is None:
        return []

    x, y, w, h = cv2.boundingRect(points)
    pad = 8
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(seed_norm.shape[1], x + w + pad)
    y1 = min(seed_norm.shape[0], y + h + pad)
    crop_mask = roi_mask[y0:y1, x0:x1]
    crop = cv2.bitwise_and(seed_norm[y0:y1, x0:x1], seed_norm[y0:y1, x0:x1], mask=crop_mask)

    detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    detected = detector.detect(crop)[0]
    if detected is None:
        return []

    candidates: list[dict[str, Any]] = []
    for raw in detected[:, 0, :]:
        fx1, fy1, fx2, fy2 = [float(v) for v in raw]
        line = (
            int(round(fx1 + x0)),
            int(round(fy1 + y0)),
            int(round(fx2 + x0)),
            int(round(fy2 + y0)),
        )
        if _line_length(line) < 8.0:
            continue
        candidate = _candidate_from_line(line, side, plate_circle, seed_norm, source)
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def _component_line_candidates(
    seed_norm: np.ndarray,
    roi_mask: np.ndarray,
    side: str,
    plate_circle: Any,
    source: str,
) -> list[dict[str, Any]]:
    """Recover obvious straight fragments that LSD misses after directional filtering."""
    orientation = str(_JIG_SIDES[side]["orientation"])
    sample = seed_norm[roi_mask > 0]
    sample = sample[sample > 0]
    if sample.size == 0:
        return []

    thresholds = sorted({
        max(6.0, float(np.percentile(sample, 75.0))),
        max(10.0, float(np.percentile(sample, 90.0))),
        max(16.0, float(np.percentile(sample, 95.0))),
    })
    candidates: list[dict[str, Any]] = []
    radius = _circle_value(plate_circle, "radius_px")
    min_major = max(35.0, radius * 0.022)

    for threshold in thresholds:
        binary = np.zeros(seed_norm.shape, dtype=np.uint8)
        binary[(seed_norm > threshold) & (roi_mask > 0)] = 255
        n_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

        for label_idx in range(1, n_labels):
            x, y, w, h, area = [int(v) for v in stats[label_idx]]
            if area < 20:
                continue

            if orientation == "horizontal":
                major = float(w)
                minor = float(h)
                if major < min_major or minor > max(14.0, major * 0.22) or major / max(minor, 1.0) < 3.2:
                    continue
                ys, xs = np.where(labels == label_idx)
                line = (
                    int(round(float(np.percentile(xs, 3.0)))),
                    int(round(float(np.median(ys)))),
                    int(round(float(np.percentile(xs, 97.0)))),
                    int(round(float(np.median(ys)))),
                )
            else:
                major = float(h)
                minor = float(w)
                if major < min_major or minor > max(14.0, major * 0.22) or major / max(minor, 1.0) < 3.2:
                    continue
                ys, xs = np.where(labels == label_idx)
                line = (
                    int(round(float(np.median(xs)))),
                    int(round(float(np.percentile(ys, 3.0)))),
                    int(round(float(np.median(xs)))),
                    int(round(float(np.percentile(ys, 97.0)))),
                )

            candidate = _candidate_from_line(line, side, plate_circle, seed_norm, source, allow_inner_axis_line=True)
            if candidate is None:
                continue
            candidate["component_area_px"] = int(area)
            candidate["component_bbox"] = [x, y, w, h]
            candidate["component_threshold_u8"] = round(float(threshold), 3)
            candidates.append(candidate)

    return _dedupe_candidates(candidates)


def _segment_interval(item: dict[str, Any], orientation: str) -> tuple[float, float]:
    x1, y1, x2, y2 = [float(v) for v in item["line"]]
    if orientation == "vertical":
        return min(y1, y2), max(y1, y2)
    return min(x1, x2), max(x1, x2)


def _group_segments_by_axis(
    candidates: list[dict[str, Any]],
    side: str,
    plate_circle: Any,
    seed_norm: np.ndarray,
    coord_tol_px: float = 10.0,
    max_gap_px: float = 120.0,
) -> list[dict[str, Any]]:
    if len(candidates) < 2:
        return []

    orientation = str(_JIG_SIDES[side]["orientation"])
    coord_groups: list[list[dict[str, Any]]] = []
    for item in sorted(candidates, key=lambda value: float(value["coord"])):
        coord = float(item["coord"])
        placed = False
        for group in coord_groups:
            weights = [max(1.0, float(member["length_px"])) for member in group]
            group_coord = float(np.average([float(member["coord"]) for member in group], weights=weights))
            if abs(coord - group_coord) <= coord_tol_px:
                group.append(item)
                placed = True
                break
        if not placed:
            coord_groups.append([item])

    grouped: list[dict[str, Any]] = []
    for coord_group in coord_groups:
        tracks: list[list[dict[str, Any]]] = []
        active: list[dict[str, Any]] = []
        active_end: float | None = None
        for item in sorted(coord_group, key=lambda member: _segment_interval(member, orientation)[0]):
            start, end = _segment_interval(item, orientation)
            if not active:
                active = [item]
                active_end = end
                continue
            assert active_end is not None
            if start - active_end <= max_gap_px:
                active.append(item)
                active_end = max(active_end, end)
            else:
                tracks.append(active)
                active = [item]
                active_end = end
        if active:
            tracks.append(active)

        for track in tracks:
            if len(track) < 2:
                continue
            weights = [max(1.0, float(member["length_px"])) for member in track]
            coord = float(np.average([float(member["coord"]) for member in track], weights=weights))
            intervals = [_segment_interval(member, orientation) for member in track]
            start = min(interval[0] for interval in intervals)
            end = max(interval[1] for interval in intervals)
            span = max(0.0, end - start)
            if span < 1.0:
                continue

            merged_intervals: list[list[float]] = []
            for interval_start, interval_end in sorted(intervals):
                if not merged_intervals or interval_start > merged_intervals[-1][1]:
                    merged_intervals.append([interval_start, interval_end])
                else:
                    merged_intervals[-1][1] = max(merged_intervals[-1][1], interval_end)
            total_segment_len = sum(max(0.0, interval[1] - interval[0]) for interval in merged_intervals)
            gap_px = max(0.0, span - total_segment_len)
            coverage_ratio = total_segment_len / max(span, 1.0)
            coord_std_px = float(np.std([float(member["coord"]) for member in track]))

            if orientation == "vertical":
                line = (int(round(coord)), int(round(start)), int(round(coord)), int(round(end)))
            else:
                line = (int(round(start)), int(round(coord)), int(round(end)), int(round(coord)))

            angle = _line_angle_deg(line)
            radial = _line_radial_stats(line, plate_circle)
            intensity = _line_intensity_stats(line, seed_norm)
            child_scores = [float(member["score"]) for member in track]
            child_p90 = [float(member["line_p90_u8"]) for member in track]
            expected_coord = _circle_value(plate_circle, "center_x_px") if orientation == "vertical" else _circle_value(plate_circle, "center_y_px")
            center_distance = abs(coord - expected_coord)
            score = (
                max(child_scores)
                + (1.25 * span)
                + (0.55 * total_segment_len)
                + (90.0 * coverage_ratio)
                + (0.45 * float(np.mean(child_p90)))
                - (1.00 * gap_px)
                - (7.0 * coord_std_px)
                - (0.85 * radial["outer_gap_px"])
                - (0.10 * center_distance)
            )
            grouped.append({
                "line": [int(v) for v in line],
                "length_px": round(_line_length(line), 3),
                "angle_deg": round(angle, 3),
                "angle_error_deg": round(_angle_error_deg(angle, orientation), 3),
                "coord": round(coord, 3),
                "center_distance_px": round(float(center_distance), 3),
                "max_radius_px": round(radial["max_radius_px"], 3),
                "p95_radius_px": round(radial["p95_radius_px"], 3),
                "closest_plate_gap_px": round(radial["closest_plate_gap_px"], 3),
                "outer_gap_px": round(radial["outer_gap_px"], 3),
                "attach_ratio": round(radial["attach_ratio"], 3),
                "line_mean_u8": round(intensity["mean"], 3),
                "line_p90_u8": round(float(np.mean(child_p90)), 3),
                "support_ratio": round(coverage_ratio, 3),
                "score": round(float(score), 3),
                "orientation": orientation,
                "side": side,
                "source": "lsd_grouped_from_04_norm",
                "group_segment_count": len(track),
                "group_span_px": round(span, 3),
                "group_gap_px": round(gap_px, 3),
                "group_coverage_ratio": round(coverage_ratio, 3),
                "group_coord_std_px": round(coord_std_px, 3),
            })
    return grouped


def _detect_side_candidates(seed_norm: np.ndarray, roi_mask: np.ndarray, side: str, plate_circle: Any, source: str) -> list[dict[str, Any]]:
    candidates = _lsd_candidates(seed_norm, roi_mask, side, plate_circle, source)
    if "directional" in source:
        candidates.extend(_component_line_candidates(seed_norm, roi_mask, side, plate_circle, "component_from_directional_04_norm"))
    candidates.extend(_group_segments_by_axis(candidates, side, plate_circle, seed_norm))
    return candidates


def _dedupe_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for candidate in candidates:
        line = tuple(int(round(float(v))) for v in candidate.get("line", []))
        key = (candidate.get("side"), line, round(float(candidate.get("coord", 0.0)), 3), candidate.get("source"))
        previous = best_by_key.get(key)
        if previous is None or float(candidate["score"]) > float(previous["score"]):
            best_by_key[key] = candidate
    return list(best_by_key.values())


def _best_with_preview(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not candidates:
        return None
    best = dict(max(candidates, key=lambda item: float(item["score"])))
    best["candidate_count"] = len(candidates)
    best["candidate_preview"] = sorted(candidates, key=lambda item: float(item["score"]), reverse=True)[:30]
    return best


def _payload_with_pair(selected: dict[str, Any], candidates: list[dict[str, Any]], pair_score: float, pair_delta_px: float, axis_center_distance_px: float) -> dict[str, Any]:
    payload = dict(selected)
    payload["candidate_count"] = len(candidates)
    payload["candidate_preview"] = sorted(candidates, key=lambda item: float(item["score"]), reverse=True)[:30]
    payload["pair_score"] = round(float(pair_score), 3)
    payload["pair_delta_px"] = round(float(pair_delta_px), 3)
    payload["axis_center_distance_px"] = round(float(axis_center_distance_px), 3)
    return payload


def _select_axis_pair(
    first_side: str,
    second_side: str,
    candidates_by_side: dict[str, list[dict[str, Any]]],
    fallback_by_side: dict[str, dict[str, Any] | None],
    plate_circle: Any,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    first_candidates = sorted(candidates_by_side[first_side], key=lambda item: float(item["score"]), reverse=True)[:40]
    second_candidates = sorted(candidates_by_side[second_side], key=lambda item: float(item["score"]), reverse=True)[:40]
    if not first_candidates or not second_candidates:
        return fallback_by_side[first_side], fallback_by_side[second_side]

    orientation = str(_JIG_SIDES[first_side]["orientation"])
    expected_coord = _circle_value(plate_circle, "center_x_px") if orientation == "vertical" else _circle_value(plate_circle, "center_y_px")
    max_pair_delta_px = max(55.0, _circle_value(plate_circle, "radius_px") * 0.09)
    best_pair: tuple[float, dict[str, Any], dict[str, Any], float, float] | None = None

    for first in first_candidates:
        first_coord = float(first["coord"])
        for second in second_candidates:
            second_coord = float(second["coord"])
            pair_delta_px = abs(first_coord - second_coord)
            if pair_delta_px > max_pair_delta_px:
                continue
            axis_center_distance_px = abs((first_coord + second_coord) * 0.5 - expected_coord)
            pair_score = (
                float(first["score"])
                + float(second["score"])
                - (3.0 * pair_delta_px)
                - (0.45 * axis_center_distance_px)
            )
            if best_pair is None or pair_score > best_pair[0]:
                best_pair = (pair_score, first, second, pair_delta_px, axis_center_distance_px)

    if best_pair is None:
        return fallback_by_side[first_side], fallback_by_side[second_side]

    pair_score, first, second, pair_delta_px, axis_center_distance_px = best_pair
    return (
        _payload_with_pair(first, candidates_by_side[first_side], pair_score, pair_delta_px, axis_center_distance_px),
        _payload_with_pair(second, candidates_by_side[second_side], pair_score, pair_delta_px, axis_center_distance_px),
    )


def _select_consistent_axis_pairs(
    candidates_by_side: dict[str, list[dict[str, Any]]],
    fallback_by_side: dict[str, dict[str, Any] | None],
    plate_circle: Any,
) -> dict[str, dict[str, Any] | None]:
    selected = dict(fallback_by_side)
    selected["top"], selected["bottom"] = _select_axis_pair("top", "bottom", candidates_by_side, fallback_by_side, plate_circle)
    selected["left"], selected["right"] = _select_axis_pair("left", "right", candidates_by_side, fallback_by_side, plate_circle)
    return selected


def _make_virtual_side_candidate(side: str, source_side: str, source: dict[str, Any], rejected: dict[str, Any], plate_circle: Any) -> dict[str, Any]:
    """Create a conservative virtual side when a tiny opposite-side fragment breaks an otherwise clear axis."""
    repaired = dict(source)
    cx = _circle_value(plate_circle, "center_x_px")
    cy = _circle_value(plate_circle, "center_y_px")
    radius = _circle_value(plate_circle, "radius_px")
    coord = float(source["coord"])
    length = max(40.0, min(float(source.get("length_px", 40.0)), radius * 0.12))
    orientation = str(_JIG_SIDES[side]["orientation"])

    if orientation == "horizontal":
        y = int(round(coord))
        if side == "left":
            x_edge = int(round(cx - radius))
            line = [x_edge, y, int(round(x_edge + length)), y]
        else:
            x_edge = int(round(cx + radius))
            line = [int(round(x_edge - length)), y, x_edge, y]
    else:
        x = int(round(coord))
        if side == "top":
            y_edge = int(round(cy - radius))
            line = [x, y_edge, x, int(round(y_edge + length))]
        else:
            y_edge = int(round(cy + radius))
            line = [x, int(round(y_edge - length)), x, y_edge]

    repaired.update({
        "line": line,
        "length_px": round(float(length), 3),
        "coord": round(float(coord), 3),
        "score": round(float(max(float(rejected.get("score", 0.0)), min(float(source.get("score", 0.0)) * 0.70, 520.0))), 3),
        "support_ratio": round(float(min(float(source.get("support_ratio", 0.0)), 0.90)), 3),
        "source": f"virtual_from_{source_side}_axis_repair",
        "virtual": True,
        "virtual_from_side": source_side,
        "replaced_side": side,
        "replaced_line": rejected.get("line"),
        "replaced_score": rejected.get("score"),
        "replaced_length_px": rejected.get("length_px"),
        "candidate_preview": [rejected],
    })
    return repaired


def _repair_unbalanced_axis_pairs(selected_by_side: dict[str, dict[str, Any] | None], plate_circle: Any) -> dict[str, dict[str, Any] | None]:
    repaired = dict(selected_by_side)
    initial_vertical = _axis_connection_assessment(selected_by_side, "top", "bottom", plate_circle)
    initial_horizontal = _axis_connection_assessment(selected_by_side, "left", "right", plate_circle)
    initial_axes = {
        ("top", "bottom"): initial_vertical,
        ("left", "right"): initial_horizontal,
    }
    radius = _circle_value(plate_circle, "radius_px")

    def maybe_repair(first_side: str, second_side: str, other_axis_connected: bool) -> None:
        axis = initial_axes[(first_side, second_side)]
        if axis["is_connected"] or not other_axis_connected:
            return
        first = selected_by_side[first_side]
        second = selected_by_side[second_side]
        if first is None or second is None:
            return

        first_len = float(first.get("length_px", 0.0))
        second_len = float(second.get("length_px", 0.0))
        if first_len >= second_len:
            strong_side, strong, weak_side, weak = first_side, first, second_side, second
        else:
            strong_side, strong, weak_side, weak = second_side, second, first_side, first

        strong_len = max(first_len, second_len)
        weak_len = min(first_len, second_len)
        strong_score = float(strong.get("score", 0.0))
        expected_coord = (
            _circle_value(plate_circle, "center_x_px")
            if str(_JIG_SIDES[first_side]["orientation"]) == "vertical"
            else _circle_value(plate_circle, "center_y_px")
        )
        strong_center_distance = abs(float(strong["coord"]) - expected_coord)

        if strong_len < max(90.0, radius * 0.055):
            return
        if strong_score < 600.0:
            return
        if weak_len > max(35.0, strong_len * 0.35):
            return
        if strong_center_distance > max(35.0, radius * 0.022):
            return
        if axis["pair_delta_px"] is not None and float(axis["pair_delta_px"]) <= float(axis["pair_delta_limit_px"]):
            return

        repaired[weak_side] = _make_virtual_side_candidate(weak_side, strong_side, strong, weak, plate_circle)

    maybe_repair("top", "bottom", bool(initial_horizontal["is_connected"]))
    maybe_repair("left", "right", bool(initial_vertical["is_connected"]))
    return repaired


def _axis_connection_assessment(selected_by_side: dict[str, dict[str, Any] | None], first_side: str, second_side: str, plate_circle: Any) -> dict[str, Any]:
    first = selected_by_side[first_side]
    second = selected_by_side[second_side]
    orientation = str(_JIG_SIDES[first_side]["orientation"])
    radius = _circle_value(plate_circle, "radius_px")
    expected_coord = _circle_value(plate_circle, "center_x_px") if orientation == "vertical" else _circle_value(plate_circle, "center_y_px")
    pair_delta_limit_px = max(70.0, radius * 0.035)
    axis_center_limit_px = max(25.0, radius * 0.014)
    result: dict[str, Any] = {
        "orientation": orientation,
        "first_side": first_side,
        "second_side": second_side,
        "pair_delta_limit_px": round(float(pair_delta_limit_px), 3),
        "axis_center_limit_px": round(float(axis_center_limit_px), 3),
        "pair_delta_px": None,
        "axis_center_distance_px": None,
        "status": "missing_axis",
        "is_connected": False,
    }
    if first is None and second is None:
        result["missing_sides"] = [first_side, second_side]
        return result
    if first is None or second is None:
        result["missing_sides"] = [first_side if first is None else second_side]
        result["status"] = "single_side_only"
        return result

    first_coord = float(first["coord"])
    second_coord = float(second["coord"])
    pair_delta_px = abs(first_coord - second_coord)
    axis_center_distance_px = abs((first_coord + second_coord) * 0.5 - expected_coord)
    connected = pair_delta_px <= pair_delta_limit_px and axis_center_distance_px <= axis_center_limit_px
    result.update({
        "missing_sides": [],
        "pair_delta_px": round(float(pair_delta_px), 3),
        "axis_center_distance_px": round(float(axis_center_distance_px), 3),
        "status": "connected" if connected else "connected_needs_review",
        "is_connected": bool(connected),
    })
    return result


def _assess_jig_detection(selected_by_side: dict[str, dict[str, Any] | None], plate_circle: Any) -> dict[str, Any]:
    selected = [item for item in selected_by_side.values() if item is not None]
    missing_sides = [side for side in _JIG_SIDES if selected_by_side[side] is None]
    virtual_sides = [
        f"{side}<-{item.get('virtual_from_side')}"
        for side, item in selected_by_side.items()
        if item is not None and bool(item.get("virtual"))
    ]
    scores = [float(item["score"]) for item in selected]
    lengths = [float(item["length_px"]) for item in selected]
    supports = [float(item.get("support_ratio", 0.0)) for item in selected]
    vertical = _axis_connection_assessment(selected_by_side, "top", "bottom", plate_circle)
    horizontal = _axis_connection_assessment(selected_by_side, "left", "right", plate_circle)
    axes = {"vertical": vertical, "horizontal": horizontal}

    total_score = float(sum(scores))
    min_score = float(min(scores)) if scores else 0.0
    total_length = float(sum(lengths))
    min_length = float(min(lengths)) if lengths else 0.0
    min_support = float(min(supports)) if supports else 0.0
    pair_deltas = [float(axis["pair_delta_px"]) for axis in axes.values() if axis["pair_delta_px"] is not None]
    axis_center_distances = [float(axis["axis_center_distance_px"]) for axis in axes.values() if axis["axis_center_distance_px"] is not None]
    max_pair_delta = max(pair_deltas) if pair_deltas else None
    max_axis_center_distance = max(axis_center_distances) if axis_center_distances else None

    radius = _circle_value(plate_circle, "radius_px")
    no_jig_center_limit_px = max(24.0, radius * 0.014)
    reasons: list[str] = []
    status = "needs_review"
    confidence = 0.50

    if missing_sides:
        status = "jig_absent"
        confidence = 0.92 if len(missing_sides) >= 2 else 0.84
        reasons.append("missing_side:" + ",".join(missing_sides))
    elif total_score < 1900.0:
        status = "jig_absent"
        confidence = 0.88
        reasons.append(f"weak_total_score:{total_score:.1f}<1900.0")
    elif min_score < 340.0 and total_score < 2400.0:
        status = "jig_absent"
        confidence = 0.82
        reasons.append(f"weak_side_score:{min_score:.1f}<340.0")
    elif max_axis_center_distance is not None and total_score < 2400.0 and max_axis_center_distance > no_jig_center_limit_px:
        status = "jig_absent"
        confidence = 0.80
        reasons.append(f"off_center_low_score:{max_axis_center_distance:.1f}>{no_jig_center_limit_px:.1f}")
    elif max_axis_center_distance is not None and total_score < 2600.0 and max_axis_center_distance > max(100.0, radius * 0.055):
        status = "jig_absent"
        confidence = 0.78
        reasons.append(f"extreme_axis_offset:{max_axis_center_distance:.1f}")
    elif max_pair_delta is not None and total_score < 2600.0 and max_pair_delta > max(160.0, radius * 0.10):
        status = "jig_absent"
        confidence = 0.78
        reasons.append(f"extreme_pair_delta:{max_pair_delta:.1f}")
    elif vertical["is_connected"] and horizontal["is_connected"]:
        status = "jig_present"
        confidence = 0.88 if total_score >= 3000.0 else 0.76
        reasons.append("four_sides_connected")
        if virtual_sides:
            confidence = min(confidence, 0.72)
            reasons.append("virtual_side:" + ",".join(virtual_sides))
    else:
        status = "needs_review"
        confidence = 0.48
        for axis_name, axis in axes.items():
            if not axis["is_connected"]:
                reasons.append(f"{axis_name}:{axis['status']}")

    if vertical["status"].startswith("missing") or horizontal["status"].startswith("missing"):
        line_connection_status = "missing_axis"
    elif vertical["status"] == "single_side_only" or horizontal["status"] == "single_side_only":
        line_connection_status = "single_side_virtual"
    elif vertical["is_connected"] and horizontal["is_connected"]:
        line_connection_status = "connected"
    else:
        line_connection_status = "connected_needs_review"

    return {
        "jig_status": status,
        "confidence": round(float(confidence), 3),
        "line_connection_status": line_connection_status,
        "reasons": reasons,
        "virtual_sides": virtual_sides,
        "missing_sides": missing_sides,
        "selected_side_count": len(selected),
        "total_score": round(total_score, 3),
        "min_score": round(min_score, 3),
        "total_length_px": round(total_length, 3),
        "min_length_px": round(min_length, 3),
        "min_support_ratio": round(min_support, 3),
        "max_pair_delta_px": None if max_pair_delta is None else round(float(max_pair_delta), 3),
        "max_axis_center_distance_px": None if max_axis_center_distance is None else round(float(max_axis_center_distance), 3),
        "axes": axes,
    }


def _anchor_from_side(side: str, info: dict[str, Any] | None, plate_circle: Any, mesh_circle: Any) -> tuple[float, float] | None:
    if info is None:
        return None
    cx = _circle_value(plate_circle, "center_x_px")
    cy = _circle_value(plate_circle, "center_y_px")
    mesh_r = _circle_value(mesh_circle, "radius_px")
    if side == "top":
        return float(info["coord"]), cy - mesh_r
    if side == "bottom":
        return float(info["coord"]), cy + mesh_r
    if side == "left":
        return cx - mesh_r, float(info["coord"])
    if side == "right":
        return cx + mesh_r, float(info["coord"])
    return None


def _line_circle_intersections(p1: tuple[float, float], p2: tuple[float, float], center: tuple[float, float], radius: float) -> list[tuple[float, float]]:
    x1, y1 = p1
    x2, y2 = p2
    cx, cy = center
    dx = x2 - x1
    dy = y2 - y1
    fx = x1 - cx
    fy = y1 - cy
    a = dx * dx + dy * dy
    b = 2.0 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - radius * radius
    disc = b * b - 4.0 * a * c
    if a <= 1e-8 or disc < 0:
        return []
    root = math.sqrt(disc)
    return [
        (x1 + ((-b - root) / (2.0 * a)) * dx, y1 + ((-b - root) / (2.0 * a)) * dy),
        (x1 + ((-b + root) / (2.0 * a)) * dx, y1 + ((-b + root) / (2.0 * a)) * dy),
    ]


def _draw_axis_inside_circle(
    shape: tuple[int, int],
    circle: Any,
    first_anchor: tuple[float, float] | None,
    second_anchor: tuple[float, float] | None,
    fallback_orientation: str,
    fallback_coord: float | None,
    thickness_px: int,
    radius_offset_px: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    mask = np.zeros(shape, dtype=np.uint8)
    cx = _circle_value(circle, "center_x_px")
    cy = _circle_value(circle, "center_y_px")
    radius = max(1.0, _circle_value(circle, "radius_px") + radius_offset_px)
    line_info: dict[str, Any] = {"status": "missing"}

    if first_anchor is not None and second_anchor is not None:
        hits = _line_circle_intersections(first_anchor, second_anchor, (cx, cy), radius)
        if len(hits) == 2:
            p1, p2 = hits
            cv2.line(mask, (int(round(p1[0])), int(round(p1[1]))), (int(round(p2[0])), int(round(p2[1]))), 255, thickness_px, cv2.LINE_AA)
            return mask, {
                "status": "paired",
                "p1": [round(float(p1[0]), 3), round(float(p1[1]), 3)],
                "p2": [round(float(p2[0]), 3), round(float(p2[1]), 3)],
            }

    if fallback_coord is None:
        return mask, line_info

    if fallback_orientation == "vertical":
        dx = float(fallback_coord) - cx
        if abs(dx) > radius:
            return mask, line_info
        half = math.sqrt(max(radius * radius - dx * dx, 0.0))
        p1 = (float(fallback_coord), cy - half)
        p2 = (float(fallback_coord), cy + half)
    else:
        dy = float(fallback_coord) - cy
        if abs(dy) > radius:
            return mask, line_info
        half = math.sqrt(max(radius * radius - dy * dy, 0.0))
        p1 = (cx - half, float(fallback_coord))
        p2 = (cx + half, float(fallback_coord))

    cv2.line(mask, (int(round(p1[0])), int(round(p1[1]))), (int(round(p2[0])), int(round(p2[1]))), 255, thickness_px, cv2.LINE_AA)
    return mask, {
        "status": "fallback_single_side",
        "p1": [round(float(p1[0]), 3), round(float(p1[1]), 3)],
        "p2": [round(float(p2[0]), 3), round(float(p2[1]), 3)],
    }


def _line_intersection(line_a: dict[str, Any], line_b: dict[str, Any]) -> tuple[float, float] | None:
    if line_a.get("p1") is None or line_a.get("p2") is None or line_b.get("p1") is None or line_b.get("p2") is None:
        return None
    ax1, ay1 = [float(v) for v in line_a["p1"]]
    ax2, ay2 = [float(v) for v in line_a["p2"]]
    bx1, by1 = [float(v) for v in line_b["p1"]]
    bx2, by2 = [float(v) for v in line_b["p2"]]
    dax, day = ax2 - ax1, ay2 - ay1
    dbx, dby = bx2 - bx1, by2 - by1
    det = dax * dby - day * dbx
    if abs(det) <= 1e-6:
        return None
    t = ((bx1 - ax1) * dby - (by1 - ay1) * dbx) / det
    return ax1 + t * dax, ay1 + t * day


def _draw_line_mask(shape: tuple[int, int], line: list[int] | None, thickness_px: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    if line is not None:
        x1, y1, x2, y2 = [int(v) for v in line]
        cv2.line(mask, (x1, y1), (x2, y2), 255, thickness_px, cv2.LINE_AA)
    return mask


def _draw_detection_overlay(gray: np.ndarray, plate_circle: Any, mesh_circle: Any, candidates_by_side: dict[str, list[dict[str, Any]]], selected_by_side: dict[str, dict[str, Any] | None]) -> np.ndarray:
    overlay = cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for side, candidates in candidates_by_side.items():
        color = tuple(int(v) for v in _JIG_SIDES[side]["color"])
        for item in candidates:
            x1, y1, x2, y2 = [int(v) for v in item["line"]]
            cv2.line(overlay, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
    for side, selected in selected_by_side.items():
        if selected is None:
            continue
        x1, y1, x2, y2 = [int(v) for v in selected["line"]]
        cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 80), 5, cv2.LINE_AA)
        cv2.putText(overlay, side, (int(round((x1 + x2) * 0.5)), int(round((y1 + y2) * 0.5))), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (0, 255, 80), 2, cv2.LINE_AA)
    cv2.circle(overlay, (int(round(_circle_value(plate_circle, "center_x_px"))), int(round(_circle_value(plate_circle, "center_y_px")))), int(round(_circle_value(plate_circle, "radius_px"))), (0, 220, 0), 3, cv2.LINE_AA)
    cv2.circle(overlay, (int(round(_circle_value(mesh_circle, "center_x_px"))), int(round(_circle_value(mesh_circle, "center_y_px")))), int(round(_circle_value(mesh_circle, "radius_px"))), (255, 150, 0), 3, cv2.LINE_AA)
    return overlay


def detect_jig_cross(gray, hv, plate_circle, mesh_circle):
    """Detect JIG presence from the 12/3/6/9 straight line fragments.

    The implementation intentionally avoids filename/product branching. It
    searches the plate-outside-mesh band, keeps direction-specific ROI
    fragments, pairs top-bottom and left-right axes, then classifies no-JIG
    conservatively from missing/weak/off-center axis evidence.
    """
    shape = gray.shape[:2]
    mesh_cx = _circle_value(mesh_circle, "center_x_px")
    mesh_cy = _circle_value(mesh_circle, "center_y_px")
    selected_thickness_px = 5

    empty = {
        "jig_x": int(round(mesh_cx)),
        "jig_y": int(round(mesh_cy)),
        "has_jig": False,
        "jig_status": "jig_absent",
        "confidence": 0.0,
        "line_connection_status": "missing_axis",
        "reasons": ["not_run"],
        "cross_angles": [],
        "cross_lines": [],
        "n_lines": 0,
        "n_lines_raw": 0,
        "total_len": 0.0,
        "debug_lines": [],
        "side_coords": {side: None for side in _JIG_SIDES},
        "selected_by_side": {side: None for side in _JIG_SIDES},
        "candidate_counts": {side: 0 for side in _JIG_SIDES},
        "step_gap_hv": None,
        "step_radial": None,
        "bridge_mask": np.zeros(shape, dtype=np.uint8),
        "virtual_axis_mask": np.zeros(shape, dtype=np.uint8),
    }

    plate_mask, outside_mask = _build_plate_outside_mesh_masks(shape, plate_circle, mesh_circle, 3.0)
    outside_seed = cv2.bitwise_and(hv, hv, mask=outside_mask)
    seed_norm = _normalize_u8(outside_seed, outside_mask)
    seed_norm[outside_mask == 0] = 0
    if np.count_nonzero(seed_norm) == 0:
        result = dict(empty)
        result["step_gap_hv"] = outside_seed
        return result

    roi_masks = _side_roi_masks(shape, plate_circle, outside_mask, 16.0, 2)
    candidates_by_side: dict[str, list[dict[str, Any]]] = {}
    fallback_by_side: dict[str, dict[str, Any] | None] = {}

    for side, cfg in _JIG_SIDES.items():
        orientation = str(cfg["orientation"])
        side_source = _directional_gray_filter(seed_norm, roi_masks[side], orientation, 7)
        directional = _detect_side_candidates(side_source, roi_masks[side], side, plate_circle, "lsd_from_directional_04_norm")
        raw = _detect_side_candidates(seed_norm, roi_masks[side], side, plate_circle, "raw04_fallback_lsd_from_04_norm")
        candidates = _dedupe_candidates(directional + raw)
        candidates_by_side[side] = candidates
        fallback_by_side[side] = _best_with_preview(candidates)

    selected_by_side = _select_consistent_axis_pairs(candidates_by_side, fallback_by_side, plate_circle)
    selected_by_side = _repair_unbalanced_axis_pairs(selected_by_side, plate_circle)
    assessment = _assess_jig_detection(selected_by_side, plate_circle)

    side_coords = {side: None if selected_by_side[side] is None else selected_by_side[side]["coord"] for side in _JIG_SIDES}
    selected_masks = {
        side: _draw_line_mask(shape, None if selected_by_side[side] is None else selected_by_side[side]["line"], selected_thickness_px)
        for side in _JIG_SIDES
    }

    top_anchor = _anchor_from_side("top", selected_by_side["top"], plate_circle, mesh_circle)
    bottom_anchor = _anchor_from_side("bottom", selected_by_side["bottom"], plate_circle, mesh_circle)
    left_anchor = _anchor_from_side("left", selected_by_side["left"], plate_circle, mesh_circle)
    right_anchor = _anchor_from_side("right", selected_by_side["right"], plate_circle, mesh_circle)

    vertical_fallback = None
    if selected_by_side["top"] is not None and selected_by_side["bottom"] is not None:
        vertical_fallback = 0.5 * (float(selected_by_side["top"]["coord"]) + float(selected_by_side["bottom"]["coord"]))
    elif selected_by_side["top"] is not None:
        vertical_fallback = float(selected_by_side["top"]["coord"])
    elif selected_by_side["bottom"] is not None:
        vertical_fallback = float(selected_by_side["bottom"]["coord"])

    horizontal_fallback = None
    if selected_by_side["left"] is not None and selected_by_side["right"] is not None:
        horizontal_fallback = 0.5 * (float(selected_by_side["left"]["coord"]) + float(selected_by_side["right"]["coord"]))
    elif selected_by_side["left"] is not None:
        horizontal_fallback = float(selected_by_side["left"]["coord"])
    elif selected_by_side["right"] is not None:
        horizontal_fallback = float(selected_by_side["right"]["coord"])

    bridge_mask = np.zeros(shape, dtype=np.uint8)
    virtual_axis_mask = np.zeros(shape, dtype=np.uint8)
    cross_lines: list[tuple[tuple[float, float], tuple[float, float]]] = []
    cross_angles: list[float] = []
    jig_x, jig_y = mesh_cx, mesh_cy
    has_jig = assessment["jig_status"] != "jig_absent"

    if has_jig:
        vertical_bridge, _vertical_bridge_line = _draw_axis_inside_circle(
            shape, mesh_circle, top_anchor, bottom_anchor, "vertical", vertical_fallback, selected_thickness_px, 2.0
        )
        horizontal_bridge, _horizontal_bridge_line = _draw_axis_inside_circle(
            shape, mesh_circle, left_anchor, right_anchor, "horizontal", horizontal_fallback, selected_thickness_px, 2.0
        )
        virtual_vertical, virtual_vertical_line = _draw_axis_inside_circle(
            shape, plate_circle, top_anchor, bottom_anchor, "vertical", vertical_fallback, selected_thickness_px, -2.0
        )
        virtual_horizontal, virtual_horizontal_line = _draw_axis_inside_circle(
            shape, plate_circle, left_anchor, right_anchor, "horizontal", horizontal_fallback, selected_thickness_px, -2.0
        )
        bridge_mask = cv2.max(vertical_bridge, horizontal_bridge)
        virtual_axis_mask = cv2.max(virtual_vertical, virtual_horizontal)
        if virtual_vertical_line.get("p1") is not None and virtual_vertical_line.get("p2") is not None:
            cross_lines.append((tuple(virtual_vertical_line["p1"]), tuple(virtual_vertical_line["p2"])))
            cross_angles.append(90.0)
        if virtual_horizontal_line.get("p1") is not None and virtual_horizontal_line.get("p2") is not None:
            cross_lines.append((tuple(virtual_horizontal_line["p1"]), tuple(virtual_horizontal_line["p2"])))
            cross_angles.append(0.0)
        intersection = _line_intersection(virtual_vertical_line, virtual_horizontal_line)
        if intersection is not None:
            jig_x, jig_y = intersection
        elif vertical_fallback is not None and horizontal_fallback is not None:
            jig_x, jig_y = float(vertical_fallback), float(horizontal_fallback)

    debug_lines = []
    for side, selected in selected_by_side.items():
        if selected is None:
            continue
        line = selected["line"]
        x1, y1, x2, y2 = [float(v) for v in line]
        debug_lines.append({
            "side": side,
            "line": tuple(line),
            "length": float(selected.get("length_px", 0.0)),
            "angle": float(selected.get("angle_deg", 0.0)),
            "mid": ((x1 + x2) * 0.5, (y1 + y2) * 0.5),
            "score": float(selected.get("score", 0.0)),
            "source": selected.get("source", ""),
        })

    selected_outside_mask = np.zeros(shape, dtype=np.uint8)
    for mask in selected_masks.values():
        selected_outside_mask = cv2.max(selected_outside_mask, mask)

    return {
        "jig_x": int(round(jig_x)),
        "jig_y": int(round(jig_y)),
        "has_jig": bool(has_jig),
        "jig_status": assessment["jig_status"],
        "confidence": assessment["confidence"],
        "line_connection_status": assessment["line_connection_status"],
        "reasons": assessment["reasons"],
        "jig_assessment": assessment,
        "cross_angles": cross_angles,
        "cross_lines": cross_lines,
        "n_lines": len(debug_lines),
        "n_lines_raw": int(sum(len(v) for v in candidates_by_side.values())),
        "total_len": round(float(sum(float(item.get("length", 0.0)) for item in debug_lines)), 3),
        "debug_lines": debug_lines,
        "side_coords": side_coords,
        "selected_by_side": selected_by_side,
        "candidate_counts": {side: len(candidates_by_side[side]) for side in _JIG_SIDES},
        "step_gap_hv": outside_seed,
        "step_radial": _draw_detection_overlay(gray, plate_circle, mesh_circle, candidates_by_side, selected_by_side),
        "selected_outside_mask": selected_outside_mask,
        "bridge_mask": bridge_mask,
        "virtual_axis_mask": virtual_axis_mask,
        "plate_mask": plate_mask,
        "outside_mask": outside_mask,
        "method": "four_roi_lsd_axis_pair_no_jig",
    }
