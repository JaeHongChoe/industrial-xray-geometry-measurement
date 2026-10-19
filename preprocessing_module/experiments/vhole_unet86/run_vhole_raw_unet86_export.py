#!/usr/bin/env python3
"""Run V7-86 YOLO + Small U-Net V-hole inference on raw X-ray images.

Raw images are converted with ``preprocess_image`` first. Known training samples
reuse their historical HV crop size; unknown products use the full HV image so
side V-holes are not clipped.

Axis selection is product-name independent: it first validates the direct YOLO
pair, then tests 12/6 and 3/9 radial candidates before falling back to the full
angle sweep for diagnostics.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


BASE_DIR = Path(__file__).resolve().parents[2]  # preprocessing_module/
for _p in (BASE_DIR / "src", BASE_DIR / "training"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from xray_3pin_detector import detect_circles, preprocess_image  # noqa: E402
from xray_3pin_detector.vhole_unet86 import (  # noqa: E402
    VHoleUNet86Detector,
    VHoleUNet86Prediction,
    apply_shape_prior,
    decode_mask_tip,
    draw_vhole_unet86_overlay,
    extract_patch,
    normalize_fine_patch,
)


CARDINAL_AXIS_TOL_DEG = 35.0
AXIS_FALLBACK_ORDER = (("12/6", 90.0), ("3/9", 0.0))
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _safe_name(path: Path, root: Path) -> str:
    rel = path.relative_to(root).with_suffix("")
    text = "__".join(rel.parts)
    text = re.sub(r"[^0-9A-Za-z가-힣_.-]+", "_", text)
    return text.strip("_")


def _load_known_hv_sizes(model_dir: Path) -> dict[str, tuple[int, int]]:
    for path in (model_dir / "known_hv_sizes.json", BASE_DIR / "known_hv_sizes.json"):
        if not path.exists():
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        return {str(k): (int(v[0]), int(v[1])) for k, v in raw.items()}
    return {}


def _extract_with_padding(img: np.ndarray, x1: int, y1: int, width: int, height: int) -> tuple[np.ndarray, dict]:
    h, w = img.shape[:2]
    x2 = x1 + width
    y2 = y1 + height
    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - w)
    pad_bottom = max(0, y2 - h)

    src = img
    if pad_left or pad_top or pad_right or pad_bottom:
        src = cv2.copyMakeBorder(
            src,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_REFLECT,
        )
        x1 += pad_left
        y1 += pad_top

    crop = src[y1:y1 + height, x1:x1 + width]
    if crop.shape[0] != height or crop.shape[1] != width:
        crop = cv2.resize(crop, (width, height), interpolation=cv2.INTER_LINEAR)

    return crop, {
        "pad_left": pad_left,
        "pad_top": pad_top,
        "pad_right": pad_right,
        "pad_bottom": pad_bottom,
    }


def _crop_hv_like(
    raw_path: Path,
    *,
    hv_crop_mode: str = "auto",
    known_sizes: dict[str, tuple[int, int]] | None = None,
) -> tuple[np.ndarray, dict]:
    known_sizes = known_sizes or {}
    gray, _morph, hv = preprocess_image(str(raw_path))
    plate_circle, mesh_circle = detect_circles(gray, hv)
    sample = raw_path.stem
    known_size = known_sizes.get(sample)

    if hv_crop_mode == "full" or (hv_crop_mode == "auto" and known_size is None):
        target_w = int(hv.shape[1])
        target_h = int(hv.shape[0])
        x1 = y1 = 0
        crop = hv.copy()
        pad = {"pad_left": 0, "pad_top": 0, "pad_right": 0, "pad_bottom": 0}
        crop_source = "full_hv"
    elif known_size is None:
        crop_size = int(round(mesh_circle.radius_px * 2.0 + 400.0))
        crop_size = max(512, min(crop_size, max(hv.shape[:2]) + 256))
        target_w = target_h = crop_size
        crop_source = "mesh_radius_fallback"
        mcx = int(round(mesh_circle.center_x_px))
        mcy = int(round(mesh_circle.center_y_px))
        x1 = int(round(mcx - target_w / 2.0))
        y1 = int(round(mcy - target_h / 2.0))
        crop, pad = _extract_with_padding(hv, x1, y1, target_w, target_h)
    else:
        target_w, target_h = known_size
        crop_source = "known_hv_size"
        mcx = int(round(mesh_circle.center_x_px))
        mcy = int(round(mesh_circle.center_y_px))
        x1 = int(round(mcx - target_w / 2.0))
        y1 = int(round(mcy - target_h / 2.0))
        crop, pad = _extract_with_padding(hv, x1, y1, target_w, target_h)

    meta = {
        "sample": sample,
        "raw_width": int(hv.shape[1]),
        "raw_height": int(hv.shape[0]),
        "crop_width": int(target_w),
        "crop_height": int(target_h),
        "crop_x1": int(x1),
        "crop_y1": int(y1),
        "crop_x2": int(x1 + target_w),
        "crop_y2": int(y1 + target_h),
        "crop_source": crop_source,
        "plate_circle": {
            "x": float(plate_circle.center_x_px),
            "y": float(plate_circle.center_y_px),
            "r": float(plate_circle.radius_px),
        },
        "mesh_circle": {
            "x": float(mesh_circle.center_x_px),
            "y": float(mesh_circle.center_y_px),
            "r": float(mesh_circle.radius_px),
        },
        **pad,
    }
    return crop, meta


def _mesh_center_in_crop(meta: dict) -> tuple[float, float, float] | None:
    mesh = meta.get("mesh_circle") or {}
    plate = meta.get("plate_circle") or {}
    try:
        mx = float(mesh["x"])
        my = float(mesh["y"])
        mr = float(mesh["r"])
        pr = float(plate.get("r", 0.0))
        # Some raw HV conversions make the mesh-circle detector latch onto a
        # tiny central feature. For ring geometry fallback, the plate circle is
        # a safer proxy than a clearly impossible mesh radius.
        if pr > 0.0 and mr < pr * 0.35:
            mx = float(plate["x"])
            my = float(plate["y"])
            mr = pr * 0.94
        return (mx - float(meta["crop_x1"]), my - float(meta["crop_y1"]), mr)
    except (KeyError, TypeError, ValueError):
        return None


def _plate_circle_in_crop(meta: dict) -> tuple[float, float, float] | None:
    plate = meta.get("plate_circle") or {}
    try:
        return (
            float(plate["x"]) - float(meta["crop_x1"]),
            float(plate["y"]) - float(meta["crop_y1"]),
            float(plate["r"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _angle_gap(a: float, b: float) -> float:
    return abs(((a - b + math.pi) % (2.0 * math.pi)) - math.pi)


def _axis_distance_deg(angle_deg: float, target_deg: float) -> float:
    return abs(((float(angle_deg) - float(target_deg) + 90.0) % 180.0) - 90.0)


def _pair_axis_angle_deg(preds: list[VHoleUNet86Prediction]) -> float | None:
    if len(preds) != 2:
        return None
    a, b = preds
    return math.degrees(math.atan2(float(b.y) - float(a.y), float(b.x) - float(a.x))) % 180.0


def _axis_info_from_pair(preds: list[VHoleUNet86Prediction]) -> dict:
    axis_angle = _pair_axis_angle_deg(preds)
    if axis_angle is None:
        return {
            "nominal_axis": None,
            "nominal_axis_deg": None,
            "axis_angle_deg": None,
            "axis_error_deg": None,
            "rotation_to_nominal_deg": None,
        }

    best_label = None
    best_target = None
    best_error = float("inf")
    for label, target in AXIS_FALLBACK_ORDER:
        err = _axis_distance_deg(axis_angle, target)
        if err < best_error:
            best_label = label
            best_target = target
            best_error = err

    if best_error > CARDINAL_AXIS_TOL_DEG:
        best_label = "off_axis"
        best_target = None
        rotation = None
    else:
        rotation = ((float(best_target) - axis_angle + 90.0) % 180.0) - 90.0

    return {
        "nominal_axis": best_label,
        "nominal_axis_deg": float(best_target) if best_target is not None else None,
        "axis_angle_deg": float(axis_angle),
        "axis_error_deg": float(best_error),
        "rotation_to_nominal_deg": float(rotation) if rotation is not None else None,
    }


def _axis_fallback_order(initial_axis: str | None) -> tuple[tuple[str, float], ...]:
    if initial_axis not in {label for label, _deg in AXIS_FALLBACK_ORDER}:
        return AXIS_FALLBACK_ORDER
    ordered = [item for item in AXIS_FALLBACK_ORDER if item[0] == initial_axis]
    ordered.extend(item for item in AXIS_FALLBACK_ORDER if item[0] != initial_axis)
    return tuple(ordered)


def _pair_plausibility(preds: list[VHoleUNet86Prediction], meta: dict, shape: tuple[int, int]) -> tuple[bool, str]:
    """Check whether the two selected V-holes look like an opposite ring pair."""
    if len(preds) != 2:
        return False, f"n_pred={len(preds)}"

    mesh = _mesh_center_in_crop(meta)
    if mesh is None:
        h, w = shape[:2]
        cx, cy, mr = w / 2.0, h / 2.0, min(w, h) / 2.5
    else:
        cx, cy, mr = mesh

    vecs = []
    radii = []
    for pred in preds:
        dx = float(pred.x) - cx
        dy = float(pred.y) - cy
        radius = math.hypot(dx, dy)
        radii.append(radius)
        vecs.append(math.atan2(dy, dx))

    if min(radii) < max(180.0, mr * 0.48):
        return False, f"inner_radius={min(radii):.1f}"
    if abs(radii[0] - radii[1]) > max(radii) * 0.42:
        return False, f"radius_mismatch={radii[0]:.1f},{radii[1]:.1f}"

    gap = _angle_gap(vecs[0], vecs[1])
    if gap < math.radians(125.0):
        return False, f"angle_gap={math.degrees(gap):.1f}"

    return True, "ok"


def _angle_jobs_for_axis(axis_deg: float | None) -> list[tuple[float, str]]:
    if axis_deg is not None:
        return [(float(axis_deg) % 180.0, "axis_fallback")]
    angle_jobs = [(0.0, "principal"), (90.0, "principal")]
    angle_jobs.extend((float(deg), "sweep") for deg in range(15, 180, 15) if deg != 90)
    return angle_jobs


def _build_radial_pairs(meta: dict, shape: tuple[int, int], axis_deg: float | None = None) -> list[dict]:
    """Build product-agnostic opposite candidate pairs around the detected rings.

    The old V7 TYPE_E diagnostic used a horizontal axis prior explicitly. Here we
    do not read the product name; instead we test both principal axes, keep a
    lower-priority 15-degree sweep, and search both mesh-radius and plate-radius
    bands. That lets 3/9 and 12/6 products share the same inference path.
    """
    mesh = _mesh_center_in_crop(meta)
    if mesh is None:
        return []
    cx, cy, mesh_radius = mesh
    plate = _plate_circle_in_crop(meta)
    h, w = shape[:2]
    pairs = []
    seen: set[tuple[int, int, int, int]] = set()

    radius_jobs = [("mesh", frac, mesh_radius * frac) for frac in (0.78, 0.86, 0.94, 1.00, 1.05, 1.10, 1.16)]
    if plate is not None:
        _pcx, _pcy, plate_radius = plate
        radius_jobs.extend(("plate", frac, plate_radius * frac) for frac in (0.70, 0.76, 0.82, 0.88, 0.94))

    angle_jobs = _angle_jobs_for_axis(axis_deg)

    for source, frac, radius in radius_jobs:
        for deg, angle_source in angle_jobs:
            pts = []
            valid = True
            for add in (0.0, 180.0):
                ang = math.radians(deg + add)
                px = cx + math.cos(ang) * radius
                py = cy + math.sin(ang) * radius
                if px < -128 or py < -128 or px > w + 128 or py > h + 128:
                    valid = False
                    break
                pts.append((px, py))
            if valid:
                key = tuple(int(round(v / 8.0)) for pt in pts for v in pt)
                if key in seen:
                    continue
                seen.add(key)
                pairs.append({
                    "pair": pts,
                    "angle_deg": float(deg),
                    "radius_fraction": float(frac),
                    "radius_px": float(radius),
                    "radius_source": source,
                    "angle_source": angle_source,
                    "axis_filter_deg": float(axis_deg) if axis_deg is not None else None,
                })
    return pairs


def _build_radial_candidates(meta: dict, shape: tuple[int, int], axis_deg: float | None = None) -> list[dict]:
    """Build independent radial candidates for pair search.

    Unlike ``_build_radial_pairs``, each side can use a different radius. This
    mirrors the older V7 geometry runner and helps TYPE_E-like products where the
    visible left/right notches do not land at exactly the same ring fraction.
    """
    mesh = _mesh_center_in_crop(meta)
    if mesh is None:
        return []
    cx, cy, mesh_radius = mesh
    plate = _plate_circle_in_crop(meta)
    h, w = shape[:2]

    radius_jobs = [("mesh", frac, mesh_radius * frac) for frac in (0.78, 0.86, 0.94, 1.00, 1.05, 1.10, 1.16)]
    if plate is not None:
        _pcx, _pcy, plate_radius = plate
        radius_jobs.extend(("plate", frac, plate_radius * frac) for frac in (0.70, 0.76, 0.82, 0.88, 0.94))

    angle_jobs = _angle_jobs_for_axis(axis_deg)

    out = []
    seen: set[tuple[int, int]] = set()
    for source, frac, radius in radius_jobs:
        for deg, angle_source in angle_jobs:
            for add in (0.0, 180.0):
                angle_deg = float(deg + add)
                ang = math.radians(angle_deg)
                px = cx + math.cos(ang) * radius
                py = cy + math.sin(ang) * radius
                if px < -128 or py < -128 or px > w + 128 or py > h + 128:
                    continue
                key = (int(round(px / 8.0)), int(round(py / 8.0)))
                if key in seen:
                    continue
                seen.add(key)
                out.append({
                    "center": (float(px), float(py)),
                    "angle_deg": angle_deg % 360.0,
                    "axis_angle_deg": float(deg),
                    "radius_fraction": float(frac),
                    "radius_px": float(radius),
                    "radius_source": source,
                    "angle_source": angle_source,
                    "axis_filter_deg": float(axis_deg) if axis_deg is not None else None,
                })
    return out


def _triangle_quality(pred: VHoleUNet86Prediction) -> float:
    c1 = np.array(pred.corner1, dtype=np.float32)
    c2 = np.array(pred.corner2, dtype=np.float32)
    tip = np.array([pred.x, pred.y], dtype=np.float32)
    base_vec = c2 - c1
    base_len = float(np.linalg.norm(base_vec))
    height = 0.0
    if base_len > 1e-6:
        tip_vec = tip - c1
        height = float(abs(base_vec[0] * tip_vec[1] - base_vec[1] * tip_vec[0]) / base_len)
    area = 0.5 * base_len * height

    base_score = float(np.clip((base_len - 16.0) / 60.0, 0.0, 1.0))
    height_score = float(np.clip((height - 14.0) / 42.0, 0.0, 1.0))
    area_score = float(np.clip((area - 260.0) / 1800.0, 0.0, 1.0))
    score = 2.0 * float(pred.confidence) + 0.45 * base_score + 0.65 * height_score + 0.35 * area_score
    if pred.mask_area < 20:
        score -= 1.1
    if height < 12.0:
        score -= 0.55
    return float(score)


def _axis_chord_bonus(
    a: dict,
    b: dict,
    *,
    axis_deg: float | None,
    mesh: tuple[float, float, float] | None,
) -> tuple[float, dict]:
    """Prefer physically long opposite pairs on the requested principal axis."""
    if axis_deg is None or mesh is None:
        return 0.0, {}

    _cx, _cy, mesh_radius = mesh
    if mesh_radius <= 1e-6:
        return 0.0, {}

    pred_a = a["pred"]
    pred_b = b["pred"]
    axis_rad = math.radians(float(axis_deg))
    ux = math.cos(axis_rad)
    uy = math.sin(axis_rad)
    dx = float(pred_b.x) - float(pred_a.x)
    dy = float(pred_b.y) - float(pred_a.y)

    axis_span = abs(dx * ux + dy * uy)
    off_axis_span = abs(-dx * uy + dy * ux)

    # Fallback candidates can include inner notches with clean masks. The V-hole
    # pair should still form a long chord across the detected ring, so use span
    # as a geometric tie-breaker after the segmentation quality score.
    span_ratio = axis_span / max(mesh_radius * 2.0, 1.0)
    span_score = float(np.clip((span_ratio - 0.84) / 0.20, 0.0, 1.0))
    off_axis_penalty = float(np.clip(off_axis_span / max(mesh_radius * 0.10, 1.0), 0.0, 1.0))
    bonus = 2.6 * span_score - 0.45 * off_axis_penalty

    return float(bonus), {
        "axis_span_px": float(axis_span),
        "axis_span_ratio": float(span_ratio),
        "axis_span_score": float(span_score),
        "off_axis_span_px": float(off_axis_span),
        "off_axis_penalty": float(off_axis_penalty),
        "axis_chord_bonus": float(bonus),
    }


def _predict_unet_at(
    detector: VHoleUNet86Detector,
    img: np.ndarray,
    coarse_x: float,
    coarse_y: float,
) -> VHoleUNet86Prediction:
    patch = extract_patch(img, coarse_x, coarse_y, detector.config.patch_size)
    patch_norm = normalize_fine_patch(patch)
    x = torch.from_numpy(patch_norm).unsqueeze(0).unsqueeze(0).float().to(detector.device)
    out = detector.mask_model(x)[0].detach().cpu()
    mask_prob = torch.sigmoid(out[0]).numpy()
    decoded = decode_mask_tip(mask_prob, float(coarse_x), float(coarse_y), img.shape, detector.config)
    decoded = apply_shape_prior(decoded, img.shape, detector.config)
    tip = decoded["tip"]
    raw_tip = decoded.get("shape_prior_before", decoded.get("raw_tip", tip))
    score = float(decoded["score"])
    return VHoleUNet86Prediction(
        x=float(tip[0]),
        y=float(tip[1]),
        raw_x=float(raw_tip[0]),
        raw_y=float(raw_tip[1]),
        corner1=tuple(float(v) for v in decoded["corner1"]),
        corner2=tuple(float(v) for v in decoded["corner2"]),
        confidence=score,
        mask_score=score,
        mask_area=int(decoded["mask_area"]),
        coarse=(float(coarse_x), float(coarse_y)),
        contour=[tuple(float(v) for v in pt) for pt in decoded["contour"]],
        shape_prior_applied=bool(decoded.get("shape_prior_applied", False)),
    )


def _predict_radial_fallback(
    detector: VHoleUNet86Detector,
    img: np.ndarray,
    meta: dict,
    *,
    axis_label: str | None = None,
    axis_deg: float | None = None,
) -> tuple[list[VHoleUNet86Prediction], list[tuple[float, float]], dict | None]:
    candidate_infos = _build_radial_candidates(meta, img.shape, axis_deg=axis_deg)
    if not candidate_infos:
        return [], [], None

    best_preds: list[VHoleUNet86Prediction] = []
    best_pair: list[tuple[float, float]] = []
    best_debug: dict | None = None
    best_score = -1e9
    mesh = _mesh_center_in_crop(meta)
    if mesh is None:
        h, w = img.shape[:2]
        cx, cy = w / 2.0, h / 2.0
        mesh_for_bonus = None
    else:
        cx, cy, _r = mesh
        mesh_for_bonus = mesh

    with torch.no_grad():
        candidates = []
        for info in candidate_infos:
            coarse_x, coarse_y = info["center"]
            pred = _predict_unet_at(detector, img, coarse_x, coarse_y)
            weak = pred.confidence < 0.15 or pred.mask_area < 20
            dx = pred.x - cx
            dy = pred.y - cy
            pred_angle = math.atan2(dy, dx)
            pred_radius = math.hypot(dx, dy)
            q = _triangle_quality(pred)
            axis_angle = math.radians(float(info["axis_angle_deg"]))
            cardinal_score = max(abs(math.cos(axis_angle)), abs(math.sin(axis_angle)))
            q += 0.45 * cardinal_score
            if info.get("angle_source") in {"principal", "axis_fallback"}:
                q += 0.35
            if weak:
                q -= 1.8
            candidates.append({
                "pred": pred,
                "coarse": (float(coarse_x), float(coarse_y)),
                "quality": float(q),
                "weak": bool(weak),
                "pred_angle": pred_angle,
                "pred_radius": float(pred_radius),
                **info,
            })

        candidates.sort(key=lambda row: row["quality"], reverse=True)
        kept = []
        for row in candidates:
            pred = row["pred"]
            if all(math.hypot(pred.x - old["pred"].x, pred.y - old["pred"].y) >= 35.0 for old in kept):
                kept.append(row)
            if len(kept) >= 24:
                break

        for i in range(len(kept)):
            for j in range(i + 1, len(kept)):
                a = kept[i]
                b = kept[j]
                preds = [a["pred"], b["pred"]]
                plausible, reason = _pair_plausibility(preds, meta, img.shape)
                if a["weak"] or b["weak"]:
                    plausible = False
                    reason = "weak_mask"

                gap = _angle_gap(a["pred_angle"], b["pred_angle"])
                if gap < math.radians(125.0):
                    continue
                radius_gap = abs(a["pred_radius"] - b["pred_radius"]) / max(a["pred_radius"], b["pred_radius"], 1.0)
                if radius_gap > 0.45:
                    continue

                score = a["quality"] + b["quality"]
                gap_score = float(np.clip((gap - math.radians(125.0)) / max(math.radians(55.0), 1e-6), 0.0, 1.0))
                score += 0.75 * gap_score
                chord_bonus, chord_debug = _axis_chord_bonus(
                    a,
                    b,
                    axis_deg=axis_deg,
                    mesh=mesh_for_bonus,
                )
                score += chord_bonus
                pair_axis = _pair_axis_angle_deg(preds)
                axis_error = None
                if axis_deg is not None and pair_axis is not None:
                    axis_error = _axis_distance_deg(pair_axis, axis_deg)
                    if axis_error > CARDINAL_AXIS_TOL_DEG:
                        plausible = False
                        reason = f"axis_error={axis_error:.1f}"
                    score += 3.0 * float(np.clip(1.0 - axis_error / max(CARDINAL_AXIS_TOL_DEG, 1e-6), 0.0, 1.0))
                else:
                    # Product-agnostic axis choice: horizontal and vertical both get
                    # a boost, but no product name is consulted.
                    axis_i = math.radians(float(a["axis_angle_deg"]))
                    axis_j = math.radians(float(b["axis_angle_deg"]))
                    cardinal_pair = (
                        max(abs(math.cos(axis_i)), abs(math.sin(axis_i)))
                        + max(abs(math.cos(axis_j)), abs(math.sin(axis_j)))
                    ) / 2.0
                    score += 8.0 * cardinal_pair
                if a.get("angle_source") in {"principal", "axis_fallback"} and b.get("angle_source") in {"principal", "axis_fallback"}:
                    score += 0.8
                if plausible:
                    score += 0.8
                else:
                    score -= 0.8

                if score > best_score:
                    best_score = score
                    best_preds = preds
                    best_pair = [a["coarse"], b["coarse"]]
                    best_debug = {
                        "score": float(score),
                        "plausible": bool(plausible),
                        "plausibility_reason": reason,
                        "axis_label": axis_label,
                        "axis_filter_deg": float(axis_deg) if axis_deg is not None else None,
                        "pair_axis_deg": float(pair_axis) if pair_axis is not None else None,
                        "axis_error_deg": float(axis_error) if axis_error is not None else None,
                        "angle_gap_deg": float(math.degrees(gap)),
                        "radius_gap_ratio": float(radius_gap),
                        **chord_debug,
                        "selected": [
                            {
                                "coarse": [float(a["coarse"][0]), float(a["coarse"][1])],
                                "tip": [float(a["pred"].x), float(a["pred"].y)],
                                "quality": float(a["quality"]),
                                "weak": bool(a["weak"]),
                                "angle_deg": float(a["angle_deg"]),
                                "axis_angle_deg": float(a["axis_angle_deg"]),
                                "radius_fraction": float(a["radius_fraction"]),
                                "radius_px": float(a["radius_px"]),
                                "radius_source": str(a["radius_source"]),
                            },
                            {
                                "coarse": [float(b["coarse"][0]), float(b["coarse"][1])],
                                "tip": [float(b["pred"].x), float(b["pred"].y)],
                                "quality": float(b["quality"]),
                                "weak": bool(b["weak"]),
                                "angle_deg": float(b["angle_deg"]),
                                "axis_angle_deg": float(b["axis_angle_deg"]),
                                "radius_fraction": float(b["radius_fraction"]),
                                "radius_px": float(b["radius_px"]),
                                "radius_source": str(b["radius_source"]),
                            },
                        ],
                        "top_candidates": [
                            {
                                "coarse": [float(row["coarse"][0]), float(row["coarse"][1])],
                                "tip": [float(row["pred"].x), float(row["pred"].y)],
                                "quality": float(row["quality"]),
                                "weak": bool(row["weak"]),
                                "axis_angle_deg": float(row["axis_angle_deg"]),
                                "radius_fraction": float(row["radius_fraction"]),
                                "radius_source": str(row["radius_source"]),
                            }
                            for row in kept[:8]
                        ],
                    }

        if best_debug is None and kept:
            # Last-resort diagnostics only. The caller will reject this if the
            # resulting pair is not plausible.
            selected = kept[:2]
            best_preds = [row["pred"] for row in selected]
            best_pair = [row["coarse"] for row in selected]
            best_debug = {
                "score": float(sum(row["quality"] for row in selected)),
                "plausible": False,
                "plausibility_reason": "no_pair",
                "axis_label": axis_label,
                "axis_filter_deg": float(axis_deg) if axis_deg is not None else None,
                "selected": [
                    {
                        "coarse": [float(row["coarse"][0]), float(row["coarse"][1])],
                        "tip": [float(row["pred"].x), float(row["pred"].y)],
                        "quality": float(row["quality"]),
                        "weak": bool(row["weak"]),
                        "axis_angle_deg": float(row["axis_angle_deg"]),
                        "radius_fraction": float(row["radius_fraction"]),
                        "radius_source": str(row["radius_source"]),
                    }
                    for row in selected
                ],
            }
    return best_preds, best_pair, best_debug


def _maybe_predict_with_fallback(
    detector: VHoleUNet86Detector,
    hv_img: np.ndarray,
    meta: dict,
) -> tuple[list[VHoleUNet86Prediction], list[tuple[float, float]], dict]:
    preds, coarse_locs = detector.predict_image(hv_img)
    plausible, reason = _pair_plausibility(preds, meta, hv_img.shape)
    axis_info = _axis_info_from_pair(preds)
    debug = {
        "initial_n": len(preds),
        "initial_plausible": bool(plausible),
        "initial_reason": reason,
        "initial_axis": axis_info,
        "fallback_used": False,
        "fallback_type": None,
        "fallback_debug": None,
        "axis_fallbacks": [],
        **axis_info,
    }
    if plausible and axis_info.get("nominal_axis") in {"12/6", "3/9"}:
        debug["axis_source"] = "initial_yolo"
        return preds, coarse_locs, debug

    seed_axis = axis_info.get("nominal_axis")
    if str(reason).startswith("angle_gap"):
        seed_axis = None
    debug["fallback_seed_axis"] = seed_axis

    for axis_label, axis_deg in _axis_fallback_order(seed_axis):
        axis_preds, axis_coarse, axis_debug = _predict_radial_fallback(
            detector,
            hv_img,
            meta,
            axis_label=axis_label,
            axis_deg=axis_deg,
        )
        axis_ok, axis_reason = _pair_plausibility(axis_preds, meta, hv_img.shape)
        if axis_debug is not None and not axis_debug.get("plausible", False):
            axis_ok = False
            axis_reason = str(axis_debug.get("plausibility_reason", axis_reason))
        axis_result_info = _axis_info_from_pair(axis_preds)
        debug["axis_fallbacks"].append({
            "axis_label": axis_label,
            "axis_deg": float(axis_deg),
            "ok": bool(axis_ok),
            "reason": axis_reason,
            "debug": axis_debug,
            **axis_result_info,
        })
        if axis_ok and axis_result_info.get("nominal_axis") == axis_label:
            debug.update(axis_result_info)
            debug["axis_source"] = "axis_fallback"
            debug["fallback_used"] = True
            debug["fallback_type"] = f"{axis_label}_axis"
            debug["fallback_debug"] = axis_debug
            return axis_preds, axis_coarse, debug

    radial_preds, radial_coarse, radial_debug = _predict_radial_fallback(detector, hv_img, meta)
    radial_ok, radial_reason = _pair_plausibility(radial_preds, meta, hv_img.shape)
    if radial_debug is not None and not radial_debug.get("plausible", False):
        radial_ok = False
        radial_reason = str(radial_debug.get("plausibility_reason", radial_reason))
    radial_axis_info = _axis_info_from_pair(radial_preds)
    debug["radial_plausible"] = bool(radial_ok)
    debug["radial_reason"] = radial_reason
    debug["fallback_debug"] = radial_debug
    debug["radial_axis"] = radial_axis_info
    if radial_ok:
        debug.update(radial_axis_info)
        debug["axis_source"] = "radial_sweep"
        debug["fallback_used"] = True
        debug["fallback_type"] = "radial_sweep"
        return radial_preds, radial_coarse, debug
    debug["axis_source"] = "unresolved"
    return preds, coarse_locs, debug


def run(
    raw_dir: Path,
    model_dir: Path,
    out_dir: Path,
    *,
    device: str | None,
    hv_crop_mode: str,
    save_hv: bool,
    vis_ext: str,
) -> None:
    vis_dir = out_dir / "visualizations"
    hv_dir = out_dir / "hv_inputs"
    vis_dir.mkdir(parents=True, exist_ok=True)
    if save_hv:
        hv_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(p for p in raw_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    detector = VHoleUNet86Detector.from_model_dir(model_dir, device=device)
    known_sizes = _load_known_hv_sizes(model_dir)

    rows = []
    started = time.perf_counter()
    print("=" * 80)
    print(" Raw Data V-hole inference with V7-86 YOLO + Small U-Net export")
    print(f" raw:   {raw_dir}")
    print(f" model: {model_dir}")
    print(f" out:   {out_dir}")
    print(f" crop:  {hv_crop_mode}")
    print(f" known: {len(known_sizes)}")
    print(f" n:     {len(image_paths)}")
    print("=" * 80)

    for idx, raw_path in enumerate(image_paths, start=1):
        row = {
            "raw_path": str(raw_path),
            "ok": False,
            "error": None,
            "predictions": [],
        }
        try:
            hv_img, meta = _crop_hv_like(raw_path, hv_crop_mode=hv_crop_mode, known_sizes=known_sizes)
            name = _safe_name(raw_path, raw_dir)
            hv_path = None
            if save_hv:
                hv_path = hv_dir / f"{name}_hv.png"
                cv2.imwrite(str(hv_path), hv_img)

            preds, coarse_locs, inference_debug = _maybe_predict_with_fallback(detector, hv_img, meta)
            vis_path = vis_dir / f"{name}_vhole86.{vis_ext.lstrip('.')}"
            draw_vhole_unet86_overlay(
                hv_img,
                preds,
                coarse_locs=coarse_locs,
                out_path=vis_path,
            )

            xoff = float(meta["crop_x1"])
            yoff = float(meta["crop_y1"])
            pred_rows = []
            for pred in preds:
                item = pred.to_dict()
                item.update({
                    "x_crop": float(pred.x),
                    "y_crop": float(pred.y),
                    "x_raw": float(pred.x + xoff),
                    "y_raw": float(pred.y + yoff),
                    "raw_tip_x_crop": float(pred.raw_x),
                    "raw_tip_y_crop": float(pred.raw_y),
                    "raw_tip_x_raw": float(pred.raw_x + xoff),
                    "raw_tip_y_raw": float(pred.raw_y + yoff),
                })
                pred_rows.append(item)

            row.update({
                "ok": True,
                "hv_path": str(hv_path) if hv_path is not None else None,
                "visualization": str(vis_path),
                "crop_meta": meta,
                "nominal_axis": inference_debug.get("nominal_axis"),
                "axis_angle_deg": inference_debug.get("axis_angle_deg"),
                "rotation_to_nominal_deg": inference_debug.get("rotation_to_nominal_deg"),
                "axis_source": inference_debug.get("axis_source"),
                "coarse_locs": [[float(x), float(y)] for x, y in coarse_locs],
                "inference_debug": inference_debug,
                "predictions": pred_rows,
            })
            tips = [(round(p["x_raw"], 1), round(p["y_raw"], 1)) for p in pred_rows]
            fb = " fallback" if inference_debug.get("fallback_used") else ""
            axis = inference_debug.get("nominal_axis") or "axis?"
            print(f"[{idx:02d}/{len(image_paths):02d}] {raw_path.stem:30s} n={len(pred_rows)} axis={axis:>4s} tips={tips} {meta['crop_source']}{fb}")
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            print(f"[{idx:02d}/{len(image_paths):02d}] {raw_path.stem:30s} ERROR {row['error']}")
        rows.append(row)

    summary = {
        "raw_dir": str(raw_dir),
        "model_dir": str(model_dir),
        "out_dir": str(out_dir),
        "num_images": len(image_paths),
        "num_ok": sum(1 for row in rows if row["ok"]),
        "num_failed": sum(1 for row in rows if not row["ok"]),
        "hv_crop_mode": hv_crop_mode,
        "save_hv": save_hv,
        "elapsed_sec": time.perf_counter() - started,
        "results": rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "predictions.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("=" * 80)
    print(f"ok={summary['num_ok']}/{summary['num_images']} failed={summary['num_failed']}")
    print(f"json={out_dir / 'predictions.json'}")
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=BASE_DIR / "raw_data")
    parser.add_argument("--model-dir", type=Path, default=BASE_DIR / "v7_86_unet_rot_model_export")
    parser.add_argument("--out-dir", type=Path, default=BASE_DIR / "vhole_raw_unet86_rot_results")
    parser.add_argument("--device", default=None)
    parser.add_argument("--hv-crop-mode", choices=["auto", "crop", "full"], default="auto")
    parser.add_argument("--save-hv", action="store_true")
    parser.add_argument("--vis-ext", default="jpg")
    args = parser.parse_args()

    run(
        args.raw_dir,
        args.model_dir,
        args.out_dir,
        device=args.device,
        hv_crop_mode=args.hv_crop_mode,
        save_hv=args.save_hv,
        vis_ext=args.vis_ext,
    )


if __name__ == "__main__":
    main()
