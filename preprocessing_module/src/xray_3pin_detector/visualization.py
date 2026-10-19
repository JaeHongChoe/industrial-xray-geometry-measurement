from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PANEL_W = 760
PANEL_H = 760
TEXT_H = 92


def _to_bgr(gray: np.ndarray) -> np.ndarray:
    if gray.ndim == 3:
        return gray.copy()
    return cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_GRAY2BGR)


def _fit_panel(img: np.ndarray, title: str, lines: list[str] | None = None, width: int = PANEL_W, height: int = PANEL_H) -> np.ndarray:
    lines = lines or []
    body_h = height - TEXT_H
    panel = np.full((height, width, 3), 248, dtype=np.uint8)
    h, w = img.shape[:2]
    scale = min(width / max(w, 1), body_h / max(h, 1))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    ox = (width - new_w) // 2
    oy = TEXT_H + (body_h - new_h) // 2
    panel[oy:oy + new_h, ox:ox + new_w] = resized
    cv2.rectangle(panel, (0, 0), (width - 1, height - 1), (220, 220, 220), 1)
    cv2.putText(panel, title, (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (20, 20, 20), 2, cv2.LINE_AA)
    y = 58
    for line in lines[:2]:
        cv2.putText(panel, line[:96], (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (65, 65, 65), 1, cv2.LINE_AA)
        y += 22
    return panel


def _draw_circle(img: np.ndarray, circle: Any, color: tuple[int, int, int], thickness: int = 4) -> None:
    cv2.circle(
        img,
        (int(round(circle.center_x_px)), int(round(circle.center_y_px))),
        int(round(circle.radius_px)),
        color,
        thickness,
        cv2.LINE_AA,
    )
    cv2.drawMarker(
        img,
        (int(round(circle.center_x_px)), int(round(circle.center_y_px))),
        color,
        cv2.MARKER_CROSS,
        42,
        4,
        cv2.LINE_AA,
    )


def _vhole_points(result: dict[str, Any]) -> list[dict[str, Any]]:
    pair = result.get("vhole", {}).get("pair")
    if not isinstance(pair, dict):
        return []
    points = []
    for key in ("point_a", "point_b"):
        point = pair.get(key)
        if isinstance(point, dict):
            points.append(point)
    return points


def _draw_vholes(img: np.ndarray, result: dict[str, Any]) -> None:
    points = _vhole_points(result)
    if len(points) < 2:
        return
    p0, p1 = points
    xy0 = (int(round(p0["x_px"])), int(round(p0["y_px"])))
    xy1 = (int(round(p1["x_px"])), int(round(p1["y_px"])))
    cv2.line(img, xy0, xy1, (255, 80, 20), 5, cv2.LINE_AA)
    for idx, point in enumerate(points, start=1):
        xy = (int(round(point["x_px"])), int(round(point["y_px"])))
        cv2.circle(img, xy, 22, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(img, xy, 32, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(
            img,
            f"V{idx} {point.get('score', 0.0):.2f}",
            (xy[0] + 34, xy[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            3,
            cv2.LINE_AA,
        )


def _draw_pins(img: np.ndarray, result: dict[str, Any]) -> None:
    pins = result.get("three_pin", {}).get("pins", [])
    for idx, pin in enumerate(pins, start=1):
        xy = (int(round(pin.get("cx", 0.0))), int(round(pin.get("cy", 0.0))))
        cv2.circle(img, xy, max(12, int(round(pin.get("radius", 8.0)))), (0, 180, 255), 4, cv2.LINE_AA)
        cv2.putText(img, f"P{idx}", (xy[0] + 18, xy[1] + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 120, 220), 3, cv2.LINE_AA)


def _draw_jig(img: np.ndarray, result: dict[str, Any]) -> None:
    jig = result.get("jig", {})
    if not jig.get("has_jig"):
        return
    xy = (int(round(jig.get("jig_x", 0.0))), int(round(jig.get("jig_y", 0.0))))
    cv2.drawMarker(img, xy, (255, 0, 255), cv2.MARKER_TILTED_CROSS, 64, 5, cv2.LINE_AA)
    cv2.putText(img, "JIG", (xy[0] + 28, xy[1] - 18), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 3, cv2.LINE_AA)
    for line in jig.get("cross_lines", [])[:8]:
        try:
            x1, y1, x2, y2 = [int(round(float(v))) for v in line]
        except Exception:
            continue
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3, cv2.LINE_AA)


def _extract_crop(gray: np.ndarray, x: float, y: float, size: int = 360) -> np.ndarray:
    h, w = gray.shape[:2]
    half = size // 2
    x1 = int(round(x)) - half
    y1 = int(round(y)) - half
    x2 = x1 + size
    y2 = y1 + size
    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - w)
    pad_bottom = max(0, y2 - h)
    src = gray
    if pad_left or pad_top or pad_right or pad_bottom:
        src = cv2.copyMakeBorder(src, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT)
        x1 += pad_left
        y1 += pad_top
    crop = src[y1:y1 + size, x1:x1 + size]
    if crop.shape[0] != size or crop.shape[1] != size:
        crop = cv2.resize(crop, (size, size))
    return crop


def _make_crop_panel(gray: np.ndarray, result: dict[str, Any]) -> np.ndarray:
    points = _vhole_points(result)
    canvas = np.full((PANEL_H, PANEL_W, 3), 248, dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (PANEL_W - 1, PANEL_H - 1), (220, 220, 220), 1)
    cv2.putText(canvas, "V-hole Zoom", (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (20, 20, 20), 2, cv2.LINE_AA)
    if not points:
        cv2.putText(canvas, "No V-hole result", (18, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 180), 2, cv2.LINE_AA)
        return canvas
    cell_w = (PANEL_W - 54) // 2
    cell_h = cell_w
    for idx, point in enumerate(points[:2]):
        crop = _to_bgr(_extract_crop(gray, point["x_px"], point["y_px"], 360))
        cv2.drawMarker(crop, (180, 180), (0, 0, 255), cv2.MARKER_CROSS, 56, 4, cv2.LINE_AA)
        crop = cv2.resize(crop, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
        x = 18 + idx * (cell_w + 18)
        y = 88
        canvas[y:y + cell_h, x:x + cell_w] = crop
        cv2.rectangle(canvas, (x, y), (x + cell_w, y + cell_h), (210, 210, 210), 1)
        label = f"V{idx + 1} angle={point.get('angle_deg', 0.0):.1f} rr={point.get('radius_ratio', 0.0):.3f}"
        cv2.putText(canvas, label, (x, y + cell_h + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (45, 45, 45), 1, cv2.LINE_AA)
    vhole = result.get("vhole", {})
    lines = [
        f"method: {vhole.get('method', '')}",
        f"axis: {vhole.get('nominal_axis')}  rot: {vhole.get('rotation_to_nominal_deg')}",
        f"score: {vhole.get('pair', {}).get('pair_score') if isinstance(vhole.get('pair'), dict) else None}",
    ]
    y = 560
    for line in lines:
        cv2.putText(canvas, line[:86], (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (45, 45, 45), 1, cv2.LINE_AA)
        y += 28
    return canvas


def make_review_image(payload: dict[str, Any]) -> np.ndarray:
    arrays = payload["arrays"]
    result = payload["result"]
    circles = payload["circles"]
    raw = arrays["raw_gray"]
    processed = arrays["processed_gray"]
    hv = arrays["hv"]

    overlay = _to_bgr(processed)
    _draw_circle(overlay, circles["plate"], (0, 210, 0), 5)
    _draw_circle(overlay, circles["mesh"], (255, 120, 0), 5)
    _draw_pins(overlay, result)
    _draw_jig(overlay, result)
    _draw_vholes(overlay, result)

    hv_bgr = cv2.applyColorMap(hv.astype(np.uint8), cv2.COLORMAP_TURBO)
    _draw_circle(hv_bgr, circles["mesh"], (255, 255, 255), 4)
    _draw_vholes(hv_bgr, result)

    summary = payload["summary"]
    raw_lines = [
        f"{summary.get('dataset', '')}",
        f"group={summary.get('product_group')} status={summary.get('status')}",
    ]
    overlay_lines = [
        f"plate-mesh dx={summary.get('plate_mesh_dx_px')} dy={summary.get('plate_mesh_dy_px')}",
        f"pin={summary.get('pin_status')} jig={summary.get('jig_status')}",
    ]
    hv_lines = [
        f"vhole={summary.get('vhole_method')}",
        f"axis={summary.get('vhole_axis')} rot={summary.get('vhole_rotation_to_axis_deg')}",
    ]

    top = np.hstack([
        _fit_panel(_to_bgr(raw), f"Raw: {summary.get('image_name')}", raw_lines),
        _fit_panel(overlay, "Geometry Overlay", overlay_lines),
    ])
    bottom = np.hstack([
        _fit_panel(hv_bgr, "HV + V-hole", hv_lines),
        _make_crop_panel(processed, result),
    ])
    review = np.vstack([top, bottom])
    header = np.full((86, review.shape[1], 3), 32, dtype=np.uint8)
    title = f"{summary.get('image_name')} | {summary.get('vhole_method')} | elapsed {summary.get('elapsed_sec')}s"
    cv2.putText(header, title[:140], (24, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.88, (255, 255, 255), 2, cv2.LINE_AA)
    detail = f"mesh_r={summary.get('mesh_radius_px')} pair_len={summary.get('vhole_pair_distance_px')} error={summary.get('vhole_error') or '-'}"
    cv2.putText(header, detail[:160], (24, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (210, 230, 255), 1, cv2.LINE_AA)
    return np.vstack([header, review])


def write_review_image(payload: dict[str, Any], output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    review = make_review_image(payload)
    ok = cv2.imwrite(str(output_path), review, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise RuntimeError(f"Failed to write review image: {output_path}")
