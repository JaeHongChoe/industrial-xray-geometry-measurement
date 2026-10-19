"""End-to-end product inference pipeline.

This module is the clean entry point for Python inference:

1. image load and preprocessing
2. plate/mesh circle detection
3. 3-pin and JIG-cross detection
4. V-hole inference with the V7 ONNX detector
5. JSON-friendly result packaging
"""

from __future__ import annotations

import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .circles import compute_eccentricity, detect_circles
from .detection import detect_3pins, detect_jig_cross
from .preprocess import _build_circle_roi_mask, load_grayscale, preprocess_image
from .products import classify_group
from .vhole_v7 import V7ONNXDetector, VHoleResult


def circle_to_dict(circle: Any | None) -> dict[str, float] | None:
    if circle is None:
        return None
    return {
        "center_x_px": round(float(circle.center_x_px), 6),
        "center_y_px": round(float(circle.center_y_px), 6),
        "radius_px": round(float(circle.radius_px), 6),
    }


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {
            str(key): _jsonable(item)
            for key, item in value.items()
            if not isinstance(item, np.ndarray)
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value if not isinstance(item, np.ndarray)]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _summarize_pins(pin_result: dict[str, Any] | None) -> dict[str, Any]:
    if not pin_result:
        return {"status": "fail", "count": 0}
    pins = []
    for item in pin_result.get("top3", []):
        pins.append({
            "cx": round(float(item.get("cx", 0.0)), 3),
            "cy": round(float(item.get("cy", 0.0)), 3),
            "area": int(item.get("area", 0)),
            "radius": round(float(item.get("radius", 0.0)), 3),
            "circularity": round(float(item.get("circularity", 0.0)), 4),
        })
    return {
        "status": "ok" if len(pins) == 3 else "partial",
        "count": len(pins),
        "pins": pins,
        "min_gap_deg": round(float(pin_result.get("min_gap", 0.0)), 3),
        "threshold_pct": pin_result.get("thresh_pct"),
        "blur_k": pin_result.get("blur_k"),
        "n_candidates": pin_result.get("n_candidates"),
        "circumscribed": circle_to_dict(pin_result.get("circumscribed")),
    }


def _summarize_jig(jig_result: dict[str, Any] | None) -> dict[str, Any]:
    if not jig_result:
        return {"status": "fail", "has_jig": False}
    status = str(jig_result.get("jig_status") or ("ok" if bool(jig_result.get("has_jig")) else "not_detected"))
    return {
        "status": status,
        "has_jig": bool(jig_result.get("has_jig")),
        "jig_x": round(float(jig_result.get("jig_x", 0.0)), 3),
        "jig_y": round(float(jig_result.get("jig_y", 0.0)), 3),
        "confidence": round(float(jig_result.get("confidence", 0.0)), 4),
        "line_connection_status": jig_result.get("line_connection_status"),
        "reasons": _jsonable(jig_result.get("reasons", [])),
        "side_coords": _jsonable(jig_result.get("side_coords", {})),
        "candidate_counts": _jsonable(jig_result.get("candidate_counts", {})),
        "method": jig_result.get("method"),
        "n_lines": int(jig_result.get("n_lines", 0)),
        "total_len": round(float(jig_result.get("total_len", 0.0)), 3),
        "cross_angles": _jsonable(jig_result.get("cross_angles", [])),
        "cross_lines": _jsonable(jig_result.get("cross_lines", [])),
    }


def infer_image(
    image_path: Path,
    detector: V7ONNXDetector,
    pixel_size_mm: float = 0.030769,
) -> dict[str, Any]:
    """Run the full inspection logic for one image."""
    image_path = Path(image_path)
    started = time.perf_counter()

    raw_gray = load_grayscale(image_path)
    processed_gray, morph, hv = preprocess_image(image_path)
    plate_circle, mesh_circle = detect_circles(processed_gray, hv)
    roi_mask = _build_circle_roi_mask(processed_gray.shape, plate_circle)
    product_group = classify_group(image_path.stem)

    eccentricity = compute_eccentricity(plate_circle, mesh_circle, pixel_size_mm)
    pin_result = detect_3pins(hv, mesh_circle)
    jig_result = detect_jig_cross(processed_gray, hv, plate_circle, mesh_circle)

    vhole_error = None
    try:
        vhole_result: VHoleResult | None = detector.detect(hv, mesh_circle, product_group)
        vhole_payload = vhole_result.to_dict()
        vhole_status = str(vhole_payload.get("status", "fail"))
        if vhole_status != "ok":
            detection = vhole_payload.get("detection", {})
            if isinstance(detection, dict):
                vhole_error = (
                    detection.get("horizontal_error")
                    or detection.get("error")
                    or detection.get("normal_reject_reason")
                )
            vhole_error = vhole_error or vhole_status
    except Exception as exc:
        vhole_error = str(exc)
        vhole_payload = {
            "status": "fail",
            "method": "v7_hv_yolo_tip_onnx",
            "error": vhole_error,
        }

    elapsed = time.perf_counter() - started
    pin_summary = _summarize_pins(pin_result)
    jig_summary = _summarize_jig(jig_result)
    summary = {
        "image_name": image_path.name,
        "image_path": str(image_path),
        "dataset": image_path.parent.name,
        "product_group": product_group,
        "status": "ok" if vhole_payload.get("status") == "ok" else "vhole_fail",
        "elapsed_sec": round(float(elapsed), 4),
        "plate_radius_px": round(float(plate_circle.radius_px), 3),
        "mesh_radius_px": round(float(mesh_circle.radius_px), 3),
        "plate_mesh_dx_px": eccentricity.get("delta_x_px"),
        "plate_mesh_dy_px": eccentricity.get("delta_y_px"),
        "plate_mesh_distance_px": eccentricity.get("distance_px"),
        "vhole_method": vhole_payload.get("method"),
        "vhole_axis": vhole_payload.get("nominal_axis"),
        "vhole_rotation_to_axis_deg": (
            round(float(vhole_payload["rotation_to_nominal_deg"]), 3)
            if vhole_payload.get("rotation_to_nominal_deg") is not None else None
        ),
        "vhole_pair_score": (
            round(float(vhole_payload["pair"]["pair_score"]), 6)
            if isinstance(vhole_payload.get("pair"), dict) else None
        ),
        "vhole_pair_distance_px": (
            vhole_payload.get("detection", {}).get("pair_distance_px")
            if isinstance(vhole_payload.get("detection"), dict) else None
        ),
        "vhole_error": vhole_error,
        "pin_status": pin_summary["status"],
        "pin_count": pin_summary["count"],
        "jig_status": jig_summary["status"],
        "jig_has_jig": jig_summary["has_jig"],
        "jig_line_connection_status": jig_summary.get("line_connection_status"),
        "jig_confidence": jig_summary.get("confidence"),
    }

    return {
        "summary": summary,
        "arrays": {
            "raw_gray": raw_gray,
            "processed_gray": processed_gray,
            "morph": morph,
            "hv": hv,
            "roi_mask": roi_mask,
        },
        "circles": {
            "plate": plate_circle,
            "mesh": mesh_circle,
        },
        "result": {
            "image": {
                "name": image_path.name,
                "path": str(image_path),
                "dataset": image_path.parent.name,
                "shape": list(raw_gray.shape[:2]),
            },
            "product_group": product_group,
            "scale": {"pixel_size_mm": pixel_size_mm},
            "plate_circle": circle_to_dict(plate_circle),
            "mesh_circle": circle_to_dict(mesh_circle),
            "eccentricity": _jsonable(eccentricity),
            "three_pin": pin_summary,
            "jig": jig_summary,
            "vhole": _jsonable(vhole_payload),
            "elapsed_sec": round(float(elapsed), 4),
        },
    }
