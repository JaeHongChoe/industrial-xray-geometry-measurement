#!/usr/bin/env python3
"""Run V-hole inference on raw images using the full-data V7 HV export.

Raw images are first converted with ``preprocess_image``. The resulting HV image
is cropped around the detected mesh center into an HV-crop-like image, then the
existing V7 export model is applied.
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

import run_coarse_fine_v7 as v7  # noqa: E402
from xray_3pin_detector import detect_circles, preprocess_image  # noqa: E402


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _safe_name(path: Path, root: Path) -> str:
    rel = path.relative_to(root).with_suffix("")
    text = "__".join(rel.parts)
    text = re.sub(r"[^0-9A-Za-z가-힣_.-]+", "_", text)
    return text.strip("_")


def _known_hv_size(sample: str) -> tuple[int, int] | None:
    candidates = [
        BASE_DIR / "vhole_hv_crops" / f"{sample}_hv.png",
        BASE_DIR / "vhole_v7_data_hv" / f"{sample}_orig_hv.png",
    ]
    for path in candidates:
        if not path.exists():
            continue
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            h, w = img.shape[:2]
            return w, h
    return None


def _extract_with_padding(img: np.ndarray, x1: int, y1: int, width: int, height: int) -> tuple[np.ndarray, dict]:
    h, w = img.shape[:2]
    x2 = x1 + width
    y2 = y1 + height
    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - w)
    pad_bottom = max(0, y2 - h)

    if pad_left or pad_top or pad_right or pad_bottom:
        img = cv2.copyMakeBorder(
            img,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_REFLECT,
        )
        x1 += pad_left
        y1 += pad_top

    crop = img[y1:y1 + height, x1:x1 + width]
    if crop.shape[0] != height or crop.shape[1] != width:
        crop = cv2.resize(crop, (width, height), interpolation=cv2.INTER_LINEAR)

    return crop, {
        "pad_left": pad_left,
        "pad_top": pad_top,
        "pad_right": pad_right,
        "pad_bottom": pad_bottom,
    }


def _crop_hv_like(raw_path: Path, hv_crop_mode: str = "auto") -> tuple[np.ndarray, dict]:
    gray, _morph, hv = preprocess_image(str(raw_path))
    plate_circle, mesh_circle = detect_circles(gray, hv)
    sample = raw_path.stem
    known_size = _known_hv_size(sample)

    if hv_crop_mode == "full" or (hv_crop_mode == "auto" and known_size is None):
        target_w = int(hv.shape[1])
        target_h = int(hv.shape[0])
        x1 = y1 = 0
        crop = hv.copy()
        pad = {
            "pad_left": 0,
            "pad_top": 0,
            "pad_right": 0,
            "pad_bottom": 0,
        }
        crop_source = "full_hv"
    elif known_size is None:
        # Forced crop mode for experiments. Service auto mode should keep unknown
        # products as full HV so side V-holes are not clipped.
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


def _load_v7_export(model_dir: Path, device: torch.device):
    from ultralytics import YOLO

    yolo_path = model_dir / "yolo_best.pt"
    tip_path = model_dir / "tip_model.pth"
    if not yolo_path.exists():
        raise FileNotFoundError(yolo_path)
    if not tip_path.exists():
        raise FileNotFoundError(tip_path)

    yolo_model = YOLO(str(yolo_path))
    tip_model = v7.TrianglePoseEstimatorV7(freeze_encoder=True).to(device)
    try:
        state = torch.load(str(tip_path), map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(str(tip_path), map_location=device)
    tip_model.load_state_dict(state)
    tip_model.eval()
    device_str = "0" if device.type == "cuda" else device.type
    return yolo_model, tip_model, device_str


def _draw_overlay(img: np.ndarray, preds: list[dict], coarse_locs, out_path: Path) -> None:
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for cx, cy in coarse_locs or []:
        cv2.circle(vis, (int(round(cx)), int(round(cy))), 14, (255, 130, 0), 2, cv2.LINE_AA)
    for idx, pred in enumerate(preds):
        tip = pred["tip"]
        raw_tip = pred.get("raw_tip", tip)
        c1 = pred.get("corner1", tip)
        c2 = pred.get("corner2", tip)
        cv2.line(vis, (int(round(c1[0])), int(round(c1[1]))), (int(round(c2[0])), int(round(c2[1]))), (255, 90, 90), 2, cv2.LINE_AA)
        cv2.line(vis, (int(round(tip[0])), int(round(tip[1]))), (int(round(c1[0])), int(round(c1[1]))), (255, 90, 90), 1, cv2.LINE_AA)
        cv2.line(vis, (int(round(tip[0])), int(round(tip[1]))), (int(round(c2[0])), int(round(c2[1]))), (255, 90, 90), 1, cv2.LINE_AA)
        cv2.circle(vis, (int(round(raw_tip[0])), int(round(raw_tip[1]))), 7, (255, 0, 255), 2, cv2.LINE_AA)
        cv2.drawMarker(vis, (int(round(tip[0])), int(round(tip[1]))), (0, 0, 255), cv2.MARKER_CROSS, 26, 2)
        cv2.putText(vis, f"V{idx + 1}", (int(round(tip[0])) + 10, int(round(tip[1])) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imwrite(str(out_path), vis)


def _yolo_candidate_boxes(
    img: np.ndarray,
    yolo_model,
    device_str: str,
    *,
    top_k: int,
    conf: float,
) -> list[dict]:
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    preds_raw = yolo_model.predict(
        img_rgb,
        imgsz=v7.YOLO_IMGSZ,
        conf=conf,
        device=device_str,
        verbose=False,
    )
    if len(preds_raw) == 0 or preds_raw[0].boxes is None:
        return []

    boxes = preds_raw[0].boxes
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    h, w = img.shape[:2]
    img_cx, img_cy = w / 2.0, h / 2.0

    candidates = []
    order = np.argsort(-confs)[:top_k]
    for idx in order:
        x1, y1, x2, y2 = [float(v) for v in xyxy[idx]]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        candidates.append({
            "source": "yolo",
            "center": (cx, cy),
            "box": (x1, y1, x2, y2),
            "box_conf": float(confs[idx]),
            "angle": math.atan2(cy - img_cy, cx - img_cx),
            "radius": math.hypot(cx - img_cx, cy - img_cy),
        })
    return candidates


def _radial_candidate_boxes(meta: dict, img_shape: tuple[int, int]) -> list[dict]:
    """Add deterministic ring candidates from the detected mesh circle.

    This is useful when YOLO misses side V-holes, especially TYPE_E where the real
    targets are near 3/9 o'clock but short notches near 12 o'clock are easier
    for YOLO to detect.
    """
    mesh = meta.get("mesh_circle", {})
    if not mesh:
        return []
    cx = float(mesh["x"]) - float(meta["crop_x1"])
    cy = float(mesh["y"]) - float(meta["crop_y1"])
    radius = float(mesh["r"])
    h, w = img_shape[:2]
    out = []
    for frac in (0.80, 0.88, 0.95, 1.00, 1.05):
        for deg in (0.0, 90.0, 180.0, 270.0):
            ang = math.radians(deg)
            px = cx + math.cos(ang) * radius * frac
            py = cy + math.sin(ang) * radius * frac
            # Keep a bit of tolerance because V-hole tips can sit near the crop edge.
            if px < -128 or py < -128 or px > w + 128 or py > h + 128:
                continue
            out.append({
                "source": "radial",
                "center": (px, py),
                "box": (px - 60.0, py - 60.0, px + 60.0, py + 60.0),
                "box_conf": 0.05,
                "angle": math.atan2(py - h / 2.0, px - w / 2.0),
                "radius": math.hypot(px - w / 2.0, py - h / 2.0),
                "radial_angle_deg": deg,
                "radial_fraction": frac,
            })
    return out


def _triangle_metrics(tip, corner1, corner2, img_shape) -> dict:
    c1 = np.array(corner1, dtype=np.float32)
    c2 = np.array(corner2, dtype=np.float32)
    t = np.array(tip, dtype=np.float32)
    base_vec = c2 - c1
    base_len = float(np.linalg.norm(base_vec))
    if base_len < 1e-6:
        height = 0.0
    else:
        height = float(abs(np.cross(base_vec, t - c1)) / base_len)
    area = 0.5 * base_len * height
    h, w = img_shape[:2]
    angle = math.atan2(float(t[1]) - h / 2.0, float(t[0]) - w / 2.0)
    radius = math.hypot(float(t[0]) - w / 2.0, float(t[1]) - h / 2.0)
    return {
        "base_len": base_len,
        "height": height,
        "area": float(area),
        "angle": angle,
        "radius": float(radius),
    }


def _candidate_quality(pred: dict, box_conf: float, metrics: dict) -> float:
    """Score one candidate using confidence plus triangle size/shape quality.

    TYPE_E side V-holes are usually recovered by radial candidate crops, not YOLO.
    Some radial crops around the outer ring produce large but meaningless
    triangles, so the fine model's tip confidence must dominate the geometric
    shape terms.
    """
    tip_conf = float(pred.get("tip_confidence", 0.0))
    base = metrics["base_len"]
    height = metrics["height"]
    area = metrics["area"]

    base_score = float(np.clip((base - 18.0) / 55.0, 0.0, 1.0))
    height_score = float(np.clip((height - 18.0) / 45.0, 0.0, 1.0))
    area_score = float(np.clip((area - 350.0) / 1800.0, 0.0, 1.0))
    score = 2.20 * tip_conf + 0.35 * box_conf + 0.55 * base_score + 0.75 * height_score + 0.45 * area_score

    # Very shallow/short triangles are often the small 12 o'clock notches in TYPE_E.
    if height < 22.0:
        score -= 0.90
    if area < 450.0:
        score -= 0.55
    if base < 18.0:
        score -= 0.40
    if tip_conf < 0.15:
        score -= 1.20
    if tip_conf < 0.05:
        score -= 1.30
    return float(score)


def _angle_gap(a: float, b: float) -> float:
    return abs(((a - b + math.pi) % (2.0 * math.pi)) - math.pi)


def _axis_alignment(angle: float, axis_prior: str) -> float:
    if axis_prior == "horizontal":
        return abs(math.cos(angle))
    if axis_prior == "vertical":
        return abs(math.sin(angle))
    return 1.0


def _axis_side(angle: float, axis_prior: str) -> float:
    if axis_prior == "horizontal":
        return math.cos(angle)
    if axis_prior == "vertical":
        return math.sin(angle)
    return 0.0


def predict_tips_geometry(
    tip_model,
    img: np.ndarray,
    yolo_model,
    device_str: str,
    *,
    device: torch.device,
    top_k: int,
    yolo_conf: float,
    min_angle_gap_deg: float,
    crop_meta: dict | None = None,
    axis_prior: str = "none",
    axis_min_alignment: float = 0.55,
) -> tuple[list[dict], list[tuple[float, float]], list[dict]]:
    """Try many YOLO candidates, then select two by triangle quality.

    This avoids selecting two short V-notches in the same angular sector.
    """
    boxes = _yolo_candidate_boxes(
        img,
        yolo_model,
        device_str,
        top_k=top_k,
        conf=yolo_conf,
    )
    if crop_meta is not None:
        boxes.extend(_radial_candidate_boxes(crop_meta, img.shape))
    if not boxes:
        preds, coarse = v7.predict_tips(tip_model, img, yolo_model, device_str, device=device)
        return preds, coarse, []

    tip_model.eval()
    candidates = []
    with torch.no_grad():
        for box in boxes:
            coarse_x, coarse_y = box["center"]
            patch = v7.extract_patch(img, coarse_x, coarse_y, v7.PATCH_SIZE)
            patch_norm = v7.normalize_fine_patch(patch)
            patch_t = torch.from_numpy(patch_norm).unsqueeze(0).unsqueeze(0).to(device)
            heatmap_3ch = tip_model(patch_t)[0].cpu().numpy()
            points, confs = v7.heatmap_to_coords(heatmap_3ch, coarse_x, coarse_y)
            raw_tip = points[0]
            corner1 = points[1]
            corner2 = points[2]
            refined_tip = v7.maybe_refine_tip(raw_tip, corner1, corner2, confs)
            pred = {
                "tip": refined_tip,
                "raw_tip": raw_tip,
                "corner1": corner1,
                "corner2": corner2,
                "tip_confidence": float(confs[0]),
                "box_conf": float(box["box_conf"]),
            }
            metrics = _triangle_metrics(refined_tip, corner1, corner2, img.shape)
            quality = _candidate_quality(pred, float(box["box_conf"]), metrics)
            candidates.append({
                "pred": pred,
                "source": box.get("source", "unknown"),
                "coarse": box["center"],
                "box": box["box"],
                "box_conf": float(box["box_conf"]),
                "quality": quality,
                "metrics": metrics,
            })

    # Candidate-level NMS by predicted tip proximity.
    candidates.sort(key=lambda row: row["quality"], reverse=True)
    kept = []
    for row in candidates:
        tip = row["pred"]["tip"]
        if all(math.hypot(tip[0] - k["pred"]["tip"][0], tip[1] - k["pred"]["tip"][1]) >= 35.0 for k in kept):
            kept.append(row)
        if len(kept) >= min(top_k, 12):
            break

    min_gap = math.radians(min_angle_gap_deg)
    best_pair = None
    best_pair_score = -1e9

    if axis_prior != "none":
        neg_side = []
        pos_side = []
        for row in kept:
            angle = row["metrics"]["angle"]
            alignment = _axis_alignment(angle, axis_prior)
            if alignment < axis_min_alignment:
                continue
            side = _axis_side(angle, axis_prior)
            if side < 0:
                neg_side.append(row)
            elif side > 0:
                pos_side.append(row)
        if neg_side and pos_side:
            best_pair = (
                max(neg_side, key=lambda row: row["quality"]),
                max(pos_side, key=lambda row: row["quality"]),
            )
            best_pair_score = best_pair[0]["quality"] + best_pair[1]["quality"]

    if best_pair is None:
        for i in range(len(kept)):
            for j in range(i + 1, len(kept)):
                a = kept[i]["metrics"]["angle"]
                b = kept[j]["metrics"]["angle"]
                gap = _angle_gap(a, b)
                if gap < min_gap:
                    continue
                axis_i = _axis_alignment(a, axis_prior)
                axis_j = _axis_alignment(b, axis_prior)
                axis_pair = (axis_i + axis_j) / 2.0
                if axis_prior != "none" and (axis_i < axis_min_alignment or axis_j < axis_min_alignment):
                    continue
                gap_score = float(np.clip((gap - min_gap) / max(math.pi - min_gap, 1e-6), 0.0, 1.0))
                axis_bonus = 2.50 * axis_pair if axis_prior != "none" else 0.0
                pair_score = kept[i]["quality"] + kept[j]["quality"] + 0.35 * gap_score + axis_bonus
                if pair_score > best_pair_score:
                    best_pair_score = pair_score
                    best_pair = (kept[i], kept[j])

    if best_pair is None:
        best_pair = tuple(kept[:2]) if len(kept) >= 2 else tuple(kept)

    selected = list(best_pair)
    preds = [row["pred"] for row in selected]
    coarse = [row["coarse"] for row in selected]
    debug = []
    for rank, row in enumerate(kept, start=1):
        debug.append({
            "rank": rank,
            "source": row.get("source", "unknown"),
            "selected": row in selected,
            "coarse": [float(row["coarse"][0]), float(row["coarse"][1])],
            "tip": [float(row["pred"]["tip"][0]), float(row["pred"]["tip"][1])],
            "box_conf": float(row["box_conf"]),
            "tip_confidence": float(row["pred"].get("tip_confidence", 0.0)),
            "quality": float(row["quality"]),
            "base_len": float(row["metrics"]["base_len"]),
            "height": float(row["metrics"]["height"]),
            "area": float(row["metrics"]["area"]),
            "angle_deg": float(math.degrees(row["metrics"]["angle"])),
            "radius": float(row["metrics"]["radius"]),
        })
    return preds, coarse, debug


def run(
    raw_dir: Path,
    model_dir: Path,
    out_dir: Path,
    device: torch.device,
    *,
    candidate_mode: str,
    yolo_top_k: int,
    yolo_conf: float,
    min_angle_gap_deg: float,
    axis_prior: str,
    axis_min_alignment: float,
    hv_crop_mode: str,
) -> None:
    crop_dir = out_dir / "hv_crops"
    vis_dir = out_dir / "visualizations"
    crop_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(p for p in raw_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    yolo_model, tip_model, device_str = _load_v7_export(model_dir, device)

    rows = []
    started = time.perf_counter()
    print("=" * 80)
    print(" Raw Data V-hole inference with V7 HV export")
    print(f" raw:   {raw_dir}")
    print(f" model: {model_dir}")
    print(f" out:   {out_dir}")
    print(f" crop:  {hv_crop_mode}")
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
            crop, meta = _crop_hv_like(raw_path, hv_crop_mode=hv_crop_mode)
            name = _safe_name(raw_path, raw_dir)
            crop_path = crop_dir / f"{name}_hv.png"
            vis_path = vis_dir / f"{name}_pred.png"
            cv2.imwrite(str(crop_path), crop)

            candidate_debug = []
            if candidate_mode == "geometry":
                sample_axis_prior = axis_prior
                if axis_prior == "auto":
                    # Product names are not available in service. Keep auto
                    # product-agnostic; use explicit horizontal/vertical only
                    # for offline diagnostics.
                    sample_axis_prior = "none"
                preds, coarse, candidate_debug = predict_tips_geometry(
                    tip_model,
                    crop,
                    yolo_model,
                    device_str,
                    device=device,
                    top_k=yolo_top_k,
                    yolo_conf=yolo_conf,
                    min_angle_gap_deg=min_angle_gap_deg,
                    crop_meta=meta,
                    axis_prior=sample_axis_prior,
                    axis_min_alignment=axis_min_alignment,
                )
            else:
                sample_axis_prior = "none"
                preds, coarse = v7.predict_tips(tip_model, crop, yolo_model, device_str, device=device)
            _draw_overlay(crop, preds, coarse, vis_path)

            xoff = float(meta["crop_x1"])
            yoff = float(meta["crop_y1"])
            pred_rows = []
            for pred in preds:
                tip_x, tip_y = pred["tip"]
                raw_tip_x, raw_tip_y = pred.get("raw_tip", pred["tip"])
                pred_rows.append({
                    "x_crop": float(tip_x),
                    "y_crop": float(tip_y),
                    "x_raw": float(tip_x + xoff),
                    "y_raw": float(tip_y + yoff),
                    "raw_x_crop": float(raw_tip_x),
                    "raw_y_crop": float(raw_tip_y),
                    "confidence": float(pred.get("tip_confidence", 0.0)),
                })

            row.update({
                "ok": True,
                "crop_path": str(crop_path),
                "visualization": str(vis_path),
                "crop_meta": meta,
                "coarse_locs": [[float(x), float(y)] for x, y in (coarse or [])],
                "candidate_mode": candidate_mode,
                "axis_prior": sample_axis_prior,
                "candidate_debug": candidate_debug,
                "predictions": pred_rows,
            })
            tips = [(round(p["x_crop"], 1), round(p["y_crop"], 1)) for p in pred_rows]
            print(f"[{idx:02d}/{len(image_paths):02d}] {raw_path.stem:18s} tips={tips} {meta['crop_source']}")
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            print(f"[{idx:02d}/{len(image_paths):02d}] {raw_path.stem:18s} ERROR {row['error']}")
        rows.append(row)

    summary = {
        "raw_dir": str(raw_dir),
        "model_dir": str(model_dir),
        "out_dir": str(out_dir),
        "num_images": len(image_paths),
        "num_ok": sum(1 for row in rows if row["ok"]),
        "num_failed": sum(1 for row in rows if not row["ok"]),
        "hv_crop_mode": hv_crop_mode,
        "elapsed_sec": time.perf_counter() - started,
        "results": rows,
    }
    (out_dir / "predictions.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("=" * 80)
    print(f"ok={summary['num_ok']}/{summary['num_images']} failed={summary['num_failed']}")
    print(f"json={out_dir / 'predictions.json'}")
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=BASE_DIR / "preprocessing_module" / "raw_data")
    parser.add_argument("--model-dir", type=Path, default=BASE_DIR / "v7_hv_model_export")
    parser.add_argument("--out-dir", type=Path, default=BASE_DIR / "preprocessing_module" / "vhole_raw_v7_results")
    parser.add_argument("--device", default=None)
    parser.add_argument("--candidate-mode", choices=["legacy", "geometry"], default="geometry")
    parser.add_argument("--yolo-top-k", type=int, default=20)
    parser.add_argument("--yolo-conf", type=float, default=0.01)
    parser.add_argument("--min-angle-gap-deg", type=float, default=45.0)
    parser.add_argument("--axis-prior", choices=["auto", "none", "horizontal", "vertical"], default="auto")
    parser.add_argument("--axis-min-alignment", type=float, default=0.55)
    parser.add_argument("--hv-crop-mode", choices=["auto", "crop", "full"], default="auto")
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = v7.DEVICE
    run(
        args.raw_dir,
        args.model_dir,
        args.out_dir,
        device,
        candidate_mode=args.candidate_mode,
        yolo_top_k=args.yolo_top_k,
        yolo_conf=args.yolo_conf,
        min_angle_gap_deg=args.min_angle_gap_deg,
        axis_prior=args.axis_prior,
        axis_min_alignment=args.axis_min_alignment,
        hv_crop_mode=args.hv_crop_mode,
    )


if __name__ == "__main__":
    main()
