#!/usr/bin/env python3
"""Verify V7 HV ONNX inference with the browser-target postprocessing path."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


PATCH_SIZE = 192
HM_SIZE = 96
YOLO_IMGSZ = 1280
YOLO_CONF = 0.05
YOLO_TOP_K = 6
TOP_PAIRS = 3
REFINE_ALPHA = 0.3
REFINE_MIN_CORNER_CONF = 0.25
REFINE_MIN_BASE = 8.0
REFINE_MAX_SHIFT = 12.0
REFINE_MIN_ANGLE_DEG = 15.0
REFINE_MAX_ANGLE_DEG = 165.0


def default_data_dir() -> Path:
    candidates = [
        Path("vhole_v7_data_hv"),
        Path("vhole_hv_crops"),
        Path(__file__).resolve().parents[1] / "vhole_hv_crops",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def extract_patch(img: np.ndarray, cx: float, cy: float, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    half = size // 2
    x1, y1 = int(cx) - half, int(cy) - half
    x2, y2 = x1 + size, y1 + size
    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - w)
    pad_bottom = max(0, y2 - h)
    if pad_left or pad_top or pad_right or pad_bottom:
        img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT)
        x1 += pad_left
        y1 += pad_top
    patch = img[y1 : y1 + size, x1 : x1 + size]
    if patch.shape[0] != size or patch.shape[1] != size:
        patch = cv2.resize(patch, (size, size))
    return patch


def normalize_fine_patch(patch: np.ndarray) -> np.ndarray:
    patch_u8 = patch.astype(np.uint8, copy=False)
    lo, hi = np.percentile(patch_u8, (1.0, 99.0))
    if hi > lo + 1.0:
        patch_f = (patch_u8.astype(np.float32) - lo) * (255.0 / (hi - lo))
        patch_u8 = np.clip(patch_f, 0, 255).astype(np.uint8)
    return patch_u8.astype(np.float32) / 255.0


def get_vhole_keypoints(json_path: Path, img_shape: tuple[int, int]) -> list[dict]:
    if not json_path.exists():
        return []
    data = json.loads(json_path.read_text(encoding="utf-8"))
    h, w = img_shape[:2]
    img_cx, img_cy = w / 2, h / 2
    keypoints = []
    for shape in data.get("shapes", []):
        if shape.get("label") != "v":
            continue
        pts = shape.get("points", [])
        if len(pts) < 3:
            continue
        tip_idx = min(range(len(pts)), key=lambda i: math.hypot(pts[i][0] - img_cx, pts[i][1] - img_cy))
        tip = (float(pts[tip_idx][0]), float(pts[tip_idx][1]))
        corners = [(float(pts[i][0]), float(pts[i][1])) for i in range(len(pts)) if i != tip_idx]
        corners.sort(key=lambda c: math.atan2(c[1] - tip[1], c[0] - tip[0]))
        keypoints.append({"tip": tip, "corner1": corners[0], "corner2": corners[1] if len(corners) > 1 else corners[0]})
    return keypoints


def letterbox_gray_to_rgb_tensor(img: np.ndarray, new_shape: int = YOLO_IMGSZ) -> tuple[np.ndarray, float, tuple[float, float]]:
    h, w = img.shape[:2]
    scale = min(new_shape / h, new_shape / w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_shape, new_shape), 114, dtype=np.uint8)
    pad_x = (new_shape - new_w) / 2.0
    pad_y = (new_shape - new_h) / 2.0
    left = int(round(pad_x - 0.1))
    top = int(round(pad_y - 0.1))
    canvas[top : top + new_h, left : left + new_w] = resized
    rgb = np.stack([canvas, canvas, canvas], axis=0).astype(np.float32) / 255.0
    return rgb[None, :, :, :], scale, (float(left), float(top))


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    out = np.empty_like(boxes)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    return out


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.7, max_det: int = 300) -> list[int]:
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = np.argsort(-scores)
    keep = []
    while order.size > 0 and len(keep) < max_det:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter + 1e-9
        order = rest[inter / union <= iou_threshold]
    return keep


def yolo_coarse_detect_onnx(img: np.ndarray, session: ort.InferenceSession) -> list[dict]:
    tensor, scale, (pad_x, pad_y) = letterbox_gray_to_rgb_tensor(img)
    output = session.run(None, {session.get_inputs()[0].name: tensor})[0][0]
    boxes_xywh = output[:4, :].T
    scores = output[4, :]
    mask = scores >= YOLO_CONF
    boxes_xywh = boxes_xywh[mask]
    scores = scores[mask]
    if len(scores) == 0:
        return []

    boxes_xyxy = xywh_to_xyxy(boxes_xywh)
    keep = nms(boxes_xyxy, scores, iou_threshold=0.7)
    boxes_xyxy = boxes_xyxy[keep]
    scores = scores[keep]

    order = np.argsort(-scores)[:YOLO_TOP_K]
    h, w = img.shape[:2]
    img_cx, img_cy = w / 2.0, h / 2.0
    all_boxes = []
    for i in order:
        x1, y1, x2, y2 = boxes_xyxy[i]
        cx = ((float(x1) + float(x2)) / 2.0 - pad_x) / scale
        cy = ((float(y1) + float(y2)) / 2.0 - pad_y) / scale
        angle = math.atan2(cy - img_cy, cx - img_cx)
        radius = math.hypot(cx - img_cx, cy - img_cy)
        all_boxes.append({"center": (cx, cy), "conf": float(scores[i]), "angle": angle, "radius": radius})

    angle_tol = math.radians(30.0)
    candidates = []
    for i in range(len(all_boxes)):
        for j in range(i + 1, len(all_boxes)):
            b1, b2 = all_boxes[i], all_boxes[j]
            angle_gap = abs(((b1["angle"] - b2["angle"] + math.pi) % (2 * math.pi)) - math.pi)
            angle_err = abs(math.pi - angle_gap)
            if angle_err > angle_tol:
                continue
            r_gap = abs(b1["radius"] - b2["radius"])
            if r_gap > max(b1["radius"], b2["radius"]) * 0.3:
                continue
            signal_score = b1["conf"] + b2["conf"]
            pair_score = signal_score - 3.0 * angle_err / math.pi
            candidates.append({"pair": [b1["center"], b2["center"]], "signal_score": signal_score, "key": (pair_score, signal_score)})
    if not candidates and len(all_boxes) >= 2:
        b1, b2 = all_boxes[0], all_boxes[1]
        signal_score = b1["conf"] + b2["conf"]
        candidates.append({"pair": [b1["center"], b2["center"]], "signal_score": signal_score, "key": (signal_score, signal_score)})
    candidates.sort(key=lambda x: x["key"], reverse=True)
    return candidates[:TOP_PAIRS]


def get_heatmap_peak(heatmap_2d: np.ndarray) -> tuple[float, float, float]:
    hm_h, hm_w = heatmap_2d.shape
    peak_y, peak_x = np.unravel_index(np.argmax(heatmap_2d), heatmap_2d.shape)
    peak_y, peak_x = int(peak_y), int(peak_x)
    peak_val = float(heatmap_2d[peak_y, peak_x])
    dx = dy = 0.0
    if 1 <= peak_x < hm_w - 1:
        left, right, center = float(heatmap_2d[peak_y, peak_x - 1]), float(heatmap_2d[peak_y, peak_x + 1]), float(heatmap_2d[peak_y, peak_x])
        denom = 2.0 * (left + right - 2.0 * center)
        if abs(denom) > 1e-6:
            dx = float(np.clip((left - right) / denom, -0.5, 0.5))
    if 1 <= peak_y < hm_h - 1:
        top, bottom, center = float(heatmap_2d[peak_y - 1, peak_x]), float(heatmap_2d[peak_y + 1, peak_x]), float(heatmap_2d[peak_y, peak_x])
        denom = 2.0 * (top + bottom - 2.0 * center)
        if abs(denom) > 1e-6:
            dy = float(np.clip((top - bottom) / denom, -0.5, 0.5))
    return peak_x + dx, peak_y + dy, peak_val


def heatmap_to_coords(heatmap_3ch: np.ndarray, coarse_x: float, coarse_y: float) -> tuple[list[tuple[float, float]], list[float]]:
    points = []
    confidences = []
    for ch in range(3):
        hm_x, hm_y, conf = get_heatmap_peak(heatmap_3ch[ch])
        orig_x = coarse_x - PATCH_SIZE / 2.0 + hm_x / HM_SIZE * PATCH_SIZE
        orig_y = coarse_y - PATCH_SIZE / 2.0 + hm_y / HM_SIZE * PATCH_SIZE
        points.append((orig_x, orig_y))
        confidences.append(conf)
    return points, confidences


def refine_tip_from_triangle(tip: tuple[float, float], corner1: tuple[float, float], corner2: tuple[float, float]) -> tuple[float, float]:
    mid_x = (corner1[0] + corner2[0]) / 2.0
    mid_y = (corner1[1] + corner2[1]) / 2.0
    base_dx = corner2[0] - corner1[0]
    base_dy = corner2[1] - corner1[1]
    base_len = math.hypot(base_dx, base_dy)
    if base_len < 1e-6:
        return tip
    perp_dx = -base_dy / base_len
    perp_dy = base_dx / base_len
    t = (tip[0] - mid_x) * perp_dx + (tip[1] - mid_y) * perp_dy
    geo_x = mid_x + t * perp_dx
    geo_y = mid_y + t * perp_dy
    return (REFINE_ALPHA * geo_x + (1 - REFINE_ALPHA) * tip[0], REFINE_ALPHA * geo_y + (1 - REFINE_ALPHA) * tip[1])


def triangle_angle_at_tip(tip: tuple[float, float], corner1: tuple[float, float], corner2: tuple[float, float]) -> float:
    v1 = (corner1[0] - tip[0], corner1[1] - tip[1])
    v2 = (corner2[0] - tip[0], corner2[1] - tip[1])
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.hypot(*v1)
    mag2 = math.hypot(*v2)
    if mag1 < 1e-6 or mag2 < 1e-6:
        return 0.0
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / (mag1 * mag2)))))


def maybe_refine_tip(raw_tip, corner1, corner2, confs):
    if min(confs[1], confs[2]) < REFINE_MIN_CORNER_CONF:
        return raw_tip
    if math.hypot(corner2[0] - corner1[0], corner2[1] - corner1[1]) < REFINE_MIN_BASE:
        return raw_tip
    angle = triangle_angle_at_tip(raw_tip, corner1, corner2)
    if angle < REFINE_MIN_ANGLE_DEG or angle > REFINE_MAX_ANGLE_DEG:
        return raw_tip
    refined = refine_tip_from_triangle(raw_tip, corner1, corner2)
    if math.hypot(refined[0] - raw_tip[0], refined[1] - raw_tip[1]) > REFINE_MAX_SHIFT:
        return raw_tip
    return refined


def predict_tips_onnx(img: np.ndarray, yolo_session: ort.InferenceSession, tip_session: ort.InferenceSession) -> tuple[list[dict], list[tuple[float, float]]]:
    candidate_infos = yolo_coarse_detect_onnx(img, yolo_session)
    if not candidate_infos:
        return [], []
    best_preds = None
    best_score = -1.0
    best_coarse = None
    best_signal = max(info["signal_score"] for info in candidate_infos)
    for info in candidate_infos:
        preds = []
        total_confidence = 0.0
        for coarse_x, coarse_y in info["pair"]:
            patch = extract_patch(img, coarse_x, coarse_y, PATCH_SIZE)
            patch_norm = normalize_fine_patch(patch)[None, None, :, :].astype(np.float32)
            heatmap_3ch = tip_session.run(None, {tip_session.get_inputs()[0].name: patch_norm})[0][0]
            points, confs = heatmap_to_coords(heatmap_3ch, coarse_x, coarse_y)
            total_confidence += confs[0]
            raw_tip, corner1, corner2 = points
            refined_tip = maybe_refine_tip(raw_tip, corner1, corner2, confs)
            preds.append({"tip": refined_tip, "raw_tip": raw_tip, "corner1": corner1, "corner2": corner2, "tip_confidence": confs[0]})
        coarse_prior = 0.05 * max(info["signal_score"], 0.0) / max(best_signal, 1.0)
        combined_score = total_confidence + coarse_prior
        if combined_score > best_score:
            best_score = combined_score
            best_preds = preds
            best_coarse = info["pair"]
    return best_preds or [], best_coarse or []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--export-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--max-images", type=int, default=12)
    args = parser.parse_args()

    yolo_session = ort.InferenceSession(str(args.export_dir / "yolo_vhole.onnx"), providers=["CPUExecutionProvider"])
    tip_session = ort.InferenceSession(str(args.export_dir / "tip_model.onnx"), providers=["CPUExecutionProvider"])

    entries = []
    for json_path in sorted(args.data_dir.glob("*_hv.json")):
        img_path = args.data_dir / f"{json_path.stem}.png"
        if not img_path.exists():
            continue
        entries.append((img_path, json_path))
    if args.max_images > 0:
        entries = entries[: args.max_images]

    all_errors = []
    failures = []
    for img_path, json_path in entries:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        gt_kps = get_vhole_keypoints(json_path, img.shape)
        gt_tips = [kp["tip"] for kp in gt_kps]
        preds, _coarse = predict_tips_onnx(img, yolo_session, tip_session)
        errors = []
        for gx, gy in gt_tips:
            errors.append(min((math.hypot(gx - p["tip"][0], gy - p["tip"][1]) for p in preds), default=9999.0))
        all_errors.extend(errors)
        ok = len(errors) >= 2 and all(err <= 5.0 for err in errors)
        if not ok:
            failures.append((img_path.stem, errors))
        print(f"{img_path.stem}: {[round(e, 3) for e in errors]} {'OK' if ok else 'MISS'}")

    finite = [e for e in all_errors if e < 9999]
    print(json.dumps({
        "num_images": len(entries),
        "num_tips": len(finite),
        "ok_images": len(entries) - len(failures),
        "mean_error_px": float(np.mean(finite)) if finite else None,
        "median_error_px": float(np.median(finite)) if finite else None,
        "max_error_px": float(max(finite)) if finite else None,
        "failures": failures,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
