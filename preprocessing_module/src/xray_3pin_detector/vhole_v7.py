from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

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

FAST_ACCEPT_MIN_TOTAL_CONF = 1.80
FAST_ACCEPT_MIN_TIP_CONF = 0.85
FAST_ACCEPT_MAX_ANGLE_ERR_DEG = 35.0
FAST_ACCEPT_MAX_RADIUS_GAP_RATIO = 0.35

STRICT_MAX_OPPOSITE_ERROR_DEG = 60.0
NORMAL_ACCEPT_MIN_TOTAL_CONF = 1.85
NORMAL_ACCEPT_MIN_TIP_CONF = 0.85
LONG_PAIR_MIN_DISTANCE_RATIO = 1.0

GEOMETRY_ACCEPT_MIN_TOTAL_CONF = 1.45
GEOMETRY_ACCEPT_MIN_TIP_CONF = 0.55
GEOMETRY_ACCEPT_MAX_OPPOSITE_ERROR_DEG = 3.0
GEOMETRY_ACCEPT_MIN_DISTANCE_RATIO = 1.85
GEOMETRY_ACCEPT_MIN_RADIUS_RATIO = 0.90
GEOMETRY_ACCEPT_MAX_RADIUS_RATIO = 1.08

HORIZONTAL_MAX_OPPOSITE_ERROR_DEG = 10.0
HORIZONTAL_MIN_TOTAL_CONF = 1.50
HORIZONTAL_AUTO_CROP_MARGIN = 1.220

MESH_CROP_MARGIN_BY_GROUP = {
    "JIG미조립": 1.052,
    "TYPE_C": 1.065,
    "TYPE_A-1": 1.059,
    "TYPE_A_미조립": 1.051,
    "TYPE_A_조립": 1.067,
    "TYPE_D": 1.051,
    "unknown": 1.060,
}


@dataclass(frozen=True)
class VHolePoint:
    x_px: float
    y_px: float
    angle_deg: float
    radius_px: float
    radius_ratio: float
    score: float
    label: str


@dataclass(frozen=True)
class VHolePair:
    point_a: VHolePoint
    point_b: VHolePoint
    pair_score: float
    opposite_error_deg: float
    guide_center_x_px: float
    guide_center_y_px: float
    guide_radius_px: float


@dataclass(frozen=True)
class VHoleResult:
    status: str
    method: str
    nominal_axis: str
    nominal_axis_deg: float
    axis_angle_deg: float | None
    rotation_to_nominal_deg: float | None
    pair: VHolePair | None
    detection: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _angle_distance_deg(angle_a: float, angle_b: float) -> float:
    return abs(((float(angle_a) - float(angle_b) + 180.0) % 360.0) - 180.0)


def _strict_opposite_error_deg(angle_a: float, angle_b: float) -> float:
    return abs(180.0 - _angle_distance_deg(angle_a, angle_b))


def _extract_patch(img: np.ndarray, cx: float, cy: float, size: int) -> np.ndarray:
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
    patch = img[y1:y1 + size, x1:x1 + size]
    if patch.shape[0] != size or patch.shape[1] != size:
        patch = cv2.resize(patch, (size, size))
    return patch


def _normalize_fine_patch(patch: np.ndarray) -> np.ndarray:
    patch_u8 = patch.astype(np.uint8, copy=False)
    lo, hi = np.percentile(patch_u8, (1.0, 99.0))
    if hi > lo + 1.0:
        patch_f = (patch_u8.astype(np.float32) - lo) * (255.0 / (hi - lo))
        patch_u8 = np.clip(patch_f, 0, 255).astype(np.uint8)
    return patch_u8.astype(np.float32) / 255.0


def _letterbox_gray_to_rgb_tensor(img: np.ndarray, new_shape: int = YOLO_IMGSZ) -> tuple[np.ndarray, float, tuple[float, float]]:
    h, w = img.shape[:2]
    scale = min(new_shape / h, new_shape / w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_shape, new_shape), 114, dtype=np.uint8)
    pad_x = (new_shape - new_w) / 2.0
    pad_y = (new_shape - new_h) / 2.0
    left = int(round(pad_x - 0.1))
    top = int(round(pad_y - 0.1))
    canvas[top:top + new_h, left:left + new_w] = resized
    rgb = np.stack([canvas, canvas, canvas], axis=0).astype(np.float32) / 255.0
    return rgb[None, :, :, :], scale, (float(left), float(top))


def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    out = np.empty_like(boxes)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    return out


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.7, max_det: int = 300) -> list[int]:
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


def _get_heatmap_peak(heatmap_2d: np.ndarray) -> tuple[float, float, float]:
    hm_h, hm_w = heatmap_2d.shape
    peak_y, peak_x = np.unravel_index(np.argmax(heatmap_2d), heatmap_2d.shape)
    peak_y, peak_x = int(peak_y), int(peak_x)
    peak_val = float(heatmap_2d[peak_y, peak_x])
    dx = dy = 0.0
    if 1 <= peak_x < hm_w - 1:
        left = float(heatmap_2d[peak_y, peak_x - 1])
        right = float(heatmap_2d[peak_y, peak_x + 1])
        center = float(heatmap_2d[peak_y, peak_x])
        denom = 2.0 * (left + right - 2.0 * center)
        if abs(denom) > 1e-6:
            dx = float(np.clip((left - right) / denom, -0.5, 0.5))
    if 1 <= peak_y < hm_h - 1:
        top = float(heatmap_2d[peak_y - 1, peak_x])
        bottom = float(heatmap_2d[peak_y + 1, peak_x])
        center = float(heatmap_2d[peak_y, peak_x])
        denom = 2.0 * (top + bottom - 2.0 * center)
        if abs(denom) > 1e-6:
            dy = float(np.clip((top - bottom) / denom, -0.5, 0.5))
    return peak_x + dx, peak_y + dy, peak_val


def _heatmap_to_coords(heatmap_3ch: np.ndarray, coarse_x: float, coarse_y: float) -> tuple[list[tuple[float, float]], list[float]]:
    points = []
    confidences = []
    for ch in range(3):
        hm_x, hm_y, conf = _get_heatmap_peak(heatmap_3ch[ch])
        orig_x = coarse_x - PATCH_SIZE / 2.0 + hm_x / HM_SIZE * PATCH_SIZE
        orig_y = coarse_y - PATCH_SIZE / 2.0 + hm_y / HM_SIZE * PATCH_SIZE
        points.append((orig_x, orig_y))
        confidences.append(conf)
    return points, confidences


def _triangle_angle_at_tip(tip: tuple[float, float], corner1: tuple[float, float], corner2: tuple[float, float]) -> float:
    v1 = (corner1[0] - tip[0], corner1[1] - tip[1])
    v2 = (corner2[0] - tip[0], corner2[1] - tip[1])
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.hypot(*v1)
    mag2 = math.hypot(*v2)
    if mag1 < 1e-6 or mag2 < 1e-6:
        return 0.0
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / (mag1 * mag2)))))


def _refine_tip_from_triangle(tip: tuple[float, float], corner1: tuple[float, float], corner2: tuple[float, float]) -> tuple[float, float]:
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
    return (REFINE_ALPHA * geo_x + (1 - REFINE_ALPHA) * tip[0],
            REFINE_ALPHA * geo_y + (1 - REFINE_ALPHA) * tip[1])


def _maybe_refine_tip(raw_tip: tuple[float, float], corner1: tuple[float, float], corner2: tuple[float, float], confs: list[float]) -> tuple[float, float]:
    if min(confs[1], confs[2]) < REFINE_MIN_CORNER_CONF:
        return raw_tip
    if math.hypot(corner2[0] - corner1[0], corner2[1] - corner1[1]) < REFINE_MIN_BASE:
        return raw_tip
    angle = _triangle_angle_at_tip(raw_tip, corner1, corner2)
    if angle < REFINE_MIN_ANGLE_DEG or angle > REFINE_MAX_ANGLE_DEG:
        return raw_tip
    refined = _refine_tip_from_triangle(raw_tip, corner1, corner2)
    if math.hypot(refined[0] - raw_tip[0], refined[1] - raw_tip[1]) > REFINE_MAX_SHIFT:
        return raw_tip
    return refined


def _rotate_model_image(img: np.ndarray, rotation_k: int) -> np.ndarray:
    if rotation_k == 0:
        return img
    if rotation_k == 1:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rotation_k == 2:
        return cv2.rotate(img, cv2.ROTATE_180)
    return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)


def _unrotate_model_point(x: float, y: float, width: int, height: int, rotation_k: int) -> tuple[float, float]:
    if rotation_k == 0:
        return float(x), float(y)
    if rotation_k == 1:
        return float(width - 1 - y), float(x)
    if rotation_k == 2:
        return float(width - 1 - x), float(height - 1 - y)
    return float(y), float(height - 1 - x)


class V7ONNXDetector:
    def __init__(self, model_dir: Path, providers: list[str] | None = None) -> None:
        model_dir = Path(model_dir)
        yolo_path = model_dir / "yolo_vhole.onnx"
        tip_path = model_dir / "tip_model.onnx"
        missing = [str(path) for path in (yolo_path, tip_path) if not path.exists()]
        if missing:
            joined = ", ".join(missing)
            raise FileNotFoundError(
                "V7 ONNX model files are not included in Git. "
                f"Place the required files before inference: {joined}"
            )
        providers = providers or ["CPUExecutionProvider"]
        self.yolo_session = ort.InferenceSession(str(yolo_path), providers=providers)
        self.tip_session = ort.InferenceSession(str(tip_path), providers=providers)

    def _yolo_coarse_detect(self, img: np.ndarray) -> list[dict[str, Any]]:
        tensor, scale, (pad_x, pad_y) = _letterbox_gray_to_rgb_tensor(img)
        output = self.yolo_session.run(None, {self.yolo_session.get_inputs()[0].name: tensor})[0][0]
        boxes_xywh = output[:4, :].T
        scores = output[4, :]
        mask = scores >= YOLO_CONF
        boxes_xywh = boxes_xywh[mask]
        scores = scores[mask]
        if len(scores) == 0:
            return []

        boxes_xyxy = _xywh_to_xyxy(boxes_xywh)
        keep = _nms(boxes_xyxy, scores, iou_threshold=0.7)
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

    def predict_tips(self, img: np.ndarray) -> tuple[list[dict[str, Any]], list[tuple[float, float]]]:
        candidate_infos = self._yolo_coarse_detect(img)
        if not candidate_infos:
            return [], []
        best_preds = None
        best_score = -1.0
        best_coarse = None
        best_signal = max(info["signal_score"] for info in candidate_infos)
        for info in candidate_infos:
            preds = []
            total_confidence = 0.0
            min_tip_confidence = 1e9
            for coarse_x, coarse_y in info["pair"]:
                patch = _extract_patch(img, coarse_x, coarse_y, PATCH_SIZE)
                patch_norm = _normalize_fine_patch(patch)[None, None, :, :].astype(np.float32)
                heatmap_3ch = self.tip_session.run(None, {self.tip_session.get_inputs()[0].name: patch_norm})[0][0]
                points, confs = _heatmap_to_coords(heatmap_3ch, coarse_x, coarse_y)
                total_confidence += confs[0]
                min_tip_confidence = min(min_tip_confidence, float(confs[0]))
                raw_tip, corner1, corner2 = points
                refined_tip = _maybe_refine_tip(raw_tip, corner1, corner2, confs)
                preds.append({
                    "tip": refined_tip,
                    "raw_tip": raw_tip,
                    "corner1": corner1,
                    "corner2": corner2,
                    "tip_confidence": confs[0],
                })
            coarse_prior = 0.05 * max(info["signal_score"], 0.0) / max(best_signal, 1.0)
            combined_score = total_confidence + coarse_prior
            if combined_score > best_score:
                best_score = combined_score
                best_preds = preds
                best_coarse = info["pair"]
            if self._fast_accept_pair(preds, total_confidence, min_tip_confidence, img.shape):
                break
        return best_preds or [], best_coarse or []

    @staticmethod
    def _fast_accept_pair(preds: list[dict[str, Any]], total_confidence: float, min_tip_confidence: float, img_shape: tuple[int, ...]) -> bool:
        if len(preds) < 2:
            return False
        if total_confidence < FAST_ACCEPT_MIN_TOTAL_CONF:
            return False
        if min_tip_confidence < FAST_ACCEPT_MIN_TIP_CONF:
            return False
        h, w = img_shape[:2]
        img_cx, img_cy = w / 2.0, h / 2.0
        p1 = preds[0]["tip"]
        p2 = preds[1]["tip"]
        a1 = math.atan2(p1[1] - img_cy, p1[0] - img_cx)
        a2 = math.atan2(p2[1] - img_cy, p2[0] - img_cx)
        angle_gap = abs(((a1 - a2 + math.pi) % (2 * math.pi)) - math.pi)
        angle_err_deg = abs(180.0 - math.degrees(angle_gap))
        if angle_err_deg > FAST_ACCEPT_MAX_ANGLE_ERR_DEG:
            return False
        r1 = math.hypot(p1[0] - img_cx, p1[1] - img_cy)
        r2 = math.hypot(p2[0] - img_cx, p2[1] - img_cy)
        return abs(r1 - r2) <= max(r1, r2, 1.0) * FAST_ACCEPT_MAX_RADIUS_GAP_RATIO

    def detect(self, hv: np.ndarray, guide_circle: Any, product_group: str = "unknown") -> VHoleResult:
        hv_model, offset_x, offset_y, crop_group, crop_margin = self._crop_to_model_square(hv, guide_circle, product_group)
        preds, coarse_points = self.predict_tips(hv_model)
        normal_pair = None
        normal_reject_reason = "missing_two_tips"
        if len(preds) >= 2:
            point_a = self._point_from_xy(
                float(preds[0]["tip"][0]) + offset_x,
                float(preds[0]["tip"][1]) + offset_y,
                guide_circle,
                preds[0].get("tip_confidence", 0.0),
                "normal",
            )
            point_b = self._point_from_xy(
                float(preds[1]["tip"][0]) + offset_x,
                float(preds[1]["tip"][1]) + offset_y,
                guide_circle,
                preds[1].get("tip_confidence", 0.0),
                "normal",
            )
            normal_pair = self._pair_from_points(point_a, point_b, guide_circle)
            normal_reject_reason = self._pair_reject_reason(normal_pair)
            if not normal_reject_reason:
                acceptance = self._pair_acceptance_label(normal_pair)
                detection = {
                    "method": "v7_hv_yolo_tip_onnx",
                    "acceptance": acceptance,
                    "coarse_point_count": len(coarse_points),
                    "model_crop_group": crop_group,
                    "model_crop_mesh_margin": round(float(crop_margin), 4),
                    "pair_distance_px": round(float(self._pair_distance(normal_pair)), 3),
                    "pair_min_distance_px": round(float(guide_circle.radius_px) * LONG_PAIR_MIN_DISTANCE_RATIO, 3),
                    "tip_confidence_sum": round(float(normal_pair.pair_score), 6),
                }
                return self._build_result(normal_pair, 90.0, "12/6", detection)

        hv_horizontal, h_offset_x, h_offset_y, _crop_group, h_crop_margin = self._crop_to_model_square_with_margin(
            hv, guide_circle, HORIZONTAL_AUTO_CROP_MARGIN, "auto_horizontal")
        try:
            horizontal_option = self._detect_horizontal_rotated_pair(hv_horizontal, h_offset_x, h_offset_y, guide_circle)
        except Exception as exc:
            if normal_pair is not None:
                detection = {
                    "method": "v7_hv_yolo_tip_onnx_invalid_normal_pair",
                    "normal_reject_reason": normal_reject_reason,
                    "horizontal_error": str(exc),
                    "model_crop_group": crop_group,
                    "model_crop_mesh_margin": round(float(crop_margin), 4),
                    "pair_distance_px": round(float(self._pair_distance(normal_pair)), 3),
                    "pair_min_distance_px": round(float(guide_circle.radius_px) * LONG_PAIR_MIN_DISTANCE_RATIO, 3),
                    "tip_confidence_sum": round(float(normal_pair.pair_score), 6),
                }
                return self._build_result(normal_pair, 90.0, "12/6", detection, status="invalid")
            return VHoleResult(
                status="fail",
                method="v7_hv_yolo_tip_onnx",
                nominal_axis="",
                nominal_axis_deg=0.0,
                axis_angle_deg=None,
                rotation_to_nominal_deg=None,
                pair=None,
                detection={
                    "method": "v7_hv_yolo_tip_onnx",
                    "normal_reject_reason": normal_reject_reason,
                    "horizontal_error": str(exc),
                },
            )
        pair = horizontal_option["pair"]
        detection = {
            "method": "v7_hv_yolo_tip_onnx_auto_horizontal_rotated",
            "coarse_point_count": horizontal_option["coarse_point_count"],
            "normal_reject_reason": normal_reject_reason,
            "normal_pair_distance_px": round(float(self._pair_distance(normal_pair)), 3) if normal_pair is not None else None,
            "model_crop_group": "auto_horizontal",
            "model_crop_mesh_margin": round(float(h_crop_margin), 4),
            "rotation": horizontal_option["rotation_label"],
            "raw_radius_ratios": [round(float(v), 6) for v in horizontal_option["raw_radius_ratios"]],
            "radius_normalization": "left_small_right_large",
            "pair_distance_px": round(float(self._pair_distance(pair)), 3),
            "pair_min_distance_px": round(float(guide_circle.radius_px) * LONG_PAIR_MIN_DISTANCE_RATIO, 3),
            "tip_confidence_sum": round(float(pair.pair_score), 6),
        }
        return self._build_result(pair, 0.0, "3/9", detection)

    @staticmethod
    def _crop_to_model_square_with_margin(img: np.ndarray, guide_circle: Any, margin: float, group: str) -> tuple[np.ndarray, float, float, str, float]:
        radius = max(1.0, float(guide_circle.radius_px))
        size = max(32, int(round(radius * 2.0 * float(margin))))
        x1 = int(round(float(guide_circle.center_x_px) - size / 2.0))
        y1 = int(round(float(guide_circle.center_y_px) - size / 2.0))
        x2 = x1 + size
        y2 = y1 + size
        h, w = img.shape[:2]
        pad_left = max(0, -x1)
        pad_top = max(0, -y1)
        pad_right = max(0, x2 - w)
        pad_bottom = max(0, y2 - h)
        src = img
        if pad_left or pad_top or pad_right or pad_bottom:
            src = cv2.copyMakeBorder(src, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT)
        crop = src[y1 + pad_top:y2 + pad_top, x1 + pad_left:x2 + pad_left]
        if crop.shape[0] != size or crop.shape[1] != size:
            crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
        return crop, float(x1), float(y1), group, float(margin)

    def _crop_to_model_square(self, img: np.ndarray, guide_circle: Any, product_group: str) -> tuple[np.ndarray, float, float, str, float]:
        group = product_group if product_group in MESH_CROP_MARGIN_BY_GROUP else "unknown"
        margin = MESH_CROP_MARGIN_BY_GROUP.get(group, MESH_CROP_MARGIN_BY_GROUP["unknown"])
        return self._crop_to_model_square_with_margin(img, guide_circle, margin, group)

    @staticmethod
    def _point_from_xy(x_px: float, y_px: float, guide_circle: Any, score: float, label: str) -> VHolePoint:
        center_x = float(guide_circle.center_x_px)
        center_y = float(guide_circle.center_y_px)
        radius = math.hypot(float(x_px) - center_x, float(y_px) - center_y)
        angle_deg = math.degrees(math.atan2(float(y_px) - center_y, float(x_px) - center_x)) % 360.0
        return VHolePoint(
            x_px=float(x_px),
            y_px=float(y_px),
            angle_deg=float(angle_deg),
            radius_px=float(radius),
            radius_ratio=float(radius) / max(float(guide_circle.radius_px), 1e-6),
            score=float(score),
            label=label,
        )

    @staticmethod
    def _pair_from_points(point_a: VHolePoint, point_b: VHolePoint, guide_circle: Any) -> VHolePair:
        return VHolePair(
            point_a=point_a,
            point_b=point_b,
            pair_score=float(point_a.score) + float(point_b.score),
            opposite_error_deg=_strict_opposite_error_deg(point_a.angle_deg, point_b.angle_deg),
            guide_center_x_px=float(guide_circle.center_x_px),
            guide_center_y_px=float(guide_circle.center_y_px),
            guide_radius_px=float(guide_circle.radius_px),
        )

    @staticmethod
    def _pair_distance(pair: VHolePair | None) -> float:
        if pair is None:
            return 0.0
        return math.hypot(
            float(pair.point_b.x_px) - float(pair.point_a.x_px),
            float(pair.point_b.y_px) - float(pair.point_a.y_px),
        )

    def _strict_pair_reject_reason(self, pair: VHolePair) -> str:
        reasons = []
        if float(pair.pair_score) < NORMAL_ACCEPT_MIN_TOTAL_CONF:
            reasons.append("low_total_confidence")
        if min(float(pair.point_a.score), float(pair.point_b.score)) < NORMAL_ACCEPT_MIN_TIP_CONF:
            reasons.append("low_tip_confidence")
        if float(pair.opposite_error_deg) > STRICT_MAX_OPPOSITE_ERROR_DEG:
            reasons.append("opposite_error")
        min_distance = max(1.0, float(pair.guide_radius_px)) * LONG_PAIR_MIN_DISTANCE_RATIO
        if self._pair_distance(pair) < min_distance:
            reasons.append("pair_distance_lt_mesh_radius")
        return ",".join(reasons)

    def _is_geometry_confirmed_pair(self, pair: VHolePair) -> bool:
        if float(pair.pair_score) < GEOMETRY_ACCEPT_MIN_TOTAL_CONF:
            return False
        if min(float(pair.point_a.score), float(pair.point_b.score)) < GEOMETRY_ACCEPT_MIN_TIP_CONF:
            return False
        if float(pair.opposite_error_deg) > GEOMETRY_ACCEPT_MAX_OPPOSITE_ERROR_DEG:
            return False
        if self._pair_distance(pair) < max(1.0, float(pair.guide_radius_px)) * GEOMETRY_ACCEPT_MIN_DISTANCE_RATIO:
            return False
        radius_ratios = (float(pair.point_a.radius_ratio), float(pair.point_b.radius_ratio))
        return all(GEOMETRY_ACCEPT_MIN_RADIUS_RATIO <= ratio <= GEOMETRY_ACCEPT_MAX_RADIUS_RATIO for ratio in radius_ratios)

    def _pair_reject_reason(self, pair: VHolePair) -> str:
        strict_reason = self._strict_pair_reject_reason(pair)
        if not strict_reason:
            return ""
        if self._is_geometry_confirmed_pair(pair):
            return ""
        return strict_reason

    def _pair_acceptance_label(self, pair: VHolePair) -> str:
        if not self._strict_pair_reject_reason(pair):
            return "strict_confidence"
        if self._is_geometry_confirmed_pair(pair):
            return "geometry_confirmed_low_confidence"
        return "rejected"

    def _detect_horizontal_rotated_pair(self, hv_model: np.ndarray, offset_x: float, offset_y: float, guide_circle: Any) -> dict[str, Any]:
        height, width = hv_model.shape[:2]
        options = []
        for label, rotation_k in (("rot90ccw", 1), ("rot90cw", 3)):
            rotated = _rotate_model_image(hv_model, rotation_k)
            preds, coarse_points = self.predict_tips(rotated)
            if len(preds) < 2:
                continue
            raw_points = []
            for pred in preds[:2]:
                local_x, local_y = _unrotate_model_point(
                    pred["tip"][0], pred["tip"][1], width, height, rotation_k)
                point = self._point_from_xy(
                    float(local_x) + float(offset_x),
                    float(local_y) + float(offset_y),
                    guide_circle,
                    pred.get("tip_confidence", 0.0),
                    f"horizontal_{label}",
                )
                raw_points.append(point)

            horizontal_error = _strict_opposite_error_deg(raw_points[0].angle_deg, raw_points[1].angle_deg)
            if horizontal_error > HORIZONTAL_MAX_OPPOSITE_ERROR_DEG:
                continue

            left_idx = 0 if _angle_distance_deg(raw_points[0].angle_deg, 180.0) <= _angle_distance_deg(raw_points[1].angle_deg, 180.0) else 1
            right_idx = 1 - left_idx
            left_angle = raw_points[left_idx].angle_deg
            right_angle = raw_points[right_idx].angle_deg
            left_score = raw_points[left_idx].score
            right_score = raw_points[right_idx].score
            small_radius, large_radius = sorted([raw_points[0].radius_px, raw_points[1].radius_px])

            center_x = float(guide_circle.center_x_px)
            center_y = float(guide_circle.center_y_px)
            point_left = self._point_from_xy(
                center_x + small_radius * math.cos(math.radians(left_angle)),
                center_y + small_radius * math.sin(math.radians(left_angle)),
                guide_circle,
                left_score,
                f"horizontal_{label}_radius_normalized",
            )
            point_right = self._point_from_xy(
                center_x + large_radius * math.cos(math.radians(right_angle)),
                center_y + large_radius * math.sin(math.radians(right_angle)),
                guide_circle,
                right_score,
                f"horizontal_{label}_radius_normalized",
            )
            pair = self._pair_from_points(point_left, point_right, guide_circle)
            if pair.pair_score < HORIZONTAL_MIN_TOTAL_CONF:
                continue
            if self._pair_distance(pair) < max(1.0, float(pair.guide_radius_px)) * LONG_PAIR_MIN_DISTANCE_RATIO:
                continue
            side_error = _angle_distance_deg(point_left.angle_deg, 180.0) + _angle_distance_deg(point_right.angle_deg, 0.0)
            radius_gap = abs(point_right.radius_px - point_left.radius_px)
            options.append({
                "pair": pair,
                "rotation_label": label,
                "coarse_point_count": len(coarse_points),
                "raw_radius_ratios": [raw_points[0].radius_ratio, raw_points[1].radius_ratio],
                "key": (pair.pair_score - 0.01 * side_error, -horizontal_error, radius_gap),
            })
        if not options:
            raise RuntimeError("Rotated V7 inference did not return a valid long horizontal pair.")
        options.sort(key=lambda item: item["key"], reverse=True)
        return options[0]

    def _build_result(
        self,
        pair: VHolePair,
        nominal_axis_deg: float,
        nominal_axis_label: str,
        detection: dict[str, Any],
        status: str = "ok",
    ) -> VHoleResult:
        axis_angle = math.degrees(math.atan2(
            float(pair.point_b.y_px) - float(pair.point_a.y_px),
            float(pair.point_b.x_px) - float(pair.point_a.x_px),
        )) % 180.0
        rotation_to_nominal = ((float(nominal_axis_deg) - axis_angle + 90.0) % 180.0) - 90.0
        return VHoleResult(
            status=status,
            method=str(detection.get("method", "")),
            nominal_axis=nominal_axis_label,
            nominal_axis_deg=float(nominal_axis_deg),
            axis_angle_deg=float(axis_angle),
            rotation_to_nominal_deg=float(rotation_to_nominal),
            pair=pair,
            detection=detection,
        )
