#!/usr/bin/env python3
"""V-hole inference for the V7-86 YOLO + Small U-Net pipeline.

This module is intentionally inference-only. Training/export code lives in
``train_vhole_unet86_export.py``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn


@dataclass(frozen=True)
class VHoleUNet86Config:
    """Runtime constants for experiment 86."""

    patch_size: int = 192
    yolo_imgsz: int = 1280
    yolo_conf: float = 0.05
    yolo_top_k: int = 6
    top_pairs: int = 3
    out_channels: int = 1
    base_channels: int = 24
    threshold: float = 0.35
    shape_prior: bool = True
    shape_prior_height_gate: float = 22.0
    shape_prior_base_min: float = 20.0
    shape_prior_ratio_max: float = 1.0
    shape_prior_a: float = 0.85
    shape_prior_b: float = 12.0
    shape_prior_hmin: float = 18.0
    shape_prior_hmax: float = 32.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VHoleUNet86Prediction:
    """One decoded V-hole prediction in original image coordinates."""

    x: float
    y: float
    raw_x: float
    raw_y: float
    corner1: tuple[float, float]
    corner2: tuple[float, float]
    confidence: float
    mask_score: float
    mask_area: int
    coarse: tuple[float, float]
    contour: list[tuple[float, float]]
    shape_prior_applied: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "x": float(self.x),
            "y": float(self.y),
            "raw_x": float(self.raw_x),
            "raw_y": float(self.raw_y),
            "corner1": [float(self.corner1[0]), float(self.corner1[1])],
            "corner2": [float(self.corner2[0]), float(self.corner2[1])],
            "confidence": float(self.confidence),
            "mask_score": float(self.mask_score),
            "mask_area": int(self.mask_area),
            "coarse": [float(self.coarse[0]), float(self.coarse[1])],
            "contour": [[float(x), float(y)] for x, y in self.contour],
            "shape_prior_applied": bool(self.shape_prior_applied),
        }


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        groups = 8 if out_ch >= 8 else 1
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SmallUNet(nn.Module):
    """Small U-Net used by V7 experiment 86."""

    def __init__(self, out_channels: int = 1, base_channels: int = 24) -> None:
        super().__init__()
        b = base_channels
        self.enc1 = ConvBlock(1, b)
        self.enc2 = ConvBlock(b, b * 2)
        self.enc3 = ConvBlock(b * 2, b * 4)
        self.bottleneck = ConvBlock(b * 4, b * 8)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(b * 8, b * 4, 2, stride=2)
        self.dec3 = ConvBlock(b * 8, b * 4)
        self.up2 = nn.ConvTranspose2d(b * 4, b * 2, 2, stride=2)
        self.dec2 = ConvBlock(b * 4, b * 2)
        self.up1 = nn.ConvTranspose2d(b * 2, b, 2, stride=2)
        self.dec1 = ConvBlock(b * 2, b)
        self.out = nn.Conv2d(b, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        bottleneck = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(bottleneck), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_to_yolo_str(device: torch.device) -> str:
    if device.type == "cuda":
        return str(device.index if device.index is not None else 0)
    return device.type


def load_hv_image(image: str | Path | np.ndarray) -> np.ndarray:
    if isinstance(image, np.ndarray):
        if image.ndim == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    img = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image}")
    return img


def extract_patch(img: np.ndarray, cx: float, cy: float, size: int) -> np.ndarray:
    """Extract a reflected-border patch centered at ``cx, cy``."""
    h, w = img.shape[:2]
    half = size // 2
    x1, y1 = int(cx) - half, int(cy) - half
    x2, y2 = x1 + size, y1 + size

    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - w)
    pad_bottom = max(0, y2 - h)

    if pad_left or pad_top or pad_right or pad_bottom:
        padded = cv2.copyMakeBorder(
            img,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_REFLECT,
        )
        x1 += pad_left
        y1 += pad_top
        patch = padded[y1:y1 + size, x1:x1 + size]
    else:
        patch = img[y1:y2, x1:x2]

    if patch.shape[0] != size or patch.shape[1] != size:
        patch = cv2.resize(patch, (size, size), interpolation=cv2.INTER_LINEAR)
    return patch


def extract_patch_constant(img: np.ndarray, cx: float, cy: float, size: int) -> np.ndarray:
    """Extract a constant-zero patch, mainly useful for mask targets."""
    h, w = img.shape[:2]
    half = size // 2
    x1, y1 = int(cx) - half, int(cy) - half
    x2, y2 = x1 + size, y1 + size

    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - w)
    pad_bottom = max(0, y2 - h)

    if pad_left or pad_top or pad_right or pad_bottom:
        padded = cv2.copyMakeBorder(
            img,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=0,
        )
        x1 += pad_left
        y1 += pad_top
        patch = padded[y1:y1 + size, x1:x1 + size]
    else:
        patch = img[y1:y2, x1:x2]

    if patch.shape[0] != size or patch.shape[1] != size:
        patch = cv2.resize(patch, (size, size), interpolation=cv2.INTER_NEAREST)
    return patch


def normalize_fine_patch(patch: np.ndarray) -> np.ndarray:
    patch_u8 = patch.astype(np.uint8, copy=False)
    lo, hi = np.percentile(patch_u8, (1.0, 99.0))
    if hi > lo + 1.0:
        patch_f = (patch_u8.astype(np.float32) - lo) * (255.0 / (hi - lo))
        patch_u8 = np.clip(patch_f, 0, 255).astype(np.uint8)
    return patch_u8.astype(np.float32) / 255.0


def yolo_coarse_detect(
    img: np.ndarray,
    yolo_model: Any,
    device_str: str,
    config: VHoleUNet86Config,
) -> list[dict[str, Any]]:
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    preds_raw = yolo_model.predict(
        img_rgb,
        imgsz=config.yolo_imgsz,
        conf=config.yolo_conf,
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

    all_boxes = []
    order = np.argsort(-confs)[:config.yolo_top_k]
    for idx in order:
        cx = float((xyxy[idx, 0] + xyxy[idx, 2]) / 2.0)
        cy = float((xyxy[idx, 1] + xyxy[idx, 3]) / 2.0)
        angle = math.atan2(cy - img_cy, cx - img_cx)
        radius = math.hypot(cx - img_cx, cy - img_cy)
        all_boxes.append({
            "center": (cx, cy),
            "conf": float(confs[idx]),
            "angle": angle,
            "radius": radius,
        })

    angle_tol = math.radians(30.0)
    candidates: list[dict[str, Any]] = []
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
            candidates.append({
                "pair": [b1["center"], b2["center"]],
                "signal_score": signal_score,
                "key": (pair_score, signal_score),
            })

    if not candidates and len(all_boxes) >= 2:
        b1, b2 = all_boxes[0], all_boxes[1]
        signal_score = b1["conf"] + b2["conf"]
        candidates.append({
            "pair": [b1["center"], b2["center"]],
            "signal_score": signal_score,
            "key": (signal_score, signal_score),
        })

    candidates.sort(key=lambda row: row["key"], reverse=True)
    return candidates[:config.top_pairs]


def _component_from_prob(prob: np.ndarray, threshold: float) -> tuple[np.ndarray | None, float]:
    thresholds = [threshold, 0.50, 0.30, 0.20, 0.12]
    best_mask = None
    best_score = -1.0
    for th in thresholds:
        binary = (prob >= th).astype(np.uint8)
        n_labels, labels, stats, _cent = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for label in range(1, n_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < 5 or area > prob.size * 0.35:
                continue
            comp = labels == label
            score = float(prob[comp].mean()) + min(area, 800) / 8000.0
            if score > best_score:
                best_score = score
                best_mask = comp.astype(np.uint8)
    return best_mask, best_score


def _two_base_corners(
    global_pts: np.ndarray,
    tip: tuple[float, float],
) -> tuple[tuple[float, float], tuple[float, float]]:
    if len(global_pts) == 0:
        return tip, tip
    tx, ty = tip
    dists = np.hypot(global_pts[:, 0] - tx, global_pts[:, 1] - ty)
    order = np.argsort(-dists)
    c1 = tuple(float(v) for v in global_pts[order[0]])
    c2 = c1
    for idx in order[1:]:
        cand = tuple(float(v) for v in global_pts[idx])
        if math.hypot(cand[0] - c1[0], cand[1] - c1[1]) >= 4.0:
            c2 = cand
            break
    return c1, c2


def decode_mask_tip(
    mask_prob: np.ndarray,
    coarse_x: float,
    coarse_y: float,
    img_shape: tuple[int, int],
    config: VHoleUNet86Config,
) -> dict[str, Any]:
    patch_size = config.patch_size
    comp, comp_score = _component_from_prob(mask_prob, config.threshold)
    h, w = img_shape[:2]
    img_cx, img_cy = w / 2.0, h / 2.0

    if comp is None:
        py, px = np.unravel_index(int(np.argmax(mask_prob)), mask_prob.shape)
        gx = coarse_x - patch_size / 2.0 + float(px)
        gy = coarse_y - patch_size / 2.0 + float(py)
        tip = (gx, gy)
        return {
            "tip": tip,
            "raw_tip": tip,
            "corner1": tip,
            "corner2": tip,
            "score": float(mask_prob[py, px]),
            "mask_area": 0,
            "contour": [],
            "shape_prior_applied": False,
        }

    contours, _hier = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        ys, xs = np.nonzero(comp.astype(bool))
        pts = np.stack([xs, ys], axis=1).astype(np.float32)
    else:
        contour = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(contour)
        pts = hull.reshape(-1, 2).astype(np.float32)

    global_pts = pts.copy()
    global_pts[:, 0] += coarse_x - patch_size / 2.0
    global_pts[:, 1] += coarse_y - patch_size / 2.0
    dist_to_center = np.hypot(global_pts[:, 0] - img_cx, global_pts[:, 1] - img_cy)
    tip_idx = int(np.argmin(dist_to_center))
    tip = tuple(float(v) for v in global_pts[tip_idx])
    c1, c2 = _two_base_corners(global_pts, tip)
    contour = [(float(x), float(y)) for x, y in global_pts]
    return {
        "tip": tip,
        "raw_tip": tip,
        "corner1": c1,
        "corner2": c2,
        "score": float(max(comp_score, mask_prob[comp.astype(bool)].mean())),
        "mask_area": int(comp.sum()),
        "contour": contour,
        "shape_prior_applied": False,
    }


def apply_shape_prior(
    decoded: dict[str, Any],
    img_shape: tuple[int, int],
    config: VHoleUNet86Config,
) -> dict[str, Any]:
    if not config.shape_prior:
        return decoded

    tip = tuple(float(v) for v in decoded["tip"])
    c1 = tuple(float(v) for v in decoded["corner1"])
    c2 = tuple(float(v) for v in decoded["corner2"])
    mid = ((c1[0] + c2[0]) / 2.0, (c1[1] + c2[1]) / 2.0)
    base = math.hypot(c2[0] - c1[0], c2[1] - c1[1])
    height = math.hypot(tip[0] - mid[0], tip[1] - mid[1])

    if base < config.shape_prior_base_min:
        return decoded
    if height >= config.shape_prior_height_gate:
        return decoded
    if height / max(base, 1e-6) > config.shape_prior_ratio_max:
        return decoded

    h, w = img_shape[:2]
    ux = w / 2.0 - mid[0]
    uy = h / 2.0 - mid[1]
    norm = math.hypot(ux, uy)
    if norm < 1e-6:
        return decoded

    target_h = config.shape_prior_a * base + config.shape_prior_b
    target_h = float(np.clip(target_h, config.shape_prior_hmin, config.shape_prior_hmax))
    repaired = (mid[0] + ux / norm * target_h, mid[1] + uy / norm * target_h)

    out = dict(decoded)
    out["tip"] = repaired
    out["shape_prior_applied"] = True
    out["shape_prior_before"] = tip
    out["shape_prior_base"] = float(base)
    out["shape_prior_height"] = float(height)
    out["shape_prior_target_height"] = float(target_h)
    return out


def _load_state_dict(path: str | Path, device: torch.device) -> dict[str, torch.Tensor]:
    try:
        state = torch.load(str(path), map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(str(path), map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported mask weight format: {path}")
    return {
        (k[7:] if isinstance(k, str) and k.startswith("module.") else k): v
        for k, v in state.items()
    }


class VHoleUNet86Detector:
    """Load YOLO + Small U-Net weights and run V-hole tip inference."""

    def __init__(
        self,
        yolo_weight: str | Path,
        mask_weight: str | Path,
        *,
        config: VHoleUNet86Config | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("ultralytics is required for V-hole YOLO inference") from exc

        self.config = config or VHoleUNet86Config()
        self.device = resolve_device(device)
        self.device_str = device_to_yolo_str(self.device)
        self.yolo_model = YOLO(str(yolo_weight))
        self.mask_model = SmallUNet(
            out_channels=self.config.out_channels,
            base_channels=self.config.base_channels,
        ).to(self.device)
        self.mask_model.load_state_dict(_load_state_dict(mask_weight, self.device))
        self.mask_model.eval()

    @classmethod
    def from_model_dir(
        cls,
        model_dir: str | Path,
        *,
        config: VHoleUNet86Config | None = None,
        device: str | torch.device | None = None,
    ) -> "VHoleUNet86Detector":
        model_dir = Path(model_dir)
        return cls(
            model_dir / "yolo_best.pt",
            model_dir / "mask_model.pth",
            config=config,
            device=device,
        )

    def predict_image(self, image: str | Path | np.ndarray) -> tuple[list[VHoleUNet86Prediction], list[tuple[float, float]]]:
        img = load_hv_image(image)
        candidate_infos = yolo_coarse_detect(img, self.yolo_model, self.device_str, self.config)
        if not candidate_infos:
            return [], []

        best_preds: list[VHoleUNet86Prediction] | None = None
        best_pair: list[tuple[float, float]] | None = None
        best_score = -1e9
        best_signal = max(float(info["signal_score"]) for info in candidate_infos)

        with torch.no_grad():
            for info in candidate_infos:
                pair = info["pair"]
                preds: list[VHoleUNet86Prediction] = []
                pair_score = 0.0
                for coarse_x, coarse_y in pair:
                    patch = extract_patch(img, coarse_x, coarse_y, self.config.patch_size)
                    patch_norm = normalize_fine_patch(patch)
                    x = torch.from_numpy(patch_norm).unsqueeze(0).unsqueeze(0).float().to(self.device)
                    out = self.mask_model(x)[0].detach().cpu()
                    mask_prob = torch.sigmoid(out[0]).numpy()
                    decoded = decode_mask_tip(
                        mask_prob,
                        float(coarse_x),
                        float(coarse_y),
                        img.shape,
                        self.config,
                    )
                    decoded = apply_shape_prior(decoded, img.shape, self.config)
                    tip = decoded["tip"]
                    raw_tip = decoded.get("shape_prior_before", decoded.get("raw_tip", tip))
                    score = float(decoded["score"])
                    pair_score += score
                    preds.append(VHoleUNet86Prediction(
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
                    ))

                coarse_prior = 0.05 * max(float(info["signal_score"]), 0.0) / max(best_signal, 1.0)
                pair_score += coarse_prior
                if pair_score > best_score:
                    best_score = pair_score
                    best_preds = preds
                    best_pair = pair

        return best_preds or [], best_pair or []

    def predict_path(self, image_path: str | Path) -> tuple[list[VHoleUNet86Prediction], list[tuple[float, float]]]:
        return self.predict_image(image_path)


def detect_vholes_unet86(
    image: str | Path | np.ndarray,
    yolo_weight: str | Path,
    mask_weight: str | Path,
    *,
    config: VHoleUNet86Config | None = None,
    device: str | torch.device | None = None,
) -> list[VHoleUNet86Prediction]:
    detector = VHoleUNet86Detector(yolo_weight, mask_weight, config=config, device=device)
    predictions, _coarse = detector.predict_image(image)
    return predictions


def draw_vhole_unet86_overlay(
    image: str | Path | np.ndarray,
    predictions: list[VHoleUNet86Prediction],
    *,
    coarse_locs: list[tuple[float, float]] | None = None,
    gt_keypoints: list[dict[str, Any]] | None = None,
    out_path: str | Path | None = None,
) -> np.ndarray:
    img = load_hv_image(image)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for clx, cly in coarse_locs or []:
        cv2.circle(vis, (int(round(clx)), int(round(cly))), 14, (255, 130, 0), 2, cv2.LINE_AA)

    for kp in gt_keypoints or []:
        pts = np.array([kp["tip"], kp["corner1"], kp["corner2"]], dtype=np.int32)
        cv2.polylines(vis, [pts], True, (0, 210, 255), 2, cv2.LINE_AA)
        tx, ty = kp["tip"]
        cv2.drawMarker(vis, (int(round(tx)), int(round(ty))), (0, 255, 255), cv2.MARKER_STAR, 22, 2)

    for pred in predictions:
        if len(pred.contour) >= 3:
            pts = np.array(pred.contour, dtype=np.int32)
            cv2.polylines(vis, [pts], True, (255, 0, 255), 2, cv2.LINE_AA)
        cv2.circle(vis, (int(round(pred.raw_x)), int(round(pred.raw_y))), 7, (255, 0, 255), 2, cv2.LINE_AA)
        cv2.drawMarker(vis, (int(round(pred.x)), int(round(pred.y))), (0, 0, 255), cv2.MARKER_CROSS, 26, 2)
        if pred.shape_prior_applied:
            cv2.line(
                vis,
                (int(round(pred.raw_x)), int(round(pred.raw_y))),
                (int(round(pred.x)), int(round(pred.y))),
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

    if out_path is not None:
        cv2.imwrite(str(out_path), vis)
    return vis
