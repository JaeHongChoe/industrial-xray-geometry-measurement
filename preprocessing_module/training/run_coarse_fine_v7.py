#!/usr/bin/env python3
"""
V-hole Tip Localization V7: Best-of-All Hybrid
================================================

Combines the proven strengths from V5, V5_Tuned, V6, and compare_coarse:

  Stage 1: YOLOv8 (fast, 100% detection)
    - No top/bottom constraint (from compare) → rotation-robust
    - Multi-candidate pair search (from V5) → best pair selection

  Stage 2: 3ch Triangle Heatmap at 96×96 resolution
    - 96×96 high-res output (from V5_Tuned) → 2× subpixel precision
    - 3ch tip+corners (from V5) → enables geometric refinement
    - ±60px jitter (from V5_Tuned) → absorbs coarse error
    - Weighted MSE + tip-weighted loss (V5_Tuned + V6)

  Refinement: Safety-guarded triangle bisector (from V6)
    - Only refine when corner confidence > threshold
    - Cap maximum shift to prevent wild corrections

Expected: ~95% accuracy (≤5px) across all folds.
"""
import argparse
import cv2
import numpy as np
import json
import math
import os
import shutil
import logging
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import timm
from tqdm import tqdm

# ============================================================================
# Configuration
# ============================================================================
HV_DIR = Path("vhole_hv_crops")

# Stage 2 — V7 tuned parameters
PATCH_SIZE = 192
HM_SIZE = 96              # [from V5_Tuned] 2× resolution (48→96)
HM_SIGMA = 5.0            # [from V5_Tuned] matched to 96×96
NUM_KEYPOINTS = 3          # [from V5] tip + corner1 + corner2
JITTER = 60                # [from V5_Tuned] absorbs coarse error better
SAMPLES_PER_TIP = 500
EPOCHS = 120               # [from V5_Tuned] more training for harder task
BATCH_SIZE = 32
LR = 5e-4                  # [from V5_Tuned] slightly lower

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

# Stage 1 YOLO parameters
YOLO_BBOX_PAD = 60
YOLO_IMGSZ = 1280
YOLO_EPOCHS = 100
YOLO_BATCH = 8
YOLO_CONF = 0.05
YOLO_TOP_K = 6             # total top-k boxes (no top/bottom split)
TOP_PAIRS = 3              # candidate pairs to try in stage 2

# Refinement — V7b tightened safety guard thresholds
REFINE_ALPHA = 0.3              # [V7b] 0.5→0.3: 과도한 보정 방지 (G055 fix)
REFINE_MIN_CORNER_CONF = 0.25   # [V7b] 0.15→0.25: 불확실한 corner일 때 skip
REFINE_MIN_BASE = 8.0
REFINE_MAX_SHIFT = 12.0          # [V7b] 20→12: 큰 shift 방지
REFINE_MIN_ANGLE_DEG = 15.0     # [V7b] 삼각형이 너무 뾰족하면 skip
REFINE_MAX_ANGLE_DEG = 165.0    # [V7b] 삼각형이 너무 평평하면 skip

# Loss weights — V6 tip-weighted
TIP_LOSS_WEIGHT = 1.0
CORNER_LOSS_WEIGHT = 0.3

# Weighted MSE peak emphasis — V5_Tuned
PEAK_WEIGHT = 5.0
BG_WEIGHT = 1.0
PEAK_THRESHOLD = 0.1


# ============================================================================
# Utility
# ============================================================================
def extract_patch(img, cx, cy, size):
    """Extract a patch centered at (cx, cy) with border reflection padding."""
    h, w = img.shape[:2]
    half = size // 2
    x1, y1 = int(cx) - half, int(cy) - half
    x2, y2 = x1 + size, y1 + size

    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - w)
    pad_bottom = max(0, y2 - h)

    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        img_padded = cv2.copyMakeBorder(
            img, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_REFLECT,
        )
        x1 += pad_left
        y1 += pad_top
        patch = img_padded[y1:y1 + size, x1:x1 + size]
    else:
        patch = img[y1:y2, x1:x2]

    if patch.shape[0] != size or patch.shape[1] != size:
        patch = cv2.resize(patch, (size, size))
    return patch


def get_vhole_keypoints(json_path, img_shape):
    """Extract 3 keypoints per V-hole from labelme JSON."""
    if not Path(json_path).exists():
        return []
    with open(json_path) as f:
        data = json.load(f)
    h, w = img_shape[0], img_shape[1]
    img_cx, img_cy = w / 2, h / 2
    keypoints = []
    for shape in data["shapes"]:
        if shape["label"] != "v":
            continue
        pts = shape["points"]
        if len(pts) < 3:
            continue
        tip_idx = min(range(len(pts)),
                      key=lambda i: math.hypot(pts[i][0] - img_cx, pts[i][1] - img_cy))
        tip = (pts[tip_idx][0], pts[tip_idx][1])
        corners = [(pts[i][0], pts[i][1]) for i in range(len(pts)) if i != tip_idx]
        corners.sort(key=lambda c: math.atan2(c[1] - tip[1], c[0] - tip[0]))
        keypoints.append({
            "tip": tip,
            "corner1": corners[0],
            "corner2": corners[1] if len(corners) > 1 else corners[0],
        })
    return keypoints


def make_gaussian_heatmap(hm_size, cx, cy, sigma=HM_SIGMA):
    yy, xx = np.mgrid[0:hm_size, 0:hm_size].astype(np.float32)
    heatmap = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    return heatmap


def normalize_fine_patch(patch):
    patch_u8 = patch.astype(np.uint8, copy=False)
    lo, hi = np.percentile(patch_u8, (1.0, 99.0))
    if hi > lo + 1.0:
        patch_f = (patch_u8.astype(np.float32) - lo) * (255.0 / (hi - lo))
        patch_u8 = np.clip(patch_f, 0, 255).astype(np.uint8)
    return patch_u8.astype(np.float32) / 255.0


def triangle_to_bbox(keypoints, img_shape=None, pad=YOLO_BBOX_PAD):
    pts = [keypoints["tip"], keypoints["corner1"], keypoints["corner2"]]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x1, y1 = min(xs) - pad, min(ys) - pad
    x2, y2 = max(xs) + pad, max(ys) + pad
    if img_shape is not None:
        h, w = img_shape[:2]
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(float(w), x2), min(float(h), y2)
    return float(x1), float(y1), float(x2), float(y2)


# ============================================================================
# Stage 1: YOLO — no top/bottom constraint
# ============================================================================
def prepare_yolo_dataset(train_entries, val_entries, fold_dir):
    fold_dir = Path(fold_dir)
    for split, split_entries in [("train", train_entries), ("val", val_entries)]:
        img_dir = fold_dir / "images" / split
        lbl_dir = fold_dir / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for e in split_entries:
            img = cv2.imread(str(e["path"]), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            dst_img = img_dir / f"{e['name']}.png"
            cv2.imwrite(str(dst_img), img_rgb)

            h, w = img.shape
            lines = []
            for kp in e["keypoints"]:
                x1, y1, x2, y2 = triangle_to_bbox(kp, img_shape=img.shape)
                bcx = ((x1 + x2) / 2.0) / w
                bcy = ((y1 + y2) / 2.0) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                lines.append(f"0 {bcx:.6f} {bcy:.6f} {bw:.6f} {bh:.6f}")

            with open(lbl_dir / f"{e['name']}.txt", "w") as f:
                f.write("\n".join(lines) + "\n")

    yaml_path = fold_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {fold_dir.resolve()}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("nc: 1\n")
        f.write("names: ['vhole']\n")
    return str(yaml_path)


def train_yolo_stage1(train_entries, val_entries, fold_dir, device=DEVICE):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics is required") from exc

    stage_dir = Path(fold_dir) / "stage1_yolo"
    stage_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = prepare_yolo_dataset(train_entries, val_entries, stage_dir / "dataset")
    device_str = "0" if device.type == "cuda" else device.type

    model = YOLO("yolov8n.pt")
    model.train(
        data=yaml_path,
        epochs=YOLO_EPOCHS,
        imgsz=YOLO_IMGSZ,
        batch=YOLO_BATCH,
        device=device_str,
        project=str(stage_dir.resolve()),
        name="train",
        exist_ok=True,
        save=False,
        plots=False,
        verbose=False,
        val=False,
    )

    best_path = stage_dir / "train" / "weights" / "best.pt"
    if best_path.exists():
        model = YOLO(str(best_path))

    # [V7] Cleanup dataset to save disk space (~565MB per fold)
    dataset_dir = stage_dir / "dataset"
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
        print(f"  🧹 Cleaned up YOLO dataset: {dataset_dir}")

    return model, device_str


def yolo_coarse_detect(img, yolo_model, device_str, top_k=YOLO_TOP_K, top_pairs=TOP_PAIRS):
    """
    [V7] YOLO coarse detection WITHOUT top/bottom constraint.
    Returns candidate pairs based on angular opposition (180° symmetry).
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    preds_raw = yolo_model.predict(
        img_rgb, imgsz=YOLO_IMGSZ, conf=YOLO_CONF,
        device=device_str, verbose=False,
    )

    if len(preds_raw) == 0 or preds_raw[0].boxes is None:
        return []

    boxes = preds_raw[0].boxes
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()

    h, w = img.shape[:2]
    img_cx, img_cy = w / 2.0, h / 2.0

    # Collect all boxes with their centers and angles
    all_boxes = []
    order = np.argsort(-confs)[:top_k]
    for i in order:
        cx = float((xyxy[i, 0] + xyxy[i, 2]) / 2.0)
        cy = float((xyxy[i, 1] + xyxy[i, 3]) / 2.0)
        angle = math.atan2(cy - img_cy, cx - img_cx)
        radius = math.hypot(cx - img_cx, cy - img_cy)
        all_boxes.append({
            "center": (cx, cy),
            "conf": float(confs[i]),
            "angle": angle,
            "radius": radius,
        })

    # Generate candidate pairs based on angular opposition (~180° apart)
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
            candidates.append({
                "pair": [b1["center"], b2["center"]],
                "signal_score": signal_score,
                "key": (pair_score, signal_score),
            })

    # Fallback: if no angular match, just use top-2 confidence
    if not candidates and len(all_boxes) >= 2:
        b1, b2 = all_boxes[0], all_boxes[1]
        signal_score = b1["conf"] + b2["conf"]
        candidates.append({
            "pair": [b1["center"], b2["center"]],
            "signal_score": signal_score,
            "key": (signal_score, signal_score),
        })

    candidates.sort(key=lambda x: x["key"], reverse=True)
    return candidates[:top_pairs]


# ============================================================================
# Stage 2: Triangle Pose Estimator — 96×96, 3ch
# ============================================================================
class TrianglePoseEstimatorV7(nn.Module):
    """
    [V7] ResNet18 encoder + extended U-Net decoder → 3ch × 96×96 heatmap.

    Combines V5_Tuned's 96×96 resolution with V5's 3-channel keypoint prediction.
    Uses f0 (stem, stride 2) for the extra skip connection.
    """

    def __init__(self, freeze_encoder=True, num_keypoints=NUM_KEYPOINTS):
        super().__init__()
        # [from V5_Tuned] out_indices includes 0 for stride-2 features
        self.encoder = timm.create_model(
            "resnet18", pretrained=True, in_chans=1,
            features_only=True, out_indices=(0, 1, 2, 3, 4),
        )

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Decoder: 6→12→24→48→96 with skip connections
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(256 + 256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128 + 128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        # [from V5_Tuned] Extra upsample: 48→96
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(64 + 64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )
        # [V7] 3ch output instead of 1ch (32 + f0=64 = 96 input channels)
        self.head = nn.Sequential(
            nn.Conv2d(32 + 64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_keypoints, 1),  # 3ch: tip, corner1, corner2
            nn.Sigmoid(),
        )

    def forward(self, x):
        self.encoder.eval()
        with torch.no_grad():
            f0, f1, f2, f3, f4 = self.encoder(x)

        x = self.up4(f4)                             # 6→12
        x = self.up3(torch.cat([x, f3], dim=1))      # 12→24
        x = self.up2(torch.cat([x, f2], dim=1))      # 24→48
        x = self.up1(torch.cat([x, f1], dim=1))      # 48→96
        x = self.head(torch.cat([x, f0], dim=1))     # 96→96, 3ch
        return x  # (B, 3, 96, 96)


# ============================================================================
# Dataset
# ============================================================================
class TrianglePoseDatasetV7(Dataset):
    """
    [V7] 3-channel heatmap targets at 96×96 with ±60px jitter.
    """

    def __init__(self, entries, samples_per_vhole=SAMPLES_PER_TIP,
                 patch_size=PATCH_SIZE, hm_size=HM_SIZE, is_train=True):
        self.samples_per_vhole = samples_per_vhole
        self.patch_size = patch_size
        self.hm_size = hm_size
        self.is_train = is_train

        self.vholes = []
        for e in entries:
            img = cv2.imread(str(e["path"]), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            for kp in e["keypoints"]:
                self.vholes.append({
                    "img": img,
                    "tip": kp["tip"],
                    "corner1": kp["corner1"],
                    "corner2": kp["corner2"],
                })

        self.length = len(self.vholes) * self.samples_per_vhole

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        vh_idx = idx % len(self.vholes)
        info = self.vholes[vh_idx]
        img = info["img"]

        points = [info["tip"], info["corner1"], info["corner2"]]
        tip = info["tip"]
        half = self.patch_size // 2

        # [V7] ±60px jitter (from V5_Tuned, was ±30 in V5)
        if self.is_train:
            cx = tip[0] + np.random.uniform(-JITTER, JITTER)
            cy = tip[1] + np.random.uniform(-JITTER, JITTER)
        else:
            cx, cy = tip[0], tip[1]

        # All keypoints relative to patch center
        rel_pts = [(p[0] - cx, p[1] - cy) for p in points]

        if self.is_train:
            ext_size = int(self.patch_size * 1.5)
            ext_patch = extract_patch(img, cx, cy, ext_size)

            angle = np.random.uniform(0, 360)
            rot_center = (ext_size // 2, ext_size // 2)
            M = cv2.getRotationMatrix2D(rot_center, angle, 1.0)
            rotated = cv2.warpAffine(
                ext_patch, M, (ext_size, ext_size),
                borderMode=cv2.BORDER_REFLECT,
            )

            margin = (ext_size - self.patch_size) // 2
            patch = rotated[margin:margin + self.patch_size,
                            margin:margin + self.patch_size]

            # Rotate all keypoint offsets
            rad = math.radians(angle)
            cos_a, sin_a = math.cos(rad), math.sin(rad)
            rel_pts = [
                (rx * cos_a + ry * sin_a, -rx * sin_a + ry * cos_a)
                for rx, ry in rel_pts
            ]

            patch = normalize_fine_patch(patch)
            patch *= np.random.uniform(0.8, 1.2)
            if np.random.rand() < 0.5:
                patch += np.random.normal(0, 0.03, patch.shape).astype(np.float32)
            patch = np.clip(patch, 0, 1)
        else:
            patch = extract_patch(img, cx, cy, self.patch_size)
            patch = normalize_fine_patch(patch)

        # Build 3-channel heatmap target at 96×96
        heatmap = np.zeros((NUM_KEYPOINTS, self.hm_size, self.hm_size), dtype=np.float32)
        for ch, (rx, ry) in enumerate(rel_pts):
            px = half + rx
            py = half + ry
            hm_x = px / self.patch_size * self.hm_size
            hm_y = py / self.patch_size * self.hm_size
            hm_x = np.clip(hm_x, 0, self.hm_size - 1)
            hm_y = np.clip(hm_y, 0, self.hm_size - 1)
            heatmap[ch] = make_gaussian_heatmap(self.hm_size, hm_x, hm_y)

        x = torch.from_numpy(patch).unsqueeze(0).float()
        target = torch.from_numpy(heatmap).float()  # (3, 96, 96)
        return x, target


# ============================================================================
# Loss: Weighted MSE + tip-weighted channels
# ============================================================================
def weighted_triangle_loss_v7(preds, targets):
    """
    [V7] Combines V6's channel weighting with V5_Tuned's peak-emphasis.
    """
    # Peak-weighted MSE per channel (from V5_Tuned)
    weight = torch.where(targets > PEAK_THRESHOLD, PEAK_WEIGHT, BG_WEIGHT)

    # Tip channel (ch 0) gets full weight, corners get reduced
    channel_weight = torch.ones_like(preds)
    channel_weight[:, 1:, :, :] = CORNER_LOSS_WEIGHT  # corners × 0.3

    combined_weight = weight * channel_weight
    loss = (combined_weight * (preds - targets) ** 2).mean()
    return loss


# ============================================================================
# Subpixel peak extraction
# ============================================================================
def get_heatmap_peak(heatmap_2d):
    """Extract subpixel peak from a single-channel heatmap."""
    hm_h, hm_w = heatmap_2d.shape
    peak_idx = np.unravel_index(np.argmax(heatmap_2d), heatmap_2d.shape)
    peak_y, peak_x = int(peak_idx[0]), int(peak_idx[1])
    peak_val = float(heatmap_2d[peak_y, peak_x])

    dx, dy = 0.0, 0.0
    if 1 <= peak_x < hm_w - 1:
        left = float(heatmap_2d[peak_y, peak_x - 1])
        right = float(heatmap_2d[peak_y, peak_x + 1])
        center = float(heatmap_2d[peak_y, peak_x])
        denom = 2.0 * (left + right - 2.0 * center)
        if abs(denom) > 1e-6:
            dx = np.clip((left - right) / denom, -0.5, 0.5)

    if 1 <= peak_y < hm_h - 1:
        top = float(heatmap_2d[peak_y - 1, peak_x])
        bottom = float(heatmap_2d[peak_y + 1, peak_x])
        center = float(heatmap_2d[peak_y, peak_x])
        denom = 2.0 * (top + bottom - 2.0 * center)
        if abs(denom) > 1e-6:
            dy = np.clip((top - bottom) / denom, -0.5, 0.5)

    return peak_x + dx, peak_y + dy, peak_val


def heatmap_to_coords(heatmap_3ch, coarse_x, coarse_y):
    """Extract 3 keypoints from 3ch heatmap → original image coords."""
    points = []
    confidences = []
    for ch in range(NUM_KEYPOINTS):
        hm_x, hm_y, conf = get_heatmap_peak(heatmap_3ch[ch])
        orig_x = coarse_x - PATCH_SIZE / 2.0 + hm_x / HM_SIZE * PATCH_SIZE
        orig_y = coarse_y - PATCH_SIZE / 2.0 + hm_y / HM_SIZE * PATCH_SIZE
        points.append((orig_x, orig_y))
        confidences.append(conf)
    return points, confidences


# ============================================================================
# Geometric refinement — V6 safety guard
# ============================================================================
def refine_tip_from_triangle(tip, corner1, corner2, alpha=REFINE_ALPHA):
    """Perpendicular bisector projection + blending."""
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

    refined_x = alpha * geo_x + (1 - alpha) * tip[0]
    refined_y = alpha * geo_y + (1 - alpha) * tip[1]
    return (refined_x, refined_y)


def triangle_angle_at_tip(tip, corner1, corner2):
    """[V7b] Compute angle (degrees) at tip vertex of triangle."""
    v1 = (corner1[0] - tip[0], corner1[1] - tip[1])
    v2 = (corner2[0] - tip[0], corner2[1] - tip[1])
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.hypot(*v1)
    mag2 = math.hypot(*v2)
    if mag1 < 1e-6 or mag2 < 1e-6:
        return 0.0
    cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cos_angle))


def maybe_refine_tip(raw_tip, corner1, corner2, confs):
    """
    [V7b] Enhanced safety-guarded refinement with triangle quality check.
    Falls back to raw tip when corners are unreliable or geometry is degenerate.
    """
    # Check 1: corner confidence
    if min(confs[1], confs[2]) < REFINE_MIN_CORNER_CONF:
        return raw_tip

    # Check 2: base length
    base_len = math.hypot(corner2[0] - corner1[0], corner2[1] - corner1[1])
    if base_len < REFINE_MIN_BASE:
        return raw_tip

    # Check 3: triangle angle at tip (degenerate triangle check)
    angle = triangle_angle_at_tip(raw_tip, corner1, corner2)
    if angle < REFINE_MIN_ANGLE_DEG or angle > REFINE_MAX_ANGLE_DEG:
        return raw_tip

    # Apply refinement with reduced alpha
    refined = refine_tip_from_triangle(raw_tip, corner1, corner2)
    shift = math.hypot(refined[0] - raw_tip[0], refined[1] - raw_tip[1])
    if shift > REFINE_MAX_SHIFT:
        return raw_tip
    return refined


# ============================================================================
# Training
# ============================================================================
def train_model(train_entries, device=DEVICE):
    train_ds = TrianglePoseDatasetV7(train_entries, is_train=True)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True,
    )

    model = TrianglePoseEstimatorV7(freeze_encoder=True).to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    n_params = sum(p.numel() for p in trainable_params)
    print(f"Trainable parameters: {n_params:,} (encoder frozen)")

    pbar = tqdm(range(EPOCHS), desc="Training Stage2")
    for epoch in pbar:
        model.train()
        total_loss = 0
        n_batch = 0
        for patches, targets in train_loader:
            patches = patches.to(device)
            targets = targets.to(device)

            preds = model(patches)
            loss = weighted_triangle_loss_v7(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batch += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batch, 1)
        pbar.set_postfix(loss=f"{avg_loss:.6f}")

    return model


# ============================================================================
# Inference — multi-candidate with Stage2 confidence selection
# ============================================================================
def predict_tips(model, img, yolo_model, device_str, device=DEVICE):
    """
    [V7] Full pipeline: YOLO coarse → multi-candidate → fine → safety-guard refine.
    """
    candidate_infos = yolo_coarse_detect(img, yolo_model, device_str)

    if len(candidate_infos) == 0:
        h, w = img.shape[:2]
        dummy = {"tip": (w/4, h/4), "raw_tip": (w/4, h/4),
                 "corner1": (w/4, h/4), "corner2": (w/4, h/4),
                 "tip_confidence": 0.0}
        return [dummy, dummy], []

    model.eval()
    best_preds = None
    best_score = -1.0
    best_coarse = None
    best_signal = max(info["signal_score"] for info in candidate_infos)

    with torch.no_grad():
        for info in candidate_infos:
            pair = info["pair"]
            preds = []
            total_confidence = 0.0

            for (coarse_x, coarse_y) in pair:
                patch = extract_patch(img, coarse_x, coarse_y, PATCH_SIZE)
                patch_norm = normalize_fine_patch(patch)
                patch_t = torch.from_numpy(patch_norm).unsqueeze(0).unsqueeze(0).to(device)

                heatmap_3ch = model(patch_t)[0].cpu().numpy()  # (3, 96, 96)
                points, confs = heatmap_to_coords(heatmap_3ch, coarse_x, coarse_y)

                total_confidence += confs[0]  # tip confidence

                raw_tip = points[0]
                corner1 = points[1]
                corner2 = points[2]

                # [V7 from V6] Safety-guarded refinement
                refined_tip = maybe_refine_tip(raw_tip, corner1, corner2, confs)

                preds.append({
                    "tip": refined_tip,
                    "raw_tip": raw_tip,
                    "corner1": corner1,
                    "corner2": corner2,
                    "tip_confidence": confs[0],
                })

            # [from V5] Score: Stage2 confidence + coarse prior
            coarse_prior = 0.05 * max(info["signal_score"], 0.0) / max(best_signal, 1.0)
            combined_score = total_confidence + coarse_prior

            if combined_score > best_score:
                best_score = combined_score
                best_preds = preds
                best_coarse = pair

    return best_preds, best_coarse


# ============================================================================
# Visualization
# ============================================================================
def visualize(img, predictions, gt_keypoints, coarse_locs, out_path):
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for clx, cly in coarse_locs:
        cv2.circle(vis, (int(clx), int(cly)), 12, (255, 150, 0), 2, cv2.LINE_AA)

    # GT: yellow stars for tips, small circles for corners
    for kp in gt_keypoints:
        tx, ty = kp["tip"]
        cv2.drawMarker(vis, (int(tx), int(ty)), (0, 255, 255), cv2.MARKER_STAR, 18, 2)
        for c in [kp["corner1"], kp["corner2"]]:
            cv2.circle(vis, (int(c[0]), int(c[1])), 4, (0, 200, 200), -1, cv2.LINE_AA)
            cv2.line(vis, (int(tx), int(ty)), (int(c[0]), int(c[1])), (0, 200, 200), 1, cv2.LINE_AA)

    # Predictions
    for i, pred in enumerate(predictions):
        tip = pred["tip"]
        c1, c2 = pred["corner1"], pred["corner2"]

        cv2.line(vis, (int(tip[0]), int(tip[1])), (int(c1[0]), int(c1[1])), (255, 100, 100), 1, cv2.LINE_AA)
        cv2.line(vis, (int(tip[0]), int(tip[1])), (int(c2[0]), int(c2[1])), (255, 100, 100), 1, cv2.LINE_AA)
        cv2.line(vis, (int(c1[0]), int(c1[1])), (int(c2[0]), int(c2[1])), (255, 100, 100), 1, cv2.LINE_AA)

        cv2.circle(vis, (int(c1[0]), int(c1[1])), 5, (255, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(vis, (int(c2[0]), int(c2[1])), 5, (255, 0, 0), -1, cv2.LINE_AA)

        cv2.circle(vis, (int(tip[0]), int(tip[1])), 8, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(vis, (int(tip[0]), int(tip[1])), 10, (0, 0, 255), 2, cv2.LINE_AA)
        label = f"V{i+1}:({tip[0]:.1f},{tip[1]:.1f})"
        cv2.putText(vis, label, (int(tip[0]) + 15, int(tip[1]) + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    for kp in gt_keypoints:
        gx, gy = kp["tip"]
        if predictions:
            dists = [math.hypot(gx - p["tip"][0], gy - p["tip"][1]) for p in predictions]
            best_idx = int(np.argmin(dists))
            best_dist = dists[best_idx]
            bp = predictions[best_idx]["tip"]
            cv2.line(vis, (int(gx), int(gy)), (int(bp[0]), int(bp[1])),
                     (0, 255, 255), 1, cv2.LINE_AA)
            mid_x = (int(gx) + int(bp[0])) // 2
            mid_y = (int(gy) + int(bp[1])) // 2
            cv2.putText(vis, f"{best_dist:.1f}px", (mid_x + 5, mid_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    cv2.imwrite(str(out_path), vis)


# ============================================================================
# Cross-Validation
# ============================================================================
def run_fold(test_group, device=DEVICE, data_dir=HV_DIR, out_prefix="vhole_cv_v7_hybrid"):
    data_dir = Path(data_dir)
    out_dir = Path(f"{out_prefix}_{test_group}")
    out_dir.mkdir(exist_ok=True)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        filename=str(out_dir / "training_log.txt"),
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
    )
    logger = logging.getLogger()

    print(f"\n{'=' * 70}")
    print(f" [V7 Hybrid] 🚀 Test Group: {test_group} 🚀")
    print(f" Stage1: YOLO (no top/bottom) | Stage2: 3ch 96×96 Triangle Pose")
    print(f" Data: {data_dir}")
    print(f"{'=' * 70}")

    # Load data
    all_entries = []
    for json_path in sorted(data_dir.glob("*.json")):
        name = json_path.stem.replace("_hv", "")
        img_path = data_dir / (json_path.stem + ".png")
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        keypoints = get_vhole_keypoints(str(json_path), img.shape)
        if len(keypoints) < 2:
            continue
        coords = [kp["tip"] for kp in keypoints]
        is_test = name.startswith(test_group)
        all_entries.append({
            "name": name, "path": img_path,
            "keypoints": keypoints, "coords": coords,
            "is_test": is_test,
        })

    train_entries = [e for e in all_entries if not e["is_test"]]
    test_entries = [e for e in all_entries if e["is_test"]]

    if not test_entries:
        print(f"No test data for {test_group}, skipping.")
        return None

    n_train_kps = sum(len(e["keypoints"]) * 3 for e in train_entries)
    print(f"Train: {len(train_entries)} images ({n_train_kps} keypoints)")
    print(f"Test:  {len(test_entries)} images")

    # ---- Stage 1: YOLO ----
    print(f"\n--- Stage 1: YOLO bbox detector (no top/bottom constraint) ---")
    yolo_model, device_str = train_yolo_stage1(
        train_entries, test_entries, out_dir, device=device,
    )

    # Stage 1 eval
    print(f"\n--- Stage 1 Coarse Eval ---")
    coarse_ok = 0
    for e in test_entries:
        img = cv2.imread(str(e["path"]), cv2.IMREAD_GRAYSCALE)
        cands = yolo_coarse_detect(img, yolo_model, device_str, top_pairs=1)
        gt_tips = e["coords"]
        if len(cands) >= 1:
            pair = cands[0]["pair"]
            errs = [min(math.hypot(gx-px, gy-py) for px, py in pair) for gx, gy in gt_tips]
            status = "OK" if all(err < 50 for err in errs) else "MISS"
            if status == "OK":
                coarse_ok += 1
            print(f"  {e['name']}: {[f'{er:.0f}px' for er in errs]} [{status}]")
        else:
            print(f"  {e['name']}: FAILED")
    print(f"  Coarse: {coarse_ok}/{len(test_entries)}")

    # ---- Stage 2: Train ----
    print(f"\n--- Stage 2: Training 3ch 96×96 Triangle Pose ---")
    model = train_model(train_entries)
    torch.save(model.state_dict(), str(out_dir / "tip_model.pth"))

    # ---- Full pipeline ----
    print(f"\n--- Full Pipeline (YOLO + 3ch 96×96 + Safety Refine) ---")
    results = []
    for e in tqdm(test_entries, desc=f"Inference CV={test_group}"):
        img = cv2.imread(str(e["path"]), cv2.IMREAD_GRAYSCALE)
        gt_kps = e["keypoints"]
        gt_tips = e["coords"]

        preds, coarse_locs = predict_tips(
            model, img, yolo_model, device_str, device=device,
        )

        errs = []
        for gx, gy in gt_tips:
            dists = [math.hypot(gx - p["tip"][0], gy - p["tip"][1]) for p in preds]
            errs.append(min(dists))

        raw_errs = []
        for gx, gy in gt_tips:
            dists = [math.hypot(gx - p["raw_tip"][0], gy - p["raw_tip"][1]) for p in preds]
            raw_errs.append(min(dists))

        ok = len(errs) >= 2 and all(err <= 5 for err in errs)
        results.append({
            "name": e["name"], "preds": preds,
            "coarse": coarse_locs if coarse_locs else [],
            "gt_kps": gt_kps, "errors": errs, "raw_errors": raw_errs, "ok": ok,
        })

        visualize(img, preds, gt_kps, coarse_locs if coarse_locs else [],
                  out_dir / f"{e['name']}_result.png")

    correct = sum(1 for r in results if r["ok"])
    all_errs = [err for r in results for err in r["errors"] if err < 9999]
    all_raw = [err for r in results for err in r["raw_errors"] if err < 9999]
    avg_err = np.mean(all_errs) if all_errs else 0
    max_err = max(all_errs) if all_errs else 0
    avg_raw = np.mean(all_raw) if all_raw else 0

    print(f"\n[V7 Fold Result: {test_group}]")
    print(f"Accuracy (<=5px): {correct}/{len(results)} ({100*correct/len(results):.1f}%)")
    print(f"Refined tip: mean={avg_err:.1f}px, max={max_err:.1f}px")
    print(f"Raw tip:     mean={avg_raw:.1f}px (for comparison)")
    for r in results:
        flag = "OK" if r["ok"] else "MISS"
        ref = [f'{e:.1f}' for e in r["errors"]]
        raw = [f'{e:.1f}' for e in r["raw_errors"]]
        print(f"  {r['name']}: refined={ref} raw={raw} [{flag}]")

    logger.info(f"Accuracy: {correct}/{len(results)} ({100*correct/len(results):.1f}%)")
    logger.info(f"Mean error: {avg_err:.1f}px (refined), {avg_raw:.1f}px (raw)")

    json_out = []
    for r in results:
        json_out.append({
            "name": r["name"],
            "tips": [{"x": p["tip"][0], "y": p["tip"][1],
                      "raw_x": p["raw_tip"][0], "raw_y": p["raw_tip"][1],
                      "c1": p["corner1"], "c2": p["corner2"]} for p in r["preds"]],
            "errors_px": r["errors"],
            "raw_errors_px": r["raw_errors"],
        })
    with open(out_dir / "final_results.json", "w") as f:
        json.dump(json_out, f, indent=2, ensure_ascii=False)

    return float(100 * correct / len(results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="V7 Hybrid: YOLO + 3ch 96×96 Triangle Pose + Safety Refine",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=["TYPE_C", "TYPE_B", "TYPE_A", "TYPE_D"],
        help="test groups to run",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=HV_DIR,
        help="directory containing *_hv.png and *_hv.json files",
    )
    parser.add_argument(
        "--out-prefix",
        default="vhole_cv_v7_hybrid",
        help="output folder prefix; final folders become <prefix>_<group>",
    )
    args = parser.parse_args()

    groups = args.groups
    fold_results = []
    for g in groups:
        acc = run_fold(g, data_dir=args.data_dir, out_prefix=args.out_prefix)
        if acc is not None:
            fold_results.append((g, acc))

    print(f"\n{'=' * 70}")
    print(f" 🎉 V7 Hybrid — Cross-Validation Results")
    print(f" Stage1: YOLO (no top/bottom) | Stage2: 3ch 96×96 + Safety Refine")
    print(f"{'=' * 70}")
    for group, acc in fold_results:
        print(f"  {group:10s}: {acc:.1f}%")
    if fold_results:
        avg_acc = np.mean([a for _, a in fold_results])
        print(f"  {'Average':10s}: {avg_acc:.1f}%")
    print(f"{'=' * 70}")
