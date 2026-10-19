#!/usr/bin/env python3
"""Export the V7 HV two-stage V-hole model bundle to ONNX.

This script produces browser-friendly model artifacts:

- yolo_vhole.onnx: stage-1 coarse V-hole detector
- tip_model.onnx: stage-2 3-keypoint heatmap refiner

The stage-2 architecture mirrors TrianglePoseEstimatorV7 from
run_coarse_fine_v7.py, but instantiates the timm encoder with
pretrained=False so export does not depend on network access.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnx
import torch
import torch.nn as nn
import timm


NUM_KEYPOINTS = 3
PATCH_SIZE = 192
HM_SIZE = 96


class TrianglePoseEstimatorV7Export(nn.Module):
    """ONNX-exportable copy of the trained V7 stage-2 model."""

    def __init__(self, num_keypoints: int = NUM_KEYPOINTS):
        super().__init__()
        self.encoder = timm.create_model(
            "resnet18",
            pretrained=False,
            in_chans=1,
            features_only=True,
            out_indices=(0, 1, 2, 3, 4),
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(256 + 256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128 + 128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(64 + 64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Conv2d(32 + 64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_keypoints, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f0, f1, f2, f3, f4 = self.encoder(x)
        x = self.up4(f4)
        x = self.up3(torch.cat([x, f3], dim=1))
        x = self.up2(torch.cat([x, f2], dim=1))
        x = self.up1(torch.cat([x, f1], dim=1))
        return self.head(torch.cat([x, f0], dim=1))


def export_tip_model(export_dir: Path, opset: int) -> Path:
    pth_path = export_dir / "tip_model.pth"
    onnx_path = export_dir / "tip_model.onnx"
    model = TrianglePoseEstimatorV7Export().eval()
    state = torch.load(pth_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    dummy = torch.zeros(1, 1, PATCH_SIZE, PATCH_SIZE, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["patch"],
        output_names=["heatmaps"],
        dynamic_axes=None,
    )
    # PyTorch's new exporter may emit a sidecar .onnx.data file. Merge it back
    # into a single ONNX file because the customer-facing prototype should be
    # easy to package into one HTML bundle.
    model_proto = onnx.load(onnx_path, load_external_data=True)
    onnx.save_model(model_proto, onnx_path, save_as_external_data=False)
    sidecar = onnx_path.with_suffix(onnx_path.suffix + ".data")
    if sidecar.exists():
        sidecar.unlink()
    onnx.checker.check_model(onnx.load(onnx_path))
    return onnx_path


def export_yolo_model(export_dir: Path, opset: int) -> Path:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics is required for YOLO ONNX export") from exc

    pt_path = export_dir / "yolo_best.pt"
    model = YOLO(str(pt_path))
    exported = model.export(
        format="onnx",
        imgsz=1280,
        opset=opset,
        dynamic=False,
        simplify=False,
        nms=False,
    )
    exported_path = Path(exported)
    target_path = export_dir / "yolo_vhole.onnx"
    if exported_path.resolve() != target_path.resolve():
        target_path.write_bytes(exported_path.read_bytes())
    onnx.checker.check_model(onnx.load(target_path))
    return target_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export V7 HV V-hole models to ONNX.")
    parser.add_argument("--export-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--skip-yolo", action="store_true")
    parser.add_argument("--skip-tip", action="store_true")
    args = parser.parse_args()

    export_dir = args.export_dir.resolve()
    artifacts: dict[str, str] = {}
    if not args.skip_tip:
        tip_path = export_tip_model(export_dir, args.opset)
        artifacts["stage2_tip_onnx"] = tip_path.name
        print(f"[ok] exported {tip_path}")
    if not args.skip_yolo:
        yolo_path = export_yolo_model(export_dir, args.opset)
        artifacts["stage1_yolo_onnx"] = yolo_path.name
        print(f"[ok] exported {yolo_path}")

    manifest_path = export_dir / "model_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.setdefault("onnx", {}).update(
        {
            **artifacts,
            "opset": args.opset,
            "stage2_input": [1, 1, PATCH_SIZE, PATCH_SIZE],
            "stage2_output": [1, NUM_KEYPOINTS, HM_SIZE, HM_SIZE],
            "stage1_input_imgsz": 1280,
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
