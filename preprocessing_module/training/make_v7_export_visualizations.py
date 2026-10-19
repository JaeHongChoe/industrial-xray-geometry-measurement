#!/usr/bin/env python3
"""Generate portable visualization images for the exported V7 HV model."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

import run_coarse_fine_v7 as v7


IMAGE_SUFFIX = "_result.jpg"
ZOOM_SUFFIX = "_zoom.png"


def _load_font(size: int) -> ImageFont.ImageFont:
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


FONT_TITLE = _load_font(24)
FONT_BODY = _load_font(18)
FONT_SMALL = _load_font(14)


def _load_entries(data_dir: Path) -> list[dict]:
    entries = []
    for json_path in sorted(data_dir.glob("*.json")):
        img_path = data_dir / f"{json_path.stem}.png"
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        keypoints = v7.get_vhole_keypoints(str(json_path), img.shape)
        if len(keypoints) < 2:
            continue
        entries.append({
            "name": json_path.stem.replace("_hv", ""),
            "path": img_path,
            "json": json_path,
            "keypoints": keypoints,
            "tips": [kp["tip"] for kp in keypoints],
        })
    return entries


def _nearest_errors(preds: list[dict], gt_tips: list[tuple[float, float]]) -> list[float]:
    errors = []
    for gx, gy in gt_tips:
        if not preds:
            errors.append(9999.0)
            continue
        errors.append(min(math.hypot(gx - p["tip"][0], gy - p["tip"][1]) for p in preds))
    return errors


def _draw_zoom_panel(
    img: np.ndarray,
    name: str,
    preds: list[dict],
    gt_kps: list[dict],
    errors: list[float],
    out_path: Path,
    crop_size: int = 260,
    scale: int = 3,
) -> None:
    h, w = img.shape[:2]
    gt_tips = [kp["tip"] for kp in gt_kps]
    panels = []

    for idx, (gx, gy) in enumerate(gt_tips):
        if preds:
            best = min(preds, key=lambda p: math.hypot(gx - p["tip"][0], gy - p["tip"][1]))
            px, py = best["tip"]
            raw_x, raw_y = best.get("raw_tip", best["tip"])
        else:
            px = py = raw_x = raw_y = float("nan")

        cx = int(round((gx + px) / 2.0)) if np.isfinite(px) else int(round(gx))
        cy = int(round((gy + py) / 2.0)) if np.isfinite(py) else int(round(gy))
        half = crop_size // 2
        x1 = max(0, min(w - crop_size, cx - half))
        y1 = max(0, min(h - crop_size, cy - half))
        x2 = min(w, x1 + crop_size)
        y2 = min(h, y1 + crop_size)
        crop = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_GRAY2BGR)

        if preds and np.isfinite(px):
            cv2.circle(crop, (int(round(raw_x - x1)), int(round(raw_y - y1))), 8,
                       (255, 120, 0), 2, cv2.LINE_AA)
            cv2.drawMarker(crop, (int(round(px - x1)), int(round(py - y1))),
                           (0, 0, 255), cv2.MARKER_CROSS, 30, 3)
            cv2.circle(crop, (int(round(px - x1)), int(round(py - y1))), 10,
                       (0, 0, 255), 2, cv2.LINE_AA)
            cv2.line(crop, (int(round(gx - x1)), int(round(gy - y1))),
                     (int(round(px - x1)), int(round(py - y1))),
                     (0, 255, 255), 2, cv2.LINE_AA)

        cv2.drawMarker(crop, (int(round(gx - x1)), int(round(gy - y1))),
                       (0, 255, 0), cv2.MARKER_STAR, 28, 3)
        cv2.circle(crop, (int(round(gx - x1)), int(round(gy - y1))), 11,
                   (0, 255, 0), 2, cv2.LINE_AA)

        crop = cv2.resize(crop, (crop_size * scale, crop_size * scale),
                          interpolation=cv2.INTER_CUBIC)
        panels.append((crop, idx, errors[idx]))

    panel_w = crop_size * scale
    panel_h = crop_size * scale
    header_h = 92
    footer_h = 42
    gap = 18
    canvas_w = panel_w * len(panels) + gap * (len(panels) + 1)
    canvas_h = header_h + panel_h + footer_h
    canvas = Image.new("RGB", (canvas_w, canvas_h), (248, 249, 250))
    draw = ImageDraw.Draw(canvas)

    status = "OK" if all(e <= 5.0 for e in errors) else "MISS"
    color = (20, 120, 65) if status == "OK" else (180, 45, 45)
    draw.text((gap, 12), f"{name}  [{status}]", font=FONT_TITLE, fill=color)
    err_txt = ", ".join(f"{e:.2f}px" for e in errors)
    draw.text((gap, 48), f"errors: {err_txt}", font=FONT_BODY, fill=(35, 40, 45))
    draw.text((gap, 72), "green=GT tip   red=refined prediction   orange=raw prediction   yellow=line to nearest error",
              font=FONT_SMALL, fill=(80, 86, 94))

    for i, (crop, idx, err) in enumerate(panels):
        x = gap + i * (panel_w + gap)
        y = header_h
        canvas.paste(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)), (x, y))
        draw.rectangle((x, y, x + panel_w - 1, y + panel_h - 1), outline=(165, 172, 180), width=2)
        draw.text((x + 8, y + panel_h + 10), f"V{idx + 1}: {err:.2f}px",
                  font=FONT_BODY, fill=(35, 40, 45))

    canvas.save(out_path, quality=95)


def _make_contact_sheet(zoom_paths: list[Path], out_path: Path) -> None:
    thumbs = []
    for path in zoom_paths:
        img = Image.open(path).convert("RGB")
        img.thumbnail((520, 300), Image.Resampling.LANCZOS)
        thumbs.append((path.stem.replace(ZOOM_SUFFIX.replace(".png", ""), ""), img.copy()))

    if not thumbs:
        return

    cols = 3
    rows = math.ceil(len(thumbs) / cols)
    cell_w = 560
    cell_h = 350
    header_h = 66
    canvas = Image.new("RGB", (cols * cell_w, header_h + rows * cell_h), (248, 249, 250))
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 16), "V7 HV final model visualization contact sheet",
              font=FONT_TITLE, fill=(24, 32, 42))

    for idx, (name, img) in enumerate(thumbs):
        row, col = divmod(idx, cols)
        x = col * cell_w + 16
        y = header_h + row * cell_h + 12
        draw.text((x, y), name, font=FONT_SMALL, fill=(45, 52, 62))
        canvas.paste(img, (x, y + 24))
        draw.rectangle((x, y + 24, x + img.width - 1, y + 24 + img.height - 1),
                       outline=(176, 184, 193), width=1)

    canvas.save(out_path, quality=90)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="vhole_v7_data_hv")
    parser.add_argument("--export-dir", default="v7_hv_model_export")
    parser.add_argument("--max-images", type=int, default=0)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    export_dir = Path(args.export_dir)
    full_dir = export_dir / "visualizations" / "full_overlay_jpg"
    zoom_dir = export_dir / "visualizations" / "zoom_review"
    sheet_dir = export_dir / "visualizations" / "summary_sheets"
    for folder in [full_dir, zoom_dir, sheet_dir]:
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True, exist_ok=True)

    device = v7.DEVICE
    device_str = "0" if device.type == "cuda" else device.type
    yolo_model = YOLO(str(export_dir / "yolo_best.pt"))
    tip_model = v7.TrianglePoseEstimatorV7(freeze_encoder=True).to(device)
    state = torch.load(str(export_dir / "tip_model.pth"), map_location=device)
    tip_model.load_state_dict(state)
    tip_model.eval()

    entries = _load_entries(data_dir)
    if args.max_images > 0:
        entries = entries[:args.max_images]

    results = []
    zoom_paths = []
    for entry in entries:
        img = cv2.imread(str(entry["path"]), cv2.IMREAD_GRAYSCALE)
        preds, coarse_locs = v7.predict_tips(tip_model, img, yolo_model, device_str, device=device)
        errors = _nearest_errors(preds, entry["tips"])
        ok = len(errors) >= 2 and all(e <= 5.0 for e in errors)

        full_path = full_dir / f"{entry['name']}{IMAGE_SUFFIX}"
        zoom_path = zoom_dir / f"{entry['name']}{ZOOM_SUFFIX}"
        v7.visualize(img, preds, entry["keypoints"], coarse_locs if coarse_locs else [], full_path)
        _draw_zoom_panel(img, entry["name"], preds, entry["keypoints"], errors, zoom_path)
        zoom_paths.append(zoom_path)

        results.append({
            "name": entry["name"],
            "errors_px": errors,
            "ok_5px": ok,
            "predictions": [
                {
                    "tip": [float(p["tip"][0]), float(p["tip"][1])],
                    "raw_tip": [float(p["raw_tip"][0]), float(p["raw_tip"][1])],
                    "corner1": [float(p["corner1"][0]), float(p["corner1"][1])],
                    "corner2": [float(p["corner2"][0]), float(p["corner2"][1])],
                    "tip_confidence": float(p["tip_confidence"]),
                }
                for p in preds
            ],
        })

    all_errors = [e for r in results for e in r["errors_px"] if e < 9999]
    summary = {
        "num_images": len(results),
        "num_tips": len(all_errors),
        "ok_images_5px": sum(1 for r in results if r["ok_5px"]),
        "mean_error_px": float(np.mean(all_errors)) if all_errors else None,
        "median_error_px": float(np.median(all_errors)) if all_errors else None,
        "max_error_px": float(max(all_errors)) if all_errors else None,
        "results": results,
    }
    with open(export_dir / "visualizations" / "final_model_visualization_results.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    _make_contact_sheet(zoom_paths, sheet_dir / "final_model_zoom_contact_sheet.jpg")

    clean_src = Path("v7_three_way_visualizations_clean")
    if clean_src.exists():
        clean_dst = export_dir / "visualizations" / "three_way_review_sheets"
        if clean_dst.exists():
            shutil.rmtree(clean_dst)
        clean_dst.mkdir(parents=True, exist_ok=True)
        for file_path in clean_src.iterdir():
            if file_path.is_file():
                shutil.copy2(file_path, clean_dst / file_path.name)

    readme = export_dir / "visualizations" / "README.md"
    readme.write_text(
        "# V7 HV visualization bundle\n\n"
        "- `full_overlay_jpg/`: compressed full-image V7 overlays.\n"
        "- `zoom_review/`: per-image zoom panels, easier to inspect around each V-hole.\n"
        "- `summary_sheets/final_model_zoom_contact_sheet.jpg`: compact contact sheet.\n"
        "- `three_way_review_sheets/`: previous HV / RawPreGray / combined comparison sheets.\n"
        "- `final_model_visualization_results.json`: final exported model prediction/error values.\n",
        encoding="utf-8",
    )

    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()
