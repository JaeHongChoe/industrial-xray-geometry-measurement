"""Command line runner for single-image and batch inference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import time
from pathlib import Path
from typing import Iterable

import cv2

from .pipeline import infer_image
from .vhole_v7 import V7ONNXDetector
from .visualization import write_review_image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SKIP_DIR_KEYWORDS = {".git", "__pycache__", "template", "templates", "템플릿"}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_model_dir() -> Path:
    return repo_root() / "models" / "v7_hv_onnx"


def slugify(text: str, max_len: int = 80) -> str:
    text = re.sub(r'[\\/:*?"<>|\s]+', "_", text.strip())
    text = re.sub(r"_+", "_", text).strip("_")
    return (text or "item")[:max_len]


def image_id(path: Path) -> str:
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:10]
    return f"{slugify(path.stem)}_{digest}"


def should_skip(path: Path, include_rotated_type_e: bool) -> bool:
    parts_lower = [part.lower() for part in path.parts]
    if any(part.startswith(".") for part in path.parts):
        return True
    if any(part.lower() == "first" for part in path.parts):
        return True
    if any(keyword in part for part in parts_lower for keyword in SKIP_DIR_KEYWORDS):
        return True
    return (not include_rotated_type_e) and any("rot90" in part.lower() for part in path.parts)


def collect_images(data_roots: Iterable[Path], include_rotated_type_e: bool = False) -> list[Path]:
    images: list[Path] = []
    seen: set[str] = set()
    for root in data_roots:
        root = Path(root)
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
                continue
            if should_skip(path, include_rotated_type_e):
                continue
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            images.append(path)
    return images


def write_summary_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_name",
        "dataset",
        "product_group",
        "status",
        "vhole_method",
        "vhole_axis",
        "vhole_rotation_to_axis_deg",
        "vhole_pair_score",
        "vhole_pair_distance_px",
        "vhole_error",
        "pin_status",
        "pin_count",
        "jig_status",
        "jig_has_jig",
        "plate_mesh_dx_px",
        "plate_mesh_dy_px",
        "plate_mesh_distance_px",
        "mesh_radius_px",
        "elapsed_sec",
        "review_image",
        "image_path",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_index_html(rows: list[dict], output_dir: Path) -> None:
    by_dataset: dict[str, list[dict]] = {}
    for row in rows:
        by_dataset.setdefault(str(row.get("dataset", "unknown")), []).append(row)

    cards = []
    for dataset, dataset_rows in sorted(by_dataset.items()):
        cards.append(f"<h2>{html.escape(dataset)} <span>{len(dataset_rows)} images</span></h2>")
        cards.append('<div class="grid">')
        for row in dataset_rows:
            review = html.escape(str(row.get("review_image", "")))
            status = html.escape(str(row.get("status", "")))
            method = html.escape(str(row.get("vhole_method", "")))
            axis = html.escape(str(row.get("vhole_axis", "")))
            name = html.escape(str(row.get("image_name", "")))
            err = html.escape(str(row.get("vhole_error") or ""))
            cards.append(
                f'<article class="card status-{status}">'
                f'<a href="{review}"><img src="{review}" loading="lazy" alt="{name}"></a>'
                f'<div class="meta"><strong>{name}</strong>'
                f'<span>{status} | {method} | axis {axis}</span>'
                f'<small>{err}</small></div></article>'
            )
        cards.append("</div>")

    body = "\n".join(cards)
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>X-ray Logic Review</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f5f6f8; color: #1f2933; }}
    header {{ padding: 24px 32px; background: #202833; color: white; }}
    header h1 {{ margin: 0 0 8px; font-size: 28px; }}
    header p {{ margin: 0; color: #cbd5e1; }}
    main {{ padding: 24px 32px 48px; }}
    h2 {{ margin: 28px 0 14px; font-size: 21px; }}
    h2 span {{ color: #64748b; font-size: 14px; font-weight: 500; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 18px; }}
    .card {{ background: white; border: 1px solid #d8dee8; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06); }}
    .card img {{ display: block; width: 100%; aspect-ratio: 1 / 1; object-fit: cover; background: #e5e7eb; }}
    .meta {{ padding: 12px 14px 14px; display: grid; gap: 5px; }}
    .meta strong {{ font-size: 14px; }}
    .meta span {{ font-size: 13px; color: #475569; }}
    .meta small {{ font-size: 12px; color: #b42318; min-height: 16px; }}
    .status-ok {{ border-color: #b7dfc1; }}
    .status-vhole_fail, .status-error {{ border-color: #f2b8b5; }}
  </style>
</head>
<body>
  <header>
    <h1>X-ray Logic Review</h1>
    <p>Generated review images for product-level inference analysis. See summary.csv for numeric details.</p>
  </header>
  <main>
    {body}
  </main>
</body>
</html>
"""
    (output_dir / "index.html").write_text(html_text, encoding="utf-8")


def write_intermediates(payload: dict, output_dir: Path, stem: str) -> None:
    arrays = payload["arrays"]
    inter_dir = output_dir / "intermediates"
    inter_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(inter_dir / f"{stem}_raw.png"), arrays["raw_gray"])
    cv2.imwrite(str(inter_dir / f"{stem}_processed.png"), arrays["processed_gray"])
    cv2.imwrite(str(inter_dir / f"{stem}_morph.png"), arrays["morph"])
    cv2.imwrite(str(inter_dir / f"{stem}_hv.png"), arrays["hv"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run X-ray 3-pin/V-hole inference.")
    parser.add_argument("--image", action="append", type=Path, default=None, help="Single image path. Can be passed multiple times.")
    parser.add_argument("--data-root", action="append", type=Path, default=None, help="Batch data root. Can be passed multiple times.")
    parser.add_argument("--model-dir", type=Path, default=default_model_dir())
    parser.add_argument("--output-dir", type=Path, default=repo_root() / "outputs" / "all_products")
    parser.add_argument("--pixel-size-mm", type=float, default=0.030769)
    parser.add_argument("--limit", type=int, default=0, help="Limit total images for quick checks. 0 means all.")
    parser.add_argument("--max-per-folder", type=int, default=0, help="Limit images per dataset folder. 0 means all.")
    parser.add_argument("--include-rotated-type-e", action="store_true", help="Include *_rot90* derived TYPE_E folders.")
    parser.add_argument("--write-intermediates", action="store_true", help="Also save raw/processed/morph/HV images.")
    return parser.parse_args()


def _select_images(args: argparse.Namespace) -> list[Path]:
    if args.image:
        return [Path(path) for path in args.image]
    if not args.data_root:
        raise SystemExit("Use --image for a single run or --data-root for a batch run.")

    images = collect_images(args.data_root, include_rotated_type_e=args.include_rotated_type_e)
    if args.max_per_folder > 0:
        counts: dict[str, int] = {}
        filtered = []
        for path in images:
            key = str(path.parent)
            count = counts.get(key, 0)
            if count >= args.max_per_folder:
                continue
            counts[key] = count + 1
            filtered.append(path)
        images = filtered
    if args.limit > 0:
        images = images[:args.limit]
    return images


def main() -> None:
    args = parse_args()
    images = _select_images(args)
    output_dir = Path(args.output_dir)
    review_root = output_dir / "review_images"
    output_dir.mkdir(parents=True, exist_ok=True)
    review_root.mkdir(parents=True, exist_ok=True)

    try:
        detector = V7ONNXDetector(args.model_dir)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from None
    rows: list[dict] = []
    full_results: list[dict] = []
    started = time.perf_counter()

    for idx, image_path in enumerate(images, start=1):
        print(f"[{idx}/{len(images)}] {image_path}")
        try:
            payload = infer_image(image_path, detector, pixel_size_mm=args.pixel_size_mm)
            dataset_slug = slugify(image_path.parent.name)
            stem = image_id(image_path)
            review_path = review_root / dataset_slug / f"{stem}_review.jpg"
            write_review_image(payload, review_path)
            if args.write_intermediates:
                write_intermediates(payload, output_dir / dataset_slug, stem)
            row = dict(payload["summary"])
            row["review_image"] = str(review_path.relative_to(output_dir))
            rows.append(row)
            full_results.append(payload["result"])
        except Exception as exc:
            rows.append({
                "image_name": image_path.name,
                "dataset": image_path.parent.name,
                "product_group": "unknown",
                "status": "error",
                "vhole_error": str(exc),
                "image_path": str(image_path),
            })
            print(f"  ERROR: {exc}")

    write_summary_csv(rows, output_dir / "summary.csv")
    (output_dir / "full_results.json").write_text(
        json.dumps(full_results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_index_html(rows, output_dir)
    elapsed = time.perf_counter() - started
    ok_count = sum(1 for row in rows if row.get("status") == "ok")
    print(f"Done: {ok_count}/{len(rows)} ok, elapsed={elapsed:.2f}s")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
