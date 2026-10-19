#!/usr/bin/env python3
"""V홀 검출 테스트 — preprocessing_module.vhole 기반

Raw Data 디렉토리의 템플릿 + vhole_hv_crops의 GT를 사용하여
preprocessing_module.vhole의 검출 정확도를 평가한다.

두 가지 모드:
  1. GT 기반 자동 템플릿 추출 (대표 이미지 잔차에서 직접 추출, 권장)
  2. 크롭 도구 템플릿 (Raw Data/템플릿/ 디렉토리 PNG + vhole_angles.json)

사용법:
  cd preprocessing_module
  python3 test_vhole.py
  python3 test_vhole.py --mode crop     # 크롭 도구 템플릿 사용
  python3 test_vhole.py --mode gt       # GT 자동 추출 (기본)
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

# preprocessing_module import
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from xray_3pin_detector.vhole import (
    classify_group,
    get_circle_from_hv,
    load_gt_coords,
    extract_templates_from_reference,
    compute_adaptive_band,
    load_template_from_directory,
    detect_vholes,
    draw_vhole_overlay,
    draw_residual_colormap,
    draw_polar_image,
    GROUP_REF_IMAGE,
    GROUP_TO_TEMPLATE_DIR,
    INNER_RADIUS_RATIO,
    OUTER_RADIUS_RATIO,
)


# ---------------------------------------------------------------------------
# 경로 설정
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]  # preprocessing_module/
HV_DIR = BASE_DIR / "vhole_hv_crops"
TEMPLATE_BASE = BASE_DIR / "Raw Data" / "템플릿"
OUT_DIR = Path(__file__).resolve().parent / "vhole_test_results"

ERR_THRESHOLD = 50  # px


# ---------------------------------------------------------------------------
# 유틸
# ---------------------------------------------------------------------------

def compute_errors(gt_coords, detections):
    """GT↔검출 최근접 거리 리스트"""
    errs = []
    for gx, gy in gt_coords:
        if detections:
            dists = [math.hypot(gx - d.x, gy - d.y) for d in detections]
            errs.append(min(dists))
        else:
            errs.append(9999)
    return errs


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="V홀 검출 테스트")
    parser.add_argument("--mode", choices=["gt", "crop"], default="gt",
                        help="템플릿 모드: gt=자동추출(권장), crop=크롭도구PNG")
    parser.add_argument("--out", type=str, default=str(OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)

    # ── 1. 이미지 분류 ──
    all_jsons = sorted(HV_DIR.glob("*.json"))
    groups = {}
    for jp in all_jsons:
        name = jp.stem.replace("_hv", "")
        img_path = HV_DIR / (jp.stem + ".png")
        if not img_path.exists():
            continue
        group = classify_group(name)
        if group not in groups:
            groups[group] = []
        groups[group].append({"name": name, "img": img_path, "json": jp})

    print(f"템플릿 모드: {args.mode}")
    print(f"출력 디렉토리: {out_dir}")
    print(f"\n=== 제품군별 이미지 ===")
    for g, imgs in sorted(groups.items()):
        print(f"  {g}: {len(imgs)}장")

    # ── 2. 제품군별 처리 ──
    all_results = []

    for group_name, img_list in sorted(groups.items()):
        if len(img_list) < 2:
            print(f"\n[{group_name}] 1장 — 스킵")
            continue

        # 대표 이미지 선택
        ref_name = GROUP_REF_IMAGE.get(group_name)
        ref = img_list[0]
        for entry in img_list:
            if entry["name"] == ref_name:
                ref = entry
                break
        infer_list = [e for e in img_list if e["name"] != ref["name"]]

        print(f"\n{'='*60}")
        print(f"[{group_name}] 대표: {ref['name']} | 추론: {len(infer_list)}장")

        # 대표 이미지 로드 + 원 검출
        ref_hv = cv2.imread(str(ref["img"]), cv2.IMREAD_GRAYSCALE)
        ref_cx, ref_cy, ref_r = get_circle_from_hv(ref_hv)
        ref_coords = load_gt_coords(str(ref["json"]))

        # GT 기반 adaptive band 계산 (GT/crop 모드 공통)
        inner_r, outer_r = compute_adaptive_band(ref_coords, ref_cx, ref_cy, ref_r)
        print(f"  circle: ({ref_cx:.0f},{ref_cy:.0f}) r={ref_r:.0f}")
        print(f"  밴드: {inner_r:.3f} ~ {outer_r:.3f}")

        # ── 템플릿 + GT angles 준비 ──
        if args.mode == "gt":
            # GT 위치에서 잔차 기반 자동 추출
            templates, gt_angles = extract_templates_from_reference(
                ref_hv, ref_cx, ref_cy, ref_r, ref_coords, inner_r, outer_r)
            for i, (ga, tmpl) in enumerate(zip(gt_angles, templates)):
                print(f"  템플릿 {i}: angle={ga:.1f}° shape={tmpl.shape}")
        else:
            # 크롭 도구 템플릿 로드
            tmpl_dir_name = GROUP_TO_TEMPLATE_DIR.get(group_name)
            if tmpl_dir_name is None:
                print(f"  [!] 템플릿 디렉토리 매핑 없음 — 스킵")
                continue
            tmpl_dir = TEMPLATE_BASE / tmpl_dir_name
            if not tmpl_dir.exists():
                print(f"  [!] 템플릿 디렉토리 없음: {tmpl_dir} — 스킵")
                continue
            templates, gt_angles, crop_band = load_template_from_directory(tmpl_dir)
            # adaptive band가 있으면 크롭 도구 기준 밴드 사용
            if crop_band is not None:
                inner_r, outer_r = crop_band
                print(f"  크롭 템플릿: {len(templates)}개, angles={[f'{a:.1f}' for a in gt_angles]}, band={inner_r:.3f}~{outer_r:.3f}")
            else:
                print(f"  크롭 템플릿: {len(templates)}개, angles={[f'{a:.1f}' for a in gt_angles]}")

        if not templates:
            print(f"  [!] 템플릿 없음 — 스킵")
            continue

        # 대표 이미지 잔차 시각화 저장
        ref_result = detect_vholes(ref_hv, templates, gt_angles,
                                    ref_cx, ref_cy, ref_r, inner_r, outer_r)
        if ref_result.residual is not None:
            cv2.imwrite(str(out_dir / f"ref_{group_name}_residual.png"),
                        draw_residual_colormap(ref_result.residual))
            cv2.imwrite(str(out_dir / f"ref_{group_name}_polar.png"),
                        draw_polar_image(ref_result.polar))

        # ── 추론 ──
        correct = 0
        group_results = []

        for entry in infer_list:
            hv = cv2.imread(str(entry["img"]), cv2.IMREAD_GRAYSCALE)
            if hv is None:
                continue

            # 검출 실행
            result = detect_vholes(hv, templates, gt_angles,
                                    inner_ratio=inner_r, outer_ratio=outer_r)

            # GT 대비 오차 계산
            gt_coords = load_gt_coords(str(entry["json"]))
            errs = compute_errors(gt_coords, result.detections)

            ok = len(errs) >= 2 and all(e <= ERR_THRESHOLD for e in errs)
            if ok:
                correct += 1

            status = "O" if ok else "X"
            err_str = ", ".join(f"{e:.0f}px" for e in errs)
            sc_str = ", ".join(f"{d.score:.3f}" for d in result.detections)
            print(f"  [{status}] {entry['name']}: err=[{err_str}]  score=[{sc_str}]")

            all_results.append({
                "group": group_name, "name": entry["name"],
                "detections": [{"x": d.x, "y": d.y, "angle": d.angle_deg,
                                "score": d.score} for d in result.detections],
                "gt": gt_coords, "errors": errs, "ok": ok,
            })

            # 시각화 저장 (3장: overlay, polar, residual)
            vis = draw_vhole_overlay(hv, result, gt_coords)
            cv2.imwrite(str(out_dir / f"{group_name}_{entry['name']}.png"), vis)

            if result.residual is not None:
                cv2.imwrite(
                    str(out_dir / f"{group_name}_{entry['name']}_residual.png"),
                    draw_residual_colormap(result.residual))

        total_infer = len(infer_list)
        pct = 100 * correct / total_infer if total_infer > 0 else 0
        print(f"  → 정확도: {correct}/{total_infer} ({pct:.1f}%)")

    # ── 3. 전체 요약 ──
    print(f"\n{'='*60}")
    print("=== 전체 요약 ===")
    total = len(all_results)
    total_ok = sum(1 for r in all_results if r["ok"])
    all_errs = [e for r in all_results for e in r["errors"] if e < 9999]

    if total > 0:
        print(f"  전체: {total_ok}/{total} ({100*total_ok/total:.1f}%)")
        if all_errs:
            print(f"  평균 오차: {np.mean(all_errs):.1f}px, "
                  f"중간값: {np.median(all_errs):.1f}px, "
                  f"최대: {max(all_errs):.1f}px")

    # 제품군별 요약
    for g in sorted(set(r["group"] for r in all_results)):
        gr = [r for r in all_results if r["group"] == g]
        gok = sum(1 for r in gr if r["ok"])
        gerrs = [e for r in gr for e in r["errors"] if e < 9999]
        avg_e = np.mean(gerrs) if gerrs else 0
        print(f"    {g}: {gok}/{len(gr)} (avg={avg_e:.1f}px)")

    # 실패 목록
    failures = [r for r in all_results if not r["ok"]]
    if failures:
        print(f"\n  실패 목록:")
        for r in failures:
            err_str = ", ".join(f"{e:.0f}px" for e in r["errors"])
            print(f"    {r['group']}/{r['name']}: [{err_str}]")

    # JSON 저장
    json_out = [{"group": r["group"], "name": r["name"],
                 "vholes": [{"x": d["x"], "y": d["y"],
                             "score": d["score"]} for d in r["detections"]],
                 "errors_px": r["errors"], "ok": r["ok"]}
                for r in all_results]
    with open(out_dir / "results.json", "w") as f:
        json.dump(json_out, f, indent=2, ensure_ascii=False)

    print(f"\n출력: {out_dir}")


if __name__ == "__main__":
    main()
