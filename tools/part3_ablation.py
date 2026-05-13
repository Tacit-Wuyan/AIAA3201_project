"""Mask-only ablation runner for Part 3.

This script reuses an existing raw-mask directory and compares several Part 3
refinement variants without rerunning SAM2 or ProPainter. It is designed for
fast evidence gathering in the report: raw SAM2, motion-core filtering,
motion-core + temporal recovery, and the full adaptive identity-continuity
version.
"""

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from part2_pipeline import ensure_dir, imread_any, imwrite_any, read_video, save_mask_video, sorted_image_paths
from part3_pipeline import adaptive_refine_masks, build_parser, binary_mask


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Part 3 mask-only ablation")
    ap.add_argument("--input-video", required=True)
    ap.add_argument("--raw-mask-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--output-figure", default="")
    ap.add_argument("--frames", default="8,24,42,62")
    ap.add_argument("--fps", type=float, default=None)

    ap.add_argument("--motion-core-percentile", type=float, default=85.0)
    ap.add_argument("--motion-core-dilate", type=int, default=25)
    ap.add_argument("--area-drop-ratio", type=float, default=0.45)
    ap.add_argument("--temporal-recovery-dilate", type=int, default=81)
    ap.add_argument("--inpaint-mask-dilate-extra", type=int, default=13)
    return ap.parse_args()


def default_refine_args() -> argparse.Namespace:
    args = build_parser().parse_args(["--input", "dummy.mp4", "--config", "dummy.yaml"])
    args.motion_core_percentile = 85.0
    args.motion_core_dilate = 25
    args.inpaint_mask_dilate_extra = 13
    args.inpaint_mask_close = 5
    return args


def copy_raw_masks(raw_mask_dir: str, out_dir: str) -> Dict[str, object]:
    ensure_dir(out_dir)
    areas: List[int] = []
    for idx, src in enumerate(sorted_image_paths(raw_mask_dir)):
        dst = os.path.join(out_dir, f"mask_{idx:05d}.png")
        shutil.copy2(src, dst)
        img = imread_any(dst, cv2.IMREAD_GRAYSCALE)
        areas.append(int((img > 0).sum()) if img is not None else 0)
    return {
        "frames": len(areas),
        "avg_raw_area": float(np.mean(areas)) if areas else 0.0,
        "avg_refined_area": float(np.mean(areas)) if areas else 0.0,
        "avg_motion_gate": 0.0,
        "avg_identity_added_area": 0.0,
        "avg_components": 0.0,
        "avg_kept": 0.0,
        "temporal_recoveries": 0,
        "temporal_clips": 0,
        "frame_stats": [{"raw_area": a, "refined_area": a} for a in areas],
    }


def summarize_variant(name: str, label: str, mask_dir: str, meta: Dict[str, object]) -> Dict[str, object]:
    areas = []
    for p in sorted_image_paths(mask_dir):
        img = imread_any(p, cv2.IMREAD_GRAYSCALE)
        areas.append(int((img > 0).sum()) if img is not None else 0)
    diffs = [abs(areas[i] - areas[i - 1]) for i in range(1, len(areas))]
    avg_area = float(np.mean(areas)) if areas else 0.0
    return {
        "variant": name,
        "label": label,
        "frames": len(areas),
        "avg_area": round(avg_area, 3),
        "min_area": int(min(areas)) if areas else 0,
        "max_area": int(max(areas)) if areas else 0,
        "area_cv": round(float(np.std(areas) / max(avg_area, 1.0)), 5) if areas else 0.0,
        "mean_abs_area_delta": round(float(np.mean(diffs)), 3) if diffs else 0.0,
        "avg_motion_gate": round(float(meta.get("avg_motion_gate", 0.0)), 4),
        "avg_identity_added_area": round(float(meta.get("avg_identity_added_area", 0.0)), 3),
        "temporal_recoveries": int(meta.get("temporal_recoveries", 0)),
        "temporal_clips": int(meta.get("temporal_clips", 0)),
    }


def render_mask(mask: np.ndarray) -> np.ndarray:
    gray = binary_mask(mask)
    out = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
    out[:, :, 2] = gray
    out[:, :, 1] = (gray * 0.45).astype(np.uint8)
    return out


def resize_cell(img: np.ndarray, width: int) -> np.ndarray:
    h, w = img.shape[:2]
    new_h = int(round(h * width / max(w, 1)))
    return cv2.resize(img, (width, new_h), interpolation=cv2.INTER_AREA)


def make_figure(
    frames: Sequence[np.ndarray],
    frame_indices: Sequence[int],
    variants: Sequence[Tuple[str, str, str]],
    out_path: str,
) -> None:
    cell_w = 245
    label_h = 32
    row_gap = 16
    col_gap = 9
    cols = [("Input", "", "input")] + [(label, mask_dir, "mask") for _, label, mask_dir in variants]

    rows: List[List[np.ndarray]] = []
    for idx in frame_indices:
        idx = min(max(idx, 0), len(frames) - 1)
        row = [resize_cell(frames[idx], cell_w)]
        for _, _, mask_dir in variants:
            mask_path = os.path.join(mask_dir, f"mask_{idx:05d}.png")
            mask = imread_any(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise RuntimeError(f"Missing mask for figure: {mask_path}")
            row.append(resize_cell(render_mask(mask), cell_w))
        rows.append(row)

    cell_h = max(img.shape[0] for row in rows for img in row)
    canvas_w = len(cols) * cell_w + (len(cols) - 1) * col_gap
    row_h = label_h + cell_h
    canvas_h = 82 + len(rows) * row_h + (len(rows) - 1) * row_gap
    canvas = np.full((canvas_h, canvas_w, 3), (246, 242, 232), np.uint8)
    cv2.putText(canvas, "Part 3 Ablation: Motion Core, Temporal Recovery, Identity Continuity", (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (35, 45, 48), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Mask-only comparison on the same raw SAM2 outputs. Smaller and steadier masks usually mean fewer static-background false positives.", (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (83, 96, 98), 1, cv2.LINE_AA)

    y0 = 82
    for r, idx in enumerate(frame_indices):
        y = y0 + r * (row_h + row_gap)
        x = 0
        for c, (title, _, _) in enumerate(cols):
            cv2.rectangle(canvas, (x, y), (x + cell_w, y + label_h), (222, 230, 223), -1)
            cv2.putText(canvas, title, (x + 8, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (38, 55, 56), 2, cv2.LINE_AA)
            if c == 0:
                cv2.putText(canvas, f"f={idx}", (x + cell_w - 52, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (99, 86, 46), 1, cv2.LINE_AA)
            img = rows[r][c]
            top = y + label_h
            pad = (cell_h - img.shape[0]) // 2
            canvas[top + pad:top + pad + img.shape[0], x:x + cell_w] = img
            cv2.rectangle(canvas, (x, top), (x + cell_w - 1, top + cell_h - 1), (193, 184, 157), 1)
            x += cell_w + col_gap

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_path, canvas)


def main() -> None:
    cli = parse_args()
    out_root = os.path.abspath(cli.output_dir)
    ensure_dir(out_root)

    frames, fps = read_video(os.path.abspath(cli.input_video))
    if not frames:
        raise RuntimeError("No frames loaded")
    fps = cli.fps if cli.fps is not None else fps

    base = default_refine_args()
    base.motion_core_percentile = cli.motion_core_percentile
    base.motion_core_dilate = cli.motion_core_dilate
    base.temporal_recovery_dilate = cli.temporal_recovery_dilate
    base.inpaint_mask_dilate_extra = cli.inpaint_mask_dilate_extra

    specs = [
        ("raw_sam2", "Raw SAM2", {"raw_copy": True}),
        ("motion_core", "Motion core", {
            "use_motion_core": True,
            "use_adaptive_threshold": False,
            "use_target_continuity": False,
            "area_drop_ratio": 0.0,
        }),
        ("motion_core_temporal", "Core + temporal", {
            "use_motion_core": True,
            "use_adaptive_threshold": False,
            "use_target_continuity": False,
            "area_drop_ratio": cli.area_drop_ratio,
        }),
        ("full_adaptive_identity", "Full adaptive", {
            "use_motion_core": True,
            "use_adaptive_threshold": True,
            "use_target_continuity": True,
            "area_drop_ratio": cli.area_drop_ratio,
        }),
    ]

    rows: List[Dict[str, object]] = []
    figure_variants: List[Tuple[str, str, str]] = []
    for name, label, overrides in specs:
        variant_dir = os.path.join(out_root, name)
        mask_dir = os.path.join(variant_dir, "masks")
        if os.path.exists(variant_dir):
            shutil.rmtree(variant_dir)
        ensure_dir(mask_dir)

        if overrides.get("raw_copy"):
            meta = copy_raw_masks(cli.raw_mask_dir, mask_dir)
        else:
            args = argparse.Namespace(**vars(base))
            for k, v in overrides.items():
                setattr(args, k, v)
            meta = adaptive_refine_masks(frames, cli.raw_mask_dir, mask_dir, args)

        save_mask_video(mask_dir, os.path.join(variant_dir, f"{name}_masks.mp4"), fps)
        with open(os.path.join(variant_dir, f"{name}_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        rows.append(summarize_variant(name, label, mask_dir, meta))
        figure_variants.append((name, label, mask_dir))

    summary_csv = os.path.join(out_root, "part3_ablation_summary.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    figure = cli.output_figure or os.path.join(out_root, "part3_ablation_comparison.png")
    frame_indices = [int(x.strip()) for x in cli.frames.split(",") if x.strip()]
    make_figure(frames, frame_indices, figure_variants, figure)
    print(json.dumps({"summary_csv": summary_csv, "figure": figure, "rows": rows}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
