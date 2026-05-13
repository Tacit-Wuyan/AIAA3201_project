"""Part 3: adaptive motion-guided mask refinement.

This runner reuses the Part 2 mask/inpainting backends, but inserts a real
Part 3 stage between them:

1. Generate raw masks using a configurable backend such as SAM2.
2. Refine masks with motion-aware component filtering and adaptive morphology.
3. Inpaint with ProPainter/OpenCV using the refined masks.

The goal is to reduce over-masking in crowded scenes while preserving enough
coverage for dynamic targets.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from part2_pipeline import (
    build_inpaint_masks,
    compose_video_from_dir,
    ensure_dir,
    extract_video_to_dir,
    imread_any,
    imwrite_any,
    inpaint_opencv,
    parse_config,
    read_video,
    run_external_command,
    save_frames,
    save_mask_video,
    sorted_image_paths,
    write_video,
)


def binary_mask(mask: np.ndarray) -> np.ndarray:
    return (mask > 127).astype(np.uint8) * 255


def compute_motion_maps(frames: Sequence[np.ndarray]) -> List[np.ndarray]:
    if not frames:
        return []
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    maps: List[np.ndarray] = []
    for i in range(len(grays)):
        if len(grays) == 1:
            maps.append(np.zeros_like(grays[0], dtype=np.float32))
            continue
        if i == 0:
            prev, curr = grays[0], grays[1]
        else:
            prev, curr = grays[i - 1], grays[i]
        flow = cv2.calcOpticalFlowFarneback(
            prev,
            curr,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=21,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        mag = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2).astype(np.float32)
        maps.append(mag)
    return maps


def postprocess(mask: np.ndarray, close_ksize: int, open_ksize: int, min_area: int) -> np.ndarray:
    out = binary_mask(mask)
    if close_ksize > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k, iterations=1)
    if open_ksize > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k, iterations=1)
    if min_area > 0:
        num, labels, stats, _ = cv2.connectedComponentsWithStats((out > 0).astype(np.uint8), connectivity=8)
        clean = np.zeros_like(out)
        for cid in range(1, num):
            if int(stats[cid, cv2.CC_STAT_AREA]) >= min_area:
                clean[labels == cid] = 255
        out = clean
    return out


def mask_centroid(mask: np.ndarray) -> Optional[Tuple[float, float]]:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def adaptive_core_percentile(raw_area_ratio: float, args: argparse.Namespace) -> float:
    pct = float(args.motion_core_percentile)
    if not args.use_adaptive_threshold:
        return clamp(pct, args.motion_core_min_percentile, args.motion_core_max_percentile)

    if raw_area_ratio > args.target_area_ratio:
        denom = max(args.target_area_ratio, 1e-6)
        excess = clamp((raw_area_ratio - args.target_area_ratio) / denom, 0.0, 1.0)
        pct += args.area_high_percentile_boost * excess
    elif raw_area_ratio < args.min_target_area_ratio:
        denom = max(args.min_target_area_ratio, 1e-6)
        shortage = clamp((args.min_target_area_ratio - raw_area_ratio) / denom, 0.0, 1.0)
        pct -= args.area_low_percentile_relax * shortage
    return clamp(pct, args.motion_core_min_percentile, args.motion_core_max_percentile)


def component_identity_score(
    component: np.ndarray,
    centroid_xy: Tuple[float, float],
    target_state: Optional[Dict[str, object]],
    args: argparse.Namespace,
) -> float:
    if not args.use_target_continuity or not target_state:
        return 0.0

    h, w = component.shape[:2]
    diag = float(np.hypot(w, h))
    search_radius = max(8.0, diag * args.target_search_radius_ratio)
    pred = target_state.get("predicted_centroid")
    dist_score = 0.0
    if pred is not None:
        px, py = pred  # type: ignore[misc]
        cx, cy = centroid_xy
        dist = float(np.hypot(cx - float(px), cy - float(py)))
        dist_score = clamp(1.0 - dist / search_radius, 0.0, 1.0)

    overlap_score = 0.0
    prev_mask = target_state.get("mask")
    if isinstance(prev_mask, np.ndarray) and int((component > 0).sum()) > 0:
        if args.identity_overlap_dilate > 1:
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (args.identity_overlap_dilate, args.identity_overlap_dilate),
            )
            support = cv2.dilate(binary_mask(prev_mask), k, iterations=1)
        else:
            support = binary_mask(prev_mask)
        overlap = cv2.bitwise_and(binary_mask(component), support)
        overlap_score = float((overlap > 0).sum() / max(1, (component > 0).sum()))

    return clamp(
        args.identity_dist_weight * dist_score + args.identity_overlap_weight * overlap_score,
        0.0,
        1.0,
    )


def update_target_state(
    refined: np.ndarray,
    prev_state: Optional[Dict[str, object]],
    args: argparse.Namespace,
) -> Optional[Dict[str, object]]:
    centroid = mask_centroid(refined)
    if centroid is None:
        return prev_state

    velocity = (0.0, 0.0)
    if prev_state and prev_state.get("centroid") is not None:
        px, py = prev_state["centroid"]  # type: ignore[misc]
        old_vx, old_vy = prev_state.get("velocity", (0.0, 0.0))  # type: ignore[misc]
        new_vx = centroid[0] - float(px)
        new_vy = centroid[1] - float(py)
        alpha = clamp(args.velocity_smoothing, 0.0, 1.0)
        velocity = (
            alpha * float(old_vx) + (1.0 - alpha) * new_vx,
            alpha * float(old_vy) + (1.0 - alpha) * new_vy,
        )

    return {
        "mask": binary_mask(refined),
        "centroid": centroid,
        "velocity": velocity,
        "predicted_centroid": (centroid[0] + velocity[0], centroid[1] + velocity[1]),
        "area": int((refined > 0).sum()),
    }


def refine_one_mask(
    raw_mask: np.ndarray,
    motion_map: np.ndarray,
    args: argparse.Namespace,
    target_state: Optional[Dict[str, object]] = None,
) -> Tuple[np.ndarray, Dict[str, object]]:
    raw = postprocess(raw_mask, close_ksize=args.pre_close, open_ksize=args.pre_open, min_area=args.min_component_area)
    h, w = raw.shape[:2]
    raw_area = int((raw > 0).sum())
    raw_area_ratio = float(raw_area / max(1, h * w))
    motion_gate = args.static_motion_thresh
    supported = raw
    motion_core_area = 0
    supported_area = int((supported > 0).sum())

    if args.use_motion_core:
        inside = motion_map[raw > 0]
        if inside.size:
            if args.adaptive_threshold_after_lock and target_state is None:
                pct = clamp(
                    float(args.motion_core_percentile),
                    args.motion_core_min_percentile,
                    args.motion_core_max_percentile,
                )
            else:
                pct = adaptive_core_percentile(raw_area_ratio, args)
            motion_gate = max(args.static_motion_thresh, float(np.percentile(inside, pct)))
        motion_core = np.where((raw > 0) & (motion_map >= motion_gate), 255, 0).astype(np.uint8)
        motion_core = postprocess(
            motion_core,
            close_ksize=args.motion_core_close,
            open_ksize=args.motion_core_open,
            min_area=args.motion_core_min_area,
        )
        motion_core_area = int((motion_core > 0).sum())
        if motion_core_area >= args.motion_core_min_area:
            if args.motion_core_dilate > 1:
                k = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (args.motion_core_dilate, args.motion_core_dilate),
                )
                motion_support = cv2.dilate(motion_core, k, iterations=1)
            else:
                motion_support = motion_core
            supported = cv2.bitwise_and(raw, motion_support)
            supported = postprocess(
                supported,
                close_ksize=args.support_close,
                open_ksize=args.support_open,
                min_area=args.min_component_area,
            )
            supported_area = int((supported > 0).sum())
            if supported_area < args.min_component_area and args.keep_best_if_empty:
                supported = raw
                supported_area = int((supported > 0).sum())

    identity_added_area = 0
    if args.use_target_continuity and target_state and isinstance(target_state.get("mask"), np.ndarray):
        prev_target_area = int(target_state.get("area", 0))
        should_activate_identity = (
            supported_area < int(prev_target_area * args.identity_activation_ratio)
            or supported_area < args.identity_activation_min_area
        )
        if should_activate_identity:
            prev_mask = binary_mask(target_state["mask"])  # type: ignore[arg-type]
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (args.identity_support_dilate, args.identity_support_dilate),
            )
            identity_support = cv2.dilate(prev_mask, k, iterations=1)
            identity_region = cv2.bitwise_and(raw, identity_support)
            identity_added_area = int((identity_region > 0).sum())
            if identity_added_area >= args.min_component_area:
                supported = cv2.bitwise_or(supported, identity_region)
                supported = postprocess(
                    supported,
                    close_ksize=args.support_close,
                    open_ksize=args.support_open,
                    min_area=args.min_component_area,
                )
                supported_area = int((supported > 0).sum())

    num, labels, stats, centroids = cv2.connectedComponentsWithStats((supported > 0).astype(np.uint8), connectivity=8)

    comps: List[Dict[str, object]] = []
    for cid in range(1, num):
        area = int(stats[cid, cv2.CC_STAT_AREA])
        if area < args.min_component_area:
            continue
        comp = labels == cid
        local = motion_map[comp]
        score = float(np.percentile(local, args.motion_percentile)) if local.size else 0.0
        area_ratio = float(area / max(1, h * w))
        centroid_xy = (float(centroids[cid][0]), float(centroids[cid][1]))
        identity_score = component_identity_score(comp.astype(np.uint8) * 255, centroid_xy, target_state, args)
        comps.append({
            "cid": cid,
            "area": area,
            "area_ratio": area_ratio,
            "motion_score": score,
            "centroid": centroid_xy,
            "identity_score": identity_score,
        })

    if not comps:
        return np.zeros_like(raw), {
            "components": 0,
            "kept": 0,
            "kept_motion_scores": [],
            "kept_identity_scores": [],
            "motion_gate": round(float(motion_gate), 4),
            "motion_core_area": motion_core_area,
            "identity_added_area": identity_added_area,
            "supported_area": supported_area,
            "raw_area": raw_area,
            "refined_area": 0,
        }

    max_motion = max(float(c["motion_score"]) for c in comps)
    for comp in comps:
        motion_norm = float(comp["motion_score"]) / max(max_motion, 1e-6)
        identity_score = float(comp["identity_score"])
        comp["rank_score"] = motion_norm + args.target_continuity_weight * identity_score

    comps.sort(key=lambda x: (float(x["rank_score"]), float(x["motion_score"]), int(x["area"])), reverse=True)
    kept: List[Dict[str, object]] = []
    for comp in comps:
        score = float(comp["motion_score"])
        identity_score = float(comp["identity_score"])
        area_ratio = float(comp["area_ratio"])
        keep = score >= args.static_motion_thresh or identity_score >= args.identity_keep_thresh
        keep = keep and area_ratio <= args.max_component_area_ratio
        if keep:
            kept.append(comp)
        if args.keep_topk > 0 and len(kept) >= args.keep_topk:
            break

    # Avoid empty masks when motion estimates are weak: keep the strongest component.
    if not kept and args.keep_best_if_empty and comps:
        kept = [comps[0]]

    out = np.zeros_like(raw)
    for comp in kept:
        cid = int(comp["cid"])
        score = float(comp["motion_score"])
        component = np.where(labels == cid, 255, 0).astype(np.uint8)
        if score >= args.fast_motion_thresh and args.fast_dilate > 1:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.fast_dilate, args.fast_dilate))
            component = cv2.dilate(component, k, iterations=1)
        elif args.base_dilate > 1:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.base_dilate, args.base_dilate))
            component = cv2.dilate(component, k, iterations=1)
        out[component > 0] = 255

    out = postprocess(out, close_ksize=args.post_close, open_ksize=args.post_open, min_area=args.min_component_area)
    return out, {
        "components": len(comps),
        "kept": len(kept),
        "kept_motion_scores": [round(float(c["motion_score"]), 4) for c in kept],
        "kept_identity_scores": [round(float(c["identity_score"]), 4) for c in kept],
        "kept_rank_scores": [round(float(c["rank_score"]), 4) for c in kept],
        "motion_gate": round(float(motion_gate), 4),
        "motion_core_area": motion_core_area,
        "identity_added_area": identity_added_area,
        "supported_area": supported_area,
        "raw_area": raw_area,
        "refined_area": int((out > 0).sum()),
    }


def adaptive_refine_masks(
    frames: Sequence[np.ndarray],
    raw_mask_dir: str,
    refined_mask_dir: str,
    args: argparse.Namespace,
) -> Dict[str, object]:
    ensure_dir(refined_mask_dir)
    raw_paths = sorted_image_paths(raw_mask_dir)
    if len(raw_paths) != len(frames):
        raise RuntimeError(f"Raw mask/frame mismatch: masks={len(raw_paths)} frames={len(frames)}")

    motion_maps = compute_motion_maps(frames)
    stats: List[Dict[str, object]] = []
    prev_mask: Optional[np.ndarray] = None
    prev_area = 0
    target_state: Optional[Dict[str, object]] = None

    for idx, (mask_path, motion) in enumerate(zip(raw_paths, motion_maps)):
        raw = imread_any(mask_path, cv2.IMREAD_GRAYSCALE)
        if raw is None:
            raise RuntimeError(f"Failed to read raw mask: {mask_path}")
        refined, info = refine_one_mask(raw, motion, args, target_state=target_state)

        # Rescue sudden area drops. Motion-only masks can be too strict when the
        # target briefly pauses or changes pose; previous-frame support keeps the
        # refinement temporally coherent without returning to the full raw mask.
        cur_area = int((refined > 0).sum())
        if (
            prev_mask is not None
            and prev_area > 0
            and args.area_drop_ratio > 0
            and cur_area < int(prev_area * args.area_drop_ratio)
        ):
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (args.temporal_recovery_dilate, args.temporal_recovery_dilate),
            )
            support = cv2.dilate(prev_mask, k, iterations=1)
            recovered = cv2.bitwise_and(binary_mask(raw), support)
            recovered = postprocess(
                recovered,
                close_ksize=args.support_close,
                open_ksize=args.support_open,
                min_area=args.min_component_area,
            )
            recovered_area = int((recovered > 0).sum())
            max_recovered_area = int(max(prev_area * args.area_recovery_max_ratio, args.min_component_area))
            if recovered_area > cur_area and recovered_area <= max_recovered_area:
                refined = recovered
                cur_area = recovered_area
                info["area_recovered_by_temporal_support"] = True
                info["recovered_area"] = recovered_area

        # Limit sudden area explosions by intersecting with a loose previous support.
        if (
            prev_mask is not None
            and prev_area > 0
            and cur_area > int(prev_area * args.area_jump_ratio)
            and args.area_jump_ratio > 1.0
        ):
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.temporal_support_dilate, args.temporal_support_dilate))
            support = cv2.dilate(prev_mask, k, iterations=1)
            clipped = cv2.bitwise_and(refined, support)
            if int((clipped > 0).sum()) >= args.min_component_area:
                refined = clipped
                info["area_clipped_by_temporal_support"] = True

        info["refined_area"] = int((refined > 0).sum())
        dst = os.path.join(refined_mask_dir, f"mask_{idx:05d}.png")
        if not imwrite_any(dst, refined):
            raise RuntimeError(f"Failed to write refined mask: {dst}")
        target_state = update_target_state(refined, target_state, args)
        prev_mask = refined
        prev_area = int((refined > 0).sum())
        stats.append(info)

    return {
        "frames": len(stats),
        "avg_raw_area": float(np.mean([s.get("raw_area", 0) for s in stats])) if stats else 0.0,
        "avg_refined_area": float(np.mean([s.get("refined_area", 0) for s in stats])) if stats else 0.0,
        "avg_motion_gate": float(np.mean([s.get("motion_gate", 0) for s in stats])) if stats else 0.0,
        "avg_identity_added_area": float(np.mean([s.get("identity_added_area", 0) for s in stats])) if stats else 0.0,
        "avg_components": float(np.mean([s.get("components", 0) for s in stats])) if stats else 0.0,
        "avg_kept": float(np.mean([s.get("kept", 0) for s in stats])) if stats else 0.0,
        "temporal_recoveries": int(sum(1 for s in stats if s.get("area_recovered_by_temporal_support"))),
        "temporal_clips": int(sum(1 for s in stats if s.get("area_clipped_by_temporal_support"))),
        "frame_stats": stats,
    }


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser("Part 3: Adaptive Motion-Guided Mask Refinement")
    ap.add_argument("--input", type=str, required=True)
    ap.add_argument("--output-dir", type=str, default="outputs/part3_adaptive")
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--mask-backend", type=str, default="sam2", choices=["sam2", "custom"])
    ap.add_argument("--inpaint-backend", type=str, default="propainter", choices=["propainter", "opencv", "custom"])
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--fallback-opencv", action="store_true")
    ap.add_argument("--output-prefix", type=str, default="part3")

    ap.add_argument("--motion-percentile", type=float, default=75.0)
    ap.add_argument("--static-motion-thresh", type=float, default=1.2)
    ap.add_argument("--fast-motion-thresh", type=float, default=4.0)
    ap.add_argument("--keep-topk", type=int, default=1)
    ap.add_argument("--keep-best-if-empty", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--min-component-area", type=int, default=120)
    ap.add_argument("--max-component-area-ratio", type=float, default=0.55)
    ap.add_argument("--base-dilate", type=int, default=1)
    ap.add_argument("--fast-dilate", type=int, default=5)
    ap.add_argument("--pre-close", type=int, default=3)
    ap.add_argument("--pre-open", type=int, default=0)
    ap.add_argument("--post-close", type=int, default=3)
    ap.add_argument("--post-open", type=int, default=0)
    ap.add_argument("--area-jump-ratio", type=float, default=2.5)
    ap.add_argument("--area-drop-ratio", type=float, default=0.45)
    ap.add_argument("--area-recovery-max-ratio", type=float, default=2.8)
    ap.add_argument("--temporal-support-dilate", type=int, default=41)
    ap.add_argument("--temporal-recovery-dilate", type=int, default=81)

    ap.add_argument("--use-adaptive-threshold", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--target-area-ratio", type=float, default=0.30)
    ap.add_argument("--min-target-area-ratio", type=float, default=0.035)
    ap.add_argument("--area-high-percentile-boost", type=float, default=3.0)
    ap.add_argument("--area-low-percentile-relax", type=float, default=10.0)
    ap.add_argument("--motion-core-min-percentile", type=float, default=65.0)
    ap.add_argument("--motion-core-max-percentile", type=float, default=88.0)
    ap.add_argument("--adaptive-threshold-after-lock", action=argparse.BooleanOptionalAction, default=True)

    ap.add_argument("--use-motion-core", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--motion-core-percentile", type=float, default=82.0)
    ap.add_argument("--motion-core-dilate", type=int, default=29)
    ap.add_argument("--motion-core-close", type=int, default=5)
    ap.add_argument("--motion-core-open", type=int, default=3)
    ap.add_argument("--motion-core-min-area", type=int, default=80)
    ap.add_argument("--support-close", type=int, default=7)
    ap.add_argument("--support-open", type=int, default=0)

    ap.add_argument("--use-target-continuity", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--target-continuity-weight", type=float, default=0.75)
    ap.add_argument("--target-search-radius-ratio", type=float, default=0.28)
    ap.add_argument("--identity-support-dilate", type=int, default=69)
    ap.add_argument("--identity-overlap-dilate", type=int, default=55)
    ap.add_argument("--identity-activation-ratio", type=float, default=0.65)
    ap.add_argument("--identity-activation-min-area", type=int, default=1200)
    ap.add_argument("--identity-keep-thresh", type=float, default=0.20)
    ap.add_argument("--identity-dist-weight", type=float, default=0.55)
    ap.add_argument("--identity-overlap-weight", type=float, default=0.45)
    ap.add_argument("--velocity-smoothing", type=float, default=0.45)

    ap.add_argument("--inpaint-mask-dilate-extra", type=int, default=11)
    ap.add_argument("--inpaint-mask-close", type=int, default=5)
    ap.add_argument("--inpaint-mask-min-area", type=int, default=80)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    output_prefix = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in args.output_prefix.strip())
    if not output_prefix:
        raise RuntimeError("--output-prefix cannot be empty")

    cfg = parse_config(args.config)
    project_root = str(Path(__file__).resolve().parent)
    input_video_abs = os.path.abspath(args.input)
    out_root = os.path.abspath(args.output_dir)

    frame_dir = os.path.join(out_root, "frames")
    raw_mask_dir = os.path.join(out_root, "raw_masks")
    mask_dir = os.path.join(out_root, "masks")
    inpaint_mask_dir = os.path.join(out_root, "inpaint_masks")
    inpaint_dir = os.path.join(out_root, "inpaint_frames")
    inpaint_video = os.path.join(out_root, "external_inpaint.mp4")

    ensure_dir(out_root)
    for folder in (frame_dir, raw_mask_dir, mask_dir, inpaint_mask_dir, inpaint_dir):
        if os.path.exists(folder):
            shutil.rmtree(folder)
        ensure_dir(folder)

    frames, fps = read_video(input_video_abs, max_frames=args.max_frames)
    if not frames:
        raise RuntimeError("No input frames loaded")
    save_frames(frames, frame_dir, prefix="frame")

    working_input_video = input_video_abs
    if args.max_frames is not None:
        working_input_video = os.path.join(out_root, "input_trimmed.mp4")
        write_video(working_input_video, frames, fps)

    base_values = {
        "input_video": working_input_video,
        "frame_dir": frame_dir,
        "mask_dir": raw_mask_dir,
        "eval_mask_dir": raw_mask_dir,
        "inpaint_mask_dir": inpaint_mask_dir,
        "inpaint_dir": inpaint_dir,
        "output_dir": out_root,
        "device": args.device,
        "project_root": project_root,
        "python_exe": sys.executable,
        "inpaint_video": inpaint_video,
    }

    key = f"{args.mask_backend}_mask_command"
    cmd = str(cfg.get(key, "")).strip()
    if not cmd:
        raise RuntimeError(f"Missing config command: {key}")
    run_external_command(cmd, base_values)
    raw_mask_count = len(sorted_image_paths(raw_mask_dir))
    if raw_mask_count == 0:
        raise RuntimeError(f"Mask backend produced no masks in {raw_mask_dir}")

    refinement_meta = adaptive_refine_masks(frames, raw_mask_dir, mask_dir, args)
    refined_mask_count = len(sorted_image_paths(mask_dir))

    inpaint_mask_count = build_inpaint_masks(
        mask_dir=mask_dir,
        out_dir=inpaint_mask_dir,
        dilate_ksize=args.inpaint_mask_dilate_extra,
        close_ksize=args.inpaint_mask_close,
        min_area=args.inpaint_mask_min_area,
    )

    inpaint_values = dict(base_values)
    inpaint_values["mask_dir"] = inpaint_mask_dir
    inpaint_values["eval_mask_dir"] = mask_dir
    inpaint_values["inpaint_mask_dir"] = inpaint_mask_dir

    if args.inpaint_backend == "opencv":
        inpaint_count = inpaint_opencv(frame_dir, inpaint_mask_dir, inpaint_dir, method="telea")
    else:
        key = f"{args.inpaint_backend}_inpaint_command"
        cmd = str(cfg.get(key, "")).strip()
        if not cmd:
            raise RuntimeError(f"Missing config command: {key}")
        try:
            run_external_command(cmd, inpaint_values)
            inpaint_count = len(sorted_image_paths(inpaint_dir))
            if inpaint_count == 0 and os.path.exists(inpaint_video):
                inpaint_count = extract_video_to_dir(inpaint_video, inpaint_dir)
            if inpaint_count == 0:
                raise RuntimeError(f"Inpaint backend produced no frames in {inpaint_dir}")
        except Exception:
            if not args.fallback_opencv:
                raise
            inpaint_count = inpaint_opencv(frame_dir, inpaint_mask_dir, inpaint_dir, method="telea")

    output_video = os.path.join(out_root, f"{output_prefix}_inpainted.mp4")
    raw_mask_video = os.path.join(out_root, f"{output_prefix}_raw_masks.mp4")
    mask_video = os.path.join(out_root, f"{output_prefix}_masks.mp4")
    inpaint_mask_video = os.path.join(out_root, f"{output_prefix}_inpaint_masks.mp4")
    compose_video_from_dir(inpaint_dir, output_video, fps)
    save_mask_video(raw_mask_dir, raw_mask_video, fps)
    save_mask_video(mask_dir, mask_video, fps)
    save_mask_video(inpaint_mask_dir, inpaint_mask_video, fps)

    meta = {
        "input": input_video_abs,
        "working_input": working_input_video,
        "fps": fps,
        "n_input_frames": len(frames),
        "output_prefix": output_prefix,
        "mask_backend": args.mask_backend,
        "inpaint_backend": args.inpaint_backend,
        "raw_mask_frames": raw_mask_count,
        "refined_mask_frames": refined_mask_count,
        "inpaint_mask_frames": inpaint_mask_count,
        "inpaint_frames": inpaint_count,
        "raw_mask_dir": raw_mask_dir,
        "mask_dir": mask_dir,
        "inpaint_mask_dir": inpaint_mask_dir,
        "output_video": output_video,
        "raw_mask_video": raw_mask_video,
        "mask_video": mask_video,
        "inpaint_mask_video": inpaint_mask_video,
        "config_file": args.config,
        "refinement": refinement_meta,
        "args": vars(args),
    }
    with open(os.path.join(out_root, f"{output_prefix}_run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
