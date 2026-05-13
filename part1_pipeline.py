import argparse
import importlib
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


DYNAMIC_COCO_IDS = {0, 1, 2, 3, 5, 7}
COCO_CLASS_ALIASES: Dict[str, int] = {
    "person": 0,
    "bicycle": 1,
    "bike": 1,
    "car": 2,
    "motorcycle": 3,
    "motorbike": 3,
    "bus": 5,
    "truck": 7,
    "sports ball": 32,
    "ball": 32,
    "tennis racket": 38,
    "racket": 38,
}


def parse_target_classes(spec: str) -> List[int]:
    ids: List[int] = []
    for raw in spec.split(","):
        item = raw.strip().lower()
        if not item:
            continue
        if item.isdigit():
            ids.append(int(item))
        elif item in COCO_CLASS_ALIASES:
            ids.append(COCO_CLASS_ALIASES[item])
        else:
            valid = ", ".join(sorted(COCO_CLASS_ALIASES))
            raise ValueError(f"Unknown target class '{raw}'. Use COCO id or one of: {valid}")
    if not ids:
        raise ValueError("--target-classes resolved to an empty list")
    return sorted(set(ids))


@dataclass
class DetectionResult:
    masks: List[np.ndarray]


class MaskExtractor:
    def __init__(self, conf: float = 0.25, device: str = "cpu", target_ids: Optional[Sequence[int]] = None):
        self.conf = conf
        self.device = device
        self.target_ids = set(target_ids or DYNAMIC_COCO_IDS)
        self._mode = "mog2"
        self._yolo = None
        self._mog2 = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=25, detectShadows=False)
        self._init_yolo()

    def _init_yolo(self) -> None:
        try:
            module = importlib.import_module("ultralytics")
            YOLO = getattr(module, "YOLO")
        except Exception:
            return
        try:
            # Keep Ultralytics settings inside current workspace when possible.
            os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(os.getcwd(), "Ultralytics"))
            self._yolo = YOLO("yolov8n-seg.pt")
            self._mode = "yolo"
        except Exception:
            self._yolo = None
            self._mode = "mog2"

    def extract(self, frame_bgr: np.ndarray) -> DetectionResult:
        if self._mode == "yolo" and self._yolo is not None:
            return self._extract_yolo(frame_bgr)
        return self._extract_mog2(frame_bgr)

    def _extract_yolo(self, frame_bgr: np.ndarray) -> DetectionResult:
        results = self._yolo.predict(frame_bgr, conf=self.conf, verbose=False, device=self.device)
        if not results:
            return DetectionResult(masks=[])
        res = results[0]
        if res.masks is None or res.boxes is None:
            return DetectionResult(masks=[])
        classes = res.boxes.cls.detach().cpu().numpy().astype(np.int32)
        mask_data = res.masks.data.detach().cpu().numpy()
        h, w = frame_bgr.shape[:2]
        out_masks: List[np.ndarray] = []
        for cls_id, m in zip(classes, mask_data):
            if int(cls_id) not in self.target_ids:
                continue
            resized = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            out_masks.append((resized > 0.5).astype(np.uint8))
        return DetectionResult(masks=out_masks)

    def _extract_mog2(self, frame_bgr: np.ndarray) -> DetectionResult:
        fg = self._mog2.apply(frame_bgr)
        fg = cv2.medianBlur(fg, 5)
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
        masks: List[np.ndarray] = []
        for i in range(1, num_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < 120:
                continue
            m = (labels == i).astype(np.uint8)
            masks.append(m)
        return DetectionResult(masks=masks)


def make_motion_dynamic_mask(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    masks: Sequence[np.ndarray],
    flow: Optional[np.ndarray] = None,
    min_points: int = 12,
    motion_thresh: float = 1.5,
    dense_motion_thresh: float = 0.8,
) -> np.ndarray:
    h, w = curr_gray.shape[:2]
    final_mask = np.zeros((h, w), dtype=np.uint8)
    if flow is None:
        flow = compute_dense_flow(prev_gray, curr_gray)
    dense_mag = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)
    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
    )
    for m in masks:
        mu8 = (m > 0).astype(np.uint8)
        if int(mu8.sum()) == 0:
            continue
        is_dynamic = False
        p0 = cv2.goodFeaturesToTrack(
            prev_gray,
            maxCorners=180,
            qualityLevel=0.01,
            minDistance=4,
            mask=(mu8 * 255),
        )
        if p0 is not None and len(p0) >= min_points:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None, **lk_params)
            if p1 is not None and st is not None:
                good = st[:, 0] == 1
                if int(good.sum()) >= min_points:
                    disp = p1[good] - p0[good]
                    mag = np.sqrt((disp[:, 0, 0] ** 2) + (disp[:, 0, 1] ** 2))
                    if float(np.median(mag)) >= motion_thresh:
                        is_dynamic = True

        if not is_dynamic:
            local_mag = dense_mag[mu8 > 0]
            if local_mag.size > 0 and float(np.percentile(local_mag, 75)) >= dense_motion_thresh:
                is_dynamic = True

        if is_dynamic:
            final_mask[mu8 > 0] = 255
    return final_mask


def dilate_mask(mask: np.ndarray, ksize: int = 9) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.dilate(mask, kernel, iterations=1)


def compute_dense_flow(prev_gray: np.ndarray, curr_gray: np.ndarray) -> np.ndarray:
    return cv2.calcOpticalFlowFarneback(
        prev_gray,
        curr_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=21,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )


def merge_instance_masks(masks: Sequence[np.ndarray], shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape
    out = np.zeros((h, w), dtype=np.uint8)
    for m in masks:
        out[m > 0] = 255
    return out


def warp_mask_with_flow(prev_mask: np.ndarray, flow: np.ndarray) -> np.ndarray:
    h, w = prev_mask.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = grid_x - flow[:, :, 0]
    map_y = grid_y - flow[:, :, 1]
    warped = cv2.remap(prev_mask, map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    return ((warped > 127).astype(np.uint8) * 255)


def refine_binary_mask(
    mask: np.ndarray,
    close_ksize: int = 7,
    open_ksize: int = 3,
    min_area: int = 120,
) -> np.ndarray:
    out = (mask > 0).astype(np.uint8) * 255
    if close_ksize > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k)
    if open_ksize > 1:
        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k2)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((out > 0).astype(np.uint8), connectivity=8)
    clean = np.zeros_like(out)
    for i in range(1, num_labels):
        if int(stats[i, cv2.CC_STAT_AREA]) >= min_area:
            clean[labels == i] = 255
    return clean


def temporal_borrow_fill(
    frames: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    max_radius: int = 30,
) -> List[np.ndarray]:
    n = len(frames)
    out = [f.copy() for f in frames]
    for i in range(n):
        if masks[i].max() == 0:
            continue
        curr = out[i]
        m = masks[i] > 0
        ys, xs = np.where(m)
        for y, x in zip(ys, xs):
            filled = False
            for r in range(1, max_radius + 1):
                p = i - r
                q = i + r
                if p >= 0 and masks[p][y, x] == 0:
                    curr[y, x] = frames[p][y, x]
                    filled = True
                    break
                if q < n and masks[q][y, x] == 0:
                    curr[y, x] = frames[q][y, x]
                    filled = True
                    break
            if not filled:
                curr[y, x] = curr[y, x]
        out[i] = curr
    return out


def spatial_inpaint_fallback(
    frames: Sequence[np.ndarray], masks: Sequence[np.ndarray], method: str = "telea"
) -> List[np.ndarray]:
    flag = cv2.INPAINT_TELEA if method.lower() == "telea" else cv2.INPAINT_NS
    out = []
    for frame, mask in zip(frames, masks):
        if mask.max() == 0:
            out.append(frame.copy())
            continue
        repaired = cv2.inpaint(frame, mask, 3, flag)
        out.append(repaired)
    return out


def merge_temporal_and_spatial(
    temporal_frames: Sequence[np.ndarray], spatial_frames: Sequence[np.ndarray], masks: Sequence[np.ndarray]
) -> List[np.ndarray]:
    out = []
    for t, s, m in zip(temporal_frames, spatial_frames, masks):
        if m.max() == 0:
            out.append(t.copy())
            continue

        # Use spatial inpainting as the primary content inside mask,
        # then feather near boundary to reduce misalignment seams.
        mk = (m > 0).astype(np.uint8) * 255
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        inner = cv2.erode(mk, k, iterations=1)
        outer = cv2.dilate(mk, k, iterations=1)
        ring = cv2.subtract(outer, inner)

        merged = t.copy()
        merged[mk > 0] = s[mk > 0]
        if int((ring > 0).sum()) > 0:
            alpha = (ring.astype(np.float32) / 255.0)[:, :, None] * 0.6
            blended = s.astype(np.float32) * alpha + t.astype(np.float32) * (1.0 - alpha)
            merged[ring > 0] = blended.astype(np.uint8)[ring > 0]

        out.append(merged.astype(np.uint8))
    return out


def read_video(path: str, max_frames: Optional[int] = None) -> Tuple[List[np.ndarray], float]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
    frames: List[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
        if max_frames is not None and len(frames) >= max_frames:
            break
    cap.release()
    return frames, fps


def write_video(path: str, frames: Sequence[np.ndarray], fps: float) -> None:
    if not frames:
        raise RuntimeError("No frames to write.")
    h, w = frames[0].shape[:2]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()


def save_masks(path: str, masks: Sequence[np.ndarray], fps: float) -> None:
    if not masks:
        return
    h, w = masks[0].shape[:2]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for m in masks:
        vis = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
        writer.write(vis)
    writer.release()


def imwrite_any(path: str, image: np.ndarray) -> bool:
    ext = os.path.splitext(path)[1] or ".png"
    try:
        ok, buf = cv2.imencode(ext, image)
        if not ok:
            return False
        buf.tofile(path)
        return True
    except Exception:
        return bool(cv2.imwrite(path, image))


def save_mask_frames(mask_dir: str, masks: Sequence[np.ndarray]) -> int:
    os.makedirs(mask_dir, exist_ok=True)
    count = 0
    for i, mask in enumerate(masks):
        path = os.path.join(mask_dir, f"mask_{i:05d}.png")
        if not imwrite_any(path, (mask > 0).astype(np.uint8) * 255):
            raise RuntimeError(f"Failed to write mask frame: {path}")
        count += 1
    return count


def expand_masks_for_inpaint(masks: Sequence[np.ndarray], extra_ksize: int) -> List[np.ndarray]:
    """Use tighter masks for evaluation and larger masks for visual inpainting."""
    if extra_ksize <= 1:
        return [(m > 0).astype(np.uint8) * 255 for m in masks]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (extra_ksize, extra_ksize))
    return [cv2.dilate((m > 0).astype(np.uint8) * 255, kernel, iterations=1) for m in masks]


def run(args: argparse.Namespace) -> None:
    frames, fps = read_video(args.input, args.max_frames)
    if len(frames) < 2:
        raise RuntimeError("Need at least 2 frames.")

    target_ids = parse_target_classes(args.target_classes)
    extractor = MaskExtractor(conf=args.conf, device=args.device, target_ids=target_ids)
    gray_prev = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)

    first_det = extractor.extract(frames[0])
    first_raw = merge_instance_masks(first_det.masks, gray_prev.shape[:2])
    if first_raw.max() > 0:
        first_mask = dilate_mask(first_raw, ksize=args.dilate)
        first_mask = refine_binary_mask(
            first_mask,
            close_ksize=args.close_ksize,
            open_ksize=args.open_ksize,
            min_area=args.min_mask_area,
        )
    else:
        first_mask = np.zeros_like(gray_prev, dtype=np.uint8)

    dynamic_masks: List[np.ndarray] = [first_mask]
    persist_left = args.persist_frames if first_mask.max() > 0 else 0
    for i in range(1, len(frames)):
        curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        det = extractor.extract(frames[i])
        raw_mask = merge_instance_masks(det.masks, curr_gray.shape[:2])
        flow = compute_dense_flow(gray_prev, curr_gray)
        dyn = make_motion_dynamic_mask(
            gray_prev,
            curr_gray,
            det.masks,
            flow=flow,
            min_points=args.min_points,
            motion_thresh=args.motion_thresh,
            dense_motion_thresh=args.dense_motion_thresh,
        )
        if args.use_detection_fallback:
            if int((dyn > 0).sum()) < args.min_dynamic_area and int((raw_mask > 0).sum()) > 0:
                dyn = raw_mask.copy()

        prev_warp = warp_mask_with_flow(dynamic_masks[-1], flow)
        raw_area = int((raw_mask > 0).sum())
        if raw_area > 0:
            persist_left = args.persist_frames
        else:
            persist_left = max(0, persist_left - 1)

        if persist_left > 0 and int((prev_warp > 0).sum()) > 0:
            if raw_area > 0:
                # Keep only overlap when detection exists to avoid excessive bleeding.
                support = cv2.bitwise_and(prev_warp, raw_mask)
                dyn = cv2.bitwise_or(dyn, support)
            elif int((dyn > 0).sum()) == 0:
                # Short-term persistence for brief miss detections.
                dyn = prev_warp

        dyn = dilate_mask(dyn, ksize=args.dilate)
        dyn = refine_binary_mask(
            dyn,
            close_ksize=args.close_ksize,
            open_ksize=args.open_ksize,
            min_area=args.min_mask_area,
        )
        dynamic_masks.append(dyn)
        gray_prev = curr_gray

    inpaint_masks = expand_masks_for_inpaint(dynamic_masks, args.inpaint_dilate_extra)
    temporal = temporal_borrow_fill(frames, inpaint_masks, max_radius=args.temporal_radius)
    spatial = spatial_inpaint_fallback(temporal, inpaint_masks, method=args.inpaint_method)
    output_frames = merge_temporal_and_spatial(temporal, spatial, inpaint_masks)

    os.makedirs(args.output_dir, exist_ok=True)
    write_video(os.path.join(args.output_dir, "inpainted.mp4"), output_frames, fps)
    save_masks(os.path.join(args.output_dir, "dynamic_mask.mp4"), dynamic_masks, fps)
    save_masks(os.path.join(args.output_dir, "inpaint_mask.mp4"), inpaint_masks, fps)
    mask_dir = os.path.join(args.output_dir, "masks")
    inpaint_mask_dir = os.path.join(args.output_dir, "inpaint_masks")
    save_mask_frames(mask_dir, dynamic_masks)
    save_mask_frames(inpaint_mask_dir, inpaint_masks)

    meta = {
        "input": args.input,
        "n_frames": len(frames),
        "fps": fps,
        "extractor_mode": extractor._mode,
        "target_class_ids": target_ids,
        "mask_dir": mask_dir,
        "inpaint_mask_dir": inpaint_mask_dir,
        "nonzero_mask_frames": int(sum(1 for m in dynamic_masks if m.max() > 0)),
        "avg_mask_area": float(np.mean([int((m > 0).sum()) for m in dynamic_masks])),
        "avg_inpaint_mask_area": float(np.mean([int((m > 0).sum()) for m in inpaint_masks])),
        "params": vars(args),
    }
    with open(os.path.join(args.output_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(json.dumps(meta, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Video Object Removal & Inpainting Baseline")
    p.add_argument("--input", type=str, required=True, help="Input video path")
    p.add_argument("--output-dir", type=str, default="outputs/run1", help="Output directory")
    p.add_argument("--max-frames", type=int, default=None, help="Optional frame limit for debug")
    p.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    p.add_argument("--device", type=str, default="cpu", help="YOLO device: cpu/cuda:0")
    p.add_argument(
        "--target-classes",
        type=str,
        default="person,bicycle,car,motorcycle,bus,truck",
        help="Comma-separated COCO ids or aliases, e.g. 'person,bicycle,sports ball,tennis racket'",
    )
    p.add_argument("--motion-thresh", type=float, default=1.0, help="Median sparse-flow magnitude threshold")
    p.add_argument("--dense-motion-thresh", type=float, default=0.8, help="Dense-flow fallback threshold")
    p.add_argument("--min-points", type=int, default=6, help="Min LK good points per instance")
    p.add_argument(
        "--use-detection-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fallback to raw detection masks",
    )
    p.add_argument("--min-dynamic-area", type=int, default=120, help="If dynamic area below this, use detection fallback")
    p.add_argument("--persist-frames", type=int, default=6, help="Mask persistence window for short miss detections")
    p.add_argument("--dilate", type=int, default=1, help="Evaluation mask dilation kernel size")
    p.add_argument("--close-ksize", type=int, default=3, help="Evaluation mask close kernel size")
    p.add_argument("--open-ksize", type=int, default=3, help="Mask open kernel size")
    p.add_argument("--min-mask-area", type=int, default=120, help="Minimum connected component area")
    p.add_argument("--temporal-radius", type=int, default=30, help="Temporal search radius")
    p.add_argument("--inpaint-dilate-extra", type=int, default=9, help="Extra dilation only for inpainting masks")
    p.add_argument("--inpaint-method", type=str, default="telea", choices=["telea", "ns"])
    return p


if __name__ == "__main__":
    parser = build_parser()
    run(parser.parse_args())


