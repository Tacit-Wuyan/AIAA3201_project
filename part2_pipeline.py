import argparse
import glob
import importlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    import yaml
except Exception:
    yaml = None


DYNAMIC_COCO_IDS = {0, 1, 2, 3, 5, 7}


@dataclass
class FrameInfo:
    path: str
    image: np.ndarray


def imread_any(path: str, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    # cv2.imread may fail on non-ASCII Windows paths; decode manually from bytes.
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, flags)
    except Exception:
        return cv2.imread(path, flags)


def imwrite_any(path: str, image: np.ndarray) -> bool:
    # cv2.imwrite may fail on non-ASCII Windows paths; encode then write bytes.
    ext = Path(path).suffix or ".png"
    try:
        ok, buf = cv2.imencode(ext, image)
        if not ok:
            return False
        buf.tofile(path)
        return True
    except Exception:
        return bool(cv2.imwrite(path, image))


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def sorted_image_paths(folder: str) -> List[str]:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    paths: List[str] = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(paths)


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
    ensure_dir(str(Path(path).parent))
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()


def save_frames(frames: Sequence[np.ndarray], out_dir: str, prefix: str = "frame") -> List[str]:
    ensure_dir(out_dir)
    paths: List[str] = []
    for i, frame in enumerate(frames):
        p = os.path.join(out_dir, f"{prefix}_{i:05d}.png")
        if not imwrite_any(p, frame):
            raise RuntimeError(f"Failed to write frame: {p}")
        paths.append(p)
    return paths


def load_frames(frame_dir: str) -> List[FrameInfo]:
    frame_paths = sorted_image_paths(frame_dir)
    out: List[FrameInfo] = []
    for p in frame_paths:
        im = imread_any(p, cv2.IMREAD_COLOR)
        if im is None:
            raise RuntimeError(f"Failed to read frame: {p}")
        out.append(FrameInfo(path=p, image=im))
    return out


def save_mask_video(mask_dir: str, out_path: str, fps: float) -> None:
    mask_paths = sorted_image_paths(mask_dir)
    if not mask_paths:
        return
    sample = imread_any(mask_paths[0], cv2.IMREAD_GRAYSCALE)
    if sample is None:
        return
    h, w = sample.shape[:2]
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for p in mask_paths:
        m = imread_any(p, cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        vis = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
        writer.write(vis)
    writer.release()


def run_external_command(command_template: str, values: Dict[str, str]) -> None:
    cmd = command_template.format(**values)
    proc = subprocess.run(cmd, shell=True)
    if proc.returncode != 0:
        raise RuntimeError(f"External command failed with code {proc.returncode}: {cmd}")


def generate_masks_open_choice(
    input_video: str,
    mask_dir: str,
    conf: float = 0.3,
    device: str = "cpu",
    track_move_threshold: float = 2.0,
    dilate_ksize: int = 7,
    max_frames: Optional[int] = None,
) -> int:
    ensure_dir(mask_dir)
    try:
        module = importlib.import_module("ultralytics")
        YOLO = getattr(module, "YOLO")
    except Exception as e:
        raise RuntimeError("ultralytics is required for mask backend 'open_choice'.") from e

    os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(os.getcwd(), "Ultralytics"))
    model = YOLO("yolov8n-seg.pt")
    results = model.track(
        source=input_video,
        conf=conf,
        device=device,
        tracker="bytetrack.yaml",
        persist=True,
        classes=sorted(DYNAMIC_COCO_IDS),
        stream=True,
        verbose=False,
    )

    prev_centers: Dict[int, Tuple[float, float]] = {}
    frame_idx = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_ksize, dilate_ksize))

    for res in results:
        if max_frames is not None and frame_idx >= max_frames:
            break
        h, w = res.orig_shape
        frame_mask = np.zeros((h, w), dtype=np.uint8)
        if res.boxes is not None and res.masks is not None and len(res.boxes) > 0:
            boxes = res.boxes
            cls_ids = boxes.cls.detach().cpu().numpy().astype(np.int32)
            ids_arr = boxes.id.detach().cpu().numpy().astype(np.int32) if boxes.id is not None else None
            xywh = boxes.xywh.detach().cpu().numpy()
            masks = res.masks.data.detach().cpu().numpy()
            for i in range(len(cls_ids)):
                cid = int(cls_ids[i])
                if cid not in DYNAMIC_COCO_IDS:
                    continue
                track_id = int(ids_arr[i]) if ids_arr is not None else -1
                cx, cy = float(xywh[i][0]), float(xywh[i][1])
                moving = True
                if track_id >= 0 and track_id in prev_centers:
                    px, py = prev_centers[track_id]
                    moving = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5 >= track_move_threshold
                if track_id >= 0:
                    prev_centers[track_id] = (cx, cy)
                if not moving:
                    continue
                m = cv2.resize(masks[i], (w, h), interpolation=cv2.INTER_NEAREST)
                frame_mask[m > 0.5] = 255

        frame_mask = cv2.dilate(frame_mask, kernel, iterations=1)
        out_path = os.path.join(mask_dir, f"mask_{frame_idx:05d}.png")
        cv2.imwrite(out_path, frame_mask)
        frame_idx += 1

    if frame_idx == 0:
        raise RuntimeError("No frames were processed while generating masks.")
    return frame_idx


def inpaint_opencv(frame_dir: str, mask_dir: str, output_frame_dir: str, method: str = "telea") -> int:
    ensure_dir(output_frame_dir)
    frame_paths = sorted_image_paths(frame_dir)
    mask_paths = sorted_image_paths(mask_dir)
    if not frame_paths:
        raise RuntimeError("No input frames found for inpainting.")
    if len(frame_paths) != len(mask_paths):
        raise RuntimeError(
            f"Frame/mask count mismatch in opencv inpaint: frames={len(frame_paths)}, masks={len(mask_paths)}"
        )
    flag = cv2.INPAINT_TELEA if method.lower() == "telea" else cv2.INPAINT_NS
    for idx, (fp, mp) in enumerate(zip(frame_paths, mask_paths)):
        frame = imread_any(fp, cv2.IMREAD_COLOR)
        mask = imread_any(mp, cv2.IMREAD_GRAYSCALE)
        if frame is None or mask is None:
            raise RuntimeError(f"Failed to read pair: {fp}, {mp}")
        out = cv2.inpaint(frame, mask, 3, flag)
        dst = os.path.join(output_frame_dir, f"inpaint_{idx:05d}.png")
        if not imwrite_any(dst, out):
            raise RuntimeError(f"Failed to write inpaint frame: {dst}")
    return len(frame_paths)


def build_inpaint_masks(
    mask_dir: str,
    out_dir: str,
    dilate_ksize: int = 9,
    close_ksize: int = 3,
    min_area: int = 80,
) -> int:
    ensure_dir(out_dir)
    paths = sorted_image_paths(mask_dir)
    if not paths:
        raise RuntimeError(f"No masks found for inpaint-mask expansion: {mask_dir}")
    dk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_ksize, dilate_ksize)) if dilate_ksize > 1 else None
    ck = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize)) if close_ksize > 1 else None
    count = 0
    for idx, src in enumerate(paths):
        m = imread_any(src, cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise RuntimeError(f"Failed to read mask for expansion: {src}")
        out = (m > 127).astype(np.uint8) * 255
        if dk is not None:
            out = cv2.dilate(out, dk, iterations=1)
        if ck is not None:
            out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, ck, iterations=1)
        if min_area > 0:
            num, labels, stats, _ = cv2.connectedComponentsWithStats((out > 0).astype(np.uint8), connectivity=8)
            clean = np.zeros_like(out)
            for cid in range(1, num):
                if int(stats[cid, cv2.CC_STAT_AREA]) >= min_area:
                    clean[labels == cid] = 255
            out = clean
        dst = os.path.join(out_dir, f"mask_{idx:05d}.png")
        if not imwrite_any(dst, out):
            raise RuntimeError(f"Failed to write expanded inpaint mask: {dst}")
        count += 1
    return count


def compose_video_from_dir(frame_dir: str, out_video: str, fps: float) -> int:
    paths = sorted_image_paths(frame_dir)
    if not paths:
        raise RuntimeError(f"No frames found in: {frame_dir}")
    first = imread_any(paths[0], cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"Failed to read first frame: {paths[0]}")
    h, w = first.shape[:2]
    ensure_dir(str(Path(out_video).parent))
    writer = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    count = 0
    for p in paths:
        im = imread_any(p, cv2.IMREAD_COLOR)
        if im is None:
            continue
        writer.write(im)
        count += 1
    writer.release()
    return count


def extract_video_to_dir(video_path: str, out_dir: str) -> int:
    ensure_dir(out_dir)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for extraction: {video_path}")
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        dst = os.path.join(out_dir, f"inpaint_{idx:05d}.png")
        if not imwrite_any(dst, frame):
            raise RuntimeError(f"Failed to write extracted frame: {dst}")
        idx += 1
    cap.release()
    return idx


def parse_config(path: Optional[str]) -> Dict[str, object]:
    if not path:
        return {}
    if yaml is None:
        raise RuntimeError("PyYAML is required for --config. Install with: pip install pyyaml")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise RuntimeError("Config root must be a dictionary.")
    return data


def main() -> None:
    ap = argparse.ArgumentParser("Part 2/3 Pipeline: Dynamic Mask + Video Inpainting")
    ap.add_argument("--input", type=str, required=True, help="Input video path")
    ap.add_argument("--output-dir", type=str, default="outputs/part2_run1")
    ap.add_argument("--config", type=str, default=None, help="YAML config for external backends")
    ap.add_argument(
        "--mask-backend",
        type=str,
        default="open_choice",
        choices=["open_choice", "sam2", "track_anything", "vggt4d", "custom"],
    )
    ap.add_argument(
        "--inpaint-backend",
        type=str,
        default="opencv",
        choices=["opencv", "propainter", "e2fgvi", "fgvc", "custom"],
    )
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--track-move-threshold", type=float, default=2.0)
    ap.add_argument("--dilate", type=int, default=7)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--fallback-opencv", action="store_true")
    ap.add_argument("--inpaint-mask-dilate-extra", type=int, default=9, help="Extra dilation only for the masks sent to inpainting")
    ap.add_argument("--inpaint-mask-close", type=int, default=3, help="Closing kernel only for inpainting masks")
    ap.add_argument("--inpaint-mask-min-area", type=int, default=80, help="Min component area for inpainting masks")
    ap.add_argument(
        "--output-prefix",
        type=str,
        default=os.environ.get("PIPELINE_OUTPUT_PREFIX", "part2"),
        help="Prefix for final videos and metadata, e.g. part2 or part3",
    )
    args = ap.parse_args()
    output_prefix = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in args.output_prefix.strip())
    if not output_prefix:
        raise RuntimeError("--output-prefix cannot be empty")

    cfg = parse_config(args.config)
    project_root = str(Path(__file__).resolve().parent)

    input_video_abs = os.path.abspath(args.input)
    out_root = os.path.abspath(args.output_dir)
    frame_dir = os.path.join(out_root, "frames")
    mask_dir = os.path.join(out_root, "masks")
    inpaint_dir = os.path.join(out_root, "inpaint_frames")
    inpaint_mask_dir = os.path.join(out_root, "inpaint_masks")
    inpaint_video = os.path.join(out_root, "external_inpaint.mp4")
    ensure_dir(out_root)
    if os.path.exists(frame_dir):
        shutil.rmtree(frame_dir)
    if os.path.exists(mask_dir):
        shutil.rmtree(mask_dir)
    if os.path.exists(inpaint_dir):
        shutil.rmtree(inpaint_dir)
    if os.path.exists(inpaint_mask_dir):
        shutil.rmtree(inpaint_mask_dir)
    ensure_dir(frame_dir)
    ensure_dir(mask_dir)
    ensure_dir(inpaint_dir)

    frames, fps = read_video(input_video_abs, max_frames=args.max_frames)
    save_frames(frames, frame_dir, prefix="frame")

    # Keep external backends frame-aligned with --max-frames by feeding a trimmed video.
    working_input_video = input_video_abs
    if args.max_frames is not None:
        working_input_video = os.path.join(out_root, "input_trimmed.mp4")
        write_video(working_input_video, frames, fps)

    mask_count = 0
    base_values = {
        "input_video": working_input_video,
        "frame_dir": frame_dir,
        "mask_dir": mask_dir,
        "eval_mask_dir": mask_dir,
        "inpaint_mask_dir": inpaint_mask_dir,
        "inpaint_dir": inpaint_dir,
        "output_dir": out_root,
        "device": args.device,
        "project_root": project_root,
        "python_exe": sys.executable,
        "inpaint_video": inpaint_video,
    }
    if args.mask_backend == "open_choice":
        mask_count = generate_masks_open_choice(
            input_video=working_input_video,
            mask_dir=mask_dir,
            conf=args.conf,
            device=args.device,
            track_move_threshold=args.track_move_threshold,
            dilate_ksize=args.dilate,
            max_frames=args.max_frames,
        )
    else:
        key = f"{args.mask_backend}_mask_command"
        cmd = str(cfg.get(key, "")).strip()
        if not cmd:
            raise RuntimeError(f"Missing config command: {key}")
        run_external_command(cmd, base_values)
        mask_count = len(sorted_image_paths(mask_dir))
        if mask_count == 0:
            raise RuntimeError(f"Mask backend '{args.mask_backend}' produced no mask files in {mask_dir}")

    mask_for_inpaint_dir = mask_dir
    inpaint_mask_count = mask_count
    if args.inpaint_mask_dilate_extra > 1 or args.inpaint_mask_close > 1:
        inpaint_mask_count = build_inpaint_masks(
            mask_dir=mask_dir,
            out_dir=inpaint_mask_dir,
            dilate_ksize=args.inpaint_mask_dilate_extra,
            close_ksize=args.inpaint_mask_close,
            min_area=args.inpaint_mask_min_area,
        )
        mask_for_inpaint_dir = inpaint_mask_dir

    inpaint_values = dict(base_values)
    inpaint_values["mask_dir"] = mask_for_inpaint_dir
    inpaint_values["inpaint_mask_dir"] = inpaint_mask_dir

    inpaint_count = 0
    if args.inpaint_backend == "opencv":
        inpaint_count = inpaint_opencv(frame_dir, mask_for_inpaint_dir, inpaint_dir, method="telea")
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
                raise RuntimeError(
                    f"Inpaint backend '{args.inpaint_backend}' produced no frames in {inpaint_dir} "
                    f"and no video at {inpaint_video}"
                )
        except Exception:
            if not args.fallback_opencv:
                raise
            inpaint_count = inpaint_opencv(frame_dir, mask_for_inpaint_dir, inpaint_dir, method="telea")

    output_video = os.path.join(out_root, f"{output_prefix}_inpainted.mp4")
    mask_video = os.path.join(out_root, f"{output_prefix}_masks.mp4")
    inpaint_mask_video = os.path.join(out_root, f"{output_prefix}_inpaint_masks.mp4")
    compose_video_from_dir(inpaint_dir, output_video, fps)
    save_mask_video(mask_dir, mask_video, fps)
    if mask_for_inpaint_dir != mask_dir:
        save_mask_video(mask_for_inpaint_dir, inpaint_mask_video, fps)

    meta = {
        "input": input_video_abs,
        "working_input": working_input_video,
        "fps": fps,
        "n_input_frames": len(frames),
        "output_prefix": output_prefix,
        "mask_backend": args.mask_backend,
        "inpaint_backend": args.inpaint_backend,
        "mask_frames": mask_count,
        "inpaint_mask_frames": inpaint_mask_count,
        "inpaint_frames": inpaint_count,
        "mask_dir": mask_dir,
        "inpaint_mask_dir": mask_for_inpaint_dir,
        "output_video": output_video,
        "mask_video": mask_video,
        "inpaint_mask_video": inpaint_mask_video if mask_for_inpaint_dir != mask_dir else None,
        "config_file": args.config,
        "args": vars(args),
    }
    with open(os.path.join(out_root, f"{output_prefix}_run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

