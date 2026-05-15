import argparse
import glob
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def imread_any(path: str, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, flags)
    except Exception:
        return cv2.imread(path, flags)


def imwrite_any(path: str, image: np.ndarray) -> bool:
    ext = Path(path).suffix or ".png"
    try:
        ok, buf = cv2.imencode(ext, image)
        if not ok:
            return False
        buf.tofile(path)
        return True
    except Exception:
        return bool(cv2.imwrite(path, image))


def sorted_image_paths(folder: str) -> List[str]:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    paths: List[str] = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(paths)


def read_video_meta(path: str) -> Tuple[float, int, int, int]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, width, height, frame_count


def save_mask_video(mask_dir: str, out_path: str, fps: float, width: int, height: int) -> int:
    mask_paths = sorted_image_paths(mask_dir)
    if not mask_paths:
        raise RuntimeError(f"No masks found in: {mask_dir}")
    ensure_dir(str(Path(out_path).parent))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    count = 0
    for p in mask_paths:
        m = imread_any(p, cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        if m.shape[:2] != (height, width):
            m = cv2.resize(m, (width, height), interpolation=cv2.INTER_NEAREST)
        vis = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
        writer.write(vis)
        count += 1
    writer.release()
    return count


def extract_video_to_dir(video_path: str, out_dir: str, prefix: str = "inpaint") -> int:
    ensure_dir(out_dir)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open generated video: {video_path}")
    count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        dst = os.path.join(out_dir, f"{prefix}_{count:05d}.png")
        if not imwrite_any(dst, frame):
            raise RuntimeError(f"Failed to write frame: {dst}")
        count += 1
    cap.release()
    return count


def require_path(path: str, label: str) -> str:
    if not os.path.exists(path):
        raise RuntimeError(f"Missing {label}: {path}")
    return path


def main() -> None:
    ap = argparse.ArgumentParser("DiffuEraser adapter")
    ap.add_argument("--input-video", type=str, required=True)
    ap.add_argument("--mask-dir", type=str, required=True)
    ap.add_argument("--inpaint-dir", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--inpaint-video", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--diffueraser-repo", type=str, default="")
    ap.add_argument("--base-model-path", type=str, default="")
    ap.add_argument("--vae-path", type=str, default="")
    ap.add_argument("--diffueraser-path", type=str, default="")
    ap.add_argument("--propainter-model-dir", type=str, default="")
    ap.add_argument("--max-img-size", type=int, default=960)
    ap.add_argument("--mask-dilation-iter", type=int, default=8)
    ap.add_argument("--ref-stride", type=int, default=10)
    ap.add_argument("--neighbor-length", type=int, default=10)
    ap.add_argument("--subvideo-length", type=int, default=50)
    ap.add_argument("--video-length", type=int, default=0, help="0 means infer from input duration")
    args = ap.parse_args()

    args.input_video = os.path.abspath(args.input_video)
    args.mask_dir = os.path.abspath(args.mask_dir)
    args.inpaint_dir = os.path.abspath(args.inpaint_dir)
    args.output_dir = os.path.abspath(args.output_dir)
    args.inpaint_video = os.path.abspath(args.inpaint_video)

    project_root = str(Path(__file__).resolve().parents[1])
    repo = args.diffueraser_repo or os.environ.get(
        "DIFFUERASER_REPO", os.path.join(project_root, "third_party", "DiffuEraser")
    )
    repo = os.path.abspath(repo)
    runner = require_path(os.path.join(repo, "run_diffueraser.py"), "DiffuEraser runner")

    weights_root = os.path.join(repo, "weights")
    base_model_path = args.base_model_path or os.environ.get(
        "DIFFUERASER_BASE_MODEL", os.path.join(weights_root, "stable-diffusion-v1-5")
    )
    vae_path = args.vae_path or os.environ.get(
        "DIFFUERASER_VAE_PATH", os.path.join(weights_root, "sd-vae-ft-mse")
    )
    diffueraser_path = args.diffueraser_path or os.environ.get(
        "DIFFUERASER_MODEL_PATH", os.path.join(weights_root, "diffuEraser")
    )
    propainter_model_dir = args.propainter_model_dir or os.environ.get(
        "DIFFUERASER_PROPAINTER_DIR", os.path.join(weights_root, "propainter")
    )

    require_path(base_model_path, "stable-diffusion-v1-5 base model")
    require_path(vae_path, "sd-vae-ft-mse")
    require_path(diffueraser_path, "DiffuEraser weights")
    require_path(propainter_model_dir, "ProPainter prior weights")

    fps, width, height, frame_count = read_video_meta(args.input_video)
    video_length = int(args.video_length)
    if video_length <= 0:
        video_length = max(1, int(math.ceil(frame_count / max(fps, 1e-6))))

    ensure_dir(args.output_dir)
    ensure_dir(args.inpaint_dir)
    ensure_dir(str(Path(args.inpaint_video).parent))

    workspace = os.path.join(args.output_dir, "diffueraser_workspace")
    ensure_dir(workspace)
    input_mask_video = os.path.join(workspace, "mask.mp4")
    saved_mask_frames = save_mask_video(args.mask_dir, input_mask_video, fps, width, height)
    if saved_mask_frames == 0:
        raise RuntimeError("DiffuEraser adapter failed because no mask frames were saved.")

    env = os.environ.copy()
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    if args.device.lower().startswith("cuda:"):
        env["CUDA_VISIBLE_DEVICES"] = args.device.split(":", 1)[1]
    elif args.device.lower() == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""

    cmd = [
        sys.executable,
        runner,
        "--input_video",
        args.input_video,
        "--input_mask",
        input_mask_video,
        "--video_length",
        str(video_length),
        "--mask_dilation_iter",
        str(args.mask_dilation_iter),
        "--max_img_size",
        str(args.max_img_size),
        "--save_path",
        workspace,
        "--ref_stride",
        str(args.ref_stride),
        "--neighbor_length",
        str(args.neighbor_length),
        "--subvideo_length",
        str(args.subvideo_length),
        "--base_model_path",
        base_model_path,
        "--vae_path",
        vae_path,
        "--diffueraser_path",
        diffueraser_path,
        "--propainter_model_dir",
        propainter_model_dir,
    ]
    proc = subprocess.run(cmd, cwd=repo, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"DiffuEraser inference failed with code {proc.returncode}")

    result_video = require_path(os.path.join(workspace, "diffueraser_result.mp4"), "DiffuEraser output video")
    shutil.copy2(result_video, args.inpaint_video)
    frame_count = extract_video_to_dir(result_video, args.inpaint_dir, prefix="inpaint")

    print(
        "DiffuEraser adapter done. "
        f"mask_video={input_mask_video}, output_video={result_video}, frames_extracted={frame_count}"
    )


if __name__ == "__main__":
    main()
