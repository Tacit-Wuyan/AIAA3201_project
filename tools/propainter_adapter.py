import argparse
import glob
import os
import shutil
import shlex
import subprocess
import sys
from pathlib import Path


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def copy_frames(src_dir: str, dst_dir: str) -> int:
    ensure_dir(dst_dir)
    paths = sorted(glob.glob(os.path.join(src_dir, "*.png")))
    if not paths:
        paths = sorted(glob.glob(os.path.join(src_dir, "*.jpg")))
    count = 0
    for i, p in enumerate(paths):
        ext = Path(p).suffix.lower()
        if ext not in [".png", ".jpg", ".jpeg"]:
            continue
        q = os.path.join(dst_dir, f"inpaint_{i:05d}.png")
        shutil.copy2(p, q)
        count += 1
    return count


def pick_latest_mp4(folder: str) -> str:
    mp4s = glob.glob(os.path.join(folder, "*.mp4"))
    if not mp4s:
        return ""
    mp4s.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return mp4s[0]


def main() -> None:
    ap = argparse.ArgumentParser("ProPainter adapter")
    ap.add_argument("--input-video", type=str, required=True)
    ap.add_argument("--mask-dir", type=str, required=True)
    ap.add_argument("--inpaint-dir", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--inpaint-video", type=str, required=True)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--propainter-repo", type=str, default="")
    ap.add_argument(
        "--propainter-extra-args",
        type=str,
        default="",
        help="Extra args forwarded to inference_propainter.py, e.g. '--resize_ratio 0.5 --subvideo_length 40'",
    )
    args = ap.parse_args()

    # Normalize to absolute paths so subprocess cwd changes do not break path resolution.
    args.input_video = os.path.abspath(args.input_video)
    args.mask_dir = os.path.abspath(args.mask_dir)
    args.inpaint_dir = os.path.abspath(args.inpaint_dir)
    args.output_dir = os.path.abspath(args.output_dir)
    args.inpaint_video = os.path.abspath(args.inpaint_video)

    project_root = str(Path(__file__).resolve().parents[1])
    repo = args.propainter_repo or os.environ.get("PROPAINTER_REPO", os.path.join(project_root, "third_party", "ProPainter"))
    infer = os.path.join(repo, "inference_propainter.py")
    if not os.path.isfile(infer):
        raise RuntimeError(
            f"Cannot find ProPainter inference script at: {infer}\n"
            "Set --propainter-repo or PROPAINTER_REPO to your ProPainter repository path."
        )

    ensure_dir(args.output_dir)
    ensure_dir(args.inpaint_dir)
    ensure_dir(str(Path(args.inpaint_video).parent))

    cmd = [
        sys.executable,
        infer,
        "-i",
        args.input_video,
        "-m",
        args.mask_dir,
        "-o",
        args.output_dir,
        "--save_frames",
    ]
    if args.device.lower().startswith("cuda"):
        cmd.append("--fp16")
    extra = args.propainter_extra_args.strip() or os.environ.get("PROPAINTER_EXTRA_ARGS", "").strip()
    if extra:
        cmd.extend(shlex.split(extra, posix=False))
    proc = subprocess.run(cmd, cwd=repo)
    if proc.returncode != 0:
        raise RuntimeError(f"ProPainter inference failed with code {proc.returncode}")

    video_stem = Path(args.input_video).stem
    candidate_root = os.path.join(args.output_dir, video_stem)
    candidate_frames = os.path.join(candidate_root, "frames")
    candidate_video = os.path.join(candidate_root, "inpaint_out.mp4")

    copied = 0
    if os.path.isdir(candidate_frames):
        copied = copy_frames(candidate_frames, args.inpaint_dir)
    if copied == 0:
        # Some forks save frames directly under output root.
        copied = copy_frames(args.output_dir, args.inpaint_dir)

    chosen_video = ""
    if os.path.isfile(candidate_video):
        chosen_video = candidate_video
    else:
        chosen_video = pick_latest_mp4(candidate_root) or pick_latest_mp4(args.output_dir)
    if chosen_video:
        shutil.copy2(chosen_video, args.inpaint_video)

    print(
        f"ProPainter adapter done. frames_copied={copied}, video="
        f"{chosen_video if chosen_video else 'none'}"
    )


if __name__ == "__main__":
    main()
