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

    base_cmd = [
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
    is_cuda = args.device.lower().startswith("cuda")
    if is_cuda:
        base_cmd.append("--fp16")
    # Force CPU path when requested; otherwise ProPainter may still pick CUDA if available.
    run_env = os.environ.copy()
    if args.device.lower().startswith("cpu"):
        run_env["CUDA_VISIBLE_DEVICES"] = ""
    if is_cuda:
        run_env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    user_extra = args.propainter_extra_args.strip() or os.environ.get("PROPAINTER_EXTRA_ARGS", "").strip()

    # For CUDA, progressively lower memory usage on OOM while keeping GPU inference.
    if user_extra:
        attempt_profiles = [("user-extra", user_extra)]
    elif is_cuda:
        attempt_profiles = [
            ("cuda-balanced", "--resize_ratio 0.67 --subvideo_length 48 --neighbor_length 8 --raft_iter 16"),
            ("cuda-safe", "--resize_ratio 0.5 --subvideo_length 32 --neighbor_length 6 --raft_iter 12"),
            ("cuda-safe-plus", "--resize_ratio 0.4 --subvideo_length 24 --neighbor_length 4 --raft_iter 8"),
        ]
    else:
        attempt_profiles = [("default", "")]

    last_returncode = None
    last_combined_output = ""
    for idx, (profile_name, profile_extra) in enumerate(attempt_profiles):
        cmd = list(base_cmd)
        if profile_extra:
            cmd.extend(shlex.split(profile_extra, posix=False))

        print(f"[ProPainter] Attempt {idx + 1}/{len(attempt_profiles)} ({profile_name})")
        proc = subprocess.run(cmd, cwd=repo, env=run_env, text=True, capture_output=True)
        if proc.stdout:
            print(proc.stdout, end="")
        if proc.stderr:
            print(proc.stderr, end="", file=sys.stderr)

        last_returncode = proc.returncode
        if proc.returncode == 0:
            break

        combined = f"{proc.stdout}\n{proc.stderr}".lower()
        last_combined_output = combined
        is_oom = ("cuda out of memory" in combined) or ("outofmemory" in combined)
        has_next = idx + 1 < len(attempt_profiles)
        if is_cuda and is_oom and has_next:
            print("[ProPainter] CUDA OOM detected, retrying with safer memory profile...")
            continue
        break

    if last_returncode != 0:
        tail = last_combined_output[-1200:] if last_combined_output else ""
        raise RuntimeError(f"ProPainter inference failed with code {last_returncode}.\n{tail}")

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
