#!/usr/bin/env python3
"""Run the requested DAVIS ablation suite:

    1. sam3 + propainter  (part2-style open-vocabulary prompt discovery + propagation)
    2. sam2 + diffueraser (main-branch Part2 mask generator + current DiffuEraser adapter)
    3. sam3 + diffueraser (part2-style open-vocabulary prompt discovery + propagation)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

import cv2


PROJECT_ROOT = Path(__file__).resolve().parent

SEQUENCE_TARGET_CLASSES: dict[str, list[str]] = {
    "bear": ["bear"],
    "blackswan": ["bird"],
    "bmx-bumps": ["person", "bicycle"],
    "bmx-trees": ["person", "bicycle"],
    "boat": ["boat"],
    "breakdance": ["person"],
    "breakdance-flare": ["person"],
    "bus": ["bus"],
    "camel": ["horse"],
    "car-roundabout": ["car"],
    "car-shadow": ["car"],
    "car-turn": ["car"],
    "cows": ["cow"],
    "dance-jump": ["person"],
    "dance-twirl": ["person"],
    "dog": ["dog"],
    "dog-agility": ["dog", "person"],
    "drift-chicane": ["car"],
    "drift-straight": ["car"],
    "drift-turn": ["car"],
    "elephant": ["elephant"],
    "flamingo": ["bird"],
    "goat": ["sheep"],
    "hike": ["person"],
    "hockey": ["person", "sports ball"],
    "horsejump-high": ["horse", "person"],
    "horsejump-low": ["horse", "person"],
    "kite-surf": ["person", "kite", "surfboard"],
    "kite-walk": ["person", "kite"],
    "libby": ["person"],
    "lucia": ["person"],
    "mallard-fly": ["bird"],
    "mallard-water": ["bird"],
    "motocross-bumps": ["person", "motorcycle"],
    "motocross-jump": ["person", "motorcycle"],
    "motorbike": ["person", "motorcycle"],
    "paragliding": ["person"],
    "paragliding-launch": ["person"],
    "parkour": ["person"],
    "rhino": ["elephant"],
    "rollerblade": ["person", "skateboard"],
    "scooter-black": ["person", "motorcycle"],
    "scooter-gray": ["person", "motorcycle"],
    "soapbox": ["car"],
    "soccerball": ["person", "sports ball"],
    "stroller": ["person"],
    "surf": ["person", "surfboard"],
    "swing": ["person"],
    "tennis": ["person", "sports ball", "tennis racket"],
    "train": ["train"],
}

MOTION_AWARE_SEQUENCES = {"tennis", "soccerball", "hockey"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Run DAVIS SAM ablation suite")
    ap.add_argument("--source-root", type=Path, default=PROJECT_ROOT)
    ap.add_argument("--davis-root", type=Path, default=PROJECT_ROOT / "DAVIS")
    ap.add_argument("--resolution", type=str, default="480p")
    ap.add_argument(
        "--videos-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "davis_sam_ablation_suite" / "videos_480p",
    )
    ap.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "davis_sam_ablation_suite",
    )
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument(
        "--experiments",
        type=str,
        default="sam3_propainter,sam2_diffueraser,sam3_diffueraser",
        help="Comma-separated experiment ids",
    )
    ap.add_argument("--sam3-propainter-python", type=Path, default=Path(sys.executable))
    ap.add_argument("--sam2-diffueraser-python", type=Path, default=Path(sys.executable))
    ap.add_argument("--sam2-diffueraser-inpaint-python", type=Path, default=Path(sys.executable))
    ap.add_argument("--sam3-diffueraser-python", type=Path, default=Path(sys.executable))
    ap.add_argument(
        "--sam3-propainter-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "part3_sam3_propainter_part2style.yaml",
    )
    ap.add_argument(
        "--sam3-diffueraser-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "part3_sam3_diffueraser_part2style.yaml",
    )
    ap.add_argument(
        "--sam3-propainter-motion-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "part3_sam3_propainter_part2style_motion.yaml",
    )
    ap.add_argument(
        "--sam3-diffueraser-motion-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "part3_sam3_diffueraser_part2style_motion.yaml",
    )
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--sequences", type=str, default="")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--fallback-opencv", action="store_true")
    ap.add_argument("--recall-thr", type=float, default=0.5)
    ap.add_argument(
        "--with-frame-metrics",
        action="store_true",
        help="Pass GT frame directories to compute PSNR/SSIM. Disabled by default because DAVIS has no inpainted-frame GT.",
    )
    return ap.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: object) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: Iterable[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def list_images(folder: Path) -> list[Path]:
    paths: list[Path] = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
        paths.extend(sorted(folder.glob(ext)))
    return sorted(paths)


def list_sequences(image_root: Path, requested: str, limit: int | None) -> list[str]:
    if requested.strip():
        seqs = [x.strip() for x in requested.split(",") if x.strip()]
    else:
        seqs = sorted([p.name for p in image_root.iterdir() if p.is_dir()])
    if limit is not None:
        seqs = seqs[:limit]
    return seqs


def images_to_video(image_dir: Path, output_path: Path, fps: float) -> int:
    frames = list_images(image_dir)
    if not frames:
        raise RuntimeError(f"No frames found in {image_dir}")
    first = cv2.imread(str(frames[0]))
    if first is None:
        raise RuntimeError(f"Failed to read first frame: {frames[0]}")
    h, w = first.shape[:2]
    ensure_dir(output_path.parent)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {output_path}")
    written = 0
    for frame_path in frames:
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue
        writer.write(frame)
        written += 1
    writer.release()
    return written


def tail_text(path: Path, max_lines: int = 40) -> str:
    if not path.is_file():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def ensure_python_imports(python_exe: Path, modules: list[str], label: str) -> None:
    code = (
        "import importlib\n"
        f"mods = {modules!r}\n"
        "missing = []\n"
        "for m in mods:\n"
        "    try:\n"
        "        importlib.import_module(m)\n"
        "    except Exception:\n"
        "        missing.append(m)\n"
        "print('\\n'.join(missing))\n"
    )
    proc = subprocess.run(
        [str(python_exe), "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Failed preflight import check for {label} with {python_exe}:\n{proc.stderr}")
    missing = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if missing:
        raise RuntimeError(f"{label} environment missing required modules: {', '.join(missing)}")


def sequence_target_classes(sequence: str) -> list[str]:
    return SEQUENCE_TARGET_CLASSES.get(sequence, ["person"])


def select_main_part2_config(source_root: Path, sequence: str) -> Path:
    name = "part2_motion_aware.yaml" if sequence in MOTION_AWARE_SEQUENCES else "part2_example.yaml"
    return source_root / "configs" / name


def generate_sam2_diffueraser_config(
    source_root: Path,
    source_config: Path,
    target_classes: list[str],
    out_path: Path,
    diffueraser_python: Path,
) -> Path:
    text = source_config.read_text(encoding="utf-8-sig")
    sam2_tool = str((source_root / "tools" / "sam2_auto_mask.py").resolve()).replace("\\", "/")
    text = text.replace("{project_root}\\\\tools\\\\sam2_auto_mask.py", sam2_tool)
    target_text = ",".join(target_classes)
    replaced, count = re.subn(
        r'--target-classes\s+\\"[^"]*\\"',
        f'--target-classes \\"{target_text}\\"',
        text,
        count=1,
    )
    if count == 0:
        raise RuntimeError(f"Failed to override target classes in {source_config}")
    diffueraser_cmd = (
        f'diffueraser_inpaint_command: "\\"{str(diffueraser_python).replace(chr(92), "/")}\\" \\"{{project_root}}/tools/diffueraser_adapter.py\\" '
        '--input-video \\"{input_video}\\" --mask-dir \\"{mask_dir}\\" --inpaint-dir \\"{inpaint_dir}\\" '
        '--output-dir \\"{output_dir}\\" --inpaint-video \\"{inpaint_video}\\" --device \\"{device}\\" '
        '--max-img-size 960 --mask-dilation-iter 8 --ref-stride 10 --neighbor-length 10 --subvideo-length 50"'
    )
    if "diffueraser_inpaint_command:" in replaced:
        replaced = re.sub(
            r'^diffueraser_inpaint_command:.*$',
            diffueraser_cmd,
            replaced,
            flags=re.MULTILINE,
        )
    else:
        replaced = replaced.rstrip() + "\n\n" + diffueraser_cmd + "\n"
    ensure_dir(out_path.parent)
    out_path.write_text(replaced, encoding="utf-8")
    return out_path


def override_target_classes(config_path: Path, target_classes: list[str], out_path: Path) -> Path:
    text = config_path.read_text(encoding="utf-8")
    target_text = ",".join(target_classes)
    replaced, count = re.subn(
        r'--target-classes\s+"[^"]*"',
        f'--target-classes "{target_text}"',
        text,
        count=1,
    )
    if count == 0:
        replaced, count = re.subn(
            r'--target-classes\s+\\"[^"]*\\"',
            f'--target-classes \\"{target_text}\\"',
            text,
            count=1,
        )
    if count == 0:
        raise RuntimeError(f"Failed to override --target-classes in {config_path}")
    ensure_dir(out_path.parent)
    out_path.write_text(replaced, encoding="utf-8")
    return out_path


def build_sam3_command(
    python_exe: Path,
    input_video: Path,
    output_dir: Path,
    config_path: Path,
    gt_mask_dir: Path,
    gt_frame_dir: Path | None,
    device: str,
    inpaint_backend: str,
    recall_thr: float,
    fallback_opencv: bool,
) -> list[str]:
    cmd = [
        str(python_exe),
        str(PROJECT_ROOT / "part3_sam3_pipeline.py"),
        "--input",
        str(input_video),
        "--output-dir",
        str(output_dir),
        "--output-prefix",
        "part3",
        "--mask-backend",
        "sam3",
        "--inpaint-backend",
        inpaint_backend,
        "--config",
        str(config_path),
        "--device",
        device,
        "--gt-mask-dir",
        str(gt_mask_dir),
        "--recall-thr",
        str(recall_thr),
    ]
    if gt_frame_dir is not None:
        cmd.extend(["--gt-frame-dir", str(gt_frame_dir)])
    if fallback_opencv:
        cmd.append("--fallback-opencv")
    return cmd


def select_sam3_config(base_config: Path, motion_config: Path, sequence: str) -> Path:
    return motion_config if sequence in MOTION_AWARE_SEQUENCES else base_config


def build_sam2_diffueraser_command(
    python_exe: Path,
    input_video: Path,
    output_dir: Path,
    config_path: Path,
    gt_mask_dir: Path,
    gt_frame_dir: Path | None,
    device: str,
    recall_thr: float,
    fallback_opencv: bool,
) -> list[str]:
    cmd = [
        str(python_exe),
        str(PROJECT_ROOT / "part3_sam3_pipeline.py"),
        "--input",
        str(input_video),
        "--output-dir",
        str(output_dir),
        "--output-prefix",
        "sam2_diffueraser",
        "--mask-backend",
        "sam2",
        "--inpaint-backend",
        "diffueraser",
        "--config",
        str(config_path),
        "--device",
        device,
        "--gt-mask-dir",
        str(gt_mask_dir),
        "--recall-thr",
        str(recall_thr),
    ]
    if gt_frame_dir is not None:
        cmd.extend(["--gt-frame-dir", str(gt_frame_dir)])
    if fallback_opencv:
        cmd.append("--fallback-opencv")
    return cmd


def summarize_result(
    experiment: str,
    sequence: str,
    output_dir: Path,
    log_path: Path,
    runtime_sec: float,
    status: str,
    meta_name: str,
    error: str = "",
    reused: bool = False,
) -> dict[str, object]:
    metrics_path = output_dir / "metrics.json"
    meta_path = output_dir / meta_name
    metrics = load_json(metrics_path) if metrics_path.is_file() else {}
    meta = load_json(meta_path) if meta_path.is_file() else {}
    mask_metrics = metrics.get("mask", {}) if isinstance(metrics, dict) else {}
    frame_metrics = metrics.get("frame", {}) if isinstance(metrics, dict) else {}
    return {
        "experiment": experiment,
        "sequence": sequence,
        "status": status,
        "reused": reused,
        "runtime_sec": round(runtime_sec, 3),
        "jm": mask_metrics.get("jm"),
        "jr": mask_metrics.get("jr"),
        "psnr": frame_metrics.get("psnr"),
        "ssim": frame_metrics.get("ssim"),
        "mask_pairs_used": mask_metrics.get("pairs_used"),
        "frame_pairs_used": frame_metrics.get("pairs_used"),
        "output_dir": str(output_dir),
        "output_video": meta.get("output_video"),
        "log_path": str(log_path),
        "error": error,
    }


def main() -> int:
    args = parse_args()
    source_root = args.source_root
    if not source_root.is_dir():
        raise RuntimeError(f"Source root not found: {source_root}")

    image_root = args.davis_root / "JPEGImages" / args.resolution
    mask_root = args.davis_root / "Annotations" / args.resolution
    if not image_root.is_dir() or not mask_root.is_dir():
        raise RuntimeError(f"DAVIS resolution folders not found under {args.davis_root}")

    requested = [x.strip() for x in args.experiments.split(",") if x.strip()]
    valid = {"sam3_propainter", "sam2_diffueraser", "sam3_diffueraser"}
    unknown = sorted(set(requested) - valid)
    if unknown:
        raise RuntimeError(f"Unknown experiments: {unknown}")

    sequences = list_sequences(image_root, args.sequences, args.limit)
    if not sequences:
        raise RuntimeError("No sequences selected")

    preflights = {
        "sam3_propainter": [
            (args.sam3_propainter_python, ["cv2", "numpy", "torch", "ultralytics", "sam3"]),
        ],
        "sam2_diffueraser": [
            (args.sam2_diffueraser_python, ["cv2", "numpy", "torch", "sam2"]),
            (args.sam2_diffueraser_inpaint_python, ["cv2", "numpy", "torch", "diffusers"]),
        ],
        "sam3_diffueraser": [
            (args.sam3_diffueraser_python, ["cv2", "numpy", "torch", "sam3", "diffusers"]),
        ],
    }
    for exp in requested:
        for python_exe, modules in preflights[exp]:
            ensure_python_imports(Path(python_exe), modules, exp)

    ensure_dir(args.videos_dir)
    ensure_dir(args.output_root)
    manifest = {
        "source_root": str(source_root),
        "davis_root": str(args.davis_root),
        "resolution": args.resolution,
        "videos_dir": str(args.videos_dir),
        "output_root": str(args.output_root),
        "device": args.device,
        "experiments": requested,
        "sequences": sequences,
        "sam3_propainter_python": str(args.sam3_propainter_python),
        "sam2_diffueraser_python": str(args.sam2_diffueraser_python),
        "sam2_diffueraser_inpaint_python": str(args.sam2_diffueraser_inpaint_python),
        "sam3_diffueraser_python": str(args.sam3_diffueraser_python),
        "sequence_target_classes": {seq: sequence_target_classes(seq) for seq in sequences},
    }
    write_json(args.output_root / "manifest.json", manifest)

    results_by_exp: dict[str, list[dict[str, object]]] = {exp: [] for exp in requested}
    meta_names = {
        "sam3_propainter": "part3_run_meta.json",
        "sam2_diffueraser": "sam2_diffueraser_run_meta.json",
        "sam3_diffueraser": "part3_run_meta.json",
    }

    for sequence in sequences:
        image_dir = image_root / sequence
        gt_mask_dir = mask_root / sequence
        gt_frame_dir = image_dir if args.with_frame_metrics else None
        if not image_dir.is_dir() or not gt_mask_dir.is_dir():
            print(f"[skip] {sequence}: missing DAVIS image or mask directory")
            continue

        input_video = args.videos_dir / f"{sequence}.mp4"
        if not input_video.is_file():
            print(f"[video] materializing {sequence} -> {input_video}")
            frame_count = images_to_video(image_dir, input_video, args.fps)
            print(f"[video] {sequence}: wrote {frame_count} frames")

        target_classes = sequence_target_classes(sequence)
        for exp in requested:
            exp_dir = args.output_root / "experiments" / exp / sequence
            log_path = args.output_root / "logs" / exp / f"{sequence}.log"
            ensure_dir(log_path.parent)
            meta_name = meta_names[exp]

            if args.skip_existing and (exp_dir / "metrics.json").is_file() and (exp_dir / meta_name).is_file():
                print(f"[reuse] {exp}/{sequence}")
                results_by_exp[exp].append(
                    summarize_result(
                        experiment=exp,
                        sequence=sequence,
                        output_dir=exp_dir,
                        log_path=log_path,
                        runtime_sec=0.0,
                        status="reused",
                        meta_name=meta_name,
                        reused=True,
                    )
                )
                continue

            if exp == "sam3_propainter":
                sam3_cfg = override_target_classes(
                    select_sam3_config(
                        args.sam3_propainter_config,
                        args.sam3_propainter_motion_config,
                        sequence,
                    ),
                    target_classes,
                    args.output_root / "generated_configs" / exp / sequence / "config.yaml",
                )
                cmd = build_sam3_command(
                    python_exe=args.sam3_propainter_python,
                    input_video=input_video,
                    output_dir=exp_dir,
                    config_path=sam3_cfg,
                    gt_mask_dir=gt_mask_dir,
                    gt_frame_dir=gt_frame_dir,
                    device=args.device,
                    inpaint_backend="propainter",
                    recall_thr=args.recall_thr,
                    fallback_opencv=args.fallback_opencv,
                )
                env = os.environ.copy()
                env.setdefault("PYTHONNOUSERSITE", "1")
                env.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")
                env.pop("PROPAINTER_REPO", None)
            elif exp == "sam3_diffueraser":
                sam3_cfg = override_target_classes(
                    select_sam3_config(
                        args.sam3_diffueraser_config,
                        args.sam3_diffueraser_motion_config,
                        sequence,
                    ),
                    target_classes,
                    args.output_root / "generated_configs" / exp / sequence / "config.yaml",
                )
                cmd = build_sam3_command(
                    python_exe=args.sam3_diffueraser_python,
                    input_video=input_video,
                    output_dir=exp_dir,
                    config_path=sam3_cfg,
                    gt_mask_dir=gt_mask_dir,
                    gt_frame_dir=gt_frame_dir,
                    device=args.device,
                    inpaint_backend="diffueraser",
                    recall_thr=args.recall_thr,
                    fallback_opencv=args.fallback_opencv,
                )
                env = os.environ.copy()
                env.setdefault("PYTHONNOUSERSITE", "1")
                env.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")
            else:
                base_cfg = select_main_part2_config(source_root, sequence)
                cfg = generate_sam2_diffueraser_config(
                    source_root=source_root,
                    source_config=base_cfg,
                    target_classes=target_classes,
                    out_path=args.output_root / "generated_configs" / exp / sequence / base_cfg.name,
                    diffueraser_python=args.sam2_diffueraser_inpaint_python,
                )
                cmd = build_sam2_diffueraser_command(
                    python_exe=args.sam2_diffueraser_python,
                    input_video=input_video,
                    output_dir=exp_dir,
                    config_path=cfg,
                    gt_mask_dir=gt_mask_dir,
                    gt_frame_dir=gt_frame_dir,
                    device=args.device,
                    recall_thr=args.recall_thr,
                    fallback_opencv=args.fallback_opencv,
                )
                env = os.environ.copy()
                env.setdefault("PYTHONNOUSERSITE", "1")
                env.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")

            print(f"[run] {exp}/{sequence}")
            start = time.time()
            proc = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            runtime_sec = time.time() - start
            log_path.write_text(proc.stdout, encoding="utf-8")

            if proc.returncode != 0:
                print(f"[fail] {exp}/{sequence} -> {log_path}")
                results_by_exp[exp].append(
                    {
                        "experiment": exp,
                        "sequence": sequence,
                        "status": "failed",
                        "reused": False,
                        "runtime_sec": round(runtime_sec, 3),
                        "jm": None,
                        "jr": None,
                        "psnr": None,
                        "ssim": None,
                        "mask_pairs_used": None,
                        "frame_pairs_used": None,
                        "output_dir": str(exp_dir),
                        "output_video": None,
                        "log_path": str(log_path),
                        "error": tail_text(log_path),
                    }
                )
                continue

            summary = summarize_result(
                experiment=exp,
                sequence=sequence,
                output_dir=exp_dir,
                log_path=log_path,
                runtime_sec=runtime_sec,
                status="ok",
                meta_name=meta_name,
            )
            results_by_exp[exp].append(summary)
            print(f"[done] {exp}/{sequence} JM={summary['jm']} JR={summary['jr']}")

    summary_root = args.output_root / "summaries"
    ensure_dir(summary_root)
    for exp, rows in results_by_exp.items():
        rows_sorted = sorted(rows, key=lambda x: x["sequence"])
        write_json(summary_root / f"{exp}_metrics.json", rows_sorted)
        fieldnames = [
            "experiment",
            "sequence",
            "status",
            "reused",
            "runtime_sec",
            "jm",
            "jr",
            "psnr",
            "ssim",
            "mask_pairs_used",
            "frame_pairs_used",
            "output_dir",
            "output_video",
            "log_path",
            "error",
        ]
        write_csv(summary_root / f"{exp}_metrics.csv", rows_sorted, fieldnames)
        ok_rows = [row for row in rows_sorted if row["status"] in {"ok", "reused"} and row["jm"] is not None]
        aggregate = {
            "experiment": exp,
            "num_sequences": len(rows_sorted),
            "num_success": sum(1 for row in rows_sorted if row["status"] in {"ok", "reused"}),
            "num_failed": sum(1 for row in rows_sorted if row["status"] == "failed"),
            "mean_jm": round(sum(float(row["jm"]) for row in ok_rows) / len(ok_rows), 6) if ok_rows else None,
            "mean_jr": round(sum(float(row["jr"]) for row in ok_rows) / len(ok_rows), 6) if ok_rows else None,
        }
        write_json(summary_root / f"{exp}_aggregate.json", aggregate)

    print(f"Batch summaries written to {summary_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
