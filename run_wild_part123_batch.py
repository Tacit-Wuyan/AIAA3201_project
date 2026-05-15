#!/usr/bin/env python3
"""Run the wild gymnastics video through Part 1 / Part 2 / Part 3 experiment suite.

No quantitative metrics are computed because the wild video has no GT masks or GT inpainted frames.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Run wild part1/part2/part3 suite")
    ap.add_argument("--source-root", type=Path, default=PROJECT_ROOT)
    ap.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "wild_part123_suite")
    ap.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "wild")
    ap.add_argument("--video-name", type=str, default="gymnastics_easy_long_12s_720p.mp4")
    ap.add_argument("--max-frames", type=int, default=100)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--part1-python", type=Path, default=Path(sys.executable))
    ap.add_argument("--part2-python", type=Path, default=Path(sys.executable))
    ap.add_argument("--part3-sam3-propainter-python", type=Path, default=Path(sys.executable))
    ap.add_argument("--part3-sam2-mask-python", type=Path, default=Path(sys.executable))
    ap.add_argument("--part3-diffueraser-python", type=Path, default=Path(sys.executable))
    ap.add_argument("--skip-existing", action="store_true")
    return ap.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: object) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def tail_text(path: Path, max_lines: int = 40) -> str:
    if not path.is_file():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def run_and_tee(cmd: list[str], cwd: Path, env: dict[str, str], log_path: Path) -> tuple[int, float]:
    ensure_dir(log_path.parent)
    start = time.time()
    env = dict(env)
    env.setdefault("PYTHONUNBUFFERED", "1")
    with log_path.open("w", encoding="utf-8") as log_f:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log_f.write(line)
            log_f.flush()
        proc.stdout.close()
        returncode = proc.wait()
    runtime_sec = time.time() - start
    return returncode, runtime_sec


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


def copy_wild_video(source_root: Path, data_root: Path, video_name: str) -> Path:
    src = source_root / "data" / "wild" / video_name
    if not src.is_file():
        raise RuntimeError(f"Wild video not found in source repo: {src}")
    dst = data_root / video_name
    ensure_dir(data_root)
    if not dst.is_file() or dst.stat().st_size != src.stat().st_size:
        shutil.copy2(src, dst)
    return dst


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
        raise RuntimeError(f"Failed to override target classes in {config_path}")
    ensure_dir(out_path.parent)
    out_path.write_text(replaced, encoding="utf-8")
    return out_path


def generate_part2_person_config(source_root: Path, out_path: Path, diffueraser_python: Path | None = None) -> Path:
    src = source_root / "configs" / "part2_example.yaml"
    text = src.read_text(encoding="utf-8-sig")
    sam2_tool = str((source_root / "tools" / "sam2_auto_mask.py").resolve()).replace("\\", "/")
    text = text.replace("{project_root}\\\\tools\\\\sam2_auto_mask.py", sam2_tool)
    replaced, count = re.subn(
        r'--target-classes\s+\\"[^"]*\\"',
        '--target-classes \\"person\\"',
        text,
        count=1,
    )
    if count == 0:
        raise RuntimeError(f"Failed to override target classes in {src}")
    if diffueraser_python is not None:
        diffueraser_cmd = (
            f'diffueraser_inpaint_command: "\\"{str(diffueraser_python).replace(chr(92), "/")}\\" \\"{{project_root}}/tools/diffueraser_adapter.py\\" '
            '--input-video \\"{input_video}\\" --mask-dir \\"{mask_dir}\\" --inpaint-dir \\"{inpaint_dir}\\" '
            '--output-dir \\"{output_dir}\\" --inpaint-video \\"{inpaint_video}\\" --device \\"{device}\\" '
            '--max-img-size 960 --mask-dilation-iter 8 --ref-stride 10 --neighbor-length 10 --subvideo-length 50"'
        )
        if "diffueraser_inpaint_command:" in replaced:
            replaced = re.sub(r"^diffueraser_inpaint_command:.*$", diffueraser_cmd, replaced, flags=re.MULTILINE)
        else:
            replaced = replaced.rstrip() + "\n\n" + diffueraser_cmd + "\n"
    ensure_dir(out_path.parent)
    out_path.write_text(replaced, encoding="utf-8")
    return out_path


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def summarize_result(name: str, output_dir: Path, log_path: Path, runtime_sec: float, status: str) -> dict[str, object]:
    meta_candidates = [
        output_dir / "run_meta.json",
        output_dir / "part2_run_meta.json",
        output_dir / "part3_run_meta.json",
        output_dir / "sam2_diffueraser_run_meta.json",
    ]
    meta_path = next((p for p in meta_candidates if p.is_file()), None)
    meta = load_json(meta_path) if meta_path is not None else {}
    output_video = None
    for key in ("output_video", "inpainted_video", "output"):
        if isinstance(meta, dict) and meta.get(key):
            output_video = meta.get(key)
            break
    if output_video is None:
        for candidate in [
            output_dir / "inpainted.mp4",
            output_dir / "part2_inpainted.mp4",
            output_dir / "part3_inpainted.mp4",
            output_dir / "sam2_diffueraser_inpainted.mp4",
        ]:
            if candidate.is_file():
                output_video = str(candidate)
                break
    return {
        "experiment": name,
        "status": status,
        "runtime_sec": round(runtime_sec, 3),
        "output_dir": str(output_dir),
        "output_video": output_video,
        "meta_path": str(meta_path) if meta_path else "",
        "log_path": str(log_path),
    }


def main() -> int:
    args = parse_args()

    source_root = args.source_root
    if not source_root.is_dir():
        raise RuntimeError(f"Source root not found: {source_root}")

    video_path = copy_wild_video(source_root, args.data_root, args.video_name)

    preflights = {
        "part1_main": (args.part1_python, ["cv2", "numpy", "torch", "ultralytics"]),
        "part2_main_propainter": (args.part2_python, ["cv2", "numpy", "torch", "sam2", "ultralytics"]),
        "part3_sam3_propainter": (args.part3_sam3_propainter_python, ["cv2", "numpy", "torch", "sam3", "ultralytics"]),
        "part3_sam2_diffueraser_mask": (args.part3_sam2_mask_python, ["cv2", "numpy", "torch", "sam2", "ultralytics"]),
        "part3_diffueraser": (args.part3_diffueraser_python, ["cv2", "numpy", "torch", "diffusers"]),
        "part3_sam3_diffueraser": (args.part3_diffueraser_python, ["cv2", "numpy", "torch", "sam3", "diffusers"]),
    }
    for label, (python_exe, modules) in preflights.items():
        ensure_python_imports(python_exe, modules, label)

    ensure_dir(args.output_root)
    generated_config_root = args.output_root / "generated_configs"
    logs_root = args.output_root / "logs"

    part2_person_cfg = generate_part2_person_config(
        source_root=source_root,
        out_path=generated_config_root / "main_part2_person.yaml",
    )
    part2_person_diffueraser_cfg = generate_part2_person_config(
        source_root=source_root,
        out_path=generated_config_root / "main_part2_person_diffueraser.yaml",
        diffueraser_python=args.part3_diffueraser_python,
    )
    sam3_propainter_cfg = override_target_classes(
        PROJECT_ROOT / "configs" / "part3_sam3_propainter_part2style.yaml",
        ["person"],
        generated_config_root / "part3_sam3_propainter_person.yaml",
    )
    sam3_diffueraser_cfg = override_target_classes(
        PROJECT_ROOT / "configs" / "part3_sam3_diffueraser_part2style.yaml",
        ["person"],
        generated_config_root / "part3_sam3_diffueraser_person.yaml",
    )

    experiments: list[tuple[str, list[str], dict[str, str]]] = [
        (
            "part1_main",
            [
                str(args.part1_python),
                str(source_root / "part1_pipeline.py"),
                "--input",
                str(video_path),
                "--output-dir",
                str(args.output_root / "part1_main"),
                "--device",
                args.device,
                "--target-classes",
                "person",
                "--max-frames",
                str(args.max_frames),
            ],
            {"PYTHONNOUSERSITE": "1", "YOLO_CONFIG_DIR": "/tmp/Ultralytics"},
        ),
        (
            "part2_main_propainter",
            [
                str(args.part2_python),
                str(source_root / "part2_pipeline.py"),
                "--input",
                str(video_path),
                "--output-dir",
                str(args.output_root / "part2_main_propainter"),
                "--output-prefix",
                "part2",
                "--mask-backend",
                "sam2",
                "--inpaint-backend",
                "propainter",
                "--config",
                str(part2_person_cfg),
                "--device",
                args.device,
                "--max-frames",
                str(args.max_frames),
            ],
            {"PYTHONNOUSERSITE": "1", "YOLO_CONFIG_DIR": "/tmp/Ultralytics"},
        ),
        (
            "part3_sam2_diffueraser",
            [
                str(args.part3_sam2_mask_python),
                str(PROJECT_ROOT / "part3_sam3_pipeline.py"),
                "--input",
                str(video_path),
                "--output-dir",
                str(args.output_root / "part3_sam2_diffueraser"),
                "--output-prefix",
                "sam2_diffueraser",
                "--mask-backend",
                "sam2",
                "--inpaint-backend",
                "diffueraser",
                "--config",
                str(part2_person_diffueraser_cfg),
                "--device",
                args.device,
                "--max-frames",
                str(args.max_frames),
            ],
            {"PYTHONNOUSERSITE": "1", "YOLO_CONFIG_DIR": "/tmp/Ultralytics"},
        ),
        (
            "part3_sam3_propainter",
            [
                str(args.part3_sam3_propainter_python),
                str(PROJECT_ROOT / "part3_sam3_pipeline.py"),
                "--input",
                str(video_path),
                "--output-dir",
                str(args.output_root / "part3_sam3_propainter"),
                "--mask-backend",
                "sam3",
                "--inpaint-backend",
                "propainter",
                "--config",
                str(sam3_propainter_cfg),
                "--device",
                args.device,
                "--max-frames",
                str(args.max_frames),
            ],
            {"PYTHONNOUSERSITE": "1", "YOLO_CONFIG_DIR": "/tmp/Ultralytics"},
        ),
        (
            "part3_sam3_diffueraser",
            [
                str(args.part3_diffueraser_python),
                str(PROJECT_ROOT / "part3_sam3_pipeline.py"),
                "--input",
                str(video_path),
                "--output-dir",
                str(args.output_root / "part3_sam3_diffueraser"),
                "--mask-backend",
                "sam3",
                "--inpaint-backend",
                "diffueraser",
                "--config",
                str(sam3_diffueraser_cfg),
                "--device",
                args.device,
                "--max-frames",
                str(args.max_frames),
            ],
            {"PYTHONNOUSERSITE": "1", "YOLO_CONFIG_DIR": "/tmp/Ultralytics"},
        ),
    ]

    results: list[dict[str, object]] = []
    manifest = {
        "source_root": str(source_root),
        "video_path": str(video_path),
        "output_root": str(args.output_root),
        "max_frames": args.max_frames,
        "device": args.device,
        "note": "Wild video has no GT; no JM/JR/PSNR/SSIM are computed.",
        "experiments": [name for name, _, _ in experiments],
    }
    write_json(args.output_root / "manifest.json", manifest)

    for name, cmd, env_updates in experiments:
        output_dir = args.output_root / name
        log_path = logs_root / f"{name}.log"
        ensure_dir(log_path.parent)

        if args.skip_existing and output_dir.is_dir() and any(output_dir.iterdir()):
            print(f"[reuse] {name}")
            results.append(summarize_result(name, output_dir, log_path, 0.0, "reused"))
            continue

        env = os.environ.copy()
        env.update(env_updates)
        if "propainter" in name:
            env.pop("PROPAINTER_REPO", None)
        print(f"[run] {name}")
        returncode, runtime_sec = run_and_tee(
            cmd,
            cwd=PROJECT_ROOT,
            env=env,
            log_path=log_path,
        )
        if returncode != 0:
            print(f"[fail] {name} -> {log_path}")
            results.append(
                {
                    "experiment": name,
                    "status": "failed",
                    "runtime_sec": round(runtime_sec, 3),
                    "output_dir": str(output_dir),
                    "output_video": "",
                    "meta_path": "",
                    "log_path": str(log_path),
                    "error": tail_text(log_path),
                }
            )
            continue
        print(f"[done] {name}")
        results.append(summarize_result(name, output_dir, log_path, runtime_sec, "ok"))

    summary_root = args.output_root / "summaries"
    ensure_dir(summary_root)
    write_json(summary_root / "wild_part123_results.json", results)
    write_csv(
        summary_root / "wild_part123_results.csv",
        results,
        ["experiment", "status", "runtime_sec", "output_dir", "output_video", "meta_path", "log_path", "error"],
    )
    print(f"Wild summaries written to {summary_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
