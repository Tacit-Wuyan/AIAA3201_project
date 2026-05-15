import argparse
import os
from pathlib import Path


def check(path: str, label: str) -> bool:
    ok = os.path.exists(path)
    print(f"[{'OK' if ok else 'MISS'}] {label}: {path}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser("Check DiffuEraser setup")
    ap.add_argument("--repo-root", type=str, default="")
    args = ap.parse_args()

    repo_root = args.repo_root or str(Path(__file__).resolve().parents[1])
    diffueraser_root = os.path.join(repo_root, "third_party", "DiffuEraser")
    weights_root = os.path.join(diffueraser_root, "weights")

    checks = [
        (os.path.join(diffueraser_root, "run_diffueraser.py"), "DiffuEraser runner"),
        (os.path.join(diffueraser_root, "requirements.txt"), "DiffuEraser requirements"),
        (os.path.join(weights_root, "diffuEraser"), "DiffuEraser model directory"),
        (os.path.join(weights_root, "stable-diffusion-v1-5"), "SD1.5 base model directory"),
        (os.path.join(weights_root, "sd-vae-ft-mse"), "VAE directory"),
        (os.path.join(weights_root, "propainter"), "ProPainter weights directory"),
    ]

    all_ok = True
    for path, label in checks:
        all_ok &= check(path, label)

    if not all_ok:
        raise SystemExit(1)
    print("DiffuEraser setup looks complete.")


if __name__ == "__main__":
    main()
