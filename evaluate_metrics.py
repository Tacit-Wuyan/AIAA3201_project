import argparse
import glob
import os
from typing import List, Sequence, Tuple

import cv2
import numpy as np


def list_images(folder: str) -> List[str]:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(folder, e)))
    return sorted(paths)


def read_mask(path: str) -> np.ndarray:
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise RuntimeError(f"Failed to read mask: {path}")
    return (m > 127).astype(np.uint8)


def read_color(path: str) -> np.ndarray:
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return im


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = np.logical_and(pred > 0, gt > 0).sum()
    union = np.logical_or(pred > 0, gt > 0).sum()
    return float(inter / (union + 1e-8))


def mask_metrics(pred_dir: str, gt_dir: str, recall_thr: float = 0.5) -> Tuple[float, float]:
    pred_paths = list_images(pred_dir)
    gt_paths = list_images(gt_dir)
    if len(pred_paths) != len(gt_paths):
        raise RuntimeError(f"Mask count mismatch: pred={len(pred_paths)} gt={len(gt_paths)}")

    ious = []
    recalls = []
    for p, g in zip(pred_paths, gt_paths):
        pm = read_mask(p)
        gm = read_mask(g)
        cur_iou = iou(pm, gm)
        ious.append(cur_iou)
        recalls.append(1.0 if cur_iou >= recall_thr else 0.0)
    jm = float(np.mean(ious)) if ious else 0.0
    jr = float(np.mean(recalls)) if recalls else 0.0
    return jm, jr


def psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    return float(cv2.PSNR(img1, img2))


def ssim_rgb(img1: np.ndarray, img2: np.ndarray) -> float:
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    x = img1.astype(np.float64)
    y = img2.astype(np.float64)
    out = []
    for ch in range(3):
        a = x[:, :, ch]
        b = y[:, :, ch]
        mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
        mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)
        mu_a2 = mu_a * mu_a
        mu_b2 = mu_b * mu_b
        mu_ab = mu_a * mu_b
        sig_a2 = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a2
        sig_b2 = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b2
        sig_ab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_ab
        ssim_map = ((2 * mu_ab + c1) * (2 * sig_ab + c2)) / ((mu_a2 + mu_b2 + c1) * (sig_a2 + sig_b2 + c2))
        out.append(float(ssim_map.mean()))
    return float(np.mean(out))


def video_quality_metrics(pred_dir: str, gt_dir: str) -> Tuple[float, float]:
    pred_paths = list_images(pred_dir)
    gt_paths = list_images(gt_dir)
    if len(pred_paths) != len(gt_paths):
        raise RuntimeError(f"Image count mismatch: pred={len(pred_paths)} gt={len(gt_paths)}")
    psnrs: List[float] = []
    ssims: List[float] = []
    for p, g in zip(pred_paths, gt_paths):
        a = read_color(p)
        b = read_color(g)
        psnrs.append(psnr(a, b))
        ssims.append(ssim_rgb(a, b))
    return float(np.mean(psnrs)), float(np.mean(ssims))


def main() -> None:
    ap = argparse.ArgumentParser("Evaluate JM/JR and PSNR/SSIM")
    ap.add_argument("--pred-mask-dir", type=str, default=None)
    ap.add_argument("--gt-mask-dir", type=str, default=None)
    ap.add_argument("--pred-frame-dir", type=str, default=None)
    ap.add_argument("--gt-frame-dir", type=str, default=None)
    ap.add_argument("--recall-thr", type=float, default=0.5)
    args = ap.parse_args()

    if args.pred_mask_dir and args.gt_mask_dir:
        jm, jr = mask_metrics(args.pred_mask_dir, args.gt_mask_dir, recall_thr=args.recall_thr)
        print(f"JM (IoU mean): {jm:.4f}")
        print(f"JR (IoU recall@{args.recall_thr}): {jr:.4f}")

    if args.pred_frame_dir and args.gt_frame_dir:
        p, s = video_quality_metrics(args.pred_frame_dir, args.gt_frame_dir)
        print(f"PSNR: {p:.4f}")
        print(f"SSIM: {s:.4f}")


if __name__ == "__main__":
    main()

