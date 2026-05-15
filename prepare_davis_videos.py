"""
Script to convert DAVIS 480p image sequences to MP4 videos
for use with the Part 3 pipeline.
"""
import os
import cv2
import glob
from pathlib import Path


def images_to_video(image_dir: str, output_path: str, fps: float = 30.0) -> None:
    """Convert a directory of sequentially named images to an MP4 video."""
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(image_dir, ext)))
    paths = sorted(paths)
    
    if not paths:
        raise RuntimeError(f"No images found in {image_dir}")
    
    first = cv2.imread(paths[0])
    if first is None:
        raise RuntimeError(f"Failed to read first frame: {paths[0]}")
    
    h, w = first.shape[:2]
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = cv2.VideoWriter(
        output_path, 
        cv2.VideoWriter_fourcc(*"mp4v"), 
        fps, 
        (w, h)
    )
    
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        writer.write(img)
    
    writer.release()
    print(f"Created {output_path} with {len(paths)} frames ({w}x{h}, {fps:.2f} fps)")


def main():
    davis_root = "DAVIS"
    output_root = "DAVIS_videos"
    fps = 30.0  # DAVIS videos are typically 30fps
    
    sequences = ["bear", "boat", "bmx-trees", "tennis"]
    
    for seq in sequences:
        image_dir = os.path.join(davis_root, "JPEGImages", "480p", seq)
        if not os.path.isdir(image_dir):
            print(f"WARNING: {image_dir} not found, skipping")
            continue
        
        output_path = os.path.join(output_root, f"{seq}.mp4")
        images_to_video(image_dir, output_path, fps)


if __name__ == "__main__":
    main()
