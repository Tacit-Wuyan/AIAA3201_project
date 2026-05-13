import argparse
import contextlib
import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch


# Common dynamic COCO classes and aliases used in this project.
CLASS_NAME_TO_ID = {
    "person": 0,
    "bicycle": 1,
    "car": 2,
    "motorcycle": 3,
    "motorbike": 3,
    "airplane": 4,
    "bus": 5,
    "train": 6,
    "truck": 7,
    "boat": 8,
    "bird": 14,
    "cat": 15,
    "dog": 16,
    "horse": 17,
    "sheep": 18,
    "cow": 19,
    "elephant": 20,
    "bear": 21,
    "zebra": 22,
    "giraffe": 23,
    "sports ball": 32,
    "sports_ball": 32,
    "skateboard": 36,
    "surfboard": 37,
    "tennis racket": 38,
    "tennis_racket": 38,
}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_mask(path: str, mask: np.ndarray) -> None:
    # Robust writing on Windows non-ASCII paths.
    ok, buf = cv2.imencode(".png", mask)
    if not ok:
        raise RuntimeError(f"Failed to encode mask: {path}")
    buf.tofile(path)


def get_video_info(video_path: str) -> Tuple[int, Tuple[int, int]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Cannot read first frame: {video_path}")
    h, w = frame.shape[:2]
    return n, (h, w)


def parse_target_classes(spec: str) -> List[int]:
    ids: List[int] = []
    for raw in spec.split(","):
        token = raw.strip().lower()
        if not token:
            continue
        token = token.replace("_", " ")
        if token.isdigit():
            cid = int(token)
        else:
            if token not in CLASS_NAME_TO_ID:
                known = ", ".join(sorted(set(CLASS_NAME_TO_ID.keys())))
                raise ValueError(f"Unknown class token '{raw}'. Known names: {known}")
            cid = CLASS_NAME_TO_ID[token]
        if cid < 0 or cid > 79:
            raise ValueError(f"COCO class id out of range [0,79]: {cid}")
        ids.append(cid)
    uniq = sorted(set(ids))
    if not uniq:
        raise ValueError("target-classes resolved to empty list")
    return uniq


def postprocess_mask(
    mask: np.ndarray,
    dilate_ksize: int,
    close_ksize: int,
    open_ksize: int,
    min_area: int,
) -> np.ndarray:
    out = (mask > 0).astype(np.uint8) * 255

    if dilate_ksize and dilate_ksize > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_ksize, dilate_ksize))
        out = cv2.dilate(out, k, iterations=1)

    if close_ksize and close_ksize > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k, iterations=1)

    if open_ksize and open_ksize > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k, iterations=1)

    if min_area and min_area > 0:
        num, labels, stats, _ = cv2.connectedComponentsWithStats((out > 0).astype(np.uint8), connectivity=8)
        cleaned = np.zeros_like(out)
        for cid in range(1, num):
            area = int(stats[cid, cv2.CC_STAT_AREA])
            if area >= min_area:
                cleaned[labels == cid] = 255
        out = cleaned

    return out


def nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> np.ndarray:
    if boxes.shape[0] == 0:
        return np.zeros((0,), dtype=np.int32)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: List[int] = []

    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]

        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])

        inter_w = np.maximum(0.0, xx2 - xx1)
        inter_h = np.maximum(0.0, yy2 - yy1)
        inter = inter_w * inter_h
        union = areas[i] + areas[rest] - inter + 1e-6
        iou = inter / union

        order = rest[iou <= iou_thresh]

    return np.asarray(keep, dtype=np.int32)


def detect_boxes_on_frame(
    frame_bgr: np.ndarray,
    yolo,
    conf: float,
    max_objs: int,
    device: str,
    target_ids: List[int],
    min_box_area_ratio: float,
    max_box_area_ratio: float,
    nms_iou: float,
) -> np.ndarray:
    results = yolo.predict(
        frame_bgr,
        conf=conf,
        classes=target_ids,
        device=device,
        verbose=False,
    )
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32)

    boxes_xyxy = results[0].boxes.xyxy.detach().cpu().numpy().astype(np.float32)
    scores = results[0].boxes.conf.detach().cpu().numpy().astype(np.float32)

    h, w = frame_bgr.shape[:2]
    frame_area = float(h * w)
    bw = np.maximum(0.0, boxes_xyxy[:, 2] - boxes_xyxy[:, 0])
    bh = np.maximum(0.0, boxes_xyxy[:, 3] - boxes_xyxy[:, 1])
    area_ratio = (bw * bh) / max(frame_area, 1.0)
    keep = (area_ratio >= min_box_area_ratio) & (area_ratio <= max_box_area_ratio)

    boxes_xyxy = boxes_xyxy[keep]
    scores = scores[keep]
    if boxes_xyxy.shape[0] == 0:
        return np.zeros((0, 4), dtype=np.float32)

    keep_idx = nms_xyxy(boxes_xyxy, scores, iou_thresh=nms_iou)
    boxes_xyxy = boxes_xyxy[keep_idx]
    scores = scores[keep_idx]

    order = np.argsort(-scores)
    boxes_xyxy = boxes_xyxy[order]
    if max_objs > 0:
        boxes_xyxy = boxes_xyxy[:max_objs]
    return boxes_xyxy


def collect_prompt_boxes(
    input_video: str,
    yolo_model: str,
    conf: float,
    max_objs_per_frame: int,
    device: str,
    prompt_interval: int,
    max_prompt_frames: int,
    target_ids: List[int],
    min_box_area_ratio: float,
    max_box_area_ratio: float,
    nms_iou: float,
) -> List[Tuple[int, np.ndarray]]:
    from ultralytics import YOLO

    os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(os.getcwd(), "Ultralytics"))
    model = YOLO(yolo_model)

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for prompts: {input_video}")

    prompts: List[Tuple[int, np.ndarray]] = []
    frame_idx = 0
    sampled = 0
    interval = max(1, int(prompt_interval))

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        should_sample = frame_idx == 0 or (frame_idx % interval == 0)
        if should_sample:
            boxes = detect_boxes_on_frame(
                frame_bgr=frame,
                yolo=model,
                conf=conf,
                max_objs=max_objs_per_frame,
                device=device,
                target_ids=target_ids,
                min_box_area_ratio=min_box_area_ratio,
                max_box_area_ratio=max_box_area_ratio,
                nms_iou=nms_iou,
            )
            if boxes.shape[0] > 0:
                for box in boxes:
                    prompts.append((frame_idx, box.astype(np.float32)))
                sampled += 1
                if sampled >= max_prompt_frames:
                    break
        frame_idx += 1

    cap.release()
    return prompts


def generate_yolo_seg_masks(
    input_video: str,
    total: int,
    h: int,
    w: int,
    yolo_seg_model: str,
    conf: float,
    device: str,
    target_ids: List[int],
) -> Dict[int, np.ndarray]:
    from ultralytics import YOLO

    os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(os.getcwd(), "Ultralytics"))
    model = YOLO(yolo_seg_model)
    results = model.predict(
        source=input_video,
        conf=conf,
        classes=target_ids,
        device=device,
        stream=True,
        verbose=False,
    )
    out: Dict[int, np.ndarray] = {}
    allowed = set(target_ids)
    for idx, res in enumerate(results):
        if idx >= total:
            break
        frame_mask = np.zeros((h, w), dtype=np.uint8)
        if res.masks is not None and res.boxes is not None and len(res.boxes) > 0:
            cls_ids = res.boxes.cls.detach().cpu().numpy().astype(np.int32)
            masks = res.masks.data.detach().cpu().numpy()
            for j, cid in enumerate(cls_ids):
                if int(cid) not in allowed:
                    continue
                m = cv2.resize(masks[j], (w, h), interpolation=cv2.INTER_NEAREST)
                frame_mask[m > 0.5] = 255
        out[idx] = frame_mask
    return out


def main() -> None:
    ap = argparse.ArgumentParser("SAM2 auto mask generation")
    ap.add_argument("--input-video", type=str, required=True)
    ap.add_argument("--mask-dir", type=str, required=True)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--sam2-model", type=str, default="facebook/sam2-hiera-large")
    ap.add_argument("--yolo-model", type=str, default="yolov8n.pt")
    ap.add_argument("--yolo-seg-model", type=str, default="yolov8n-seg.pt")
    ap.add_argument(
        "--target-classes",
        type=str,
        default="person,bicycle,sports ball,tennis racket",
        help="Comma-separated COCO class names/ids to remove",
    )

    # Robust prompt strategy for reappearing objects.
    ap.add_argument("--prompt-interval", type=int, default=20, help="Sample detection prompts every N frames")
    ap.add_argument("--max-prompt-frames", type=int, default=20, help="Max sampled frames used as SAM2 prompts")
    ap.add_argument("--max-objs-per-prompt-frame", type=int, default=2)
    ap.add_argument("--min-box-area-ratio", type=float, default=0.0008)
    ap.add_argument("--max-box-area-ratio", type=float, default=0.65)
    ap.add_argument("--nms-iou", type=float, default=0.6)
    ap.add_argument("--mask-dilate", type=int, default=11)
    ap.add_argument("--mask-close", type=int, default=7)
    ap.add_argument("--mask-open", type=int, default=0)
    ap.add_argument("--min-mask-area", type=int, default=80)

    ap.add_argument("--yolo-fallback", action="store_true", help="Fallback/merge with YOLO-seg masks")
    ap.add_argument(
        "--fallback-min-area",
        type=int,
        default=200,
        help="If SAM2 mask area on a frame is below this, use YOLO fallback mask for that frame",
    )
    ap.add_argument(
        "--fallback-union",
        action="store_true",
        help="Union SAM2 and YOLO-seg masks on each frame (stronger recall, more aggressive masks)",
    )
    args = ap.parse_args()

    target_ids = parse_target_classes(args.target_classes)
    print(f"Target classes resolved to COCO ids: {target_ids}")

    ensure_dir(args.mask_dir)
    n_frames, (h, w) = get_video_info(args.input_video)

    prompt_pairs = collect_prompt_boxes(
        input_video=args.input_video,
        yolo_model=args.yolo_model,
        conf=args.conf,
        max_objs_per_frame=args.max_objs_per_prompt_frame,
        device=args.device,
        prompt_interval=args.prompt_interval,
        max_prompt_frames=args.max_prompt_frames,
        target_ids=target_ids,
        min_box_area_ratio=args.min_box_area_ratio,
        max_box_area_ratio=args.max_box_area_ratio,
        nms_iou=args.nms_iou,
    )

    if len(prompt_pairs) == 0:
        total = max(1, n_frames)
        if args.yolo_fallback:
            yolo_masks = generate_yolo_seg_masks(
                input_video=args.input_video,
                total=total,
                h=h,
                w=w,
                yolo_seg_model=args.yolo_seg_model,
                conf=args.conf,
                device=args.device,
                target_ids=target_ids,
            )
            for i in range(total):
                m = yolo_masks.get(i, np.zeros((h, w), dtype=np.uint8))
                m = postprocess_mask(
                    m,
                    dilate_ksize=args.mask_dilate,
                    close_ksize=args.mask_close,
                    open_ksize=args.mask_open,
                    min_area=args.min_mask_area,
                )
                write_mask(os.path.join(args.mask_dir, f"mask_{i:05d}.png"), m)
            print(f"No prompt boxes found, used YOLO fallback masks: {total} frames.")
            return

        for i in range(total):
            write_mask(os.path.join(args.mask_dir, f"mask_{i:05d}.png"), np.zeros((h, w), dtype=np.uint8))
        print("No prompt boxes detected, wrote empty masks.")
        return

    try:
        from sam2.sam2_video_predictor import SAM2VideoPredictor
    except Exception as e:
        raise RuntimeError(
            "Cannot import SAM2. Install it first, e.g., pip install -e <path_to_sam2_repo>"
        ) from e

    predictor = SAM2VideoPredictor.from_pretrained(args.sam2_model)
    state = predictor.init_state(args.input_video)

    amp_ctx = contextlib.nullcontext()
    if args.device.lower().startswith("cuda"):
        amp_ctx = torch.autocast("cuda", dtype=torch.bfloat16)

    with torch.inference_mode(), amp_ctx:
        # Add prompts on multiple keyframes so reappearing objects can be recovered.
        for i, (frame_idx, box) in enumerate(prompt_pairs):
            obj_id = i + 1
            predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=int(frame_idx),
                obj_id=obj_id,
                box=box,
            )

        mask_map: Dict[int, np.ndarray] = {}
        max_idx = -1
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(state):
            combined = np.zeros((h, w), dtype=bool)
            for j in range(len(out_obj_ids)):
                m = out_mask_logits[j].detach().cpu().numpy()
                m = np.squeeze(m)
                combined = np.logical_or(combined, m > 0.0)
            mask_map[int(out_frame_idx)] = (combined.astype(np.uint8) * 255)
            max_idx = max(max_idx, int(out_frame_idx))

    total = max(n_frames, max_idx + 1)
    if total <= 0:
        total = 1

    replaced = 0
    unioned = 0
    if args.yolo_fallback:
        yolo_masks = generate_yolo_seg_masks(
            input_video=args.input_video,
            total=total,
            h=h,
            w=w,
            yolo_seg_model=args.yolo_seg_model,
            conf=args.conf,
            device=args.device,
            target_ids=target_ids,
        )
        for i in range(total):
            sm = mask_map.get(i, np.zeros((h, w), dtype=np.uint8))
            ym = yolo_masks.get(i, np.zeros((h, w), dtype=np.uint8))
            s_area = int((sm > 0).sum())
            y_area = int((ym > 0).sum())

            if args.fallback_union:
                if y_area > 0:
                    merged = np.where((sm > 0) | (ym > 0), 255, 0).astype(np.uint8)
                    mask_map[i] = merged
                    unioned += 1
            elif s_area < args.fallback_min_area and y_area > 0:
                mask_map[i] = ym
                replaced += 1

    for i in range(total):
        m = mask_map.get(i)
        if m is None:
            m = np.zeros((h, w), dtype=np.uint8)
        m = postprocess_mask(
            m,
            dilate_ksize=args.mask_dilate,
            close_ksize=args.mask_close,
            open_ksize=args.mask_open,
            min_area=args.min_mask_area,
        )
        write_mask(os.path.join(args.mask_dir, f"mask_{i:05d}.png"), m)

    if args.yolo_fallback:
        if args.fallback_union:
            print(
                f"SAM2 masks saved with YOLO union: {total} frames, unioned={unioned}, "
                f"prompts={len(prompt_pairs)} -> {Path(args.mask_dir).resolve()}"
            )
        else:
            print(
                f"SAM2 masks saved with YOLO fallback: {total} frames, replaced={replaced}, "
                f"prompts={len(prompt_pairs)} -> {Path(args.mask_dir).resolve()}"
            )
    else:
        print(f"SAM2 masks saved: {total} frames, prompts={len(prompt_pairs)} -> {Path(args.mask_dir).resolve()}")


if __name__ == "__main__":
    main()
