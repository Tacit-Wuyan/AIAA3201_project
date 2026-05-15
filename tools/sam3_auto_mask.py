import argparse
import contextlib
import importlib
import json
import os
import sys
import types
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import numpy as np
import torch


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
    "skateboard": 36,
    "surfboard": 37,
    "sports ball": 32,
    "sports_ball": 32,
    "kite": 33,
    "tennis racket": 38,
    "tennis_racket": 38,
}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_mask(path: str, mask: np.ndarray) -> None:
    ok, buf = cv2.imencode(".png", mask)
    if not ok:
        raise RuntimeError(f"Failed to encode mask: {path}")
    buf.tofile(path)


def list_images(folder: str) -> List[str]:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    paths: List[str] = []
    for ext in exts:
        paths.extend(str(p) for p in sorted(Path(folder).glob(ext)))
    return sorted(paths)


def parse_target_classes(spec: str) -> List[str]:
    names: List[str] = []
    for raw in spec.split(","):
        token = raw.strip().lower().replace("_", " ")
        if not token:
            continue
        if token not in CLASS_NAME_TO_ID:
            known = ", ".join(sorted(set(CLASS_NAME_TO_ID.keys())))
            raise ValueError(f"Unknown class token '{raw}'. Known names: {known}")
        if token not in names:
            names.append(token)
    if not names:
        raise ValueError("target-classes resolved to empty list")
    return names


def class_name_to_coco_id(name: str) -> int:
    return CLASS_NAME_TO_ID[name.lower().replace("_", " ")]


def get_video_info(video_path: str) -> Tuple[int, int, int]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Cannot read first frame: {video_path}")
    h, w = frame.shape[:2]
    return num_frames, h, w


def iter_sampled_frames(video_path: str, prompt_interval: int) -> Sequence[Tuple[int, np.ndarray]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for prompt discovery: {video_path}")
    samples: List[Tuple[int, np.ndarray]] = []
    frame_idx = 0
    interval = max(1, int(prompt_interval))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx == 0 or frame_idx % interval == 0:
            samples.append((frame_idx, frame))
        frame_idx += 1
    cap.release()
    return samples


def bbox_xyxy_to_xywh(box_xyxy: Sequence[float]) -> List[float]:
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def bbox_xywh_to_xyxy(box_xywh: Sequence[float]) -> List[float]:
    x, y, w, h = [float(v) for v in box_xywh]
    return [x, y, x + max(0.0, w), y + max(0.0, h)]


def clamp_box_xyxy(box_xyxy: Sequence[float], image_w: int, image_h: int) -> List[float]:
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    x1 = min(max(0.0, x1), float(image_w - 1))
    y1 = min(max(0.0, y1), float(image_h - 1))
    x2 = min(max(x1 + 1.0, x2), float(image_w))
    y2 = min(max(y1 + 1.0, y2), float(image_h))
    return [x1, y1, x2, y2]


def expand_box_xyxy(
    box_xyxy: Sequence[float],
    image_w: int,
    image_h: int,
    padding_ratio: float,
) -> List[float]:
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    pad_w = max(2.0, (x2 - x1) * float(padding_ratio))
    pad_h = max(2.0, (y2 - y1) * float(padding_ratio))
    return clamp_box_xyxy([x1 - pad_w, y1 - pad_h, x2 + pad_w, y2 + pad_h], image_w, image_h)


def bbox_iou_xyxy(box_a: Optional[Sequence[float]], box_b: Optional[Sequence[float]]) -> float:
    if box_a is None or box_b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def bbox_center_distance(box_a: Optional[Sequence[float]], box_b: Optional[Sequence[float]]) -> float:
    if box_a is None or box_b is None:
        return float("inf")
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]
    acx = 0.5 * (ax1 + ax2)
    acy = 0.5 * (ay1 + ay2)
    bcx = 0.5 * (bx1 + bx2)
    bcy = 0.5 * (by1 + by2)
    return float(((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5)


def mask_to_bbox_xyxy(mask: np.ndarray) -> Optional[List[float]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def read_frame_at(video_path: str, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for random frame access: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Cannot read frame {frame_idx} from video: {video_path}")
    return frame


def run_yolo_detections(
    model,
    frame: np.ndarray,
    target_classes: Sequence[str],
    conf: float,
    device: str,
) -> List[Dict[str, object]]:
    target_map = {class_name_to_coco_id(name): name for name in target_classes}
    results = model.predict(frame, conf=conf, device=device, verbose=False)
    detections: List[Dict[str, object]] = []
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return detections

    boxes = results[0].boxes
    cls_ids = boxes.cls.detach().cpu().numpy().astype(np.int32)
    confs = boxes.conf.detach().cpu().numpy().astype(np.float32)
    xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32)
    for idx in range(len(cls_ids)):
        class_id = int(cls_ids[idx])
        if class_id not in target_map:
            continue
        detections.append(
            {
                "class_name": target_map[class_id],
                "class_id": class_id,
                "conf": float(confs[idx]),
                "bbox_xyxy": [float(v) for v in xyxy[idx].tolist()],
                "bbox_xywh": bbox_xyxy_to_xywh(xyxy[idx].tolist()),
            }
        )
    return detections


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


def score_boxes_by_motion(
    prev_gray: Optional[np.ndarray],
    curr_gray: np.ndarray,
    boxes_xyxy: np.ndarray,
    padding_ratio: float = 0.05,
) -> np.ndarray:
    if boxes_xyxy.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    if prev_gray is None:
        return np.zeros((boxes_xyxy.shape[0],), dtype=np.float32)

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        curr_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=21,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    mag = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)
    h, w = curr_gray.shape[:2]
    scores: List[float] = []
    for box in boxes_xyxy:
        x1, y1, x2, y2 = [float(v) for v in box]
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        pad_x = bw * padding_ratio
        pad_y = bh * padding_ratio
        xx1 = max(0, int(round(x1 - pad_x)))
        yy1 = max(0, int(round(y1 - pad_y)))
        xx2 = min(w, int(round(x2 + pad_x)))
        yy2 = min(h, int(round(y2 + pad_y)))
        if xx2 <= xx1 or yy2 <= yy1:
            scores.append(0.0)
            continue
        local = mag[yy1:yy2, xx1:xx2]
        if local.size == 0:
            scores.append(0.0)
            continue
        scores.append(float(np.percentile(local, 75)))
    return np.asarray(scores, dtype=np.float32)


def generate_yolo_seg_masks(
    input_video: str,
    total: int,
    h: int,
    w: int,
    yolo_seg_model: str,
    conf: float,
    device: str,
    target_classes: Sequence[str],
) -> Dict[int, np.ndarray]:
    from ultralytics import YOLO

    os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(os.getcwd(), "Ultralytics"))
    model = YOLO(yolo_seg_model)
    target_ids = {class_name_to_coco_id(name) for name in target_classes}
    results = model.predict(
        source=input_video,
        conf=conf,
        classes=sorted(target_ids),
        device=device,
        stream=True,
        verbose=False,
    )
    out: Dict[int, np.ndarray] = {}
    for idx, res in enumerate(results):
        if idx >= total:
            break
        frame_mask = np.zeros((h, w), dtype=np.uint8)
        if res.masks is not None and res.boxes is not None and len(res.boxes) > 0:
            cls_ids = res.boxes.cls.detach().cpu().numpy().astype(np.int32)
            masks = res.masks.data.detach().cpu().numpy()
            for j, cid in enumerate(cls_ids):
                if int(cid) not in target_ids:
                    continue
                m = cv2.resize(masks[j], (w, h), interpolation=cv2.INTER_NEAREST)
                frame_mask[m > 0.5] = 255
        out[idx] = frame_mask
    return out


def generate_yolo_box_masks(
    input_video: str,
    total: int,
    h: int,
    w: int,
    yolo_model,
    conf: float,
    device: str,
    target_classes: Sequence[str],
    padding_ratio: float,
    min_box_area_ratio: float,
    max_box_area_ratio: float,
    nms_iou: float,
    max_objs_per_frame: int,
) -> Dict[int, np.ndarray]:
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for YOLO box fallback: {input_video}")
    out: Dict[int, np.ndarray] = {}
    frame_idx = 0
    while frame_idx < total:
        ok, frame = cap.read()
        if not ok:
            break
        frame_mask = np.zeros((h, w), dtype=np.uint8)
        detections = run_yolo_detections(
            model=yolo_model,
            frame=frame,
            target_classes=target_classes,
            conf=conf,
            device=device,
        )
        if detections:
            det_boxes = np.asarray([det["bbox_xyxy"] for det in detections], dtype=np.float32)
            det_scores = np.asarray([float(det["conf"]) for det in detections], dtype=np.float32)
            bw = np.maximum(0.0, det_boxes[:, 2] - det_boxes[:, 0])
            bh = np.maximum(0.0, det_boxes[:, 3] - det_boxes[:, 1])
            area_ratio = (bw * bh) / max(1.0, float(w * h))
            keep = (area_ratio >= float(min_box_area_ratio)) & (area_ratio <= float(max_box_area_ratio))
            if np.any(keep):
                det_boxes = det_boxes[keep]
                det_scores = det_scores[keep]
                keep_idx = nms_xyxy(det_boxes, det_scores, iou_thresh=nms_iou)
                det_boxes = det_boxes[keep_idx]
                order = np.argsort(-det_scores[keep_idx])
                det_boxes = det_boxes[order]
                if max_objs_per_frame > 0:
                    det_boxes = det_boxes[:max_objs_per_frame]
                for box in det_boxes:
                    x1, y1, x2, y2 = [int(round(v)) for v in expand_box_xyxy(box.tolist(), w, h, padding_ratio)]
                    frame_mask[y1:y2, x1:x2] = 255
        out[frame_idx] = frame_mask
        frame_idx += 1
    cap.release()
    return out


def aggregate_boxes_xyxy(boxes_xyxy: Sequence[Sequence[float]]) -> Optional[List[float]]:
    if not boxes_xyxy:
        return None
    arr = np.asarray(boxes_xyxy, dtype=np.float32)
    return [float(v) for v in np.median(arr, axis=0).tolist()]


def box_area_ratio_xyxy(box_xyxy: Optional[Sequence[float]], image_w: int, image_h: int) -> float:
    if box_xyxy is None:
        return 0.0
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return float(area / max(1.0, float(image_w * image_h)))


def box_slenderness_xyxy(box_xyxy: Optional[Sequence[float]]) -> float:
    if box_xyxy is None:
        return float("inf")
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    return float(max(w / h, h / w))


def build_temporal_consensus_detection(
    input_video: str,
    yolo_model,
    class_name: str,
    anchor_frame_idx: int,
    anchor_box_xyxy: Sequence[float],
    conf: float,
    device: str,
    temporal_window_radius: int,
    image_w: int,
    image_h: int,
) -> Dict[str, object]:
    num_frames, _, _ = get_video_info(input_video)
    if num_frames <= 0:
        raise RuntimeError(f"Video has no frames: {input_video}")
    anchor_frame_idx = max(0, min(int(anchor_frame_idx), num_frames - 1))
    radius = max(0, int(temporal_window_radius))
    frame_indices = sorted(
        {
            idx
            for idx in range(
                max(0, anchor_frame_idx - radius),
                min(num_frames, anchor_frame_idx + radius + 1),
            )
        }
    )
    matched_boxes = [[float(v) for v in anchor_box_xyxy]]
    matched_confs = [1.0]
    per_frame_matches: List[Dict[str, object]] = []
    for frame_idx in frame_indices:
        if frame_idx == int(anchor_frame_idx):
            continue
        frame = read_frame_at(input_video, frame_idx)
        detections = [
            det
            for det in run_yolo_detections(
                model=yolo_model,
                frame=frame,
                target_classes=[class_name],
                conf=conf,
                device=device,
            )
            if str(det["class_name"]) == class_name
        ]
        matched = match_detection_to_history(detections, anchor_box_xyxy)
        if matched is None:
            continue
        matched_box = [float(v) for v in matched["bbox_xyxy"]]
        matched_boxes.append(matched_box)
        matched_confs.append(float(matched["conf"]))
        per_frame_matches.append(
            {
                "frame_idx": int(frame_idx),
                "bbox_xyxy": matched_box,
                "conf": float(matched["conf"]),
                "iou_to_anchor": bbox_iou_xyxy(anchor_box_xyxy, matched_box),
            }
        )

    consensus_box = aggregate_boxes_xyxy(matched_boxes) or [float(v) for v in anchor_box_xyxy]
    mean_iou = float(np.mean([bbox_iou_xyxy(consensus_box, box) for box in matched_boxes])) if matched_boxes else 0.0
    match_ratio = float(len(matched_boxes) / max(1, len(frame_indices)))
    mean_conf = float(np.mean(matched_confs)) if matched_confs else 0.0
    return {
        "bbox_xyxy": clamp_box_xyxy(consensus_box, image_w, image_h),
        "bbox_xywh": bbox_xyxy_to_xywh(consensus_box),
        "temporal_match_count": int(len(matched_boxes)),
        "temporal_frame_count": int(len(frame_indices)),
        "temporal_match_ratio": match_ratio,
        "temporal_mean_iou": mean_iou,
        "temporal_mean_conf": mean_conf,
        "matched_frames": per_frame_matches,
        "area_ratio": box_area_ratio_xyxy(consensus_box, image_w, image_h),
        "slenderness": box_slenderness_xyxy(consensus_box),
    }


def discover_initial_prompts(
    input_video: str,
    yolo_model,
    target_classes: Sequence[str],
    conf: float,
    device: str,
    prompt_interval: int,
    max_prompt_frames_per_class: int,
    init_search_sampled_frames: int,
    image_w: int,
    image_h: int,
    temporal_window_radius: int,
    prompt_filter: str,
    motion_prompt_topk: int,
    motion_score_thresh: float,
    motion_box_padding: float,
    max_objs_per_frame: int,
    min_box_area_ratio: float,
    max_box_area_ratio: float,
    nms_iou: float,
) -> Dict[str, List[Dict[str, object]]]:
    found: Dict[str, List[Dict[str, object]]] = {name: [] for name in target_classes}
    sampled_frames = iter_sampled_frames(input_video, prompt_interval)
    sampled_frames = list(sampled_frames[: max(1, init_search_sampled_frames)])
    per_class_candidates: Dict[str, List[Dict[str, object]]] = {name: [] for name in target_classes}
    prev_gray: Optional[np.ndarray] = None

    for frame_idx, frame in sampled_frames:
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = run_yolo_detections(
            model=yolo_model,
            frame=frame,
            target_classes=target_classes,
            conf=conf,
            device=device,
        )
        filtered_detections: List[Dict[str, object]] = []
        if detections:
            det_boxes = np.asarray([det["bbox_xyxy"] for det in detections], dtype=np.float32)
            det_scores = np.asarray([float(det["conf"]) for det in detections], dtype=np.float32)
            bw = np.maximum(0.0, det_boxes[:, 2] - det_boxes[:, 0])
            bh = np.maximum(0.0, det_boxes[:, 3] - det_boxes[:, 1])
            area_ratio = (bw * bh) / max(1.0, float(image_w * image_h))
            keep = (area_ratio >= float(min_box_area_ratio)) & (area_ratio <= float(max_box_area_ratio))
            if np.any(keep):
                det_boxes = det_boxes[keep]
                det_scores = det_scores[keep]
                kept_dets = [det for det, flag in zip(detections, keep.tolist()) if flag]
                keep_idx = nms_xyxy(det_boxes, det_scores, iou_thresh=nms_iou)
                kept_dets = [kept_dets[int(i)] for i in keep_idx.tolist()]
                det_boxes = det_boxes[keep_idx]
                if prompt_filter == "motion" and det_boxes.shape[0] > 0:
                    motion_scores = score_boxes_by_motion(
                        prev_gray=prev_gray,
                        curr_gray=curr_gray,
                        boxes_xyxy=det_boxes,
                        padding_ratio=motion_box_padding,
                    )
                    order = np.argsort(-motion_scores)
                    keep_motion: List[int] = []
                    for idx in order.tolist():
                        if float(motion_scores[int(idx)]) < motion_score_thresh:
                            continue
                        keep_motion.append(int(idx))
                        if motion_prompt_topk > 0 and len(keep_motion) >= motion_prompt_topk:
                            break
                    kept_dets = [kept_dets[i] for i in keep_motion] if keep_motion else []
                kept_dets.sort(key=lambda x: float(x["conf"]), reverse=True)
                if max_objs_per_frame > 0:
                    kept_dets = kept_dets[: max_objs_per_frame]
                filtered_detections = kept_dets
        for det in filtered_detections:
            class_name = str(det["class_name"])
            per_class_candidates[class_name].append(
                {
                    "frame_idx": int(frame_idx),
                    "bbox_xyxy": det["bbox_xyxy"],
                    "bbox_xywh": det["bbox_xywh"],
                    "conf": float(det["conf"]),
                }
            )
        prev_gray = curr_gray

    for class_name, candidates in per_class_candidates.items():
        scored: List[Dict[str, object]] = []
        for cand in candidates:
            consensus = build_temporal_consensus_detection(
                input_video=input_video,
                yolo_model=yolo_model,
                class_name=class_name,
                anchor_frame_idx=int(cand["frame_idx"]),
                anchor_box_xyxy=cand["bbox_xyxy"],
                conf=conf,
                device=device,
                temporal_window_radius=temporal_window_radius,
                image_w=image_w,
                image_h=image_h,
            )
            init_score = (
                0.45 * float(cand["conf"])
                + 0.35 * float(consensus["temporal_mean_iou"])
                + 0.20 * float(consensus["temporal_match_ratio"])
            )
            scored.append(
                {
                    **cand,
                    **consensus,
                    "init_score": float(init_score),
                }
            )
        scored.sort(key=lambda x: float(x["init_score"]), reverse=True)
        found[class_name] = scored[: max(1, max_prompt_frames_per_class)]
    return found


def discover_legacy_prompt_frames(
    input_video: str,
    yolo_model,
    target_classes: Sequence[str],
    conf: float,
    device: str,
    prompt_interval: int,
    max_prompt_frames_per_class: int,
) -> Dict[str, List[int]]:
    found: Dict[str, List[int]] = {name: [] for name in target_classes}
    sampled_frames = iter_sampled_frames(input_video, prompt_interval)
    per_class_limit = max(1, int(max_prompt_frames_per_class))
    for frame_idx, frame in sampled_frames:
        detections = run_yolo_detections(
            model=yolo_model,
            frame=frame,
            target_classes=target_classes,
            conf=conf,
            device=device,
        )
        seen_in_frame = {str(det["class_name"]) for det in detections}
        for class_name in target_classes:
            if class_name not in seen_in_frame:
                continue
            if len(found[class_name]) >= per_class_limit:
                continue
            found[class_name].append(int(frame_idx))
    return found


def discover_text_prompt_frames_scored(
    input_video: str,
    yolo_model,
    target_classes: Sequence[str],
    conf: float,
    device: str,
    prompt_interval: int,
    max_prompt_frames_per_class: int,
    image_w: int,
    image_h: int,
    temporal_window_radius: int,
) -> Dict[str, List[int]]:
    sampled_frames = iter_sampled_frames(input_video, prompt_interval)
    candidates_by_class: Dict[str, List[Dict[str, object]]] = {name: [] for name in target_classes}
    for frame_idx, frame in sampled_frames:
        detections = run_yolo_detections(
            model=yolo_model,
            frame=frame,
            target_classes=target_classes,
            conf=conf,
            device=device,
        )
        for det in detections:
            class_name = str(det["class_name"])
            consensus = build_temporal_consensus_detection(
                input_video=input_video,
                yolo_model=yolo_model,
                class_name=class_name,
                anchor_frame_idx=int(frame_idx),
                anchor_box_xyxy=det["bbox_xyxy"],
                conf=conf,
                device=device,
                temporal_window_radius=temporal_window_radius,
                image_w=image_w,
                image_h=image_h,
            )
            score = (
                0.45 * float(det["conf"])
                + 0.35 * float(consensus["temporal_match_ratio"])
                + 0.20 * float(consensus["temporal_mean_iou"])
            )
            candidates_by_class[class_name].append(
                {
                    "frame_idx": int(frame_idx),
                    "score": float(score),
                }
            )

    found: Dict[str, List[int]] = {name: [] for name in target_classes}
    min_gap = max(1, int(prompt_interval) // 2)
    per_class_limit = max(1, int(max_prompt_frames_per_class))
    for class_name, candidates in candidates_by_class.items():
        ranked = sorted(candidates, key=lambda x: (float(x["score"]), int(x["frame_idx"])), reverse=True)
        picked: List[int] = []
        for cand in ranked:
            frame_idx = int(cand["frame_idx"])
            if any(abs(frame_idx - prev) < min_gap for prev in picked):
                continue
            picked.append(frame_idx)
            if len(picked) >= per_class_limit:
                break
        found[class_name] = sorted(picked)
    return found


def discover_box_prompt_infos_for_frames(
    input_video: str,
    yolo_model,
    class_to_frames: Dict[str, List[int]],
    conf: float,
    device: str,
    image_w: int,
    image_h: int,
    temporal_window_radius: int,
) -> Dict[str, List[Dict[str, object]]]:
    found: Dict[str, List[Dict[str, object]]] = {name: [] for name in class_to_frames.keys()}
    for class_name, frame_indices in class_to_frames.items():
        for frame_idx in frame_indices:
            frame = read_frame_at(input_video, int(frame_idx))
            detections = [
                det
                for det in run_yolo_detections(
                    model=yolo_model,
                    frame=frame,
                    target_classes=[class_name],
                    conf=conf,
                    device=device,
                )
                if str(det["class_name"]) == class_name
            ]
            if not detections:
                continue
            best_det = max(detections, key=lambda x: float(x["conf"]))
            consensus = build_temporal_consensus_detection(
                input_video=input_video,
                yolo_model=yolo_model,
                class_name=class_name,
                anchor_frame_idx=int(frame_idx),
                anchor_box_xyxy=best_det["bbox_xyxy"],
                conf=conf,
                device=device,
                temporal_window_radius=temporal_window_radius,
                image_w=image_w,
                image_h=image_h,
            )
            found[class_name].append(
                {
                    "frame_idx": int(frame_idx),
                    "bbox_xyxy": [float(v) for v in consensus["bbox_xyxy"]],
                    "bbox_xywh": [float(v) for v in consensus["bbox_xywh"]],
                    "conf": float(best_det["conf"]),
                    "temporal_match_ratio": float(consensus["temporal_match_ratio"]),
                    "temporal_mean_iou": float(consensus["temporal_mean_iou"]),
                    "area_ratio": float(consensus["area_ratio"]),
                    "slenderness": float(consensus["slenderness"]),
                }
            )
    return found


def _install_edt_shim() -> None:
    # Official SAM3 inference imports a Triton EDT kernel even when we only need
    # standard video prediction. On Windows, Triton is typically unavailable, so
    # we replace it with a CPU/OpenCV-compatible fallback that preserves behavior.
    edt_mod = types.ModuleType("sam3.model.edt")

    def edt_triton(data: torch.Tensor) -> torch.Tensor:
        assert data.dim() == 3
        device = data.device
        arr = data.detach().to("cpu").numpy()
        outs = []
        for mask in arr:
            dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 0)
            outs.append(torch.from_numpy(dt))
        return torch.stack(outs, dim=0).to(device=device, dtype=torch.float32)

    edt_mod.edt_triton = edt_triton
    sys.modules["sam3.model.edt"] = edt_mod


def _install_nms_shim() -> None:
    nms_mod = types.ModuleType("sam3.perflib.triton.nms")

    def nms_triton(ious: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
        ious_np = ious.float().detach().cpu().numpy()
        scores_np = scores.float().detach().cpu().numpy()
        order = scores_np.argsort()[::-1]
        kept = []
        while order.size > 0:
            idx = int(order[0])
            kept.append(idx)
            remain = np.where(ious_np[idx, order[1:]] <= iou_threshold)[0]
            order = order[remain + 1]
        return torch.tensor(kept, dtype=torch.int64, device=scores.device)

    nms_mod.nms_triton = nms_triton
    sys.modules["sam3.perflib.triton.nms"] = nms_mod


def _install_cc_shim() -> None:
    cc_mod = types.ModuleType("sam3.perflib.triton.connected_components")

    def connected_components_triton(input_tensor: torch.Tensor):
        if input_tensor.dim() == 4 and input_tensor.shape[1] == 1:
            work = input_tensor[:, 0]
        elif input_tensor.dim() == 3:
            work = input_tensor
        else:
            raise ValueError("Input tensor must be (B,H,W) or (B,1,H,W)")

        if work.shape[0] == 0:
            empty = torch.zeros_like(input_tensor, dtype=torch.int32)
            return empty, empty.clone()

        labels_list = []
        counts_list = []
        for sample in work.detach().to("cpu").numpy():
            sample_u8 = sample.astype(np.uint8)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(sample_u8, connectivity=8)
            counts = np.zeros_like(labels, dtype=np.int32)
            for cid in range(1, num_labels):
                counts[labels == cid] = int(stats[cid, cv2.CC_STAT_AREA])
            labels_list.append(torch.from_numpy(labels.astype(np.int32)))
            counts_list.append(torch.from_numpy(counts))

        labels_tensor = torch.stack(labels_list, dim=0).to(device=input_tensor.device)
        counts_tensor = torch.stack(counts_list, dim=0).to(device=input_tensor.device)
        if input_tensor.dim() == 4:
            labels_tensor = labels_tensor.unsqueeze(1)
            counts_tensor = counts_tensor.unsqueeze(1)
        return labels_tensor, counts_tensor

    cc_mod.connected_components_triton = connected_components_triton
    sys.modules["sam3.perflib.triton.connected_components"] = cc_mod


def install_runtime_shims() -> List[str]:
    used_shims: List[str] = []
    module_to_installer = {
        "sam3.model.edt": _install_edt_shim,
        "sam3.perflib.triton.nms": _install_nms_shim,
        "sam3.perflib.triton.connected_components": _install_cc_shim,
    }
    for module_name, installer in module_to_installer.items():
        try:
            importlib.import_module(module_name)
        except Exception:
            installer()
            used_shims.append(module_name)
    return used_shims


def load_hq_sam_predictor(
    checkpoint_path: str,
    model_type: str,
    device: str,
):
    from segment_anything_hq import SamPredictor, sam_model_registry

    model = sam_model_registry[model_type](checkpoint=checkpoint_path)
    model.to(device=device)
    model.eval()
    return SamPredictor(model)


def _cast_nested_tensor_fp32(value):
    if torch.is_tensor(value):
        return value.to(dtype=torch.float32)
    if isinstance(value, list):
        return [_cast_nested_tensor_fp32(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_cast_nested_tensor_fp32(v) for v in value)
    return value


def normalize_hq_sam_predictor_state_fp32(predictor) -> None:
    predictor.features = _cast_nested_tensor_fp32(predictor.features)
    predictor.interm_features = _cast_nested_tensor_fp32(predictor.interm_features)


def predict_with_hq_sam_box(
    predictor,
    box_xyxy: Sequence[float],
    multimask_output: bool = True,
    hq_token_only: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    box = predictor.transform.apply_boxes(np.asarray(box_xyxy, dtype=np.float32), predictor.original_size)
    box_torch = torch.as_tensor(box, dtype=torch.float32, device=predictor.device)[None, :]
    amp_ctx = contextlib.nullcontext()
    if predictor.device.type == "cuda":
        amp_ctx = torch.autocast(device_type="cuda", enabled=False)
    with torch.inference_mode(), amp_ctx:
        masks, iou_predictions, low_res_masks = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=box_torch,
            mask_input=None,
            multimask_output=multimask_output,
            return_logits=False,
            hq_token_only=hq_token_only,
        )
    masks_np = masks[0].detach().to(dtype=torch.float32).cpu().numpy()
    iou_predictions_np = iou_predictions[0].detach().to(dtype=torch.float32).cpu().numpy()
    low_res_masks_np = low_res_masks[0].detach().to(dtype=torch.float32).cpu().numpy()
    return masks_np, iou_predictions_np, low_res_masks_np


def load_sam3_predictor(
    sam3_repo: str,
    checkpoint_path: str,
    compile_model: bool,
    async_loading_frames: bool,
):
    project_root = Path(__file__).resolve().parents[1]
    triton_cache_dir = project_root / ".cache" / "triton"
    triton_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TRITON_CACHE_DIR", str(triton_cache_dir))
    sys.path.insert(0, sam3_repo)
    used_shims = install_runtime_shims()
    from sam3.model_builder import build_sam3_predictor

    predictor = build_sam3_predictor(
        checkpoint_path=checkpoint_path,
        version="sam3",
        compile=compile_model,
        async_loading_frames=async_loading_frames,
    )
    return predictor, used_shims


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


def apply_yolo_constraint(
    mask: np.ndarray,
    constraint_box_xyxy: Optional[Sequence[float]],
    padding_ratio: float,
) -> np.ndarray:
    if constraint_box_xyxy is None:
        return mask
    h, w = mask.shape[:2]
    expanded = expand_box_xyxy(constraint_box_xyxy, w, h, padding_ratio)
    x1, y1, x2, y2 = [int(round(v)) for v in expanded]
    box_mask = np.zeros_like(mask, dtype=np.uint8)
    box_mask[y1:y2, x1:x2] = 255
    if np.count_nonzero(mask & box_mask) == 0:
        return mask
    return np.where(box_mask > 0, mask, 0).astype(np.uint8)


def is_mask_anomalous(
    current_mask: np.ndarray,
    previous_mask: Optional[np.ndarray],
    current_box: Optional[Sequence[float]],
    previous_box: Optional[Sequence[float]],
    min_mask_area: int,
    area_ratio_min: float,
    area_ratio_max: float,
    center_jump_factor: float,
) -> Optional[str]:
    area = int(np.count_nonzero(current_mask))
    if area < max(1, min_mask_area):
        return "mask_too_small"

    if previous_mask is None:
        return None

    previous_area = int(np.count_nonzero(previous_mask))
    if previous_area >= max(1, min_mask_area):
        ratio = area / max(1.0, float(previous_area))
        if ratio < area_ratio_min:
            return "area_drop"
        if ratio > area_ratio_max:
            return "area_blowup"

    if current_box is not None and previous_box is not None:
        jump = bbox_center_distance(current_box, previous_box)
        px1, py1, px2, py2 = [float(v) for v in previous_box]
        prev_diag = max(1.0, ((px2 - px1) ** 2 + (py2 - py1) ** 2) ** 0.5)
        if jump > center_jump_factor * prev_diag:
            return "center_jump"

    return None


def match_detection_to_history(
    detections: Sequence[Dict[str, object]],
    history_box_xyxy: Optional[Sequence[float]],
) -> Optional[Dict[str, object]]:
    if not detections:
        return None
    if history_box_xyxy is None:
        return max(detections, key=lambda d: float(d["conf"]))

    best_det: Optional[Dict[str, object]] = None
    best_score = -1.0
    hx1, hy1, hx2, hy2 = [float(v) for v in history_box_xyxy]
    history_diag = max(1.0, ((hx2 - hx1) ** 2 + (hy2 - hy1) ** 2) ** 0.5)
    for det in detections:
        det_box = det["bbox_xyxy"]
        iou = bbox_iou_xyxy(history_box_xyxy, det_box)
        distance = bbox_center_distance(history_box_xyxy, det_box)
        center_score = max(0.0, 1.0 - distance / (2.5 * history_diag))
        conf_score = float(det["conf"])
        score = 0.6 * iou + 0.25 * center_score + 0.15 * conf_score
        if score > best_score:
            best_score = score
            best_det = det
    if best_score < 0.08:
        return None
    return best_det


def should_use_text_guidance(
    prompt_info: Dict[str, object],
    temporal_match_ratio_min: float,
    temporal_mean_iou_min: float,
    small_area_ratio_max: float,
    slenderness_max: float,
) -> bool:
    match_ratio = float(prompt_info.get("temporal_match_ratio", 0.0))
    mean_iou = float(prompt_info.get("temporal_mean_iou", 0.0))
    area_ratio = float(prompt_info.get("area_ratio", 0.0))
    slenderness = float(prompt_info.get("slenderness", 999.0))
    if match_ratio < temporal_match_ratio_min:
        return True
    if mean_iou < temporal_mean_iou_min:
        return True
    if area_ratio <= small_area_ratio_max:
        return True
    if slenderness >= slenderness_max:
        return True
    return False


def add_detection_prompt(
    predictor,
    session_id: str,
    frame_idx: int,
    class_name: Optional[str],
    bbox_xywh: Sequence[float],
    image_w: int,
    image_h: int,
    output_prob_thresh: float,
    clear_old_boxes: bool,
) -> None:
    x, y, w, h = [float(v) for v in bbox_xywh]
    normalized_box = [
        x / max(1.0, float(image_w)),
        y / max(1.0, float(image_h)),
        w / max(1.0, float(image_w)),
        h / max(1.0, float(image_h)),
    ]
    predictor.handle_request(
        {
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": int(frame_idx),
            "text": class_name,
            "bounding_boxes": [normalized_box],
            "bounding_box_labels": [1],
            "rel_coordinates": True,
            "clear_old_boxes": clear_old_boxes,
            "output_prob_thresh": output_prob_thresh,
        }
    )


def load_supervised_box_prompts(
    gt_mask_dir: str,
    frame_idx: int,
    min_component_area: int,
    image_w: int,
    image_h: int,
) -> List[Dict[str, object]]:
    mask_paths = list_images(gt_mask_dir)
    if not mask_paths:
        raise RuntimeError(f"No GT mask images found in {gt_mask_dir}")
    if frame_idx < 0 or frame_idx >= len(mask_paths):
        raise RuntimeError(f"supervised-init-frame-idx {frame_idx} out of range for {len(mask_paths)} masks")

    mask = cv2.imread(mask_paths[frame_idx], cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Failed to read GT mask: {mask_paths[frame_idx]}")
    mask_bin = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
    prompts: List[Dict[str, object]] = []
    component_id = 0
    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area < int(min_component_area):
            continue
        x = int(stats[label_idx, cv2.CC_STAT_LEFT])
        y = int(stats[label_idx, cv2.CC_STAT_TOP])
        width = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        height = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
        box_xyxy = clamp_box_xyxy([x, y, x + width, y + height], image_w, image_h)
        prompts.append(
            {
                "frame_idx": int(frame_idx),
                "component_id": int(component_id),
                "bbox_xyxy": box_xyxy,
                "bbox_xywh": bbox_xyxy_to_xywh(box_xyxy),
                "area": area,
            }
        )
        component_id += 1
    if not prompts:
        raise RuntimeError(
            f"Supervised init mask {mask_paths[frame_idx]} has no connected component >= {min_component_area} pixels"
        )
    return prompts


def load_supervised_mask_prompts(
    gt_mask_dir: str,
    frame_idx: int,
    min_component_area: int,
    image_w: int,
    image_h: int,
) -> List[Dict[str, object]]:
    mask_paths = list_images(gt_mask_dir)
    if not mask_paths:
        raise RuntimeError(f"No GT mask images found in {gt_mask_dir}")
    if frame_idx < 0 or frame_idx >= len(mask_paths):
        raise RuntimeError(f"supervised-init-frame-idx {frame_idx} out of range for {len(mask_paths)} masks")

    mask = cv2.imread(mask_paths[frame_idx], cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Failed to read GT mask: {mask_paths[frame_idx]}")
    mask_bin = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
    prompts: List[Dict[str, object]] = []
    component_id = 0
    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area < int(min_component_area):
            continue
        component_mask = (labels == label_idx).astype(np.uint8)
        x = int(stats[label_idx, cv2.CC_STAT_LEFT])
        y = int(stats[label_idx, cv2.CC_STAT_TOP])
        width = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        height = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
        box_xyxy = clamp_box_xyxy([x, y, x + width, y + height], image_w, image_h)
        prompts.append(
            {
                "frame_idx": int(frame_idx),
                "component_id": int(component_id),
                "bbox_xyxy": box_xyxy,
                "bbox_xywh": bbox_xyxy_to_xywh(box_xyxy),
                "area": area,
                "mask": component_mask,
            }
        )
        component_id += 1
    if not prompts:
        raise RuntimeError(
            f"Supervised init mask {mask_paths[frame_idx]} has no connected component >= {min_component_area} pixels"
        )
    return prompts


def load_supervised_union_mask_prompt(
    gt_mask_dir: str,
    frame_idx: int,
    min_component_area: int,
    image_w: int,
    image_h: int,
) -> Dict[str, object]:
    mask_paths = list_images(gt_mask_dir)
    if not mask_paths:
        raise RuntimeError(f"No GT mask images found in {gt_mask_dir}")
    if frame_idx < 0 or frame_idx >= len(mask_paths):
        raise RuntimeError(f"supervised-init-frame-idx {frame_idx} out of range for {len(mask_paths)} masks")

    mask = cv2.imread(mask_paths[frame_idx], cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Failed to read GT mask: {mask_paths[frame_idx]}")
    mask_bin = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
    merged = np.zeros_like(mask_bin, dtype=np.uint8)
    kept_area = 0
    kept_components = 0
    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area < int(min_component_area):
            continue
        merged[labels == label_idx] = 1
        kept_area += area
        kept_components += 1
    if kept_area <= 0:
        raise RuntimeError(
            f"Supervised init mask {mask_paths[frame_idx]} has no connected component >= {min_component_area} pixels"
        )
    box_xyxy = mask_to_bbox_xyxy((merged > 0).astype(np.uint8) * 255)
    if box_xyxy is None:
        raise RuntimeError(f"Failed to compute bbox from GT mask: {mask_paths[frame_idx]}")
    box_xyxy = clamp_box_xyxy(box_xyxy, image_w, image_h)
    return {
        "frame_idx": int(frame_idx),
        "bbox_xyxy": box_xyxy,
        "bbox_xywh": bbox_xyxy_to_xywh(box_xyxy),
        "area": int(kept_area),
        "component_count": int(kept_components),
        "mask": merged.astype(np.uint8),
    }


def add_text_prompt(
    predictor,
    session_id: str,
    frame_idx: int,
    class_name: str,
    output_prob_thresh: float,
) -> None:
    predictor.handle_request(
        {
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": int(frame_idx),
            "text": class_name,
            "output_prob_thresh": output_prob_thresh,
        }
    )


def stream_propagation_masks(
    predictor,
    session_id: str,
    start_frame_index: int,
    propagation_direction: str,
    max_frame_num_to_track: int | None,
    output_prob_thresh: float,
) -> Dict[int, np.ndarray]:
    outputs: Dict[int, np.ndarray] = {}
    for item in predictor.handle_stream_request(
        {
            "type": "propagate_in_video",
            "session_id": session_id,
            "propagation_direction": propagation_direction,
            "start_frame_index": int(start_frame_index),
            "max_frame_num_to_track": max_frame_num_to_track,
            "output_prob_thresh": output_prob_thresh,
        }
    ):
        frame_idx = int(item["frame_index"])
        out = item["outputs"]
        masks = out.get("out_binary_masks")
        if masks is None or len(masks) == 0:
            continue
        if isinstance(masks, torch.Tensor):
            masks = masks.detach().cpu().numpy()
        masks = np.asarray(masks)
        if masks.ndim == 4:
            masks = masks[:, 0]
        frame_mask = np.any(masks > 0, axis=0).astype(np.uint8) * 255
        outputs[frame_idx] = frame_mask
    return outputs


def propagate_text_prompt_only(
    predictor,
    input_video: str,
    class_name: str,
    prompt_frame: int,
    output_prob_thresh: float,
    offload_video_to_cpu: bool,
    offload_state_to_cpu: bool,
    max_frame_num_to_track: int | None,
    propagation_direction: str,
) -> Dict[int, np.ndarray]:
    response = predictor.handle_request(
        {
            "type": "start_session",
            "resource_path": input_video,
            "offload_video_to_cpu": offload_video_to_cpu,
            "offload_state_to_cpu": offload_state_to_cpu,
        }
    )
    session_id = response["session_id"]
    try:
        add_text_prompt(
            predictor=predictor,
            session_id=session_id,
            frame_idx=prompt_frame,
            class_name=class_name,
            output_prob_thresh=output_prob_thresh,
        )
        return stream_propagation_masks(
            predictor=predictor,
            session_id=session_id,
            start_frame_index=prompt_frame,
            propagation_direction=propagation_direction,
            max_frame_num_to_track=max_frame_num_to_track,
            output_prob_thresh=output_prob_thresh,
        )
    finally:
        predictor.handle_request({"type": "close_session", "session_id": session_id})


def propagate_with_box_prompt_only(
    predictor,
    input_video: str,
    prompt_frame: int,
    bbox_xywh: Sequence[float],
    image_w: int,
    image_h: int,
    output_prob_thresh: float,
    offload_video_to_cpu: bool,
    offload_state_to_cpu: bool,
    max_frame_num_to_track: int | None,
    propagation_direction: str,
) -> Dict[int, np.ndarray]:
    response = predictor.handle_request(
        {
            "type": "start_session",
            "resource_path": input_video,
            "offload_video_to_cpu": offload_video_to_cpu,
            "offload_state_to_cpu": offload_state_to_cpu,
        }
    )
    session_id = response["session_id"]
    try:
        add_detection_prompt(
            predictor=predictor,
            session_id=session_id,
            frame_idx=prompt_frame,
            class_name=None,
            bbox_xywh=bbox_xywh,
            image_w=image_w,
            image_h=image_h,
            output_prob_thresh=output_prob_thresh,
            clear_old_boxes=True,
        )
        return stream_propagation_masks(
            predictor=predictor,
            session_id=session_id,
            start_frame_index=prompt_frame,
            propagation_direction=propagation_direction,
            max_frame_num_to_track=max_frame_num_to_track,
            output_prob_thresh=output_prob_thresh,
        )
    finally:
        predictor.handle_request({"type": "close_session", "session_id": session_id})


def propagate_with_mask_prompt_only(
    predictor,
    input_video: str,
    prompt_frame: int,
    component_mask: np.ndarray,
    output_prob_thresh: float,
    offload_video_to_cpu: bool,
    offload_state_to_cpu: bool,
    max_frame_num_to_track: int | None,
    propagation_direction: str,
    obj_id: int = 0,
) -> Dict[int, np.ndarray]:
    response = predictor.handle_request(
        {
            "type": "start_session",
            "resource_path": input_video,
            "offload_video_to_cpu": offload_video_to_cpu,
            "offload_state_to_cpu": offload_state_to_cpu,
        }
    )
    session_id = response["session_id"]
    try:
        _session_add_mask_prompt(
            predictor=predictor,
            session_id=session_id,
            frame_idx=prompt_frame,
            obj_id=obj_id,
            component_mask=component_mask,
        )
        return stream_propagation_masks(
            predictor=predictor,
            session_id=session_id,
            start_frame_index=prompt_frame,
            propagation_direction=propagation_direction,
            max_frame_num_to_track=max_frame_num_to_track,
            output_prob_thresh=output_prob_thresh,
        )
    finally:
        predictor.handle_request({"type": "close_session", "session_id": session_id})


def _session_add_mask_prompt(
    predictor,
    session_id: str,
    frame_idx: int,
    obj_id: int,
    component_mask: np.ndarray,
) -> None:
    session = predictor._get_session(session_id)
    inference_state = session["state"]
    predictor._extend_expiration_time(session)
    mask_tensor = torch.from_numpy((component_mask > 0).astype(np.float32)).to(predictor.device)
    if mask_tensor.ndim != 2:
        raise ValueError("component_mask must be a 2D array")
    if "tracker_inference_states" in inference_state:
        # SAM3 video inference internally mixes tracker/bootstrap logic that is not fully
        # compatible with tensors created under torch.inference_mode(). Run the bootstrap
        # path under no_grad with inference_mode temporarily disabled.
        with torch.inference_mode(False), torch.no_grad():
            if hasattr(predictor.model, "_prepare_backbone_feats"):
                predictor.model._prepare_backbone_feats(inference_state, int(frame_idx), reverse=False)
            if not inference_state["tracker_inference_states"] or not inference_state["tracker_metadata"]:
                predictor.model.add_fake_objects_to_inference_state(
                    inference_state=inference_state,
                    num_objects=max(1, int(obj_id) + 1),
                    frame_idx=int(frame_idx),
                )
        tracker_states = predictor.model._get_tracker_inference_states_by_obj_ids(inference_state, [int(obj_id)])
        if not tracker_states:
            raise RuntimeError(f"Failed to resolve tracker state for obj_id={obj_id}")
        tracker_state = tracker_states[0]
        if predictor.device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                predictor.model.tracker.add_new_mask(
                    inference_state=tracker_state,
                    frame_idx=int(frame_idx),
                    obj_id=int(obj_id),
                    mask=mask_tensor,
                    add_mask_to_memory=True,
                )
                predictor.model.tracker.propagate_in_video_preflight(tracker_state, run_mem_encoder=True)
        else:
            predictor.model.tracker.add_new_mask(
                inference_state=tracker_state,
                frame_idx=int(frame_idx),
                obj_id=int(obj_id),
                mask=mask_tensor,
                add_mask_to_memory=True,
            )
            predictor.model.tracker.propagate_in_video_preflight(tracker_state, run_mem_encoder=True)
    else:
        if predictor.device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                predictor.model.tracker.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=int(frame_idx),
                    obj_id=int(obj_id),
                    mask=mask_tensor,
                    add_mask_to_memory=True,
                )
                predictor.model.tracker.propagate_in_video_preflight(inference_state, run_mem_encoder=True)
        else:
            predictor.model.tracker.add_new_mask(
                inference_state=inference_state,
                frame_idx=int(frame_idx),
                obj_id=int(obj_id),
                mask=mask_tensor,
                add_mask_to_memory=True,
            )
            predictor.model.tracker.propagate_in_video_preflight(inference_state, run_mem_encoder=True)


def propagate_with_mask_reprompt(
    predictor,
    input_video: str,
    prompt_frame: int,
    component_mask: np.ndarray,
    output_prob_thresh: float,
    offload_video_to_cpu: bool,
    offload_state_to_cpu: bool,
    max_frame_num_to_track: int | None,
    propagation_direction: str,
    reprompt_interval: int,
    reprompt_min_mask_area: int,
    obj_id: int = 0,
) -> Tuple[Dict[int, np.ndarray], List[Dict[str, object]]]:
    response = predictor.handle_request(
        {
            "type": "start_session",
            "resource_path": input_video,
            "offload_video_to_cpu": offload_video_to_cpu,
            "offload_state_to_cpu": offload_state_to_cpu,
        }
    )
    session_id = response["session_id"]
    outputs: Dict[int, np.ndarray] = {}
    events: List[Dict[str, object]] = []
    current_start = int(prompt_frame)
    current_direction = str(propagation_direction)
    interval = max(0, int(reprompt_interval))

    try:
        _session_add_mask_prompt(
            predictor=predictor,
            session_id=session_id,
            frame_idx=current_start,
            obj_id=obj_id,
            component_mask=component_mask,
        )
        last_reprompt_frame = current_start
        while True:
            chunk_limit = max_frame_num_to_track
            if interval > 0:
                if chunk_limit is None:
                    chunk_limit = interval
                else:
                    chunk_limit = min(int(chunk_limit), interval)
            chunk_outputs = stream_propagation_masks(
                predictor=predictor,
                session_id=session_id,
                start_frame_index=current_start,
                propagation_direction=current_direction,
                max_frame_num_to_track=chunk_limit,
                output_prob_thresh=output_prob_thresh,
            )
            for out_idx, mask in chunk_outputs.items():
                outputs[out_idx] = mask
            if interval <= 0 or current_direction == "backward" or not chunk_outputs:
                break

            candidate_frames = [idx for idx in sorted(chunk_outputs.keys()) if idx > last_reprompt_frame]
            if not candidate_frames:
                break
            reprompt_frame = int(candidate_frames[-1])
            reprompt_mask = chunk_outputs[reprompt_frame]
            reprompt_area = int(np.count_nonzero(reprompt_mask))
            if reprompt_area < int(reprompt_min_mask_area):
                events.append(
                    {
                        "event": "reprompt_skipped",
                        "frame_idx": reprompt_frame,
                        "reason": "pred_mask_too_small",
                        "mask_area": reprompt_area,
                    }
                )
                break
            _session_add_mask_prompt(
                predictor=predictor,
                session_id=session_id,
                frame_idx=reprompt_frame,
                obj_id=obj_id,
                component_mask=(reprompt_mask > 0).astype(np.uint8),
            )
            events.append(
                {
                    "event": "self_reprompt",
                    "frame_idx": reprompt_frame,
                    "mask_area": reprompt_area,
                }
            )
            if max_frame_num_to_track is not None:
                consumed = max(0, reprompt_frame - current_start)
                max_frame_num_to_track = max(0, int(max_frame_num_to_track) - consumed)
                if max_frame_num_to_track <= 0:
                    break
            current_start = reprompt_frame
            last_reprompt_frame = reprompt_frame
            current_direction = "forward"
    finally:
        predictor.handle_request({"type": "close_session", "session_id": session_id})

    return outputs, events


def propagate_with_yolo_redetect(
    predictor,
    yolo_model,
    input_video: str,
    class_name: str,
    initial_prompt: Dict[str, object],
    output_prob_thresh: float,
    offload_video_to_cpu: bool,
    offload_state_to_cpu: bool,
    max_frame_num_to_track: int | None,
    conf: float,
    device: str,
    redetect_interval: int,
    min_reprompt_gap: int,
    yolo_constraint_padding: float,
    apply_yolo_mask_constraint: bool,
    anomaly_area_ratio_min: float,
    anomaly_area_ratio_max: float,
    anomaly_center_jump_factor: float,
    min_mask_area: int,
    image_w: int,
    image_h: int,
    initial_propagation_direction: str,
    temporal_window_radius: int,
    enable_redetect: bool = True,
) -> Tuple[Dict[int, np.ndarray], List[Dict[str, object]]]:
    response = predictor.handle_request(
        {
            "type": "start_session",
            "resource_path": input_video,
            "offload_video_to_cpu": offload_video_to_cpu,
            "offload_state_to_cpu": offload_state_to_cpu,
        }
    )
    session_id = response["session_id"]
    outputs: Dict[int, np.ndarray] = {}
    events: List[Dict[str, object]] = []
    current_start = int(initial_prompt["frame_idx"])
    current_box_xywh = [float(v) for v in initial_prompt["bbox_xywh"]]
    last_constraint_box_xyxy = bbox_xywh_to_xyxy(current_box_xywh)
    previous_mask: Optional[np.ndarray] = None
    previous_box: Optional[List[float]] = None
    last_reprompt_frame = -10**9
    reprompt_attempts: Dict[int, int] = {}
    current_direction = initial_propagation_direction

    try:
        add_detection_prompt(
            predictor=predictor,
            session_id=session_id,
            frame_idx=current_start,
            class_name=class_name,
            bbox_xywh=current_box_xywh,
            image_w=image_w,
            image_h=image_h,
            output_prob_thresh=output_prob_thresh,
            clear_old_boxes=True,
        )

        while True:
            restart_from: Optional[int] = None
            restart_box_xywh: Optional[List[float]] = None
            restart_reason: Optional[str] = None
            for item in predictor.handle_stream_request(
                {
                    "type": "propagate_in_video",
                    "session_id": session_id,
                    "propagation_direction": current_direction,
                    "start_frame_index": int(current_start),
                    "max_frame_num_to_track": max_frame_num_to_track,
                    "output_prob_thresh": output_prob_thresh,
                }
            ):
                frame_idx = int(item["frame_index"])
                out = item["outputs"]
                masks = out.get("out_binary_masks")
                frame_mask = None
                if masks is not None and len(masks) > 0:
                    if isinstance(masks, torch.Tensor):
                        masks = masks.detach().cpu().numpy()
                    masks = np.asarray(masks)
                    if masks.ndim == 4:
                        masks = masks[:, 0]
                    frame_mask = np.any(masks > 0, axis=0).astype(np.uint8) * 255
                if frame_mask is None:
                    continue

                raw_box = mask_to_bbox_xyxy(frame_mask)
                anomaly_reason = is_mask_anomalous(
                    current_mask=frame_mask,
                    previous_mask=previous_mask,
                    current_box=raw_box,
                    previous_box=previous_box,
                    min_mask_area=min_mask_area,
                    area_ratio_min=anomaly_area_ratio_min,
                    area_ratio_max=anomaly_area_ratio_max,
                    center_jump_factor=anomaly_center_jump_factor,
                )

                should_redetect = False
                if enable_redetect:
                    if (
                        redetect_interval > 0
                        and frame_idx > current_start
                        and ((frame_idx - current_start) % redetect_interval == 0)
                    ):
                        should_redetect = True
                    if (
                        frame_idx >= current_start
                        and anomaly_reason is not None
                        and frame_idx - last_reprompt_frame >= min_reprompt_gap
                    ):
                        should_redetect = True

                matched_det: Optional[Dict[str, object]] = None
                if should_redetect:
                    frame = read_frame_at(input_video, frame_idx)
                    detections = [
                        det
                        for det in run_yolo_detections(
                            model=yolo_model,
                            frame=frame,
                            target_classes=[class_name],
                            conf=conf,
                            device=device,
                        )
                        if str(det["class_name"]) == class_name
                    ]
                    matched_det = match_detection_to_history(detections, raw_box or previous_box or last_constraint_box_xyxy)
                    if matched_det is not None:
                        consensus_det = build_temporal_consensus_detection(
                            input_video=input_video,
                            yolo_model=yolo_model,
                            class_name=class_name,
                            anchor_frame_idx=frame_idx,
                            anchor_box_xyxy=matched_det["bbox_xyxy"],
                            conf=conf,
                            device=device,
                            temporal_window_radius=temporal_window_radius,
                            image_w=image_w,
                            image_h=image_h,
                        )
                        matched_det = {
                            **matched_det,
                            "bbox_xyxy": [float(v) for v in consensus_det["bbox_xyxy"]],
                            "bbox_xywh": [float(v) for v in consensus_det["bbox_xywh"]],
                            "temporal_match_ratio": float(consensus_det["temporal_match_ratio"]),
                            "temporal_mean_iou": float(consensus_det["temporal_mean_iou"]),
                            "slenderness": float(consensus_det["slenderness"]),
                            "area_ratio": float(consensus_det["area_ratio"]),
                        }
                        last_constraint_box_xyxy = [float(v) for v in matched_det["bbox_xyxy"]]

                constrained_mask = frame_mask
                if apply_yolo_mask_constraint:
                    constrained_mask = apply_yolo_constraint(
                        frame_mask,
                        constraint_box_xyxy=last_constraint_box_xyxy,
                        padding_ratio=yolo_constraint_padding,
                    )
                outputs[frame_idx] = constrained_mask
                previous_mask = constrained_mask
                previous_box = mask_to_bbox_xyxy(constrained_mask)

                if (
                    enable_redetect
                    and
                    anomaly_reason is not None
                    and matched_det is not None
                    and frame_idx > current_start
                    and reprompt_attempts.get(frame_idx, 0) < 2
                ):
                    restart_from = frame_idx
                    restart_box_xywh = [float(v) for v in matched_det["bbox_xywh"]]
                    restart_reason = anomaly_reason
                    reprompt_attempts[frame_idx] = reprompt_attempts.get(frame_idx, 0) + 1
                    events.append(
                        {
                            "event": "reprompt",
                            "frame_idx": frame_idx,
                            "class_name": class_name,
                            "reason": anomaly_reason,
                            "matched_conf": float(matched_det["conf"]),
                            "matched_bbox_xywh": restart_box_xywh,
                            "temporal_match_ratio": float(matched_det.get("temporal_match_ratio", 0.0)),
                            "temporal_mean_iou": float(matched_det.get("temporal_mean_iou", 0.0)),
                        }
                    )
                    predictor.handle_request({"type": "cancel_propagation", "session_id": session_id})
                    break

                if matched_det is not None:
                    events.append(
                        {
                            "event": "redetect",
                            "frame_idx": frame_idx,
                            "class_name": class_name,
                            "reason": anomaly_reason or "scheduled_check",
                            "matched_conf": float(matched_det["conf"]),
                            "matched_bbox_xywh": [float(v) for v in matched_det["bbox_xywh"]],
                            "temporal_match_ratio": float(matched_det.get("temporal_match_ratio", 0.0)),
                            "temporal_mean_iou": float(matched_det.get("temporal_mean_iou", 0.0)),
                        }
                    )

            if restart_from is None or restart_box_xywh is None:
                break

            add_detection_prompt(
                predictor=predictor,
                session_id=session_id,
                frame_idx=restart_from,
                class_name=class_name,
                bbox_xywh=restart_box_xywh,
                image_w=image_w,
                image_h=image_h,
                output_prob_thresh=output_prob_thresh,
                clear_old_boxes=False,
            )
            current_start = restart_from
            current_box_xywh = restart_box_xywh
            last_constraint_box_xyxy = bbox_xywh_to_xyxy(current_box_xywh)
            last_reprompt_frame = restart_from
            current_direction = "forward"
            events.append(
                {
                    "event": "restart",
                    "frame_idx": restart_from,
                    "class_name": class_name,
                    "reason": restart_reason,
                }
            )
    finally:
        predictor.handle_request({"type": "close_session", "session_id": session_id})

    return outputs, events


def choose_hq_refine_box(
    input_video: str,
    yolo_model,
    frame_idx: int,
    class_name: str,
    merged_mask: np.ndarray,
    conf: float,
    device: str,
    temporal_window_radius: int,
    image_w: int,
    image_h: int,
) -> Optional[List[float]]:
    current_mask_box = mask_to_bbox_xyxy(merged_mask)
    frame = read_frame_at(input_video, frame_idx)
    detections = [
        det
        for det in run_yolo_detections(
            model=yolo_model,
            frame=frame,
            target_classes=[class_name],
            conf=conf,
            device=device,
        )
        if str(det["class_name"]) == class_name
    ]
    matched_det = match_detection_to_history(detections, current_mask_box)
    if matched_det is not None:
        consensus = build_temporal_consensus_detection(
            input_video=input_video,
            yolo_model=yolo_model,
            class_name=class_name,
            anchor_frame_idx=frame_idx,
            anchor_box_xyxy=matched_det["bbox_xyxy"],
            conf=conf,
            device=device,
            temporal_window_radius=temporal_window_radius,
            image_w=image_w,
            image_h=image_h,
        )
        return [float(v) for v in consensus["bbox_xyxy"]]
    if current_mask_box is not None:
        return [float(v) for v in current_mask_box]
    return None


def refine_keyframes_with_hq_sam(
    predictor,
    input_video: str,
    yolo_model,
    merged_masks: Dict[int, np.ndarray],
    refine_specs: Sequence[Dict[str, object]],
    conf: float,
    device: str,
    temporal_window_radius: int,
    image_w: int,
    image_h: int,
    min_overlap_iou: float,
    min_area_ratio: float,
    max_area_ratio: float,
    boundary_band_ksize: int,
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    frame_cache: Dict[int, np.ndarray] = {}
    used_keys: set[Tuple[int, str]] = set()
    for spec in refine_specs:
        frame_idx = int(spec["frame_idx"])
        class_name = str(spec["class_name"])
        dedupe_key = (frame_idx, class_name)
        if dedupe_key in used_keys:
            continue
        used_keys.add(dedupe_key)
        if frame_idx not in merged_masks:
            continue
        refine_box = choose_hq_refine_box(
            input_video=input_video,
            yolo_model=yolo_model,
            frame_idx=frame_idx,
            class_name=class_name,
            merged_mask=merged_masks[frame_idx],
            conf=conf,
            device=device,
            temporal_window_radius=temporal_window_radius,
            image_w=image_w,
            image_h=image_h,
        )
        if refine_box is None:
            records.append(
                {
                    "frame_idx": frame_idx,
                    "class_name": class_name,
                    "reason": str(spec.get("reason", "unknown")),
                    "status": "skipped_no_box",
                }
            )
            continue
        frame = frame_cache.get(frame_idx)
        if frame is None:
            frame = read_frame_at(input_video, frame_idx)
            frame_cache[frame_idx] = frame
        predictor.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        normalize_hq_sam_predictor_state_fp32(predictor)
        masks, scores, _ = predict_with_hq_sam_box(
            predictor=predictor,
            box_xyxy=refine_box,
            multimask_output=True,
            hq_token_only=True,
        )
        if masks is None or len(masks) == 0:
            records.append(
                {
                    "frame_idx": frame_idx,
                    "class_name": class_name,
                    "reason": str(spec.get("reason", "unknown")),
                    "status": "skipped_no_mask",
                }
            )
            continue
        best_idx = int(np.argmax(np.asarray(scores, dtype=np.float32)))
        best_mask = (np.asarray(masks[best_idx]).astype(np.uint8) > 0).astype(np.uint8) * 255
        refined = merged_masks[frame_idx].copy()
        x1, y1, x2, y2 = [int(round(v)) for v in clamp_box_xyxy(refine_box, image_w, image_h)]
        existing_region = refined[y1:y2, x1:x2]
        hq_region = best_mask[y1:y2, x1:x2]
        existing_bin = (existing_region > 0).astype(np.uint8)
        hq_bin = (hq_region > 0).astype(np.uint8)
        inter = int(np.logical_and(existing_bin > 0, hq_bin > 0).sum())
        union = int(np.logical_or(existing_bin > 0, hq_bin > 0).sum())
        overlap_iou = float(inter / max(1, union))
        area_ratio = float(hq_bin.sum() / max(1.0, float(existing_bin.sum()))) if existing_bin.sum() > 0 else float("inf")
        if (
            existing_bin.sum() == 0
            or overlap_iou < float(min_overlap_iou)
            or area_ratio < float(min_area_ratio)
            or area_ratio > float(max_area_ratio)
        ):
            records.append(
                {
                    "frame_idx": frame_idx,
                    "class_name": class_name,
                    "reason": str(spec.get("reason", "unknown")),
                    "status": "skipped_inconsistent",
                    "box_xyxy": [float(v) for v in refine_box],
                    "score": float(np.asarray(scores, dtype=np.float32)[best_idx]),
                    "overlap_iou": overlap_iou,
                    "area_ratio": area_ratio,
                }
            )
            continue
        if np.count_nonzero(existing_region) > 0:
            refined_region = existing_region.copy()
            band_ksize = max(1, int(boundary_band_ksize))
            if band_ksize > 1:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (band_ksize, band_ksize))
                existing_u8 = (existing_bin * 255).astype(np.uint8)
                hq_u8 = (hq_bin * 255).astype(np.uint8)
                existing_outer = cv2.dilate(existing_u8, kernel, iterations=1)
                existing_inner = cv2.erode(existing_u8, kernel, iterations=1)
                hq_outer = cv2.dilate(hq_u8, kernel, iterations=1)
                hq_inner = cv2.erode(hq_u8, kernel, iterations=1)
                boundary_band = (
                    ((existing_outer > 0) & (existing_inner == 0))
                    | ((hq_outer > 0) & (hq_inner == 0))
                )
            else:
                boundary_band = (existing_bin > 0) | (hq_bin > 0)
            hq_region_u8 = np.where(hq_region > 0, 255, 0).astype(np.uint8)
            refined_region[boundary_band] = hq_region_u8[boundary_band]
            refined[y1:y2, x1:x2] = refined_region
        else:
            refined = np.maximum(refined, best_mask)
        merged_masks[frame_idx] = refined
        records.append(
            {
                "frame_idx": frame_idx,
                "class_name": class_name,
                "reason": str(spec.get("reason", "unknown")),
                "status": "refined",
                "box_xyxy": [float(v) for v in refine_box],
                "score": float(np.asarray(scores, dtype=np.float32)[best_idx]),
                "overlap_iou": overlap_iou,
                "area_ratio": area_ratio,
                "boundary_band_ksize": int(boundary_band_ksize),
            }
        )
    return records


def main() -> None:
    ap = argparse.ArgumentParser("SAM3 automatic video mask generation with local checkpoint support")
    ap.add_argument("--input-video", type=str, required=True)
    ap.add_argument("--mask-dir", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument(
        "--target-classes",
        type=str,
        default="person,bicycle,sports ball,tennis racket",
        help="Comma-separated COCO class names used for YOLO11 initialization/redetection and SAM3 prompts",
    )
    ap.add_argument("--yolo-model", type=str, default="yolo11n.pt")
    ap.add_argument("--prompt-interval", type=int, default=18)
    ap.add_argument("--yolo-seg-model", type=str, default="yolov8n-seg.pt")
    ap.add_argument(
        "--text-prompt-interval",
        type=int,
        default=0,
        help="Optional sampling interval for late-object text prompts. Uses --prompt-interval when <= 0.",
    )
    ap.add_argument("--init-search-sampled-frames", type=int, default=3)
    ap.add_argument("--max-prompt-frames-per-class", type=int, default=1)
    ap.add_argument("--temporal-window-radius", type=int, default=2)
    ap.add_argument("--output-prob-thresh", type=float, default=0.5)
    ap.add_argument("--mask-dilate", type=int, default=5)
    ap.add_argument("--mask-close", type=int, default=3)
    ap.add_argument("--mask-open", type=int, default=0)
    ap.add_argument("--inpaint-mask-dilate-extra", type=int, default=4)
    ap.add_argument("--inpaint-mask-close-extra", type=int, default=2)
    ap.add_argument("--min-mask-area", type=int, default=80)
    ap.add_argument(
        "--prompt-filter",
        type=str,
        default="class",
        choices=["class", "motion"],
        help="Use class-only prompt boxes or keep only locally moving boxes during prompt discovery.",
    )
    ap.add_argument("--motion-prompt-topk", type=int, default=1)
    ap.add_argument("--motion-score-thresh", type=float, default=1.0)
    ap.add_argument("--motion-box-padding", type=float, default=0.05)
    ap.add_argument("--max-objs-per-prompt-frame", type=int, default=2)
    ap.add_argument("--min-box-area-ratio", type=float, default=0.0006)
    ap.add_argument("--max-box-area-ratio", type=float, default=0.75)
    ap.add_argument("--nms-iou", type=float, default=0.6)
    ap.add_argument("--yolo-fallback", action="store_true")
    ap.add_argument("--fallback-min-area", type=int, default=80)
    ap.add_argument("--fallback-union", action="store_true")
    ap.add_argument(
        "--force-yolo-fallback-classes",
        type=str,
        default="",
        help="Classes that should always receive per-frame YOLO-seg fallback when SAM3 propagation is missing or too weak.",
    )
    ap.add_argument(
        "--force-yolo-box-fallback-classes",
        type=str,
        default="",
        help="Classes that should receive per-frame YOLO-box fallback when both SAM3 and YOLO-seg are weak.",
    )
    ap.add_argument("--force-yolo-fallback-min-propagated", type=int, default=1)
    ap.add_argument("--force-yolo-box-padding", type=float, default=0.2)
    ap.add_argument("--redetect-interval", type=int, default=12)
    ap.add_argument("--min-reprompt-gap", type=int, default=6)
    ap.add_argument("--yolo-constraint-padding", type=float, default=0.15)
    ap.add_argument("--apply-yolo-mask-constraint", action="store_true")
    ap.add_argument("--anomaly-area-ratio-min", type=float, default=0.35)
    ap.add_argument("--anomaly-area-ratio-max", type=float, default=2.8)
    ap.add_argument("--anomaly-center-jump-factor", type=float, default=1.6)
    ap.add_argument("--text-guidance-match-ratio-min", type=float, default=0.60)
    ap.add_argument("--text-guidance-mean-iou-min", type=float, default=0.35)
    ap.add_argument("--text-guidance-small-area-max", type=float, default=0.01)
    ap.add_argument("--text-guidance-slenderness-max", type=float, default=4.5)
    ap.add_argument("--sam3-repo", type=str, default="")
    ap.add_argument("--checkpoint-path", type=str, default="")
    ap.add_argument("--hq-sam-checkpoint", type=str, default="")
    ap.add_argument("--hq-sam-model-type", type=str, default="vit_tiny")
    ap.add_argument("--hq-sam-device", type=str, default="")
    ap.add_argument("--hq-sam-keyframe-refine", action="store_true")
    ap.add_argument("--hq-sam-max-keyframes-per-class", type=int, default=3)
    ap.add_argument(
        "--hq-sam-refine-text-prompts",
        action="store_true",
        help="Also refine text-prompt late-object keyframes with HQ-SAM. Disabled by default to avoid over-correcting thin auxiliary objects.",
    )
    ap.add_argument("--hq-sam-min-overlap-iou", type=float, default=0.8)
    ap.add_argument("--hq-sam-min-area-ratio", type=float, default=0.6)
    ap.add_argument("--hq-sam-max-area-ratio", type=float, default=1.5)
    ap.add_argument(
        "--hq-sam-boundary-band-ksize",
        type=int,
        default=9,
        help="Only replace pixels in a narrow boundary band when applying HQ-SAM refinement.",
    )
    ap.add_argument("--offload-video-to-cpu", action="store_true")
    ap.add_argument("--offload-state-to-cpu", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--async-loading-frames", action="store_true")
    ap.add_argument(
        "--guidance-mode",
        type=str,
        default="yolo_box_redetect",
        choices=["yolo_box_redetect", "legacy_text"],
        help="Use the current YOLO-box+redetect pipeline or the older text-prompt-only SAM3 baseline.",
    )
    ap.add_argument(
        "--aux-target-classes",
        type=str,
        default="",
        help="Optional classes treated as late auxiliary objects when propagation-direction is both; they stay forward-only.",
    )
    ap.add_argument(
        "--text-prompt-classes",
        type=str,
        default="",
        help="Optional manual overrides that force legacy multi-keyframe text prompts.",
    )
    ap.add_argument(
        "--small-object-box-classes",
        type=str,
        default="sports ball",
        help="Optional classes that should use YOLO box prompts instead of text prompts when discovered as late objects.",
    )
    ap.add_argument(
        "--small-object-box-padding",
        type=float,
        default=1.0,
        help="Extra relative padding applied to late small-object YOLO boxes before SAM3 prompting.",
    )
    ap.add_argument(
        "--small-object-track-window",
        type=int,
        default=12,
        help="Forward propagation window for late small objects. Use <=0 to disable the limit.",
    )
    ap.add_argument(
        "--text-prompt-max-per-class",
        type=int,
        default=2,
        help="Maximum number of discovered keyframes per class for text-prompt guidance.",
    )
    ap.add_argument(
        "--propagation-direction",
        type=str,
        default="both",
        choices=["both", "forward", "backward"],
    )
    ap.add_argument(
        "--max-frame-num-to-track",
        type=int,
        default=None,
        help="Optional cap on the number of frames propagated per prompt run, useful for smoke tests",
    )
    ap.add_argument(
        "--supervised-init-mask-dir",
        type=str,
        default="",
        help="Optional GT mask directory. When set, use the selected GT mask frame as the initial supervised prompt instead of YOLO discovery.",
    )
    ap.add_argument("--supervised-init-frame-idx", type=int, default=0)
    ap.add_argument("--supervised-min-component-area", type=int, default=16)
    ap.add_argument(
        "--supervised-prompt-mode",
        type=str,
        default="union_mask",
        choices=["union_mask", "component_masks"],
        help="Use a single union GT mask prompt or split the first-frame GT into connected-component mask prompts.",
    )
    ap.add_argument(
        "--supervised-reprompt-interval",
        type=int,
        default=0,
        help="Every N propagated frames, re-add the current predicted mask to SAM3 memory. Disabled when <=0.",
    )
    ap.add_argument(
        "--supervised-reprompt-min-mask-area",
        type=int,
        default=32,
        help="Skip self re-prompt when the predicted mask area on the refresh frame is smaller than this threshold.",
    )
    ap.add_argument("--dry-run-init", action="store_true")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    mask_dir = os.path.abspath(args.mask_dir)
    ensure_dir(mask_dir)
    inpaint_mask_dir = str(Path(mask_dir).parent / "masks_inpaint")
    ensure_dir(inpaint_mask_dir)

    sam3_repo = os.path.abspath(args.sam3_repo or str(project_root.parent / "sam3_repo"))
    checkpoint_path = os.path.abspath(args.checkpoint_path or str(project_root.parent / "sam3" / "sam3.pt"))
    hq_sam_checkpoint = os.path.abspath(
        args.hq_sam_checkpoint or str(project_root / "third_party" / "hq_sam" / "sam_hq_vit_tiny.pth")
    )
    hq_sam_device = args.hq_sam_device.strip() or args.device

    yolo_model_path = args.yolo_model
    if not os.path.isabs(yolo_model_path) and not os.path.isfile(yolo_model_path):
        yolo_model_path = str((project_root / yolo_model_path).resolve())
    yolo_model_path = os.path.abspath(yolo_model_path)

    if not os.path.isdir(sam3_repo):
        raise RuntimeError(f"SAM3 repo not found: {sam3_repo}")
    if not os.path.isfile(checkpoint_path):
        raise RuntimeError(f"SAM3 checkpoint not found: {checkpoint_path}")

    predictor, used_shims = load_sam3_predictor(
        sam3_repo=sam3_repo,
        checkpoint_path=checkpoint_path,
        compile_model=args.compile,
        async_loading_frames=args.async_loading_frames,
    )
    if args.dry_run_init:
        print(
            "SAM3 predictor initialized successfully from "
            f"repo={sam3_repo} checkpoint={checkpoint_path} shims={used_shims}"
        )
        return

    supervised_init = bool(args.supervised_init_mask_dir.strip())
    yolo_model = None
    if supervised_init:
        target_classes = []
    else:
        if not os.path.isfile(yolo_model_path):
            raise RuntimeError(f"YOLO model not found: {yolo_model_path}")

        from ultralytics import YOLO

        os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(os.getcwd(), "Ultralytics"))
        yolo_model = YOLO(yolo_model_path)
        target_classes = parse_target_classes(args.target_classes)
    aux_target_classes = set(parse_target_classes(args.aux_target_classes)) if args.aux_target_classes.strip() else set()
    manual_text_prompt_classes = set(parse_target_classes(args.text_prompt_classes)) if args.text_prompt_classes.strip() else set()
    manual_text_prompt_classes &= set(target_classes)
    small_object_box_classes = set(parse_target_classes(args.small_object_box_classes)) if args.small_object_box_classes.strip() else set()
    small_object_box_classes &= set(target_classes)
    force_yolo_fallback_classes = set(parse_target_classes(args.force_yolo_fallback_classes)) if args.force_yolo_fallback_classes.strip() else set()
    force_yolo_fallback_classes &= set(target_classes)
    force_yolo_box_fallback_classes = set(parse_target_classes(args.force_yolo_box_fallback_classes)) if args.force_yolo_box_fallback_classes.strip() else set()
    force_yolo_box_fallback_classes &= set(target_classes)

    n_frames, h, w = get_video_info(args.input_video)
    merged_masks: Dict[int, np.ndarray] = {i: np.zeros((h, w), dtype=np.uint8) for i in range(n_frames)}
    prompt_runs_meta: List[Dict[str, object]] = []
    redetect_events: List[Dict[str, object]] = []
    hq_sam_refine_meta: List[Dict[str, object]] = []

    run_count = 0
    initial_prompts: Dict[str, object]
    if supervised_init:
        gt_mask_dir_abs = os.path.abspath(args.supervised_init_mask_dir)
        if args.supervised_prompt_mode == "component_masks":
            supervised_prompts = load_supervised_mask_prompts(
                gt_mask_dir=gt_mask_dir_abs,
                frame_idx=int(args.supervised_init_frame_idx),
                min_component_area=int(args.supervised_min_component_area),
                image_w=w,
                image_h=h,
            )
            initial_prompts = {
                "supervised_gt_mask_dir": gt_mask_dir_abs,
                "supervised_init_frame_idx": int(args.supervised_init_frame_idx),
                "supervised_prompt_mode": args.supervised_prompt_mode,
                "supervised_mask_prompts": [
                    {
                        "frame_idx": int(item["frame_idx"]),
                        "component_id": int(item["component_id"]),
                        "bbox_xyxy": [float(v) for v in item["bbox_xyxy"]],
                        "bbox_xywh": [float(v) for v in item["bbox_xywh"]],
                        "area": int(item["area"]),
                    }
                    for item in supervised_prompts
                ],
            }
        else:
            supervised_prompts = [
                {
                    **load_supervised_union_mask_prompt(
                        gt_mask_dir=gt_mask_dir_abs,
                        frame_idx=int(args.supervised_init_frame_idx),
                        min_component_area=int(args.supervised_min_component_area),
                        image_w=w,
                        image_h=h,
                    ),
                    "component_id": 0,
                }
            ]
            initial_prompts = {
                "supervised_gt_mask_dir": gt_mask_dir_abs,
                "supervised_init_frame_idx": int(args.supervised_init_frame_idx),
                "supervised_prompt_mode": args.supervised_prompt_mode,
                "supervised_mask_prompts": [
                    {
                        "frame_idx": int(supervised_prompts[0]["frame_idx"]),
                        "component_id": 0,
                        "bbox_xyxy": [float(v) for v in supervised_prompts[0]["bbox_xyxy"]],
                        "bbox_xywh": [float(v) for v in supervised_prompts[0]["bbox_xywh"]],
                        "area": int(supervised_prompts[0]["area"]),
                        "component_count": int(supervised_prompts[0].get("component_count", 1)),
                    }
                ],
            }
        for prompt_info in supervised_prompts:
            frame_idx = int(prompt_info["frame_idx"])
            component_id = int(prompt_info["component_id"])
            print(
                f"[SAM3] supervised init component={component_id} "
                f"frame={frame_idx} direction='{args.propagation_direction}'"
            )
            if int(args.supervised_reprompt_interval) > 0:
                outputs, reprompt_events = propagate_with_mask_reprompt(
                    predictor=predictor,
                    input_video=args.input_video,
                    prompt_frame=frame_idx,
                    component_mask=prompt_info["mask"],
                    output_prob_thresh=args.output_prob_thresh,
                    offload_video_to_cpu=args.offload_video_to_cpu,
                    offload_state_to_cpu=args.offload_state_to_cpu,
                    max_frame_num_to_track=args.max_frame_num_to_track,
                    propagation_direction=args.propagation_direction,
                    reprompt_interval=args.supervised_reprompt_interval,
                    reprompt_min_mask_area=args.supervised_reprompt_min_mask_area,
                    obj_id=component_id,
                )
                redetect_events.extend(
                    [
                        {
                            **event,
                            "class_name": f"supervised_component_{component_id}",
                        }
                        for event in reprompt_events
                    ]
                )
            else:
                outputs = propagate_with_mask_prompt_only(
                    predictor=predictor,
                    input_video=args.input_video,
                    prompt_frame=frame_idx,
                    component_mask=prompt_info["mask"],
                    output_prob_thresh=args.output_prob_thresh,
                    offload_video_to_cpu=args.offload_video_to_cpu,
                    offload_state_to_cpu=args.offload_state_to_cpu,
                    max_frame_num_to_track=args.max_frame_num_to_track,
                    propagation_direction=args.propagation_direction,
                    obj_id=component_id,
                )
            for out_idx, mask in outputs.items():
                merged_masks[out_idx] = np.maximum(merged_masks[out_idx], mask)
            prompt_runs_meta.append(
                {
                    "prompt_text": f"supervised_component_{component_id}",
                    "prompt_frame": frame_idx,
                    "guidance_submode": "supervised_gt_mask",
                    "initial_bbox_xywh": [float(v) for v in prompt_info["bbox_xywh"]],
                    "component_area": int(prompt_info["area"]),
                    "component_count": int(prompt_info.get("component_count", 1)),
                    "propagated_frame_count": len(outputs),
                    "propagation_direction": args.propagation_direction,
                    "max_frame_num_to_track": args.max_frame_num_to_track,
                    "self_reprompt_interval": int(args.supervised_reprompt_interval),
                }
            )
            run_count += 1
    elif args.guidance_mode == "legacy_text":
        prompt_frames = discover_legacy_prompt_frames(
            input_video=args.input_video,
            yolo_model=yolo_model,
            target_classes=target_classes,
            conf=args.conf,
            device=args.device,
            prompt_interval=args.prompt_interval,
            max_prompt_frames_per_class=args.max_prompt_frames_per_class,
        )
        initial_prompts = {"prompt_frames": prompt_frames}
        for class_name, frame_indices in prompt_frames.items():
            for frame_idx in frame_indices:
                print(f"[SAM3] legacy init class='{class_name}' frame={frame_idx}")
                outputs = propagate_text_prompt_only(
                    predictor=predictor,
                    input_video=args.input_video,
                    class_name=class_name,
                    prompt_frame=int(frame_idx),
                    output_prob_thresh=args.output_prob_thresh,
                    offload_video_to_cpu=args.offload_video_to_cpu,
                    offload_state_to_cpu=args.offload_state_to_cpu,
                    max_frame_num_to_track=args.max_frame_num_to_track,
                    propagation_direction=args.propagation_direction,
                )
                for out_idx, mask in outputs.items():
                    merged_masks[out_idx] = np.maximum(merged_masks[out_idx], mask)
                prompt_runs_meta.append(
                    {
                        "prompt_text": class_name,
                        "prompt_frame": int(frame_idx),
                        "propagated_frame_count": len(outputs),
                        "propagation_direction": args.propagation_direction,
                        "max_frame_num_to_track": args.max_frame_num_to_track,
                    }
                )
                run_count += 1
        if run_count == 0:
            print("[SAM3] Legacy text mode found no prompt frames. Writing empty masks.")
    else:
        prompt_infos_by_class = discover_initial_prompts(
            input_video=args.input_video,
            yolo_model=yolo_model,
            target_classes=target_classes,
            conf=args.conf,
            device=args.device,
            prompt_interval=args.prompt_interval,
            max_prompt_frames_per_class=args.max_prompt_frames_per_class,
            init_search_sampled_frames=args.init_search_sampled_frames,
            image_w=w,
            image_h=h,
            temporal_window_radius=args.temporal_window_radius,
            prompt_filter=args.prompt_filter,
            motion_prompt_topk=args.motion_prompt_topk,
            motion_score_thresh=args.motion_score_thresh,
            motion_box_padding=args.motion_box_padding,
            max_objs_per_frame=args.max_objs_per_prompt_frame,
            min_box_area_ratio=args.min_box_area_ratio,
            max_box_area_ratio=args.max_box_area_ratio,
            nms_iou=args.nms_iou,
        )
        auto_text_prompt_classes = {
            class_name
            for class_name, prompt_infos in prompt_infos_by_class.items()
            if prompt_infos
            and should_use_text_guidance(
                prompt_infos[0],
                temporal_match_ratio_min=args.text_guidance_match_ratio_min,
                temporal_mean_iou_min=args.text_guidance_mean_iou_min,
                small_area_ratio_max=args.text_guidance_small_area_max,
                slenderness_max=args.text_guidance_slenderness_max,
            )
        }
        text_prompt_classes = manual_text_prompt_classes | auto_text_prompt_classes
        text_prompt_frames_by_class = discover_text_prompt_frames_scored(
            input_video=args.input_video,
            yolo_model=yolo_model,
            target_classes=[name for name in target_classes if name in text_prompt_classes],
            conf=args.conf,
            device=args.device,
            prompt_interval=(args.text_prompt_interval if int(args.text_prompt_interval) > 0 else args.prompt_interval),
            max_prompt_frames_per_class=args.text_prompt_max_per_class,
            image_w=w,
            image_h=h,
            temporal_window_radius=args.temporal_window_radius,
        )
        small_object_box_frames_by_class = {
            class_name: frame_indices
            for class_name, frame_indices in text_prompt_frames_by_class.items()
            if class_name in small_object_box_classes
        }
        small_object_box_prompts_by_class = discover_box_prompt_infos_for_frames(
            input_video=args.input_video,
            yolo_model=yolo_model,
            class_to_frames=small_object_box_frames_by_class,
            conf=args.conf,
            device=args.device,
            image_w=w,
            image_h=h,
            temporal_window_radius=args.temporal_window_radius,
        )
        text_prompt_frames_by_class = {
            class_name: frame_indices
            for class_name, frame_indices in text_prompt_frames_by_class.items()
            if class_name not in small_object_box_classes
        }
        prompt_infos_by_class = {
            class_name: prompt_infos
            for class_name, prompt_infos in prompt_infos_by_class.items()
            if class_name not in text_prompt_classes
        }
        for class_name, prompt_infos in small_object_box_prompts_by_class.items():
            if prompt_infos:
                prompt_infos_by_class.setdefault(class_name, []).extend(prompt_infos)
        initial_prompts = {
            "box_guided": prompt_infos_by_class,
            "text_prompt_frames": text_prompt_frames_by_class,
            "auto_text_prompt_classes": sorted(auto_text_prompt_classes),
            "manual_text_prompt_classes": sorted(manual_text_prompt_classes),
            "small_object_box_classes": sorted(small_object_box_classes),
            "small_object_box_prompts": small_object_box_prompts_by_class,
        }

        for class_name, frame_indices in text_prompt_frames_by_class.items():
            for frame_idx in frame_indices:
                print(
                    f"[SAM3] text-guided init class='{class_name}' frame={frame_idx} "
                    f"direction='{args.propagation_direction}'"
                )
                outputs = propagate_text_prompt_only(
                    predictor=predictor,
                    input_video=args.input_video,
                    class_name=class_name,
                    prompt_frame=int(frame_idx),
                    output_prob_thresh=args.output_prob_thresh,
                    offload_video_to_cpu=args.offload_video_to_cpu,
                    offload_state_to_cpu=args.offload_state_to_cpu,
                    max_frame_num_to_track=args.max_frame_num_to_track,
                    propagation_direction=args.propagation_direction,
                )
                for out_idx, mask in outputs.items():
                    merged_masks[out_idx] = np.maximum(merged_masks[out_idx], mask)
                prompt_runs_meta.append(
                    {
                        "prompt_text": class_name,
                        "prompt_frame": int(frame_idx),
                        "guidance_submode": "text_prompt",
                        "propagated_frame_count": len(outputs),
                        "propagation_direction": args.propagation_direction,
                        "max_frame_num_to_track": args.max_frame_num_to_track,
                    }
                )
                run_count += 1

        for class_name, prompt_infos in prompt_infos_by_class.items():
            for prompt_info in prompt_infos:
                frame_idx = int(prompt_info["frame_idx"])
                effective_initial_direction = args.propagation_direction
                effective_prompt_info = prompt_info
                effective_max_track = args.max_frame_num_to_track
                effective_redetect_interval = args.redetect_interval
                enable_redetect = True
                if effective_initial_direction == "both" and class_name in aux_target_classes:
                    effective_initial_direction = "forward"
                if class_name in small_object_box_classes:
                    effective_initial_direction = "forward"
                    enable_redetect = False
                    effective_redetect_interval = 0
                    small_prompt = dict(prompt_info)
                    expanded_xyxy = expand_box_xyxy(
                        prompt_info["bbox_xyxy"],
                        image_w=w,
                        image_h=h,
                        padding_ratio=args.small_object_box_padding,
                    )
                    small_prompt["bbox_xyxy"] = expanded_xyxy
                    small_prompt["bbox_xywh"] = bbox_xyxy_to_xywh(expanded_xyxy)
                    effective_prompt_info = small_prompt
                    if int(args.small_object_track_window) > 0:
                        effective_max_track = int(args.small_object_track_window)
                print(
                    f"[SAM3] init class='{class_name}' frame={frame_idx} "
                    f"direction='{effective_initial_direction}'"
                )
                outputs, events = propagate_with_yolo_redetect(
                    predictor=predictor,
                    yolo_model=yolo_model,
                    input_video=args.input_video,
                    class_name=class_name,
                    initial_prompt=effective_prompt_info,
                    output_prob_thresh=args.output_prob_thresh,
                    offload_video_to_cpu=args.offload_video_to_cpu,
                    offload_state_to_cpu=args.offload_state_to_cpu,
                    max_frame_num_to_track=effective_max_track,
                    conf=args.conf,
                    device=args.device,
                    redetect_interval=effective_redetect_interval,
                    min_reprompt_gap=args.min_reprompt_gap,
                    yolo_constraint_padding=args.yolo_constraint_padding,
                    apply_yolo_mask_constraint=args.apply_yolo_mask_constraint,
                    anomaly_area_ratio_min=args.anomaly_area_ratio_min,
                    anomaly_area_ratio_max=args.anomaly_area_ratio_max,
                    anomaly_center_jump_factor=args.anomaly_center_jump_factor,
                    min_mask_area=args.min_mask_area,
                    image_w=w,
                    image_h=h,
                    initial_propagation_direction=effective_initial_direction,
                    temporal_window_radius=args.temporal_window_radius,
                    enable_redetect=enable_redetect,
                )
                for out_idx, mask in outputs.items():
                    merged_masks[out_idx] = np.maximum(merged_masks[out_idx], mask)
                redetect_events.extend(events)
                prompt_runs_meta.append(
                    {
                        "prompt_text": class_name,
                        "prompt_frame": int(frame_idx),
                        "guidance_submode": (
                            "yolo_box_small_object"
                            if class_name in small_object_box_classes
                            else "yolo_box_redetect"
                        ),
                        "initial_bbox_xywh": [float(v) for v in effective_prompt_info["bbox_xywh"]],
                        "temporal_match_ratio": float(effective_prompt_info.get("temporal_match_ratio", 0.0)),
                        "temporal_mean_iou": float(effective_prompt_info.get("temporal_mean_iou", 0.0)),
                        "area_ratio": float(effective_prompt_info.get("area_ratio", 0.0)),
                        "slenderness": float(effective_prompt_info.get("slenderness", 0.0)),
                        "propagated_frame_count": len(outputs),
                        "propagation_direction": (
                            f"{effective_initial_direction}_then_forward_restarts"
                            if events and enable_redetect
                            else effective_initial_direction
                        ),
                        "max_frame_num_to_track": effective_max_track,
                        "redetect_event_count": len(events),
                        "small_object_box_padding": (
                            args.small_object_box_padding if class_name in small_object_box_classes else 0.0
                        ),
                    }
                )
                run_count += 1

        if run_count == 0:
            print("[SAM3] No initial prompt boxes discovered from YOLO11. Writing empty masks.")

    if args.hq_sam_keyframe_refine:
        if os.path.isfile(hq_sam_checkpoint):
            per_class_frames: Dict[str, List[Dict[str, object]]] = {}
            for prompt_run in prompt_runs_meta:
                if int(prompt_run.get("propagated_frame_count", 0)) <= 0:
                    continue
                class_name = str(prompt_run["prompt_text"])
                frame_idx = int(prompt_run["prompt_frame"])
                guidance_submode = str(prompt_run.get("guidance_submode", ""))
                if guidance_submode == "text_prompt" and not args.hq_sam_refine_text_prompts:
                    continue
                reason = (
                    "late_object_prompt"
                    if frame_idx > 0 and guidance_submode == "text_prompt"
                    else "prompt_keyframe"
                )
                per_class_frames.setdefault(class_name, []).append(
                    {"frame_idx": frame_idx, "class_name": class_name, "reason": reason}
                )
            for event in redetect_events:
                class_name = str(event["class_name"])
                per_class_frames.setdefault(class_name, []).append(
                    {"frame_idx": int(event["frame_idx"]), "class_name": class_name, "reason": str(event["event"])}
                )
            refine_specs: List[Dict[str, object]] = []
            per_class_limit = max(1, int(args.hq_sam_max_keyframes_per_class))
            for class_name, specs in per_class_frames.items():
                unique_by_frame: Dict[int, Dict[str, object]] = {}
                for spec in specs:
                    unique_by_frame.setdefault(int(spec["frame_idx"]), spec)
                ordered = [unique_by_frame[idx] for idx in sorted(unique_by_frame.keys())[:per_class_limit]]
                refine_specs.extend(ordered)
            if refine_specs:
                print(f"[HQ-SAM] refining {len(refine_specs)} keyframe/class pairs")
                hq_predictor = load_hq_sam_predictor(
                    checkpoint_path=hq_sam_checkpoint,
                    model_type=args.hq_sam_model_type,
                    device=hq_sam_device,
                )
                hq_sam_refine_meta = refine_keyframes_with_hq_sam(
                    predictor=hq_predictor,
                    input_video=args.input_video,
                    yolo_model=yolo_model,
                    merged_masks=merged_masks,
                    refine_specs=refine_specs,
                    conf=args.conf,
                    device=args.device,
                    temporal_window_radius=args.temporal_window_radius,
                    image_w=w,
                    image_h=h,
                    min_overlap_iou=args.hq_sam_min_overlap_iou,
                    min_area_ratio=args.hq_sam_min_area_ratio,
                    max_area_ratio=args.hq_sam_max_area_ratio,
                    boundary_band_ksize=args.hq_sam_boundary_band_ksize,
                )
        else:
            print(f"[HQ-SAM] checkpoint not found, skip refinement: {hq_sam_checkpoint}")

    failed_force_yolo_classes: List[str] = []
    if force_yolo_fallback_classes or force_yolo_box_fallback_classes:
        propagated_by_class: Dict[str, int] = {}
        for prompt_run in prompt_runs_meta:
            class_name = str(prompt_run.get("prompt_text", ""))
            propagated = int(prompt_run.get("propagated_frame_count", 0))
            propagated_by_class[class_name] = max(propagated_by_class.get(class_name, 0), propagated)
        failed_force_yolo_classes = sorted(
            class_name
            for class_name in (force_yolo_fallback_classes | force_yolo_box_fallback_classes)
            if int(propagated_by_class.get(class_name, 0)) < int(args.force_yolo_fallback_min_propagated)
        )

    yolo_seg_masks: Dict[int, np.ndarray] = {}
    forced_yolo_seg_masks: Dict[int, np.ndarray] = {}
    forced_yolo_box_masks: Dict[int, np.ndarray] = {}
    replaced = 0
    unioned = 0
    if args.yolo_fallback:
        yolo_seg_model_path = args.yolo_seg_model
        if not os.path.isabs(yolo_seg_model_path) and not os.path.isfile(yolo_seg_model_path):
            yolo_seg_model_path = str((project_root / yolo_seg_model_path).resolve())
        yolo_seg_model_path = os.path.abspath(yolo_seg_model_path)
        yolo_seg_masks = generate_yolo_seg_masks(
            input_video=args.input_video,
            total=n_frames,
            h=h,
            w=w,
            yolo_seg_model=yolo_seg_model_path,
            conf=args.conf,
            device=args.device,
            target_classes=target_classes,
        )
    if failed_force_yolo_classes:
        yolo_seg_model_path = args.yolo_seg_model
        if not os.path.isabs(yolo_seg_model_path) and not os.path.isfile(yolo_seg_model_path):
            yolo_seg_model_path = str((project_root / yolo_seg_model_path).resolve())
        yolo_seg_model_path = os.path.abspath(yolo_seg_model_path)
        forced_yolo_seg_masks = generate_yolo_seg_masks(
            input_video=args.input_video,
            total=n_frames,
            h=h,
            w=w,
            yolo_seg_model=yolo_seg_model_path,
            conf=args.conf,
            device=args.device,
            target_classes=[c for c in failed_force_yolo_classes if c in force_yolo_fallback_classes],
        )
        if any(c in force_yolo_box_fallback_classes for c in failed_force_yolo_classes):
            forced_yolo_box_masks = generate_yolo_box_masks(
                input_video=args.input_video,
                total=n_frames,
                h=h,
                w=w,
                yolo_model=yolo_model,
                conf=args.conf,
                device=args.device,
                target_classes=[c for c in failed_force_yolo_classes if c in force_yolo_box_fallback_classes],
                padding_ratio=args.force_yolo_box_padding,
                min_box_area_ratio=args.min_box_area_ratio,
                max_box_area_ratio=args.max_box_area_ratio,
                nms_iou=args.nms_iou,
                max_objs_per_frame=args.max_objs_per_prompt_frame,
            )

    for frame_idx in range(n_frames):
        merged_mask = merged_masks[frame_idx]
        if args.yolo_fallback:
            sam_area = int(np.count_nonzero(merged_mask))
            yolo_mask = yolo_seg_masks.get(frame_idx, np.zeros((h, w), dtype=np.uint8))
            yolo_area = int(np.count_nonzero(yolo_mask))
            if args.fallback_union:
                if yolo_area > 0:
                    merged_mask = np.where((merged_mask > 0) | (yolo_mask > 0), 255, 0).astype(np.uint8)
                    unioned += 1
            elif sam_area < args.fallback_min_area and yolo_area > 0:
                merged_mask = yolo_mask
                replaced += 1
        forced_seg_mask = forced_yolo_seg_masks.get(frame_idx)
        if forced_seg_mask is not None and int(np.count_nonzero(forced_seg_mask)) > 0:
            merged_mask = np.where((merged_mask > 0) | (forced_seg_mask > 0), 255, 0).astype(np.uint8)
        forced_box_mask = forced_yolo_box_masks.get(frame_idx)
        if forced_box_mask is not None and int(np.count_nonzero(forced_box_mask)) > 0:
            if int(np.count_nonzero(merged_mask)) < int(args.fallback_min_area):
                merged_mask = np.where((merged_mask > 0) | (forced_box_mask > 0), 255, 0).astype(np.uint8)
        eval_mask = postprocess_mask(
            merged_mask,
            dilate_ksize=args.mask_dilate,
            close_ksize=args.mask_close,
            open_ksize=args.mask_open,
            min_area=args.min_mask_area,
        )
        inpaint_mask = postprocess_mask(
            merged_mask,
            dilate_ksize=max(1, int(args.mask_dilate) + int(args.inpaint_mask_dilate_extra)),
            close_ksize=max(1, int(args.mask_close) + int(args.inpaint_mask_close_extra)),
            open_ksize=args.mask_open,
            min_area=args.min_mask_area,
        )
        write_mask(os.path.join(mask_dir, f"mask_{frame_idx:05d}.png"), eval_mask)
        write_mask(os.path.join(inpaint_mask_dir, f"mask_{frame_idx:05d}.png"), inpaint_mask)

    meta = {
        "input_video": os.path.abspath(args.input_video),
        "mask_dir": mask_dir,
        "inpaint_mask_dir": inpaint_mask_dir,
        "sam3_repo": sam3_repo,
        "checkpoint_path": checkpoint_path,
        "hq_sam_checkpoint": hq_sam_checkpoint,
        "hq_sam_device": hq_sam_device,
        "yolo_model_path": yolo_model_path,
        "device": args.device,
        "supervised_init_mask_dir": os.path.abspath(args.supervised_init_mask_dir) if supervised_init else "",
        "supervised_init_frame_idx": int(args.supervised_init_frame_idx),
        "supervised_prompt_mode": args.supervised_prompt_mode,
        "supervised_reprompt_interval": int(args.supervised_reprompt_interval),
        "supervised_reprompt_min_mask_area": int(args.supervised_reprompt_min_mask_area),
        "runtime_shims": used_shims,
        "guidance_mode": args.guidance_mode,
        "target_classes": target_classes,
        "initial_prompts": initial_prompts,
        "prompt_runs": prompt_runs_meta,
        "redetect_events": redetect_events,
        "hq_sam_keyframe_refine": bool(args.hq_sam_keyframe_refine),
        "hq_sam_refine_text_prompts": bool(args.hq_sam_refine_text_prompts),
        "hq_sam_refine_events": hq_sam_refine_meta,
        "frame_count": n_frames,
        "image_height": h,
        "image_width": w,
        "output_prob_thresh": args.output_prob_thresh,
        "yolo_guidance": {
            "yolo_model": Path(yolo_model_path).name,
            "redetect_interval": args.redetect_interval,
            "min_reprompt_gap": args.min_reprompt_gap,
            "apply_yolo_mask_constraint": args.apply_yolo_mask_constraint,
            "yolo_constraint_padding": args.yolo_constraint_padding,
            "anomaly_area_ratio_min": args.anomaly_area_ratio_min,
            "anomaly_area_ratio_max": args.anomaly_area_ratio_max,
            "anomaly_center_jump_factor": args.anomaly_center_jump_factor,
            "prompt_filter": args.prompt_filter,
            "motion_prompt_topk": args.motion_prompt_topk,
            "motion_score_thresh": args.motion_score_thresh,
            "motion_box_padding": args.motion_box_padding,
            "max_objs_per_prompt_frame": args.max_objs_per_prompt_frame,
            "min_box_area_ratio": args.min_box_area_ratio,
            "max_box_area_ratio": args.max_box_area_ratio,
            "nms_iou": args.nms_iou,
            "yolo_fallback": bool(args.yolo_fallback),
            "fallback_min_area": args.fallback_min_area,
            "fallback_union": bool(args.fallback_union),
            "fallback_replaced": replaced,
            "fallback_unioned": unioned,
            "force_yolo_fallback_classes": sorted(force_yolo_fallback_classes),
            "force_yolo_box_fallback_classes": sorted(force_yolo_box_fallback_classes),
            "force_yolo_fallback_min_propagated": args.force_yolo_fallback_min_propagated,
            "force_yolo_box_padding": args.force_yolo_box_padding,
            "failed_force_yolo_classes": failed_force_yolo_classes,
        },
        "postprocess": {
            "mask_dilate": args.mask_dilate,
            "mask_close": args.mask_close,
            "mask_open": args.mask_open,
            "inpaint_mask_dilate_extra": args.inpaint_mask_dilate_extra,
            "inpaint_mask_close_extra": args.inpaint_mask_close_extra,
            "min_mask_area": args.min_mask_area,
            "hq_sam_min_overlap_iou": args.hq_sam_min_overlap_iou,
            "hq_sam_min_area_ratio": args.hq_sam_min_area_ratio,
            "hq_sam_max_area_ratio": args.hq_sam_max_area_ratio,
            "hq_sam_boundary_band_ksize": args.hq_sam_boundary_band_ksize,
        },
    }
    meta_path = str(Path(mask_dir).parent / "sam3_run_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(
        f"SAM3 masks saved: frames={n_frames}, prompt_runs={run_count}, "
        f"repo={sam3_repo}, checkpoint={checkpoint_path}, mask_dir={mask_dir}, shims={used_shims}"
    )


if __name__ == "__main__":
    main()
