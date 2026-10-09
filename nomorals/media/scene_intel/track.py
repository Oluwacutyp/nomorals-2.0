"""Character tracking across a film: YOLOv8 (person detect) + OSNet (re-ID)
+ ByteTrack-style association.

Mined pipeline (rap-cv-task, player_reid): this exact stack is proven on
commodity hardware. OSNet is 2.2M params — phone-viable.

Models are loaded lazily and cached. Everything degrades honestly when a
model is missing: no fake tracks, ever.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class Detection:
    frame_t: float          # seconds
    bbox: tuple[float, float, float, float]  # x1,y1,x2,y2 (0-1 normalized)
    track_id: int           # local track id within this run
    embedding: list[float] = field(default_factory=list)  # OSNet vector


@dataclass
class CharacterTrack:
    char_id: str            # "char_0", "char_1", ...
    detections: list[Detection] = field(default_factory=list)
    total_screen_s: float = 0.0
    first_seen: float = 0.0
    last_seen: float = 0.0
    # Representative embedding = mean of detection embeddings
    mean_embedding: list[float] = field(default_factory=list)


class ModelUnavailable(Exception):
    pass


def _load_yolo():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ModelUnavailable(
            "ultralytics not installed — pip install ultralytics") from exc
    return YOLO("yolov8n.pt")  # ~6MB, auto-downloaded on first use


def _load_osnet():
    try:
        import torch
        import torchreid
    except ImportError as exc:
        raise ModelUnavailable(
            "torch/torchreid not installed") from exc
    model = torchreid.models.build_model(
        name="osnet_x1_0", num_classes=1000, pretrained=True)
    model.eval()
    return model, torch


def _cosine(a: list[float], b: list[float]) -> float:
    import math
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-9
    nb = math.sqrt(sum(x * x for x in b)) or 1e-9
    return dot / (na * nb)


def _iou(a: tuple[float, float, float, float],
         b: tuple[float, float, float, float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + ab - inter + 1e-9)


def track_characters(src: str, *, sample_fps: float = 2.0,
                     reid_threshold: float = 0.55,
                     workdir: str | None = None) -> list[CharacterTrack]:
    """Track distinct characters across a film.

    Detection at sample_fps (2 fps default — films don't need 30fps for
    identity). ByteTrack-style IoU association within shots; OSNet cosine
    similarity re-identifies across scene cuts.

    Returns character tracks sorted by total screen time (char_0 = most).
    Raises ModelUnavailable with an honest message when the stack is missing.
    """
    yolo = _load_yolo()
    osnet, torch = _load_osnet()

    try:
        import cv2
    except ImportError as exc:
        raise ModelUnavailable("opencv not installed — pip install opencv-python") from exc

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {src}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    stride = max(1, int(round(src_fps / sample_fps)))

    # ByteTrack-lite state
    active: dict[int, dict] = {}   # track_id -> {bbox, missed, embedding}
    next_tid = 0
    all_dets: list[Detection] = []
    frame_i = 0

    import numpy as np
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_i % stride == 0:
            t = frame_i / src_fps
            h, w = frame.shape[:2]
            results = yolo(frame, verbose=False)[0]
            dets_now: list[tuple[tuple, list[float]]] = []
            for box in results.boxes:
                cls = int(box.cls[0])
                if cls != 0:  # person class only
                    continue
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
                bbox = (x1 / w, y1 / h, x2 / w, y2 / h)
                # OSNet embedding on the crop
                crop = frame[int(y1):int(y2), int(x1):int(x2)]
                if crop.size == 0:
                    continue
                crop = cv2.resize(crop, (128, 256))
                tensor = torch.from_numpy(
                    crop[:, :, ::-1].transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
                with torch.no_grad():
                    emb = osnet(tensor).squeeze(0).tolist()
                dets_now.append((bbox, emb))

            # Associate: IoU match to active tracks, else new track
            matched_active: set[int] = set()
            for bbox, emb in dets_now:
                best_tid, best_iou = -1, 0.3  # IoU gate
                for tid, tr in active.items():
                    if tid in matched_active:
                        continue
                    iou = _iou(bbox, tr["bbox"])
                    if iou > best_iou:
                        best_tid, best_iou = tid, iou
                if best_tid >= 0:
                    tid = best_tid
                    matched_active.add(tid)
                    active[tid].update(bbox=bbox, missed=0, embedding=emb)
                else:
                    tid = next_tid
                    next_tid += 1
                    active[tid] = {"bbox": bbox, "missed": 0, "embedding": emb}
                all_dets.append(Detection(frame_t=round(t, 3), bbox=bbox,
                                          track_id=tid, embedding=emb))
            # Age out missed tracks (keep 5s of disappearance)
            for tid in list(active):
                if tid not in matched_active:
                    active[tid]["missed"] += 1
                    if active[tid]["missed"] > int(5 * sample_fps):
                        del active[tid]
        frame_i += 1
    cap.release()

    # Cross-scene re-ID: merge local tracks by OSNet cosine similarity.
    # Group detections by track, mean embedding per track, then cluster.
    by_track: dict[int, list[Detection]] = {}
    for d in all_dets:
        by_track.setdefault(d.track_id, []).append(d)

    track_embs: list[tuple[int, list[float]]] = []
    for tid, ds in by_track.items():
        if not ds or not ds[0].embedding:
            continue
        n = len(ds[0].embedding)
        mean = [sum(d.embedding[i] for d in ds) / len(ds) for i in range(n)]
        track_embs.append((tid, mean))

    # Greedy clustering by cosine similarity
    clusters: list[list[int]] = []  # list of track-id groups
    cluster_embs: list[list[float]] = []
    for tid, emb in track_embs:
        placed = False
        for ci, cemb in enumerate(cluster_embs):
            if _cosine(emb, cemb) >= reid_threshold:
                clusters[ci].append(tid)
                # update running mean
                n = len(clusters[ci])
                cluster_embs[ci] = [(a * (n - 1) + b) / n
                                    for a, b in zip(cemb, emb)]
                placed = True
                break
        if not placed:
            clusters.append([tid])
            cluster_embs.append(emb)

    characters: list[CharacterTrack] = []
    for ci, tids in enumerate(clusters):
        ds = [d for tid in tids for d in by_track[tid]]
        ds.sort(key=lambda d: d.frame_t)
        screen_s = len(ds) / sample_fps
        characters.append(CharacterTrack(
            char_id=f"char_{ci}",
            detections=ds,
            total_screen_s=round(screen_s, 1),
            first_seen=ds[0].frame_t,
            last_seen=ds[-1].frame_t,
            mean_embedding=cluster_embs[ci],
        ))
    characters.sort(key=lambda c: c.total_screen_s, reverse=True)
    # Re-label by screen time rank
    for i, c in enumerate(characters):
        c.char_id = f"char_{i}"
    return characters
