"""Frame-level video inference with OpenCV.

Samples N frames uniformly across the video, runs the classifier on each,
and aggregates results by averaging softmax probabilities.
"""
from __future__ import annotations

from collections import Counter
from typing import List

import cv2
import numpy as np

from .config import CLASS_NAMES
from .inference import Predictor, get_predictor


def sample_frames(video_path: str, num_frames: int = 8) -> List[np.ndarray]:
    """Uniformly sample `num_frames` frames from the video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise ValueError(f"Video has no readable frames: {video_path}")

    n = min(num_frames, total)
    indices = np.linspace(0, total - 1, n).astype(int)

    frames: List[np.ndarray] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()

    if not frames:
        raise ValueError("Failed to read any frames from video")
    return frames


def predict_video(
    video_path: str,
    num_frames: int = 8,
    predictor: Predictor | None = None,
) -> dict:
    """Run inference on sampled frames and aggregate.

    Returns:
        {
            "num_frames_sampled": int,
            "per_frame": [{"frame_index", "class_id", "class_name", "confidence"}, ...],
            "aggregate": {"class_id", "class_name", "confidence"},
            "vote_counts": {class_name: count, ...},
        }
    """
    predictor = predictor or get_predictor()
    frames = sample_frames(video_path, num_frames=num_frames)

    per_frame = []
    prob_sum = np.zeros(len(CLASS_NAMES), dtype=np.float64)
    votes: Counter = Counter()

    for i, frame in enumerate(frames):
        result = predictor.predict_frame(frame)
        per_frame.append({
            "frame_index": i,
            "class_id": result["class_id"],
            "class_name": result["class_name"],
            "confidence": result["confidence"],
        })
        votes[result["class_name"]] += 1
        for j, name in enumerate(CLASS_NAMES):
            prob_sum[j] += result["probs"][name]

    avg_probs = prob_sum / len(frames)
    agg_class_id = int(np.argmax(avg_probs))
    aggregate = {
        "class_id": agg_class_id,
        "class_name": CLASS_NAMES[agg_class_id],
        "confidence": float(avg_probs[agg_class_id]),
    }

    return {
        "num_frames_sampled": len(frames),
        "per_frame": per_frame,
        "aggregate": aggregate,
        "vote_counts": dict(votes),
    }
