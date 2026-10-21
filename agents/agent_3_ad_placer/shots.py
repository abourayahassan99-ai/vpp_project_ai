"""
Shot-cut detection (Agent 3).

Edited videos jump between camera positions (a hard cut). A planar tracker cannot follow a surface
across a cut, so the tracker is told where cuts are, drops its state there, and a new anchor is used
for the next shot (see video_placer.place_ad_in_video).

Detector: Bhattacharyya distance between consecutive HSV histograms of tiny frames. A cut is a spike
that is large in absolute terms AND large relative to the recent frame-to-frame variation (fast handheld
motion changes the histogram gradually and does not trigger it).

It can also fire when only the picture *on a TV screen* changes. That is harmless here: cuts are only used
to bound the frame range of one anchor, so a false boundary costs at most one extra anchor.
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np


def detect_cuts(video_path: str, abs_threshold: float = 0.28, rel_factor: float = 4.0, min_gap: int = 8) -> List[int]:
    """Frame indices at which a new shot starts (the first frame after the cut)."""
    cap = cv2.VideoCapture(str(video_path))
    dists: List[float] = []
    prev = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256])
        h = cv2.normalize(h, h, 1.0, 0.0, cv2.NORM_L1).astype(np.float32)
        dists.append(0.0 if prev is None else float(cv2.compareHist(prev, h, cv2.HISTCMP_BHATTACHARYYA)))
        prev = h
    cap.release()
    d = np.asarray(dists)
    candidates = []
    cuts: List[int] = []
    for i in range(1, len(d)):
        lo, hi = max(1, i - 15), min(len(d), i + 16)
        neigh = np.concatenate([d[lo:i], d[i + 1:hi]])
        local = float(np.median(neigh)) if len(neigh) else 0.0
        if d[i] >= abs_threshold and d[i] >= rel_factor * max(local, 0.02):
            if not candidates or i - candidates[-1] >= min_gap:
                candidates.append(i)
    cuts = candidates
    return cuts


def shot_ranges(cuts: List[int], total_frames: int) -> List[Tuple[int, int]]:
    """[(first_frame, last_frame), ...] for every shot, given cut positions."""
    edges = [0] + [c for c in cuts if 0 < c < total_frames] + [total_frames]
    return [(edges[i], edges[i + 1] - 1) for i in range(len(edges) - 1)]
