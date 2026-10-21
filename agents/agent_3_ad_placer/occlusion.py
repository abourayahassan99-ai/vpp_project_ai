"""
Foreground / occlusion estimation for Agent 3 (plane-space background subtraction).

The anchor frame is warped into a small canonical rectangle (the "plane texture"). For every frame
the current frame is warped into the same rectangle through the tracked quad and compared with it.
Pixels that differ strongly are something standing in front of the surface (hand, person, object).
The resulting mask lives in plane space, so the compositor warps it with the same homography as the
ad and the ad is simply not drawn there.

Limits (be aware of them):
* Works for surfaces whose appearance is static -- a switched-off TV, a blank wall, a poster. It is
  NOT usable when the surface shows changing video: everything differs. In that case the estimator
  detects an implausibly large "foreground" and returns ``None`` (no occlusion handling) rather than
  punching holes in the ad. ``is_usable_for`` lets the caller disable it up front.
* Depth/parallax of non-planar details shows up as small differences; morphology and the minimum
  blob size remove most of it.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from .geometry import quad_dimensions
from .schemas import AdPlacerConfig


class OcclusionEstimator:
    def __init__(self, anchor_frame: np.ndarray, anchor_quad: np.ndarray, cfg: AdPlacerConfig, long_side: int = 256):
        self.cfg = cfg
        qw, qh = quad_dimensions(np.asarray(anchor_quad, np.float32))
        if qw >= qh:
            self.size = (long_side, int(np.clip(round(long_side * qh / max(qw, 1)), 32, 512)))
        else:
            self.size = (int(np.clip(round(long_side * qw / max(qh, 1)), 32, 512)), long_side)
        cw, ch = self.size
        self._rect = np.float32([[0, 0], [cw, 0], [cw, ch], [0, ch]])
        self._ref, _ = self._to_plane(anchor_frame, anchor_quad)
        self._ref_med = np.median(self._ref.reshape(-1, 3), axis=0)
        self._prev: Optional[np.ndarray] = None
        self.unreliable_frames = 0
        self.frames = 0

    def _to_plane(self, frame: np.ndarray, quad: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        M = cv2.getPerspectiveTransform(np.asarray(quad, np.float32).reshape(4, 2), self._rect)
        plane = cv2.warpPerspective(frame, M, self.size, flags=cv2.INTER_AREA)
        inside = cv2.warpPerspective(np.full(frame.shape[:2], 255, np.uint8), M, self.size, flags=cv2.INTER_NEAREST)
        return plane, inside > 0

    @staticmethod
    def is_usable_for(anchor_frame: np.ndarray, anchor_quad: np.ndarray, max_mean_luma: float = 70.0) -> bool:
        """Static-appearance check: a bright (switched-on) screen cannot be background-subtracted."""
        mask = np.zeros(anchor_frame.shape[:2], np.uint8)
        cv2.fillConvexPoly(mask, np.round(anchor_quad).astype(np.int32), 255)
        gray = cv2.cvtColor(anchor_frame, cv2.COLOR_BGR2GRAY)
        return float(gray[mask > 0].mean()) <= max_mean_luma

    def reset(self) -> None:
        self._prev = None

    def estimate(self, frame: np.ndarray, quad: np.ndarray) -> Optional[np.ndarray]:
        """Plane-space float32 mask in [0, 1] (1 = occluded), or None when it cannot be trusted."""
        self.frames += 1
        cur, inside = self._to_plane(frame, quad)
        if inside.mean() < 0.25:
            return None
        cur_f = cv2.GaussianBlur(cur, (0, 0), 1.2).astype(np.float32)
        ref_f = cv2.GaussianBlur(self._ref, (0, 0), 1.2).astype(np.float32)
        # photometric normalisation: remove the global per-channel offset (exposure / white balance drift)
        med = np.median(cur_f[inside].reshape(-1, 3), axis=0)
        diff = np.abs((cur_f - med) - (ref_f - self._ref_med)).mean(axis=2) / 255.0
        thr = self.cfg.occlusion_threshold
        m = np.clip((diff - 0.6 * thr) / (0.8 * thr), 0.0, 1.0)
        m[~inside] = 0.0
        # Ignore a thin border: corner error of a pixel or two makes the wall/bezel strip look like a
        # "foreground" and would flicker; an occluder crossing the edge still covers the rest of it.
        bx, by = max(3, int(0.04 * self.size[0])), max(3, int(0.04 * self.size[1]))
        m[:by, :] = 0
        m[-by:, :] = 0
        m[:, :bx] = 0
        m[:, -bx:] = 0
        hard = (m > 0.5).astype(np.uint8)
        # opening with a large kernel removes thin strips (a few px of edge misalignment, reflections along the bezel);
        # a real occluder (hand, person, object) is thick
        k = max(5, int(0.045 * min(self.size)) | 1)
        hard = cv2.morphologyEx(hard, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        hard = cv2.morphologyEx(hard, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        # keep only blobs big enough to be real objects
        n, lab, stats, _ = cv2.connectedComponentsWithStats(hard, connectivity=8)
        keep = np.zeros_like(hard)
        min_area = 0.015 * hard.size
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                keep[lab == i] = 1
        frac = float(keep.sum()) / float(max(inside.sum(), 1))
        if frac > self.cfg.occlusion_max_fraction:
            self.unreliable_frames += 1
            self._prev = None
            return None
        mask = cv2.dilate(keep, np.ones((5, 5), np.uint8)).astype(np.float32)
        mask = cv2.GaussianBlur(mask, (0, 0), 1.5)
        if self._prev is not None:  # temporal hysteresis: no flicker of the mask edge
            mask = np.maximum(mask, 0.6 * self._prev)
        self._prev = mask
        return mask
