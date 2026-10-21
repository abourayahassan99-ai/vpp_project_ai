"""
Surface-aware perspective tracking for Agent 3.

Equivalent of After Effects' "perspective corner pin" tracker, built for an offline pipeline.

Concepts (kept separate on purpose)
-----------------------------------
* Anchor   : one reference frame + a 4-corner quad (TL, TR, BR, BL) that defines the plane. Corner
             order is fixed here and never re-sorted afterwards.
* Plane points : image features on the surface's plane (a ring around / along the edge of the quad,
             *not* the screen interior, which may be dark or show moving video). Every point stores
             its coordinates in the anchor frame.
* Homography : each frame solves H(anchor -> frame) directly from those anchor-referenced points
             with RANSAC, so errors do not accumulate frame after frame. quad_t = H(anchor quad).
* Re-anchoring : every few frames AKAZE descriptors of the anchor ring are matched to the current
             frame; this cancels slow drift and is also how the surface is *re-acquired* after loss.
* State machine : TRACKING -> DEGRADED (coasting) -> LOST -> REACQUIRE -> TRACKING. A LOST surface
             produces no quad, so no ad is ever drawn from stale coordinates.
* Smoothing : after the raw pass, each contiguous segment is smoothed with a confidence-weighted
             constant-velocity Kalman filter + RTS backward pass (see smoothing.py).

Backward compatibility: ``FrameSource``, ``track_surface`` and ``track_surface_continuous`` keep
their old signatures; the latter now returns ``Dict[frame_idx, quad]`` only for frames where the
surface was actually present.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .geometry import order_corners, quad_dimensions, quad_inside_ratio, quad_is_plausible
from .schemas import AdPlacerConfig
from .edge_refine import EdgeRefiner
from .smoothing import smooth_segment

logger = logging.getLogger(__name__)

TRACKING, DEGRADED, LOST, REACQUIRE = "TRACKING", "DEGRADED", "LOST", "REACQUIRE"

_LK_PARAMS = dict(
    winSize=(31, 31), maxLevel=4,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)


# --------------------------------------------------------------------------- #
# Frame source (kept for compatibility; now avoids needless seeking)
# --------------------------------------------------------------------------- #
class FrameSource:
    """Video reader with context-manager support. Sequential reads never seek."""

    def __init__(self, source_path):
        self.source_path = str(source_path)
        self.cap = cv2.VideoCapture(self.source_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._pos = 0  # index of the next frame the decoder will return
        logger.info("FrameSource %s: %d frames, %dx%d @ %.2f fps",
                    self.source_path, self.total_frames, self.width, self.height, self.fps)

    @property
    def is_video(self) -> bool:
        return self.cap is not None and self.cap.isOpened() and self.total_frames > 1

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    def read(self, idx: Optional[int] = None, timestamp_sec: Optional[float] = None, *args, **kwargs):
        if not self.cap or not self.cap.isOpened():
            return None
        if idx is None and timestamp_sec is not None and self.fps > 0:
            idx = int(round(timestamp_sec * self.fps))
        if idx is not None and idx != self._pos:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            self._pos = int(idx)
        ret, frame = self.cap.read()
        if ret:
            self._pos += 1
        return frame if ret else None

    def reset(self) -> None:
        if self.cap and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._pos = 0

    def release(self) -> None:
        if self.cap and self.cap.isOpened():
            self.cap.release()


def _rewind_source(source) -> None:
    if hasattr(source, "reset"):
        source.reset()
    elif hasattr(source, "cap") and source.cap is not None:
        source.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass
class TrackResult:
    """Output of :class:`PlanarSurfaceTracker` (all coordinates in full-resolution pixels)."""

    quads: Dict[int, np.ndarray] = field(default_factory=dict)       # frame -> (4,2) TL,TR,BR,BL
    confidence: Dict[int, float] = field(default_factory=dict)       # 0..1 (0 = interpolated)
    opacity: Dict[int, float] = field(default_factory=dict)          # 0..1 fade for the ad
    states: Dict[int, str] = field(default_factory=dict)             # raw state per processed frame
    total_frames: int = 0
    fps: float = 30.0
    size: Tuple[int, int] = (0, 0)                                   # (w, h)
    metrics: Dict[str, object] = field(default_factory=dict)


@dataclass
class _Rec:
    quad: np.ndarray            # working-scale quad
    conf: float                 # 0 => predicted / unmeasured
    kind: str                   # "measured" | "coast"
    verified: bool = False      # geometry backed by a strong anchor registration / many inliers


def _smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


# --------------------------------------------------------------------------- #
# Tracker
# --------------------------------------------------------------------------- #
class PlanarSurfaceTracker:
    def __init__(self, cfg: Optional[AdPlacerConfig] = None):
        self.cfg = cfg or AdPlacerConfig()
        self._akaze = cv2.AKAZE_create(threshold=0.0005)
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.stats: Dict[str, int] = {}
        self._a_desc = None
        self._a_kp: List[Tuple[float, float]] = []
        self._exclude_interior = False
        self._last_cov = 1.0

    # ----- small helpers --------------------------------------------------- #
    def _bump(self, key: str, n: int = 1) -> None:
        self.stats[key] = self.stats.get(key, 0) + n

    def _ring_mask(self, quad: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
        """Region of the *surface plane* used for features: a band along the quad's edge + a ring outside."""
        h, w = shape
        c = self.cfg
        poly = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(poly, np.round(quad).astype(np.int32), 255)
        qw, qh = quad_dimensions(quad)
        m = max(min(qw, qh), 8.0)
        r_out = int(max(6, c.track_ring_ratio * m))
        outer = cv2.dilate(poly, cv2.getStructuringElement(cv2.MORPH_RECT, (2 * r_out + 1, 2 * r_out + 1)))
        if self._exclude_interior or c.track_band_margin <= 0:
            inner = poly  # screen shows moving video: never track its interior
        else:
            # static surface (dark switched-off TV, wall, poster): its edge band holds the strongest
            # corners (TV corners!) and keeps the model well conditioned on all four sides
            r_in = int(np.clip(c.track_band_margin, 2, 0.2 * m))
            inner = cv2.erode(poly, cv2.getStructuringElement(cv2.MORPH_RECT, (2 * r_in + 1, 2 * r_in + 1)))
        return cv2.subtract(outer, inner)

    def _confidence(self, n_in: int, n_used: int, rmse: float) -> float:
        c = self.cfg
        c_n = np.clip(n_in / (3.0 * c.track_min_inliers), 0, 1)
        c_r = np.clip((n_in / max(n_used, 1)) / 0.7, 0, 1)
        c_e = np.exp(-rmse / max(c.track_ransac_px, 1e-3))
        return float(max(0.05, c_n * (0.5 + 0.5 * c_r) * (0.5 + 0.5 * c_e)))

    def _homography(self, src: np.ndarray, dst: np.ndarray):
        if len(src) < 4:
            return None, None
        try:
            H, m = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, self.cfg.track_ransac_px,
                                      maxIters=2000, confidence=0.995)
        except cv2.error:
            H, m = cv2.findHomography(src, dst, cv2.RANSAC, self.cfg.track_ransac_px)
        if H is None or m is None or not np.isfinite(H).all():
            return None, None
        return H, m.reshape(-1).astype(bool)

    def _fit_model(self, src: np.ndarray, dst: np.ndarray, ref_area: float):
        """
        Fit anchor->frame with a model matched to how well the points constrain it.
        Points that only cover a thin strip of the plane (typical: cables below a dark TV, plain wall
        above) cannot determine perspective; a full homography then extrapolates wildly to the far
        corners. Coverage = convex-hull area of the inlier points / current quad area.
            >= 0.35 : homography (8 DoF)      0.12-0.35 : affine (6 DoF)      < 0.12 : similarity (4 DoF)
        Returns (H 3x3, inlier mask, model name) or (None, None, None).
        """
        H, inl = self._homography(src, dst)
        if H is None:
            return None, None, None
        pts = dst[inl].reshape(-1, 2).astype(np.float32)
        cov = 1.0
        if len(pts) >= 3 and ref_area > 1.0:
            cov = float(cv2.contourArea(cv2.convexHull(pts))) / ref_area
        self._last_cov = cov
        if cov >= self.cfg.track_cov_homography:
            return H, inl, "homography"
        fn = cv2.estimateAffine2D if cov >= self.cfg.track_cov_affine else cv2.estimateAffinePartial2D
        A, m = fn(src, dst, method=cv2.RANSAC, ransacReprojThreshold=self.cfg.track_ransac_px)
        if A is None or m is None:
            return H, inl, "homography"
        self._bump("reduced_model_frames")
        return np.vstack([A, [0.0, 0.0, 1.0]]), m.reshape(-1).astype(bool), ("affine" if fn is cv2.estimateAffine2D else "similarity")

    # ----- anchor registration (AKAZE) ------------------------------------- #
    def _register_to_anchor(self, gray: np.ndarray, hint_quad: Optional[np.ndarray] = None, min_inliers: int = 15):
        """Match anchor ring descriptors to ``gray``. Returns dict or None."""
        if self._a_desc is None or len(self._a_kp) < 8:
            return None
        mask = None
        if hint_quad is not None:
            mask = self._ring_mask(hint_quad, gray.shape)
            mask = cv2.dilate(mask, np.ones((31, 31), np.uint8))
        kp, desc = self._akaze.detectAndCompute(gray, mask)
        if desc is None or len(kp) < 8:
            return None
        knn = self._bf.knnMatch(self._a_desc, desc, k=2)
        good = [p[0] for p in knn if len(p) == 2 and p[0].distance < 0.8 * p[1].distance]
        if len(good) < max(8, self.cfg.track_min_inliers):
            return None
        src = np.float32([self._a_kp[m.queryIdx] for m in good])
        dst = np.float32([kp[m.trainIdx].pt for m in good])
        ref_area = abs(cv2.contourArea((hint_quad if hint_quad is not None else self._a_quad).astype(np.float32)))
        H, inl, _ = self._fit_model(src.reshape(-1, 1, 2), dst.reshape(-1, 1, 2), ref_area)
        if H is None or inl.sum() < max(self.cfg.track_min_inliers, min_inliers) or inl.sum() < 0.3 * len(good):
            return None
        quad = cv2.perspectiveTransform(self._a_quad.reshape(-1, 1, 2), H).reshape(4, 2)
        if not quad_is_plausible(quad, gray.shape):
            return None
        proj = cv2.perspectiveTransform(src[inl].reshape(-1, 1, 2), H).reshape(-1, 2)
        rmse = float(np.sqrt(np.mean(np.sum((proj - dst[inl]) ** 2, axis=1))))
        return dict(H=H, quad=quad, pts_a=src[inl].reshape(-1, 1, 2), pts_c=dst[inl].reshape(-1, 1, 2),
                    n_in=int(inl.sum()), n_used=len(good), rmse=rmse)

    # ----- anchor setup ---------------------------------------------------- #
    def _setup_anchor(self, anchor_gray: np.ndarray, anchor_quad: np.ndarray) -> None:
        self._a_gray = anchor_gray
        self._a_quad = anchor_quad.astype(np.float32)
        mask = self._ring_mask(self._a_quad, anchor_gray.shape)
        kps, desc = self._akaze.detectAndCompute(anchor_gray, mask)
        self._a_kp = [k.pt for k in kps] if kps else []
        self._a_desc = desc
        logger.info("Anchor: %d AKAZE keypoints on the plane ring", len(self._a_kp))
        if len(self._a_kp) < 30:
            logger.warning("Few anchor features (%d): tracking will rely on LK only and re-acquisition "
                           "after loss may fail. Pick a sharper anchor frame or a larger ring.", len(self._a_kp))

    def _fresh_points(self, gray, quad, existing_c, H_inv, n_max):
        """New good features on the ring, expressed in anchor coordinates via H_inv."""
        mask = self._ring_mask(quad, gray.shape)
        if existing_c is not None:
            for x, y in existing_c.reshape(-1, 2):
                cv2.circle(mask, (int(x), int(y)), 6, 0, -1)
        new_c = cv2.goodFeaturesToTrack(gray, maxCorners=n_max, qualityLevel=0.01, minDistance=7,
                                        mask=mask, blockSize=5)
        if new_c is None:
            return None, None
        new_a = cv2.perspectiveTransform(new_c, H_inv)
        return new_a.astype(np.float32), new_c.astype(np.float32)

    # ----- main entry ------------------------------------------------------ #
    def run(
        self,
        video_path: str,
        anchor_idx: int,
        anchor_quad: np.ndarray,
        anchor_frame: Optional[np.ndarray] = None,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        progress: Optional[Callable[[int, int], None]] = None,
        exclude_interior: bool = False,
    ) -> TrackResult:
        """
        ``exclude_interior=True`` keeps the quad interior out of the feature region; use it when the
        surface shows changing content (a switched-on TV). Static surfaces keep an inner edge band.
        """
        c = self.cfg
        self._exclude_interior = exclude_interior
        t0 = time.perf_counter()
        self.stats = {}
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        W, Hh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        scale = min(1.0, c.track_work_width / float(W))
        ww, wh = int(round(W * scale)), int(round(Hh * scale))

        def to_gray(img):
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return cv2.resize(g, (ww, wh), interpolation=cv2.INTER_AREA) if scale < 1 else g

        if anchor_frame is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, anchor_idx)
            ok, anchor_frame = cap.read()
            if not ok:
                cap.release()
                raise RuntimeError(f"Could not read anchor frame {anchor_idx}")
        a_quad_w = np.asarray(anchor_quad, np.float32).reshape(4, 2) * scale
        if not quad_is_plausible(a_quad_w, (wh, ww)):
            cap.release()
            raise ValueError("Anchor quad is not a plausible convex quadrilateral: "
                             f"{np.asarray(anchor_quad).round(1).tolist()}")
        self._setup_anchor(to_gray(anchor_frame), a_quad_w)
        self._edges = EdgeRefiner(self._a_gray, a_quad_w, bezel_ratio=c.anchor_bezel_ratio) if c.track_edge_refine else None
        if self._edges is not None:
            logger.info("Edge refinement: %d/4 sides have a usable edge in the anchor", self._edges.usable_sides)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        end = (total - 1) if end_frame is None else min(end_frame, total - 1)

        records: Dict[int, _Rec] = {}
        pending: Dict[int, _Rec] = {}
        trailing: List[int] = []
        raw_states: Dict[int, str] = {}

        state = LOST
        ref_gray = None
        pts_a = pts_c = None
        last_quad = None
        vel = np.zeros((4, 2), np.float32)
        H_inv = np.eye(3)
        coast = 0
        streak = 0
        since_reanchor = 0
        snapshots = []
        need_snap = False
        t = start_frame

        while t <= end:
            ok, frame = cap.read()
            if not ok:
                break
            gray = to_gray(frame)

            # ---------------- acquisition at the anchor frame itself ---------------- #
            if t == anchor_idx:
                a_c = cv2.goodFeaturesToTrack(self._a_gray, maxCorners=c.track_max_features, qualityLevel=0.01,
                                              minDistance=7, mask=self._ring_mask(self._a_quad, self._a_gray.shape),
                                              blockSize=5)
                if a_c is not None and len(a_c) >= 8:
                    pts_a = pts_c = a_c.astype(np.float32)
                    ref_gray, last_quad, vel = gray, self._a_quad.copy(), np.zeros((4, 2), np.float32)
                    state, coast, streak, since_reanchor = TRACKING, 0, c.track_reacquire_confirm, 0
                    records.update(pending)
                    pending.clear()
                    records[t] = _Rec(last_quad.copy(), 1.0, "measured", verified=True)
                    snapshots.append(dict(frame=t, gray=gray.copy(), pts_a=pts_a.copy(), pts_c=pts_c.copy(),
                                          quad=last_quad.copy()))
                    need_snap = False
                    raw_states[t] = state
                    t += 1
                    continue

            # ---------------- LOST: periodic re-acquisition --------------------------- #
            if state == LOST:
                raw_states[t] = LOST
                if (t - start_frame) % max(1, c.track_reacquire_interval) == 0:
                    self._bump("reacquire_attempts")
                    reg = self._register_to_anchor(gray, min_inliers=c.track_acquire_min_inliers)
                    if reg is not None:
                        self._bump("reacquire_success")
                        pts_a, pts_c, ref_gray = reg["pts_a"], reg["pts_c"], gray
                        last_quad, vel = reg["quad"].astype(np.float32), np.zeros((4, 2), np.float32)
                        # re-check against the anchor again very soon: a fresh registration is the weakest estimate
                        state, coast, streak = REACQUIRE, 0, 1
                        since_reanchor = max(0, c.track_reanchor_interval - 4)
                        conf = self._confidence(reg["n_in"], reg["n_used"], reg["rmse"])
                        pending[t] = _Rec(last_quad.copy(), conf, "measured", verified=reg["n_in"] >= 40)
                        need_snap = True
                        if reg["n_in"] >= 40:
                            snapshots.append(dict(frame=t, gray=gray.copy(), pts_a=pts_a.copy(), pts_c=pts_c.copy(),
                                                  quad=last_quad.copy()))
                            need_snap = False
                        raw_states[t] = REACQUIRE
                        logger.info("frame %d: surface re-acquired (%d inliers)", t, reg["n_in"])
                t += 1
                if progress and t % 200 == 0:
                    progress(t, end)
                continue

            # ---------------- TRACKING / DEGRADED / REACQUIRE: LK + anchored H ------- #
            quad_t = None
            conf = 0.0
            success = False
            step = self._lk_step(ref_gray, gray, pts_a, pts_c, last_quad, vel, coast, (wh, ww))
            if step is not None:
                success, quad_t, conf = True, step["quad"], step["conf"]
                pts_a, pts_c, H_inv = step["pts_a"], step["pts_c"], step["H_inv"]

            if success:
                coast = 0
                trailing.clear()
                since_reanchor += 1
                verified = int(len(pts_c)) >= 40 and conf >= 0.6
                if len(pts_c) < 0.6 * c.track_max_features:  # replenish features on the plane
                    na, nc = self._fresh_points(gray, quad_t, pts_c, H_inv, c.track_max_features - len(pts_c))
                    if na is not None:
                        pts_a, pts_c = np.vstack([pts_a, na]), np.vstack([pts_c, nc])
                if since_reanchor >= c.track_reanchor_interval:  # drift correction against the anchor
                    since_reanchor = 0
                    reg = self._register_to_anchor(gray, hint_quad=quad_t)
                    if reg is not None:
                        size = max(quad_dimensions(quad_t))
                        diff = float(np.mean(np.linalg.norm(reg["quad"] - quad_t, axis=1)))
                        if diff <= 0.5 * size and reg["n_in"] >= max(15, c.track_min_inliers):
                            quad_t = reg["quad"]
                            pts_a, pts_c = reg["pts_a"], reg["pts_c"]
                            conf = max(conf, self._confidence(reg["n_in"], reg["n_used"], reg["rmse"]))
                            verified = reg["n_in"] >= 40
                            if reg["n_in"] < 40:  # weak registration: look again soon
                                since_reanchor = max(0, c.track_reanchor_interval - 5)
                            self._bump("reanchor_accepted")
                        else:
                            self._bump("reanchor_rejected")
                vel = (quad_t - last_quad).astype(np.float32)
                last_quad, ref_gray = quad_t.astype(np.float32), gray
                out_quad = last_quad
                if self._edges is not None:
                    out_quad, n_edge = self._edges.refine(gray, last_quad)
                    self._bump("edge_sides_corrected", n_edge)
                rec = _Rec(out_quad.copy(), conf, "measured", verified=verified)
                if need_snap and verified:
                    snapshots.append(dict(frame=t, gray=gray.copy(), pts_a=pts_a.copy(), pts_c=pts_c.copy(),
                                          quad=last_quad.copy()))
                    need_snap = False
                if state == REACQUIRE:
                    streak += 1
                    pending[t] = rec
                    raw_states[t] = REACQUIRE
                    if streak >= c.track_reacquire_confirm:
                        records.update(pending)
                        pending.clear()
                        state = TRACKING
                else:
                    state = TRACKING if conf >= 0.45 else DEGRADED
                    records[t] = rec
                    raw_states[t] = state
            else:
                self._bump("failed_frames")
                if state == REACQUIRE:
                    pending.clear()
                    need_snap = False
                    state, pts_a, pts_c = LOST, None, None
                    raw_states[t] = LOST
                else:
                    coast += 1
                    if coast > c.track_max_coast_frames:
                        for f in trailing:
                            records.pop(f, None)
                        trailing.clear()
                        need_snap = False
                        state, pts_a, pts_c = LOST, None, None
                        raw_states[t] = LOST
                        logger.info("frame %d: surface LOST", t)
                    else:
                        state = DEGRADED
                        raw_states[t] = DEGRADED
                        pred = (last_quad + vel * 0.5).astype(np.float32)  # damped coasting
                        records[t] = _Rec(pred, 0.0, "coast")
                        trailing.append(t)
            t += 1
            if progress and t % 200 == 0:
                progress(t, end)

        cap.release()
        if c.track_backfill:
            for snap in snapshots:
                self._backfill(video_path, snap, start_frame, to_gray, records, raw_states, (wh, ww))
        return self._finalise(records, raw_states, scale, fps, total, (W, Hh), (wh, ww),
                              time.perf_counter() - t0, start_frame, end)

    # ----- one LK + anchored-homography step (shared by forward tracking and backward fill) --------- #
    def _lk_step(self, ref_gray, gray, pts_a, pts_c, last_quad, vel, coast, wshape):
        """Track ``pts_c`` from ``ref_gray`` to ``gray``, fit the anchor homography and gate the result."""
        c = self.cfg
        wh, ww = wshape
        if pts_c is None or len(pts_c) < 6:
            return None
        p1, st1, _ = cv2.calcOpticalFlowPyrLK(ref_gray, gray, pts_c, None, **_LK_PARAMS)
        p0r, st0, _ = cv2.calcOpticalFlowPyrLK(gray, ref_gray, p1, None, **_LK_PARAMS)
        fb = np.linalg.norm(pts_c - p0r, axis=2).reshape(-1)
        inb = ((p1[:, 0, 0] > -ww * 0.1) & (p1[:, 0, 0] < ww * 1.1) &
               (p1[:, 0, 1] > -wh * 0.1) & (p1[:, 0, 1] < wh * 1.1))
        keep = (st1.reshape(-1) == 1) & (st0.reshape(-1) == 1) & (fb < c.track_fb_error_px) & inb
        if keep.sum() < 6:
            return None
        a_k, c_k = pts_a[keep], p1[keep]
        H, inl, _model = self._fit_model(a_k, c_k, abs(cv2.contourArea(last_quad.astype(np.float32))))
        if H is None or inl.sum() < c.track_min_inliers:
            return None
        quad_t = cv2.perspectiveTransform(self._a_quad.reshape(-1, 1, 2), H).reshape(4, 2)
        proj = cv2.perspectiveTransform(a_k[inl], H).reshape(-1, 2)
        rmse = float(np.sqrt(np.mean(np.sum((proj - c_k[inl].reshape(-1, 2)) ** 2, axis=1))))
        area_prev = abs(cv2.contourArea(last_quad.astype(np.float32)))
        area_new = abs(cv2.contourArea(quad_t.astype(np.float32)))
        ratio = area_new / max(area_prev, 1e-6)
        size = max(quad_dimensions(last_quad))
        jump = float(np.mean(np.linalg.norm(quad_t - (last_quad + vel), axis=1)))
        gap = coast + 1  # LK baseline spans this many frames after failures
        lim = c.track_max_scale_change ** gap
        if not (quad_is_plausible(quad_t, (wh, ww)) and 1.0 / lim <= ratio <= lim
                and jump <= c.track_jump_tolerance * size * gap
                and quad_inside_ratio(quad_t, (wh, ww)) > 0.05):
            return None
        return dict(quad=quad_t, conf=self._confidence(int(inl.sum()), int(keep.sum()), rmse),
                    pts_a=a_k[inl], pts_c=c_k[inl], H_inv=np.linalg.inv(H))

    # ----- backward fill ----------------------------------------------------------------------------- #
    def _backfill(self, video_path, snap, floor_frame, to_gray, records, raw_states, wshape):
        """
        Track BACKWARDS from the first verified frame of an episode so the ad can appear from the first frame the
        surface is reliably tracked. Every step passes exactly the same gates as forward tracking, must reach
        ``track_backfill_min_conf``, and stops after two consecutive failures. Frames that already hold verified
        data are never overwritten (and stop the walk).
        """
        c = self.cfg
        t0 = snap["frame"]
        lo = max(floor_frame, t0 - c.track_backfill_max)
        if t0 - 1 < lo:
            return 0
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
        grays = {}
        for j in range(lo, t0):
            ok, fr = cap.read()
            if not ok:
                break
            grays[j] = to_gray(fr)
        cap.release()
        ref_gray, pts_a, pts_c = snap["gray"], snap["pts_a"], snap["pts_c"]
        last_quad, vel, coast, done = snap["quad"].copy(), np.zeros((4, 2), np.float32), 0, 0
        for j in range(t0 - 1, lo - 1, -1):
            if j not in grays:
                break
            existing = records.get(j)
            if existing is not None and existing.verified:
                break
            step = self._lk_step(ref_gray, grays[j], pts_a, pts_c, last_quad, vel, coast, wshape)
            if step is None or step["conf"] < c.track_backfill_min_conf:
                coast += 1
                if coast > 2:
                    break
                continue
            coast = 0
            quad_t, pts_a, pts_c = step["quad"], step["pts_a"], step["pts_c"]
            if len(pts_c) < 0.6 * c.track_max_features:
                na, nc = self._fresh_points(grays[j], quad_t, pts_c, step["H_inv"], c.track_max_features - len(pts_c))
                if na is not None:
                    pts_a, pts_c = np.vstack([pts_a, na]), np.vstack([pts_c, nc])
            vel = (quad_t - last_quad).astype(np.float32)
            last_quad, ref_gray = quad_t.astype(np.float32), grays[j]
            out_quad = last_quad
            if self._edges is not None:
                out_quad, n_edge = self._edges.refine(grays[j], last_quad)
                self._bump("edge_sides_corrected", n_edge)
            records[j] = _Rec(out_quad.copy(), step["conf"], "measured", verified=True)
            raw_states[j] = "BACKFILL"
            done += 1
        if done:
            self._bump("backfilled_frames", done)
        return done

    # ----- post-processing --------------------------------------------------- #
    def _finalise(self, records, raw_states, scale, fps, total, size, wshape, elapsed, start_frame, end) -> TrackResult:
        c = self.cfg
        res = TrackResult(total_frames=total, fps=fps, size=size, states=raw_states)
        frames = sorted(records)
        if not frames:
            res.metrics = dict(self.stats, tracked_frames=0, processing_seconds=round(elapsed, 2))
            return res

        # contiguous runs; runs separated by short gaps are merged (the filter interpolates the gap)
        runs: List[List[int]] = [[frames[0]]]
        for f in frames[1:]:
            if f == runs[-1][-1] + 1:
                runs[-1].append(f)
            else:
                runs.append([f])
        segments: List[List[int]] = [runs[0]]
        for r in runs[1:]:
            if r[0] - segments[-1][-1] - 1 <= c.track_max_interp_gap:
                segments[-1] = segments[-1] + r
            else:
                segments.append(r)

        n_interp = 0
        for seg in segments:
            idx = np.arange(seg[0], seg[-1] + 1)
            q = np.full((len(idx), 4, 2), np.nan, np.float64)
            cf = np.zeros(len(idx))
            for f in seg:
                q[f - idx[0]] = records[f].quad
                cf[f - idx[0]] = records[f].conf
            for k in range(len(idx)):  # NaN rows (gaps) get a finite placeholder with zero weight
                if not np.isfinite(q[k]).all():
                    q[k] = q[k - 1]
            n_interp += int(np.sum(cf == 0))
            sm = smooth_segment(q, cf, c.smooth_process_noise, c.smooth_measurement_noise, c.smooth_speed_ref)
            # probation: a segment that began with an unverified registration stays invisible until a
            # verified frame appears (never show an ad on geometry we do not trust)
            ver = [f for f in seg if records[f].verified]
            first_ok = ver[0] if ver else None
            if first_ok is None:
                self._bump("segments_never_verified")

            for k, f in enumerate(idx):
                vis = quad_inside_ratio(sm[k], wshape)
                v = _smoothstep((vis - c.track_min_visible_ratio) / max(c.track_min_visible_ratio, 1e-3))
                ramp = 1.0
                if first_ok is None or f < first_ok:
                    ramp = 0.0
                elif c.fade_frames > 0:  # fade at segment ends unless the segment touches the clip boundary
                    if idx[0] > start_frame:
                        ramp = min(ramp, _smoothstep((f - first_ok + 1) / (c.fade_frames + 1)))
                    if idx[-1] < end:
                        ramp = min(ramp, _smoothstep((idx[-1] - f + 1) / (c.fade_frames + 1)))
                res.quads[int(f)] = (sm[k] / scale).astype(np.float32)
                res.confidence[int(f)] = float(cf[k])
                res.opacity[int(f)] = float(v * ramp)

        confs = [records[f].conf for f in frames if records[f].kind == "measured"]
        res.metrics = dict(
            self.stats,
            tracked_frames=len(res.quads),
            visible_frames=int(sum(1 for o in res.opacity.values() if o > 0.01)),
            interpolated_frames=n_interp,
            segments=len(segments),
            mean_confidence=round(float(np.mean(confs)), 3) if confs else 0.0,
            processing_seconds=round(elapsed, 2),
            processing_fps=round((end - start_frame + 1) / max(elapsed, 1e-6), 1),
            work_scale=round(scale, 3),
        )
        return res


# --------------------------------------------------------------------------- #
# Backwards-compatible API
# --------------------------------------------------------------------------- #
def track_surface_continuous(
    source,
    key_idx: int,
    key_quad: np.ndarray,
    first_seen_frame: Optional[int] = None,
    max_frames: Optional[int] = None,
    **kwargs,
) -> Dict[int, np.ndarray]:
    """
    Old entry point. Now tracks both directions from the anchor and returns quads only for frames
    where the surface is really present. ``max_frames`` bounds the range around ``key_idx``.
    Pass ``cfg=AdPlacerConfig(...)`` to tune the tracker.
    """
    cfg = kwargs.get("cfg") or AdPlacerConfig()
    path = getattr(source, "source_path", None)
    if path is None:
        raise ValueError("track_surface_continuous needs a FrameSource (with .source_path)")
    key_quad = order_corners(np.asarray(key_quad, np.float32))
    anchor = source.read(key_idx)
    start = 0 if max_frames is None else max(0, key_idx - max_frames)
    end = None if max_frames is None else key_idx + max_frames
    result = PlanarSurfaceTracker(cfg).run(path, key_idx, key_quad, anchor_frame=anchor,
                                           start_frame=start, end_frame=end)
    return result.quads


def track_surface(*args, **kwargs):
    """Wrapper kept for backwards compatibility."""
    return track_surface_continuous(*args, **kwargs)


__all__ = ["FrameSource", "PlanarSurfaceTracker", "TrackResult", "track_surface", "track_surface_continuous",
           "TRACKING", "DEGRADED", "LOST", "REACQUIRE"]
