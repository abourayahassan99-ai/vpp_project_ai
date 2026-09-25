import cv2
import numpy as np
from typing import Dict, Optional, Tuple, List
from .geometry import order_corners, is_valid_quad

class FrameSource:
    """كائن لقراءة وتوفير الفريمات من الفيديو يدعم الـ Context Manager."""

    def __init__(self, source_path):
        self.source_path = str(source_path)
        self.cap = cv2.VideoCapture(self.source_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"FrameSource path: {self.source_path}")
        print(f"FrameSource total frames: {self.total_frames}")
    @property
    def is_video(self) -> bool:
        """تأكيد أن المصدر عبارة عن فيديو مفتوح ويعمل."""
        return self.cap is not None and self.cap.isOpened() and self.total_frames > 1

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
    

    def read(self, idx: int, timestamp_sec: Optional[float] = None, *args, **kwargs) -> Optional[np.ndarray]:
        if not self.cap or not self.cap.isOpened():
            return None
        
        if idx is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        elif timestamp_sec is not None and self.fps > 0:
            frame_from_time = int(timestamp_sec * self.fps)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_from_time)

        ret, frame = self.cap.read()
        return frame if ret else None

    def reset(self) -> None:
        if self.cap and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def release(self) -> None:
        if self.cap and self.cap.isOpened():
            self.cap.release()
def _rewind_source(source) -> None:
    """Rewind video source safely."""
    if hasattr(source, "reset"):
        source.reset()
    elif hasattr(source, "cap") and source.cap is not None:
        source.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    elif hasattr(source, "set"):
        source.set(cv2.CAP_PROP_POS_FRAMES, 0)


def _get_band_mask(frame_shape: tuple, quad: np.ndarray, margin: int = 25) -> np.ndarray:
    """Create a mask for a band around the quad boundary (excludes dark screen center)."""
    h, w = frame_shape[:2]
    outer_mask = np.zeros((h, w), dtype=np.uint8)
    inner_mask = np.zeros((h, w), dtype=np.uint8)

    quad_int = quad.astype(np.int32)
    cv2.fillConvexPoly(outer_mask, quad_int, 255)

    # Erode to get inner area
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (margin, margin))
    inner_mask = cv2.erode(outer_mask, kernel)

    band_mask = cv2.subtract(outer_mask, inner_mask)
    return band_mask


def validate_homography(H: np.ndarray, key_quad: np.ndarray, new_quad: np.ndarray, frame_shape: tuple) -> bool:
    """Validate Homography matrix to prevent scale explosion and extreme drift."""
    if H is None or H.shape != (3, 3):
        return False

    # 1. Determinant Check (affine scale change)
    det = np.linalg.det(H[:2, :2])
    if det < 0.2 or det > 4.0:
        return False

    # 2. Quad Convexity and Area Check
    if not is_valid_quad(new_quad, frame_shape=frame_shape):
        return False

    # 3. Area ratio check relative to key frame quad
    key_area = cv2.contourArea(key_quad.astype(np.int32))
    new_area = cv2.contourArea(new_quad.astype(np.int32))

    if key_area > 0:
        ratio = new_area / key_area
        if ratio < 0.15 or ratio > 3.5:
            return False

    return True


def track_surface_continuous(
    source,
    key_idx: int,
    key_quad: np.ndarray,
    first_seen_frame: Optional[int] = None,
    max_frames: Optional[int] = None,
    **kwargs
) -> Dict[int, np.ndarray]:
    """Frame-by-frame tracker using Forward-Backward Optical Flow and Homography validation."""
    _rewind_source(source)
    key_quad = order_corners(key_quad)
    quads: Dict[int, np.ndarray] = {key_idx: key_quad}

    # Fetch initial frame
    prev_frame = source.read(key_idx) if hasattr(source, "read") else None
    if prev_frame is None:
        return quads

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_quad = key_quad.copy()

    # Track Forward
    curr_idx = key_idx + 1
    frame_count = 0

    while True:
        if max_frames is not None and frame_count >= max_frames:
            break

        curr_frame = source.read(curr_idx) if hasattr(source, "read") else None
        if curr_frame is None:
            break

        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # 1. Feature detection in outer band
        band_mask = _get_band_mask(prev_frame.shape, curr_quad, margin=20)
        p0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=5, mask=band_mask)

        if p0 is not None and len(p0) >= 8:
            # 2. Forward-Backward LK Optical Flow
            p1, st1, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None)
            p0r, st0, _ = cv2.calcOpticalFlowPyrLK(curr_gray, prev_gray, p1, None)

            # Filter points based on FB error
            fb_err = np.linalg.norm(p0 - p0r, axis=2).reshape(-1)
            valid = (st1.reshape(-1) == 1) & (st0.reshape(-1) == 1) & (fb_err < 1.5)

            pts_prev = p0[valid]
            pts_curr = p1[valid]

            if len(pts_prev) >= 6:
                H, inliers = cv2.findHomography(pts_prev, pts_curr, cv2.RANSAC, 3.0)
                if H is not None:
                    # Transform current quad
                    pts_src = curr_quad.reshape(-1, 1, 2)
                    pts_dst = cv2.perspectiveTransform(pts_src, H).reshape(4, 2)
                    candidate_quad = order_corners(pts_dst)

                    # Validate H before accepting
                    if validate_homography(H, key_quad, candidate_quad, curr_frame.shape):
                        curr_quad = candidate_quad
                        quads[curr_idx] = curr_quad.copy()

        prev_gray = curr_gray
        curr_idx += 1
        frame_count += 1

    # Offline Smoothing Pass (Moving Average)
    smoothed_quads = _smooth_quad_trajectory(quads)
    return smoothed_quads


def _smooth_quad_trajectory(quads: Dict[int, np.ndarray], window_size: int = 5) -> Dict[int, np.ndarray]:
    """Smooth corner coordinates across frames to prevent jitter."""
    sorted_keys = sorted(quads.keys())
    if len(sorted_keys) < window_size:
        return quads

    smoothed = {}
    pts_array = np.array([quads[k] for k in sorted_keys])  # (N, 4, 2)

    kernel = np.ones(window_size) / window_size
    smoothed_pts = np.zeros_like(pts_array)

    for c in range(4):
        for coord in range(2):
            smoothed_pts[:, c, coord] = np.convolve(pts_array[:, c, coord], kernel, mode='same')

    for i, k in enumerate(sorted_keys):
        smoothed[k] = order_corners(smoothed_pts[i])

    return smoothed


def track_surface(*args, **kwargs):
    """Wrapper function for backwards compatibility."""
    return track_surface_continuous(*args, **kwargs)


__all__ = ["FrameSource", "track_surface", "track_surface_continuous"]