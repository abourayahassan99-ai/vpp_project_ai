"""
Geometry for Agent 3: polygon validation, corner fitting / ordering, and perspective warping.

Everything here is pure (no I/O, no config object) so it can be unit-tested in isolation.
Quads are always (4, 2) float32 arrays ordered TL, TR, BR, BL (clockwise on screen).
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import cv2
import numpy as np

from .schemas import PreparedAd


# --------------------------------------------------------------------------- #
# Polygon parsing / validation
# --------------------------------------------------------------------------- #
def parse_polygon(points: Any, frame_w: int, frame_h: int) -> Optional[np.ndarray]:
    """Convert raw ``polygon_points`` to an (N, 2) float32 array, or None if unusable."""
    if points is None:
        return None
    try:
        pts = np.asarray(points, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if pts.ndim != 2 or pts.shape[1] != 2:
        return None
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) == 0:
        return None
    if pts.min() >= 0.0 and pts.max() <= 1.5:  # normalised coordinates
        pts = pts * np.array([frame_w, frame_h], dtype=np.float32)
    return pts


def polygon_is_valid(pts: Optional[np.ndarray], min_area_px: float = 16.0) -> bool:
    """>= 4 distinct points whose convex hull has a non-trivial area (i.e. not collinear)."""
    if pts is None or len(pts) < 4:
        return False
    if len(np.unique(np.round(pts, 1), axis=0)) < 4:
        return False
    hull = cv2.convexHull(pts.astype(np.float32))
    return len(hull) >= 3 and cv2.contourArea(hull) >= min_area_px


# --------------------------------------------------------------------------- #
# Corner ordering / fitting
# --------------------------------------------------------------------------- #
def order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points predictably: TL, TR, BR, BL based on polar angle relative to centroid."""
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    if len(pts) != 4:
        return pts

    # Filter invalid/NaN points
    if not np.isfinite(pts).all():
        return pts

    center = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    sorted_indices = np.argsort(angles)
    pts_sorted = pts[sorted_indices]

    # Roll array so Top-Left (smallest sum of X+Y) is first
    sums = pts_sorted[:, 0] + pts_sorted[:, 1]
    tl_idx = int(np.argmin(sums))
    ordered = np.roll(pts_sorted, -tl_idx, axis=0)

    return ordered.astype(np.float32)


def is_valid_quad(quad: np.ndarray, frame_shape: tuple = None, max_area_ratio: float = 5.0) -> bool:
    """Check if a quad is convex and physically plausible."""
    if quad is None or len(quad) != 4:
        return False
    if not np.isfinite(quad).all():
        return False

    quad_int = quad.astype(np.int32)
    if not cv2.isContourConvex(quad_int):
        return False

    area = cv2.contourArea(quad_int)
    if area < 10.0:
        return False

    if frame_shape is not None:
        h, w = frame_shape[:2]
        if area > (w * h * max_area_ratio):
            return False

    return True

def fit_quad(pts: np.ndarray) -> Tuple[np.ndarray, str]:
    """
    Reduce an arbitrary polygon to 4 ordered corners.

    Order of preference: hull that already has 4 vertices -> approxPolyDP quad that keeps
    >= 85 % of the hull area -> minimum-area rotated rectangle (fallback).
    """
    hull = cv2.convexHull(pts.astype(np.float32))
    hull_pts = hull.reshape(-1, 2)
    hull_area = cv2.contourArea(hull)

    if len(hull_pts) == 4:
        return order_corners(hull_pts), "polygon_4_points"

    perimeter = cv2.arcLength(hull, True)
    for eps in np.linspace(0.005, 0.10, 20):
        approx = cv2.approxPolyDP(hull, eps * perimeter, True)
        if len(approx) == 4:
            quad = approx.reshape(4, 2)
            if cv2.contourArea(quad.astype(np.float32)) >= 0.85 * hull_area:
                return order_corners(quad), "approx_poly_dp"
            break
        if len(approx) < 4:
            break

    box = cv2.boxPoints(cv2.minAreaRect(hull))
    return order_corners(box), "min_area_rect"


# --------------------------------------------------------------------------- #
# Measurements & masks
# --------------------------------------------------------------------------- #
def quad_dimensions(quad: np.ndarray) -> Tuple[float, float]:
    """Mean (width, height) of an ordered TL, TR, BR, BL quad."""
    w = (np.linalg.norm(quad[1] - quad[0]) + np.linalg.norm(quad[2] - quad[3])) / 2.0
    h = (np.linalg.norm(quad[3] - quad[0]) + np.linalg.norm(quad[2] - quad[1])) / 2.0
    return float(w), float(h)


def quad_dimensions_from_mask(mask: np.ndarray) -> Tuple[float, float]:
    """(width, height) of a binary mask's bounding box."""
    _, _, w, h = cv2.boundingRect(mask)
    return float(w), float(h)


def quad_to_mask(quad: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """Filled quad as a uint8 mask (255 inside) with the frame's height/width."""
    h, w = shape[:2]
    mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(mask, np.round(quad).astype(np.int32), 255)
    return mask


# --------------------------------------------------------------------------- #
# Ad preparation & perspective warp
# --------------------------------------------------------------------------- #
def prepare_ad(
    ad_bgr: np.ndarray, ad_alpha: np.ndarray, quad: np.ndarray, fit_mode: str = "cover"
) -> Tuple[PreparedAd, Tuple[int, int]]:
    """Crop/resize the ad to the surface's frontal size (aspect-correct with ``fit_mode='cover'``)."""
    qw, qh = quad_dimensions(quad)
    dest_w = int(np.clip(round(qw), 8, 4096))
    dest_h = int(np.clip(round(qh), 8, 4096))
    ah, aw = ad_bgr.shape[:2]

    if fit_mode == "cover":
        target_ar, ad_ar = dest_w / dest_h, aw / ah
        if ad_ar > target_ar:  # ad too wide -> crop left/right
            new_w = max(1, int(round(ah * target_ar)))
            x0 = (aw - new_w) // 2
            ad_bgr, ad_alpha = ad_bgr[:, x0 : x0 + new_w], ad_alpha[:, x0 : x0 + new_w]
        else:  # ad too tall -> crop top/bottom
            new_h = max(1, int(round(aw / target_ar)))
            y0 = (ah - new_h) // 2
            ad_bgr, ad_alpha = ad_bgr[y0 : y0 + new_h], ad_alpha[y0 : y0 + new_h]

    shrinking = ad_bgr.shape[0] * ad_bgr.shape[1] > dest_w * dest_h
    interp = cv2.INTER_AREA if shrinking else cv2.INTER_CUBIC
    bgr = cv2.resize(ad_bgr, (dest_w, dest_h), interpolation=interp)
    alpha = cv2.resize(ad_alpha, (dest_w, dest_h), interpolation=interp)
    return PreparedAd(bgr=bgr, alpha=np.clip(alpha, 0.0, 1.0).astype(np.float32)), (dest_w, dest_h)


def warp_ad(
    prepared: PreparedAd, quad: np.ndarray, frame_shape: Tuple[int, ...]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Warp the prepared ad into ``quad``. Returns (warped_bgr, warped_alpha, 3x3 matrix)."""
    h, w = prepared.bgr.shape[:2]
    fh, fw = frame_shape[:2]
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    matrix = cv2.getPerspectiveTransform(src, quad.astype(np.float32))
    # Replicate borders so no dark fringe appears at the seam; alpha is warped separately.
    warped = cv2.warpPerspective(
        prepared.bgr, matrix, (fw, fh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )
    warped_alpha = cv2.warpPerspective(
        prepared.alpha, matrix, (fw, fh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    return warped, warped_alpha, matrix

# أضفها في نهاية geometry.py

def warp_ad_to_quad(ad_img: np.ndarray, dst_quad: np.ndarray, frame_shape: tuple) -> Tuple[np.ndarray, np.ndarray]:
    """
    إسقاط الإعلان على أركان الشاشة (dst_quad) حتى لو كانت الأركان خارج أبعاد الفريم.
    """
    h_frame, w_frame = frame_shape[:2]
    h_ad, w_ad = ad_img.shape[:2]

    # أركان الإعلان الأصلية
    src_corners = np.array([
        [0, 0],
        [w_ad - 1, 0],
        [w_ad - 1, h_ad - 1],
        [0, h_ad - 1]
    ], dtype=np.float32)

    # dst_quad يحتوي على الإحداثيات الفعلية حتى لو كانت بالسالب أو أكبر من أبعاد الصورة
    dst_corners = dst_quad.reshape(4, 2).astype(np.float32)

    # حساب Homography
    M = cv2.getPerspectiveTransform(src_corners, dst_corners)

    # Warp لأسطح الفريم كاملة
    warped_ad = cv2.warpPerspective(ad_img, M, (w_frame, h_frame), flags=cv2.INTER_LINEAR)

    # إنشاء قناع (Mask)
    mask = np.zeros((h_frame, w_frame), dtype=np.uint8)
    cv2.fillConvexPoly(mask, dst_corners.astype(np.int32), 255)

    return warped_ad, mask