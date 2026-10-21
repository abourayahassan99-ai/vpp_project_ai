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

def _intersect(l1, l2):
    """Intersection of two cv2.fitLine results (vx, vy, x0, y0); None when parallel."""
    (vx1, vy1, x1, y1), (vx2, vy2, x2, y2) = l1, l2
    det = vx1 * (-vy2) - vy1 * (-vx2)
    if abs(det) < 1e-6:
        return None
    t = ((x2 - x1) * (-vy2) - (y2 - y1) * (-vx2)) / det
    return np.array([x1 + t * vx1, y1 + t * vy1], np.float32)


def refine_quad_by_lines(contour: np.ndarray, quad: np.ndarray, tol_ratio: float = 0.06) -> np.ndarray:
    """
    Re-fit each side of ``quad`` as a robust line through the contour points that lie near it, then
    intersect neighbouring lines. This yields true trapezoids (perspective) instead of a rotated
    rectangle, and removes corner error caused by rounded segmentation masks.
    Falls back to ``quad`` when a side has too little support or the result is implausible.
    """
    pts = np.asarray(contour, np.float32).reshape(-1, 2)
    quad = np.asarray(quad, np.float32).reshape(4, 2)
    diag = float(np.linalg.norm(quad.max(0) - quad.min(0)))
    tol = max(2.0, tol_ratio * diag)
    # densify contour so long straight edges have support along their whole length
    dense = []
    for i in range(len(pts)):
        a, b = pts[i], pts[(i + 1) % len(pts)]
        n = max(2, int(np.linalg.norm(b - a) / 2.0))
        dense.append(a + (b - a) * np.linspace(0, 1, n, endpoint=False)[:, None])
    dense = np.concatenate(dense)
    lines = []
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        ab = b - a
        L = float(np.linalg.norm(ab))
        if L < 1e-3:
            return quad
        u = ab / L
        rel = dense - a
        t = rel @ u
        d = np.abs(rel[:, 0] * u[1] - rel[:, 1] * u[0])
        sel = dense[(d < tol) & (t > 0.1 * L) & (t < 0.9 * L)]
        if len(sel) < 6:
            return quad
        lines.append(cv2.fitLine(sel, cv2.DIST_HUBER, 0, 0.01, 0.01).reshape(-1))
    out = []
    for i in range(4):
        p = _intersect(lines[i - 1], lines[i])  # corner i = side (i-1) ∩ side i
        if p is None:
            return quad
        out.append(p)
    out = np.array(out, np.float32)
    # reject if a corner moved implausibly far
    if np.linalg.norm(out - quad, axis=1).max() > 0.25 * diag:
        return quad
    return out


def fit_quad(pts: np.ndarray) -> Tuple[np.ndarray, str]:
    """
    Reduce an arbitrary polygon to 4 ordered corners.

    Preference: hull with 4 vertices -> approxPolyDP quad keeping >= 85 % of the hull area, refined by
    robust line fitting -> minimum-area rectangle, also line-refined (fallback).
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
                q = order_corners(quad)
                return order_corners(refine_quad_by_lines(hull_pts, q)), "approx_poly_dp+lines"
            break
        if len(approx) < 4:
            break

    box = order_corners(cv2.boxPoints(cv2.minAreaRect(hull)))
    return order_corners(refine_quad_by_lines(hull_pts, box, tol_ratio=0.04)), "min_area_rect+lines"


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


# =========================================================================== #
# Planar-surface helpers used by the tracker / compositor
# =========================================================================== #
def unit_square() -> np.ndarray:
    return np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])


def adjust_quad(quad: np.ndarray, inset: float) -> np.ndarray:
    """
    Grow (inset < 0, "overscan") or shrink (inset > 0) the quad *in the surface plane*, so the change
    stays perspective-correct. ``inset`` is a fraction of the quad's width/height on each side.
    """
    quad = np.asarray(quad, np.float32).reshape(4, 2)
    if abs(inset) < 1e-6:
        return quad
    H = cv2.getPerspectiveTransform(unit_square(), quad)
    pts = np.float32([[inset, inset], [1 - inset, inset], [1 - inset, 1 - inset], [inset, 1 - inset]])
    return cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H).reshape(4, 2)


def quad_inside_ratio(quad: np.ndarray, frame_shape: Tuple[int, ...]) -> float:
    """Fraction of the quad's area that lies inside the frame (supersampled polygon clip)."""
    h, w = frame_shape[:2]
    q = np.asarray(quad, np.float32).reshape(4, 2)
    area = abs(cv2.contourArea(q))
    if area < 1.0:
        return 0.0
    x0, y0 = np.floor(q.min(0)).astype(int)
    x1, y1 = np.ceil(q.max(0)).astype(int)
    x0c, y0c, x1c, y1c = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    if x1c <= x0c or y1c <= y0c:
        return 0.0
    # rasterise at reduced resolution inside the bounding box for speed
    scale = min(1.0, 128.0 / max(x1 - x0, y1 - y0, 1))
    bw, bh = max(2, int((x1 - x0) * scale)), max(2, int((y1 - y0) * scale))
    mask = np.zeros((bh, bw), np.uint8)
    pts = ((q - [x0, y0]) * scale).astype(np.int32)
    cv2.fillConvexPoly(mask, pts, 255)
    cx0, cy0 = int((x0c - x0) * scale), int((y0c - y0) * scale)
    cx1, cy1 = int(np.ceil((x1c - x0) * scale)), int(np.ceil((y1c - y0) * scale))
    inside = np.count_nonzero(mask[cy0:cy1, cx0:cx1])
    total = max(np.count_nonzero(mask), 1)
    return float(inside) / float(total)


def quad_is_plausible(quad: np.ndarray, frame_shape: Tuple[int, ...], max_area_ratio: float = 6.0,
                      min_angle: float = 25.0, max_aspect: float = 8.0) -> bool:
    """Convex, finite, sane interior angles / aspect, not absurdly larger than the frame."""
    q = np.asarray(quad, np.float32).reshape(4, 2)
    if not np.isfinite(q).all():
        return False
    if not is_valid_quad(q, frame_shape, max_area_ratio):
        return False
    for i in range(4):
        a, b, c = q[i - 1] - q[i], q[(i + 1) % 4] - q[i], None
        cosang = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
        ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
        if ang < min_angle or ang > 180 - min_angle:
            return False
    w, h = quad_dimensions(q)
    return min(w, h) > 2 and max(w, h) / max(min(w, h), 1e-6) < max_aspect


def quad_roi(quad: np.ndarray, frame_shape: Tuple[int, ...], pad: int = 3) -> Optional[Tuple[int, int, int, int]]:
    """Integer (x0, y0, x1, y1) of the quad's bounding box clipped to the frame, or None if off-screen."""
    h, w = frame_shape[:2]
    q = np.asarray(quad, np.float32).reshape(4, 2)
    x0, y0 = np.floor(q.min(0)).astype(int) - pad
    x1, y1 = np.ceil(q.max(0)).astype(int) + pad
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return int(x0), int(y0), int(x1), int(y1)


def quad_aa_mask(quad: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
    """Anti-aliased (sub-pixel accurate) float32 coverage mask of the quad inside ``roi``."""
    x0, y0, x1, y1 = roi
    shift = 4
    pts = np.round((np.asarray(quad, np.float32) - [x0, y0]) * (1 << shift)).astype(np.int32)
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillConvexPoly(m, pts, 255, lineType=cv2.LINE_AA, shift=shift)
    return m.astype(np.float32) / 255.0


def warp_plane_to_roi(src: np.ndarray, quad: np.ndarray, roi: Tuple[int, int, int, int],
                      interp: int = cv2.INTER_LINEAR) -> np.ndarray:
    """
    Warp a plane-space image (ad, occlusion mask...) so its full rectangle lands on ``quad``,
    rendering only the ``roi`` window. Borders are replicated so no dark fringe can appear;
    geometric coverage must come from ``quad_aa_mask``.
    Source corners are pixel *edges* (0..w, 0..h) so the quad maps to the true image rectangle.
    """
    x0, y0, x1, y1 = roi
    h, w = src.shape[:2]
    srcq = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    M = cv2.getPerspectiveTransform(srcq, np.asarray(quad, np.float32).reshape(4, 2))
    T = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], np.float64)
    return cv2.warpPerspective(src, T @ M, (x1 - x0, y1 - y0), flags=interp, borderMode=cv2.BORDER_REPLICATE)



def estimate_plane_aspect(quad: np.ndarray, frame_shape: Tuple[int, ...]) -> float:
    """
    True width/height ratio of the rectangle that projects to ``quad`` (TL, TR, BR, BL), recovered from
    perspective (Zhang & He, "Whiteboard scanning"), assuming square pixels and the principal point at
    the image centre. Falls back to the mean edge-length ratio when the quad is (near) frontal, where
    perspective carries no information, or when the estimate is not finite.
    """
    q = np.asarray(quad, np.float64).reshape(4, 2)
    w0, h0 = quad_dimensions(q.astype(np.float32))
    fallback = float(np.clip(w0 / max(h0, 1e-6), 0.2, 8.0))
    fh, fw = frame_shape[:2]
    u0, v0 = fw / 2.0, fh / 2.0
    m1, m2, m4, m3 = [np.array([x, y, 1.0]) for x, y in (q[0], q[1], q[2], q[3])]  # TL, TR, BR, BL
    try:
        k2 = np.dot(np.cross(m1, m4), m3) / np.dot(np.cross(m2, m4), m3)
        k3 = np.dot(np.cross(m1, m4), m2) / np.dot(np.cross(m3, m4), m2)
        n2, n3 = k2 * m2 - m1, k3 * m3 - m1
        if abs(n2[2] * n3[2]) < 1e-9:
            return fallback
        f2 = -1.0 / (n2[2] * n3[2]) * (
            (n2[0] * n3[0] - (n2[0] * n3[2] + n2[2] * n3[0]) * u0 + n2[2] * n3[2] * u0 ** 2)
            + (n2[1] * n3[1] - (n2[1] * n3[2] + n2[2] * n3[1]) * v0 + n2[2] * n3[2] * v0 ** 2)
        )
        if not np.isfinite(f2) or f2 <= 0:
            return fallback
        f = np.sqrt(f2)
        A = np.array([[f, 0, u0], [0, f, v0], [0, 0, 1.0]])
        Ai = np.linalg.inv(A)
        M = Ai.T @ Ai
        r2 = float(n2 @ M @ n2) / float(n3 @ M @ n3)
        if not np.isfinite(r2) or r2 <= 0:
            return fallback
        ratio = float(np.sqrt(r2))
        # sanity: a perspective-derived estimate far from the apparent ratio means a bad fit
        if not (0.4 * fallback <= ratio <= 2.5 * fallback) or not (0.2 <= ratio <= 8.0):
            return fallback
        return ratio
    except (ZeroDivisionError, np.linalg.LinAlgError, FloatingPointError):
        return fallback
