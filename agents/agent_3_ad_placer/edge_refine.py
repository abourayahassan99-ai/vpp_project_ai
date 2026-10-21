"""
Edge detection for planar display surfaces (Agent 3).

Two jobs, one detector
----------------------
1. ``snap_quad_to_display``  (once, at the anchor frame): Agent 1's segmentation polygon is loose (10-12 px
   outside the TV in the supplied footage). The 4 sides are re-fitted to the real image edges of the *active
   display area*.
2. ``EdgeRefiner``  (every frame): keeps each tracked side on the same physical edge, using the position the
   edge had relative to the quad in the anchor frame.

The detector looks, along the outward normal of every side, for the INNERMOST significant dark -> brighter
step when scanning outwards from the glass. Two kinds of step occur on a dark (switched-off) TV:

  * outer step : glass/bezel -> bright wall. The bezel itself is a thin black rim that is not separable from
                 the glass, so a small bezel allowance (``bezel_ratio`` x the surface's smaller side) is applied.
  * inner step : glass -> a lighter bezel strip (the bottom of the TV in the supplied footage). That step IS the
                 active-area boundary, so only ~1 px is applied.

The kind is decided from the brightness *outside* the step relative to the wall, not from a fixed per-side rule.
Sides whose edge cannot be supported (cables, shadows) are inferred from the supported sides.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .geometry import _intersect, quad_is_plausible, quad_dimensions

logger = logging.getLogger(__name__)

_N_SAMPLES = 32


def _outward_normals(quad: np.ndarray) -> np.ndarray:
    e = np.roll(quad, -1, axis=0) - quad
    n = np.stack([e[:, 1], -e[:, 0]], axis=1)  # outward for clockwise (screen) ordering
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)


def _side_points(quad: np.ndarray, i: int, shape: Tuple[int, int], margin: int) -> np.ndarray:
    a, b = quad[i], quad[(i + 1) % 4]
    t = np.linspace(0.10, 0.90, _N_SAMPLES, dtype=np.float32)
    p = a[None, :] + (b - a)[None, :] * t[:, None]
    h, w = shape
    ok = (p[:, 0] > margin) & (p[:, 0] < w - 1 - margin) & (p[:, 1] > margin) & (p[:, 1] < h - 1 - margin)
    return p[ok]


def _profiles(gray: np.ndarray, pts: np.ndarray, normal: np.ndarray, radius: int) -> np.ndarray:
    offs = np.arange(-radius, radius + 1, dtype=np.float32)
    xs = (pts[:, 0:1] + normal[0] * offs[None, :]).astype(np.float32)
    ys = (pts[:, 1:2] + normal[1] * offs[None, :]).astype(np.float32)
    return cv2.remap(gray, xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _first_steps(prof: np.ndarray, wall_ref: float, dark_frac: float = 0.55):
    """
    Innermost significant dark->brighter step for every profile row.
    Returns arrays (pos, step, outside_mean, rise): ``pos`` is the mid-ramp position (sub-pixel index) or NaN;
    ``rise`` is where the profile starts leaving the dark plateau (the start of the transition).
    """
    K, M = prof.shape
    cs = np.concatenate([np.zeros((K, 1), np.float32), np.cumsum(prof, axis=1)], axis=1)
    k = np.arange(3, M - 3)
    inside = (cs[:, k] - cs[:, k - 3]) / 3.0            # mean of prof[k-3 .. k-1]
    outside = (cs[:, k + 4] - cs[:, k + 1]) / 3.0       # mean of prof[k+1 .. k+3]
    step = outside - inside
    min_step = max(5.0, 0.07 * wall_ref)
    cand = (step >= min_step) & (inside <= dark_frac * wall_ref)
    pos = np.full(K, np.nan)
    rise = np.full(K, np.nan)
    st = np.zeros(K)
    out = np.zeros(K)
    for r in range(K):
        idx = np.flatnonzero(cand[r])
        if len(idx) == 0:
            continue
        j = idx[0]
        while j + 1 < len(step[r]) and cand[r, j + 1] and step[r, j + 1] >= step[r, j]:
            j += 1                                      # climb to the local maximum of this step
        d = step[r, max(j - 1, 0):j + 2]
        sub = 0.0
        if len(d) == 3 and (d[0] - 2 * d[1] + d[2]) < -1e-6:
            sub = float(np.clip(0.5 * (d[0] - d[2]) / (d[0] - 2 * d[1] + d[2]), -0.5, 0.5))
        kk = k[j]
        pos[r] = kk + sub
        st[r] = step[r, j]
        out[r] = outside[r, j]
        lo = max(0, kk - 10)
        plateau = float(np.median(prof[r, lo:max(lo + 1, kk - 5)]))
        thr = plateau + max(2.5, 0.06 * (outside[r, j] - plateau))
        m = kk
        while m > 0 and prof[r, m] > thr:
            m -= 1
        if m < kk and prof[r, m + 1] > prof[r, m]:
            frac = (thr - prof[r, m]) / max(prof[r, m + 1] - prof[r, m], 1e-6)
            rise[r] = m + float(np.clip(frac, 0, 1))
        else:
            rise[r] = float(m)
    return pos, st, out, rise


def _fit_line(points: np.ndarray, tol: float = 1.6, min_frac: float = 0.45) -> Optional[Tuple[np.ndarray, int]]:
    """RANSAC line through ``points`` (handles tilted edges and cable/shadow outliers); refit on the inliers."""
    pts = np.asarray(points, np.float32)
    n = len(pts)
    if n < 8:
        return None
    rng = np.random.default_rng(7)
    best = None
    for _ in range(80):
        i, j = rng.choice(n, 2, replace=False)
        d = pts[j] - pts[i]
        L = float(np.linalg.norm(d))
        if L < 1e-3:
            continue
        d /= L
        res = np.abs((pts[:, 0] - pts[i, 0]) * d[1] - (pts[:, 1] - pts[i, 1]) * d[0])
        inl = res < tol
        c = int(inl.sum())
        if best is None or c > best[0]:
            best = (c, inl)
    if best is None or best[0] < max(8, min_frac * n):
        return None
    ln = cv2.fitLine(pts[best[1]], cv2.DIST_HUBER, 0, 0.01, 0.01).reshape(-1)
    res = np.abs((pts[:, 0] - ln[2]) * ln[1] - (pts[:, 1] - ln[3]) * ln[0])
    keep = res < tol
    if keep.sum() >= 8:
        ln = cv2.fitLine(pts[keep], cv2.DIST_HUBER, 0, 0.01, 0.01).reshape(-1)
    return ln, int(keep.sum())


def _wall_value(pr: np.ndarray) -> float:
    """Wall brightness from profiles: the bright end of the OUTER half. Robust when the quad starts inside the
    screen, so that its nearest 'outside' samples are still glass."""
    half = pr[:, pr.shape[1] // 2 + pr.shape[1] // 5:]
    return float(np.median(np.percentile(half, 85, axis=1)))


def _tails(gray, quad, normals, radius, margin=2):
    """Median brightness just outside each side (used to estimate the wall brightness).

    Profiles that run past the image border read the replicated border pixel, which is the nearest visible
    wall pixel, so thin strips of wall (TV close to the frame edge) still give a usable value."""
    vals = []
    for i in range(4):
        p = _side_points(quad, i, gray.shape, 3)
        if len(p) < 6:
            continue
        pr = _profiles(gray, p, normals[i], radius)
        vals.append(_wall_value(pr))
    return vals


def _side_walls(gray, quad, normals, radius):
    """Wall brightness next to EACH side (NaN only when that side has no sample points inside the frame)."""
    out = np.full(4, np.nan)
    for i in range(4):
        p = _side_points(quad, i, gray.shape, 3)
        if len(p) < 6:
            continue
        out[i] = _wall_value(_profiles(gray, p, normals[i], radius))
    return out


def _wall_for(side_walls, i, fallback):
    return float(side_walls[i]) if np.isfinite(side_walls[i]) else float(fallback)


def detect_side(gray, quad, i, normals, radius, wall_ref, bezel_px: float = 0.0, dark_frac: float = 0.55):
    """
    Active-display boundary of side ``i`` near ``quad``. Returns dict(offsets, base, steps, outside, strong) or None.
    ``offsets`` are signed distances (px) along the outward normal from the quad side to the boundary:
    the start of the rise out of the dark plateau; on sharp, strong (wall) edges it is additionally kept at least
    ``bezel_px`` inside the mid-ramp, because a thin black rim is not separable from the glass there.
    """
    p = _side_points(quad, i, gray.shape, 3)
    if len(p) < 10:
        return None
    pr = _profiles(gray, p, normals[i], radius)
    pos, st, out, rise = _first_steps(pr, wall_ref, dark_frac)
    ok = ~np.isnan(pos)
    if ok.sum() < 10:
        return None
    strong = float(np.median(st[ok])) >= 0.45 * wall_ref
    mid = pos[ok] - radius
    boundary = rise[ok] - radius - 0.5
    if strong and bezel_px > 0:
        boundary = np.minimum(boundary, mid - bezel_px)
    return dict(offsets=boundary, base=p[ok], steps=st[ok], outside=out[ok], strong=strong)


def _line_through(side_pts: np.ndarray):
    return _fit_line(side_pts)


def snap_quad_to_display(
    frame_bgr: np.ndarray,
    quad: np.ndarray,
    bezel_ratio: float = 0.018,
    search_ratio: float = 0.20,
) -> Tuple[Optional[np.ndarray], dict]:
    """
    Snap a loose quad to the active display area of a dark screen. Returns (quad or None, info).
    ``None`` means the evidence was insufficient / implausible and the caller must keep its own quad.
    """
    info: dict = {"sides": {}, "ok": False}
    q0 = np.asarray(quad, np.float32).reshape(4, 2)
    h, w = frame_bgr.shape[:2]
    gray = cv2.GaussianBlur(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), (0, 0), 1.0).astype(np.float32)
    qw, qh = quad_dimensions(q0)
    size = min(qw, qh)
    radius = int(np.clip(search_ratio * size, 10, 48))
    normals = _outward_normals(q0)

    tails = _tails(gray, q0, normals, radius)
    if len(tails) < 1:
        info["reason"] = "surface mostly outside the frame"
        return None, info
    # with few visible sides the median would mix in dark (cable / shadow) sides; the brightest tail is the wall
    wall_ref = float(np.median(tails)) if len(tails) >= 3 else float(max(tails))
    side_walls = _side_walls(gray, q0, normals, radius)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(mask, np.round(q0).astype(np.int32), 255)
    inner = float(np.median(gray[mask > 0]))
    if inner > 0.6 * wall_ref or wall_ref < 40:
        info["reason"] = f"screen is not darker than its surroundings (inside {inner:.0f}, wall {wall_ref:.0f})"
        return None, info

    bezel = bezel_ratio * size
    disp = [None] * 4        # signed displacement of the fitted ACTIVE-AREA side from the input side (px, +outward)
    lines = [None] * 4
    kinds = [""] * 4
    for i in range(4):
        d = detect_side(gray, q0, i, normals, radius, _wall_for(side_walls, i, wall_ref), bezel)
        if d is None:
            continue
        kinds[i] = "wall-edge" if d["strong"] else "soft-edge"
        pts = d["base"] + normals[i][None, :] * d["offsets"][:, None]
        fit = _fit_line(pts, tol=2.2)
        if fit is None:
            info["sides"][i] = dict(kind=kinds[i], supported=False, n=int(len(pts)))
            continue
        ln, n_in = fit
        if n_in < 10:
            info["sides"][i] = dict(kind=kinds[i], supported=False, n=n_in)
            continue
        # angle sanity vs the input side
        a, b = q0[i], q0[(i + 1) % 4]
        dvec = (b - a) / (np.linalg.norm(b - a) + 1e-9)
        ang = np.degrees(np.arccos(np.clip(abs(ln[0] * dvec[0] + ln[1] * dvec[1]), 0, 1)))
        if ang > 7.0:
            info["sides"][i] = dict(kind=kinds[i], supported=False, n=n_in, angle=round(float(ang), 1))
            continue
        if ln[0] * dvec[0] + ln[1] * dvec[1] < 0:
            ln[0], ln[1] = -ln[0], -ln[1]
        mid = (a + b) / 2
        disp[i] = float(np.dot(np.array([ln[2], ln[3]]) - mid, normals[i]))
        lines[i] = ln
        info["sides"][i] = dict(kind=kinds[i], supported=True, n=n_in, shift_px=round(disp[i], 1),
                                 step=round(float(np.median(d["steps"])), 1))

    good = [i for i in range(4) if lines[i] is not None]
    if len(good) < 3:
        info["reason"] = f"only {len(good)}/4 sides have a reliable edge"
        return None, info
    # unsupported side: keep the input side moved by the median shift of the supported ones
    for i in range(4):
        if lines[i] is None:
            shift = float(np.median([disp[j] for j in good]))
            a, b = q0[i], q0[(i + 1) % 4]
            mid = (a + b) / 2 + normals[i] * shift
            dvec = (b - a) / (np.linalg.norm(b - a) + 1e-9)
            lines[i] = np.array([dvec[0], dvec[1], mid[0], mid[1]], np.float32)
            info["sides"].setdefault(i, {})["inferred_shift_px"] = round(shift, 1)
    out = []
    for i in range(4):
        pnt = _intersect(lines[i - 1], lines[i])
        if pnt is None:
            info["reason"] = "degenerate intersection"
            return None, info
        out.append(pnt)
    q1 = np.array(out, np.float32)
    # validation
    if not quad_is_plausible(q1, frame_bgr.shape):
        info["reason"] = "snapped quad is not a plausible convex quadrilateral"
        return None, info
    move = float(np.linalg.norm(q1 - q0, axis=1).max())
    area_ratio = abs(cv2.contourArea(q1)) / max(abs(cv2.contourArea(q0)), 1e-6)
    info.update(max_corner_move=round(move, 1), area_ratio=round(area_ratio, 3), wall=round(wall_ref, 1))
    if move > 0.35 * size or not (0.55 <= area_ratio <= 1.05):
        info["reason"] = f"snap moved the quad too far (move {move:.1f}px, area x{area_ratio:.2f})"
        return None, info
    info["ok"] = True
    return q1, info


# --------------------------------------------------------------------------- #
# Per-frame refinement
# --------------------------------------------------------------------------- #
class EdgeRefiner:
    """Keeps tracked quad sides on the real display edges, using the anchor frame to calibrate each side."""

    def __init__(self, anchor_gray: np.ndarray, anchor_quad: np.ndarray, search_ratio: float = 0.10,
                 bezel_ratio: float = 0.018):
        self.search_ratio = search_ratio
        self.bezel_ratio = bezel_ratio
        self.a_quad = np.asarray(anchor_quad, np.float32).reshape(4, 2)
        self.offset: List[float] = [0.0] * 4
        self.strength: List[float] = [0.0] * 4
        self.valid: List[bool] = [False] * 4
        g = cv2.GaussianBlur(anchor_gray, (0, 0), 1.0).astype(np.float32)
        self.enabled = False
        qw, qh = quad_dimensions(self.a_quad)
        radius = int(np.clip(0.2 * min(qw, qh), 10, 48))      # same search range as snap_quad_to_display
        normals = _outward_normals(self.a_quad)
        tails = _tails(g, self.a_quad, normals, radius)
        if len(tails) < 1:
            return
        wall = float(np.median(tails)) if len(tails) >= 3 else float(max(tails))
        side_walls = _side_walls(g, self.a_quad, normals, radius)
        m = np.zeros(g.shape, np.uint8)
        cv2.fillConvexPoly(m, np.round(self.a_quad).astype(np.int32), 255)
        if float(np.median(g[m > 0])) > 0.6 * wall or wall < 40:
            return                                   # not a dark display: nothing to calibrate against
        for i in range(4):
            d = detect_side(g, self.a_quad, i, normals, radius, _wall_for(side_walls, i, wall),
                            self.bezel_ratio * min(qw, qh))
            if d is None:
                continue
            fit = _fit_line(d["base"] + normals[i][None, :] * d["offsets"][:, None], tol=2.2)
            if fit is None and not (len(d["offsets"]) >= 12 and float(np.std(d["offsets"])) < 3.0):
                continue                             # genuinely inconsistent (cables, shadows): do not use this side
            self.offset[i] = float(np.median(d["offsets"]))
            self.strength[i] = float(np.median(d["steps"]))
            self.valid[i] = True
        self.enabled = sum(self.valid) >= 2

    @property
    def usable_sides(self) -> int:
        return sum(self.valid)

    def refine(self, gray: np.ndarray, quad: np.ndarray) -> Tuple[np.ndarray, int]:
        """Returns (refined quad, number of sides corrected). Falls back to ``quad`` unchanged."""
        return self._refine_once(gray, quad)

    def _refine_once(self, gray: np.ndarray, quad: np.ndarray) -> Tuple[np.ndarray, int]:
        quad = np.asarray(quad, np.float32).reshape(4, 2)
        if not self.enabled:
            return quad, 0
        g = cv2.GaussianBlur(gray, (0, 0), 1.0).astype(np.float32) if gray.dtype != np.float32 else gray
        sizes = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)
        a_sizes = np.linalg.norm(np.roll(self.a_quad, -1, axis=0) - self.a_quad, axis=1)
        radius = int(np.clip(self.search_ratio * float(min(sizes.min(), np.linalg.norm(quad[0] - quad[3]))), 6, 36))
        normals = _outward_normals(quad)
        tails = _tails(g, quad, normals, radius)
        if len(tails) < 1:
            return quad, 0
        wall = float(np.median(tails)) if len(tails) >= 3 else float(max(tails))
        side_walls = _side_walls(g, quad, normals, radius)
        cur_w, cur_h = quad_dimensions(quad)
        bezel_px = self.bezel_ratio * min(cur_w, cur_h)
        lines, corrected = [], 0
        for i in range(4):
            a, b = quad[i], quad[(i + 1) % 4]
            dvec = (b - a) / (np.linalg.norm(b - a) + 1e-9)
            line = np.array([dvec[0], dvec[1], a[0], a[1]], np.float32)
            if self.valid[i] and sizes[i] > 12:
                d = detect_side(g, quad, i, normals, radius, _wall_for(side_walls, i, wall), bezel_px, dark_frac=0.85)
                if d is not None and float(np.median(d["steps"])) >= 0.4 * self.strength[i]:
                    scale = sizes[i] / max(a_sizes[i], 1e-6)
                    s = d["offsets"] - self.offset[i] * scale
                    fit = _fit_line(d["base"] + normals[i][None, :] * s[:, None], tol=1.8)
                    if fit is not None and fit[1] >= max(8, 0.4 * len(s)):
                        ln = fit[0]
                        ang = np.degrees(np.arccos(np.clip(abs(ln[0] * dvec[0] + ln[1] * dvec[1]), 0, 1)))
                        if ang < 12.0:
                            if ln[0] * dvec[0] + ln[1] * dvec[1] < 0:
                                ln[0], ln[1] = -ln[0], -ln[1]
                            line, corrected = ln, corrected + 1
            lines.append(line)
        if corrected == 0:
            return quad, 0
        out = []
        for i in range(4):
            pnt = _intersect(lines[i - 1], lines[i])
            if pnt is None:
                return quad, 0
            out.append(pnt)
        out = np.array(out, np.float32)
        if np.linalg.norm(out - quad, axis=1).max() > 1.6 * radius + 0.1 * float(sizes.max()):
            return quad, 0
        return out, corrected
