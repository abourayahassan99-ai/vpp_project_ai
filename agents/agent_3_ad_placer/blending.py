"""
Compositing for Agent 3: feathered alpha blending, Poisson seamless cloning and the default
"hybrid" edge-band blend.

Modes
-----
alpha         Feathered alpha composite (Gaussian-blurred, slightly eroded mask).
normal_clone  ``cv2.seamlessClone(NORMAL_CLONE)`` over the whole surface.
mixed_clone   ``cv2.seamlessClone(MIXED_CLONE)`` over the whole surface.
hybrid        Poisson (mixed clone) only decides the seam: its weight is 1 on / outside the mask
              border and fades to 0 ``clone_edge_band_ratio * min(w, h)`` pixels inside, so the ad
              interior keeps its lighting-corrected colours.

Poisson solves for colours that agree with the surroundings, so the explicit ``normal_clone`` /
``mixed_clone`` modes will pull a bright ad towards a dark bezel; that is why ``hybrid`` is default.
If Poisson cloning raises, every clone mode falls back to feathered alpha blending.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from .geometry import quad_dimensions, quad_dimensions_from_mask, quad_to_mask, warp_ad
from .schemas import AdPlacerConfig, PlacementError, PreparedAd

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Masks
# --------------------------------------------------------------------------- #
def feather_params(quad: np.ndarray, cfg: AdPlacerConfig) -> Tuple[float, int]:
    """(Gaussian sigma, erosion in px) scaled to the surface size."""
    qw, qh = quad_dimensions(quad)
    sigma = float(np.clip(cfg.feather_ratio * min(qw, qh), 1.0, 12.0))
    return sigma, int(round(sigma))


def build_masks(quad: np.ndarray, shape: Tuple[int, ...], sigma: float, erode_px: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (binary core mask uint8, feathered alpha float32)."""
    poly = quad_to_mask(quad, shape)
    core = poly
    if erode_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1))
        eroded = cv2.erode(poly, kernel)
        core = eroded if eroded.any() else poly
    soft = cv2.GaussianBlur(core.astype(np.float32) / 255.0, (0, 0), sigma)
    return core, soft


# --------------------------------------------------------------------------- #
# Poisson cloning
# --------------------------------------------------------------------------- #
def poisson_clone(frame: np.ndarray, warped: np.ndarray, core: np.ndarray, flag: int) -> np.ndarray:
    """seamlessClone on the mask's bounding box, padded so masks touching the border are safe."""
    pad = 16
    f = cv2.copyMakeBorder(frame, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    s = cv2.copyMakeBorder(warped, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    m = cv2.copyMakeBorder(core, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    x, y, w, h = cv2.boundingRect(m)
    if w < 3 or h < 3:
        raise ValueError("mask too small for Poisson cloning")
    x0, y0, x1, y1 = x - 2, y - 2, x + w + 2, y + h + 2  # margin stays inside the padding
    cloned = cv2.seamlessClone(
        np.ascontiguousarray(s[y0:y1, x0:x1]),
        f,
        np.ascontiguousarray(m[y0:y1, x0:x1]),
        (x + w // 2, y + h // 2),
        flag,
    )
    return np.ascontiguousarray(cloned[pad:-pad, pad:-pad])


# --------------------------------------------------------------------------- #
# Blending
# --------------------------------------------------------------------------- #
def blend(
    frame: np.ndarray,
    warped: np.ndarray,
    alpha: np.ndarray,
    core: np.ndarray,
    mode: str,
    cfg: AdPlacerConfig,
) -> Tuple[np.ndarray, str, Optional[str]]:
    """Blend ``warped`` into ``frame``. Returns (image, mode actually used, optional note)."""
    a3 = alpha[..., None]
    alpha_out = np.clip(
        frame.astype(np.float32) * (1.0 - a3) + warped.astype(np.float32) * a3 + 0.5, 0, 255
    ).astype(np.uint8)
    if mode == "alpha":
        return alpha_out, "alpha", None

    flag = cv2.NORMAL_CLONE if mode == "normal_clone" else cv2.MIXED_CLONE
    try:
        cloned = poisson_clone(frame, warped, core, flag)
    except (cv2.error, ValueError) as exc:
        msg = f"Poisson cloning failed ({str(exc).splitlines()[0][:120]}); used feathered alpha blending instead."
        logger.warning(msg)
        return alpha_out, "alpha", msg

    if mode != "hybrid":
        # Explicit normal/mixed clone: respect the request (see module docstring for the colour-bleed caveat).
        return cloned, mode, None

    qw, qh = quad_dimensions_from_mask(core)
    band = max(4.0, cfg.clone_edge_band_ratio * min(qw, qh))
    dist = cv2.distanceTransform(core, cv2.DIST_L2, 3)
    w = np.clip(1.0 - dist / band, 0.0, 1.0)[..., None]
    out = cloned.astype(np.float32) * w + alpha_out.astype(np.float32) * (1.0 - w)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8), "hybrid(edge_poisson+alpha)", None


from typing import Any, Dict, Optional, Tuple  # noqa: E402  (kept for the video compositor below)

from .geometry import adjust_quad, quad_aa_mask, quad_dimensions, quad_roi, warp_plane_to_roi  # noqa: E402


def _motion_blur(img: np.ndarray, motion: np.ndarray, shutter: float, max_px: float) -> np.ndarray:
    """Directional blur matching camera motion during the exposure (length = shutter * |motion|)."""
    length = min(float(np.hypot(*motion)) * shutter, max_px)
    if length < 1.5:
        return img
    k = int(round(length)) | 1  # odd kernel size
    kernel = np.zeros((k, k), np.float32)
    ang = np.arctan2(motion[1], motion[0])
    c = k // 2
    cv2.line(kernel, (int(round(c - np.cos(ang) * c)), int(round(c - np.sin(ang) * c))),
             (int(round(c + np.cos(ang) * c)), int(round(c + np.sin(ang) * c))), 1.0, 1)
    kernel /= max(kernel.sum(), 1e-6)
    return cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE)


def _reflection_layer(region: np.ndarray, aa: np.ndarray, quad: np.ndarray, cfg: AdPlacerConfig) -> np.ndarray:
    """
    Light that the glass reflects (tube-light glare, room tint) is still there when the screen shows content, and
    adds to the emitted light. The original frame IS that reflection while the screen is off, so a low-passed copy
    of the original screen pixels (restricted to the screen interior, so no bezel / wall bleeds in) is added to the ad.
    It is (a) capped, (b) only taken from dark screen pixels (a hand or cable in front is bright and is left out)
    and (c) scaled by ``reflection_strength``, so the ad keeps its contrast and readability.
    """
    qw, qh = quad_dimensions(quad)
    sigma = float(np.clip(cfg.reflection_blur_ratio * min(qw, qh), 2.0, 14.0))
    inner = (aa > 0.99).astype(np.float32)
    k = max(3, int(round(0.01 * min(qw, qh))) | 1)
    inner = cv2.erode(inner, np.ones((k, k), np.uint8))
    num = cv2.GaussianBlur(region * inner[..., None], (0, 0), sigma)
    den = cv2.GaussianBlur(inner, (0, 0), sigma)[..., None]
    refl = np.where(den > 0.05, num / np.maximum(den, 1e-3), 0.0)
    luma = refl.mean(axis=2, keepdims=True)
    T = cfg.reflection_dark_threshold
    w = np.clip((T - luma) / (0.35 * T), 0.0, 1.0)            # 1 on a dark screen, 0 where something bright is in front
    add = np.minimum(refl, cfg.reflection_cap) * w * cfg.reflection_strength
    return add.astype(np.float32)


def composite(
    frame: np.ndarray,
    prepared: PreparedAd,
    quad: np.ndarray,
    mode: str,
    cfg: AdPlacerConfig,
    *,
    opacity: float = 1.0,
    occlusion: Optional[np.ndarray] = None,
    motion: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Warp the prepared ad onto ``quad`` (TL, TR, BR, BL, already ordered by the tracker) and blend it
    into ``frame`` -- the per-frame video compositor.

    * ``opacity``   : 0..1 from the tracker (fade in / out, visibility).
    * ``occlusion`` : optional float mask in *plane space* (1 = something is in front of the surface);
                      it is warped with the same homography as the ad.
    * ``motion``    : (dx, dy) corner-centre speed in px/frame, used for motion-blur matching.

    The ad is composited with an anti-aliased quad mask and replicated borders, so there is no dark
    fringe and no black border. ``mode`` is accepted for compatibility; Poisson / hybrid blending is
    for stills (see :func:`blend`) because per-frame Poisson solves flicker in video.
    """
    quad = np.asarray(quad, np.float32).reshape(4, 2)
    draw_quad = adjust_quad(quad, cfg.screen_inset_ratio)
    info: Dict[str, Any] = {"blending": {"mode": "video_alpha", "requested_mode": mode, "status": "placed"}}
    roi = quad_roi(draw_quad, frame.shape)
    if roi is None or opacity <= 0.001 or not np.isfinite(draw_quad).all():
        info["blending"]["status"] = "off_screen" if roi is None else "hidden"
        return frame, info

    x0, y0, x1, y1 = roi
    warped = warp_plane_to_roi(prepared.bgr, draw_quad, roi)
    alpha = warp_plane_to_roi(prepared.alpha, draw_quad, roi)
    if motion is not None and cfg.motion_blur_shutter > 0:
        warped = _motion_blur(warped, np.asarray(motion, np.float32), cfg.motion_blur_shutter, cfg.max_motion_blur_px)

    aa = quad_aa_mask(draw_quad, roi)
    a = aa * alpha * float(opacity)
    warped_f = warped.astype(np.float32)
    if cfg.reflection_strength > 0:
        warped_f = np.clip(warped_f + _reflection_layer(frame[y0:y1, x0:x1].astype(np.float32), aa, draw_quad, cfg),
                           0, 255)
        info["blending"]["reflection"] = True
    if occlusion is not None:
        occ = warp_plane_to_roi(occlusion.astype(np.float32), draw_quad, roi)
        a = a * (1.0 - np.clip(occ, 0.0, 1.0))
        info["blending"]["occluded_fraction"] = round(float((occ > 0.5).mean()), 4)

    region = frame[y0:y1, x0:x1].astype(np.float32)
    out = frame.copy()
    out[y0:y1, x0:x1] = np.clip(region * (1.0 - a[..., None]) + warped_f * a[..., None] + 0.5,
                                0, 255).astype(np.uint8)
    return out, info
