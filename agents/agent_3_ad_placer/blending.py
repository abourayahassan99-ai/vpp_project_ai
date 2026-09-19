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


from typing import Any, Dict, Tuple
import cv2
import numpy as np
from .schemas import AdPlacerConfig, PlacementError, PreparedAd


def composite(
    frame: np.ndarray, prepared: PreparedAd, quad: np.ndarray, mode: str, cfg: AdPlacerConfig
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Warp the prepared ad into ``quad`` and blend it into ``frame`` cleanly."""
    from .geometry import order_corners
    
    quad = order_corners(quad)
    h, w = frame.shape[:2]

    # Calculate transformation
    ad_h, ad_w = prepared.bgr.shape[:2]
    src_pts = np.array([[0, 0], [ad_w, 0], [ad_w, ad_h], [0, ad_h]], dtype=np.float32)
    
    matrix = cv2.getPerspectiveTransform(src_pts, quad.astype(np.float32))
    warped = cv2.warpPerspective(prepared.bgr, matrix, (w, h))
    warped_alpha = cv2.warpPerspective(prepared.alpha, matrix, (w, h))

    mask = (warped_alpha > 0.1).astype(np.uint8) * 255

    # Safe return if surface is off-screen or invisible
    if np.count_nonzero(mask) == 0:
        return frame, {
            "transformation_matrix": matrix.tolist(),
            "blending": {"mode": mode, "status": "off_screen"}
        }

    # Blending
    inv_alpha = 1.0 - warped_alpha[..., None]
    out_frame = (frame * inv_alpha + warped * warped_alpha[..., None]).astype(np.uint8)

    return out_frame, {
        "transformation_matrix": matrix.tolist(),
        "blending": {"mode": mode, "status": "placed"}
    }