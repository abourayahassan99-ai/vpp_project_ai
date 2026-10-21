"""
Lighting and colour matching for Agent 3 (LAB statistics transfer).

Emissive surfaces (TV / monitor / screen) generate their own light and are usually dark when the
frame was captured, so matching the ad to the surface interior would make it unreadable. For
those types the ambient light is measured from a ring around the surface instead, and the
correction is applied more weakly.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

from .geometry import quad_dimensions, quad_to_mask
from .schemas import AdPlacerConfig, LightingParams, PreparedAd

logger = logging.getLogger(__name__)


def is_emissive(surface_type: str, cfg: AdPlacerConfig) -> bool:
    return surface_type in cfg.emissive_surface_types


def to_lab(bgr_u8: np.ndarray) -> np.ndarray:
    """uint8 BGR -> float32 LAB (L in 0-100, a/b roughly -127..127)."""
    return cv2.cvtColor(bgr_u8.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)


def lab_stats(lab: np.ndarray, mask: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
    """(mean L, std L, mean a, mean b) over ``mask``; None when fewer than 30 pixels are selected."""
    px = lab[mask]
    if px.shape[0] < 30:
        return None
    return float(px[:, 0].mean()), float(px[:, 0].std()), float(px[:, 1].mean()), float(px[:, 2].mean())


def estimate_lighting(
    frame: np.ndarray,
    prepared: PreparedAd,
    quad: np.ndarray,
    surface_type: str,
    cfg: AdPlacerConfig,
) -> LightingParams:
    """Measure the destination lighting and derive the brightness / contrast / colour-temperature shift."""
    poly_bool = quad_to_mask(quad, frame.shape) > 0

    emissive = is_emissive(surface_type, cfg)
    strength = cfg.emissive_lighting_strength if emissive else cfg.lighting_strength

    qw, qh = quad_dimensions(quad)
    ring_px = int(max(5, 0.15 * min(qw, qh)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring_px + 1, 2 * ring_px + 1))
    ring_bool = (cv2.dilate(poly_bool.astype(np.uint8) * 255, kernel) > 0) & ~poly_bool

    frame_lab = to_lab(frame)
    if emissive:
        order = [("surrounding_ring", ring_bool), ("frame_outside_surface", ~poly_bool)]
    else:
        order = [("surface_roi", poly_bool), ("surrounding_ring", ring_bool)]
    ref_name, ref_stats = "none", None
    for name, mask in order:
        ref_stats = lab_stats(frame_lab, mask)
        if ref_stats is not None:
            ref_name = name
            break

    ad_lab = to_lab(prepared.bgr)
    ad_stats = lab_stats(ad_lab, prepared.alpha > 0.5)
    if ref_stats is None or ad_stats is None:
        logger.warning("Could not estimate lighting statistics; leaving ad colours unchanged.")
        return LightingParams(strength=0.0, reference="none")

    t_mean, t_std, t_a, t_b = ref_stats
    s_mean, s_std, s_a, s_b = ad_stats
    lo, hi = cfg.contrast_gain_range
    contrast = float(np.clip(1.0 + strength * (t_std / max(s_std, 1e-3) - 1.0), lo, hi))
    shift = float(np.clip(strength * (t_mean - s_mean), -cfg.max_brightness_shift, cfg.max_brightness_shift))
    if emissive and not cfg.emissive_match_luminance:
        # A screen emits its own light: matching the ad's luminance / contrast to the (dark) surroundings would
        # only dull it. Keep the ad's tone; only the (weak) chroma adaptation below remains.
        contrast, shift = 1.0, 0.0
    cmax = cfg.max_chroma_shift
    return LightingParams(
        strength=strength,
        reference=ref_name,
        src_mean_L=s_mean,
        brightness_shift=shift,
        contrast_gain=contrast,
        a_shift=float(np.clip(strength * (t_a - s_a), -cmax, cmax)),
        b_shift=float(np.clip(strength * (t_b - s_b), -cmax, cmax)),
        target_mean_L=t_mean,
        target_std_L=t_std,
    )


def apply_lighting(bgr_u8: np.ndarray, params: LightingParams) -> np.ndarray:
    """Apply the LAB correction to a uint8 BGR image (no-op when no reference could be measured)."""
    if params.reference == "none":
        return bgr_u8
    lab = to_lab(bgr_u8)
    lab[..., 0] = np.clip(
        (lab[..., 0] - params.src_mean_L) * params.contrast_gain + params.src_mean_L + params.brightness_shift, 0, 100
    )
    lab[..., 1] = np.clip(lab[..., 1] + params.a_shift, -127, 127)
    lab[..., 2] = np.clip(lab[..., 2] + params.b_shift, -127, 127)
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)