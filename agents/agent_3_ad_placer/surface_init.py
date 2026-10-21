"""
Anchor initialisation for Agent 3: turn an Agent-1 polygon (or a manual quad) into the ordered,
validated 4-corner quad that the tracker uses as its plane definition.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

from .geometry import (adjust_quad, fit_quad, order_corners, parse_polygon, polygon_is_valid,
                       quad_dimensions, quad_inside_ratio, quad_is_plausible)
from .edge_refine import snap_quad_to_display
from .schemas import AdPlacerConfig, PlacementError

logger = logging.getLogger(__name__)


def build_anchor_quad(
    frame: np.ndarray,
    polygon_points: Any = None,
    manual_quad: Any = None,
    surface_type: str = "",
    cfg: Optional[AdPlacerConfig] = None,
) -> Tuple[np.ndarray, str, List[str]]:
    """
    Returns (quad TL,TR,BR,BL float32, method, warnings).

    Priority: ``manual_quad`` (4 points, any order; the exact usable surface) > ``polygon_points``
    (YOLO segmentation polygon: robust line-fit trapezoid, then a small in-plane inset from the
    object's outer bezel to the screen for emissive types).
    """
    cfg = cfg or AdPlacerConfig()
    h, w = frame.shape[:2]
    warnings: List[str] = []

    if manual_quad is not None:
        q = np.asarray(manual_quad, np.float32).reshape(-1, 2)
        if len(q) != 4:
            raise PlacementError("manual_quad must contain exactly 4 points")
        quad, method = order_corners(q), "manual"
    else:
        pts = parse_polygon(polygon_points, w, h)
        if not polygon_is_valid(pts):
            raise PlacementError("Surface polygon is missing or degenerate; supply manual_quad instead.")
        quad, method = fit_quad(pts)
        if surface_type in cfg.emissive_surface_types:
            snapped, info = (snap_quad_to_display(frame, quad, cfg.anchor_bezel_ratio)
                             if cfg.snap_anchor_to_edges else (None, {"reason": "disabled"}))
            if snapped is not None:
                quad, method = snapped, method + "+edge_snap"
                logger.info("anchor snapped to display edges: %s", {k: v for k, v in info.items() if k != "ok"})
            else:
                warnings.append("could not snap the anchor to the display edges (%s); the segmentation polygon is "
                                "usually 5-15 px too large, so the ad may overlap the bezel" % info.get("reason"))
                if cfg.anchor_bezel_inset:
                    quad = adjust_quad(quad, cfg.anchor_bezel_inset)
                    method += "+bezel_inset"

    if not quad_is_plausible(quad, frame.shape):
        raise PlacementError(f"Anchor quad is not plausible (convex, sane angles/aspect): {quad.round(1).tolist()}")
    inside = quad_inside_ratio(quad, frame.shape)
    if inside < 0.8:
        warnings.append(f"only {inside:.0%} of the surface is inside the anchor frame; choose a frame where it is "
                        "fully visible for the best tracking")
    qw, qh = quad_dimensions(quad)
    if min(qw, qh) < 0.05 * min(w, h):
        warnings.append("surface is very small in the anchor frame; tracking features will be scarce")
    for msg in warnings:
        logger.warning(msg)
    return quad.astype(np.float32), method, warnings
