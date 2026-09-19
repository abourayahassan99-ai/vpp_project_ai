"""Agent 3 - Ad Placement and Seamless Blending (vpp_project_ai).

    from agents.agent_3_ad_placer import AdPlacerAgent, AdPlacerConfig, Agent3Output
    from agents.agent_3_ad_placer import geometry, lighting, blending, tracking
"""

from . import blending, geometry, lighting, schemas, tracking
from .agent import AdPlacerAgent
from .geometry import fit_quad, order_corners, parse_polygon, polygon_is_valid
from .schemas import (
    VALID_BLEND_MODES,
    VALID_FIT_MODES,
    AdPlacerConfig,
    Agent3Output,
    BlendingStatus,
    LightingParams,
    PlacementError,
    PreparedAd,
    SurfaceCandidate,
)
from .tracking import FrameSource

__all__ = [
    # agent
    "AdPlacerAgent",
    "AdPlacerConfig",
    # schemas
    "Agent3Output",
    "BlendingStatus",
    "SurfaceCandidate",
    "LightingParams",
    "PreparedAd",
    "PlacementError",
    "VALID_BLEND_MODES",
    "VALID_FIT_MODES",
    # geometry helpers
    "fit_quad",
    "order_corners",
    "parse_polygon",
    "polygon_is_valid",
    # tracking
    "FrameSource",
    # sub-modules
    "schemas",
    "geometry",
    "lighting",
    "blending",
    "tracking",
]