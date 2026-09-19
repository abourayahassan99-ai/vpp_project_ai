from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

VALID_BLEND_MODES = ("hybrid", "alpha", "normal_clone", "mixed_clone")
VALID_FIT_MODES = ("cover", "stretch")


class PlacementError(RuntimeError):
    """Expected, recoverable failure (turned into a failed BlendingStatus by the agent)."""


# --------------------------------------------------------------------------- #
# Public output
# --------------------------------------------------------------------------- #
class BlendingStatus(BaseModel):
    success: bool
    message: str


class Agent3Output(BaseModel):
    selected_surface_id: Optional[Union[int, str]] = None
    selected_frame_id: Optional[int] = None
    blending_status: BlendingStatus
    output_image_path: Optional[str] = None
    output_video_path: Optional[str] = None
    placement_metrics: Dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Stage-to-stage models
# --------------------------------------------------------------------------- #
class SurfaceCandidate(BaseModel):
    """A validated Agent 1 surface, with its fitted quad and selection score."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    object_id: Optional[Union[int, str]] = None
    frame_id: int
    first_seen_frame: Optional[int] = None
    timestamp_sec: Optional[float] = None
    blur_score: Optional[float] = None
    surface_type: str
    confidence: float
    polygon: np.ndarray  # (N, 2) float32, raw polygon in pixels
    quad: np.ndarray  # (4, 2) float32, ordered TL, TR, BR, BL
    fit_method: str
    area_ratio: float
    score: float = 0.0
    score_breakdown: Dict[str, float] = Field(default_factory=dict)


class LightingParams(BaseModel):
    """LAB-space correction applied to the ad."""

    strength: float = 0.0
    reference: str = "none"
    src_mean_L: float = 0.0
    brightness_shift: float = 0.0
    contrast_gain: float = 1.0
    a_shift: float = 0.0
    b_shift: float = 0.0
    target_mean_L: float = 0.0
    target_std_L: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.model_dump().items()}


@dataclass
class PreparedAd:
    bgr: np.ndarray  # uint8 (h, w, 3)
    alpha: np.ndarray  # float32 (h, w) in [0, 1]


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class AdPlacerConfig:
    # I/O
    agent1_output_path: Path = Path("data/agent1_output.json")
    agent2_output_path: Path = Path("data/agent2_output.json")
    output_image_path: Path = Path("data/ad_placement_output.jpg")
    output_video_path: Path = Path("data/ad_placement_output.mp4")
    jpeg_quality: int = 95

    # Video Render Settings
    video_max_frames: Optional[int] = None
    video_fourcc: str = "mp4v"

    # Tracker Settings
    track_min_inliers: int = 12
    track_max_scale_change: float = 1.8
    track_band_margin: int = 25

    # Surface selection
    min_confidence: float = 0.30
    min_area_ratio: float = 0.0005
    preferred_surface_types: Tuple[str, ...] = ("tv", "television", "monitor", "screen", "display")
    emissive_surface_types: Tuple[str, ...] = ("tv", "television", "monitor", "screen", "display", "laptop")

    # Geometry
    fit_mode: str = "cover"

    # Lighting
    lighting_strength: float = 0.6
    emissive_lighting_strength: float = 0.3
    contrast_gain_range: Tuple[float, float] = (0.7, 1.3)
    max_brightness_shift: float = 25.0
    max_chroma_shift: float = 20.0

    # Blending
    blend_mode: str = "hybrid"
    feather_ratio: float = 0.02
    clone_edge_band_ratio: float = 0.10