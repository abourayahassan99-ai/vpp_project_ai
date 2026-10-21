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

    # Tracker settings (planar, anchored tracker -- see tracking.py)
    track_work_width: int = 640            # tracker works on frames downscaled to this width
    track_min_inliers: int = 12            # min RANSAC inliers for a frame to count as measured
    track_max_scale_change: float = 1.8    # max frame-to-frame quad area ratio (and its inverse)
    track_band_margin: int = 25            # width (working-scale px) of the on-surface edge band used for features
    track_ring_ratio: float = 0.5          # feature ring extends this fraction of min(w,h) outside the quad
    track_max_features: int = 250
    track_cov_homography: float = 0.15     # min point-coverage (hull / quad area) to trust a full homography
    track_cov_affine: float = 0.05         # below this only a similarity transform is fitted
    track_edge_refine: bool = True         # snap the quad sides to the surface's real edges each frame
    track_fb_error_px: float = 1.5         # forward-backward LK rejection threshold
    track_ransac_px: float = 3.0
    track_acquire_min_inliers: int = 22    # stricter inlier requirement for (re-)acquiring the surface
    track_reanchor_interval: int = 20      # AKAZE re-registration against the anchor every N frames
    track_reacquire_interval: int = 5      # while LOST, try to re-acquire every N frames
    track_max_coast_frames: int = 8        # DEGRADED frames tolerated before the surface is declared LOST
    track_min_visible_ratio: float = 0.15  # fraction of quad area inside the frame required to draw the ad
    track_reacquire_confirm: int = 3       # consecutive good frames before the ad fades back in
    track_max_interp_gap: int = 12         # fill gaps between reliable frames up to this length
    track_jump_tolerance: float = 0.35     # max corner jump vs. prediction, as a fraction of quad size

    # Temporal smoothing (constant-velocity Kalman + RTS backward pass)
    smooth_process_noise: float = 4.0      # working-scale px^2, acceleration noise
    smooth_measurement_noise: float = 1.0  # working-scale px^2 at confidence 1
    smooth_speed_ref: float = 3.0          # corner speed (px/frame) at which the smoothing noise doubles (adaptive)

    # Appearance in time
    fade_frames: int = 6                   # opacity ramp when the ad appears / disappears

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

    # Surface geometry refinement
    snap_anchor_to_edges: bool = True      # re-fit the anchor quad to the real display edges (see edge_refine.py)
    anchor_bezel_ratio: float = 0.018      # thin-bezel allowance (x smaller side) on sharp, strong wall edges only
    screen_inset_ratio: float = 0.0        # per-frame draw inset in the plane (fraction of size); <0 = overscan
    anchor_bezel_inset: float = 0.012      # applied once to YOLO object quads of emissive types (bezel -> screen)

    # Multi-shot handling (edited videos with cuts)
    multi_anchor: bool = True              # start a new tracking anchor in each shot where the first one cannot reach
    max_extra_anchors: int = 8
    surface_aspect: Optional[float] = None # true width/height of the surface (None = 16:9 for screens, else estimated)
    ad_max_render_long_side: int = 1280    # resolution the ad is prepared at (a mip pyramid is built from it)

    # Screen realism
    reflection_strength: float = 0.5       # 0 disables; glass reflections added to the ad (kept subtle)
    reflection_cap: float = 38.0           # max brightness (0-255) the reflection may add
    reflection_dark_threshold: float = 95.0  # reflections are only taken from screen pixels darker than this
    reflection_blur_ratio: float = 0.03    # low-pass sigma as a fraction of the screen's smaller side
    emissive_match_luminance: bool = False # False: never darken / flatten the ad towards the dark wall for screens

    # Backward fill (ad from the first reliably verified frame)
    track_backfill: bool = True
    track_backfill_max: int = 90           # max frames tracked backwards from a verified frame
    track_backfill_min_conf: float = 0.4

    # Realism
    use_occlusion: bool = False            # opt-in: only reliable for static surfaces (switched-off TV, wall); verify on your footage
    occlusion_threshold: float = 0.25      # normalised plane-space difference regarded as foreground
    occlusion_max_fraction: float = 0.55   # above this the difference is a content change, not an occluder
    motion_blur_shutter: float = 0.5       # fraction of frame time the shutter is open (0 disables)
    max_motion_blur_px: float = 24.0

    # Blending
    blend_mode: str = "alpha"   # tight tracked quad: Poisson ("hybrid") bleeds the dark bezel into the ad edges
    feather_ratio: float = 0.02
    clone_edge_band_ratio: float = 0.10