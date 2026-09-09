from typing import List, Tuple, Optional, Dict, Any
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# ==========================================
# 1. Agent 1: Object Detection Schemas
# ==========================================

class SurfaceROI(BaseModel):
    """
    Represents the detected region of an object.

    Agent 1 only reports what was detected.
    It does NOT decide whether the object is suitable
    for advertisement placement.
    """

    surface_type: str = Field(
        description="Detected object/surface name"
    )

    confidence: float = Field(
        description="YOLO detection confidence"
    )

    polygon_points: List[Tuple[int, int]] = Field(
        description="Segmentation polygon points"
    )


class DetectedSurface(BaseModel):
    """
    Represents one detected object in the video.
    """

    object_id: Optional[int] = Field(
        default=None,
        description="Tracking ID assigned by YOLO when available"
    )

    # First time YOLO detected this tracked object
    first_seen_frame: int = Field(
        default=0,
        description="First frame where this object was detected"
    )

    first_seen_timestamp_sec: float = Field(
        default=0.0,
        description="Timestamp of first detection"
    )

    # Best observation of the object
    frame_id: int = Field(
        description="Frame of the best-confidence observation"
    )

    timestamp_sec: float = Field(
        description="Timestamp of the best-confidence observation"
    )

    blur_score: float = Field(
        description="Laplacian variance clarity score"
    )

    roi: SurfaceROI = Field(
        description="Detected object region"
    )


class CandidateFrameOutput(BaseModel):
    """
    Agent 1 output.

    Agent 1 dynamically detects objects from the video.

    It does NOT make the final advertisement placement decision.
    """

    detected_surfaces: List[DetectedSurface] = Field(
        default_factory=list,
        description="Objects detected throughout the video"
    )

    total_surfaces: int = Field(
        default=0,
        description="Number of unique detected object candidates"
    )

    feedback_prompt: str = Field(
        default="",
        description="Summary of detected objects"
    )


# ==========================================
# 2. Agent 2: Scene Context Schemas
# ==========================================

class SceneContextOutput(BaseModel):
    scene_type: str = Field(
        description="Classification: dialogue, action, static, close_up"
    )

    motion_level: str = Field(
        description="Motion dynamics: Low, Medium, High"
    )

    has_occlusion_risk: bool = Field(
        description="True if humans/objects cross over target surface"
    )

    lighting_condition: str = Field(
        description="Environment light: Normal, Dark, High-Contrast"
    )

    recommended_opacity: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Suggested blending opacity"
    )


# ==========================================
# 3. Agent 3: Ad Fitting Engine Schemas
# ==========================================

class CompositeOutput(BaseModel):
    output_video_path: str = Field(
        description="Path to rendered video with ad placed"
    )

    homography_matrix_applied: bool = Field(
        default=True
    )

    poisson_blending_success: bool = Field(
        default=True
    )

    processing_time_sec: float


# ==========================================
# 4. Shared LangGraph Pipeline State
# ==========================================

class GraphState(TypedDict):
    # Global Inputs
    video_path: str
    ad_asset_path: str

    # Agent Inter-communication Outputs
    candidate_frame: Optional[Dict[str, Any]]
    scene_context: Optional[Dict[str, Any]]
    composite_result: Optional[Dict[str, Any]]

    # Execution Tracking
    current_agent: str
    errors: List[str]