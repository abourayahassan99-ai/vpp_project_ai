from typing import List, Tuple, Optional, Dict, Any
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# ==========================================
# 1. Agent 1: Candidate Frame & ROI Schemas
# ==========================================
class SurfaceROI(BaseModel):
    surface_type: str = Field(description="Detected target surface type (e.g., wall, billboard, screen)")
    confidence: float = Field(description="Detection confidence score")
    polygon_points: List[Tuple[int, int]] = Field(description="4 corner coordinates [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]")

class CandidateFrameOutput(BaseModel):
    frame_id: int = Field(description="Frame index selected from raw video")
    timestamp_sec: float = Field(description="Timestamp in seconds")
    blur_score: float = Field(description="Laplacian variance clarity score")
    roi: SurfaceROI
    feedback_prompt: str = Field(description="VLM-generated description explaining frame selection")


# ==========================================
# 2. Agent 2: Scene Context Schemas
# ==========================================
class SceneContextOutput(BaseModel):
    scene_type: str = Field(description="Classification: dialogue, action, static, close_up")
    motion_level: str = Field(description="Motion dynamics: Low, Medium, High")
    has_occlusion_risk: bool = Field(description="True if humans/objects cross over the target surface ROI")
    lighting_condition: str = Field(description="Environment light: Normal, Dark, High-Contrast")
    recommended_opacity: float = Field(default=1.0, ge=0.0, le=1.0, description="Suggested blending opacity")


# ==========================================
# 3. Agent 3: Ad Fitting Engine Schemas
# ==========================================
class CompositeOutput(BaseModel):
    output_video_path: str = Field(description="Path to the rendered video with ad placed")
    homography_matrix_applied: bool = Field(default=True)
    poisson_blending_success: bool = Field(default=True)
    processing_time_sec: float


# ==========================================
# 4. Shared LangGraph Pipeline State
# ==========================================
class GraphState(TypedDict):
    # Global Inputs
    video_path: str
    ad_asset_path: str
    
    # Agent Inter-communication Outputs
    candidate_frame: Optional[Dict[str, Any]]  # Agent 1 output serialized as dict
    scene_context: Optional[Dict[str, Any]]    # Agent 2 output serialized as dict
    composite_result: Optional[Dict[str, Any]]  # Agent 3 output serialized as dict
    
    # Execution Tracking
    current_agent: str
    errors: List[str]