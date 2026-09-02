import os
import cv2
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from agents.agent_1_frame_selector.agent import FrameSelectorAgent
from core.state import CandidateFrameOutput

TEST_VIDEO_PATH = "tests/temp_test_video.mp4"

@pytest.fixture(scope="module")
def generate_synthetic_video():
    """Generates a 30-frame synthetic video containing clean target shapes."""
    os.makedirs("tests", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(TEST_VIDEO_PATH, fourcc, 10.0, (640, 480))

    for i in range(30):
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 200
        cv2.rectangle(frame, (150, 100), (490, 380), (50, 50, 50), -1)
        cv2.putText(frame, "AD SURFACE", (220, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Apply intentional heavy blur to first 10 frames
        if i < 10:
            frame = cv2.GaussianBlur(frame, (51, 51), 0)

        out.write(frame)

    out.release()
    yield TEST_VIDEO_PATH

    if os.path.exists(TEST_VIDEO_PATH):
        os.remove(TEST_VIDEO_PATH)

def test_agent_1_frame_selection(generate_synthetic_video):
    # Mock YOLO model inference so unit test isolates Agent 1's filtering logic
    with patch("agents.agent_1_frame_selector.agent.YOLO") as MockYOLO:
        mock_model_instance = MagicMock()
        MockYOLO.return_value = mock_model_instance
        
        mock_result = MagicMock()
        mock_box = MagicMock()
        mock_box.cls.tolist.return_value = [62]  # Target class ID (tv)
        
        mock_xyxy = MagicMock()
        mock_xyxy.cpu.return_value.numpy.return_value = np.array([150, 100, 490, 380])
        mock_box.xyxy = [mock_xyxy]
        
        mock_conf = MagicMock()
        mock_conf.cpu.return_value.numpy.return_value = 0.95
        mock_box.conf = [mock_conf]
        
        mock_result.masks = [MagicMock()]
        mock_result.boxes = mock_box
        mock_model_instance.return_value = [mock_result]
        mock_model_instance.names = {62: "tv"}

        agent = FrameSelectorAgent(blur_threshold=50.0)
        candidate = agent.extract_candidate_frame(generate_synthetic_video, frame_interval=2)
        
        assert candidate is not None, "Agent 1 failed to return any candidate frame."
        assert isinstance(candidate, CandidateFrameOutput), "Output does not conform to CandidateFrameOutput schema."
        assert candidate.blur_score >= 50.0, f"Selected frame blur score ({candidate.blur_score}) is below threshold."
        assert candidate.frame_id >= 10, "Agent failed to filter out early blurred frames."
        
        print(f"\n[SUCCESS] Agent 1 Selected Frame: {candidate.frame_id}")
        print(f"Timestamp: {candidate.timestamp_sec}s | Clarity Score: {candidate.blur_score}")
        print(f"Target ROI: {candidate.roi.surface_type} | Points: {candidate.roi.polygon_points}")