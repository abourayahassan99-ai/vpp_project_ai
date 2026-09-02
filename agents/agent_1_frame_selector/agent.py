import cv2
import numpy as np
from typing import Optional, List
from ultralytics import YOLO

from core.state import CandidateFrameOutput, SurfaceROI
from agents.agent_1_frame_selector.blur_detector import BlurDetector

class FrameSelectorAgent:
    def __init__(self, yolo_model_path: str = "yolov8n-seg.pt", blur_threshold: float = 100.0):
        """
        Agent 1: Filters out motion-blurred frames and extracts target surface ROIs.
        """
        self.blur_detector = BlurDetector(threshold=blur_threshold)
        self.model = YOLO(yolo_model_path)
        
        # Target COCO classes for planar surfaces (e.g., tv/monitor, laptop, book, board)
        self.target_class_ids = [62, 63, 73]  # COCO IDs for tv, laptop, book/board

    def extract_candidate_frame(self, video_path: str, frame_interval: int = 10) -> Optional[CandidateFrameOutput]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video file at {video_path}")

        frame_id = 0
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        best_candidate: Optional[CandidateFrameOutput] = None
        highest_blur_score = 0.0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process every Nth frame to reduce processing latency
            if frame_id % frame_interval == 0:
                blur_score = self.blur_detector.calculate_blur_score(frame)
                
                # Filter out blurry frames early
                if blur_score >= self.blur_detector.threshold and blur_score > highest_blur_score:
                    # Run segmentation model
                    results = self.model(frame, verbose=False)[0]
                    
                    if results.masks is not None:
                        for idx, class_id in enumerate(results.boxes.cls.tolist()):
                            if int(class_id) in self.target_class_ids:
                                # Extract bounding box corners as surface polygon [(x1,y1), (x2,y2), ...]
                                bbox = results.boxes.xyxy[idx].cpu().numpy().astype(int)
                                x1, y1, x2, y2 = bbox
                                polygon_points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                                conf = float(results.boxes.conf[idx].cpu().numpy())

                                surface_roi = SurfaceROI(
                                    surface_type=self.model.names[int(class_id)],
                                    confidence=round(conf, 4),
                                    polygon_points=polygon_points
                                )

                                highest_blur_score = blur_score
                                best_candidate = CandidateFrameOutput(
                                    frame_id=frame_id,
                                    timestamp_sec=round(frame_id / fps, 2),
                                    blur_score=round(blur_score, 2),
                                    roi=surface_roi,
                                    feedback_prompt=f"Selected sharp frame with target surface '{surface_roi.surface_type}'"
                                )
                                break  # Take highest confidence ROI in the frame

            frame_id += 1

        cap.release()
        return best_candidate