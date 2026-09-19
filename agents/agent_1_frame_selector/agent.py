import time
from typing import Optional, Dict, Any, List

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from core.state import (
    CandidateFrameOutput,
    DetectedSurface,
    SurfaceROI,
)

from .blur_detector import BlurDetector


class FrameSelectorAgent:
    """
    Agent 1
    =========
    Object Discovery + Tracking.
    """

    def __init__(
        self,
        model_path: str = "yolov8n-seg.pt",
        detection_confidence: float = 0.30,
        target_class_ids: Optional[List[int]] = None,
        blur_threshold: float = 15.0,
        detection_interval: int = 5,
        imgsz: int = 512,
        blur_resize_width: int = 640,
        device: Optional[Any] = None,
    ):
        self.model_path = model_path
        self.detection_confidence = detection_confidence
        self.target_class_ids = target_class_ids
        self.blur_threshold = blur_threshold
        self.detection_interval = max(1, detection_interval)
        self.imgsz = imgsz
        self.blur_resize_width = blur_resize_width

        if device is not None:
            self.device = device
        else:
            self.device = 0 if torch.cuda.is_available() else "cpu"

        self.use_half = bool(torch.cuda.is_available())
        self.model = YOLO(self.model_path)
        self.blur_detector = BlurDetector(threshold=self.blur_threshold)

    def _calculate_fast_blur_score(self, frame: np.ndarray) -> float:
        if frame is None or frame.size == 0:
            return 0.0

        height, width = frame.shape[:2]
        if width > self.blur_resize_width:
            new_height = int(height * self.blur_resize_width / width)
            small_frame = cv2.resize(
                frame,
                (self.blur_resize_width, new_height),
                interpolation=cv2.INTER_AREA,
            )
        else:
            small_frame = frame

        return self.blur_detector.calculate_blur_score(small_frame)

    def _detect_and_track(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        track_kwargs = {
            "persist": True,
            "conf": self.detection_confidence,
            "classes": self.target_class_ids,
            "imgsz": self.imgsz,
            "device": self.device,
            "verbose": False,
            "tracker": "bytetrack.yaml",
        }

        if self.use_half:
            track_kwargs["half"] = True

        results = self.model.track(frame, **track_kwargs)
        if not results:
            return []

        result = results[0]
        if result.boxes is None:
            return []

        detections = []
        boxes = result.boxes
        names = result.names
        masks_xy = result.masks.xy if result.masks is not None else None

        for index in range(len(boxes)):
            box = boxes[index]
            confidence = float(box.conf[0].item())
            class_id = int(box.cls[0].item())
            object_name = names[class_id]

            track_id = None
            if box.id is not None:
                track_id = int(box.id[0].item())

            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy.tolist()

            polygon = []
            if masks_xy is not None and index < len(masks_xy):
                raw_polygon = masks_xy[index]
                polygon = [(int(x), int(y)) for x, y in raw_polygon]

            if len(polygon) < 3:
                polygon = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

            detections.append(
                {
                    "object_id": track_id,
                    "class_id": class_id,
                    "object_name": object_name,
                    "confidence": confidence,
                    "polygon": polygon,
                    "bbox": (x1, y1, x2, y2),
                }
            )

        return detections

    def process_video(self, video_path: str) -> CandidateFrameOutput:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        best_objects: Dict[Any, Dict[str, Any]] = {}
        frame_id = 0
        processing_start = time.perf_counter()

        while True:
            success, frame = cap.read()
            if not success:
                break

            should_detect = (frame_id % self.detection_interval == 0)

            if should_detect:
                detections = self._detect_and_track(frame)
                blur_score = self._calculate_fast_blur_score(frame)
                timestamp_sec = frame_id / fps

                for detection in detections:
                    object_id = detection["object_id"]
                    object_name = detection["object_name"]
                    confidence = detection["confidence"]
                    polygon = detection["polygon"]

                    if object_id is not None:
                        object_key = ("track", object_id)
                    else:
                        object_key = (
                            "class",
                            detection["class_id"],
                            object_name,
                        )

                    if object_key not in best_objects:
                        best_objects[object_key] = {
                            "object_id": object_id,
                            "object_name": object_name,
                            "first_seen_frame": frame_id,
                            "first_seen_timestamp_sec": timestamp_sec,
                            "best_frame_id": frame_id,
                            "best_timestamp_sec": timestamp_sec,
                            "best_confidence": confidence,
                            "best_blur_score": blur_score,
                            "best_polygon": polygon,
                            "trajectory": [],
                        }

                        print(
                            f"[FOUND] {object_name} "
                            f"confidence={confidence:.2f} "
                            f"frame={frame_id} "
                            f"time={timestamp_sec:.2f} "
                            f"track_id={object_id}"
                        )
                    else:
                        current = best_objects[object_key]
                        if confidence > current["best_confidence"]:
                            current["best_confidence"] = confidence
                            current["best_frame_id"] = frame_id
                            current["best_timestamp_sec"] = timestamp_sec
                            current["best_blur_score"] = blur_score
                            current["best_polygon"] = polygon

                    # إضافة بيانات الفريم الحالي داخل الـ trajectory
                    best_objects[object_key]["trajectory"].append(
                        {
                            "frame_id": frame_id,
                            "timestamp_sec": timestamp_sec,
                            "confidence": confidence,
                            "polygon": polygon,
                        }
                    )

            frame_id += 1  # <--- داخل حلقة while مباشرة

        cap.release()

        processing_time = time.perf_counter() - processing_start

        # بناء كائنات المخرجات
        detected_surfaces = []
        for data in best_objects.values():
            roi = SurfaceROI(
                surface_type=data["object_name"],
                confidence=data["best_confidence"],
                polygon_points=data["best_polygon"],
            )

            detected_surface = DetectedSurface(
                object_id=data["object_id"],
                first_seen_frame=data["first_seen_frame"],
                first_seen_timestamp_sec=data["first_seen_timestamp_sec"],
                frame_id=data["best_frame_id"],
                timestamp_sec=data["best_timestamp_sec"],
                blur_score=data["best_blur_score"],
                roi=roi,
                trajectory=data["trajectory"],
            )

            detected_surfaces.append(detected_surface)

        detected_surfaces.sort(
            key=lambda x: x.roi.confidence, reverse=True
        )

        object_names = [item.roi.surface_type for item in detected_surfaces]
        if object_names:
            feedback_prompt = (
                "Agent 1 detected: " + ", ".join(object_names)
            )
        else:
            feedback_prompt = "Agent 1 did not detect any objects."

        output = CandidateFrameOutput(
            detected_surfaces=detected_surfaces,
            total_surfaces=len(detected_surfaces),
            feedback_prompt=feedback_prompt,
        )

        video_duration = total_frames / fps if fps > 0 else 0.0
        processing_fps = (
            total_frames / processing_time if processing_time > 0 else 0.0
        )
        speed_ratio = processing_fps / fps if fps > 0 else 0.0

        print("\n==========================================")
        print("AGENT 1 PERFORMANCE")
        print("==========================================")
        print(f"Video duration: {video_duration:.2f} sec")
        print(f"Processing time: {processing_time:.2f} sec")
        print(f"Processing FPS: {processing_fps:.2f}")
        print(f"Realtime ratio: {speed_ratio:.2f}x")
        print(f"YOLO every {self.detection_interval} frame(s)")
        print(f"Inference image size: {self.imgsz}")
        print(f"Device: {self.device}")
        print(f"Unique objects: {len(detected_surfaces)}")
        print("==========================================")

        return output