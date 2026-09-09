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

    Responsibilities:
    - Detect all objects supported by the YOLO model.
    - Track objects when tracking IDs are available.
    - Extract object location/polygon.
    - Store first-seen information.
    - Store the best-confidence observation.
    - Calculate blur score only when YOLO runs.

    Agent 1 does NOT decide whether an object is suitable
    for advertisement placement.
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

        # Run YOLO every N frames.
        # Between YOLO calls, the last tracked result is reused
        # for visualization, but new objects are recorded only
        # when YOLO actually runs.
        self.detection_interval = max(1, detection_interval)

        # Smaller inference size = faster processing.
        self.imgsz = imgsz

        # Blur is calculated on a small image instead of the
        # original 2560x1440 frame.
        self.blur_resize_width = blur_resize_width

        # Automatically use GPU when available.
        if device is not None:
            self.device = device
        else:
            self.device = 0 if torch.cuda.is_available() else "cpu"

        self.use_half = bool(torch.cuda.is_available())

        self.model = YOLO(self.model_path)

        self.blur_detector = BlurDetector(
            threshold=self.blur_threshold
        )

    # ---------------------------------------------------------
    # Fast blur calculation
    # ---------------------------------------------------------

    def _calculate_fast_blur_score(
        self,
        frame: np.ndarray
    ) -> float:
        """
        Calculate blur on a resized frame.

        This avoids running Laplacian variance over the full
        2560x1440 frame every time.
        """

        if frame is None or frame.size == 0:
            return 0.0

        height, width = frame.shape[:2]

        if width > self.blur_resize_width:
            new_height = int(
                height * self.blur_resize_width / width
            )

            small_frame = cv2.resize(
                frame,
                (self.blur_resize_width, new_height),
                interpolation=cv2.INTER_AREA,
            )
        else:
            small_frame = frame

        return self.blur_detector.calculate_blur_score(
            small_frame
        )

    # ---------------------------------------------------------
    # YOLO detection + tracking
    # ---------------------------------------------------------

    def _detect_and_track(
        self,
        frame: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """
        Run YOLO segmentation + tracking on one frame.
        """

        # Build tracking arguments dynamically.
        # On CPU, do not pass half=False because newer
        # Ultralytics versions warn that half is deprecated.
        track_kwargs = {
            "persist": True,
            "conf": self.detection_confidence,
            "classes": self.target_class_ids,
            "imgsz": self.imgsz,
            "device": self.device,
            "verbose": False,
        }

        # Use half precision only when CUDA/GPU is available.
        if self.use_half:
            track_kwargs["half"] = True

        results = self.model.track(
            frame,
            **track_kwargs,
        )

        if not results:
            return []

        result = results[0]

        if result.boxes is None:
            return []

        detections = []

        boxes = result.boxes

        names = result.names

        masks_xy = None

        if result.masks is not None:
            masks_xy = result.masks.xy

        for index in range(len(boxes)):
            box = boxes[index]

            confidence = float(
                box.conf[0].item()
            )

            class_id = int(
                box.cls[0].item()
            )

            object_name = names[class_id]

            # Tracking ID
            track_id = None

            if box.id is not None:
                track_id = int(
                    box.id[0].item()
                )

            # Bounding box
            xyxy = (
                box.xyxy[0]
                .cpu()
                .numpy()
                .astype(int)
            )

            x1, y1, x2, y2 = xyxy.tolist()

            # Segmentation polygon
            polygon = []

            if (
                masks_xy is not None
                and index < len(masks_xy)
            ):
                raw_polygon = masks_xy[index]

                polygon = [
                    (int(x), int(y))
                    for x, y in raw_polygon
                ]

            # Fallback to bounding box if segmentation
            # polygon is unavailable.
            if len(polygon) < 3:
                polygon = [
                    (x1, y1),
                    (x2, y1),
                    (x2, y2),
                    (x1, y2),
                ]

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

    # ---------------------------------------------------------
    # Main processing
    # ---------------------------------------------------------

    def process_video(
        self,
        video_path: str,
    ) -> CandidateFrameOutput:

        cap = cv2.VideoCapture(
            video_path
        )

        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open video: {video_path}"
            )

        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        if fps <= 0:
            fps = 30.0

        total_frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        best_objects: Dict[Any, Dict[str, Any]] = {}

        frame_id = 0

        last_detections: List[
            Dict[str, Any]
        ] = []

        processing_start = (
            time.perf_counter()
        )

        while True:

            success, frame = cap.read()

            if not success:
                break

            # -------------------------------------------------
            # YOLO only every N frames
            # -------------------------------------------------

            should_detect = (
                frame_id
                % self.detection_interval
                == 0
            )

            if should_detect:

                detections = (
                    self._detect_and_track(
                        frame
                    )
                )

                # Save latest detections for optional
                # visualization.
                last_detections = detections

                # Calculate blur ONLY when YOLO runs.
                blur_score = (
                    self._calculate_fast_blur_score(
                        frame
                    )
                )

                timestamp_sec = (
                    frame_id / fps
                )

                # ---------------------------------------------
                # Store object observations
                # ---------------------------------------------

                for detection in detections:

                    object_id = detection[
                        "object_id"
                    ]

                    object_name = detection[
                        "object_name"
                    ]

                    confidence = detection[
                        "confidence"
                    ]

                    polygon = detection[
                        "polygon"
                    ]

                    # Prefer tracking ID.
                    #
                    # If YOLO did not provide one,
                    # use class name as a fallback.
                    if object_id is not None:
                        object_key = (
                            "track",
                            object_id,
                        )
                    else:
                        object_key = (
                            "class",
                            detection[
                                "class_id"
                            ],
                            object_name,
                        )

                    # -----------------------------------------
                    # First time seeing this object
                    # -----------------------------------------

                    if object_key not in best_objects:

                        best_objects[
                            object_key
                        ] = {
                            "object_id": object_id,

                            "object_name": object_name,

                            "first_seen_frame": (
                                frame_id
                            ),

                            "first_seen_timestamp_sec": (
                                timestamp_sec
                            ),

                            "best_frame_id": (
                                frame_id
                            ),

                            "best_timestamp_sec": (
                                timestamp_sec
                            ),

                            "best_confidence": (
                                confidence
                            ),

                            "best_blur_score": (
                                blur_score
                            ),

                            "best_polygon": (
                                polygon
                            ),
                        }

                        print(
                            f"[FOUND] "
                            f"{object_name} "
                            f"confidence={confidence:.2f} "
                            f"frame={frame_id} "
                            f"time={timestamp_sec:.2f} "
                            f"track_id={object_id}"
                        )

                    # -----------------------------------------
                    # Better observation found
                    # -----------------------------------------

                    else:

                        current = best_objects[
                            object_key
                        ]

                        if (
                            confidence
                            > current[
                                "best_confidence"
                            ]
                        ):

                            current[
                                "best_confidence"
                            ] = confidence

                            current[
                                "best_frame_id"
                            ] = frame_id

                            current[
                                "best_timestamp_sec"
                            ] = timestamp_sec

                            current[
                                "best_blur_score"
                            ] = blur_score

                            current[
                                "best_polygon"
                            ] = polygon

            frame_id += 1

        cap.release()

        processing_time = (
            time.perf_counter()
            - processing_start
        )

        # -----------------------------------------------------
        # Build final Agent 1 output
        # -----------------------------------------------------

        detected_surfaces = []

        for data in best_objects.values():

            roi = SurfaceROI(
                surface_type=data[
                    "object_name"
                ],

                confidence=data[
                    "best_confidence"
                ],

                polygon_points=data[
                    "best_polygon"
                ],
            )

            detected_surface = DetectedSurface(
                object_id=data[
                    "object_id"
                ],

                first_seen_frame=data[
                    "first_seen_frame"
                ],

                first_seen_timestamp_sec=data[
                    "first_seen_timestamp_sec"
                ],

                frame_id=data[
                    "best_frame_id"
                ],

                timestamp_sec=data[
                    "best_timestamp_sec"
                ],

                blur_score=data[
                    "best_blur_score"
                ],

                roi=roi,
            )

            detected_surfaces.append(
                detected_surface
            )

        # Highest confidence first
        detected_surfaces.sort(
            key=lambda x: x.roi.confidence,
            reverse=True,
        )

        object_names = [
            item.roi.surface_type
            for item in detected_surfaces
        ]

        if object_names:

            feedback_prompt = (
                "Agent 1 detected: "
                + ", ".join(object_names)
            )

        else:

            feedback_prompt = (
                "Agent 1 did not detect any objects."
            )

        output = CandidateFrameOutput(
            detected_surfaces=detected_surfaces,
            total_surfaces=len(
                detected_surfaces
            ),
            feedback_prompt=feedback_prompt,
        )

        # -----------------------------------------------------
        # Performance information
        # -----------------------------------------------------

        video_duration = (
            total_frames / fps
            if fps > 0
            else 0.0
        )

        processing_fps = (
            total_frames / processing_time
            if processing_time > 0
            else 0.0
        )

        speed_ratio = (
            processing_fps / fps
            if fps > 0
            else 0.0
        )

        print(
            "\n=========================================="
        )

        print(
            "AGENT 1 PERFORMANCE"
        )

        print(
            "=========================================="
        )

        print(
            f"Video duration: "
            f"{video_duration:.2f} sec"
        )

        print(
            f"Processing time: "
            f"{processing_time:.2f} sec"
        )

        print(
            f"Processing FPS: "
            f"{processing_fps:.2f}"
        )

        print(
            f"Realtime ratio: "
            f"{speed_ratio:.2f}x"
        )

        print(
            f"YOLO every "
            f"{self.detection_interval} frame(s)"
        )

        print(
            f"Inference image size: "
            f"{self.imgsz}"
        )

        print(
            f"Device: "
            f"{self.device}"
        )

        print(
            f"Unique objects: "
            f"{len(detected_surfaces)}"
        )

        print(
            "=========================================="
        )

        return output