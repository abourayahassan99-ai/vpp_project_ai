import json
import os

import cv2
import numpy as np
from ultralytics import YOLO


MODEL_PATH = "yolov8n-seg.pt"

CONFIDENCE = 0.30

DETECTION_INTERVAL = 5

IMAGE_SIZE = 512

DEFAULT_OUTPUT_VIDEO = "data/tracked_surface_output.mp4"

DEFAULT_OUTPUT_JSON = "data/agent1_output.json"


def polygon_to_quad(polygon: np.ndarray) -> list:
    """
    تحويل القناع المضلع (Polygon)
    إلى 4 زوايا (Quad) متناسقة لإسقاط الإعلان.
    """

    rect = cv2.minAreaRect(polygon)

    box = cv2.boxPoints(rect)

    return np.int32(box).tolist()


def draw_result(frame, result):
    """
    رسم نتائج الـ detection/tracking على الفريم.
    """

    if result is None:
        return frame

    output = frame.copy()

    names = result.names

    boxes = result.boxes

    if boxes is None:
        return output

    masks_xy = (
        result.masks.xy
        if result.masks is not None
        else None
    )

    for index in range(len(boxes)):

        box = boxes[index]

        confidence = float(
            box.conf[0].item()
        )

        class_id = int(
            box.cls[0].item()
        )

        object_name = names[class_id]

        track_id = (
            int(box.id[0].item())
            if box.id is not None
            else None
        )

        # -------------------------
        # Polygon
        # -------------------------

        polygon = None

        if (
            masks_xy is not None
            and index < len(masks_xy)
        ):
            polygon = np.array(
                masks_xy[index],
                dtype=np.int32
            )

        # -------------------------
        # Bounding Box
        # -------------------------

        xyxy = (
            box.xyxy[0]
            .cpu()
            .numpy()
            .astype(int)
        )

        x1, y1, x2, y2 = xyxy.tolist()

        # -------------------------
        # Draw Polygon
        # -------------------------

        if polygon is not None and len(polygon) >= 3:

            overlay = output.copy()

            cv2.fillPoly(
                overlay,
                [polygon],
                (0, 255, 0)
            )

            output = cv2.addWeighted(
                overlay,
                0.18,
                output,
                0.82,
                0
            )

            cv2.polylines(
                output,
                [polygon],
                True,
                (0, 255, 0),
                2
            )

        # -------------------------
        # Draw Bounding Box
        # -------------------------

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # -------------------------
        # Label
        # -------------------------

        if track_id is not None:

            label = (
                f"{object_name} "
                f"ID:{track_id} "
                f"({confidence:.2f})"
            )

        else:

            label = (
                f"{object_name} "
                f"({confidence:.2f})"
            )

        text_y = max(
            25,
            y1 - 10
        )

        cv2.putText(
            output,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    return output


def track_and_save_video(
    input_path,
    output_path=DEFAULT_OUTPUT_VIDEO,
    json_output_path=DEFAULT_OUTPUT_JSON
):
    """
    تشغيل Agent 1 على فيديو معين.

    Parameters
    ----------
    input_path:
        مسار الفيديو اللي هنشتغل عليه.

    output_path:
        مسار فيديو الـ preview الناتج.

    json_output_path:
        مسار ملف JSON الناتج من Agent 1.
    """

    # -------------------------
    # Create Output Directories
    # -------------------------

    os.makedirs(
        os.path.dirname(output_path) or ".",
        exist_ok=True
    )

    os.makedirs(
        os.path.dirname(json_output_path) or ".",
        exist_ok=True
    )

    # -------------------------
    # Load YOLO Model
    # -------------------------

    print("Loading YOLO model...")

    model = YOLO(MODEL_PATH)

    # -------------------------
    # Open Video
    # -------------------------

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open video: {input_path}"
        )

    # -------------------------
    # Video Information
    # -------------------------

    fps = (
        cap.get(cv2.CAP_PROP_FPS)
        or 30.0
    )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    print()
    print("==========================================")
    print("AGENT 1 - TRACKING VIDEO")
    print("==========================================")
    print(f"Input Video: {input_path}")
    print(f"Resolution: {width} x {height}")
    print(f"FPS: {fps}")
    print(f"Total Frames: {total_frames}")
    print("==========================================")

    # -------------------------
    # Video Writer
    # -------------------------

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    if not writer.isOpened():

        cap.release()

        raise RuntimeError(
            f"Could not create output video: {output_path}"
        )

    # -------------------------
    # Tracking Variables
    # -------------------------

    frame_count = 0

    yolo_runs = 0

    last_result = None

    # بيانات التتبع
    tracks_data = {}

    # -------------------------
    # Process Video
    # -------------------------

    while True:

        success, frame = cap.read()

        if not success:
            break

        # -------------------------
        # Run YOLO Every N Frames
        # -------------------------

        if frame_count % DETECTION_INTERVAL == 0:

            results = model.track(
                frame,
                persist=True,
                conf=CONFIDENCE,
                imgsz=IMAGE_SIZE,
                verbose=False
            )

            if results:

                last_result = results[0]

                yolo_runs += 1

                if (
                    last_result.boxes is not None
                    and last_result.boxes.id is not None
                ):

                    boxes = last_result.boxes

                    masks_xy = (
                        last_result.masks.xy
                        if last_result.masks is not None
                        else []
                    )

                    # -------------------------
                    # Process Detected Objects
                    # -------------------------

                    for idx, box in enumerate(boxes):

                        track_id = int(
                            box.id[0].item()
                        )

                        class_id = int(
                            box.cls[0].item()
                        )

                        class_name = (
                            last_result.names[class_id]
                        )

                        conf = float(
                            box.conf[0].item()
                        )

                        # Bounding Box

                        bbox = (
                            box.xyxy[0]
                            .cpu()
                            .numpy()
                            .astype(int)
                            .tolist()
                        )

                        # -------------------------
                        # Polygon + Quad
                        # -------------------------

                        if idx < len(masks_xy):

                            poly_pts = (
                                masks_xy[idx]
                                .astype(int)
                                .tolist()
                            )

                            quad_pts = polygon_to_quad(
                                masks_xy[idx]
                            )

                        else:

                            poly_pts = []

                            quad_pts = []

                        # -------------------------
                        # Create Track
                        # -------------------------

                        if track_id not in tracks_data:

                            tracks_data[track_id] = {

                                "track_id": track_id,

                                "class_name": class_name,

                                "first_seen_frame": frame_count,

                                "frames": []
                            }

                        # -------------------------
                        # Save Frame Data
                        # -------------------------

                        tracks_data[
                            track_id
                        ]["frames"].append({

                            "frame_id": frame_count,

                            "confidence": round(
                                conf,
                                3
                            ),

                            "bbox": bbox,

                            "polygon": poly_pts,

                            "quad": quad_pts
                        })

        # -------------------------
        # Draw Preview
        # -------------------------

        output_frame = draw_result(
            frame,
            last_result
        )

        cv2.putText(
            output_frame,
            f"Frame: {frame_count + 1}/{total_frames}",
            (30, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        writer.write(
            output_frame
        )

        frame_count += 1

    # -------------------------
    # Release Resources
    # -------------------------

    cap.release()

    writer.release()

    # -------------------------
    # Build Agent 1 JSON
    # -------------------------

    agent1_export = {

        "video_path": str(input_path),

        "resolution": [
            width,
            height
        ],

        "fps": fps,

        "total_frames": frame_count,

        "tracks": list(
            tracks_data.values()
        )
    }

    # -------------------------
    # Save JSON
    # -------------------------

    with open(
        json_output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            agent1_export,
            f,
            indent=4,
            ensure_ascii=False
        )

    # -------------------------
    # Final Log
    # -------------------------

    print()
    print("==========================================")
    print("AGENT 1 DONE")
    print("==========================================")
    print(f"Processed frames: {frame_count}")
    print(f"YOLO runs: {yolo_runs}")
    print(f"Detected tracks: {len(tracks_data)}")
    print(f"Saved Video: {output_path}")
    print(f"Saved JSON: {json_output_path}")
    print("==========================================")

    return agent1_export


# ============================================================
# Standalone Test
# ============================================================

if __name__ == "__main__":

    input_video = r"D:\GP\agent2\input\video.mp4"

    track_and_save_video(
        input_path=input_video,
        output_path=DEFAULT_OUTPUT_VIDEO,
        json_output_path=DEFAULT_OUTPUT_JSON
    )