import os

import cv2
import numpy as np
from ultralytics import YOLO


MODEL_PATH = "yolov8n-seg.pt"

CONFIDENCE = 0.30

DETECTION_INTERVAL = 5

IMAGE_SIZE = 512

DEFAULT_OUTPUT = "data/tracked_surface_output.mp4"


def draw_result(
    frame,
    result,
):

    if result is None:
        return frame

    output = frame.copy()

    names = result.names

    boxes = result.boxes

    if boxes is None:
        return output

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

        track_id = None

        if box.id is not None:
            track_id = int(
                box.id[0].item()
            )

        # -----------------------------------------------------
        # Polygon
        # -----------------------------------------------------

        polygon = None

        if (
            masks_xy is not None
            and index < len(masks_xy)
        ):

            polygon = np.array(
                masks_xy[index],
                dtype=np.int32,
            )

        # -----------------------------------------------------
        # Bounding box
        # -----------------------------------------------------

        xyxy = (
            box.xyxy[0]
            .cpu()
            .numpy()
            .astype(int)
        )

        x1, y1, x2, y2 = xyxy.tolist()

        # -----------------------------------------------------
        # Draw polygon
        # -----------------------------------------------------

        if polygon is not None and len(polygon) >= 3:

            overlay = output.copy()

            cv2.fillPoly(
                overlay,
                [polygon],
                (0, 255, 0),
            )

            output = cv2.addWeighted(
                overlay,
                0.18,
                output,
                0.82,
                0,
            )

            cv2.polylines(
                output,
                [polygon],
                True,
                (0, 255, 0),
                2,
            )

        # -----------------------------------------------------
        # Draw bounding box
        # -----------------------------------------------------

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        # -----------------------------------------------------
        # Label
        # -----------------------------------------------------

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
            y1 - 10,
        )

        cv2.putText(
            output,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return output


def track_and_save_video(
    input_path,
    output_path=DEFAULT_OUTPUT,
):

    os.makedirs(
        os.path.dirname(output_path)
        or ".",
        exist_ok=True,
    )

    model = YOLO(
        MODEL_PATH
    )

    cap = cv2.VideoCapture(
        input_path
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open video: {input_path}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 30.0

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height),
    )

    frame_count = 0

    yolo_runs = 0

    unique_ids = set()

    last_result = None

    print(
        "=========================================="
    )

    print(
        "TRACKING VIDEO"
    )

    print(
        "=========================================="
    )

    print(
        f"Input: {input_path}"
    )

    print(
        f"Output: {output_path}"
    )

    print(
        f"Resolution: {width}x{height}"
    )

    print(
        f"FPS: {fps:.2f}"
    )

    print(
        f"Total frames: {total_frames}"
    )

    print(
        f"YOLO interval: "
        f"every {DETECTION_INTERVAL} frames"
    )

    print(
        f"Image size: {IMAGE_SIZE}"
    )

    while True:

        success, frame = cap.read()

        if not success:
            break

        # -----------------------------------------------------
        # Run YOLO every N frames
        # -----------------------------------------------------

        if (
            frame_count
            % DETECTION_INTERVAL
            == 0
        ):

            results = model.track(
                frame,
                persist=True,
                conf=CONFIDENCE,
                imgsz=IMAGE_SIZE,
                verbose=False,
            )

            if results:

                last_result = results[0]

                yolo_runs += 1

                if (
                    last_result.boxes is not None
                    and last_result.boxes.id is not None
                ):

                    ids = (
                        last_result.boxes.id
                        .cpu()
                        .numpy()
                        .astype(int)
                        .tolist()
                    )

                    unique_ids.update(ids)

        # -----------------------------------------------------
        # Draw latest tracking result
        # -----------------------------------------------------

        output_frame = draw_result(
            frame,
            last_result,
        )

        # Frame counter
        cv2.putText(
            output_frame,
            (
                f"Frame: "
                f"{frame_count + 1}/"
                f"{total_frames}"
            ),
            (30, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(
            output_frame
        )

        frame_count += 1

    cap.release()

    writer.release()

    print(
        "\n=========================================="
    )

    print(
        "DONE"
    )

    print(
        "=========================================="
    )

    print(
        f"Processed frames: "
        f"{frame_count}"
    )

    print(
        f"YOLO runs: "
        f"{yolo_runs}"
    )

    print(
        f"Unique tracking IDs: "
        f"{len(unique_ids)}"
    )

    print(
        f"Saved to: "
        f"{output_path}"
    )


if __name__ == "__main__":

    input_video = (
        r"D:\GP\agent2\input\video.mp4"
    )

    track_and_save_video(
        input_video,
        DEFAULT_OUTPUT,
    )