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
    """تحويل القناع المضلع (Polygon) إلى 4 زوايا (Quad) متناسقة لإسقاط الإعلان."""
    rect = cv2.minAreaRect(polygon)
    box = cv2.boxPoints(rect)
    return np.int32(box).tolist()


def draw_result(frame, result):
    if result is None:
        return frame

    output = frame.copy()
    names = result.names
    boxes = result.boxes

    if boxes is None:
        return output

    masks_xy = result.masks.xy if result.masks is not None else None

    for index in range(len(boxes)):
        box = boxes[index]
        confidence = float(box.conf[0].item())
        class_id = int(box.cls[0].item())
        object_name = names[class_id]
        track_id = int(box.id[0].item()) if box.id is not None else None

        # Polygon
        polygon = None
        if masks_xy is not None and index < len(masks_xy):
            polygon = np.array(masks_xy[index], dtype=np.int32)

        # Bounding box
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = xyxy.tolist()

        # Draw polygon
        if polygon is not None and len(polygon) >= 3:
            overlay = output.copy()
            cv2.fillPoly(overlay, [polygon], (0, 255, 0))
            output = cv2.addWeighted(overlay, 0.18, output, 0.82, 0)
            cv2.polylines(output, [polygon], True, (0, 255, 0), 2)

        # Draw bounding box
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Label
        label = f"{object_name} ID:{track_id} ({confidence:.2f})" if track_id is not None else f"{object_name} ({confidence:.2f})"
        text_y = max(25, y1 - 10)
        cv2.putText(output, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    return output


def track_and_save_video(input_path, output_path=DEFAULT_OUTPUT_VIDEO, json_output_path=DEFAULT_OUTPUT_JSON):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(json_output_path) or ".", exist_ok=True)

    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frame_count = 0
    yolo_runs = 0
    last_result = None

    # هيكل بيانات للتتبع والتصدير لـ Agent 3
    tracks_data = {}

    print("==========================================")
    print("TRACKING VIDEO & EXPORTING JSON")
    print("==========================================")

    while True:
        success, frame = cap.read()
        if not success:
            break

        # تشغيل YOLO كل N فريمات
        if frame_count % DETECTION_INTERVAL == 0:
            results = model.track(frame, persist=True, conf=CONFIDENCE, imgsz=IMAGE_SIZE, verbose=False)

            if results:
                last_result = results[0]
                yolo_runs += 1

                if last_result.boxes is not None and last_result.boxes.id is not None:
                    boxes = last_result.boxes
                    masks_xy = last_result.masks.xy if last_result.masks is not None else []

                    for idx, box in enumerate(boxes):
                        track_id = int(box.id[0].item())
                        class_id = int(box.cls[0].item())
                        class_name = last_result.names[class_id]
                        conf = float(box.conf[0].item())
                        bbox = box.xyxy[0].cpu().numpy().astype(int).tolist()

                        # استخراج المضلع والـ 4 زوايا (Quad)
                        poly_pts = masks_xy[idx].astype(int).tolist() if idx < len(masks_xy) else []
                        quad_pts = polygon_to_quad(masks_xy[idx]) if idx < len(masks_xy) else []

                        if track_id not in tracks_data:
                            tracks_data[track_id] = {
                                "track_id": track_id,
                                "class_name": class_name,
                                "first_seen_frame": frame_count,
                                "frames": []
                            }

                        tracks_data[track_id]["frames"].append({
                            "frame_id": frame_count,
                            "confidence": round(conf, 3),
                            "bbox": bbox,
                            "polygon": poly_pts,
                            "quad": quad_pts
                        })

        # رسم المخرجات لإنشاء فيديو المعاينة
        output_frame = draw_result(frame, last_result)
        cv2.putText(output_frame, f"Frame: {frame_count + 1}/{total_frames}", (30, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

        writer.write(output_frame)
        frame_count += 1

    cap.release()
    writer.release()

    # حفظ بيانات التتبع في ملف JSON
    agent1_export = {
        "video_path": input_path,
        "resolution": [width, height],
        "fps": fps,
        "total_frames": frame_count,
        "tracks": list(tracks_data.values())
    }

    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(agent1_export, f, indent=4, ensure_ascii=False)

    print("\n==========================================")
    print("DONE EXPORTING")
    print("==========================================")
    print(f"Processed frames: {frame_count}")
    print(f"Saved Video: {output_path}")
    print(f"Saved JSON Metadata: {json_output_path}")


if __name__ == "__main__":
    input_video = r"D:\GP\agent2\input\video.mp4"
    track_and_save_video(input_video, DEFAULT_OUTPUT_VIDEO, DEFAULT_OUTPUT_JSON)