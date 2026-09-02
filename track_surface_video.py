import cv2
import numpy as np
from ultralytics import YOLO

def track_and_save_video(input_video_path="tv_sample.mp4", output_video_path="data/tracked_output.mp4"):
    print(f"[INFO] Processing input video: {input_video_path}...")
    cap = cv2.VideoCapture(input_video_path)
    
    if not cap.isOpened():
        print(f"[ERROR] Could not open video file: {input_video_path}")
        return

    # Extract original video specs
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    print("[INFO] Loading YOLOv8-seg model...")
    model = YOLO("yolov8n-seg.pt")

    # COCO Class IDs (62: TV, 63: laptop)
    TARGET_CLASSES = [62, 63]
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        results = model(frame, verbose=False)[0]

        if results.masks is not None:
            for idx, class_id in enumerate(results.boxes.cls.tolist()):
                if int(class_id) in TARGET_CLASSES:
                    confidence = float(results.boxes.conf[idx])
                    if confidence >= 0.3:
                        # Extract polygon mask points
                        polygon = results.masks.xy[idx]
                        pts = np.array(polygon, np.int32).reshape((-1, 1, 2))

                        # 1. Draw green outline tracking polygon
                        cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=3)

                        # 2. Add semi-transparent green highlight on screen region
                        overlay = frame.copy()
                        cv2.fillPoly(overlay, [pts], (0, 255, 0))
                        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

                        label = f"{model.names[int(class_id)]} ({confidence:.2f})"
                        cv2.putText(
                            frame, label,
                            (pts[0][0][0], max(30, pts[0][0][1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
                        )
                        break

        out.write(frame)

        if frame_count % 30 == 0 or frame_count == total_frames:
            print(f"[INFO] Rendered frame {frame_count}/{total_frames}")

    cap.release()
    out.release()
    print(f"\n[SUCCESS] Tracked video successfully saved to: {output_video_path}")

if __name__ == "__main__":
    track_and_save_video()
