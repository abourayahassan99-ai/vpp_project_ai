import cv2
import numpy as np
from agents.agent_1_frame_selector.agent import FrameSelectorAgent

def test_real_video():
    video_path = "tv_sample.mp4"
    output_image_path = "data/candidate_output.jpg"

    print(f"[INFO] Initializing Agent 1 on {video_path}...")
    agent = FrameSelectorAgent(yolo_model_path="yolov8n-seg.pt", blur_threshold=15.0)

    # Process video every 5 frames
    result = agent.extract_candidate_frame(video_path, frame_interval=5)

    if result is None:
        print("[FAIL] Agent 1 did not find any clear frame with a target surface.")
        return

    print("\n=== AGENT 1 EXTRACTION SUCCESS ===")
    print(f"Selected Frame ID : {result.frame_id}")
    print(f"Timestamp        : {result.timestamp_sec} seconds")
    print(f"Clarity Score    : {result.blur_score}")
    print(f"Target Surface   : {result.roi.surface_type} (Conf: {result.roi.confidence})")
    print(f"Polygon Points   : {result.roi.polygon_points}")

    # Draw ROI onto the frame to visually verify detection
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, result.frame_id)
    ret, frame = cap.read()
    cap.release()

    if ret and frame is not None:
        pts = np.array(result.roi.polygon_points, np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=3)
        cv2.putText(
            frame,
            f"{result.roi.surface_type} ({result.roi.confidence})",
            (result.roi.polygon_points[0][0], max(20, result.roi.polygon_points[0][1] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        cv2.imwrite(output_image_path, frame)
        print(f"[SUCCESS] Visual verification image saved to: {output_image_path}")

if __name__ == "__main__":
    test_real_video()