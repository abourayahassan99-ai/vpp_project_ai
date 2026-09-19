import json
import os
import time
from pathlib import Path
from agents.agent_1_frame_selector.agent import FrameSelectorAgent

VIDEO_PATH = r"D:\vpp_project_ai\tv_sample.mp4"
MODEL_PATH = "yolov8n-seg.pt"
CONFIDENCE = 0.25
DETECTION_INTERVAL = 1
IMAGE_SIZE = 512
OUTPUT_DIR = Path("data")

def main():
    print("==========================================")
    print("AGENT 1: OBJECT DISCOVERY + TRACKING")
    print("==========================================")

    agent = FrameSelectorAgent(
        model_path=MODEL_PATH,
        detection_confidence=CONFIDENCE,
        target_class_ids=None,
        blur_threshold=15.0,
        detection_interval=DETECTION_INTERVAL,
        imgsz=IMAGE_SIZE,
        blur_resize_width=640,
    )

    print(f"Video: {VIDEO_PATH}")
    print(f"Model: {MODEL_PATH}")
    print(f"Confidence: {CONFIDENCE}")
    print(f"YOLO interval: every {DETECTION_INTERVAL} frame(s)")
    print(f"YOLO image size: {IMAGE_SIZE}")

    print("\nStarting Agent 1...\n")

    start = time.perf_counter()
    result = agent.process_video(VIDEO_PATH)
    elapsed = time.perf_counter() - start

    print("\n==========================================")
    print("FINAL AGENT 1 RESULT")
    print("==========================================")
    print(f"Total unique objects: {result.total_surfaces}")

    for item in result.detected_surfaces:
        print("\n------------------------------------------")
        print(f"Object: {item.roi.surface_type}")
        print(f"Track ID: {item.object_id}")
        print(f"Confidence: {item.roi.confidence:.3f}")
        print(f"First detected frame: {item.first_seen_frame} ({item.first_seen_timestamp_sec:.2f}s)")
        print(f"Best observation frame: {item.frame_id} ({item.timestamp_sec:.2f}s)")
        print(f"Blur score: {item.blur_score:.2f}")
        print(f"Polygon points: {len(item.roi.polygon_points)}")

    print("\n==========================================")
    print("FEEDBACK")
    print("==========================================")
    print(result.feedback_prompt)

    print("\n==========================================")
    print("PERFORMANCE METRICS")
    print("==========================================")
    print(f"Total Elapsed Time: {elapsed:.2f} seconds")

    # حفظ مخرجات JSON بشكل آمن
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "agent1_output.json"

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result.model_dump(), f, ensure_ascii=False, indent=2)
        print(f"\nJSON saved to: {output_path.resolve()}")
    except Exception as e:
        print(f"\nCould not save JSON: {e}")

if __name__ == "__main__":
    main()