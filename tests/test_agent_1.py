import json
import time

from agents.agent_1_frame_selector.agent import FrameSelectorAgent


VIDEO_PATH = r"D:\GP\agent2\input\video.mp4"
MODEL_PATH = "yolov8n-seg.pt"

CONFIDENCE = 0.30

# Speed optimization
DETECTION_INTERVAL = 5
IMAGE_SIZE = 512


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
    print("Target classes: ALL YOLO classes")
    print(
        f"YOLO interval: "
        f"every {DETECTION_INTERVAL} frame(s)"
    )
    print(
        f"YOLO image size: "
        f"{IMAGE_SIZE}"
    )

    print("\nStarting Agent 1...\n")

    start = time.perf_counter()

    result = agent.process_video(
        VIDEO_PATH
    )

    elapsed = time.perf_counter() - start

    # ---------------------------------------------------------
    # Final result
    # ---------------------------------------------------------

    print("\n==========================================")
    print("FINAL AGENT 1 RESULT")
    print("==========================================")

    print(
        f"Total unique objects: "
        f"{result.total_surfaces}"
    )

    for item in result.detected_surfaces:

        print("\n------------------------------------------")

        print(
            f"Object: "
            f"{item.roi.surface_type}"
        )

        print(
            f"Track ID: "
            f"{item.object_id}"
        )

        print(
            f"Confidence: "
            f"{item.roi.confidence:.3f}"
        )

        print(
            f"First detected frame: "
            f"{item.first_seen_frame}"
        )

        print(
            f"First detected time: "
            f"{item.first_seen_timestamp_sec:.2f}s"
        )

        print(
            f"Best observation frame: "
            f"{item.frame_id}"
        )

        print(
            f"Best observation time: "
            f"{item.timestamp_sec:.2f}s"
        )

        print(
            f"Blur score: "
            f"{item.blur_score:.2f}"
        )

        print(
            f"Polygon points: "
            f"{len(item.roi.polygon_points)}"
        )

        print(
            f"Polygon: "
            f"{item.roi.polygon_points[:5]}"
        )

    # ---------------------------------------------------------
    # Feedback
    # ---------------------------------------------------------

    print("\n==========================================")
    print("FEEDBACK")
    print("==========================================")

    print(result.feedback_prompt)

    # ---------------------------------------------------------
    # Performance
    # ---------------------------------------------------------

    print("\n==========================================")
    print("TOTAL TEST TIME")
    print("==========================================")

    print(
        f"{elapsed:.2f} seconds"
    )

    # ---------------------------------------------------------
    # Optional JSON export
    # ---------------------------------------------------------

    output_path = (
        "data/agent1_output.json"
    )

    json_data = result.model_dump()

    try:

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                json_data,
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(
            f"\nJSON saved to: "
            f"{output_path}"
        )

    except Exception as e:

        print(
            f"\nCould not save JSON: {e}"
        )


if __name__ == "__main__":
    main()