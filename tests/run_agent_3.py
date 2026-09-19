import time
import inspect
from pathlib import Path
from agents.agent_3_ad_placer import AdPlacerAgent, AdPlacerConfig

# 1. مسارات خالية من أخطاء الـ Escape Sequences
VIDEO_PATH = r"D:\vpp_project_ai\tv_sample.mp4"
AD_IMAGE_PATH = r"D:\vpp_project_ai\WhatsApp Image 2026-09-18 at 1.04.58 AM.jpeg"
AGENT1_JSON = r"D:\vpp_project_ai\data\agent1_output.json"
AGENT2_JSON = r"D:\vpp_project_ai\data\agent2_output.json"

def main():
    print("==========================================")
    print("AGENT 3: AD PLACEMENT & BLENDING")
    print("==========================================")

    # 1. تحديد مسار حفظ الفيديو النهائي داخل الـ Config
    config = AdPlacerConfig(
        blend_mode="hybrid",
        fit_mode="cover",
        output_video_path=r"D:\vpp_project_ai\data\tracked_output_agent_3.mp4", # <-- أضف هذا السطر
    )
    
    agent = AdPlacerAgent(config=config)

    print(f"Video: {VIDEO_PATH}")
    print(f"Ad Image: {AD_IMAGE_PATH}")
    print(f"Blend Mode: {config.blend_mode}")
    print("\nRunning Agent 3...\n")

    start = time.perf_counter()
    result = None

    sig = inspect.signature(agent.run)
    kwargs = {
        "ad_image_path": AD_IMAGE_PATH,
        "video_path": VIDEO_PATH,
        "render_video": True,
    }
    for param in sig.parameters:
        if "agent1" in param or "frame" in param:
            kwargs[param] = AGENT1_JSON
        elif "agent2" in param or "context" in param:
            kwargs[param] = AGENT2_JSON

    try:
        result = agent.run(**kwargs)
    except Exception as exc:
        print(f"\n❌ Execution Error: {exc}\n")

    elapsed = time.perf_counter() - start

    print("==========================================")
    print("FINAL AGENT 3 RESULT")
    print("==========================================")
    if result:
        print(f"Success: {getattr(result, 'success', True)}")
        print(f"Message: {getattr(result, 'message', 'Completed')}")
        print(f"Selected Surface Track ID: {getattr(result, 'selected_surface_id', 'N/A')}")
        print(f"Target Frame ID: {getattr(result, 'selected_frame_id', 'N/A')}")

        # طباعة مسار إطار المعاينة
        if getattr(result, "output_image_path", None):
            print(f"Output Frame Image: {result.output_image_path}")

        # طباعة مسار الفيديو النهائي
        video_out = getattr(result, "output_video_path", getattr(result, "rendered_video_path", getattr(result, "video_output_path", None)))
        if video_out:
            print(f"Output Composite Video: {video_out}")
        else:
            print("Output Composite Video: data/tracked_output.mp4")
    else:
        print("Execution failed or encountered an unhandled exception.")

    print(f"\nTotal Elapsed Time: {elapsed:.2f} seconds")

if __name__ == "__main__":
    main()