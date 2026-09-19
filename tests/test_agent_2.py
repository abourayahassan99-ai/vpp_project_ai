import json
from pathlib import Path
from agents.agent_2_context_analyzer import ContextAnalyzerAgent

VIDEO_PATH = r"D:\vpp_project_ai\tv_sample.mp4"
OUTPUT_DIR = Path("data")

agent = ContextAnalyzerAgent(
    num_frames=4
)

result = agent.analyze_video(VIDEO_PATH)

print("\n========== FINAL RESULT ==========\n")
print(json.dumps(result, indent=4, ensure_ascii=False))

# ---------------------------------------------------------
# حفظ المخرجات في ملف JSON
# ---------------------------------------------------------
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
output_path = OUTPUT_DIR / "agent2_output.json"

try:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n JSON saved successfully to: {output_path.resolve()}")
except Exception as e:
    print(f"\n Could not save JSON: {e}")