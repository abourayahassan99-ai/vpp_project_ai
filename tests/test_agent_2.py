import json

from agents.agent_2_context_analyzer import ContextAnalyzerAgent


VIDEO_PATH = r"D:\GP\agent2\input\video3.mp4"


agent = ContextAnalyzerAgent(
    num_frames=8
)

result = agent.analyze_video(VIDEO_PATH)

print("\n========== FINAL RESULT ==========\n")

print(
    json.dumps(
        result,
        indent=4,
        ensure_ascii=False
    )
)