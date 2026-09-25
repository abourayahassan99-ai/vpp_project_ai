from pathlib import Path
import json
import shutil
import uuid

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse

from agents.agent_1_frame_selector.agent import FrameSelectorAgent
from agents.agent_2_context_analyzer.agent import ContextAnalyzerAgent
from agents.agent_3_ad_placer.agent import AdPlacerAgent

# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="VPP AI API",
    description="Virtual Product Placement AI API",
    version="1.0.0"
)


# ============================================================
# DIRECTORIES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

UPLOADS_DIR = BASE_DIR / "uploads"
OUTPUTS_DIR = BASE_DIR / "outputs"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "message": "VPP AI API is running!"
    }


# ============================================================
# PROCESS VIDEO
# ============================================================

@app.post("/process")
async def process_video(
    video: UploadFile = File(...),
    advertisement: UploadFile = File(...),
    prompt: str = Form(...)
):

    # --------------------------------------------------------
    # 1. Create unique job ID
    # --------------------------------------------------------

    job_id = uuid.uuid4().hex[:8]

    # --------------------------------------------------------
    # 2. Save uploaded video
    # --------------------------------------------------------

    video_filename = f"{job_id}_{video.filename}"
    video_path = UPLOADS_DIR / video_filename

    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    # --------------------------------------------------------
    # 3. Save uploaded advertisement
    # --------------------------------------------------------

    ad_filename = f"{job_id}_{advertisement.filename}"
    advertisement_path = UPLOADS_DIR / ad_filename

    with open(advertisement_path, "wb") as buffer:
        shutil.copyfileobj(advertisement.file, buffer)

    # --------------------------------------------------------
    # 4. Run Agent 1
    # --------------------------------------------------------

    agent1 = FrameSelectorAgent()

    agent1_result = agent1.process_video(
        str(video_path)
    )
    # --------------------------------------------------------
    # 5. Run Agent 2
    # --------------------------------------------------------

    agent2 = ContextAnalyzerAgent()

    agent2_result = agent2.analyze_video(
        str(video_path)
    )
     # --------------------------------------------------------
    # 6. Save Agent 1 and Agent 2 outputs
    # --------------------------------------------------------

    agent1_output_path = OUTPUTS_DIR / f"{job_id}_agent1.json"
    agent2_output_path = OUTPUTS_DIR / f"{job_id}_agent2.json"

    with open(agent1_output_path, "w", encoding="utf-8") as f:
        json.dump(
            agent1_result.model_dump(),
            f,
            indent=4,
            ensure_ascii=False
        )

    with open(agent2_output_path, "w", encoding="utf-8") as f:
        json.dump(
            agent2_result,
            f,
            indent=4,
            ensure_ascii=False
        )
    # --------------------------------------------------------
    # 7. Run Agent 3
    # --------------------------------------------------------

    agent3 = AdPlacerAgent()

    result = agent3.run(
        ad_image_path=advertisement_path,
        video_path=video_path,
        render_video=True,
        user_prompt=prompt,
        agent1_output_path=agent1_output_path,
        agent2_output_path=agent2_output_path
    )

    # --------------------------------------------------------
    # 8. Check result
    # --------------------------------------------------------

    if not result.blending_status.success:

        return {
            "status": "failed",
            "job_id": job_id,
            "prompt": prompt,
            "message": result.blending_status.message,
            "result": result.model_dump()
        }

    # --------------------------------------------------------
    # 9. Copy output video to API outputs folder
    # --------------------------------------------------------

    output_video_url = None
    output_image_url = None

    if result.output_video_path:

        source_video = Path(result.output_video_path)

        if source_video.exists():

            final_video_name = f"{job_id}_result.mp4"
            final_video_path = OUTPUTS_DIR / final_video_name

            shutil.copy2(
                source_video,
                final_video_path
            )

            output_video_url = f"/outputs/{final_video_name}"

    # --------------------------------------------------------
    # 10. Copy output image to API outputs folder
    # --------------------------------------------------------

    if result.output_image_path:

        source_image = Path(result.output_image_path)

        if source_image.exists():

            final_image_name = f"{job_id}_result.jpg"
            final_image_path = OUTPUTS_DIR / final_image_name

            shutil.copy2(
                source_image,
                final_image_path
            )

            output_image_url = f"/outputs/{final_image_name}"

    # --------------------------------------------------------
    # 11. Return API response
    # --------------------------------------------------------

    return {
        "status": "completed",
        "job_id": job_id,
        "prompt": prompt,

        "output_video_url": output_video_url,
        "output_image_url": output_image_url,

        "result": result.model_dump()
    }


# ============================================================
# GET OUTPUT FILE
# ============================================================

@app.get("/outputs/{filename}")
def get_output(filename: str):

    # Prevent path traversal
    safe_filename = Path(filename).name

    file_path = OUTPUTS_DIR / safe_filename

    if not file_path.exists():
        return {
            "status": "error",
            "message": "Output file not found."
        }

    return FileResponse(
        path=file_path,
        filename=safe_filename
    )