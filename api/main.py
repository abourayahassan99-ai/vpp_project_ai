from pathlib import Path
import json
import shutil
import uuid
import re
from typing import Literal, Optional

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from agents.agent_1_frame_selector.agent import FrameSelectorAgent
from agents.agent_2_context_analyzer.agent import ContextAnalyzerAgent
from agents.agent_3_ad_placer.agent import AdPlacerAgent, PLACEMENT_SURFACE_TYPES
from agents.agent_3_ad_placer.schemas import AdPlacerConfig
from agents.agent_3_ad_placer.tracking import FrameSource

# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="VPP AI API",
    description="Virtual Product Placement AI API",
    version="1.0.0"
)
agent2_instance = None


# ============================================================
# DIRECTORIES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

UPLOADS_DIR = BASE_DIR / "uploads"
OUTPUTS_DIR = BASE_DIR / "outputs"
JOBS_DIR = BASE_DIR / "jobs"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
JOBS_DIR.mkdir(parents=True, exist_ok=True)


class PlacementRequest(BaseModel):
    mode: Literal["direct", "recommendation"] = Field(
        description="direct applies the prompt to the best candidate; recommendation locks to the selected candidate."
    )
    selected_surface_id: Optional[str] = Field(
        default=None,
        description="Required for recommendation mode; optional for the matched candidate from the prompt in the first /jobs/analyze request."
    )
    prompt: Optional[str] = Field(
        default=None,
        description="Optional placement preference/instruction passed to Agent 3."
    )


def _job_dir(job_id: str) -> Path:
    if not re.fullmatch(r"[a-f0-9]{8}", job_id):
        raise HTTPException(status_code=404, detail="Job not found.")
    path = JOBS_DIR / job_id
    if not path.is_dir():
        raise HTTPException(status_code=404, detail="Job not found.")
    return path


def _candidate_explanation(candidate, rank: int) -> dict:
    parts = candidate.score_breakdown
    points = {key: round(float(value) * 100, 1) for key, value in parts.items()}
    scene_fit = points.get("scene_compatibility", 0.0)
    clarity = points.get("sharpness", 0.0)
    if scene_fit >= 18:
        reason = f"{candidate.surface_type} matches the scene context well"
    elif candidate.surface_type in ("tv", "television", "monitor", "screen", "display"):
        reason = f"{candidate.surface_type} is a preferred display surface"
    else:
        reason = f"{candidate.surface_type} is a detected usable surface"
    if clarity < 4:
        reason += "; the preview frame may be relatively blurry"
    score = round(float(candidate.score) * 100, 1)
    risks = []
    if candidate.confidence < 0.55:
        risks.append("Agent 1 has low confidence in this surface detection; check the preview edges.")
    if candidate.area_ratio < 0.003:
        risks.append("The detected surface is small; fine ad text may be hard to read.")
    if score < 60:
        risks.append("This candidate has a lower suitability score than the stronger options.")
    return {
        "candidate_id": str(candidate.object_id),
        "rank": rank,
        "surface_type": candidate.surface_type,
        "frame_id": candidate.frame_id,
        "timestamp_sec": candidate.timestamp_sec,
        "suitability_score": score,
        "score_scale": "0-100; higher is a better fit",
        "score_breakdown_points": points,
        "detection_confidence": round(float(candidate.confidence), 3),
        "reason": reason,
        "risk_notes": risks,
    }


def _make_recommendation_previews(
    job_id: str, video_path: Path, ad_path: Path, candidates: list,
    agent1_path: Path, agent2_path: Path,
) -> list:
    job_dir = _job_dir(job_id)
    results = []
    with FrameSource(video_path) as source:
        for rank, candidate in enumerate(candidates, start=1):
            filename = f"recommendation_{rank}.jpg"
            preview_path = job_dir / filename
            if candidate.object_id is not None:
                preview_config = AdPlacerConfig(
                    agent1_output_path=agent1_path,
                    agent2_output_path=agent2_path,
                    output_image_path=preview_path,
                )
                preview_result = AdPlacerAgent(preview_config).run(
                    ad_image_path=ad_path,
                    video_path=video_path,
                    render_video=False,
                    agent1_output_path=agent1_path,
                    agent2_output_path=agent2_path,
                    selected_surface_id=str(candidate.object_id),
                )
                if preview_result.blending_status.success and preview_path.is_file():
                    item = _candidate_explanation(candidate, rank)
                    item["risk_notes"].extend(
                        str(warning) for warning in
                        preview_result.placement_metrics.get("anchor_warnings", [])
                    )
                    item["preview_url"] = f"/jobs/{job_id}/files/{filename}"
                    item["preview_type"] = "ad_composite"
                    item["preview_scope"] = "anchor_frame_only"
                    results.append(item)
                    continue

            frame = source.read(candidate.frame_id, candidate.timestamp_sec)
            if frame is None:
                continue
            quad = np.asarray(candidate.quad, dtype=np.int32).reshape(-1, 1, 2)
            preview = frame.copy()
            cv2.polylines(preview, [quad], True, (0, 220, 80), 4, cv2.LINE_AA)
            x, y = candidate.quad[0]
            label = f"#{rank} {candidate.surface_type} | score {candidate.score * 100:.0f}/100"
            cv2.putText(preview, label, (max(8, int(x)), max(30, int(y) - 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 80), 2, cv2.LINE_AA)
            if not cv2.imwrite(str(job_dir / filename), preview):
                continue
            item = _candidate_explanation(candidate, rank)
            item["preview_url"] = f"/jobs/{job_id}/files/{filename}"
            item["preview_type"] = "highlighted_surface"
            results.append(item)
    return results


def _prompt_candidate_payload(candidates: list, frame_w: int, frame_h: int) -> list:
    payload = []
    for candidate in candidates:
        center = np.asarray(candidate.quad, dtype=np.float32).mean(axis=0)
        payload.append({
            "candidate_id": str(candidate.object_id),
            "surface_type": candidate.surface_type,
            "suitability_score": round(float(candidate.score) * 100, 1),
            "detection_confidence": round(float(candidate.confidence), 3),
            "area_ratio": round(float(candidate.area_ratio), 5),
            "frame_id": candidate.frame_id,
            "center_x_ratio": round(float(center[0] / max(frame_w, 1)), 3),
            "center_y_ratio": round(float(center[1] / max(frame_h, 1)), 3),
        })
    return payload


def _agent2_mentions_display(agent2_result: dict) -> bool:
    """Return true when Agent 2 saw a plausible planar display needing a detector retry."""
    parts = []
    for item in agent2_result.get("frame_descriptions", []):
        if isinstance(item, dict):
            parts.append(str(item.get("caption", "")))
    scene = agent2_result.get("scene_analysis", {})
    if isinstance(scene, dict):
        for value in scene.values():
            parts.extend(map(str, value if isinstance(value, list) else [value]))
    text = " ".join(parts).lower()
    return re.search(r"\b(tv|television|monitor|screen|display)\b", text) is not None


def _agent2_mentions_chroma_screen(agent2_result: dict) -> bool:
    parts = [str(item.get("caption", "")) for item in agent2_result.get("frame_descriptions", [])
             if isinstance(item, dict)]
    text = " ".join(parts).lower()
    return "green screen" in text or "chroma key" in text


def _append_tv_fallback(agent1_data: dict, fallback_result) -> int:
    """Append high-resolution TV detections with IDs that cannot collide with Agent 1 IDs."""
    surfaces = agent1_data.setdefault("detected_surfaces", [])
    ids = [int(item["object_id"]) for item in surfaces
           if isinstance(item, dict) and str(item.get("object_id", "")).isdigit()]
    next_id = max(ids, default=0) + 1000
    added = 0
    fallback_surfaces = (fallback_result.get("detected_surfaces", [])
                         if isinstance(fallback_result, dict)
                         else fallback_result.detected_surfaces)
    for surface in fallback_surfaces:
        item = surface if isinstance(surface, dict) else surface.model_dump()
        item["object_id"] = next_id
        next_id += 1
        surfaces.append(item)
        added += 1
    agent1_data["total_surfaces"] = len(surfaces)
    labels = [str(item.get("roi", {}).get("surface_type", "")) for item in surfaces]
    agent1_data["feedback_prompt"] = "Agent 1 detected: " + ", ".join(labels) if labels else "Agent 1 did not detect any objects."
    return added


def _detect_green_display(video_path: Path, sample_interval: int = 5) -> dict:
    """Detect a large chroma-green planar screen when the object detector misses the TV class."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"detected_surfaces": []}
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_index = 0
    samples = []
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % sample_interval == 0:
                h, w = frame.shape[:2]
                work_w = min(w, 1280)
                scale_x, scale_y = w / work_w, h / max(1, int(round(h * work_w / w)))
                work_frame = (cv2.resize(frame, (work_w, int(round(h * work_w / w))),
                                         interpolation=cv2.INTER_AREA)
                              if work_w < w else frame)
                hsv = cv2.cvtColor(work_frame, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, np.array([35, 70, 70]), np.array([90, 255, 255]))
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                frame_area = float(work_frame.shape[0] * work_frame.shape[1])
                best = None
                for contour in contours:
                    area = float(cv2.contourArea(contour))
                    if area < 0.01 * frame_area:
                        continue
                    perimeter = cv2.arcLength(contour, True)
                    poly = cv2.approxPolyDP(contour, 0.02 * perimeter, True).reshape(-1, 2)
                    if len(poly) != 4 or not cv2.isContourConvex(poly.reshape(-1, 1, 2)):
                        continue
                    edges = np.linalg.norm(np.roll(poly, -1, axis=0) - poly, axis=1)
                    surf_w = 0.5 * (edges[0] + edges[2])
                    surf_h = 0.5 * (edges[1] + edges[3])
                    aspect = max(surf_w, surf_h) / max(min(surf_w, surf_h), 1.0)
                    x, y, bw, bh = cv2.boundingRect(contour)
                    rectangularity = area / max(float(bw * bh), 1.0)
                    if not (1.15 <= aspect <= 2.8 and rectangularity >= 0.72):
                        continue
                    if best is None or area > best[0]:
                        best = (area, poly)
                if best is not None:
                    poly = best[1].astype(np.float32)
                    poly[:, 0] *= scale_x
                    poly[:, 1] *= scale_y
                    poly = np.rint(poly).astype(int)
                    contour_area = best[0]
                    confidence = float(min(0.96, 0.82 + 0.14 * min(contour_area / (0.08 * frame_area), 1.0)))
                    samples.append({
                        "frame_id": frame_index,
                        "timestamp_sec": frame_index / fps,
                        "confidence": confidence,
                        "polygon": poly.astype(int).tolist(),
                    })
            frame_index += 1
    finally:
        cap.release()

    if not samples:
        return {"detected_surfaces": []}
    best_sample = max(samples, key=lambda item: item["confidence"])
    surface = {
        "object_id": 0,
        "first_seen_frame": samples[0]["frame_id"],
        "first_seen_timestamp_sec": samples[0]["timestamp_sec"],
        "frame_id": best_sample["frame_id"],
        "timestamp_sec": best_sample["timestamp_sec"],
        "blur_score": 100.0,
        "roi": {
            "surface_type": "tv",
            "confidence": best_sample["confidence"],
            "polygon_points": best_sample["polygon"],
            "exact_active_area": True,
        },
        "trajectory": samples,
    }
    return {"detected_surfaces": [surface]}


def _has_agent1_display(agent1_data: dict) -> bool:
    display_names = {"tv", "television", "monitor", "screen", "display"}
    return any(
        str((surface.get("roi") or {}).get("surface_type", "")).strip().lower() in display_names
        for surface in agent1_data.get("detected_surfaces", [])
        if isinstance(surface, dict)
    )


def _run_display_fallback(agent1_data: dict, agent2_result: dict, video_path: Path) -> tuple[int, Optional[str]]:
    """Augment Agent 1 when Agent 2 sees a display that Agent 1 missed."""
    if not _agent2_mentions_display(agent2_result):
        return 0, None
    try:
        added = 0
        if _agent2_mentions_chroma_screen(agent2_result):
            added = _append_tv_fallback(agent1_data, _detect_green_display(video_path))
        if not added:
            tv_result = FrameSelectorAgent(
                detection_confidence=0.20,
                target_class_ids=[62],  # COCO class 62 is "tv" for yolov8n-seg.pt
                detection_interval=15,
                imgsz=768,
            ).process_video(str(video_path))
            added = _append_tv_fallback(agent1_data, tv_result)
        if added:
            return added, "Agent 1 missed the display at the default resolution; a targeted TV/chroma-screen fallback was used."
        return 0, "Agent 2 described a display, but neither targeted TV detection nor the chroma-screen fallback could locate its surface."
    except Exception as exc:
        return 0, f"Targeted display detection fallback failed: {exc}"


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

@app.post(
    "/process",
    summary="Legacy full placement (Agent 3)",
    description="Runs Agent 1, Agent 2, and the full Agent 3 placement pipeline, including video tracking and rendering.",
)
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

    global agent2_instance

    if agent2_instance is None:
        print("Initializing Agent 2 for the first time...")
        # Fewer sampled frames reduce CPU time and memory pressure on laptops.
        agent2_instance = ContextAnalyzerAgent(
            num_frames=8,
            aggregator_model_name="Qwen/Qwen2.5-0.5B-Instruct",
        )

    agent2_result = agent2_instance.analyze_video(
        str(video_path)
)

    # Keep the legacy /process route in sync with /jobs/analyze: Agent 2 can identify a TV
    # that the default Agent 1 pass misses (especially a chroma-green display).
    agent1_data = agent1_result.model_dump()
    process_warnings = []
    if _agent2_mentions_display(agent2_result) and not _has_agent1_display(agent1_data):
        added, warning = _run_display_fallback(agent1_data, agent2_result, video_path)
        if added:
            print(f"Added {added} display candidate(s) using the targeted fallback.")
        if warning:
            process_warnings.append(warning)
     # --------------------------------------------------------
    # 6. Save Agent 1 and Agent 2 outputs
    # --------------------------------------------------------

    agent1_output_path = OUTPUTS_DIR / f"{job_id}_agent1.json"
    agent2_output_path = OUTPUTS_DIR / f"{job_id}_agent2.json"

    with open(agent1_output_path, "w", encoding="utf-8") as f:
        json.dump(
            agent1_data,
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
            "warnings": process_warnings,
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
        "warnings": process_warnings,

        "output_video_url": output_video_url,
        "output_image_url": output_image_url,

        "result": result.model_dump()
    }


# ============================================================
# GET OUTPUT FILE
# ============================================================

@app.post(
    "/jobs/analyze",
    summary="Analyze and return recommendations (or run direct placement)",
    description=(
        "Recommendation mode returns Agent 3 single-frame previews only; it does not render a final video. "
        "After the user selects a surface, POST /jobs/{job_id}/place runs Agent 3's full tracking and video compositor. "
        "Direct mode with a clear prompt runs that same full Agent 3 pipeline within this request."
    ),
)
def analyze_job(
    video: UploadFile = File(...),
    advertisement: UploadFile = File(...),
    mode: Literal["direct", "recommendation"] = Form("recommendation"),
    prompt: Optional[str] = Form(None),
):
    """Analyze uploads; direct mode interprets its prompt and can place in this same request."""
    job_id = uuid.uuid4().hex[:8]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=False)
    video_ext = Path(video.filename or "video.mp4").suffix or ".mp4"
    ad_ext = Path(advertisement.filename or "advertisement.png").suffix or ".png"
    video_path = job_dir / f"video{video_ext}"
    ad_path = job_dir / f"advertisement{ad_ext}"
    with video_path.open("wb") as target:
        shutil.copyfileobj(video.file, target)
    with ad_path.open("wb") as target:
        shutil.copyfileobj(advertisement.file, target)

    warnings = []
    try:
        agent1_result = FrameSelectorAgent().process_video(str(video_path))
        agent1_path = job_dir / "agent1.json"
        agent1_data = agent1_result.model_dump()
        agent1_path.write_text(
            json.dumps(agent1_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent 1 analysis failed: {exc}") from exc

    global agent2_instance
    try:
        if agent2_instance is None:
            print("Initializing lightweight Agent 2 for the first time...")
            agent2_instance = ContextAnalyzerAgent(
                num_frames=8,
                aggregator_model_name="Qwen/Qwen2.5-0.5B-Instruct",
            )
        agent2_result = agent2_instance.analyze_video(str(video_path))
    except Exception as exc:
        # Agent 1 candidates still have useful confidence/clarity scores. Let the
        # client proceed while making the missing scene context explicit.
        warnings.append(f"Scene analysis unavailable; recommendations use visual detection only: {exc}")
        agent2_result = {"frame_descriptions": [], "scene_analysis": {}}
    agent2_path = job_dir / "agent2.json"
    agent2_path.write_text(json.dumps(agent2_result, indent=2, ensure_ascii=False), encoding="utf-8")

    placer = AdPlacerAgent()
    scene_sections = placer._scene_sections(agent2_result)
    frame_w = frame_h = 0
    try:
        with FrameSource(video_path) as source:
            frame_w, frame_h = source.width, source.height
            candidates = placer.rank_surfaces(
                agent1_result.model_dump(), scene_sections, source.width, source.height
            )
        candidates = [
            candidate for candidate in candidates
            if candidate.surface_type.strip().lower() in PLACEMENT_SURFACE_TYPES
        ]
        # On high-resolution footage, Agent 1 can miss a TV that Agent 2 sees.
        if not candidates:
            added, fallback_warning = _run_display_fallback(agent1_data, agent2_result, video_path)
            if added:
                agent1_path.write_text(
                    json.dumps(agent1_data, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                candidates = placer.rank_surfaces(
                    agent1_data, scene_sections, source.width, source.height
                )
                candidates = [
                    candidate for candidate in candidates
                    if candidate.surface_type.strip().lower() in PLACEMENT_SURFACE_TYPES
                ]
            if fallback_warning:
                warnings.append(fallback_warning)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not build placement recommendations: {exc}") from exc

    interpretation = None
    prompt_interpreter_warning = None
    if mode == "direct":
        if prompt and prompt.strip() and candidates:
            candidate_payload = _prompt_candidate_payload(candidates, frame_w, frame_h)
            try:
                interpretation = agent2_instance.interpret_placement_request(
                    prompt.strip(), agent2_result, candidate_payload
                )
            except Exception as exc:
                prompt_interpreter_warning = str(exc)
                interpretation = {
                    "intent_clear": False,
                    "confidence": 0.0,
                    "matched_candidate_ids": [],
                    "target_description": "",
                    "reason": "The request could not be matched reliably to a detected surface.",
                    "clarification_question": "Choose a recommendation or name the surface more specifically.",
                }
        else:
            interpretation = {
                "intent_clear": False,
                "confidence": 0.0,
                "matched_candidate_ids": [],
                "target_description": "",
                "reason": "No clear placement target was provided." if not prompt else "No eligible surface was detected.",
                "clarification_question": "Name a surface such as a monitor, TV, poster, or billboard.",
            }

    # Direct mode only needs previews when the prompt is ambiguous. Recommendation
    # mode always returns the choices. Both use the same saved Agent 1/2 analysis.
    needs_recommendations = mode == "recommendation" or not interpretation or not interpretation["intent_clear"]
    recommendations = []
    if needs_recommendations and candidates:
        try:
            recommendations = _make_recommendation_previews(
                job_id, video_path, ad_path, candidates, agent1_path, agent2_path
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Could not build placement recommendations: {exc}") from exc

    (job_dir / "job.json").write_text(json.dumps({
        "job_id": job_id,
        "video": video_path.name,
        "advertisement": ad_path.name,
        "status": "awaiting_placement",
        "warnings": warnings,
    }, indent=2), encoding="utf-8")
    (job_dir / "recommendations.json").write_text(
        json.dumps(recommendations, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if mode == "direct" and interpretation and interpretation["intent_clear"]:
        placement_response = place_job(
            job_id,
            PlacementRequest(
                mode="direct",
                selected_surface_id=interpretation["matched_candidate_ids"][0],
                prompt=prompt,
            ),
        )
        placement_response["prompt_interpretation"] = interpretation
        placement_response["assistant_message"] = (
            f"I understood your request and placed the ad on {interpretation['target_description'] or 'the matched surface'}."
        )
        return placement_response

    if mode == "direct":
        return {
            "job_id": job_id,
            "status": "needs_recommendation" if recommendations else "no_eligible_surfaces",
            "mode": "direct",
            "prompt": prompt,
            "requires_user_choice": bool(recommendations),
            "prompt_interpretation": interpretation,
            "assistant_message": (
                "I could not match your prompt to one clear available surface. Review these recommendations or make the prompt more specific."
                if recommendations else
                "No supported advertising surface was found in this video."
            ),
            "warnings": warnings + ([{"type": "prompt_interpreter_unavailable", "message": prompt_interpreter_warning}]
                                    if prompt_interpreter_warning else []),
            "recommendations": recommendations,
        }

    return {
        "job_id": job_id,
        "status": "awaiting_placement",
        "mode": "recommendation",
        "message": (
            "Analysis complete. Choose a recommendation or submit direct mode to place the ad."
            if recommendations else
            "Analysis complete, but no supported advertising surface was found. Try a video with a TV, monitor, laptop, poster, or billboard."
        ),
        "warnings": warnings,
        "recommendations": recommendations,
    }


@app.post(
    "/jobs/{job_id}/place",
    summary="Run full Agent 3 placement and video tracking",
    description=(
        "Runs AdPlacerAgent.run(render_video=True) with the selected surface, using Agent 3's surface initialization, "
        "planar tracking, temporal smoothing, edge refinement, per-frame compositing, and video rendering."
    ),
)
def place_job(job_id: str, request: PlacementRequest):
    """Render a direct placement or lock Agent 3 to a recommended surface."""
    job_dir = _job_dir(job_id)
    info_path = job_dir / "job.json"
    if not info_path.exists():
        raise HTTPException(status_code=404, detail="Job metadata not found.")
    job_info = json.loads(info_path.read_text(encoding="utf-8"))
    if request.mode == "recommendation" and request.selected_surface_id is None:
        raise HTTPException(status_code=422, detail="selected_surface_id is required in recommendation mode.")

    config = AdPlacerConfig(
        agent1_output_path=job_dir / "agent1.json",
        agent2_output_path=job_dir / "agent2.json",
        output_image_path=job_dir / "result.jpg",
        output_video_path=job_dir / "result.mp4",
    )
    placer = AdPlacerAgent(config)
    result = placer.run(
        ad_image_path=job_dir / job_info["advertisement"],
        video_path=job_dir / job_info["video"],
        render_video=True,
        user_prompt=request.prompt,
        agent1_output_path=config.agent1_output_path,
        agent2_output_path=config.agent2_output_path,
        selected_surface_id=request.selected_surface_id,
    )
    if not result.blending_status.success:
        job_info["status"] = "placement_failed"
        info_path.write_text(json.dumps(job_info, indent=2), encoding="utf-8")
        raise HTTPException(status_code=422, detail={
            "status": "placement_failed",
            "job_id": job_id,
            "message": result.blending_status.message,
            "result": result.model_dump(),
        })

    job_info["status"] = "completed"
    job_info["selected_surface_id"] = result.selected_surface_id
    job_info["selected_frame_id"] = result.selected_frame_id
    info_path.write_text(json.dumps(job_info, indent=2), encoding="utf-8")

    placement_warnings = list(result.placement_metrics.get("anchor_warnings", []))
    video_info = result.placement_metrics.get("video", {})
    if video_info.get("error"):
        placement_warnings.append(f"Video render warning: {video_info['error']}")

    return {
        "status": "completed",
        "job_id": job_id,
        "mode": request.mode,
        "prompt": request.prompt,
        "selected_surface_id": result.selected_surface_id,
        "selected_frame_id": result.selected_frame_id,
        "placement_score": result.placement_metrics.get("selection_score"),
        "score_breakdown": result.placement_metrics.get("score_breakdown", {}),
        "placement_warnings": placement_warnings,
        "output_image_url": f"/jobs/{job_id}/files/result.jpg",
        "output_video_url": f"/jobs/{job_id}/files/result.mp4" if result.output_video_path else None,
        "result": result.model_dump(),
    }


@app.get("/jobs/{job_id}/files/{filename}")
def get_job_file(job_id: str, filename: str):
    job_dir = _job_dir(job_id)
    safe_name = Path(filename).name
    file_path = job_dir / safe_name
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Job file not found.")
    return FileResponse(path=file_path, filename=safe_name)


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
