"""
Agent 3 - Ad Placement and Seamless Blending Agent (vpp_project_ai): orchestrator.

Pipeline position
-----------------
    Agent 1 (surface detection)   -> data/agent1_output.json
    Agent 2 (scene understanding) -> data/agent2_output.json
    Agent 3 (this package)        -> data/ad_placement_output.jpg  (+ optional video)

Module map
----------
    schemas.py    Pydantic models (Agent3Output, BlendingStatus, SurfaceCandidate, LightingParams), config
    geometry.py   polygon validation, corner fitting/ordering, ad preparation (fit_mode), perspective warp
    lighting.py   LAB brightness / contrast / colour-temperature matching, emissive-surface handling
    blending.py   feathered alpha, Poisson clone (normal / mixed), hybrid edge-band blend
    tracking.py   FrameSource, optical-flow + RANSAC homography tracking
    agent.py      this file: reads Agent 1 + Agent 2, scores and selects the surface, orchestrates the rest

Selection score (Agent 1 + Agent 2 combined)
--------------------------------------------
    0.55 * confidence
  + 0.25 * scene compatibility (0.4 type mentioned in Agent 2 text, 0.3 preferred type, 0.3 listed under
                                "Persistent Objects")
  + 0.10 * area (capped at 10 % of the frame)
  + 0.10 * sharpness (blur_score / (blur_score + 100), assuming higher = sharper)

Assumptions
-----------
* ``polygon_points`` are pixel coordinates in the resolution of the frame read here. If every coordinate
  lies in [0, 1.5] they are treated as normalised and scaled by the frame size.
* Video rendering re-uses the key frame's lighting correction and uses feathered alpha blending only
  (Poisson per frame is slow and flickers).
* ``run()`` is the public entry point. Adapt the constructor / signature if your project has a shared
  ``BaseAgent`` class.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from .blending import composite
from .geometry import fit_quad, parse_polygon, polygon_is_valid, prepare_ad
from .lighting import apply_lighting, estimate_lighting
from .schemas import (
    VALID_BLEND_MODES,
    VALID_FIT_MODES,
    AdPlacerConfig,
    Agent3Output,
    BlendingStatus,
    PlacementError,
    PreparedAd,
    SurfaceCandidate,
)
from .tracking import FrameSource, track_surface

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Agent 2 text helpers
# --------------------------------------------------------------------------- #
def _flatten_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.lower()
    if isinstance(value, dict):
        return " ".join(f"{k} {_flatten_text(v)}" for k, v in value.items()).lower()
    if isinstance(value, (list, tuple, set)):
        return " ".join(_flatten_text(v) for v in value)
    return str(value).lower()


def _has_word(text: str, term: str) -> bool:
    return bool(re.search(rf"\b{re.escape(term)}\b", text))


SURFACE_SYNONYMS: Dict[str, Tuple[str, ...]] = {
    "tv": ("tv", "television", "screen", "monitor", "display"),
    "television": ("tv", "television", "screen"),
    "monitor": ("monitor", "screen", "display", "computer"),
    "screen": ("screen", "tv", "monitor", "display"),
    "display": ("display", "screen", "monitor"),
    "wall": ("wall", "walls"),
    "poster": ("poster", "picture", "frame"),
    "billboard": ("billboard", "hoarding", "signboard"),
}


# --------------------------------------------------------------------------- #
# The agent
# --------------------------------------------------------------------------- #
class AdPlacerAgent:
    """Agent 3: selects a surface, then warps, colour-matches and seamlessly blends an ad into it."""

    name = "agent_3_ad_placer"

    def __init__(self, config: Optional[AdPlacerConfig] = None):
        self.cfg = config or AdPlacerConfig()
        if self.cfg.blend_mode not in VALID_BLEND_MODES:
            raise ValueError(f"blend_mode must be one of {VALID_BLEND_MODES}")
        if self.cfg.fit_mode not in VALID_FIT_MODES:
            raise ValueError(f"fit_mode must be one of {VALID_FIT_MODES}")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def run(
        self,
        ad_image_path: Union[str, Path],
        video_path: Optional[Union[str, Path]] = None,
        background_image_path: Optional[Union[str, Path]] = None,
        render_video: bool = False,
    ) -> Agent3Output:
        """
        Place the ad on the best surface.

        ``video_path`` supplies the frames referenced by Agent 1 (``frame_id``). For tests or
        single-image pipelines, ``background_image_path`` can be used instead.
        Expected failures never raise: they come back as ``blending_status.success == False``.
        """
        surface_id: Optional[Union[int, str]] = None
        frame_id: Optional[int] = None
        metrics: Dict[str, Any] = {}
        try:
            source_path = video_path or background_image_path
            if source_path is None:
                raise PlacementError("Provide either video_path or background_image_path.")

            # Inputs: Agent 1 (required) and Agent 2 (optional but strongly recommended) ---------
            agent1 = self._load_json(self.cfg.agent1_output_path, required=True)
            agent2 = self._load_json(self.cfg.agent2_output_path, required=False)
            ad_bgr, ad_alpha = self._load_ad(ad_image_path)

            with FrameSource(source_path) as source:
                # 1. Target selection (Agent 1 candidates x Agent 2 scene context) --------------
                scene_sections = self._scene_sections(agent2)
                candidates = self.rank_surfaces(agent1, scene_sections, source.width, source.height)
                if not candidates:
                    raise PlacementError("No valid target surface found in Agent 1 output.")
                target = candidates[0]
                surface_id, frame_id = target.object_id, target.frame_id
                metrics.update(
                    surface_type=target.surface_type,
                    detection_confidence=round(target.confidence, 4),
                    selection_score=round(target.score, 4),
                    score_breakdown={k: round(v, 4) for k, v in target.score_breakdown.items()},
                    polygon_fit_method=target.fit_method,
                    target_corners=target.quad.round(2).tolist(),
                    num_candidates=len(candidates),
                )

                frame = source.read(target.frame_id, target.timestamp_sec)
                if frame is None:
                    raise PlacementError(
                        f"Could not read frame {target.frame_id} (t={target.timestamp_sec}) from {source.path}."
                    )

                # 2-4. Geometry -> lighting -> blending -----------------------------------------
                prepared, dest_size = prepare_ad(ad_bgr, ad_alpha, target.quad, self.cfg.fit_mode)
                lighting = estimate_lighting(frame, prepared, target.quad, target.surface_type, self.cfg)
                prepared.bgr = apply_lighting(prepared.bgr, lighting)
                composite_img, info = composite(frame, prepared, target.quad, self.cfg.blend_mode, self.cfg)

                metrics.update(
                    transformation_matrix=info["transformation_matrix"],
                    ad_fit_mode=self.cfg.fit_mode,
                    ad_prepared_size=list(dest_size),
                    lighting_adjustment=lighting.to_dict(),
                    blending=info["blending"],
                )

                # 5. Save key frame ---------------------------------------------------------------
                out_path = Path(self.cfg.output_image_path)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                params = [int(cv2.IMWRITE_JPEG_QUALITY), int(self.cfg.jpeg_quality)]
                if not cv2.imwrite(str(out_path), composite_img, params):
                    raise PlacementError(f"Failed to write output image to {out_path}")

                message = "Ad placed successfully."
                if info["blending"].get("note"):
                    message += " " + info["blending"]["note"]

                # Optional video ------------------------------------------------------------------
                video_path_out: Optional[str] = None
                if render_video:
                    video_path_out, video_info = self._safe_render_video(source, target, prepared, agent1)
                    metrics["video"] = video_info
                    if video_path_out is None:
                        message += f" Video not rendered: {video_info.get('error')}"

            return Agent3Output(
                selected_surface_id=surface_id,
                selected_frame_id=frame_id,
                blending_status=BlendingStatus(success=True, message=message),
                output_image_path=str(out_path),
                output_video_path=video_path_out,
                placement_metrics=metrics,
            )
        except PlacementError as exc:
            logger.warning("Agent 3 could not complete placement: %s", exc)
            return self._failure(str(exc), surface_id, frame_id, metrics)
        except Exception as exc:  # noqa: BLE001 - agents must not crash the pipeline
            logger.exception("Agent 3 unexpected error")
            return self._failure(f"Unexpected error: {type(exc).__name__}: {exc}", surface_id, frame_id, metrics)

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_json(path: Path, required: bool) -> Dict[str, Any]:
        path = Path(path)
        if not path.exists():
            if required:
                raise PlacementError(f"Required file not found: {path}")
            logger.warning("Optional file not found: %s (continuing without scene context)", path)
            return {}
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except json.JSONDecodeError as exc:
            if required:
                raise PlacementError(f"Invalid JSON in {path}: {exc}") from exc
            logger.warning("Invalid JSON in %s: %s", path, exc)
            return {}
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _load_ad(path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
        """Load the ad as uint8 BGR plus a float32 alpha channel (all ones if the file has none)."""
        path = Path(path)
        if not path.exists():
            raise PlacementError(f"Ad image not found: {path}")
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise PlacementError(f"Could not decode ad image: {path}")
        if img.dtype != np.uint8:
            img = cv2.convertScaleAbs(img, alpha=255.0 / np.iinfo(img.dtype).max)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        alpha = np.ones(img.shape[:2], dtype=np.float32)
        if img.shape[2] == 4:
            alpha = img[:, :, 3].astype(np.float32) / 255.0
            img = img[:, :, :3]
        return np.ascontiguousarray(img), alpha
    # ------------------------------------------------------------------ #
    # Geometry & Trajectory Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _order_quad(pts: np.ndarray) -> np.ndarray:
        """ترتيب الأركان الأربعة صراحة لمنع انقضاض/قلب الإعلان"""
        pts = pts.reshape(4, 2).astype(np.float32)
        rect = np.zeros((4, 2), dtype=np.float32)
        
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # Top-Left
        rect[2] = pts[np.argmax(s)]  # Bottom-Right

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # Top-Right
        rect[3] = pts[np.argmax(diff)]  # Bottom-Left
        return rect

    @staticmethod
    def _interpolate_trajectory(trajectory: List[Dict[str, Any]], frame_w: int, frame_h: int) -> Dict[int, np.ndarray]:
        """استكمال الفريمات المفقودة بين كل نقطتين في Trajectory الخاص بـ Agent 1"""
        if not trajectory:
            return {}

        sorted_traj = sorted(trajectory, key=lambda x: x["frame_id"])
        frame_quads: Dict[int, np.ndarray] = {}

        for i in range(len(sorted_traj) - 1):
            f_start = sorted_traj[i]["frame_id"]
            f_end = sorted_traj[i + 1]["frame_id"]

            poly_start = parse_polygon(sorted_traj[i]["polygon"], frame_w, frame_h)
            poly_end = parse_polygon(sorted_traj[i + 1]["polygon"], frame_w, frame_h)

            quad_start, _ = fit_quad(poly_start)
            quad_end, _ = fit_quad(poly_end)

            steps = f_end - f_start
            if steps <= 0:
                frame_quads[f_start] = quad_start
                continue

            for step in range(steps):
                f_curr = f_start + step
                t = step / float(steps)
                # استكمال خطي للأركان بين الفريمات
                interp_q = (1.0 - t) * quad_start + t * quad_end
                frame_quads[f_curr] = interp_q

        # إضافة الفريم الأخير
        last_f = sorted_traj[-1]["frame_id"]
        last_poly = parse_polygon(sorted_traj[-1]["polygon"], frame_w, frame_h)
        last_q, _ = fit_quad(last_poly)
        frame_quads[last_f] = last_q

        return frame_quads

    @staticmethod
    def _scene_sections(agent2: Dict[str, Any]) -> Dict[str, str]:
        """
        Lower-cased text per Agent 2 section (``Scene Type``, ``Main Objects``, ``Persistent Objects``,
        ``Visible Surfaces``, ``Spatial Relationships``, ``Scene Context``), keyed by a normalised name.
        """
        analysis = agent2.get("scene_analysis", agent2) if isinstance(agent2, dict) else {}
        sections: Dict[str, str] = {}
        if isinstance(analysis, dict):
            for key, value in analysis.items():
                sections[re.sub(r"[\s_]+", " ", str(key).strip().lower())] = _flatten_text(value)
        return sections

    # ------------------------------------------------------------------ #
    # 1. Target surface selection
    # ------------------------------------------------------------------ #
    def rank_surfaces(
        self,
        agent1: Dict[str, Any],
        scene_sections: Dict[str, str],
        frame_w: int,
        frame_h: int,
    ) -> List[SurfaceCandidate]:
        """Validate every detected surface and return the usable ones, best first."""
        frame_area = float(frame_w * frame_h)
        min_area_px = max(16.0, self.cfg.min_area_ratio * frame_area)
        scene_text = " ".join(scene_sections.values())
        persistent_text = scene_sections.get("persistent objects", "")

        candidates: List[SurfaceCandidate] = []
        for idx, surf in enumerate(agent1.get("detected_surfaces") or []):
            if not isinstance(surf, dict):
                continue
            roi = surf.get("roi") or {}
            object_id = surf.get("object_id", idx)
            confidence = self._to_float(roi.get("confidence", surf.get("confidence")), 0.0)
            surface_type = str(roi.get("surface_type", "unknown")).strip().lower()

            if confidence < self.cfg.min_confidence:
                logger.info("Skip %s: confidence %.2f < %.2f", object_id, confidence, self.cfg.min_confidence)
                continue
            polygon = parse_polygon(roi.get("polygon_points"), frame_w, frame_h)
            if not polygon_is_valid(polygon, min_area_px):
                logger.info("Skip %s: missing / degenerate polygon", object_id)
                continue

            quad, method = fit_quad(polygon)
            area_ratio = float(cv2.contourArea(quad)) / frame_area
            if area_ratio < self.cfg.min_area_ratio or not cv2.isContourConvex(quad.reshape(-1, 1, 2)):
                logger.info("Skip %s: quad too small or non-convex", object_id)
                continue

            frame_id = surf.get("frame_id", surf.get("first_seen_frame", 0))
            first_seen = surf.get("first_seen_frame")
            cand = SurfaceCandidate(
                object_id=object_id,
                frame_id=int(frame_id) if frame_id is not None else 0,
                first_seen_frame=int(first_seen) if first_seen is not None else None,
                timestamp_sec=self._to_float(surf.get("timestamp_sec"), None),
                blur_score=self._to_float(surf.get("blur_score"), None),
                surface_type=surface_type,
                confidence=confidence,
                polygon=polygon,
                quad=quad,
                fit_method=method,
                area_ratio=area_ratio,
            )
            self._score(cand, scene_text, persistent_text)
            candidates.append(cand)

        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates

    def _score(self, cand: SurfaceCandidate, scene_text: str, persistent_text: str) -> None:
        terms = SURFACE_SYNONYMS.get(cand.surface_type, (cand.surface_type,))
        in_scene = float(any(_has_word(scene_text, t) for t in terms))
        persistent = float(any(_has_word(persistent_text, t) for t in terms))
        preferred = float(cand.surface_type in self.cfg.preferred_surface_types)
        compat = 0.4 * in_scene + 0.3 * preferred + 0.3 * persistent
        area = min(1.0, cand.area_ratio / 0.10)
        sharp = 0.5 if cand.blur_score is None else cand.blur_score / (cand.blur_score + 100.0)

        cand.score_breakdown = {
            "confidence": 0.55 * cand.confidence,
            "scene_compatibility": 0.25 * compat,
            "area": 0.10 * area,
            "sharpness": 0.10 * sharp,
        }
        cand.score = float(sum(cand.score_breakdown.values()))

    @staticmethod
    def _to_float(value: Any, default: Optional[float]) -> Optional[float]:
        try:
            out = float(value)
            return out if math.isfinite(out) else default
        except (TypeError, ValueError):
            return default

    # ------------------------------------------------------------------ #
    # 5. Optional video rendering (tracking.py finds the quads, blending.py draws them)
    # ------------------------------------------------------------------ #
    def _safe_render_video(
        self, source: FrameSource, target: SurfaceCandidate, prepared: PreparedAd, agent1: Dict[str, Any]
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        if not source.is_video:
            return None, {"rendered": False, "error": "background is a still image, not a video"}
        try:
            return self.render_video(source, target, prepared, agent1)
        except Exception as exc:
            logger.exception("Video rendering failed")
            return None, {"rendered": False, "error": f"{type(exc).__name__}: {exc}"}

    def render_video(
        self, 
        source: FrameSource, 
        target: SurfaceCandidate, 
        prepared: PreparedAd,
        agent1_data: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """Track the surface and render the ad dynamically with smooth tracking."""
        
        # 1. التتبع بواسطة Optical Flow
        tracked_quads = track_surface(
            source,
            key_idx=target.frame_id,
            key_quad=target.quad,
            first_seen_frame=target.first_seen_frame,
            max_frames=self.cfg.video_max_frames,
            
        )

        # 2. استخراج واستكمال المسار من Agent 1
        traj_quads = {}
        for surf in agent1_data.get("detected_surfaces") or []:
            if surf.get("object_id") == target.object_id and "trajectory" in surf:
                traj_quads = self._interpolate_trajectory(surf["trajectory"], source.width, source.height)
                break

        # دمج التتبع: الأولوية لـ Optical Flow ثم Trajectory لملء الفجوات
        all_quads = {**traj_quads, **tracked_quads}

        out_path = Path(self.cfg.output_video_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*self.cfg.video_fourcc),
            source.fps or 25.0,
            (source.width, source.height),
        )

        if not writer.isOpened():
            raise PlacementError(f"VideoWriter could not open {out_path}.")

        written = 0
        prev_quad: Optional[np.ndarray] = None
        alpha = 0.65  # معامل التنعيم (EMA) لتثبيت الزوم ومنع الاهتزاز

        try:
            idx = 0
            while True:
                frame = source.read(idx)
                if frame is None:
                    break

                quad = all_quads.get(idx)
                if quad is not None:
                    # ضبط ترتيب الأركان
                    ordered_q = self._order_quad(quad)

                    # تطبيق التنعيم (EMA) لتتبع سلس مع حركة الكاميرا والـ Zoom
                    if prev_quad is not None:
                        smoothed_q = alpha * ordered_q + (1.0 - alpha) * prev_quad
                    else:
                        smoothed_q = ordered_q

                    prev_quad = smoothed_q
                    frame, _ = composite(frame, prepared, smoothed_q, "alpha", self.cfg)
                else:
                    prev_quad = None  # إعادة تعيين التنعيم في حال اختفاء الشاشة

                writer.write(frame)
                written += 1
                idx += 1
        finally:
            writer.release()

        return str(out_path), {
            "rendered": True,
            "frames_written": written,
            "tracked_frames": len(all_quads),
        }

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _failure(
        self,
        message: str,
        selected_surface_id: Optional[Union[int, str]] = None,
        selected_frame_id: Optional[int] = None,
        placement_metrics: Optional[Dict[str, Any]] = None,
    ) -> Agent3Output:
        """Helper to format a standardized failure output."""
        return Agent3Output(
            selected_surface_id=selected_surface_id,
            selected_frame_id=selected_frame_id,
            blending_status=BlendingStatus(success=False, message=message),
            output_image_path=None,
            output_video_path=None,
            placement_metrics=placement_metrics or {},
        )

# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Agent 3: ad placement and seamless blending")
    parser.add_argument("--ad", required=True, help="Path to the ad image (PNG with alpha supported)")
    parser.add_argument("--video", help="Source video containing the key frame")
    parser.add_argument("--background", help="Still image to use instead of a video")
    parser.add_argument("--agent1", default="data/agent1_output.json")
    parser.add_argument("--agent2", default="data/agent2_output.json")
    parser.add_argument("--out-image", default="data/ad_placement_output.jpg")
    parser.add_argument("--out-video", default="data/ad_placement_output.mp4")
    parser.add_argument("--blend-mode", default="hybrid", choices=VALID_BLEND_MODES)
    parser.add_argument("--render-video", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = AdPlacerConfig(
        agent1_output_path=Path(args.agent1),
        agent2_output_path=Path(args.agent2),
        output_image_path=Path(args.out_image),
        output_video_path=Path(args.out_video),
        blend_mode=args.blend_mode,
    )
    result = AdPlacerAgent(config).run(
        ad_image_path=args.ad,
        video_path=args.video,
        background_image_path=args.background,
        render_video=args.render_video,
    )
    print(json.dumps(result.model_dump(), indent=2, default=str))
    return 0 if result.blending_status.success else 1


if __name__ == "__main__":
    raise SystemExit(main())