"""
Video placement driver for Agent 3.

    cuts -> anchors (one per shot) -> planar tracker per anchor -> merge -> per-frame composite -> mp4

Stages are deliberately separate:
    1. surface_init.build_anchor_quad   (Agent-1 polygon or manual corners -> ordered quad)
    2. tracking.PlanarSurfaceTracker     (corner tracking + homography + recovery + smoothing)
    3. occlusion.OcclusionEstimator      (optional foreground mask in plane space)
    4. blending.composite                (perspective warp + AA mask + opacity + occlusion)

Edited videos contain cuts. A tracker cannot follow a surface across a cut, so every shot that the first
anchor does not reach gets its own anchor, seeded from Agent 1's per-frame polygons of the same object
(``anchor_candidates``). Every anchor keeps its own lighting, ad preparation and occlusion reference.

    python -m ad_placer.video_placer --video in.mp4 --ad ad.png --anchor-frame 170 \
        --quad "102,332 709,347 707,692 87,681" --out out.mp4
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .blending import composite
from .geometry import estimate_plane_aspect, parse_polygon, polygon_is_valid
from .lighting import apply_lighting, estimate_lighting
from .occlusion import OcclusionEstimator
from .schemas import AdPlacerConfig, PlacementError, PreparedAd
from .shots import detect_cuts, shot_ranges
from .surface_init import build_anchor_quad
from .tracking import PlanarSurfaceTracker, TrackResult

logger = logging.getLogger(__name__)

_FRONTAL_SCREEN_ASPECT = 16.0 / 9.0


def load_ad(path: str):
    """Load an ad image as (uint8 BGR, float32 alpha in [0,1]); honours PNG transparency."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise PlacementError(f"Could not read ad image: {path}")
    if img.dtype != np.uint8:
        img = cv2.convertScaleAbs(img, alpha=255.0 / np.iinfo(img.dtype).max)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return img[..., :3].copy(), img[..., 3].astype(np.float32) / 255.0
    return img, np.ones(img.shape[:2], np.float32)


def _read_frame(video_path: str, idx: int) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def _find_ffmpeg() -> Optional[str]:
    """System ffmpeg, else the binary bundled with the optional ``imageio-ffmpeg`` package."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg  # optional:  pip install imageio-ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001 - any import / lookup failure just means "not available"
        return None


def _reencode_h264(tmp_path: str, source_path: str, out_path: str) -> bool:
    """Re-encode to H.264 (+ copy the source audio). False if no ffmpeg is available (output stays MPEG-4)."""
    exe = _find_ffmpeg()
    if exe is None:
        logger.warning("No ffmpeg found: the output will be MPEG-4 Part 2 (softer, blockier than H.264). "
                       "Install ffmpeg or run `pip install imageio-ffmpeg`.")
        return False
    cmd = [exe, "-y", "-loglevel", "error", "-i", tmp_path, "-i", str(source_path),
           "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "libx264", "-preset", "medium", "-crf", "16",
           "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", out_path]
    try:
        subprocess.run(cmd, check=True)
        return True
    except (subprocess.CalledProcessError, OSError) as exc:
        logger.warning("ffmpeg re-encode failed (%s); keeping the MPEG-4 output.", exc)
        return False


# --------------------------------------------------------------------------- #
# Per-anchor rendering context (ad preparation, lighting, occlusion reference)
# --------------------------------------------------------------------------- #
@dataclass
class _Anchor:
    frame_idx: int
    quad: np.ndarray
    levels: List[PreparedAd]          # mip pyramid of the lighting-corrected ad, largest first
    occ: Optional[OcclusionEstimator]
    exclude_interior: bool
    lighting: Dict[str, Any]
    aspect: float
    note: str = ""


def _surface_aspect(quad: np.ndarray, frame_shape, surface_type: str, cfg: AdPlacerConfig) -> float:
    if cfg.surface_aspect:
        return float(cfg.surface_aspect)
    est = estimate_plane_aspect(quad, frame_shape)
    if surface_type in cfg.emissive_surface_types and abs(est / _FRONTAL_SCREEN_ASPECT - 1.0) < 0.20:
        return _FRONTAL_SCREEN_ASPECT  # screens are 16:9; the estimate only confirms it is not a portrait/ultrawide
    return est


def _build_levels(ad_bgr: np.ndarray, ad_alpha: np.ndarray, aspect: float, cfg: AdPlacerConfig) -> List[PreparedAd]:
    """Crop the ad to the surface's true aspect ('cover') or stretch it, then build a mip pyramid."""
    ah, aw = ad_bgr.shape[:2]
    if cfg.fit_mode == "cover":
        if aw / ah > aspect:
            nw = max(1, int(round(ah * aspect)))
            x0 = (aw - nw) // 2
            ad_bgr, ad_alpha = ad_bgr[:, x0:x0 + nw], ad_alpha[:, x0:x0 + nw]
        else:
            nh = max(1, int(round(aw / aspect)))
            y0 = (ah - nh) // 2
            ad_bgr, ad_alpha = ad_bgr[y0:y0 + nh], ad_alpha[y0:y0 + nh]
    ah, aw = ad_bgr.shape[:2]
    long_px = int(min(cfg.ad_max_render_long_side, max(aw, ah)))   # never upscale beyond the ad's own resolution
    if aspect >= 1:
        w0, h0 = long_px, max(16, int(round(long_px / aspect)))
    else:
        w0, h0 = max(16, int(round(long_px * aspect))), long_px
    levels = []
    w, h = w0, h0
    while True:
        interp = cv2.INTER_AREA if (aw * ah > w * h) else cv2.INTER_CUBIC
        bgr = cv2.resize(ad_bgr, (w, h), interpolation=interp)
        alpha = np.clip(cv2.resize(ad_alpha, (w, h), interpolation=interp), 0, 1).astype(np.float32)
        levels.append(PreparedAd(bgr=bgr, alpha=alpha))
        if max(w, h) <= 96:
            break
        w, h = max(8, w // 2), max(8, h // 2)
    return levels


def _pick_level(levels: List[PreparedAd], quad: np.ndarray) -> PreparedAd:
    """Smallest pyramid level that is still at least as wide as the surface on screen (no aliasing, no blur)."""
    qw = 0.5 * (np.linalg.norm(quad[1] - quad[0]) + np.linalg.norm(quad[2] - quad[3]))
    qh = 0.5 * (np.linalg.norm(quad[3] - quad[0]) + np.linalg.norm(quad[2] - quad[1]))
    need = max(qw, qh * levels[0].bgr.shape[1] / levels[0].bgr.shape[0])
    best = levels[0]
    for lv in levels:
        if lv.bgr.shape[1] >= need:
            best = lv
        else:
            break
    return best


def _make_anchor(frame: np.ndarray, idx: int, quad: np.ndarray, ad_bgr, ad_alpha, surface_type: str,
                 cfg: AdPlacerConfig, aspect: Optional[float] = None) -> _Anchor:
    emissive = surface_type in cfg.emissive_surface_types
    static_ok = OcclusionEstimator.is_usable_for(frame, quad)
    aspect = aspect or _surface_aspect(quad, frame.shape, surface_type, cfg)
    levels = _build_levels(ad_bgr, ad_alpha, aspect, cfg)
    light = estimate_lighting(frame, levels[-1] if len(levels) == 1 else levels[min(1, len(levels) - 1)],
                              quad, surface_type, cfg)
    levels = [PreparedAd(bgr=apply_lighting(lv.bgr, light), alpha=lv.alpha) for lv in levels]
    occ, note = None, "disabled by config"
    if cfg.use_occlusion:
        if static_ok:
            occ, note = OcclusionEstimator(frame, quad, cfg), "enabled"
        else:
            note = "disabled: surface is bright / shows content (background subtraction unreliable)"
    return _Anchor(idx, quad.astype(np.float32), levels, occ, emissive and not static_ok, light.to_dict(), aspect, note)


# --------------------------------------------------------------------------- #
# Anchor candidates (Agent 1 trajectory) and merging
# --------------------------------------------------------------------------- #
def _candidate_score(c: Dict[str, Any], w: int, h: int) -> float:
    poly = parse_polygon(c.get("polygon"), w, h)
    if poly is None or not polygon_is_valid(poly):
        return -1.0
    x, y, bw, bh = cv2.boundingRect(poly.astype(np.int32))
    margin = 4
    touches = sum([x <= margin, y <= margin, x + bw >= w - margin, y + bh >= h - margin])
    size = min(1.0, ((bw * bh) / float(w * h) / 0.03) ** 0.5)
    return float(c.get("confidence", 0.0)) * (1.0 - 0.25 * touches) * (0.3 + 0.7 * size)


def _merge(results: List[Tuple[int, TrackResult]]) -> Tuple[TrackResult, Dict[int, int]]:
    """Per frame, keep the result with the highest opacity (then confidence). Returns (merged, source anchor)."""
    base = results[0][1]
    merged = TrackResult(total_frames=base.total_frames, fps=base.fps, size=base.size)
    source: Dict[int, int] = {}
    for aid, r in results:
        for f, q in r.quads.items():
            o, c = r.opacity.get(f, 0.0), r.confidence.get(f, 0.0)
            if f not in merged.quads or (o, c) > (merged.opacity[f], merged.confidence[f]):
                merged.quads[f], merged.opacity[f], merged.confidence[f] = q, o, c
                source[f] = aid
        for f, st in r.states.items():
            merged.states.setdefault(f, st)
    return merged, source


def place_ad_in_video(
    video_path: str,
    ad_path: str,
    output_path: str,
    anchor_frame_idx: int,
    polygon_points: Any = None,
    manual_quad: Any = None,
    surface_type: str = "tv",
    cfg: Optional[AdPlacerConfig] = None,
    debug_video_path: Optional[str] = None,
    reencode_h264: bool = True,
    anchor_candidates: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Track the surface through ``video_path`` and render the ad onto it. Returns a metrics dict (for
    ``Agent3Output.placement_metrics``). Raises ``PlacementError`` for expected failures.

    ``anchor_candidates``: optional list of ``{"frame_id", "polygon", "confidence"}`` (Agent 1's trajectory of
    the same object). Used to start an extra anchor in every shot the primary anchor cannot reach.
    """
    cfg = cfg or AdPlacerConfig()
    t_start = time.perf_counter()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise PlacementError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    anchor = _read_frame(video_path, anchor_frame_idx)
    if anchor is None:
        raise PlacementError(f"Could not read anchor frame {anchor_frame_idx}")

    # 1. primary anchor ---------------------------------------------------------- #
    quad0, method, warnings = build_anchor_quad(anchor, polygon_points, manual_quad, surface_type, cfg)
    ad_bgr, ad_alpha = load_ad(ad_path)
    anchors: Dict[int, _Anchor] = {0: _make_anchor(anchor, anchor_frame_idx, quad0, ad_bgr, ad_alpha, surface_type, cfg)}
    shared_aspect = anchors[0].aspect

    # 2. tracking: primary over the whole video, then one extra anchor per uncovered shot -------- #
    tracker = PlanarSurfaceTracker(cfg)
    primary = tracker.run(video_path, anchor_frame_idx, quad0, anchor_frame=anchor,
                          exclude_interior=anchors[0].exclude_interior)
    results: List[Tuple[int, TrackResult]] = [(0, primary)]
    shot_log: List[Dict[str, Any]] = []
    cuts: List[int] = []
    if cfg.multi_anchor and anchor_candidates:
        cuts = detect_cuts(video_path)
        cands = [c for c in anchor_candidates if c.get("polygon") is not None]
        for (s0, s1) in shot_ranges(cuts, total):
            if len(anchors) - 1 >= cfg.max_extra_anchors:
                break
            merged, _ = _merge(results)
            in_shot = [c for c in cands if s0 <= int(c["frame_id"]) <= s1]
            if not in_shot:
                continue
            uncovered = [c for c in in_shot if merged.opacity.get(int(c["frame_id"]), 0.0) < 0.5]
            if len(uncovered) < max(1, 0.25 * len(in_shot)):
                continue
            scored = sorted(((_candidate_score(c, W, H), c) for c in in_shot), key=lambda t: -t[0])
            for sc, best in scored[:2]:  # best candidate, then one fallback
                if sc < cfg.min_confidence * 0.3:
                    break
                fidx = int(best["frame_id"])
                fr = _read_frame(video_path, fidx)
                if fr is None:
                    continue
                try:
                    q, _, _ = build_anchor_quad(fr, best["polygon"], None, surface_type, cfg)
                except PlacementError as exc:
                    logger.info("shot %d-%d: candidate %d rejected (%s)", s0, s1, fidx, exc)
                    continue
                aid = len(anchors)
                anchors[aid] = _make_anchor(fr, fidx, q, ad_bgr, ad_alpha, surface_type, cfg, aspect=shared_aspect)
                res = tracker.run(video_path, fidx, q, anchor_frame=fr, start_frame=s0, end_frame=s1,
                                  exclude_interior=anchors[aid].exclude_interior)
                results.append((aid, res))
                nvis = sum(1 for o in res.opacity.values() if o > 0.5)
                shot_log.append(dict(shot=[s0, s1], anchor_frame=fidx, visible_frames=nvis))
                logger.info("shot %d-%d: extra anchor at frame %d -> %d visible frames", s0, s1, fidx, nvis)
                if nvis > 0.3 * (s1 - s0 + 1):
                    break
    tr, source = _merge(results)
    tr.metrics = dict(primary.metrics)
    if not tr.quads:
        raise PlacementError("Surface could not be tracked in this video (no frame passed validation).")

    # 3. render ------------------------------------------------------------------ #
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    tmp_dir = tempfile.mkdtemp(prefix="adplace_")
    tmp_out = os.path.join(tmp_dir, "render.mp4")
    writer = cv2.VideoWriter(tmp_out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    dbg = cv2.VideoWriter(debug_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H)) if debug_video_path else None
    if not writer.isOpened():
        raise PlacementError(f"Could not create output video: {tmp_out}")

    cap = cv2.VideoCapture(str(video_path))
    n_drawn = n_occ = 0
    last_end = min(total, cfg.video_max_frames) if cfg.video_max_frames else total
    f = 0
    cur_src = None
    try:
        while f < last_end:
            ok, frame = cap.read()
            if not ok:
                break
            q = tr.quads.get(f)
            out = frame
            if q is not None and tr.opacity.get(f, 0.0) > 0.001:
                aid = source[f]
                ctx = anchors[aid]
                if aid != cur_src and ctx.occ is not None:
                    ctx.occ.reset()
                cur_src = aid
                prev_q, next_q = tr.quads.get(f - 1), tr.quads.get(f + 1)
                if source.get(f - 1) != aid:
                    prev_q = None
                if source.get(f + 1) != aid:
                    next_q = None
                motion = None
                if prev_q is not None or next_q is not None:
                    a, b = (prev_q if prev_q is not None else q), (next_q if next_q is not None else q)
                    span = (1 if prev_q is not None else 0) + (1 if next_q is not None else 0)
                    motion = (b.mean(0) - a.mean(0)) / max(span, 1)
                mask = ctx.occ.estimate(frame, q) if ctx.occ is not None else None
                out, info = composite(frame, _pick_level(ctx.levels, q), q, cfg.blend_mode, cfg,
                                      opacity=tr.opacity[f], occlusion=mask, motion=motion)
                n_drawn += info["blending"]["status"] == "placed"
                n_occ += int(mask is not None and info["blending"].get("occluded_fraction", 0) > 0.005)
            else:
                cur_src = None
            writer.write(out)
            if dbg is not None:
                d = out.copy()
                if q is not None:
                    cv2.polylines(d, [np.round(q).astype(np.int32)], True, (0, 0, 255), 2, cv2.LINE_AA)
                    for p in q:
                        cv2.circle(d, tuple(int(v) for v in p), 6, (0, 255, 0), -1)
                cv2.putText(d, f"{f} anchor={source.get(f, '-')} c={tr.confidence.get(f, 0):.2f} o={tr.opacity.get(f, 0):.2f}",
                            (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                dbg.write(d)
            f += 1
    finally:
        cap.release()
        writer.release()
        if dbg is not None:
            dbg.release()

    encoded = _reencode_h264(tmp_out, video_path, output_path) if reencode_h264 else False
    if not encoded:
        shutil.move(tmp_out, output_path)
    shutil.rmtree(tmp_dir, ignore_errors=True)

    covered = sum(1 for o in tr.opacity.values() if o > 0.5)
    metrics = dict(
        anchor_frame=anchor_frame_idx, anchor_method=method, anchor_quad=quad0.round(1).tolist(),
        anchor_warnings=warnings, frames_written=f, frames_with_ad=int(n_drawn), frames_covered=int(covered),
        frames_with_occlusion=int(n_occ), occlusion=anchors[0].note,
        surface_aspect=round(shared_aspect, 3), ad_levels=[lv.bgr.shape[1] for lv in anchors[0].levels][:3],
        shots=[list(r) for r in shot_ranges(cuts, total)] if cuts else [[0, total - 1]],
        anchors=[dict(id=i, frame=a.frame_idx, occlusion=a.note) for i, a in anchors.items()],
        extra_anchor_runs=shot_log, lighting=anchors[0].lighting, tracking=tr.metrics,
        output_video_path=output_path, h264=encoded,
        encoder=("libx264" if encoded else "mpeg4 (no ffmpeg found - install ffmpeg or imageio-ffmpeg)"), total_seconds=round(time.perf_counter() - t_start, 2),
    )
    logger.info("Placement done: %s", metrics)
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description="Place an ad on a tracked planar surface in a video.")
    p.add_argument("--video", required=True)
    p.add_argument("--ad", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--anchor-frame", type=int, required=True)
    p.add_argument("--quad", help='4 corner points "x,y x,y x,y x,y" in the anchor frame (full-resolution pixels)')
    p.add_argument("--surface-type", default="tv")
    p.add_argument("--debug-video")
    p.add_argument("--no-occlusion", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    quad = None
    if args.quad:
        quad = [[float(v) for v in pt.split(",")] for pt in args.quad.split()]
    cfg = AdPlacerConfig(use_occlusion=not args.no_occlusion)
    m = place_ad_in_video(args.video, args.ad, args.out, args.anchor_frame, manual_quad=quad,
                          surface_type=args.surface_type, cfg=cfg, debug_video_path=args.debug_video)
    print(m)


if __name__ == "__main__":
    main()
