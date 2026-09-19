"""
Tests for Agent 3 (Ad Placement and Seamless Blending).

Runs with pytest:            pytest tests/test_agent_3.py -v
...or standalone:            python tests/test_agent_3.py

Everything is synthetic: a textured "room" with a dark TV drawn at a known perspective quad, a
colourful ad, and hand-written agent1/agent2 JSON files, so no real footage or model is needed.
"""

from __future__ import annotations

import inspect
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest import mock

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.agent_3_ad_placer import (  # noqa: E402
    VALID_BLEND_MODES,
    AdPlacerAgent,
    AdPlacerConfig,
    fit_quad,
    order_corners,
    parse_polygon,
    polygon_is_valid,
)

W, H = 960, 540
TV_QUAD = np.array([[300, 150], [640, 170], [630, 380], [310, 360]], dtype=np.float32)  # TL, TR, BR, BL
DEFAULT_SCENE = {
    "Scene Type": "Living room",
    "Main Objects": ["sofa", "tv", "coffee table"],
    "Persistent Objects": ["tv", "sofa"],
    "Visible Surfaces": ["tv screen", "wall", "floor"],
    "Spatial Relationships": "The tv is mounted on the wall facing the sofa.",
    "Scene Context": "Evening, indoor, lamp on.",
}


# --------------------------------------------------------------------------- #
# Synthetic data helpers
# --------------------------------------------------------------------------- #
def make_scene(wall_gray: int = 150, seed: int = 0) -> np.ndarray:
    """Textured wall (gives optical-flow features) with a dark TV at TV_QUAD."""
    rng = np.random.default_rng(seed)
    wall = (wall_gray, wall_gray, wall_gray)
    img = np.full((H, W, 3), wall, np.uint8)
    for _ in range(300):
        x, y = int(rng.integers(0, W)), int(rng.integers(0, H))
        sw, sh = int(rng.integers(4, 22)), int(rng.integers(4, 22))
        colour = np.clip(np.array(wall) + rng.integers(-45, 45, 3), 0, 255).tolist()
        cv2.rectangle(img, (x, y), (x + sw, y + sh), colour, -1)
    quad_i = TV_QUAD.astype(np.int32)
    cv2.fillConvexPoly(img, quad_i, (28, 26, 24))
    cv2.polylines(img, [quad_i], True, (8, 8, 8), 6)
    return img


def make_ad(w: int = 600, h: int = 300, corner_marks: bool = False) -> np.ndarray:
    xs = np.linspace(0, 255, w, dtype=np.uint8)
    ad = np.zeros((h, w, 3), np.uint8)
    ad[..., 0], ad[..., 1], ad[..., 2] = xs[None, :], 160, 255 - xs[None, :]
    cv2.putText(ad, "AD", (w // 3, h // 2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255, 255, 255), 8)
    if corner_marks:  # TL red, TR green, BR blue, BL yellow (BGR)
        s = min(w, h) // 3
        ad[:s, :s] = (0, 0, 255)
        ad[:s, -s:] = (0, 255, 0)
        ad[-s:, -s:] = (255, 0, 0)
        ad[-s:, :s] = (0, 255, 255)
    return ad


def tv_surface(object_id: Any = 7, frame_id: int = 0, polygon: Optional[List[List[float]]] = None, **roi_kw: Any) -> Dict[str, Any]:
    roi = {"surface_type": "tv", "confidence": 0.9, "polygon_points": (TV_QUAD if polygon is None else polygon)}
    roi.update(roi_kw)
    roi["polygon_points"] = np.asarray(roi["polygon_points"], dtype=float).tolist() if roi["polygon_points"] is not None else None
    return {
        "object_id": object_id,
        "frame_id": frame_id,
        "first_seen_frame": 0,
        "timestamp_sec": 0.0,
        "blur_score": 250.0,
        "roi": roi,
    }


def setup_case(
    root: Path,
    surfaces: List[Dict[str, Any]],
    background: Optional[np.ndarray] = None,
    ad: Optional[np.ndarray] = None,
    **config_overrides: Any,
):
    """Write agent1/agent2 JSON + images into ``root`` and return (config, ad_path, background_path)."""
    root.mkdir(parents=True, exist_ok=True)
    data = root / "data"
    data.mkdir(exist_ok=True)
    (data / "agent1_output.json").write_text(json.dumps({"detected_surfaces": surfaces}))
    (data / "agent2_output.json").write_text(json.dumps({"scene_analysis": DEFAULT_SCENE}))
    bg_path, ad_path = root / "background.png", root / "ad.png"
    cv2.imwrite(str(bg_path), make_scene() if background is None else background)
    cv2.imwrite(str(ad_path), make_ad() if ad is None else ad)
    config = AdPlacerConfig(
        agent1_output_path=data / "agent1_output.json",
        agent2_output_path=data / "agent2_output.json",
        output_image_path=data / "ad_placement_output.jpg",
        output_video_path=data / "ad_placement_output.avi",
        **config_overrides,
    )
    return config, ad_path, bg_path


def quad_mask(quad: np.ndarray, scale: float = 1.0, dilate_px: int = 0) -> np.ndarray:
    centre = quad.mean(axis=0)
    q = (centre + (quad - centre) * scale).astype(np.int32)
    mask = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(mask, q, 255)
    if dilate_px:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        mask = cv2.dilate(mask, k)
    return mask > 0


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def test_order_corners_is_permutation_invariant_and_clockwise():
    angle = np.deg2rad(20)
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rect = (np.array([[0, 0], [200, 0], [200, 100], [0, 100]]) @ rot.T + [300, 200]).astype(np.float32)
    expected = order_corners(rect)
    rng = np.random.default_rng(1)
    for _ in range(10):
        shuffled = rect[rng.permutation(4)]
        assert np.allclose(order_corners(shuffled), expected)
    # clockwise on screen (y down) -> positive shoelace sum
    x, y = expected[:, 0], expected[:, 1]
    assert np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) > 0
    assert expected[0].sum() == expected.sum(axis=1).min()  # first point is top-left


def test_polygon_validation_and_normalised_coordinates():
    assert polygon_is_valid(TV_QUAD)
    assert not polygon_is_valid(np.array([[0, 0], [10, 10], [20, 20], [30, 30]], np.float32))  # collinear
    assert not polygon_is_valid(np.array([[0, 0], [10, 0], [10, 10]], np.float32))  # < 4 points
    assert not polygon_is_valid(None)
    assert parse_polygon("garbage", 100, 100) is None
    assert parse_polygon([[1, 2, 3]], 100, 100) is None
    pts = parse_polygon([[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]], 200, 100)
    assert np.allclose(pts.max(axis=0), [180, 90])


def test_fit_quad_handles_non_quadrilaterals():
    quad, method = fit_quad(TV_QUAD)
    assert method == "polygon_4_points" and quad.shape == (4, 2)

    # TV quad with jittered midpoints -> 8-point polygon
    mids = (TV_QUAD + np.roll(TV_QUAD, -1, axis=0)) / 2 + np.array([1.5, -1.5])
    octagon = np.vstack([TV_QUAD, mids]).astype(np.float32)
    quad, method = fit_quad(octagon)
    assert quad.shape == (4, 2) and method in ("approx_poly_dp", "min_area_rect")
    assert np.abs(quad - order_corners(TV_QUAD)).max() < 6

    # a circle-ish blob has no sensible quad -> minimum-area rectangle fallback
    t = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    blob = np.stack([400 + 100 * np.cos(t), 300 + 100 * np.sin(t)], axis=1).astype(np.float32)
    quad, method = fit_quad(blob)
    assert quad.shape == (4, 2) and method == "min_area_rect"


# --------------------------------------------------------------------------- #
# Surface selection
# --------------------------------------------------------------------------- #
def test_selection_prefers_valid_high_confidence_tv(tmp_path: Path):
    wall_poly = [[100, 50], [860, 60], [860, 480], [100, 470]]
    surfaces = [
        tv_surface(1, polygon=[[0, 0], [10, 10], [20, 20], [30, 30]]),  # degenerate polygon
        tv_surface(2),  # the good one
        tv_surface(3, polygon=wall_poly, surface_type="wall", confidence=0.9),
        tv_surface(4, confidence=0.1),  # below min confidence
        {"object_id": 5, "frame_id": 0, "roi": {"surface_type": "tv", "confidence": 0.99}},  # no polygon
    ]
    config, _, _ = setup_case(tmp_path, surfaces)
    agent = AdPlacerAgent(config)
    sections = agent._scene_sections({"scene_analysis": DEFAULT_SCENE})
    ranked = agent.rank_surfaces({"detected_surfaces": surfaces}, sections, W, H)
    ids = [c.object_id for c in ranked]
    assert ids[0] == 2
    assert 3 in ids and not ({1, 4, 5} & set(ids))


# --------------------------------------------------------------------------- #
# End-to-end key frame placement
# --------------------------------------------------------------------------- #
def test_end_to_end_keyframe(tmp_path: Path):
    background = make_scene()
    config, ad_path, bg_path = setup_case(tmp_path, [tv_surface(7)], background=background)
    result = AdPlacerAgent(config).run(ad_path, background_image_path=bg_path)

    assert result.blending_status.success, result.blending_status.message
    assert result.selected_surface_id == 7 and result.selected_frame_id == 0
    out = cv2.imread(result.output_image_path)
    assert out is not None and out.shape == background.shape

    metrics = result.placement_metrics
    assert np.array(metrics["transformation_matrix"]).shape == (3, 3)
    for key in ("brightness_shift", "contrast_gain", "a_shift", "b_shift", "strength"):
        assert key in metrics["lighting_adjustment"]
    assert metrics["lighting_adjustment"]["reference"] == "surrounding_ring"  # TV is emissive

    # the screen interior changed a lot ...
    inner = quad_mask(TV_QUAD, scale=0.6)
    assert np.abs(out.astype(int) - background.astype(int))[inner].mean() > 20
    # ... while everything >30 px away from the TV is untouched (compare against a JPEG round-trip of the input)
    ref = cv2.imdecode(cv2.imencode(".jpg", background, [cv2.IMWRITE_JPEG_QUALITY, 95])[1], cv2.IMREAD_COLOR)
    far = ~quad_mask(TV_QUAD, dilate_px=30)
    assert np.abs(out.astype(int) - ref.astype(int))[far].max() <= 1


def test_ad_orientation_matches_surface_corners(tmp_path: Path):
    config, ad_path, bg_path = setup_case(
        tmp_path, [tv_surface()], ad=make_ad(corner_marks=True), fit_mode="stretch", blend_mode="alpha"
    )
    result = AdPlacerAgent(config).run(ad_path, background_image_path=bg_path)
    assert result.blending_status.success
    out = cv2.imread(result.output_image_path).astype(int)
    centre = TV_QUAD.mean(axis=0)

    def sample(corner: np.ndarray) -> np.ndarray:
        x, y = (centre + 0.8 * (corner - centre)).astype(int)
        return out[y - 3 : y + 4, x - 3 : x + 4].reshape(-1, 3).mean(axis=0)  # B, G, R

    tl, tr, br, bl = (sample(c) for c in TV_QUAD)
    assert tl[2] > tl[1] + 30 and tl[2] > tl[0] + 30  # red
    assert tr[1] > tr[2] + 30 and tr[1] > tr[0] + 30  # green
    assert br[0] > br[1] + 30 and br[0] > br[2] + 30  # blue
    assert bl[2] > bl[0] + 30 and bl[1] > bl[0] + 30  # yellow


def test_lighting_follows_scene_brightness(tmp_path: Path):
    means = {}
    for name, wall in (("dim", 55), ("bright", 205)):
        config, ad_path, bg_path = setup_case(
            tmp_path / name, [tv_surface()], background=make_scene(wall_gray=wall), blend_mode="alpha"
        )
        result = AdPlacerAgent(config).run(ad_path, background_image_path=bg_path)
        assert result.blending_status.success
        gray = cv2.cvtColor(cv2.imread(result.output_image_path), cv2.COLOR_BGR2GRAY)
        means[name] = float(gray[quad_mask(TV_QUAD, scale=0.6)].mean())
    assert means["bright"] > means["dim"] + 3, means


def test_non_quad_polygon_uses_fallback(tmp_path: Path):
    mids = (TV_QUAD + np.roll(TV_QUAD, -1, axis=0)) / 2 + np.array([1.5, -1.5])
    polygon = np.vstack([TV_QUAD, mids]).tolist()
    config, ad_path, bg_path = setup_case(tmp_path, [tv_surface(polygon=polygon)])
    result = AdPlacerAgent(config).run(ad_path, background_image_path=bg_path)
    assert result.blending_status.success
    assert result.placement_metrics["polygon_fit_method"] in ("approx_poly_dp", "min_area_rect")


def test_all_blend_modes_run(tmp_path: Path):
    for mode in VALID_BLEND_MODES:
        config, ad_path, bg_path = setup_case(tmp_path / mode, [tv_surface()], blend_mode=mode)
        result = AdPlacerAgent(config).run(ad_path, background_image_path=bg_path)
        assert result.blending_status.success, (mode, result.blending_status.message)
        assert Path(result.output_image_path).exists()
        used = result.placement_metrics["blending"]["mode_used"]
        assert used.startswith("hybrid") if mode == "hybrid" else used == mode, (mode, used)  # no silent fallback


def test_hybrid_keeps_ad_interior_but_changes_seam(tmp_path: Path):
    outs = {}
    for mode in ("alpha", "hybrid"):
        config, ad_path, bg_path = setup_case(tmp_path / mode, [tv_surface()], blend_mode=mode)
        result = AdPlacerAgent(config).run(ad_path, background_image_path=bg_path)
        assert result.blending_status.success
        outs[mode] = cv2.imread(result.output_image_path).astype(int)
    diff = np.abs(outs["hybrid"] - outs["alpha"])
    assert diff[quad_mask(TV_QUAD, scale=0.7)].mean() < 3.0  # interior identical up to JPEG noise
    seam = quad_mask(TV_QUAD, dilate_px=4) & ~quad_mask(TV_QUAD, scale=0.9)
    assert diff[seam].mean() > 3.0  # ...but the seam band is actually Poisson-adjusted


def test_poisson_failure_falls_back_to_alpha(tmp_path: Path):
    config, ad_path, bg_path = setup_case(tmp_path, [tv_surface()], blend_mode="mixed_clone")
    with mock.patch.object(cv2, "seamlessClone", side_effect=cv2.error("boom")):
        result = AdPlacerAgent(config).run(ad_path, background_image_path=bg_path)
    assert result.blending_status.success
    assert result.placement_metrics["blending"]["mode_used"] == "alpha"
    assert "alpha" in result.blending_status.message.lower()


def test_transparent_png_ad_is_supported(tmp_path: Path):
    ad = cv2.cvtColor(make_ad(), cv2.COLOR_BGR2BGRA)
    ad[:, : ad.shape[1] // 2, 3] = 0  # left half transparent
    config, ad_path, bg_path = setup_case(tmp_path, [tv_surface()], ad=ad, blend_mode="alpha", fit_mode="stretch")
    result = AdPlacerAgent(config).run(ad_path, background_image_path=bg_path)
    assert result.blending_status.success
    out = cv2.imread(result.output_image_path)
    bg = cv2.imread(str(bg_path))
    left = quad_mask(TV_QUAD, scale=0.25)
    left[:, int(TV_QUAD.mean(axis=0)[0]) :] = False  # left of the screen centre
    assert np.abs(out.astype(int) - bg.astype(int))[left].mean() < 12  # background shows through


# --------------------------------------------------------------------------- #
# Failure handling: never raise, always report
# --------------------------------------------------------------------------- #
def test_missing_polygon_reports_failure(tmp_path: Path):
    surface = tv_surface()
    surface["roi"].pop("polygon_points")
    config, ad_path, bg_path = setup_case(tmp_path, [surface])
    result = AdPlacerAgent(config).run(ad_path, background_image_path=bg_path)
    assert not result.blending_status.success
    assert "No valid target surface" in result.blending_status.message
    assert result.output_image_path is None and not config.output_image_path.exists()


def test_missing_inputs_report_failure(tmp_path: Path):
    config, ad_path, bg_path = setup_case(tmp_path, [tv_surface()])
    agent = AdPlacerAgent(config)
    assert not agent.run(tmp_path / "nope.png", background_image_path=bg_path).blending_status.success
    assert not agent.run(ad_path, background_image_path=tmp_path / "nope.png").blending_status.success
    assert not agent.run(ad_path).blending_status.success  # no frame source at all
    config.agent1_output_path.unlink()
    assert not agent.run(ad_path, background_image_path=bg_path).blending_status.success


def test_missing_agent2_is_tolerated(tmp_path: Path):
    config, ad_path, bg_path = setup_case(tmp_path, [tv_surface()])
    config.agent2_output_path.unlink()
    result = AdPlacerAgent(config).run(ad_path, background_image_path=bg_path)
    assert result.blending_status.success


# --------------------------------------------------------------------------- #
# Video: tracked overlay
# --------------------------------------------------------------------------- #
def test_video_rendering_tracks_surface(tmp_path: Path):
    n_frames, key = 20, 5
    base = make_scene()
    video_path = tmp_path / "input.avi"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 25.0, (W, H))
    if not writer.isOpened():
        print("  (skipped: MJPG VideoWriter unavailable)")
        return
    shifts = [(2.0 * i, 0.5 * i) for i in range(n_frames)]
    for dx, dy in shifts:
        m = np.float32([[1, 0, dx], [0, 1, dy]])
        writer.write(cv2.warpAffine(base, m, (W, H), borderMode=cv2.BORDER_REPLICATE))
    writer.release()

    key_quad = TV_QUAD + np.array(shifts[key], dtype=np.float32)
    surface = tv_surface(polygon=key_quad.tolist(), frame_id=key)
    surface["first_seen_frame"] = 0
    config, ad_path, _ = setup_case(tmp_path, [surface], video_fourcc="MJPG")
    result = AdPlacerAgent(config).run(ad_path, video_path=video_path, render_video=True)

    assert result.blending_status.success, result.blending_status.message
    assert result.selected_frame_id == key
    video = result.placement_metrics["video"]
    assert video["rendered"], video
    assert video["first_frame"] == 0 and video["last_frame"] == n_frames - 1
    assert video["frames_written"] == n_frames
    expected_last = TV_QUAD + np.array(shifts[-1], dtype=np.float32)
    assert np.abs(np.array(video["last_frame_corners"]) - order_corners(expected_last)).max() < 4.0

    cap = cv2.VideoCapture(result.output_video_path)
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == n_frames
    cap.release()


# --------------------------------------------------------------------------- #
# Standalone runner
# --------------------------------------------------------------------------- #
def _run_all() -> int:
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    failures = 0
    for name, fn in tests:
        kwargs = {}
        if "tmp_path" in inspect.signature(fn).parameters:
            kwargs["tmp_path"] = Path(tempfile.mkdtemp(prefix=f"agent3_{name}_"))
        try:
            fn(**kwargs)
            where = f"  [{kwargs['tmp_path']}]" if kwargs else ""
            print(f"PASS  {name}{where}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())