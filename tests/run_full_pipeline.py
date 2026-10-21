"""
Full pipeline test:  Agent 1 (detect + track)  ->  Agent 2 (scene context)  ->  Agent 3 (ad placement)

Put this file in  tests/run_full_pipeline.py  and run from the project root:

    python -m tests.run_full_pipeline --video data/input.mp4 --ad data/ad.png

Every stage writes its output so you can inspect it, and a stage can be skipped (re-using its JSON)
with --skip-agent1 / --skip-agent2. Outputs go to data/ (or --out-dir):

    agent1_output.json            Agent 1 result
    agent1_selected_surface.jpg   anchor frame with the polygon / quad Agent 3 will use (CHECK THIS FIRST)
    agent2_output.json            Agent 2 result
    ad_placement_output.jpg       key-frame composite
    ad_placement_output.mp4       final video
    ad_placement_debug.mp4        same video with the tracked quad / state / confidence drawn on it
    contact_sheet.jpg             12 frames from the final video side by side

If an import below does not match your folder/class names, change the 3 lines marked  # ADJUST.
"""

import argparse
import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np

# ADJUST: module paths / class names of your agents
from agents.agent_1_frame_selector.agent import FrameSelectorAgent  # ADJUST
from agents.agent_2_context_analyzer.agent import ContextAnalyzerAgent  # ADJUST
from agents.agent_3_ad_placer.agent import AdPlacerAgent  # ADJUST
from agents.agent_3_ad_placer.schemas import AdPlacerConfig
from agents.agent_3_ad_placer.video_placer import place_ad_in_video  # noqa: F401  (import check)


def dump(obj):
    """pydantic v2 / v1 / dict -> plain dict"""
    for name in ("model_dump", "dict"):
        if hasattr(obj, name):
            return getattr(obj, name)()
    return obj


def banner(text):
    print("\n" + "=" * 62 + f"\n{text}\n" + "=" * 62)


def contact_sheet(video, out_path, n=12, tile_w=240):
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    tiles = []
    for i in np.linspace(0, max(total - 1, 0), n, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if not ok:
            continue
        f = cv2.resize(f, (tile_w, int(f.shape[0] * tile_w / f.shape[1])))
        cv2.putText(f, str(i), (5, 22), 0, 0.7, (0, 255, 255), 2)
        tiles.append(f)
    cap.release()
    if not tiles:
        return
    h = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
    while len(tiles) % 6:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(out_path), np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--ad", required=True)
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--prompt", default=None, help='optional user prompt, e.g. "put the ad on the TV"')
    ap.add_argument("--skip-agent1", action="store_true")
    ap.add_argument("--skip-agent2", action="store_true")
    ap.add_argument("--blend-mode", default="alpha", choices=["alpha", "hybrid", "normal_clone", "mixed_clone"])
    a = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p1, p2 = out / "agent1_output.json", out / "agent2_output.json"
    times = {}

    # ---------------- Agent 1 ---------------- #
    banner("AGENT 1: object detection + tracking")
    t = time.perf_counter()
    if a.skip_agent1 and p1.exists():
        print(f"(skipped, using {p1})")
    else:
        res1 = FrameSelectorAgent().process_video(a.video)
        p1.write_text(json.dumps(dump(res1), indent=2, default=str), encoding="utf-8")
    times["agent1"] = time.perf_counter() - t
    a1 = json.loads(p1.read_text(encoding="utf-8"))
    surfaces = a1.get("detected_surfaces") or []
    print(f"detected {len(surfaces)} object(s):")
    for s in surfaces:
        roi = s.get("roi", {})
        print(f"  id={s.get('object_id')}  {roi.get('surface_type')}  conf={roi.get('confidence', 0):.2f}  "
              f"anchor_frame={s.get('frame_id')}  blur={s.get('blur_score')}")
    if not surfaces:
        raise SystemExit("Agent 1 found nothing: stop here (lower detection_confidence or check the video).")

    # ---------------- Agent 2 ---------------- #
    banner("AGENT 2: scene context")
    t = time.perf_counter()
    if a.skip_agent2 and p2.exists():
        print(f"(skipped, using {p2})")
    else:
        res2 = ContextAnalyzerAgent(num_frames=8).analyze_video(a.video)
        p2.write_text(json.dumps(res2, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    times["agent2"] = time.perf_counter() - t
    sa = json.loads(p2.read_text(encoding="utf-8")).get("scene_analysis", {})
    print("scene type:", sa.get("Scene Type"), "| surfaces:", sa.get("Visible Surfaces"))

    # ---------------- Agent 3 ---------------- #
    banner("AGENT 3: tracking + ad placement")
    t = time.perf_counter()
    cfg = AdPlacerConfig(
        agent1_output_path=p1, agent2_output_path=p2,
        output_image_path=out / "ad_placement_output.jpg",
        output_video_path=out / "ad_placement_output.mp4",
        blend_mode=a.blend_mode,
    )
    result = AdPlacerAgent(cfg).run(
        ad_image_path=a.ad, video_path=a.video, render_video=True, user_prompt=a.prompt,
    )
    times["agent3"] = time.perf_counter() - t
    r = result.model_dump()
    m = r["placement_metrics"]
    print("success :", r["blending_status"]["success"], "|", r["blending_status"]["message"])
    print("surface :", r["selected_surface_id"], "(", m.get("surface_type"), ") anchor frame", r["selected_frame_id"])
    print("quad    :", m.get("target_corners"), "via", m.get("polygon_fit_method"))
    for w in m.get("anchor_warnings", []):
        print("WARNING :", w)
    v = m.get("video", {})
    if v.get("rendered"):
        trk = v.get("tracking", {})
        print(f"video   : {v['frames_written']} frames written, ad drawn on {v.get('frames_with_ad')}, "
              f"occlusion on {v.get('frames_with_occlusion')} ({v.get('occlusion')})")
        print(f"tracking: mean_conf={trk.get('mean_confidence')} segments={trk.get('segments')} "
              f"reacquired={trk.get('reacquire_success', 0)} failed_frames={trk.get('failed_frames', 0)} "
              f"speed={trk.get('processing_fps')} fps")
    else:
        print("VIDEO NOT RENDERED:", v.get("error"))

    # ---------------- Review material ---------------- #
    banner("REVIEW FILES")
    # 1) did Agent 1/3 pick the right surface?  draw the anchor quad on the anchor frame
    cap = cv2.VideoCapture(str(a.video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(r["selected_frame_id"] or 0))
    ok, fr = cap.read()
    cap.release()
    if ok and m.get("target_corners"):
        q = np.array(m["target_corners"], np.int32)
        cv2.polylines(fr, [q], True, (0, 0, 255), 3)
        for i, pt in enumerate(q):
            cv2.circle(fr, tuple(int(x) for x in pt), 7, (0, 255, 0), -1)
            cv2.putText(fr, ["TL", "TR", "BR", "BL"][i], (int(pt[0]) + 8, int(pt[1]) - 8), 0, 0.8, (0, 255, 255), 2)
        cv2.imwrite(str(out / "agent1_selected_surface.jpg"), fr)
        print("check first :", out / "agent1_selected_surface.jpg")
    if r["output_video_path"]:
        contact_sheet(r["output_video_path"], out / "contact_sheet.jpg")
        print("final video :", r["output_video_path"])
        print("contact     :", out / "contact_sheet.jpg")
    print("key frame   :", r["output_image_path"])
    print("\nstage times :", {k: f"{v:.1f}s" for k, v in times.items()})
    (out / "full_pipeline_report.json").write_text(json.dumps(r, indent=2, default=str), encoding="utf-8")
    print("report      :", out / "full_pipeline_report.json")


if __name__ == "__main__":
    main()
