# VPP AI API contract

Base URL during local development: `http://127.0.0.1:8000`

The first request uploads the video and advertisement and runs Agent 1 and Agent 2. The same request supports both user flows: direct prompt placement, or recommendations. In direct mode, Agent 2 interprets the prompt against its scene analysis and Agent 1's eligible detected surfaces. A clear match is rendered immediately in this request. An unclear prompt returns recommendations so the user can choose one. There is no separate prompt interpretation request.

## 1. Analyze and either place directly or return recommendations

`POST /jobs/analyze` (`multipart/form-data`)

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `video` | video file | yes | Input video |
| `advertisement` | image file | yes | Advertisement artwork |
| `mode` | `direct` or `recommendation` | no | Defaults to `recommendation` |
| `prompt` | string | no | In `direct` mode, describes the intended surface; in recommendation mode, optional context |

### Direct prompt flow

Submit the video, ad, `mode=direct`, and the prompt together in this first request. For example, in Swagger `/docs`, select `direct` in the `mode` field and enter a prompt such as `Place the ad on the TV screen`.

If the prompt clearly matches an eligible detected surface, this same request renders the placement and returns `status: "completed"` with `output_video_url` and `output_image_url`. If the prompt is vague, unsupported by the scene, or cannot be matched confidently, the response is `status: "needs_recommendation"` with a clarification, previews, scores, and risk notes. The user can then choose a candidate by calling `/jobs/{job_id}/place` in recommendation mode.

Example completed response:

```json
{
  "status": "completed",
  "job_id": "a1b2c3d4",
  "mode": "direct",
  "prompt": "Place the ad on the TV screen",
  "selected_surface_id": 1,
  "output_image_url": "/jobs/a1b2c3d4/files/result.jpg",
  "output_video_url": "/jobs/a1b2c3d4/files/result.mp4",
  "prompt_interpretation": {
    "intent_clear": true,
    "matched_candidate_ids": ["1"],
    "target_description": "TV screen",
    "reason": "The detected TV screen matches the prompt."
  }
}
```

Example unclear response:

```json
{
  "job_id": "a1b2c3d4",
  "status": "needs_recommendation",
  "mode": "direct",
  "requires_user_choice": true,
  "prompt_interpretation": {
    "intent_clear": false,
    "clarification_question": "Which visible surface should receive the ad?"
  },
  "recommendations": []
}
```

The actual `recommendations` array contains the available candidates, each with its `candidate_id`, surface type, suitability score (0-100; higher is better), score breakdown, preview URL, and any `risk_notes`.

### Recommendation flow

Submit `mode=recommendation` (or omit `mode`) with the same video and ad. This request analyzes the scene and returns candidate previews and scores; it does not render the final video yet. The client should display each preview, score, and risk notes, then submit the chosen `candidate_id` as `selected_surface_id` to `/jobs/{job_id}/place`.

Recommendations filter Agent 1 detections to supported planar placement types (screens, laptops, billboards/posters, walls, pictures/frames, books, doors, and windows). People, plants, bowls, couches, and other non-target objects are not offered. Scores use Agent 3's weighted ranking: confidence, scene compatibility, area, and sharpness. `preview_url` is relative to the API base URL. These are single anchor-frame composites for comparing surfaces, not the final tracked video; after a choice, `/place` runs the full Agent 3 video algorithm.

If Agent 2 describes a display but Agent 1 finds no eligible surface, `/jobs/analyze` makes a targeted TV detection retry. For green-screen footage it can locate the large chroma rectangle geometrically; otherwise it retries YOLO at a higher input resolution. This conditional fallback can add processing time, and its use or failure is reported in `warnings`.

## 2. Place a selected recommendation (or continue after an unclear prompt)

`POST /jobs/{job_id}/place` (`application/json`)

This endpoint is used after recommendation mode, or after direct mode returned `needs_recommendation`. It reuses the saved Agent 1/Agent 2 analysis and does not rerun video analysis.

```json
{
  "mode": "recommendation",
  "selected_surface_id": "1",
  "prompt": "Keep the logo centered and fully visible"
}
```

The AI locks Agent 3 to the chosen candidate and runs the same full `AdPlacerAgent.run(render_video=True)` pipeline as direct placement: surface initialization, planar tracking, temporal smoothing, edge refinement, per-frame compositing, and video encoding. The optional prompt is retained for traceability; it does not currently change the geometry or blending behavior after a candidate is locked.

Example response:

```json
{
  "status": "completed",
  "job_id": "a1b2c3d4",
  "mode": "recommendation",
  "selected_surface_id": 1,
  "selected_frame_id": 0,
  "placement_score": 0.842,
  "output_image_url": "/jobs/a1b2c3d4/files/result.jpg",
  "output_video_url": "/jobs/a1b2c3d4/files/result.mp4",
  "placement_warnings": []
}
```

## Job files and errors

- `GET /jobs/{job_id}/files/{filename}` returns recommendation previews and completed outputs.
- A missing or expired `job_id` returns `404`.
- Missing `selected_surface_id` in recommendation mode returns `422`.
- If no usable candidate or placement fails, placement returns `422` with an error detail.
- If Agent 2 scene analysis runs out of resources, the API can still return Agent 1 based recommendations and includes the issue in `warnings`.

Interactive OpenAPI docs: `/docs`.
