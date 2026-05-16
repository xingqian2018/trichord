# Scene Text English V1 — Structured Caption Analysis

**Data path:** `gcs:nv-00-10206-vfm/debug/xingqianx/synthetic_data/synthetic_scene_text_v1/prompt/part000000/`
**Date:** 2026-05-12
**Samples inspected:** 4 (keys `000000000000`, `000000000234`, `000000000413`, `000000000654`)
**Generation model:** `qwen3-235b-a22b-instruct`

---

## Schema Overview

Each sample contains a `promptgen_structured_caption` field with:
- `entities` — list of scene objects, each with: `identifier`, `description`, `location`, `relation`, `materials`, `text_and_signage` (optional), `grounding_dense`, `grounding_dense_downsampled`
- `image_style`
- `camera_details`

---

## Finding A — Spatial Depth Inconsistency

### Description
Each entity is assigned a `location` (depth zone: foreground / midground / background) independently. This assignment does not propagate through the `relation` field, causing physically adjacent or stacked objects to land in different depth zones.

### Observed Cases

**Sample `000000000000` (cabin workbench):**
- `wooden ruler` → `center middle foreground`, but rests *on top of* `green paint can` → `center middle midground`
- `handwritten notepad` → `left middle foreground`, but sits *on* `workbench` → `left middle midground`

**Sample `000000000654` (bar entrance):**
- `beer tap handle label` → `lower left foreground`, but described as inside the bar, *behind* the `glass entrance door` → `center midground` and `interior bar counter` → `center background`
- `metal trash bin` → `lower left foreground`, but described as *next to the streetlight pole* which is 3 meters from the entrance (midground distance)

**Sample `000000000234` (boutique interior):**
- `floor lamp` → `lower left foreground`, but relation says *standing behind the sofa* → `lower left midground` (depth reversal: behind = closer to camera?)

### Additional: Left/Right Mismatch (Sample `000000000234`)
- `floor lamp` entity relation: "to the *left* of the seating area"
- `camera_details`: "warm-toned fill from the floor lamp on the *right* rear"
- Direct contradiction between entity field and camera_details.

### Root Cause
Depth zone is assigned per-entity in isolation. The `relation` field (which encodes spatial dependency) is not consulted during location assignment, leading to violations of the basic constraint: *if object A rests on / is behind / is part of object B, A and B must share compatible depth zones.*

---

## Finding B — OCR Text Coverage Across Fields

### Design Intent
`text_and_signage` is intended to hold only the exact visible characters (pure OCR), analogous to what a text detector would return.

### Observed Inconsistency in `text_and_signage`

In sample `000000000000`, all 5 text-bearing entities have font/style descriptions appended to the OCR text:

| Identifier | text_and_signage |
|---|---|
| wooden ruler on workbench | `"12 INCHES" in small engraved serif font along the edge` |
| handwritten notepad | `"Cut pine boards: 8" x 10" x 2" — 6 pcs" in smudged graphite handwriting` |
| wall calendar | `"CABIN BUILDING SCHEDULE ..." in bold black print` |
| tape measure on floor | `"MAX 25 FT" in red block numerals near the retraction housing` |
| window decal | `"INCH BY INCH, WE BUILD" in white sans-serif letters` |

Samples `000000000654`, `000000000413`, `000000000234` are clean — `text_and_signage` contains only the quoted characters.

This is a model instruction-following inconsistency, not a schema issue.

### OCR Text Coverage Across Entity Fields

For all 5 text-bearing entities in sample `000000000000`:

| Field | Contains OCR text? |
|---|---|
| `identifier` | ❌ Never |
| `description` | ❌ Never |
| `grounding_dense` | ✅ Always |
| `grounding_dense_downsampled` | ✅ Always |

The `identifier` describes what the object *is* (e.g. "tape measure on floor"), never what it *reads*. The `description` likewise omits the text content. The OCR characters appear exclusively in `text_and_signage` as a dedicated field, and are redundantly embedded inline within both grounding prose fields.

### Implication
- If `text_and_signage` is malformed (as in sample `000000000000`), the actual displayed characters are still recoverable from `grounding_dense` / `grounding_dense_downsampled` via text extraction.
- Downstream tasks relying on exact-match OCR from `text_and_signage` would fail on the malformed entries; falling back to the grounding fields requires parsing prose.

---

## Summary

| Finding | Severity | Affected Samples | Recoverable? |
|---|---|---|---|
| Depth zone ignores `relation` constraints | Medium | All 4 samples | N/A (generation issue) |
| Left/right mismatch between entity and camera_details | Low | `000000000234` | N/A |
| `text_and_signage` includes font descriptions instead of pure OCR | Medium | `000000000000` (1/4) | ✅ via grounding fields |
| OCR text absent from `identifier` and `description` | By design | All samples | ✅ via grounding fields |
