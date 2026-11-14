# VLM artifact-check criteria v1

Derived bottom-up from 487 artifact observations on 224 images (Open T2I Pilot:
krea2 / ideogram4 / flux3 / qwen21 × 56 prompts). Layer A needs only the image and is
the part used on unknown inputs. Layer B is used only when the generation prompt is
available. Layer C is a policy flag.

Prevalence = images (of 224) with the defect at any severity / at moderate-or-worse,
from the silver labels in image_labels_v1.json.

## How the VLM applies a criterion

For every criterion the VLM must:
1. Evaluate the **gate**. If the gate is false, output `n/a` (not `absent`).
2. Inspect the named regions and answer the **check**.
3. If present, give **severity** (minor / moderate / severe, anchors below) and
   **evidence**: location + what is literally seen (quote rendered text verbatim).
4. Never infer a defect from what the image "should" contain in Layer A.

Severity anchors (global):
- severe: the first thing a viewer notices, or it breaks the image's purpose.
- moderate: noticeable at normal viewing size.
- minor: only on close inspection / zoom.

## Layer A — image-only artifacts

### A01 render_failure (2 / 2)
- Gate: always.
- Check: Is the image a placeholder, blank, single colour, corrupt, heavy noise, or a
  system message (e.g. "Image blocked by safety filter") instead of real content?
- Severity: always severe. If present, skip all other criteria.
- Pilot: ideogram4 22bcc159, be6b1589.

### A02 primary_text_error (27 / 17)
- Gate: image contains intended text — headline, title, sign, label, button, menu, UI
  text, or any text that is the subject of the image.
- Check: Transcribe every such string letter by letter. Any misspelling, missing /
  extra / swapped letter, broken glyph, duplicated word inside a string, truncation
  of a word, or malformed digits? For non-Latin scripts: do the glyphs form real words
  in one consistent script?
- Severity: severe if the main headline/title is wrong; moderate for other intended
  text; minor for a single subtle slip in secondary intended text.
- Pilot: "HOLD YOUR GROTIIND" (krea2 9020fa4d), "HAPPY TALS" (qwen21 c055516a),
  "HAPPY 31ST BIRTHDA" (ideogram4 6242cfd5), "약과 · 중정糕" (qwen21 99b8498e).
- Boundary: incidental text not part of the design → A03.

### A03 background_pseudo_text (38 / 7)
- Gate: image contains incidental text (signage, billboards, body copy, logos, plates,
  jersey marks, UI micro-labels).
- Check: Is any of it illegible pseudo-letters or nonsense words?
- Severity: minor for small far text; moderate if prominent (large billboard, logo
  near the subject).
- Pilot: "VOTAKE", "TIMEDOGTI", newspaper body copy, sponsor logos.

### A04 hand_anatomy (16 / 2)
- Gate: a human (or humanoid) hand or foot is visible.
- Check: Count digits on each visible hand. Are any fused, missing, extra, mitten-like,
  wrong length, missing thumb? Is the grip physically possible (palm and fingertips on
  consistent sides of the held object)? Toes included.
- Pilot: palm facing camera while fingertips curl over phone front (ideogram4 538c255f);
  mitten hands holding glasses (krea2 6242cfd5).
- Boundary: whose body a hand belongs to → A05.

### A05 body_anatomy (21 / 8)
- Gate: a person or animal body (beyond face) is visible.
- Check: Limb count, length, proportion and joint direction plausible? Torso, neck,
  legs continuous? When bodies overlap, can every limb be assigned to exactly one body?
- Pilot: kabaddi defender holding his own leg (ideogram4 9020fa4d); two knees, one
  foot (krea2 be6b1589); torso shrinking to a stalk (qwen21 edd35a41).
- Boundary: outlines physically blend into each other → A08; clean outlines but
  unclear ownership → A05.

### A06 face_anatomy (5 / 0)
- Gate: a face (human/animal/character) is visible, including background figures.
- Check: Eyes, teeth, features coherent and symmetric as appropriate? Background
  faces smeared/featureless? Several faces near-identical clones?
- Pilot: smeared supporters (flux3 1c557ff8), identical mascot faces (qwen21 db9a4cce).

### A07 phantom_duplicate (12 / 7)
- Gate: always.
- Check: Is there an extra copy or fragment of something that should exist once —
  extra head, body-less head, second person in an impossible place, doubled pole,
  extra wheel, duplicated UI card?
- Pilot: second beagle head on a stretched neck (ideogram4 c055516a); second officer
  inside the car (krea2 750da150); disembodied bearded head (flux3 cde42482).
- Boundary: a plausible, complete extra instance (7 students instead of 6) is B01.

### A08 object_fusion (22 / 3)
- Gate: always.
- Check: At the boundary between two separate things, do outlines blend so they
  cannot be separated (objects, animals, hands, props)?
- Pilot: llama bodies merging, one with no head (krea2 3ea00508); arm blending into a
  hand holding a glass (qwen21 6242cfd5).

### A09 floating_unsupported (24 / 6)
- Gate: always.
- Check: Does any object hover with no support, hang from nothing, or a part sit
  disconnected from its structure?
- Pilot: pizza slice hovering with no hand (ideogram4 6dbcce59); turbine blade with
  no tower (flux3 3615dbbf).
- Boundary: support exists but forces/fluids/light act wrongly → A12.

### A10 impossible_structure (26 / 8)
- Gate: always.
- Check: Could every manufactured object / building / device physically exist and
  function as drawn (hinges, frames, screens, tools, circuits, joints)?
- Pilot: laptop keyboard on the camera side while lid shows its back (ideogram4
  fe071b7f); UI card larger than the phone (ideogram4 538c255f); bike-frame truss
  (flux3 80b1d2ec).
- Boundary: coherent object that is just not the requested one → B02.

### A11 perspective_depth (15 / 5)
- Gate: scene has depth (not a flat graphic).
- Check: Do parallel lines converge consistently? Is front/behind ordering consistent
  for every overlap? Is scale consistent with distance?
- Pilot: floorboards fanning outward against room perspective (ideogram4 18cf827d).

### A12 physics_lighting (26 / 8)
- Gate: always.
- Check: Are shadows, reflections, light sources, heat, fluids, smoke/steam and
  gravity consistent within the scene?
- Pilot: mirror reflects nothing (ideogram4 cbe42980); bare hand on glowing crucible
  (flux3 745c1740); water spraying upward from above the spout (flux3 be3a0578).
- Boundary: one element lit / styled differently from the rest → A15.

### A13 repetition_stamping (19 / 4)
- Gate: scene contains multiple similar items or a pattern.
- Check: Are instances suspiciously identical (cloned), or is a repeated pattern
  broken (some copies differ in an inconsistent way)?
- Pilot: stamped snowflakes (ideogram4 a7919ff9); clip border with tabs on only some
  clips (qwen21 22bcc159).

### A14 texture_anomaly (15 / 2)
- Gate: always.
- Check: Melted, waxy/plastic, glassy, crumpled or noisy surfaces; meaningless
  smudges, blobs or scribbles that are not objects?
- Pilot: anvil dripping like liquid (krea2 2dd7264f); crumpled-paper gown (qwen21
  cde42482); chalk-dust blotch (ideogram4 db9a4cce).

### A15 compositing_seam (8 / 5)
- Gate: always.
- Check: Does any element look pasted in — hard cut-out edges, no contact shadow,
  lighting or rendering style different from the rest, collage seam, decal/print that
  ignores the curvature of its surface?
- Pilot: photo panel joined to illustration panels (ideogram4 13d33ca6); colour tiger
  in a charcoal room (ideogram4 18cf827d); flat text on a curved mug (ideogram4 fe071b7f).

### A16 non_content_overlay (6 / 5)
- Gate: always.
- Check: Watermark, signature, seal, fake transparency checkerboard, subtitle bar, or
  large padding canvas around the real content?
- Pilot: artist signature (ideogram4 edd35a41); red seal (ideogram4 745c1740);
  checkerboard baked into pixels (ideogram4 2dd7264f); cover in 40% of a grey canvas
  (ideogram4 5a7ef18d).

### A17 framing_crop (9 / 2)
- Gate: always.
- Check: Is a key subject or text cut awkwardly by the frame edge (head, headline,
  sign word)?
- Pilot: smith's head cut off (krea2 2dd7264f); "LUCKY COME" (krea2 0514a3c0).

### A18 layout_alignment (7 / 1)
- Gate: image is or contains a designed graphic / UI / table / menu.
- Check: Are grids, rows and columns aligned; do labels sit with their values; do
  elements collide with each other or with the subject?
- Pilot: names and times in separate day columns (ideogram4 538c255f).

### A19 context_incongruity (8 / 0)
- Gate: always.
- Check: Is a well-formed object present where it makes no sense (fire on a gym floor,
  painted lane lines on a dirt road, dangling cable next to an aquarium)?
- Pilot: open fires on gym floor (qwen21 4999c6f4).

## Layer B — prompt-conditioned (gate: prompt available)

| Code | Check | Prevalence |
|---|---|---|
| B01 count_mismatch | Count each requested quantity; does it match? | 12 / 8 |
| B02 missing_or_wrong_object | Every requested object present and the right kind (incl. species, period)? | 35 / 11 |
| B03 action_pose_interaction | Requested action, pose, gesture, expression, contact (who holds what) correct? | 18 / 7 |
| B04 text_content_mismatch | Legible rendered text equals the quoted prompt text; no unrequested text? | 11 / 0 |
| B05 attribute_style_view | Style/medium, camera angle, focus, light/time, colour, material, age as requested? | 29 / 6 |
| B06 spatial_layout | Relative placement / arrangement as requested? | 11 / 5 |
| B07 forbidden_or_unrequested | Anything the prompt excluded ("no lens flare", "invisible supports") present? | 9 / 0 |

## Layer C — risk flag

| Code | Check |
|---|---|
| C01 real_brand_person_landmark | Real trademark, logo, club crest, identifiable real person or landmark reproduced? |

## Boundary rules (summary)

| Confusable pair | Rule |
|---|---|
| A02 vs A03 | Text that is the subject or part of the design → A02, regardless of size. Incidental → A03. |
| A02 vs A07 vs B04 | Duplication inside one string → A02; duplicated UI/data item → A07; whole legible block repeated elsewhere → B04. |
| A04 vs A05 | Shape/digits/grip of a hand → A04; which body it belongs to → A05. |
| A05 vs A08 | Outlines blend → A08; outlines clean but ownership unclear → A05. |
| A07 vs B01 | Visibly anomalous copy (fragment, no body, impossible place) → A07; plausible complete extra → B01. |
| A09 vs A12 | No support at all → A09; support exists but physics wrong (incl. floating on water) → A12. |
| A12 vs A15 | Wrong light inside a coherent scene → A12; one element lit/styled unlike the rest → A15. |
| A10 vs B02 | Object could not exist/function → A10; coherent but not the requested object → B02. |
| B02 vs B05 | Different noun/category → B02; same object, wrong attribute → B05. |

## VLM output schema (per image)

```json
{
  "A01": {"status": "absent"},
  "A02": {"status": "present", "severity": "severe",
          "evidence": [{"location": "headline, top center", "seen": "HOLD YOUR GROTIIND"}]},
  "A04": {"status": "n/a"},
  "...": "one entry per criterion code"
}
```

`status` ∈ {present, absent, n/a}. `evidence` required when present.
