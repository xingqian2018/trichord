# VLM filter rubric — synthetic T2I training pairs

Each item is one (image, caption) pair produced by a T2I model from a user prompt and
then re-captioned. The pair is meant to become T2I training data. Judge whether it is
safe to train on. Look at the actual image (Read the image_path) before judging; never
judge from the prompt or caption alone.

## Checks

1. **render_failure** — blank, placeholder ("Image blocked by safety filter"), solid color,
   corrupt, or heavily noised image. Always `reject`.
2. **prompt_adherence** (1-5) — are the subject, count, attributes, layout, style and
   setting of the prompt present? 5 = everything; 3 = one notable element missing/wrong;
   1 = largely a different image.
3. **text_accuracy** (1-5, or null when the prompt asks for no legible text) — compare
   every quoted string in the prompt with what is actually rendered. Read the letters.
   5 = all strings exact; 4 = one tiny slip in minor text; 3 = a few misspellings or
   missing/extra items; 2 = most strings wrong; 1 = gibberish. For non-Latin scripts
   (Korean, Arabic, ...) judge whether the glyphs form the requested words.
   Also note unrequested gibberish text appearing in the image.
4. **artifacts** (1-5) — anatomy (hands, fingers, limbs, faces, eyes), merged/duplicated
   objects, impossible geometry, broken perspective, melted textures, seams, watermarks.
   5 = none visible; 3 = noticeable but localized; 1 = dominates the image.
5. **aesthetic** (1-5) — composition, lighting, coherence as a finished image.
6. **caption_consistency** (1-5) — does the caption describe THIS image? Pay special
   attention to captions that copy text/details from the prompt that the image does not
   actually show (e.g. caption lists "Mon..Sun" but image shows "Mon Tue Wed Fru Sun").
   5 = every claim verifiable in the image; 3 = one or two false claims; 1 = mostly false.

## Verdict

- `reject`  — render_failure, or prompt_adherence <= 2, or artifacts <= 2,
              or text_accuracy <= 2.
- `fix_caption` — image is acceptable (no reject condition) but caption_consistency <= 3:
              the image is usable only if the caption is rewritten from the image.
- `keep`    — everything else.

Be strict and concrete. `issues` must name the specific defect you saw (e.g.
"'Fru' instead of 'Thu'; Sat column missing", "left hand has 6 fingers"), not generic words.

## Output

A JSON array, one object per item, in input order:

{"uuid": str, "model": str, "verdict": "keep"|"fix_caption"|"reject",
 "render_failure": bool, "prompt_adherence": int, "text_accuracy": int|null,
 "artifacts": int, "aesthetic": int, "caption_consistency": int,
 "issues": [str, ...]}
