# Task: HTML Poster (T2I image required)

Design a polished poster as one complete HTML document — it must read as a designed poster, not a webpage or slide. Use the topic/requirements from the conversation. Never ask questions: when the brief is broad or vague, decide the specifics yourself (subject, angle, audience, tone) and be creative; the topic may name only a category, in which case pick one concrete, well-chosen instance of it. Never invent facts (dates, prices, stats, contacts) — write copy that stands without them.

Write finished, topic-specific copy: a strong headline, a concise supporting line, only necessary details, a CTA only if warranted. No placeholders ("Lorem ipsum", "Your title here", dummy URLs/logos/contacts).

Design: one dominant focal point, clear type hierarchy, intentional spacing/alignment, a limited coordinated palette, strong contrast, balanced imagery/negative space. Size the canvas to the aspect ratio given in the brief. Use CSS for everything CSS can render cleanly (backgrounds, gradients, shapes, type); text stays real HTML, never baked into an image.

**At least one original photograph/illustration is mandatory as the focal point** — an all-CSS composition with no generated image fails the brief. Pick a subject yourself if the brief doesn't specify one.

Each required visual takes two tool calls, made **before** writing the final HTML. A poster may use several images: do both steps once per visual, each with its own request, its own `aspect_ratio` for that slot, and a distinct `output_path`.

1. Call `upsample_prompt` with a short request for the image and the aspect ratio of its layout slot:

```
upsample_prompt(
  prompt="Short image request: subject, setting, medium/style, mood, and where space is reserved for overlaid HTML text. No text/logos/watermarks in the image.",
  aspect_ratio="16:9"           # one of 1:1, 16:9, 9:16, 4:3, 3:4 — match the layout slot
)
```

It returns the image expanded into a dense structured JSON description (the format Cosmos3 renders from).

2. Read the JSON and fix only real errors: anything that contradicts your request or the layout (a subject in the space reserved for text, text, logos or watermarks in the image, the wrong medium or mood). Do not rewrite it otherwise. Then call `generate_image` with that JSON object as `prompt`, the same `aspect_ratio`, and a new file path:

```
generate_image(
  prompt={...},                 # the structured JSON object from upsample_prompt, fixed where needed
  aspect_ratio="16:9",
  output_path="/tmp/poster_agent/<new file name>.png"  # absolute path under /tmp/poster_agent/
)
```

`generate_image` returns the absolute path of the saved image. Use that path **verbatim** as the `src` of a real `<img>`, put position/size CSS and `object-fit` on the `<img>` (or its wrapper), and add real alt text. Never write an `<img src>` that a `generate_image` call in this conversation did not return — no placeholders, directives, or invented paths. If `generate_image` reports a problem with the JSON, fix exactly that and call it again; if an image fails or does not fit, call `upsample_prompt` again with an adjusted request, or redesign that slot in CSS. Only deliver once every required image exists and the poster has been rendered and reviewed.

Before delivering, render and review: call `render_poster` with the full HTML (`html`), the target aspect ratio (`aspect_ratio`) and an absolute path under /tmp/poster_agent/ (`output_path`); it returns the path of a ~2048px PNG. Then call `poster_visual_critic` with the brief (`prompt`) and that path (`image_url`). If it says the poster needs another pass, fix the must-fix points (adjust HTML/CSS or regenerate an image), then render and review again under a new filename — at most two rounds. Deliver only the version that was last rendered and reviewed.

Output: one short sentence naming the concept, then one fenced ```html block with the complete document.
