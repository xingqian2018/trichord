# Task: HTML Poster (T2I image required, expressive type, no small text)

Design a polished poster as one complete HTML document — it must read as a designed poster, not a webpage or slide. Use the topic/requirements from the conversation. Never ask questions: when the brief is broad or vague, decide the specifics yourself (subject, angle, audience, tone) and be creative; the topic may name only a category, in which case pick one concrete, well-chosen instance of it. Never invent facts (dates, prices, stats, contacts) — write copy that stands without them.

Write finished, topic-specific copy: a strong headline, a concise supporting line, only necessary details, a CTA only if warranted. No placeholders ("Lorem ipsum", "Your title here", dummy URLs/logos/contacts).

Design: one dominant focal point, clear type hierarchy, intentional spacing/alignment, a limited coordinated palette, strong contrast, balanced imagery/negative space. Size the canvas to the aspect ratio given in the brief. Use CSS for everything CSS can render cleanly (backgrounds, gradients, shapes, type); text stays real HTML, never baked into an image.

## Typography: expressive fonts from Google Fonts

Type is a main design element of this poster, not an afterthought. Load fonts from Google Fonts with a `<link>` in `<head>`, for example:

```
<link href="https://fonts.googleapis.com/css2?family=Abril+Fatface&family=Work+Sans:wght@400;600&display=swap" rel="stylesheet">
```

- Only Google Fonts families render. System font names (Georgia, Helvetica Neue, Avenir, Segoe UI, Arial and the like) are not installed and fall back to a plain default font, so never rely on them; give every `font-family` a Google Fonts family first and a generic family (`serif`, `sans-serif`, `cursive`, `monospace`) last.
- Give the headline, and other display text such as a big number, a date or a short slogan, a font with character that fits the topic and mood: high-contrast display serifs (Abril Fatface, Playfair Display, DM Serif Display, Bodoni Moda, Rozha One), heavy or condensed display sans (Anton, Bebas Neue, Oswald, Archivo Black, Big Shoulders Display, League Gothic), slab and western (Alfa Slab One, Rye, Ultra), script and brush (Pacifico, Lobster, Yellowtail, Kaushan Script, Permanent Marker, Great Vibes, Satisfy), retro and decorative (Monoton, Righteous, Shrikhand, Bungee, Fascinate, Limelight, Cinzel Decorative), hand-drawn and playful (Caveat Brush, Fredoka, Chewy, Bangers, Gloria Hallelujah), elegant and editorial (Cormorant Garamond, Italiana, Marcellus, Syne, Unbounded). These are examples, not a closed list: any Google Fonts family works. Vary your choice across posters instead of defaulting to the same few.
- Pair the display font with a clearly readable text font for the supporting line and details (Inter, Work Sans, Manrope, DM Sans, Source Serif 4, Lora, Libre Franklin, Nunito, and similar). Use at most three families on one poster.
- Use type expressively where it serves the idea: scale contrast between headline and the rest, tight or wide letter-spacing, weight contrast, a headline that interacts with the image or the layout. Keep every word legible: script and decorative fonts only at large display sizes, never for small text or long lines.

## No small text

Every word on the poster must be readable at a glance from a few meters away. Measure sizes against the shorter side of the canvas (S):

- Headline: at least 7% of S, usually much larger.
- Supporting line and body text: at least 3% of S.
- The smallest text anywhere, including labels, captions, dates, venue lines, credits and footers: at least 2.2% of S. On a 1080 px wide portrait canvas that is 24 px; on a 1600x900 canvas it is 20 px.

If the copy does not fit at these sizes, cut words, merge or drop details, or remove a block instead of shrinking the type. No fine print, no long paragraphs, no dense lists. Light or thin weights need extra size and contrast.

## Images

**At least one original photograph/illustration is mandatory as the focal point** — an all-CSS composition with no generated image fails the brief. Pick a subject yourself if the brief doesn't specify one.

Each required visual takes two tool calls, made **before** writing the final HTML. A poster may use several images: do both steps once per visual, each with its own request, its own `aspect_ratio` for that slot, and a distinct `output_path`.

1. Call `upsample_prompt` with a short request for the image and the aspect ratio of its layout slot:

```
upsample_prompt(
  prompt="Short image request: subject, setting, medium/style, mood, and where space is reserved for overlaid HTML text. No text/logos/watermarks in the image.",
  aspect_ratio="3:2"            # one of 16:9, 3:2, 4:3, 1:1, 3:4, 2:3, 9:16 — match the layout slot
)
```

It returns the image expanded into a dense structured JSON description (the format Cosmos3 renders from).

2. Read the JSON and fix only real errors: anything that contradicts your request or the layout (a subject in the space reserved for text, text, logos or watermarks in the image, the wrong medium or mood). Do not rewrite it otherwise. Then call `generate_image` with that JSON object as `prompt`, the same `aspect_ratio`, and a new file path:

```
generate_image(
  prompt={...},                 # the structured JSON object from upsample_prompt, fixed where needed
  aspect_ratio="3:2",
  output_path="/tmp/poster_agent/<new file name>.png"  # absolute path under /tmp/poster_agent/
)
```

`generate_image` returns the absolute path of the saved image. Use that path **verbatim** as the `src` of a real `<img>`, put position/size CSS and `object-fit` on the `<img>` (or its wrapper), and add real alt text. Never write an `<img src>` that a `generate_image` call in this conversation did not return — no placeholders, directives, or invented paths. If `generate_image` reports a problem with the JSON, fix exactly that and call it again; if an image fails or does not fit, call `upsample_prompt` again with an adjusted request, or redesign that slot in CSS. Only deliver once every required image exists and the poster has been rendered and reviewed.

## Review and deliver

Before delivering, render and review: call `render_poster` with the full HTML (`html`), the target aspect ratio (`aspect_ratio`) and an absolute path under /tmp/poster_agent/ (`output_path`); it returns the path of a ~2048px PNG. Then call `poster_visual_critic` with the brief (`prompt`) and that path (`image_url`). Check the render yourself as well: any text that looks small, any font that came out as a plain default instead of the font you chose, clipped or colliding text. If anything needs another pass, fix the must-fix points (adjust HTML/CSS or regenerate an image), then render and review again under a new filename — at most two rounds. Deliver only the version that was last rendered and reviewed.

Output: one short sentence naming the concept, then one fenced ```html block with the complete document.
