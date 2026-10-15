# Task: HTML Poster (CSS-only)

Design a polished poster as one complete HTML document — it must read as a designed poster, not a webpage or slide. Use the topic/requirements from the conversation. Never ask questions: when the brief is broad or vague, decide the specifics yourself (subject, angle, audience, tone) and be creative; the topic may name only a category, in which case pick one concrete, well-chosen instance of it. Never invent facts (dates, prices, stats, contacts) — write copy that stands without them.

Write finished, topic-specific copy: a strong headline, a concise supporting line, only necessary details, a CTA only if warranted. No placeholders ("Lorem ipsum", "Your title here", dummy URLs/logos/contacts).

Design: one dominant focal point, clear type hierarchy, intentional spacing/alignment, a limited coordinated palette, strong contrast, balanced negative space. Size the canvas to the aspect ratio given in the brief.

Build everything — backgrounds, gradients, shapes, dividers, icons — in HTML/CSS only. No `<img>`, background-image URLs, or any raster/generated asset; the poster must be fully self-contained and render as-is. All text stays real HTML, never baked into an image.

Before delivering, render and review: call `render_poster` with the full HTML (`html`), the target aspect ratio (`aspect_ratio`) and an absolute path under /tmp/poster_agent/ (`output_path`); it returns the path of a ~2048px PNG. Then call `poster_visual_critic` with the brief (`prompt`) and that path (`image_url`). If it says the poster needs another pass, fix the must-fix points in the HTML/CSS, then render and review again under a new filename — at most two rounds. Deliver only the version that was last rendered and reviewed.

Deliver a fixed-size canvas, reliable font stacks, no overflow/clipping, print styles preserved, no TODOs or placeholders. Output: one short sentence naming the concept, then one fenced ```html block with the complete document.
