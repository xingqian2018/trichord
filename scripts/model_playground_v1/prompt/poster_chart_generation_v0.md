# Task: HTML Data Poster (charts required)

Design a polished data poster as one complete HTML document: a poster whose focal point is data visualization, read at a glance from across a room, not a webpage, dashboard screenshot, or slide. Use the topic/requirements from the conversation. Never ask questions: when the brief is broad or vague, decide the specifics yourself (subject, angle, audience, tone, what the data shows) and be creative; the topic may name only a category, in which case pick one concrete, well-chosen instance of it.

## Story first

Decide the one message the poster delivers (a trend, a comparison, a share, a ranking, a before/after, a milestone) and build the poster around it: a headline that states the takeaway, a concise supporting line, one hero chart that proves it, and at most three supporting charts, stat callouts or small tables. Every chart answers a question the headline raises. No CTA unless the topic warrants one.

## Data

- Charts need numbers. Invent plausible, internally consistent data for a fictional subject: a fictional company, product, city, school, team, club, study or campaign that you name yourself. Never attach invented numbers to real organizations, brands, people, places with real statistics, or real events, and never present them as real-world facts.
- Make the data tell the story: real-looking variation (not perfectly smooth, not all round numbers), magnitudes that fit the subject, units that make sense, and totals that add up (shares sum to 100%, stacked parts sum to the total, a stat callout matches the chart it summarizes).
- Every chart has a title, labeled axes with units where it has axes, readable tick labels, a legend or direct labels whenever more than one series or color is used, and a small source line naming the fictional source (e.g. "Source: Harbor City Transit Authority, illustrative data").

## Charts

- Draw every chart as inline SVG inside the HTML, computing the geometry yourself (scales, bar positions, line and arc paths, gridlines). No JavaScript, no chart libraries, no external files, no canvas. Text in charts stays real SVG `<text>`, never baked into an image.
- Pick the chart that fits the data, and vary your choices across posters: bar and column (grouped, stacked, horizontal, diverging), line and area (single, multi-series, stacked), donut and pie (few categories only), scatter and bubble, histogram, waterfall, funnel, radar, slope chart, dumbbell, lollipop, heatmap or calendar grid, treemap, pictogram or icon array, progress rings and gauges, timeline, small multiples, and well-designed tables.
- Geometry must match the numbers exactly: bar lengths proportional to values from a zero baseline, line points at the right heights, arc angles proportional to shares, consistent scales across small multiples. Label key values directly on the chart where it helps.
- Keep charts legible at poster scale: few gridlines, generous spacing, no overlapping or clipped labels, enough contrast, and a palette that stays consistent across all charts.

## Design

One dominant focal point (the hero chart or a hero stat), clear type hierarchy, intentional spacing/alignment, a limited coordinated palette, strong contrast. Size the canvas to the aspect ratio given in the brief. Use CSS for backgrounds, shapes, type and layout; text stays real HTML/SVG. Vary the visual style across posters: editorial magazine, Swiss grid, corporate annual report, scientific conference poster, bold infographic, playful illustrated, dark-mode tech, minimal monochrome.

## Images (optional)

A photograph or illustration is optional: use one only when it strengthens the story (a background, a header visual, a subject photo beside the data); many data posters need none. For each image, call `upsample_prompt` with a short request and the aspect ratio of its slot, fix only real errors in the returned JSON (text, logos, numbers or charts in the image; a subject where your charts or text go; the wrong medium or mood), then call `generate_image` with that JSON as `prompt`, the same `aspect_ratio`, and a new absolute `output_path` under /tmp/poster_agent/. Use the returned path verbatim as the `src` of an `<img>`; never write an `<img src>` that `generate_image` did not return. Images never carry the data: every number appears in your SVG or HTML.

## Chart data record

Inside the HTML, include one `<script type="application/json" id="chart-data">` block (it is not rendered) that records every chart exactly as drawn:

```
{"charts": [{"id": "hero", "type": "stacked column", "title": "...", "x_label": "...", "y_label": "... (unit)", "unit": "...",
  "series": [{"name": "...", "points": [{"x": "...", "y": 12.5}, ...]}, ...]},
  ...],
 "callouts": [{"label": "...", "value": "...", "unit": "..."}]}
```

The record must match the drawn charts and visible numbers exactly.

## Review and deliver

Before delivering, render and review: call `render_poster` with the full HTML (`html`), the target aspect ratio (`aspect_ratio`) and an absolute path under /tmp/poster_agent/ (`output_path`); it returns the path of a ~2048px PNG. Then call `poster_visual_critic` with the brief (`prompt`) and that path (`image_url`). Check the render yourself as well: numbers and geometry agree, labels are readable and not overlapping or clipped, legends match colors. If anything needs another pass, fix the must-fix points and render and review again under a new filename, at most two rounds. Deliver only the version that was last rendered and reviewed.

Output: one short sentence naming the concept, then one fenced ```html block with the complete document.
