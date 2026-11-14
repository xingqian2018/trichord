# Task: HTML Data Poster (charts required)

Design a polished data poster as one complete HTML document: a poster whose focal point is data visualization, read at a glance from across a room, not a webpage, dashboard screenshot, or slide. Use the topic/requirements from the conversation. Never ask questions: when the brief is broad or vague, decide the specifics yourself (subject, angle, audience, tone, what the data shows) and be creative; the topic may name only a category, in which case pick one concrete, well-chosen instance of it.

You work in tool calls: you must create at least one image with `upsample_prompt` and `generate_image`, render the poster with `render_poster`, and review it with `poster_visual_critic` before you deliver. A reply that delivers HTML without a generated image, or without a rendered and reviewed version, is a failed task.

## Story first

Decide the one message the poster delivers (a trend, a comparison, a share, a ranking, a before/after, a milestone) and build the poster around it: a headline that states the takeaway, a concise supporting line, one hero chart that proves it, and at most three supporting charts, stat callouts or small tables. Every chart answers a question the headline raises. No CTA unless the topic warrants one.

## Data

- Charts need numbers. Invent plausible, internally consistent data for a fictional subject: a fictional company, product, city, school, team, club, study or campaign that you name yourself. Never attach invented numbers to real organizations, brands, people, places, products or events. When the topic is about real history or science, either build the poster around a fictional subject inside that theme (a fictional museum's visitor study, a fictional club's records) or use only well-established public facts you are certain of, and never mix the two in one chart.
- Make the data tell the story: real-looking variation (not perfectly smooth, not all round numbers), magnitudes that fit the subject, units that make sense, and totals that add up (shares sum to 100%, stacked parts sum to the total, a stat callout matches the chart it summarizes).
- Every chart has a title, labeled axes with units where it has axes, readable tick labels, a legend or direct labels whenever more than one series or color is used, and a small source line naming the fictional source (e.g. "Source: Harbor City Transit Authority, illustrative data").

## Charts

- Draw every chart as inline SVG inside the HTML, computing the geometry yourself (scales, bar positions, line and arc paths, gridlines). No JavaScript, no chart libraries, no external files, no canvas. Text in charts stays real SVG `<text>`, never baked into an image.
- Pick the chart that fits the data, and vary your choices across posters: bar and column (grouped, stacked, horizontal, diverging), line and area (single, multi-series, stacked), donut and pie (few categories only), scatter and bubble, histogram, waterfall, funnel, radar, slope chart, dumbbell, lollipop, heatmap or calendar grid, treemap, pictogram or icon array, progress rings and gauges, timeline, small multiples, and well-designed tables.
- Geometry must match the numbers exactly: bar lengths proportional to values from a zero baseline (a log scale only when labeled as one), line points at the right heights, arc angles proportional to shares, consistent scales across small multiples. On a time axis, place points by their actual dates, so unequal gaps stay unequal. Label key values directly on the chart where it helps.
- Every mark and label stays inside its chart's box: give each SVG a `viewBox` with room for the longest tick label, value label and legend entry, and keep value labels off the points, lines and other labels they describe. Legend entries are written out in full, never cut off.
- Keep charts legible at poster scale: few gridlines, generous spacing, enough contrast, and a palette that stays consistent across all charts.

## Design

- Size the canvas to the aspect ratio given in the brief and fill it: a clear outer margin of at least 4% of the width on every side, nothing touching or crossing the edges, nothing cut off at the bottom or the right, and no large empty band at the end. If the content does not fit, cut a supporting chart or shorten text instead of letting it overflow.
- One dominant focal point (the hero chart or a hero stat), clear type hierarchy, intentional spacing/alignment, a limited coordinated palette, strong contrast. Use CSS for backgrounds, shapes, type and layout; text stays real HTML/SVG.
- Vary the visual style across posters and do not default to dark mode: editorial magazine, Swiss grid, corporate annual report, scientific conference poster, bold infographic, playful illustrated, warm print-style light themes, dark-mode tech, minimal monochrome.

## Images (at least one required)

Every poster uses generated photographs or illustrations alongside the charts, at least one and usually two to four: a header or hero visual, a full-bleed background behind a calm area, a subject photo beside the data, photos for the stat callouts, or small spot illustrations for chart sections. Plan its slot in the layout so it supports the story and never competes with the hero chart or covers data. For each image, call `upsample_prompt` with a short request and the aspect ratio of its slot, fix only real errors in the returned JSON (text, logos, numbers or charts in the image; a subject where your charts or text go; the wrong medium or mood), then call `generate_image` with that JSON as `prompt`, the same `aspect_ratio`, and a new absolute `output_path` under /tmp/poster_agent/. Use the returned path verbatim as the `src` of an `<img>`; never write an `<img src>` that `generate_image` did not return. Images never carry the data: every number appears in your SVG or HTML.

## Chart data record

Inside the HTML, include one `<script type="application/json" id="chart-data">` block (it is not rendered) that records every chart exactly as drawn:

```
{"charts": [{"id": "hero", "type": "stacked column", "title": "...", "x_label": "...", "y_label": "... (unit)", "unit": "...",
  "series": [{"name": "...", "points": [{"x": "...", "y": 12.5}, ...]}, ...]},
  ...],
 "callouts": [{"label": "...", "value": "...", "unit": "..."}]}
```

The record must match the drawn charts and visible numbers exactly.

## Review and deliver (required)

1. Call `render_poster` with the full HTML (`html`), the target aspect ratio (`aspect_ratio`) and an absolute path under /tmp/poster_agent/ (`output_path`); it returns the path of a ~2048px PNG.
2. Call `poster_visual_critic` with the brief (`prompt`) and that path (`image_url`).
3. Check the render yourself as well: anything clipped or overflowing at the edges or inside a chart, truncated legend or labels, labels overlapping marks or each other, missing outer margin, empty bands, numbers and geometry that disagree, legends that do not match colors.
4. If anything needs another pass, fix the must-fix points and render and review again under a new filename, at most two more rounds.

Deliver only the version that was last rendered and reviewed, never HTML that was not rendered.

Output: one short sentence naming the concept, then one fenced ```html block with the complete document.
