#!/usr/bin/env python3
"""Render a reports_xx markdown report to a self-contained HTML page (no external assets).

Usage: python3 reports_xx/render_html.py reports_xx/<report>.md [out.html]
Default output: same path with .html.
"""
import html
import re
import sys
from pathlib import Path

from markdown_it import MarkdownIt

CSS = """
:root { --fg:#1f2328; --muted:#59636e; --border:#d1d9e0; --head:#f6f8fa; --good:#1a7f37; --bad:#cf222e; --accent:#0969da; }
@media (prefers-color-scheme: dark) {
  :root { --fg:#e6edf3; --muted:#9198a1; --border:#3d444d; --head:#151b23; --good:#3fb950; --bad:#f85149; --accent:#4493f8; }
  body { background:#0d1117; }
}
body { color:var(--fg); font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif; margin:0; }
main { max-width:1200px; margin:0 auto; padding:24px 32px 64px; }
nav { position:sticky; top:0; background:inherit; border-bottom:1px solid var(--border); padding:8px 32px; font-size:13px;
      backdrop-filter:blur(6px); z-index:1; }
nav a { color:var(--accent); text-decoration:none; margin-right:14px; white-space:nowrap; }
h1 { font-size:26px; border-bottom:1px solid var(--border); padding-bottom:8px; }
h2 { font-size:20px; margin-top:36px; border-bottom:1px solid var(--border); padding-bottom:4px; }
code { background:var(--head); padding:1px 5px; border-radius:4px; font-size:12.5px; }
table { border-collapse:collapse; margin:12px 0; display:block; overflow-x:auto; max-width:100%; }
th, td { border:1px solid var(--border); padding:5px 10px; white-space:nowrap; }
th { background:var(--head); font-weight:600; }
tr:nth-child(even) td { background:color-mix(in srgb, var(--head) 50%, transparent); }
td { font-variant-numeric:tabular-nums; }
.neg { color:var(--good); }
.pos { color:var(--bad); }
hr { border:0; border-top:1px solid var(--border); margin:28px 0; }
em { color:var(--muted); }
"""

# dCE values like (−0.047), −0.0034, +0.0007: negative = better (green), positive = worse (red).
NEG = re.compile(r"(?<![\w.])(−\d+\.\d+)")
POS = re.compile(r"(?<![\w.])(\+\d+\.\d+)")


def color_deltas(fragment: str) -> str:
    # Only touch table cells, so prose like "1.15×" or dates stay as-is.
    def cell(m):
        body = NEG.sub(r'<span class="neg">\1</span>', m.group(2))
        body = POS.sub(r'<span class="pos">\1</span>', body)
        return m.group(1) + body + m.group(3)

    return re.sub(r"(<td[^>]*>)(.*?)(</td>)", cell, fragment, flags=re.S)


def main():
    src = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_suffix(".html")
    md = MarkdownIt("commonmark", {"html": False}).enable("table")
    tokens = md.parse(src.read_text())

    # Give h2 headings ids for the nav bar.
    nav, n = [], 0
    for i, t in enumerate(tokens):
        if t.type == "heading_open" and t.tag == "h2":
            n += 1
            anchor = f"s{n}"
            t.attrSet("id", anchor)
            nav.append((anchor, re.split(r"[:(：（]", tokens[i + 1].content)[0].strip()))

    body = color_deltas(md.renderer.render(tokens, md.options, {}))
    title = next((tokens[i + 1].content for i, t in enumerate(tokens) if t.type == "heading_open" and t.tag == "h1"), src.stem)
    nav_html = "".join(f'<a href="#{a}">{html.escape(text)}</a>' for a, text in nav)
    out.write_text(
        f'<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{html.escape(title)}</title><style>{CSS}</style></head>\n"
        f"<body><nav>{nav_html}</nav><main>\n{body}</main></body></html>\n"
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
