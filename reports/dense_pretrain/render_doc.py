#!/usr/bin/env python3
"""reports_xx/render_doc.py - render a documentation Markdown file (a guide, not a results report) to a self-contained HTML page.

    <python with markdown_it> reports_xx/render_doc.py reports_xx/<doc>.md [out.html]     (default output: same name with .html)
    e.g. $U/envs/dpl/bin/python render_doc.py doc_isolated_eval_guide.md doc_isolated_eval_guide_en.md   (several files at once)

    <python> reports_xx/render_doc.py --embed-source <doc>_en.md      the Markdown source is stored inside the page (not shown), so the
                                                                      .md file can be deleted and the page is still editable
    <python> reports_xx/render_doc.py --extract <doc>_en.html [out.md]   writes that source back out (default: same name with .md)

Same look as render_html.py (its CSS is reused), with what a text-heavy document needs and that script does not have: table cells
that wrap, styled code blocks and quotes, `lang` taken from the file name (`*_en.md` -> en, otherwise zh), a nav bar of the h2 titles.
The system python has no markdown_it: use the dpl env (users/xingqianx/envs/dpl/bin/python).
"""
import html
import sys
from pathlib import Path

from markdown_it import MarkdownIt

from render_html import CSS as BASE_CSS

CSS = BASE_CSS + """
main { max-width:1080px; }
nav { background:color-mix(in srgb, Canvas 92%, transparent); }
p, li { line-height:1.65; }
h3 { font-size:16px; margin-top:26px; }
table { display:table; width:100%; }
th, td { white-space:normal; vertical-align:top; text-align:left; overflow-wrap:anywhere; }
th { white-space:nowrap; }
td:first-child { min-width:8em; }
td code, li code, p code { overflow-wrap:anywhere; }
pre { background:var(--head); border:1px solid var(--border); border-radius:6px; padding:10px 14px; overflow-x:auto; line-height:1.45; }
pre code { background:none; padding:0; font-size:12.5px; white-space:pre; }
blockquote { margin:14px 0; padding:8px 16px; border-left:4px solid var(--accent); background:var(--head); border-radius:0 6px 6px 0; }
blockquote p { margin:4px 0; }
"""


SRC_OPEN, SRC_CLOSE = '<script type="text/markdown" id="md-source">', "</script>"


def render(src: Path, out: Path, embed: bool = False) -> None:
    md = MarkdownIt("commonmark", {"html": False}).enable("table")
    text = src.read_text()
    tokens = md.parse(text)
    nav, n = [], 0
    for i, t in enumerate(tokens):
        if t.type == "heading_open" and t.tag == "h2":
            n += 1
            t.attrSet("id", f"s{n}")
            nav.append((f"s{n}", tokens[i + 1].content))
    body = md.renderer.render(tokens, md.options, {})
    title = next((tokens[i + 1].content for i, t in enumerate(tokens) if t.type == "heading_open" and t.tag == "h1"), src.stem)
    lang = "en" if src.stem.endswith("_en") else "zh"
    nav_html = "".join(f'<a href="#{a}">{html.escape(text)}</a>' for a, text in nav)
    out.write_text(f'<!doctype html>\n<html lang="{lang}"><head><meta charset="utf-8">'
                   f'<meta name="viewport" content="width=device-width, initial-scale=1">'
                   f"<title>{html.escape(title)}</title><style>{CSS}</style></head>\n"
                   f"<body><nav>{nav_html}</nav><main>\n{body}</main>"
                   + (f"\n{SRC_OPEN}{html.escape(text, quote=False)}{SRC_CLOSE}" if embed else "") + "</body></html>\n")
    print(f"wrote {out}" + (" (Markdown source embedded)" if embed else ""))


def extract(page: Path, out: Path) -> None:
    h = page.read_text()
    if SRC_OPEN not in h:
        sys.exit(f"{page} has no embedded Markdown source")
    a = h.index(SRC_OPEN) + len(SRC_OPEN)
    out.write_text(html.unescape(h[a:h.index(SRC_CLOSE, a)]))
    print(f"wrote {out}")


def main() -> None:
    argv = sys.argv[1:]
    if argv and argv[0] == "--extract":
        page = Path(argv[1])
        extract(page, Path(argv[2]) if len(argv) > 2 else page.with_suffix(".md"))
        return
    embed = "--embed-source" in argv
    args = [Path(a) for a in argv if a != "--embed-source"]
    if len(args) == 2 and args[1].suffix == ".html":
        render(args[0], args[1], embed)
        return
    for src in args:
        render(src, src.with_suffix(".html"), embed)


if __name__ == "__main__":
    main()
