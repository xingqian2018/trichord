# -----------------------------------------------------------------------------
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This codebase constitutes NVIDIA proprietary technology and is strictly
# confidential. Any unauthorized reproduction, distribution, or disclosure
# of this code, in whole or in part, outside NVIDIA is strictly prohibited
# without prior written consent.
#
# For inquiries regarding the use of this code in other NVIDIA proprietary
# projects, please contact Cosmos Lab at cosmoslab@exchange.nvidia.com.
# -----------------------------------------------------------------------------
"""Check how much of the visible HTML text a poster caption carries.

Usage:
    python scripts/t2i_ocr_caption/check_text_coverage.py \
        --captions /path/to/caption_sp_for_html_poster_v0.jsonl \
        --input gs://.../poster_..._slice_from_maintable_YYYYmmdd.lance

For every caption row the HTML is looked up by id in the Lance table, the text nodes of <body> are
extracted (skipping script/style/template/noscript and inline display:none), and two numbers are
reported: node recall (the whole node appears in the caption, or in the concatenation of its quoted
segments so a node quoted line by line still counts; case- and whitespace-insensitive)
and word recall (fraction of the node words that appear in the caption).
"""

import argparse
import json
import re
import sys
import unicodedata
from html import unescape
from html.parser import HTMLParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from caption_gateway import load_lance_items  # noqa: E402

SKIP_TAGS = {"script", "style", "template", "noscript", "head", "title", "svg"}
VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}
HIDDEN_STYLE = re.compile(r"display\s*:\s*none|visibility\s*:\s*hidden", re.IGNORECASE)
WORD = re.compile(r"\w+", re.UNICODE)
QUOTED = re.compile(r'"([^"]*)"')


class TextNodes(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[bool] = []
        self.nodes: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in VOID_TAGS:
            return
        hidden = tag in SKIP_TAGS or bool(HIDDEN_STYLE.search(dict(attrs).get("style") or "")) or "hidden" in dict(attrs)
        self.stack.append(hidden)

    def handle_endtag(self, tag):
        if tag in VOID_TAGS:
            return
        if self.stack:
            self.stack.pop()

    def handle_data(self, data):
        text = " ".join(data.split())
        if text and not any(self.stack) and WORD.search(text):
            self.nodes.append(text)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", unescape(text)).lower()
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return " ".join(text.split())


def visible_text_nodes(html: str) -> list[str]:
    parser = TextNodes()
    parser.feed(html)
    return parser.nodes


def coverage(html: str, caption: str) -> dict:
    nodes = visible_text_nodes(html)
    cap = normalize(caption)
    joined = normalize(" ".join(QUOTED.findall(unicodedata.normalize("NFKC", caption).replace("\u201c", '"').replace("\u201d", '"'))))
    cap_words = set(WORD.findall(cap))
    hit_nodes, total_words, hit_words, missing = 0, 0, 0, []
    for node in nodes:
        norm = normalize(node)
        words = WORD.findall(norm)
        total_words += len(words)
        hit_words += sum(w in cap_words for w in words)
        if norm in cap or norm in joined:
            hit_nodes += 1
        else:
            missing.append(node)
    return {
        "num_nodes": len(nodes),
        "node_recall": hit_nodes / len(nodes) if nodes else 1.0,
        "word_recall": hit_words / total_words if total_words else 1.0,
        "missing": missing,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--captions", required=True, nargs="+")
    parser.add_argument("--input", required=True)
    parser.add_argument("--id_column", default="uuid")
    parser.add_argument("--image_column", default="image_s3_range")
    parser.add_argument("--html_column", default="caption_original_html")
    parser.add_argument("--show_missing", type=int, default=0)
    args = parser.parse_args()

    rows_by_file = {}
    for path in args.captions:
        rows = {}
        with open(path) as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    if row.get("caption"):
                        rows[row["id"]] = row
        rows_by_file[path] = rows
    wanted = set().union(*(set(r) for r in rows_by_file.values()))
    args.sample, args.seed = 0, 0
    html_by_id = {item["id"]: item.get("caption_html", "") for item in load_lance_items(args, wanted_ids=wanted)}

    for path, rows in rows_by_file.items():
        stats = [(row_id, coverage(html_by_id[row_id], row["caption"]), len(row["caption"].split())) for row_id, row in rows.items()]
        n = len(stats)
        print(f"== {Path(path).name}: {n} captions")
        print(f"   node recall {sum(s['node_recall'] for _, s, _ in stats) / n:.3f}  word recall {sum(s['word_recall'] for _, s, _ in stats) / n:.3f}"
              f"  perfect {sum(s['node_recall'] == 1.0 for _, s, _ in stats)}/{n}  words avg {sum(w for _, _, w in stats) / n:.0f} max {max(w for _, _, w in stats)}")
        for row_id, s, _ in sorted(stats, key=lambda x: x[1]["node_recall"])[: args.show_missing]:
            print(f"   {row_id[:8]} nodes {s['num_nodes']} node_recall {s['node_recall']:.2f} missing {s['missing'][:6]}")


if __name__ == "__main__":
    main()
