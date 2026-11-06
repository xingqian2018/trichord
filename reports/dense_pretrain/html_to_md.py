#!/usr/bin/env python3
"""reports_xx/html_to_md.py - agent-readable Markdown mirror of the eval report HTML (stdlib only).

The HTML from gen_eval_html.py keeps most numbers inside inline-SVG charts (hover tooltips, data-tip) and behind toggle
buttons. This walks the zh HTML in document order and writes everything as text:
  - every chart       -> a table rebuilt from its hover tooltips (rows = lines, columns = rungs; long x axes transposed)
  - every toggle pane -> a labelled block ([data-metric] / [data-ref] / [data-win]), so all metric / window variants are present
  - LR-grid cells     -> m*, eta*, delta text, plus the LR-scan points of each cell's pop-up curve after the table
  - 3-D fit planes    -> fit coefficients + the per-cell optimum points
gen_eval_md.py calls convert() and adds the preface and the raw-number appendix.

    python3 reports_xx/html_to_md.py [in.html] [out.md]     # body only, for debugging
"""
import json
import os
import re
import sys
from html.parser import HTMLParser

VOID = {"br", "img", "meta", "link", "input", "hr", "col", "source", "wbr", "area", "base", "embed", "param", "track"}
INLINE = {"b", "strong", "i", "em", "span", "code", "sup", "sub", "a", "small", "br", "label", "u", "abbr"}
SKIP = {"head", "script", "style", "nav", "colgroup", "button", "title"}
PANE_ATTR = (("data-metric", "data-m", "指标"), ("data-ref", "data-r", "视图"), ("data-win", "data-w", "窗口"))
MAX_ROWS = 40          # a chart with more x points than this is thinned evenly (first and last kept)
MAX_STEP_ROWS = 12     # the same for x axes of training steps (router-health curves, 24 points each)


class Node:
    __slots__ = ("tag", "a", "kids", "parent")

    def __init__(self, tag, a, parent):
        self.tag, self.a, self.kids, self.parent = tag, a, [], parent

    def cls(self):
        return set(self.a.get("class", "").split())

    def desc(self):
        for k in self.kids:
            if isinstance(k, Node):
                yield k
                yield from k.desc()


class Tree(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = self.cur = Node("root", {}, None)

    def handle_starttag(self, tag, attrs):
        n = Node(tag, {k: (v or "") for k, v in attrs}, self.cur)
        self.cur.kids.append(n)
        if tag not in VOID:
            self.cur = n

    def handle_startendtag(self, tag, attrs):           # <line .../> etc.: a leaf, do not descend
        self.cur.kids.append(Node(tag, {k: (v or "") for k, v in attrs}, self.cur))

    def handle_endtag(self, tag):
        n = self.cur
        while n is not self.root and n.tag != tag:
            n = n.parent
        if n is not self.root:
            self.cur = n.parent

    def handle_data(self, d):
        self.cur.kids.append(d)


def strip_tags(s):
    s = re.sub(r"<br\s*/?>", "\n", s)
    s = re.sub(r"<sup>(.*?)</sup>", r"^\1", s)
    s = re.sub(r"<sub>(.*?)</sub>", r"_\1", s)
    return re.sub(r"<[^>]+>", "", s)


def norm(s):
    s = re.sub(r"[ \t\r\f\v]+", " ", s)
    s = re.sub(r" *\n *", "\n", s)
    return s.strip()


def inline(n):
    """Markdown inline text of a node (SVG and buttons dropped)."""
    if isinstance(n, str):
        return n
    if n.tag in ("svg", "script", "style", "button"):
        return ""
    if n.tag == "br":
        return "\n"
    s = "".join(inline(k) for k in n.kids)
    if n.tag in ("b", "strong"):
        t = s.strip()
        return f"**{t}**" if t else s
    if n.tag == "code":
        return f"`{s}`"
    if n.tag == "sup":
        return "^" + s
    if n.tag == "sub":
        return "_" + s
    if n.tag in ("div", "p", "li", "tr", "td", "th"):
        return " " + s + " "
    return s


def cell_text(n):
    """Table-cell text: block children joined with ' · ', line breaks as <br>."""
    parts = []

    def rec(x):
        if isinstance(x, str):
            parts.append(x)
            return
        if x.tag in ("svg", "button", "script", "style"):
            return
        if x.tag == "br":
            parts.append("\n")
            return
        if x.tag in ("div", "p", "ul", "li", "table"):
            parts.append("\x00")
            for k in x.kids:
                rec(k)
            parts.append("\x00")
            return
        parts.append(inline(x))

    for k in n.kids:
        rec(k)
    segs = [norm(s) for s in "".join(parts).split("\x00")]
    s = " · ".join(x for x in segs if x).replace("**", "")   # bold in cells is visual emphasis only
    return s.replace("|", "\\|").replace("\n", "<br>")


def has_block(n):
    return any(isinstance(k, Node) and (k.tag not in INLINE or has_block(k)) for k in n.kids)


def chart_rows(svg):
    """Parse the data-tip tooltips of one chart -> (x labels, series names, {(x, s): value}, notes)."""
    xs, ss, val, notes = [], [], {}, []
    for t in (d.a["data-tip"] for d in svg.desc() if "data-tip" in d.a):
        parts = [norm(strip_tags(p)).replace("●", "").strip() for p in re.split(r"<br\s*/?>", t)]
        if len(parts) == 1:                                  # a single "name: value" point
            head, lines = "", parts
        else:
            head, lines = parts[0], parts[1:]
        if head not in xs:
            xs.append(head)
        for ln in lines:
            if not ln:
                continue
            if ": " in ln:
                s, v = ln.rsplit(": ", 1)
            else:
                s, v = "", ln
            s = s.strip() or "值"
            key = s
            k = 2
            while (head, key) in val:                        # two lines with the same name at one x
                key = f"{s} ({k})"
                k += 1
            if key not in ss:
                ss.append(key)
            val[(head, key)] = v.strip()
    return xs, ss, val, notes


def md_table(head, rows):
    esc = lambda c: (c or " ").replace("\n", "<br>")
    out = ["| " + " | ".join(esc(c) for c in head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(esc(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def chart_md(svg, legend=()):
    xs, ss, val, _ = chart_rows(svg)
    if not ss:
        return ""
    if legend:                                               # tooltips list lines by value; use the legend's fixed order where names match
        rank = {s: i for i, s in enumerate(legend)}
        ss = sorted(ss, key=lambda s: (rank.get(s, len(rank)), ss.index(s)))
    e = lambda c: c.replace("|", "\\|")
    if len(xs) <= 8 and len(ss) > 1:                         # rows = lines, columns = rungs / x points
        return md_table(["线"] + [e(x) or "x" for x in xs], [[e(s)] + [e(val.get((x, s), "·")) for x in xs] for s in ss])
    note = ""
    idx = list(range(len(xs)))
    cap = MAX_STEP_ROWS if all("step" in x for x in xs) else MAX_ROWS
    if len(xs) > cap:                                        # training-step curves (router health) are thinned; LR scans are not
        step = (len(xs) - 1) / (cap - 1)
        idx = sorted({round(i * step) for i in range(cap)})
        note = f"\n\n（共 {len(xs)} 个点，均匀取 {len(idx)} 个，含首尾）"
    return md_table(["横轴"] + [e(s) for s in ss], [[e(xs[i])] + [e(val.get((xs[i], s), "·")) for s in ss] for i in idx]) + note


class Conv:
    def __init__(self, root):
        self.out, self.para, self.legend = [], [], []
        self.labels = {}                                   # (id(scope node), button attr, key) -> button text
        for b in root.desc():
            if b.tag != "button":
                continue
            for _, battr, _ in PANE_ATTR:
                if battr in b.a:
                    sc = b.parent
                    while sc is not None and not ({"card", "pane"} & sc.cls()) and sc.tag not in ("section", "main"):
                        sc = sc.parent
                    self.labels[(id(sc), battr, b.a[battr])] = norm(inline(b) or "".join(k for k in b.kids if isinstance(k, str)))
        self.tabs = {b.a["data-tab"]: norm("".join(k for k in b.kids if isinstance(k, str))) for b in root.desc() if b.tag == "button" and "data-tab" in b.a}

    def flush(self):
        t = norm("".join(self.para))
        self.para = []
        if t:
            self.out.append(t)

    def block(self, s):
        self.flush()
        if s:
            self.out.append(s)

    def label(self, n, attr, battr):
        key = n.a[attr]
        p = n.parent
        while p is not None:
            lab = self.labels.get((id(p), battr, key))
            if lab:
                return lab
            p = p.parent
        return key

    def walk(self, n, hoff):
        for k in n.kids:
            if isinstance(k, str):
                self.para.append(k)
            else:
                self.node(k, hoff)

    def node(self, n, hoff):
        t, c = n.tag, n.cls()
        if "legend" in c:                                    # not printed (the chart tables carry the names); kept for row order
            self.legend = [norm(inline(k)) for k in n.kids if isinstance(k, Node)]
            return
        if t in SKIP or n.a.get("id") == "tip" or {"toggle", "rtoggle"} & c:
            return
        if t == "svg":
            if "data-tip" in n.a or any("data-tip" in d.a for d in n.desc()):
                if any("pie" in p.cls() for p in self.ancestors(n)):   # the pie repeats the table next to it
                    return
                self.block(chart_md(n, self.legend))
            return
        if "plane3d" in c:
            self.block(self.plane(n))
            return
        if t in INLINE and not has_block(n):
            self.para.append(inline(n))
            return
        m = re.fullmatch(r"h([1-6])", t)
        if m:
            self.block("#" * min(6, int(m.group(1)) + hoff) + " " + norm(inline(n)).replace("\n", " "))
            return
        if t == "table":
            self.table(n)
            return
        if t in ("ul", "ol"):
            self.flush()
            self.out.append("\n".join(self.lst(n, 0)))
            return
        if t == "section" and "pane" in c:
            self.block(f"## 页签：{self.tabs.get(n.a.get('data-pane'), n.a.get('data-pane'))}")
            self.walk(n, hoff + 1)
            self.flush()
            return
        for attr, battr, kind in PANE_ATTR:
            if attr in n.a:
                self.block(f"**〔{kind}：{self.label(n, attr, battr)}〕**")
                self.walk(n, hoff)
                self.flush()
                return
        if "ct" in c:                                        # chart / card title
            self.block(f"**{norm(inline(n)).replace(chr(10), ' ')}**")
            return
        self.flush()
        self.walk(n, hoff)
        self.flush()

    @staticmethod
    def ancestors(n):
        p = n.parent
        while p is not None:
            yield p
            p = p.parent

    def lst(self, n, depth):
        lines = []
        for li in (k for k in n.kids if isinstance(k, Node) and k.tag == "li"):
            txt = norm("".join(inline(k) for k in li.kids if not (isinstance(k, Node) and k.tag in ("ul", "ol")))).replace("\n", " ")
            lines.append("  " * depth + "- " + txt)
            for sub in (k for k in li.kids if isinstance(k, Node) and k.tag in ("ul", "ol")):
                lines += self.lst(sub, depth + 1)
        return lines

    def table(self, n):
        trs = [d for d in n.desc() if d.tag == "tr" and next(p for p in self.ancestors(d) if p.tag == "table") is n]
        grid, pend, pops = [], {}, []                        # pend: column -> [text, rows left]
        head_rows = 0
        for r, tr in enumerate(trs):
            cells = [k for k in tr.kids if isinstance(k, Node) and k.tag in ("td", "th")]
            if tr.parent.tag == "thead" or (r == head_rows and cells and all(k.tag == "th" for k in cells) and r == 0):
                head_rows += 1
            row, ci, it = [], 0, iter(cells)
            while True:
                if ci in pend:
                    row.append(pend[ci][0])
                    pend[ci][1] -= 1
                    if pend[ci][1] == 0:
                        del pend[ci]
                    ci += 1
                    continue
                cell = next(it, None)
                if cell is None:
                    if any(k >= ci for k in pend):
                        row.append("")
                        ci += 1
                        continue
                    break
                txt = cell_text(cell)
                svgs = [s for s in cell.desc() if s.tag == "svg" and any("data-tip" in d.a for d in s.desc())]
                for s in svgs:
                    pops.append((r, ci, s))
                cs, rs = int(cell.a.get("colspan") or 1), int(cell.a.get("rowspan") or 1)
                for j in range(cs):
                    row.append(txt)
                    if rs > 1:
                        pend[ci + j] = [txt, rs - 1]
                ci += cs
            grid.append(row)
        if not grid:
            return
        w = max(len(r) for r in grid)
        grid = [r + [""] * (w - len(r)) for r in grid]
        if head_rows == 0:
            head, body = [""] * w, grid
        else:
            head = [" / ".join(dict.fromkeys(x for x in col if x)) for col in zip(*grid[:head_rows])]
            body = grid[head_rows:]
        self.block(md_table(head, body))
        for r, ci, s in pops:                                # LR-scan curves behind the grid cells
            rl = re.sub(r"<br>.*", "", next((x for x in grid[r] if x), ""))
            col = head[ci] if ci < len(head) else ""
            where = " / ".join(x for x in dict.fromkeys((rl, col)) if x)
            self.block(f"**LR 扫描点：{where}**（格子：{grid[r][ci].replace('<br>', ' · ')}）\n\n" + chart_md(s))

    @staticmethod
    def plane(n):
        d = json.loads(n.a["data-plane"])
        lab = d.get("lab", {})
        xa = d.get("x") or {"ref": 8}
        xname = lab.get("L", "L")
        ref = xa.get("ref", 8)
        f = lambda c: "—" if not c else f"η = {c[0]:.5g} × ({xname}/{ref:g})^{c[1]:.4g} × (T/10.4B)^{c[2]:.4g}"
        lines = ["**3D 拟合图的数据**（横轴 " + xname + "，纵轴 " + lab.get("T", "T") + "，高度 " + lab.get("eta", "η*") + "）", "",
                 f"- 灰（参考）：{f(d.get('arch'))}", f"- 橙（我们拟合的）：{f(d.get('fit'))}", f"- 青（Boxiang）：{f(d.get('box'))}"]
        if d.get("fix"):
            lines.append(f"- fix：{f(d['fix'])}")
        rows = [[p.get("lab", ""), f"{p['X']:.4g}", f"{p['T'] / 1e9:.1f}B", f"{p['eta']:.5f}", "是" if p.get("ex") else ""] for p in d.get("pts", [])]
        return "\n".join(lines) + "\n\n" + md_table(["格子", xname, "T", "实测最优 η*", "外推"], rows)


def convert(path):
    tr = Tree()
    tr.feed(open(path, encoding="utf-8").read())
    body = next((d for d in tr.root.desc() if d.tag == "body"), tr.root)
    c = Conv(tr.root)
    c.walk(body, 0)
    c.flush()
    md = "\n\n".join(x for x in c.out if x.strip())
    return re.sub(r"\n{3,}", "\n\n", md) + "\n"


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "qwen3_scaling_ladder_eval.html")
    md = convert(src)
    if len(sys.argv) > 2:
        open(sys.argv[2], "w").write(md)
    else:
        sys.stdout.write(md)
