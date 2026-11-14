#!/usr/bin/env python3
"""reports_xx/prod_isodoc_section.py - the report's tab "prod iso-doc" (user 2026-11-13: "请+一个iso-doc测试tab", "就是prod iso-doc",
"记得测试应该是分为8k vs 32k"): doc-isolated val loss of the PRODUCTION run's checkpoints, by training step, at the 8K and the 32K window.

Two kinds of result per checkpoint and window, the same quantity where both exist (same documents, same tokens, same scoring code):
  full    results_xx/<RUN>[_W32K]_it<N>.json   the 115-split suite of run_doc_eval.sh (stage_prod_ckpt.py + collect_doc_eval.py): the 7
                                              collections and overall_tsqw, both windows from one model load
  pooled  results_xx/prod_isodoc/<RUN>[_W32K]_iter<N>.json   boxiangw's loss-parity doc evals (collect_prod_isodoc.py): pooled sets, so
                                              only the collections he scored (val_v2, v3small) and no overall_tsqw
A full result replaces the pooled one of the same checkpoint and window. Reference rows come from the ordinary doc-isolated results
(R of gen_eval_html.load()). The chart buttons are the seven classic metric options; the table lists every collection, with the columns
of the reference tab's table (TABLE_COLS; model size and vocab size first). A checkpoint / window without a score shows "-".

    from prod_isodoc_section import prod_tab
    prod_tab(lang, esc, R, ladder_size, sizes, vocab) -> (tab label, pane HTML), or None when there is no result yet
"""
import glob
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results_xx", "prod_isodoc")   # pooled sets
FULL = os.path.join(HERE, "..", "results_xx")                     # 115-split suite
RUN = "ULTRA59B_MUONH_KD_QBK3"
SHORT = "59B"
COL = "--c1"
WINS = [(8192, "8K", ""), (32768, "32K", "_W32K")]     # (window, label, Change-ID suffix of the full and of the reference results)
# the classic metric options, as METRICS of gen_eval_html.py (user 11-13: "我们不是经典的 overall_tsqw overall_tw overall_macro val_v2
# uncheatable codefresh private_internal 选项吗"): always all seven; a metric a checkpoint has no score for shows "-"
METRICS = [("00_overall_tsqw", "overall_tsqw"), ("00_overall_tw", "overall_tw"), ("00_overall_macro", "overall_macro"), ("01_val_v2_overall_tw", "val_v2"),
           ("02_uncheatable_tw", "uncheatable"), ("03_codefresh_tw", "codefresh"), ("04_private_internal_tw", "private_internal")]
# the table's columns (user 11-13 20:55: "这个要列出全部的，就是 model size / vocab size / overall_tsqw / overall_tw / overall_macro / val_v2 /
# uncheatable / codefresh / v3small / private_internal / pile_test / pile_10k"): REF_COLS of gen_eval_html.py, i.e. the columns of the
# 外部参考 tab's doc-isolated table, after "model size" and "vocab size". The chart buttons stay the seven classic options above.
TABLE_COLS = [("00_overall_tsqw", "overall_tsqw"), ("00_overall_tw", "overall_tw"), ("00_overall_macro", "overall_macro"), ("01_val_v2_overall_tw", "val_v2"),
              ("02_uncheatable_tw", "uncheatable"), ("03_codefresh_tw", "codefresh"), ("04_v3small_tw", "v3small"),
              ("04_private_internal_tw", "private_internal"), ("05_pile_test_tw", "pile_test"), ("05_pile_10k_tw", "pile_10k")]
VOCAB = 151936             # the production run's padded vocab (its argv: --padded-vocab-size 151936), the Qwen3 embedding rows
# a reference whose run record is not in the shared results: model size and vocab size from its architecture twin (the KD arm of data
# derisk round 2 is the LATENT2_E2X architecture; KD changes the loss, not the model)
SIZE_AS = {"PROD1_DPSK_LATENT2_E2X_KD_A0P8_S64Y_MIMO25_PPFULL": "PROD1_DPSK_LATENT2_E2X"}
# reference rows: (Change ID, rung, text key, colour, drawn as a line in the charts). All four share the Qwen3 token ids, so their
# loss per token is comparable with the production run's (the MiMo teacher is far below the others: table only, it would flatten the charts)
REFS = [("PROD1_DPSK", "q5", "r_p5", "--c3", True),
        ("PROD1_DPSK_LATENT2_E2X", "q4", "r_ce", "--c3", False),
        ("PROD1_DPSK_LATENT2_E2X_KD_A0P8_S64Y_MIMO25_PPFULL", "q3", "r_kd", "--c3", False),
        ("QWEN3_30B_A3B_BASE", "ref", "r_qwen", "--muted", True),
        ("MIMO_V25_PRO_BASE", "ref", "r_teacher", "--muted", False)]
STATS = [("loss", "val_loss"), ("bpb", "bpb")]
# The pooled-set evals (boxiangw's loss-parity doc evals, results_xx/prod_isodoc/<RUN>_iter<N>.json) are NOT shown (user 11-13: "这个是boxiang的
# 分数对吗，这个暂时不要放上来"): the tab holds only the scores of OUR full-suite evals. The files stay on disk; True shows them again.
SHOW_POOLED = False
TEACHER = {"mimo-v2.5-pro-base": "MiMo-V2.5-Pro-Base", "mimo-v2.6-pro-rl": "MiMo-V2.6-Pro-RL"}   # teacher-cache directory name -> model name
TX = {"zh": {
    "tab": "prod iso-doc",
    "what": "生产 run {run}（{size}；{L} 层，hidden {h}，{E} 个 expert 选 {k}；KD teacher {teacher}，α {alpha}；训练长度 {seq}，共 {total} 步）各 checkpoint 的 doc-isolated val loss。",
    "how": "测法和 doc-isolated 页相同：每篇验证文档单独成一个样本，不带别的文档的上下文，打分代码是同一份。窗口分两种：8K 和所有旧结果可比；32K 和这个 run 的训练长度一致。",
    "kinds": "结果有两种。全量：我们的 115 个子集、7 个集合，有 overall_tsqw，8K 和 32K 出自同一次模型加载。合并集合：Boxiang 的流程给 downstream 评测做 loss 对齐用的评测，只有 val_v2 和 v3small，一个集合一个 loss。两者的文档数和 token 数相同，同一个集合的数值是同一个量，可以和 doc-isolated 页直接对比。全量评测分三个部分并行跑，先跑完的集合先显示，还没跑完的显示 “-”。",
    "ours_only": "结果只来自我们自己的全量评测：115 个子集、7 个集合，8K 和 32K 出自同一次模型加载。全量评测分三个部分并行跑，先跑完的集合先显示，还没跑完的显示 “-”。",
    "w_line": "{w} 窗口：{parts}。", "w_full": "全量 = step {its}", "w_part": "全量已跑完一部分 = step {its}", "w_pooled": "只有合并集合 = step {its}", "w_none": "还没有结果，下面显示 “-”",
    "h_curve": "val loss 随训练步数的变化",
    "curve_note": "用上面的按钮换指标，用卡片里的 8K / 32K 页签换窗口（图和下面的表一起切换）。横轴是训练步数，下面一行是占全程的比例。虚线是参考：PROD1_DPSK q5（ladder 上训练完成的 checkpoint）和外部基线 Qwen3-30B-A3B-Base，取同一窗口的数值。",
    "ylab": "val loss（nats / token）", "no_win": "{w} 窗口还没有这个指标的结果。", "step": "step", "ckpt": "checkpoint", "prog": "训练进度", "src": "来源",
    "win": "{w} 窗口", "full": "全量", "part": "全量（已跑完的部分）", "pooled": "合并集合", "ours": "我们", "ref_h": "参考",
    "r_kd": "ladder E2X + KD q3（teacher MiMo-V2.5）", "r_ce": "ladder E2X q4（CE）",
    "r_qwen": "Qwen3-30B-A3B-Base", "r_teacher": "MiMo-V2.5-Pro-Base（KD teacher）",
    "r_p5": "PROD1_DPSK · q5", "c_p5": "PROD1_DPSK · q5", "c_qwen": "Qwen3-30B-A3B-Base"},
      "en": {
    "tab": "prod iso-doc",
    "what": "Doc-isolated val loss of the checkpoints of the production run {run} ({size}; {L} layers, hidden {h}, top-{k} of {E} experts; KD teacher {teacher}, α {alpha}; trained at {seq}, {total} steps in all).",
    "how": "Same protocol as the doc-isolated tab: every validation document is its own sample with no context from other documents, scored by the same code. Two windows: 8K is comparable with every earlier result; 32K matches the length this run trains at.",
    "kinds": "There are two kinds of result. Full: our 115 subsets in 7 collections, with overall_tsqw, 8K and 32K from one model load. Pooled: the evals Boxiang's pipeline runs as loss-parity reference of the downstream eval, val_v2 and v3small only, one loss per collection. Both score the same documents and tokens, so a collection's value is the same quantity in either and compares directly with the doc-isolated tab. A full eval runs as three parallel parts: collections that are finished are shown, the others show “-” until their part is done.",
    "ours_only": "Only the results of our own full-suite evals are shown: 115 subsets in 7 collections, 8K and 32K from one model load. A full eval runs as three parallel parts: collections that are finished are shown, the others show “-” until their part is done.",
    "w_line": "{w} window: {parts}.", "w_full": "full = step {its}", "w_part": "full, parts finished so far = step {its}", "w_pooled": "pooled only = step {its}", "w_none": "no result yet, shown as “-” below",
    "h_curve": "Val loss vs. training step",
    "curve_note": "The buttons above switch the metric; the 8K / 32K tabs in the card switch the window (the chart and the table below together). The x axis is the training step, with the share of the whole run below it. Dashed lines are references at the same window: PROD1_DPSK q5 (a finished ladder checkpoint) and the external baseline Qwen3-30B-A3B-Base.",
    "ylab": "val loss (nats / token)", "no_win": "No result for this metric at the {w} window yet.", "step": "step", "ckpt": "checkpoint", "prog": "training progress", "src": "source",
    "win": "{w} window", "full": "full", "part": "full (parts finished so far)", "pooled": "pooled", "ours": "ours", "ref_h": "References",
    "r_kd": "ladder E2X + KD q3 (teacher MiMo-V2.5)", "r_ce": "ladder E2X q4 (CE)",
    "r_qwen": "Qwen3-30B-A3B-Base", "r_teacher": "MiMo-V2.5-Pro-Base (KD teacher)",
    "r_p5": "PROD1_DPSK · q5", "c_p5": "PROD1_DPSK · q5", "c_qwen": "Qwen3-30B-A3B-Base"}}


def load():
    """{window: {iteration: {"metrics": {key: {loss, bpb, ..}}, "kind": full | part | pooled, "also": [kinds that filled gaps], "job",
    "date", "by", "train"}}} of RUN."""
    found = []                                            # (rank, window, iteration, entry); rank 0 pooled, 1 parts of a full eval, 2 full
    for p in glob.glob(os.path.join(RESULTS, RUN + "*_iter*.json")):
        d = json.load(open(p))
        if d.get("run") == RUN:
            s, kind = d["source"], d.get("kind", "pooled")
            if kind == "pooled" and not SHOW_POOLED:
                continue
            found.append((0 if kind == "pooled" else 1, int(d["window"]), int(d["iteration"]),
                          {"metrics": d["metrics"], "kind": "part" if kind == "partial" else kind, "job": s["eval_job"], "date": s["evaluated_at"],
                           "by": s["by"], "train": d.get("train")}))
    for win, _, suf in WINS:
        for p in glob.glob(os.path.join(FULL, f"{RUN}{suf}_it[0-9]*.json")):
            d = json.load(open(p))
            if d.get("change") == RUN + suf:
                found.append((2, win, int(d["iteration"]), {"metrics": d["metrics"], "kind": "full", "job": d["eval_job"], "date": d["evaluated_at"],
                                                           "by": "xingqianx", "train": None}))
    out = {}
    for rank, win, it, e in sorted(found, key=lambda x: x[:3]):   # ascending rank: the better source overrides, the others fill what it lacks
        old = out.setdefault(win, {}).get(it)
        if old:
            e = {**e, "metrics": {**old["metrics"], **e["metrics"]}, "train": e["train"] or old["train"],
                 "also": [x for x in old.get("also", []) + [old["kind"]] if x != e["kind"]]}
        out[win][it] = e
    return out


def _legend(esc, items):
    """items = [(name, colour var, dashed)], the report's legend markup."""
    sw = lambda c, dashed: f"repeating-linear-gradient(90deg,var({c}) 0 4px,transparent 4px 7px)" if dashed else f"var({c})"
    return '<div class="legend">' + "".join(f'<span><i style="background:{sw(c, d)}"></i>{esc(n)}</span>' for n, c, d in items) + "</div>"


def _chart(esc, t, pts, refs, total, w=620, h=320):
    """pts = [(step, value)] of the production run; refs = [(name, colour var, value)] drawn as dashed horizontal lines."""
    ml, mr, mt, mb = 56, 178, 14, 42
    xmax = max(10000, math.ceil(max(s for s, _ in pts) * 1.06 / 5000) * 5000)
    tick = next(s for s in (2000, 5000, 10000, 20000, 50000) if xmax / s <= 8)
    ys = [v for _, v in pts] + [v for _, _, v in refs]
    lo, hi = min(ys), max(ys)
    pad = (hi - lo) * 0.08 or 0.01
    lo, hi = lo - pad, hi + pad
    X = lambda s: ml + s / xmax * (w - ml - mr)
    Y = lambda v: mt + (hi - v) / (hi - lo) * (h - mt - mb)
    pct = lambda s: f"{100 * s / total:.1f}%" if total else ""
    o = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img">']
    for i in range(5):
        yv = lo + (hi - lo) * i / 4
        o.append(f'<line x1="{ml}" x2="{w - mr}" y1="{Y(yv):.1f}" y2="{Y(yv):.1f}" stroke="var(--grid)" stroke-width="1"/>')
        o.append(f'<text x="{ml - 6}" y="{Y(yv) + 4:.1f}" text-anchor="end">{yv:.3f}</text>')
    for s in range(0, xmax + 1, tick):
        o.append(f'<text x="{X(s):.1f}" y="{h - mb + 16}" text-anchor="middle">{s // 1000}k</text>')
        if s and total:
            o.append(f'<text x="{X(s):.1f}" y="{h - mb + 29}" text-anchor="middle" style="fill:var(--muted)">{pct(s)}</text>')
    o.append(f'<line x1="{ml}" x2="{w - mr}" y1="{h - mb}" y2="{h - mb}" stroke="var(--axis)"/>')
    o.append(f'<text x="12" y="{(h - mb + mt) / 2:.0f}" transform="rotate(-90 12 {(h - mb + mt) / 2:.0f})" text-anchor="middle">{esc(t["ylab"])}</text>')
    lab = sorted([[Y(v), n, c, v] for n, c, v in refs])
    for i in range(1, len(lab)):                         # end labels of the reference lines: pushed apart when closer than a text line
        if lab[i][0] - lab[i - 1][0] < 13:
            lab[i][0] = lab[i - 1][0] + 13
    for n, c, v in refs:
        o.append(f'<line x1="{ml}" x2="{w - mr}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="var({c})" stroke-width="1.6" stroke-dasharray="5 4"/>')
    for y, n, c, v in lab:
        o.append(f'<text x="{w - mr + 6}" y="{y + 4:.1f}" style="fill:var({c});font-weight:600">{esc(n)} {v:.4f}</text>')
    d = " ".join(f"{'M' if i == 0 else 'L'}{X(s):.1f},{Y(v):.1f}" for i, (s, v) in enumerate(pts))
    o.append(f'<path d="{d}" fill="none" stroke="var({COL})" stroke-width="2" stroke-linejoin="round"/>')
    for s, v in pts:
        o.append(f'<circle cx="{X(s):.1f}" cy="{Y(v):.1f}" r="4.5" fill="var({COL})" stroke="var(--surface-1)" stroke-width="2"/>')
    ls, lv = pts[-1]
    o.append(f'<text class="lbl" x="{X(ls):.1f}" y="{Y(lv) - 10:.1f}" text-anchor="middle">{lv:.4f}</text>')
    for i, (s, v) in enumerate(pts):                     # one hover band per checkpoint: the run and the references, largest on top
        a = X(s) - (X(s) - X(pts[i - 1][0])) / 2 if i else ml
        b = X(s) + (X(pts[i + 1][0]) - X(s)) / 2 if i + 1 < len(pts) else w - mr
        rows = sorted([(SHORT, COL, v)] + list(refs), key=lambda r: r[2], reverse=True)
        tip = f"<b>{t['step']} {s:,} · {pct(s)}</b>" + "".join(f'<br><span style="color:var({c})">●</span> {esc(n)}: {y:.4f}' for n, c, y in rows)
        o.append(f'<rect class="hit" x="{a:.1f}" y="{mt}" width="{b - a:.1f}" height="{h - mt - mb}" data-tip="{esc(tip)}"/>')
    o.append("</svg>")
    return "".join(o)


def prod_tab(lang, esc, R, ladder_size=None, sizes=None, vocab=None):
    """vocab(Change ID, rung) -> vocab size of a reference row (gen_eval_html: REF_VOCAB / ladder_vocab), or None."""
    t = TX[lang]
    res = load()
    if not res:
        return None
    sizes = sizes or {}
    tr = next((d["train"] for w in res.values() for d in w.values() if d.get("train")), None) or {}
    total = tr.get("total_iterations")
    cols = list(METRICS)
    sep = "、" if lang == "zh" else ", "
    kfmt = lambda n: f"{n // 1024}K" if n else "?"
    size = next((v for k, v in sizes.items() if k.startswith(RUN)), "")
    what = t["what"].format(run=RUN, size=size or "?", L=tr.get("num_layers", "?"), h=tr.get("hidden_size", "?"), E=tr.get("num_experts", "?"),
                            k=tr.get("moe_router_topk", "?"), teacher=TEACHER.get(tr.get("kd_teacher"), tr.get("kd_teacher") or "?"), alpha=tr.get("kd_alpha", "?"),
                            seq=kfmt(tr.get("seq_length")), total=f"{total:,}" if total else "?")
    notes = [what, t["how"], t["kinds"] if SHOW_POOLED else t["ours_only"]]
    for win, wl, _ in WINS:
        parts = []
        for kind in ("full", "part", "pooled"):
            its_k = sorted(it for it, d in res.get(win, {}).items() if d["kind"] == kind)
            if its_k:
                parts.append(t["w_" + kind].format(its=sep.join(f"{i:,}" for i in its_k)))
        notes.append(t["w_line"].format(w=wl, parts=("；" if lang == "zh" else "; ").join(parts) if parts else t["w_none"]))
    out = f'<div class="card"><ul class="f">{"".join(f"<li>{esc(x)}</li>" for x in notes)}</ul></div>'

    def ref_val(ch, q, suf, k, sec="loss"):
        m = R.get((ch + suf, q))
        return m[k].get(sec) if m and k in m else None

    # ---- one card, 8K / 32K as tabs (user 11-13: "8K 32k 变成tab"): per window ONE chart (the metric buttons choose which; the x axis is
    # the training iteration, so it is this tab's own chart, not the rung charts of the doc-isolated tab) and ONE table
    cell = lambda v: f"<td>{v:.4f}</td>" if v is not None else "<td>-</td>"
    its = sorted({it for w in res.values() for it in w})
    head = (f'<th class="l">{esc(t["ckpt"])}</th><th>{esc(t["prog"])}</th><th>model size</th><th>vocab size</th>'
            + "".join(f"<th>{esc(n)}</th>" for _, n in TABLE_COLS))
    txt = lambda v: f"<td>{esc(v)}</td>" if v else "<td>-</td>"
    wid = {win: f"w{win // 1024}" for win, _, _ in WINS}
    charts, tables = "", ""
    for i, (win, wl, suf) in enumerate(WINS):
        charts += f'<div data-win="{wid[win]}"{" hidden" if i else ""}>'
        for j, (k, n) in enumerate(cols):
            pts = [(it, d["metrics"][k]["loss"]) for it, d in sorted(res.get(win, {}).items()) if k in d["metrics"]]
            charts += f'<div data-metric="{k}"{" hidden" if j else ""} style="max-width:900px;margin:0 auto"><div class="ct">{esc(n)} · {esc(t["win"].format(w=wl))}</div>'
            if pts:
                refs = [(t["c_" + key[2:]], c, ref_val(ch, q, suf, k)) for ch, q, key, c, draw in REFS if draw and ref_val(ch, q, suf, k) is not None]
                charts += _legend(esc, [(SHORT, COL, False)] + [(rn, c, True) for rn, c, _ in refs]) + _chart(esc, t, pts, refs, total, w=820, h=360)
            else:
                charts += f'<p class="note" style="margin:8px 0">{esc(t["no_win"].format(w=wl))}</p>'
            charts += "</div>"
        charts += "</div>"
        tables += f'<div data-win="{wid[win]}"{" hidden" if i else ""}>'
        for si, (sec, _) in enumerate(STATS):
            rows = ""
            for it in its:
                d = res.get(win, {}).get(it)
                rows += (f'<tr><td class="l">{SHORT} · {esc(t["step"])} {it:,}</td><td>{f"{100 * it / total:.1f}%" if total else "-"}</td>'
                         + txt(size) + txt(f"{VOCAB:,}")
                         + "".join(cell(d["metrics"][k].get(sec) if d and k in d["metrics"] else None) for k, _ in TABLE_COLS) + "</tr>")
            first = True
            for ch, q, key, _, _ in REFS:
                vals = [ref_val(ch, q, suf, k, sec) for k, _ in TABLE_COLS]
                if all(v is None for v in vals):
                    continue
                tw = SIZE_AS.get(ch, ch)
                sz = sizes.get(ch) or ((ladder_size(tw, q) or ("",))[0] if ladder_size and q != "ref" else "")
                vc = vocab(tw, q) if vocab else None
                trc = ' class="sep"' if first else ""
                rows += (f'<tr{trc}><td class="l">{esc(t[key])}</td><td>{esc(t["ref_h"]) if first else ""}</td>'
                         + txt(sz) + txt(f"{vc:,}" if vc else "") + "".join(cell(v) for v in vals) + "</tr>")
                first = False
            tables += f'<div data-ref="{sec}"{" hidden" if si else ""}><div style="overflow-x:auto"><table style="display:table;width:100%"><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div></div>'
        tables += "</div>"
    out += f'<div class="msec"><h2>{esc(t["h_curve"])}</h2><p class="note">{esc(t["curve_note"])}</p>'
    out += '<div class="toggle" style="margin-top:0">' + "".join(f'<button data-m="{k}" class="{"on" if j == 0 else ""}">{esc(n)}</button>' for j, (k, n) in enumerate(cols)) + "</div>"
    out += '<div class="card"><div class="rtoggle wtoggle">' + "".join(
        f'<button class="{"on" if i == 0 else ""}" data-w="{wid[win]}">{wl}</button>' for i, (win, wl, _) in enumerate(WINS)) + "</div>"
    out += charts
    out += '<div class="rtoggle" style="margin-top:18px">' + "".join(f'<button class="{"on" if i == 0 else ""}" data-r="{sec}">{esc(n)}</button>' for i, (sec, n) in enumerate(STATS)) + "</div>"
    out += tables + "</div></div>"
    return t["tab"], out
