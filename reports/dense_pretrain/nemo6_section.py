"""reports_xx/nemo6_section.py - the "全量测评范围" chapter of the eval report's 外部参考 tab (user 11-05: its own chapter after
"外部参考：Pretrain doc-isolated 测试", not piled into it): the tree of every benchmark in the suite. Imported by gen_eval_html.py.

The tree is built from reports_xx/nemo6/nemo6_report.yml (the config the runs use), so it cannot drift from what is evaluated: every task
of the config must appear exactly once in BENCH below (asserted). n-shot = the value lm_eval recorded for the pilot model
(results_xx/nemo6/QWEN3_30B_A3B_BASE.json), else the config's num_fewshot, else the number in the task name; left blank when unknown.
"""
import html
import json
import os
import re

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = os.path.join(HERE, "nemo6", "nemo6_report.yml")
PILOT = os.path.join(HERE, "..", "results_xx", "nemo6", "QWEN3_30B_A3B_BASE.json")

# Two big parts, non-generation first (user 11-05 "generation 和 non generation … 两个 big session，non generation 放在前面").
# Non-generation = lm_eval output_type multiple_choice (the log-likelihood of each option after the prompt is compared, nothing is
# generated). Checked 11-05 against the configs in the raw lm_eval results (Global-MMLU / INCLUDE: all 48 / 45 subtasks); every other
# task is generate_until. Asserted below to match the report metrics (multiple choice <=> only acc / acc_norm).
NONGEN = {"commonsense_qa", "arc_challenge_llama", "hellaswag", "openbookqa", "piqa", "winogrande", "social_iqa", "truthfulqa_mc2",
          "race", "adlr_global_mmlu", "adlr_include"}
# kind -> zh / en name, short how (results chapter), long how (scope chapter)
KINDS = [
    ("ll", "非生成类 benchmark", "Non-generation benchmarks", "多选题，按选项概率打分，不生成", "multiple choice scored by option likelihood, nothing generated",
     "多选题：把每个选项接在题目后面，比较模型给各选项的对数概率（acc_norm 按长度归一化），概率最高的选项即答案。不生成文本，所以分数不受回答格式和停止符影响。",
     "Multiple choice: each option is appended to the prompt and the option log-likelihoods are compared (acc_norm: length-normalised); "
     "the most likely option is the answer. Nothing is generated, so answer format and stop strings do not matter."),
    ("gen", "生成类 benchmark", "Generation benchmarks", "生成回答后打分", "answers generated, then scored",
     "模型生成回答（贪心一次，或每题采样 32 次），再抽取答案比对、执行代码，或由 LLM 裁判判对错。分数也受回答格式影响（见结果表里的 ‡）。",
     "The model generates an answer (one greedy decode, or 32 samples per item), which is then matched, executed or graded by an LLM judge. "
     "Scores also depend on the answer format (see ‡ in the result tables)."),
]
# (kind, task_group of the config) -> zh / en name, how it is scored; display order inside each kind. reading / aggregate hold both kinds.
CATS = [
    ("ll", "reasoning", "常识推理", "Commonsense reasoning", "英文多选题", "English multiple choice"),
    ("ll", "reading", "阅读理解", "Reading comprehension", "读一篇文章，答多选题", "multiple choice on a passage"),
    ("ll", "aggregate", "多语言考试", "Multilingual exams", "知识考试多选题，8–9 种非英语语言（含中文），每种语言一个分",
     "knowledge-exam multiple choice in 8-9 non-English languages (incl. Chinese), one score per language"),
    ("gen", "code", "代码", "Code", "生成代码并执行打分", "generated code is executed"),
    ("gen", "aggregate", "综合考试", "Exams", "生成作答（多为 CoT），抽取最终选项 / 答案比对", "generated answer (mostly CoT), final choice / answer extracted"),
    ("gen", "math", "数学", "Math", "生成解答，比对最终答案", "generated solution, final answer checked"),
    ("gen", "reading", "事实问答", "Factual QA", "短答案生成，精确匹配", "short generated answers, exact match"),
    ("gen", "knowledge", "知识", "Knowledge", "LLM 裁判 gpt-oss-120b 判对错", "LLM judge gpt-oss-120b"),
    ("gen", "multilingual", "多语言", "Multilingual", "翻译", "translation"),
]
# benchmark: (category, display name, detail zh, detail en, [(task, variant zh, variant en)])
BENCH = [
    ("code", "MBPP", "sanitized", "sanitized", [("mbpp_sanitized_3shot_sampled", "", "")]),
    ("code", "HumanEval", "", "", [("humaneval_sampled", "", "")]),
    ("code", "LiveCodeBench", "release v6", "release v6", [("lcb_codegen", "greedy", "greedy"), ("lcb_codegen_sampled", "采样", "sampled")]),
    ("code", "BigCodeBench", "", "", [("bigcodebench_complete_greedy", "complete", "complete"), ("bigcodebench_instruct_greedy", "instruct", "instruct")]),
    ("aggregate", "MMLU-Pro", "", "", [("mmlu_pro_5_shot_base", "base", "base"), ("mmlu_pro_0_shot_sampled", "采样", "sampled")]),
    ("aggregate", "MMLU", "string 版", "string", [("mmlu_str", "", "")]),
    ("aggregate", "MMLU-Redux", "string 版", "string", [("mmlu_redux_str", "", "")]),
    ("aggregate", "AGIEval-en", "CoT", "CoT", [("agieval_en_cot", "", "")]),
    ("aggregate", "GPQA-Diamond", "CoT", "CoT", [("gpqa_diamond_cot_n_shot", "greedy", "greedy"), ("gpqa_diamond_cot_n_shot_sampled", "采样", "sampled")]),
    ("aggregate", "SuperGPQA", "CoT", "CoT", [("supergpqa_main_cot_n_shot", "", "")]),
    ("aggregate", "HLE", "纯文本题", "text only", [("hle_text_sampled", "", "")]),
    ("aggregate", "Global-MMLU", "", "", [("adlr_global_mmlu", "", "")]),
    ("aggregate", "MGSM", "native CoT", "native CoT", [("adlr_mgsm", "", "")]),
    ("aggregate", "INCLUDE", "base-44", "base-44", [("adlr_include", "", "")]),
    ("math", "GSM8K", "CoT", "CoT", [("gsm8k_cot_nemo", "", "")]),
    ("math", "Minerva MATH", "", "", [("minerva_math_nemo", "", "")]),
    ("math", "MATH-500", "", "", [("math_500_sampled", "", "")]),
    ("math", "AIME 2024", "", "", [("aime24", "", "")]),
    ("math", "AIME 2025", "", "", [("aime25", "", "")]),
    ("math", "HMMT Feb 2025", "", "", [("hmmt_sampled", "", "")]),
    ("reasoning", "CommonsenseQA", "", "", [("commonsense_qa", "", "")]),
    ("reasoning", "ARC-Challenge", "", "", [("arc_challenge_llama", "", "")]),
    ("reasoning", "HellaSwag", "", "", [("hellaswag", "", "")]),
    ("reasoning", "OpenBookQA", "", "", [("openbookqa", "", "")]),
    ("reasoning", "PIQA", "", "", [("piqa", "", "")]),
    ("reasoning", "WinoGrande", "", "", [("winogrande", "", "")]),
    ("reasoning", "Social IQa", "", "", [("social_iqa", "", "")]),
    ("reasoning", "TruthfulQA", "MC2", "MC2", [("truthfulqa_mc2", "", "")]),
    ("reading", "TriviaQA", "", "", [("trivia_qa", "", "")]),
    ("reading", "PopQA", "", "", [("popqa_5shot", "", "")]),
    ("reading", "Natural Questions", "", "", [("naturalqs_5shot", "", "")]),
    ("reading", "SimpleQA", "", "", [("simpleqa_5shot", "", "")]),
    ("reading", "RACE", "", "", [("race", "", "")]),
    ("knowledge", "SimpleQA-Verified", "", "", [("simpleqa_verified_25shot", "", "")]),
    ("knowledge", "FACTS-parametric", "每题判 3 次", "judged 3x per item", [("facts_parametric_25shot", "", "")]),
    ("multilingual", "WMT24++", "英 → 55 种语言，报一个总分", "en → 55 languages, one overall score", [("wmt24pp_en_xx", "", "")]),
]


def _bkind(b):
    """kind of a BENCH entry (all its variants share it)."""
    ks = {"ll" if tn in NONGEN else "gen" for tn, _, _ in b[4]}
    assert len(ks) == 1, f"nemo6: mixed kinds in {b[1]}"
    return ks.pop()


# small categories shown together in the results (user 11-05: "多语言 和知识合并"; the non-generation part is one 11-column table):
# (kind, group) -> categories appended to its section
MERGE = {("ll", "reasoning"): [("ll", "reading"), ("ll", "aggregate")], ("gen", "knowledge"): [("gen", "multilingual")]}
MERGED_INTO = {k for v in MERGE.values() for k in v}
# the generation exams are too wide for one table (user 11-05): two tables, each with its own curve
SPLIT = {("gen", "aggregate"): [("MMLU 系 · AGIEval", "MMLU family · AGIEval", ["MMLU-Pro", "MMLU", "MMLU-Redux", "AGIEval-en"]),
                                ("GPQA · SuperGPQA · HLE · MGSM", "GPQA · SuperGPQA · HLE · MGSM", ["GPQA-Diamond", "SuperGPQA", "HLE", "MGSM"])]}
KCSS = ("<style>h3.nt-k{font-size:16.5px;margin:32px 0 6px;padding-top:14px;border-top:1px solid var(--axis)}"
        "h3.nt-k .nt-s{font-size:12.5px;font-weight:400;color:var(--text-secondary);margin-left:4px}h4.nt-cat{font-size:15px;margin:22px 0 4px}</style>")
# language lists of the multilingual tasks (their report aliases are per language)
LANGS = {"adlr_global_mmlu": "global_mmlu_", "adlr_mgsm": "mgsm_native_cot_", "adlr_include": "include_base_44_"}
MET = {"zh": {"exact_match": "EM", "acc": "acc", "acc_norm": "acc_norm", "chrf++": "chrF++", "judge_accuracy": "裁判准确率", "f1": "F1",
              "correct": "correct", "not_attempted": "not_attempted", "level_5_exact_match": "EM (Level 5)", "math500_exact_match": "EM (MATH-500 子集)"},
       "en": {"exact_match": "EM", "acc": "acc", "acc_norm": "acc_norm", "chrf++": "chrF++", "judge_accuracy": "judge accuracy", "f1": "F1",
              "correct": "correct", "not_attempted": "not_attempted", "level_5_exact_match": "EM (Level 5)", "math500_exact_match": "EM (MATH-500 subset)"}}
TXT = {
    "zh": {"h": "全量测评范围",
           "note": "下游 benchmark 全量评测，配置是 Boxiang 的 ADLR lm-eval 的 <code>nemo6_report.yml</code>（与他评测我们 production 模型时完全相同）。"
                   "协议：few-shot base prompt，不套 chat 模板；模型用 vLLM 起服务。下面是测哪些 benchmark；各模型的分数另起一章。",
           "tree_note": "先按是否生成分成两大部分（非生成类在前），每部分里再按配置的 task_group 分类；reading 和 aggregate 两个 task_group "
                        "两种都有，按任务拆到两边。一个 benchmark 可能有两种跑法（贪心一次 / 每题采样 32 次），各算一个任务。",
           "root": "nemo6", "stats": "{k} 大部分 · {c} 类 · {t} 任务 · {b} 个 benchmark · {m} 个指标",
           "cnt": "{t} 任务 · {m} 指标", "shot": "{n}-shot", "x_lang": "{m} × {n} 种语言",
           "chips": {"greedy": "greedy", "sampled": "采样 ×{r}", "judge": "LLM 裁判", "exec": "执行代码"},
           "legend": "生成类的跑法：greedy = 贪心解码一次；采样 ×32 = 每题采样 32 次，报 pass@1（平均）和 pass@32（至少对一次）；"
                     "执行代码 = 生成的代码跑测试用例；LLM 裁判 = gpt-oss-120b 判对错。"},
    "en": {"h": "Full evaluation scope",
           "note": "Downstream benchmarks, Boxiang's ADLR lm-eval config <code>nemo6_report.yml</code> (the same one he uses for our production models). "
                   "Protocol: few-shot base prompts, no chat template; models served with vLLM. Below: which benchmarks are run; the scores get their own chapter.",
           "tree_note": "Two parts by whether the model generates (non-generation first), then the config's task_group inside each part; the "
                        "reading and aggregate task_groups hold both kinds and are split by task. A benchmark may run twice (greedy once / 32 samples "
                        "per item); each run is one task.",
           "root": "nemo6", "stats": "{k} parts · {c} categories · {t} tasks · {b} benchmarks · {m} metrics",
           "cnt": "{t} tasks · {m} metrics", "shot": "{n}-shot", "x_lang": "{m} × {n} languages",
           "chips": {"greedy": "greedy", "sampled": "{r} samples", "judge": "LLM judge", "exec": "code exec"},
           "legend": "Generation modes: greedy = one greedy decode; 32 samples = 32 samples per item, pass@1 (mean) and pass@32 (any correct); "
                     "code exec = generated code runs the test cases; LLM judge = gpt-oss-120b grades."},
}
CSS = """<style>
.nt-root{display:flex;flex-wrap:wrap;align-items:baseline;gap:6px 14px}.nt-root b{font-size:16px}.nt-root .nt-s{color:var(--text-secondary)}
.nt-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,360px),1fr));gap:12px;margin-top:10px}
.nt-grid .card{margin:0}.nt-h{display:flex;flex-wrap:wrap;align-items:baseline;gap:4px 10px;margin-bottom:2px}.nt-h b{font-size:15px}
.nt-h .nt-s{color:var(--text-secondary);font-size:12px}.nt-how{color:var(--text-secondary);font-size:12px;margin-bottom:6px}
.nt-tree,.nt-tree ul{list-style:none;margin:0;padding:0}.nt-tree ul{margin-left:12px}
.nt-tree li{position:relative;padding:3px 0 3px 16px;line-height:1.45}
.nt-tree li::before{content:"";position:absolute;left:0;top:0;bottom:0;border-left:1px solid var(--axis)}
.nt-tree li:last-child::before{bottom:auto;height:14px}
.nt-tree li::after{content:"";position:absolute;left:0;top:14px;width:11px;border-top:1px solid var(--axis)}
.nt-b{font-weight:600}.nt-d,.nt-m{color:var(--text-secondary);font-size:12px}.nt-lang{display:block;color:var(--muted);font-size:11.5px}
.nt-c{display:inline-block;font-size:11px;line-height:17px;padding:0 7px;border-radius:999px;margin-left:4px;white-space:nowrap}
.nt-c.greedy{background:color-mix(in srgb,var(--s1) 14%,transparent);color:var(--s1)}
.nt-c.sampled{background:color-mix(in srgb,var(--s2) 14%,transparent);color:var(--s2)}
.nt-c.judge{background:color-mix(in srgb,var(--s4) 16%,transparent);color:var(--s4)}
.nt-c.exec{background:color-mix(in srgb,var(--s3) 14%,transparent);color:var(--s3)}
.nt-leg{color:var(--text-secondary);font-size:12px;margin:8px 0 0}
</style>"""


def _e(s):
    return html.escape(str(s), quote=False)


def _shots(cfg_task, pilot_rows):
    ns = {m.get("n_shot") for m in pilot_rows if isinstance(m.get("n_shot"), int)}
    if len(ns) == 1:
        return ns.pop()
    if isinstance(cfg_task.get("num_fewshot"), int):
        return cfg_task["num_fewshot"]
    m = re.search(r"(\d+)_?shot", cfg_task["task_name"])
    return int(m.group(1)) if m else None


def _mode(t):
    """generation task -> chips: greedy / sampled (+ judge / exec); none for non-generation (its part's header says it)."""
    reps = int((t.get("overrides") or {}).get("repeats", 1))
    mets = {rm["metric"].split(",")[0] for rm in t["report_metric"]}
    assert (t["task_name"] in NONGEN) == (mets <= {"acc", "acc_norm"}), f"nemo6: NONGEN out of sync for {t['task_name']}"
    if t["task_name"] in NONGEN:
        return []
    chips = []
    if t["task_group"] == "code":
        chips.append(("exec", None))
    if mets & {"judge_accuracy"}:
        chips.append(("judge", None))
    if reps > 1:
        chips.append(("sampled", reps))
    else:
        chips.append(("greedy", None))
    return chips


def _metrics(t, lang):
    tn = t["task_name"]
    names = []
    for rm in t["report_metric"]:
        n = rm["metric"].split(",")[0]
        n = re.sub(r"\[\d+\]$", "", n)                              # pass@1[32] -> pass@1
        n = MET[lang].get(n, n)
        if n not in names:
            names.append(n)
    if tn == "mmlu_str":
        return "EM（总分 / STEM）" if lang == "zh" else "EM (overall / STEM)", ""
    if tn in LANGS:
        langs = [rm["alias"].replace(LANGS[tn], "") for rm in t["report_metric"]]
        return TXT[lang]["x_lang"].format(m=" / ".join(names), n=len(langs)), " · ".join(langs)
    return " · ".join(names), ""


def nemo6_chapter(lang, esc=_e):
    """HTML of the nemo6 chapter (currently: the benchmark tree). Empty string if the config is missing."""
    if not os.path.exists(CFG):
        return ""
    tx = TXT[lang]
    cfg = yaml.safe_load(open(CFG))["tasks"]
    by = {t["task_name"]: t for t in cfg}
    listed = [tn for b in BENCH for tn, _, _ in b[4]]
    assert sorted(listed) == sorted(by), f"nemo6 tree out of sync with config: {set(by) ^ set(listed)}"
    pilot = {}
    if os.path.exists(PILOT):
        for ts in json.load(open(PILOT))["groups"].values():
            for x in ts:
                pilot[x["task"]] = x["metrics"]
    n_met = sum(len(t["report_metric"]) for t in cfg)

    def row(tn, label):
        t = by[tn]
        s = _shots(t, pilot.get(tn, []))
        m, langs = _metrics(t, lang)
        chips = "".join(f'<span class="nt-c {k}">{esc(tx["chips"][k].format(r=r))}</span>' for k, r in _mode(t))
        bits = [b for b in (label, tx["shot"].format(n=s) if s is not None else "") if b]
        return (f'<span class="nt-d">{esc(" · ".join(bits))}</span>{chips} <span class="nt-m">→ {esc(m)}</span>'
                + (f'<span class="nt-lang">{esc(langs)}</span>' if langs else ""))

    def card(kind, key, name, how):
        bs = [b for b in BENCH if b[0] == key and _bkind(b) == kind]
        tasks = [tn for b in bs for tn, _, _ in b[4]]
        mets = sum(len(by[tn]["report_metric"]) for tn in tasks)
        items = []
        for _, bname, dzh, den, variants in bs:
            d = dzh if lang == "zh" else den
            head = f'<span class="nt-b">{esc(bname)}</span>' + (f' <span class="nt-d">{esc(d)}</span>' if d else "")
            if len(variants) == 1:
                tn, vz, ve = variants[0]
                items.append(f"<li>{head} {row(tn, vz if lang == 'zh' else ve)}</li>")
            else:
                sub = "".join(f"<li>{row(tn, vz if lang == 'zh' else ve)}</li>" for tn, vz, ve in variants)
                items.append(f"<li>{head}<ul>{sub}</ul></li>")
        return (f'<div class="card"><div class="nt-h"><b>{esc(name)}</b>'
                f'<span class="nt-s">{esc(key)} · {esc(tx["cnt"].format(t=len(tasks), m=mets))}</span></div>'
                f'<div class="nt-how">{esc(how)}</div><ul class="nt-tree">{"".join(items)}</ul></div>')

    for t in cfg:
        _mode(t)                                               # asserts NONGEN == the acc / acc_norm-only tasks
    assert {(_bkind(b), b[0]) for b in BENCH} <= {c[:2] for c in CATS}, "nemo6: a benchmark has no (kind, task_group) category"
    zh = lang == "zh"
    parts = []
    for kind, kzh, ken, _, _, lzh, len_ in KINDS:                    # two big parts, non-generation first (user 11-05)
        ktasks = [tn for b in BENCH if _bkind(b) == kind for tn, _, _ in b[4]]
        kmets = sum(len(by[tn]["report_metric"]) for tn in ktasks)
        cards = [card(kind, key, czh if zh else cen, hzh if zh else hen) for k, key, czh, cen, hzh, hen in CATS if k == kind]
        parts.append(f'<h3 class="nt-k">{esc(kzh if zh else ken)} <span class="nt-s">{esc(tx["cnt"].format(t=len(ktasks), m=kmets))}</span></h3>'
                     f'<p class="note">{esc(lzh if zh else len_)}</p><div class="nt-grid">{"".join(cards)}</div>')
    stats = tx["stats"].format(k=len(KINDS), c=len(CATS), t=len(cfg), b=len(BENCH), m=n_met)
    return (f'{CSS}{KCSS}<h2>{esc(tx["h"])}</h2><p class="note">{tx["note"]}</p>'
            f'<p class="note">{esc(tx["tree_note"])}</p>'
            f'<div class="card nt-root"><b>{esc(tx["root"])}</b><span class="nt-s">{esc(stats)}</span><span class="nt-c greedy">greedy</span>'
            f'<span class="nt-c sampled">{esc(tx["chips"]["sampled"].format(r=32))}</span><span class="nt-c exec">{esc(tx["chips"]["exec"])}</span>'
            f'<span class="nt-c judge">{esc(tx["chips"]["judge"])}</span></div>'
            f'{"".join(parts)}<p class="nt-leg">{esc(tx["legend"])}</p>')


# ---------------------------------------------------------------------------------------------------------------------------------------
# Chapter "全量测评结果" (user 11-05: one doc-isolated-style table per category): rows = reference models (REF_MODELS order / names /
# sizes of gen_eval_html.py), columns = the category's report metrics (x100). Multilingual tasks: language mean in the main table,
# per-language tables folded below. Cells whose task had >= EMPTY_FLAG empty answers (collect_nemo6.py empty_frac) are greyed + ‡.
RESULTS = os.path.join(HERE, "..", "results_xx", "nemo6")
EMPTY_FLAG = 0.2
STXT = {
    "zh": {"h": "全量测评结果", "note": "分数 ×100（准确率 / pass@k 为 %，WMT24++ 为 chrF++）。点表头排序（第一次从高到低）；<b>粗体</b> = 该列最高；"
                                         "· = 还没跑出来（job 在跑，跑完后更新）。模型名、大小与上面 Pretrain doc-isolated 表一致。",
           "st_h": "进度", "st_head": ["模型", "大小", "已出分指标", "状态"], "done": "完成", "run": "在跑", "post": "后训练",
           "mean": "平均（{n} 种语言）", "langs": "各语言明细（{b}）",
           "flag": "‡ 灰色 = 这个模型在这一项有大量空回答（悬停看比例）：它在 few-shot 的 “A:” 后先输出换行，而这一项遇到换行就停止生成，"
                   "回答被截成空字符串，分数不代表模型能力。只出现在部分后训练模型上（如 Qwen3.8-27B；MiMo-V2.6-Pro-RL 没有这个问题）。",
           "empty": "空回答 {p:.0f}%", "ylab": ["最低", "最高", "每列归一化"], "out": "范围外", "cn": True,
           "fam": {"Qwen3.8": ("蓝色系", "Qwen3.8", "由浅到深，按模型大小"), "Qwen3": ("橙色系", "Qwen3", "由浅到深，按模型大小"),
                   "MiMo": ("青绿色系", "MiMo", "由浅到深"), "DeepSeek": ("黄色", "DeepSeek", ""), "Kimi": ("品红", "Kimi", ""),
                   "Mistral": ("紫色", "Mistral", "")},
           "norm": "曲线：每一列在各模型之间做 min–max 归一化（该列最低 = 0，最高 = 1），只看相对位置，纵轴没有绝对意义。0 分、‡ 空回答、以及明显偏低的分（同时低于该列 Q1 − 1.5×IQR 和中位数的一半，至少 4 个模型有分时才判）不参与归一化，画在图下方的“范围外”一栏（空心点，线照样连过去）；只有一个模型有分的列不画。颜色按模型家族，同一家族用同一色系的深浅（Qwen3 和 Qwen3.8 各一个色系，按模型大小由浅到深）。鼠标放在图上显示原始分（与表格相同）。"},
    "en": {"h": "Full evaluation results", "note": "Scores x100 (accuracy / pass@k in %, chrF++ for WMT24++). Click a header to sort (first click: high to low); "
                                                   "<b>bold</b> = best in the column; · = not out yet (job running, updated when done). Model names / sizes as in the Pretrain doc-isolated table above.",
           "st_h": "Progress", "st_head": ["model", "size", "metrics scored", "status"], "done": "done", "run": "running", "post": "post-trained",
           "mean": "mean ({n} languages)", "langs": "Per language ({b})",
           "flag": "‡ grey = many empty answers from this model on this task (hover for the share): it starts with a newline after the few-shot \"A:\", "
                   "the task stops generation at a newline, so the answer is cut to an empty string; the score does not reflect the model. "
                   "Only some post-trained models (e.g. Qwen3.8-27B; MiMo-V2.6-Pro-RL is fine).",
           "empty": "{p:.0f}% empty answers", "ylab": ["min", "max", "per-column normalised"], "out": "out of range", "cn": False,
           "fam": {"Qwen3.8": ("Blues", "Qwen3.8", "light to dark by model size"), "Qwen3": ("Oranges", "Qwen3", "light to dark by model size"),
                   "MiMo": ("Aquas", "MiMo", "light to dark"), "DeepSeek": ("Yellow", "DeepSeek", ""), "Kimi": ("Magenta", "Kimi", ""),
                   "Mistral": ("Violet", "Mistral", "")},
           "norm": "Curves: each column min-max normalised across the models (column min = 0, max = 1), relative positions only, the y axis has no absolute meaning. 0, ‡ (empty answers) and significantly low scores (below both Q1 − 1.5×IQR and half the median of the column, judged when >= 4 models are scored) are left out of the normalisation and drawn in the \"out of range\" strip below (hollow points, still joined by the line); columns with a single scored model are not drawn. Colour = model family, models of a family = shades of one hue (Qwen3 and Qwen3.8 each their own, light to dark by size). Hover: raw scores (as in the table)."},
}


def _load_results():
    out = {}
    for f in sorted(glob_json(RESULTS)):
        try:
            d = json.load(open(f))
            out[d["name"]] = d
        except Exception:
            pass
    return out


def glob_json(d):
    return [os.path.join(d, x) for x in os.listdir(d) if x.endswith(".json")] if os.path.isdir(d) else []


def _cell_lookup(d):
    """(task, alias, metric) -> (value, task empty_frac) for one model's summary JSON."""
    out, ef = {}, {}
    for ts in d["groups"].values():
        for t in ts:
            ef[t["task"]] = t.get("empty_frac")
            for m in t["metrics"]:
                out[(t["task"], m["alias"], m["metric"])] = m["value"]
    return out, ef


def _mshort(metric, lang):
    n = re.sub(r"\[\d+\]$", "", metric.split(",")[0])
    return MET[lang].get(n, n)


# curve above each table (user 11-05): x = the table's columns, y = per-column min-max normalisation across the models ("每个指标的正则",
# no requirement on the y axis). Colour (second pass, user): one hue per model FAMILY, the models of a family = steps of that hue (Qwen3 +
# Qwen3.8 = one blue family, 5 steps light -> dark by size; MiMo = 2 orange steps); solid lines only; legend grouped by family.
# Family hues = the report's categorical slots in their fixed order (blue, orange, aqua, yellow, magenta: adjacent pairs validated in both
# modes); the family steps validated as ordinal ramps (monotone L, step gaps, light end >= 2:1) in both modes. Identity is also carried by
# the grouped legend, the hover list and the table. Out of range (hollow points in a strip under the plot; not in the min / max, not joined
# by the line): 0, ‡ (empty answers) and significantly low scores (< Q1 - 1.5 IQR AND < half the column median, >= 4 models scored).
# The line still runs to an out-of-range point (user 11-05); it breaks only where a model has no score yet.
FAMILIES = [("Qwen3.8", ["QWEN38_27B", "QWEN38_FLASH_NEXT", "QWEN38_MAX"]), ("Qwen3", ["QWEN3_30B_A3B_BASE", "QWEN3_235B_A22B"]),
            ("MiMo", ["MIMO_V25_PRO_BASE", "MIMO_V26_PRO_RL"]), ("DeepSeek", ["DEEPSEEK_V4_FLASH_BASE"]),
            ("Kimi", ["KIMI_K3"]), ("Mistral", ["MISTRAL_LARGE4"])]
# 11-05 (user "qwen3.8还是和qwen3区分下色系"): Qwen3.8 = blue steps (3, by size), Qwen3 = orange steps (2, by size), MiMo = aqua steps (2);
# DeepSeek yellow (--c4), Kimi magenta (--c5), Mistral violet. Steps validated --ordinal in both modes. The 6th hue: every candidate
# sits < 15 normal-vision dE from one other colour (green 8.6 vs MiMo's dark aqua, red 7.1 vs Qwen3 orange, violet 10.5 / 9.8 vs the
# Qwen3.8 mid blue) -> violet, the largest worst case; the grouped legend, the hover names and the table carry identity as well.
COLOR = {**{c: f"--nq{i + 1}" for i, c in enumerate(FAMILIES[0][1])}, **{c: f"--no{i + 1}" for i, c in enumerate(FAMILIES[1][1])},
         **{c: f"--nm{i + 1}" for i, c in enumerate(FAMILIES[2][1])},
         "DEEPSEEK_V4_FLASH_BASE": "--c4", "KIMI_K3": "--c5", "MISTRAL_LARGE4": "--nv"}
SHADES = (".viz-root{--nq1:#86b6ef;--nq2:#2a78d6;--nq3:#104281;--no1:#eb6834;--no2:#b4471d;--nm1:#1baf7a;--nm2:#0f7a53;--nv:#6250d6}"
          '@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{--nq1:#9ec5f4;--nq2:#3987e5;--nq3:#184f95;'
          "--no1:#f0a47f;--no2:#d95926;--nm1:#7fcfa9;--nm2:#199e70;--nv:#9085e9}}")
SHORT = {"LiveCodeBench": "LCB", "BigCodeBench": "BCB", "GPQA-Diamond": "GPQA-D", "Natural Questions": "NQ", "SimpleQA-Verified": "SimpleQA-V",
         "FACTS-parametric": "FACTS", "CommonsenseQA": "CSQA", "ARC-Challenge": "ARC-C", "HMMT Feb 2025": "HMMT 25", "AIME 2024": "AIME 24",
         "AIME 2025": "AIME 25", "Minerva MATH": "Minerva", "Global-MMLU": "G-MMLU", "Social IQa": "SIQA", "OpenBookQA": "OBQA",
         "WinoGrande": "WinoG", "Natural Questions EM": "NQ"}


def _smooth(pts):
    """SVG path through pts [(x, y)] (x increasing) as a monotone cubic (Fritsch-Carlson, = d3.curveMonotoneX; user 11-05: 曲线 instead
    of a polyline): passes through every point and never overshoots between two neighbours (no invented peaks / dips)."""
    n = len(pts)
    if n < 3:
        return " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(pts))
    dx = [pts[i + 1][0] - pts[i][0] for i in range(n - 1)]
    m = [(pts[i + 1][1] - pts[i][1]) / dx[i] if dx[i] else 0.0 for i in range(n - 1)]
    t = [m[0]] + [0.0 if m[i - 1] * m[i] <= 0 else (m[i - 1] + m[i]) / 2 for i in range(1, n - 1)] + [m[-1]]
    for k in range(n - 1):
        if m[k] == 0:
            t[k] = t[k + 1] = 0.0
            continue
        a, b = t[k] / m[k], t[k + 1] / m[k]
        if a * a + b * b > 9:
            tau = 3 / (a * a + b * b) ** 0.5
            t[k], t[k + 1] = tau * a * m[k], tau * b * m[k]
    d = [f"M{pts[0][0]:.1f},{pts[0][1]:.1f}"]
    for k in range(n - 1):
        (x0, y0), (x1, y1), h = pts[k], pts[k + 1], dx[k] / 3
        d.append(f"C{x0 + h:.1f},{y0 + t[k] * h:.1f} {x1 - h:.1f},{y1 - t[k + 1] * h:.1f} {x1:.1f},{y1:.1f}")
    return " ".join(d)


def _curve(ccols, models, esc, tx, W=1000, H=400):
    """ccols = [(line1, line2, getter)]; models = [(code, display name)]. Grouped legend + inline SVG (viewBox W x H, width 100%)."""
    import statistics
    n = len(ccols)
    rot = n > 9                                            # dense x axis: one-line labels at -30 deg (no overlap), else two lines
    ml, mr, mt, mb, band = 58, 14, 12, (92 if rot else 46), 24
    pts = {c: [g(c) for _, _, g in ccols] for c, _ in models}
    norm = {c: [None] * n for c, _ in models}
    out = {c: [False] * n for c, _ in models}
    for j in range(n):
        vals = [(c, pts[c][j][0], pts[c][j][1]) for c, _ in models if pts[c][j][0] is not None]
        good = [v for _, v, f in vals if not f and v > 0]
        fence = None
        if len(good) >= 4:                                 # "significantly low" = below the Tukey fence AND below half the median
            q1, med, q3 = statistics.quantiles(good, n=4, method="inclusive")
            fence = min(q1 - 1.5 * (q3 - q1), 0.5 * med)
        ins = []
        for c, v, f in vals:
            if f or v <= 0 or (fence is not None and v < fence):
                out[c][j] = True
            else:
                ins.append((c, v))
        if len(ins) >= 2:
            lo, hi = min(v for _, v in ins), max(v for _, v in ins)
            for c, v in ins:
                norm[c][j] = (v - lo) / (hi - lo) if hi > lo else 0.5
    shown = [c for c, _ in models if any(x is not None for x in norm[c]) or any(out[c])]
    if not shown:
        return ""
    ybot = H - mb - band                                   # bottom of the normalised area; the strip below it = out of range
    X = lambda j: ml + (j + 0.5) * (W - ml - mr) / n
    Y = lambda y: mt + 8 + (1 - y) * (ybot - mt - 16)
    yo = ybot + band / 2 + 2
    o = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img">']
    for k in range(5):                                     # recessive grid, no numbers: the y axis is relative only
        o.append(f'<line x1="{ml}" x2="{W - mr}" y1="{Y(k / 4):.1f}" y2="{Y(k / 4):.1f}" stroke="var(--grid)" stroke-width="1"/>')
    o.append(f'<text x="{ml - 6}" y="{Y(1) + 4:.1f}" text-anchor="end">{esc(tx["ylab"][1])}</text>'
             f'<text x="{ml - 6}" y="{Y(0) + 4:.1f}" text-anchor="end">{esc(tx["ylab"][0])}</text>'
             f'<text x="12" y="{Y(0.5):.0f}" transform="rotate(-90 12 {Y(0.5):.0f})" text-anchor="middle">{esc(tx["ylab"][2])}</text>')
    if True:                                               # out-of-range strip: on every chart, even with no point in it (user 11-05)
        o.append(f'<rect x="{ml}" y="{ybot + 3:.1f}" width="{W - ml - mr}" height="{band - 3}" fill="var(--grid)" opacity="0.35"/>'
                 f'<text x="{ml - 6}" y="{yo + 4:.1f}" text-anchor="end" style="fill:var(--muted)">{esc(tx["out"])}</text>')
    o.append(f'<line x1="{ml}" x2="{W - mr}" y1="{H - mb}" y2="{H - mb}" stroke="var(--axis)"/>')
    for j, (l1, l2, _) in enumerate(ccols):               # x labels
        if rot:
            x0, y0 = X(j) + 4, H - mb + 14
            o.append(f'<text x="{x0:.1f}" y="{y0}" text-anchor="end" transform="rotate(-30 {x0:.1f} {y0})" style="font-size:11px">'
                     f'<tspan class="lbl" style="font-size:11px">{esc(l1)}</tspan> <tspan style="fill:var(--muted)">{esc(l2)}</tspan></text>')
        else:
            o.append(f'<text x="{X(j):.1f}" y="{H - mb + 16}" text-anchor="middle" class="lbl" style="font-size:11px">{esc(l1)}</text>'
                     f'<text x="{X(j):.1f}" y="{H - mb + 30}" text-anchor="middle" style="fill:var(--muted);font-size:10.5px">{esc(l2)}</text>')
    pos = {c: [None] * n for c in shown}                  # (x, y, in_range): out-of-range points sit in the strip, side by side
    for j in range(n):
        oc = [c for c in shown if out[c][j]]
        for c in shown:
            if norm[c][j] is not None:
                pos[c][j] = (X(j), Y(norm[c][j]), True)
            elif out[c][j]:
                i = oc.index(c)
                pos[c][j] = (X(j) + (i - (len(oc) - 1) / 2) * 9, yo, False)
    for c in shown:                                        # solid lines; they also run down to out-of-range points (user 11-05),
        col = COLOR.get(c, "--text-secondary")             # and break only where a model has no score yet
        seg = []
        for pt in pos[c] + [None]:
            if pt is not None:
                seg.append(pt[:2])
                continue
            if len(seg) > 1:
                o.append(f'<path d="{_smooth(seg)}" fill="none" stroke="var({col})" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
            seg = []
    for c in shown:                                        # markers: filled in range, hollow out of range
        col = COLOR.get(c, "--text-secondary")
        for pt in pos[c]:
            if pt is None:
                continue
            x, y, inr = pt
            o.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{4 if inr else 3.5}" fill="var({"--" + col[2:] if inr else "--surface-1"})" '
                     f'stroke="var({"--surface-1" if inr else col})" stroke-width="2"/>')
    colw = (W - ml - mr) / n
    names = dict(models)
    for j, (l1, l2, _) in enumerate(ccols):               # hover band per column: raw scores (as in the table), highest first
        rows = [(c, pts[c][j]) for c, _ in models if pts[c][j][0] is not None]
        if not rows:
            continue
        def val(c, v, f):
            return f"{v * 100:.1f}" + ("‡" if f else "")   # raw score only, as in the table (user 11-05: "hover只是放raw score")
        tip = f"<b>{esc(l1)} · {esc(l2)}</b>" + "".join(
            f'<br><span style="color:var({COLOR.get(c, "--text-secondary")})">●</span> {esc(names[c])}: {esc(val(c, v, f))}'
            for c, (v, f) in sorted(rows, key=lambda r: r[1][0], reverse=True))
        o.append(f'<rect class="hit" x="{X(j) - colw / 2:.1f}" y="{mt}" width="{colw:.1f}" height="{H - mt - mb}" data-tip="{esc(tip)}"/>')
    o.append("</svg>")
    # legend in the KD-teacher section's form (user 11-05): one line per colour family, swatches light -> dark, then
    # "<colour>：<family> — member / member (由浅到深)"; only the models drawn in this chart
    lines = []
    for fam, codes in FAMILIES + [("·", [c for c in shown if c not in COLOR])]:
        items = [c for c in codes if c in shown]
        if not items:
            continue
        cw, desc, order = tx["fam"].get(fam, ("", fam, ""))
        sw = "".join(f'<i style="display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:2px;'
                     f'background:var({COLOR.get(c, "--text-secondary")})"></i>' for c in items)
        sep = "：" if tx is not None and tx.get("cn") else ": "
        if len(items) == 1 and len(codes) == 1:
            txt = f"{cw}{sep}{names[items[0]]}"
        else:
            txt = f"{cw}{sep}{desc} — " + " / ".join(names[c] for c in items)
            if order and len(items) > 1:
                txt += f"（{order}）" if tx.get("cn") else f" ({order})"
        lines.append(f"<div>{sw} {esc(txt)}</div>")
    return f'<div class="kd25fam" style="font-size:12.5px;line-height:1.7;margin:2px 0 6px">{"".join(lines)}</div>{"".join(o)}'


def nemo6_scores(lang, esc, ref_models, ref_size):
    if not os.path.exists(CFG):
        return ""
    tx, tt = STXT[lang], TXT[lang]
    cfg = yaml.safe_load(open(CFG))["tasks"]
    by = {t["task_name"]: t for t in cfg}
    res = _load_results()
    models = [(c, n) for c, n, _ in ref_models if c in res]
    if not models:
        return ""
    look = {c: _cell_lookup(res[c]) for c, _ in models}
    n_all = sum(len(t["report_metric"]) for t in cfg)

    def fmt(v):
        return f"{v * 100:.1f}"

    def name_cell(c, n):
        tag = f' <span class="nt-d">{esc(tx["post"])}</span>' if res[c].get("post_trained") else ""
        return f'<td class="l"><b>{esc(n)}</b>{tag}</td><td class="l nt-d">{esc(ref_size.get(c, ""))}</td>'

    def table(cols, rows_extra=""):
        """cols: [(header html, getter(code) -> (value|None, flag_text|None))]"""
        vals = {c: [g(c) for _, g, *_ in cols] for c, _ in models}
        best = []
        for j in range(len(cols)):
            vs = [vals[c][j][0] for c, _ in models if vals[c][j][0] is not None and not vals[c][j][1]]
            best.append(max(vs) if len(vs) > 1 and "not_attempted" not in cols[j][0] else None)   # no "best" for a refusal rate
        head = ('<tr><th class="l">' + esc(tx["st_head"][0]) + '</th><th class="l">' + esc(tx["st_head"][1]) + "</th>"
                + "".join(f"<th>{h}</th>" for h, *_ in cols) + "</tr>")
        body = []
        for c, n in models:
            tds = []
            for j, (v, flag) in enumerate(vals[c]):
                if v is None:
                    tds.append('<td data-v="">·</td>')
                elif flag:
                    tds.append(f'<td class="nt-flag" data-v="{v}" title="{esc(flag)}">{fmt(v)}‡</td>')
                else:
                    s = fmt(v)
                    tds.append(f'<td data-v="{v}">{"<b>" + s + "</b>" if best[j] is not None and v == best[j] else s}</td>')
            body.append(f'<tr data-s="1">{name_cell(c, n)}{"".join(tds)}</tr>')
        flagged = any(f for c, _ in models for _, f in vals[c])
        return (f'<table class="rsort nt-tab" data-hi="1"><thead>{head}</thead><tbody>{"".join(body)}</tbody></table>', flagged)

    def getter(task, alias, metric):
        def g(c):
            cells, ef = look[c]
            v = cells.get((task, alias, metric))
            e = ef.get(task)
            flag = tx["empty"].format(p=e * 100) if isinstance(e, (int, float)) and e >= EMPTY_FLAG and v is not None else None
            return (v if isinstance(v, (int, float)) else None), flag
        return g

    def mean_getter(task):
        rms = by[task]["report_metric"]

        def g(c):
            cells, _ = look[c]
            vs = [cells.get((task, rm["alias"], rm["metric"])) for rm in rms]
            return (sum(vs) / len(vs) if vs and all(isinstance(v, (int, float)) for v in vs) else None), None
        return g

    out = [f'<h2>{esc(tx["h"])}</h2><p class="note">{tx["note"]}</p><p class="note">{esc(tx["norm"])}</p>']   # no progress table (user 11-05: "进度 这个不要")
    zh_ = lang == "zh"
    for kind, kzh, ken, szh, sen, _, _ in KINDS:          # two big parts, non-generation first (user 11-05)
        ktasks = [tn for b in BENCH if _bkind(b) == kind for tn, _, _ in b[4]]
        kmets = sum(len(by[tn]["report_metric"]) for tn in ktasks)
        out.append(f'<h3 class="nt-k">{esc(kzh if zh_ else ken)} <span class="nt-s">{esc(szh if zh_ else sen)} · '
                   f'{esc(tt["cnt"].format(t=len(ktasks), m=kmets))}</span></h3>')
        out.extend(_kind_sections(kind, lang, esc, tx, by, table, getter, mean_getter, models))
    css = ("<style>.nt-tab th{white-space:normal;vertical-align:bottom;min-width:64px}.nt-tab td.nt-flag{color:var(--muted);font-style:italic}"
           ".nt-fn{margin:8px 0 0}.nt-fold{margin-top:8px}.nt-fold summary{cursor:pointer;color:var(--text-secondary);font-size:13px}"
           ".nt-lg{flex-wrap:wrap;gap:4px 22px;margin:2px 0 4px}.nt-lg i{width:16px}.nt-lgg{display:inline-flex;flex-wrap:wrap;gap:2px 10px;align-items:center}"
           ".nt-lgg b{color:var(--text-primary);font-weight:600}.nt-tab{margin-top:10px}" + SHADES + "</style>")
    return css + KCSS + "".join(out)


def _kind_sections(kind, lang, esc, tx, by, table, getter, mean_getter, models):
    """the category sections (h4, curve + table per card) of one big part."""
    out = []
    zh_ = lang == "zh"
    for _, key, zh, en, how_zh, how_en in [c for c in CATS if c[0] == kind]:
        cat = (kind, key)
        if cat in MERGED_INTO:                             # shown inside another category's section
            continue
        cats = [cat] + MERGE.get(cat, [])
        benches = [b for k in cats for b in BENCH if (_bkind(b), b[0]) == k]
        label = key
        if MERGE.get(cat):
            allc = [next(c for c in CATS if c[:2] == k) for k in cats]
            zh, en = " · ".join(c[2] for c in allc), " · ".join(c[3] for c in allc)
            how_zh, how_en = "；".join(f"{c[2]}：{c[4]}" for c in allc), "; ".join(f"{c[3]}: {c[5]}" for c in allc)
            label = " · ".join(k[1] for k in cats)
        parts = SPLIT.get(cat) or [(None, None, [b[1] for b in benches])]
        sec = (f'<h4 class="nt-cat">{esc(zh if zh_ else en)} <span class="nt-d">{esc(label)}</span></h4>'
               f'<p class="note">{esc(how_zh if zh_ else how_en)}</p>')
        for ptz, pte, pnames in parts:
            cols, folds = [], []
            for _, bname, _, _, variants in [b for b in benches if b[1] in pnames]:
                for tn, vz, ve in variants:
                    t = by[tn]
                    v = vz if lang == "zh" else ve
                    top = esc(bname) + (f' <span class="nt-d">{esc(v)}</span>' if v else "")
                    l1 = SHORT.get(bname, bname) + (f" {v}" if v else "")          # chart x label, line 1
                    if tn in LANGS:
                        m = _mshort(t["report_metric"][0]["metric"], lang)
                        cols.append((f'{top}<br><span class="nt-d">{esc(m)} · {esc(tx["mean"].format(n=len(t["report_metric"])))}</span>', mean_getter(tn),
                                     l1, f'{m} {"平均" if lang == "zh" else "mean"}', True))
                        lcols = [(esc(rm["alias"].replace(LANGS[tn], "")), getter(tn, rm["alias"], rm["metric"])) for rm in t["report_metric"]]
                        folds.append((bname, table(lcols)[0]))
                        continue
                    for rm in t["report_metric"]:
                        sub = _mshort(rm["metric"], lang)
                        if tn == "mmlu_str":
                            sub = ("EM 总分" if lang == "zh" else "EM overall") if rm["alias"] == "mmlu_str" else "EM STEM"
                        cols.append((f'{top}<br><span class="nt-d">{esc(sub)}</span>', getter(tn, rm["alias"], rm["metric"]),
                                     l1, sub, "not_attempted" not in rm["metric"]))      # a refusal rate is not a score: table only
            tbl, flagged = table(cols)
            chart = _curve([(l1, l2, g) for _, g, l1, l2, ok in cols if ok], models, esc, tx)
            title = f'<div class="ct">{esc(ptz if lang == "zh" else pte)}</div>' if ptz else ""
            sec += (f'<div class="card">{title}{chart}{tbl}' + (f'<p class="note nt-fn">{esc(tx["flag"])}</p>' if flagged else ""))
            for bname, ft in folds:
                sec += f'<details class="nt-fold"><summary>{esc(tx["langs"].format(b=bname))}</summary>{ft}</details>'
            sec += "</div>"
        out.append(sec)
    return out
