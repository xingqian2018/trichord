"""reports_xx/cosmos_eval_section.py - the full-evaluation chapters of the eval report, on the protocol of the unified eval harness
(user 11-13: "新的 baseline 全量测评结果在这里，以这个版本为准"; same day: "新的是66个指标，不再是nemov6的75个，根据这个彻底改写 外部参考，和
downstream eval 两个tab页面"): "全量测评范围" (benchmark tree, key tasks marked), "全量测评重点" (the key tier: 19 tasks, one headline metric
each, three tabs) and "全量测评结果" (all 66 report metrics). gen_eval_html.py puts them in the 外部参考 tab (reference models) and, with
per-model colours and the show / hide boxes of "实验列表", in the downstream eval tab (our checkpoints). The production run's checkpoints
(JSONs with "track") are rows there like the other checkpoints; above "实验列表" ce_steps() draws
the key tier against the training step, charts only (user 11-13 evening: "实验列表上面还是把一个 blocked chart，是重点指标，根据 x-axis 训练时间
变化标注出来"; the earlier chapter with notes and tables was deleted that afternoon on "这个删除"). The nemo6 chapters (75 metrics) are no longer rendered; nemo6_section.py only supplies the shared curve, colours and CSS.

Scores: results_xx/cosmos_eval/<MODEL>.json, written by reports_xx/cosmos_eval/collect_cosmos_eval.py from runs/cosmos_eval_downstream (read-only).
Task list: reports_xx/cosmos_eval/cosmos_ultra_ladder.yml = the harness's nvbench/cosmos_ultra_ladder.yml (36 tasks, 66 report metrics). Every
task of the list must appear exactly once in BENCH, and every benchmark in exactly one results tab (asserted), so the chapters cannot drift
from what was evaluated. The curve and the table markup are the ones the nemo6 chapters used (same JS, same md mirror).
"""
import json
import math
import os
import re

import yaml

from nemo6_section import CSS, KCSS, SHADES, SHORT as _NSHORT, STXT as _NSTXT, _curve, _e

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = os.path.join(HERE, "cosmos_eval", "cosmos_ultra_ladder.yml")
RESULTS = os.path.join(HERE, "..", "results_xx", "cosmos_eval")
PILOT = "QWEN3_30B_A3B_BASE"                               # n-shot of the tree = what lm_eval recorded for this model
EMPTY_FLAG = 0.2      # a task with >= 20 % empty answers: its scores are withdrawn ("-", never 0; user 11-06), as in the nemo6 chapters and as
EMPTY_NOTE = 0.05     # the harness's own gen_health flag "INVALID: empty answers"; 5-20 %: the score stays, marked, share on hover
# non-generation = letter / option likelihood (asserted below: exactly the tasks whose report metrics are acc / acc_norm / acc_ties_shared)
NONGEN = {"commonsense_qa", "arc_challenge_llama", "hellaswag", "openbookqa", "piqa", "winogrande", "social_iqa", "truthfulqa_mc2",
          "race_high_all", "mmlu", "mmlu_redux", "global_mmlu_zh", "include_base_44_chinese"}
RATES = {"no_answer", "not_attempted"}                     # a rate, not a score: table only, never withdrawn
NO_CURVE = RATES | {"acc_ties_shared"}                     # table only (acc_ties_shared repeats acc within a point)
ZERO_SHOT = {"openbookqa", "piqa", "social_iqa", "race_high_all"}   # lm_eval records no n-shot: the question alone (task yamls)
KINDS = [
    ("ll", "非生成类 benchmark", "Non-generation benchmarks",
     "多选题：比较模型给各选项（或选项字母）的对数概率，概率最高的即答案（acc_norm 按长度归一化）。不生成文本，分数不受回答格式和停止符影响。",
     "Multiple choice: the log-likelihoods of the options (or option letters) are compared and the most likely one is the answer "
     "(acc_norm: length-normalised). Nothing is generated, so answer format and stop strings do not matter."),
    ("gen", "生成类 benchmark", "Generation benchmarks",
     "模型生成回答（贪心一次，或每题采样 32 次），再抽取答案比对、执行代码，或由 LLM 裁判判对错。",
     "The model generates an answer (one greedy decode, or 32 samples per item), which is then matched, executed or graded by an LLM judge."),
]
# (kind, task_group of the config) -> zh / en name, how it is scored; display order inside each kind
CATS = [
    ("ll", "reasoning", "常识推理", "Commonsense reasoning", "英文多选题", "English multiple choice"),
    ("ll", "reading", "阅读理解", "Reading comprehension", "读一篇文章，答多选题", "multiple choice on a passage"),
    ("ll", "aggregate", "知识考试", "Knowledge exams", "知识考试多选题，按选项字母的概率打分", "knowledge-exam multiple choice, scored by letter likelihood"),
    ("gen", "code", "代码", "Code", "生成代码并执行打分", "generated code is executed"),
    ("gen", "aggregate", "综合考试", "Exams", "生成作答（CoT），抽取最终选项 / 答案比对", "generated answer (CoT), final choice / answer extracted"),
    ("gen", "math", "数学", "Math", "生成解答，比对最终答案", "generated solution, final answer checked"),
    ("gen", "reading", "事实问答", "Factual QA", "短答案生成，取回答的第一行比对", "short generated answers, the first line is matched"),
    ("gen", "knowledge", "知识", "Knowledge", "LLM 裁判判对错", "graded by an LLM judge"),
]
# benchmark: (task_group, display name, detail zh, detail en, [(task, variant zh, variant en)])
BENCH = [
    ("code", "MBPP", "sanitized", "sanitized", [("mbpp_sanitized_3shot_sampled", "", "")]),
    ("code", "HumanEval", "", "", [("humaneval_sampled", "", "")]),
    ("code", "LiveCodeBench", "release v6", "release v6", [("lcb_codegen", "greedy", "greedy"), ("lcb_codegen_sampled", "采样", "sampled")]),
    ("aggregate", "MMLU-Pro", "官方 CoT 提示", "official CoT prompt", [("mmlu_pro_5_shot_base", "", "")]),
    ("aggregate", "AGIEval-en", "CoT，8 个子集", "CoT, 8 sets", [("agieval_en_cot", "", "")]),
    ("aggregate", "GPQA-Diamond", "CoT", "CoT", [("gpqa_diamond_cot_n_shot", "greedy", "greedy"), ("gpqa_diamond_cot_n_shot_sampled", "采样", "sampled")]),
    ("aggregate", "SuperGPQA", "CoT", "CoT", [("supergpqa_main_cot_n_shot", "", "")]),
    ("aggregate", "MGSM-zh", "native CoT，中文", "native CoT, Chinese", [("mgsm_native_cot_zh", "", "")]),
    ("aggregate", "MMLU", "字母似然", "letter likelihood", [("mmlu", "", "")]),
    ("aggregate", "MMLU-Redux", "字母似然，5,330 道核对无误的题", "letter likelihood, the 5,330 questions found correct", [("mmlu_redux", "", "")]),
    ("aggregate", "Global-MMLU-zh", "中文", "Chinese", [("global_mmlu_zh", "", "")]),
    ("aggregate", "INCLUDE-zh", "base-44，中文", "base-44, Chinese", [("include_base_44_chinese", "", "")]),
    ("math", "GSM8K", "CoT", "CoT", [("gsm8k_cot_nemo", "NeMo 提示", "NeMo prompt"), ("gsm8k_cot", "Wei 提示", "Wei prompt")]),
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
    ("reading", "RACE-high", "全部 3,498 题", "all 3,498 questions", [("race_high_all", "", "")]),
    ("reading", "TriviaQA", "", "", [("trivia_qa", "", "")]),
    ("reading", "PopQA", "官方协议", "official protocol", [("popqa_15shot", "", "")]),
    ("reading", "Natural Questions", "", "", [("naturalqs_5shot", "", "")]),
    ("reading", "SimpleQA", "", "", [("simpleqa_5shot", "", "")]),
    ("knowledge", "SimpleQA-Verified", "", "", [("simpleqa_verified_25shot", "", "")]),
    ("knowledge", "FACTS-parametric", "", "", [("facts_parametric_25shot", "", "")]),
]
# tabs of the results chapter, non-generation first: (zh, en, [(part title zh, en, [benchmark names])]); one curve + one table per part
SECTIONS = [
    ("常识推理 · 阅读理解", "Commonsense reasoning · Reading comprehension",
     [(None, None, ["CommonsenseQA", "ARC-Challenge", "HellaSwag", "OpenBookQA", "PIQA", "WinoGrande", "Social IQa", "TruthfulQA", "RACE-high"])]),
    ("知识考试", "Knowledge exams", [(None, None, ["MMLU", "MMLU-Redux", "Global-MMLU-zh", "INCLUDE-zh"])]),
    ("代码", "Code", [(None, None, ["MBPP", "HumanEval", "LiveCodeBench"])]),
    ("综合考试", "Exams", [("MMLU-Pro · AGIEval", "MMLU-Pro · AGIEval", ["MMLU-Pro", "AGIEval-en"]),
                       ("GPQA · SuperGPQA · MGSM", "GPQA · SuperGPQA · MGSM", ["GPQA-Diamond", "SuperGPQA", "MGSM-zh"])]),
    ("数学", "Math", [(None, None, ["GSM8K", "Minerva MATH", "MATH-500", "AIME 2024", "AIME 2025", "HMMT Feb 2025"])]),
    ("事实问答 · 知识", "Factual QA · Knowledge",
     [(None, None, ["TriviaQA", "PopQA", "Natural Questions", "SimpleQA", "SimpleQA-Verified", "FACTS-parametric"])]),
]
# the key tier "全量测评重点" (decided 11-13 in session 统一测试): 19 of the 36 tasks, one headline metric each, in three tabs
# (知识 / 推理 8, 数学 8, 代码 3); the other 17 tasks and the other 47 metrics appear in "全量测评结果" only: (key, zh, en, [(alias, metric)])
FOCUS_TABS = [
    ("kr", "知识 / 推理", "Knowledge / reasoning",
     [("mmlu", "acc,none"), ("mmlu_pro_5_shot_base", "exact_match_official,tigerlab-extract"), ("agieval_en_cot", "macro-exact_match,custom-extract"),
      ("hellaswag", "acc_norm,none"), ("winogrande", "acc,none"), ("openbookqa", "acc_norm,none"), ("truthfulqa_mc2", "acc,none"),
      ("race_high_all", "acc,none")]),
    # user 11-13 (session 统一测试): "LiveCodeBench 贪心版 ... HMMT、AIME 2024、AIME 2025、GSM8K 官方 8-shot 版 全部变为重点". Their headline
    # metric: pass@1 like the other sampled tasks; gsm8k_cot by strict-match (the first metric of the task list; flexible-extract stays in
    # the results chapter). Same day: "数学 / 代码 分成两个吧": one tab for math, one for code
    ("math", "数学", "Math",
     [("gsm8k_cot_nemo", "exact_match,none"), ("gsm8k_cot", "exact_match,strict-match"), ("math_500_sampled", "pass@1[32],math_verify"),
      ("minerva_math_nemo", "exact_match,none"), ("aime24", "pass@1[32],math_verify"), ("aime25", "pass@1[32],math_verify"),
      ("hmmt_sampled", "pass@1[32],math_verify"), ("mgsm_native_cot_zh", "exact_match,symbolic-match")]),
    ("code", "代码", "Code",
     [("humaneval_sampled", "pass@1[32],evalplus_sanitized"), ("mbpp_sanitized_3shot_sampled", "pass@1[32],evalplus_sanitized"),
      ("lcb_codegen", "pass@1[1],score")]),   # user 11-14: "LiveCodeBench 采样 pass@1 这个不再是重点，livecodebench一个就好"
]
# metric labels: (alias, metric) first, then the metric name alone
LABEL = {
    ("mmlu_pro_5_shot_base", "exact_match_official,tigerlab-extract"): ("EM（官方）", "EM (official)"),
    ("mmlu_pro_5_shot_base", "exact_match,tigerlab-extract"): ("EM（未作答算错）", "EM (no answer = wrong)"),
    ("gsm8k_cot", "exact_match,strict-match"): ("EM strict", "EM strict"),
    ("gsm8k_cot", "exact_match,flexible-extract"): ("EM flexible", "EM flexible"),
    ("minerva_math_nemo", "exact_match_hendrycks,none"): ("EM（Hendrycks）", "EM (Hendrycks)"),
    ("trivia_qa", "exact_match_wiki,remove_whitespace"): ("EM（Wiki 子集）", "EM (Wiki rows)"),
    ("popqa_15shot", "acc_substring,remove_whitespace"): ("acc（子串，官方）", "acc (substring, official)"),
    ("agieval_en_cot", "macro-exact_match,custom-extract"): ("EM 宏平均（官方）", "EM macro (official)"),
    ("agieval_en_cot", "macro-exact_match,no-fallback"): ("EM 宏平均（无兜底）", "EM macro (no fallback)"),
    ("agieval_en_cot", "exact_match,custom-extract"): ("EM（按题数加权）", "EM (size-weighted)"),
    ("mmlu", "acc,none"): ("acc 总分", "acc overall"),
    ("mmlu", "acc_ties_shared,none"): ("acc 总分（并列均分）", "acc overall (ties shared)"),
    ("mmlu_stem", "acc,none"): ("acc STEM", "acc STEM"),
    ("mmlu_stem", "acc_ties_shared,none"): ("acc STEM（并列均分）", "acc STEM (ties shared)"),
}
GENERIC = {"exact_match": ("EM", "EM"), "acc": ("acc", "acc"), "acc_norm": ("acc_norm", "acc_norm"),
           "acc_ties_shared": ("acc（并列均分）", "acc (ties shared)"), "judge_accuracy": ("裁判准确率", "judge accuracy"), "f1": ("F1", "F1"),
           "correct": ("correct", "correct"), "not_attempted": ("not_attempted", "not_attempted"), "no_answer": ("未作答率", "no answer"),
           "majority": ("多数票", "majority vote"), "pass@1": ("pass@1", "pass@1"), "pass@32": ("pass@32", "pass@32")}
SHORT = {**_NSHORT, "Global-MMLU-zh": "G-MMLU-zh", "RACE-high": "RACE-h"}
TX = {
    "zh": {"h_scope": "全量测评范围", "h_focus": "全量测评重点", "h_res": "全量测评结果",
           "note": "全量测评 = 统一评测 harness lm-evaluation-harness（ultra-prod 分支）的任务清单 <code>nvbench/cosmos_ultra_ladder.yml</code>："
                   "{t} 个任务、{m} 个指标，取代原来 nemo6 的 40 个任务、75 个指标。各任务的 prompt、few-shot、停止符和答案抽取已按 benchmark "
                   "官方实现对齐。重点测评 = 其中 {kn} 个任务，每个取一个主分数（下面的树里标了“重点”，主分数加粗）。分数来自 "
                   "<code>runs/cosmos_eval_downstream</code>（11-12 起）；数学五项（MATH-500 / Minerva / AIME 24 / AIME 25 / HMMT）取数学 prompt 修正后的重测；"
                   "MiMo-V2.5-Pro-Base 取补上文档结束 stop token 后的整套重测。旧协议（nemo6）的分数和这一版不可比，不再放在报告里。",
           "root": "cosmos_ultra_ladder", "stats": "{k} 大部分 · {c} 类 · {t} 任务 · {b} 个 benchmark · {m} 个指标 · 重点 {kn} 个任务",
           "cnt": "{t} 任务 · {m} 指标", "shot": "{n}-shot",
           "chips": {"greedy": "greedy", "sampled": "采样 ×{r}", "judge": "LLM 裁判", "exec": "执行代码", "key": "重点"},
           "legend": "重点 = 进“全量测评重点”一章的任务，加粗的是它在那一章用的主分数。生成类的跑法：greedy = 贪心解码一次；采样 ×32 = 每题采样 32 次，"
                     "报 pass@1（32 次的平均）和 pass@32（至少对一次）；GPQA 采样版报 pass@1 和多数票（32 次里出现最多的答案）。执行代码 = 生成的代码"
                     "跑测试用例。acc（并列均分）= 最高分选项并列时按 1/k 计分。",
           "focus_note": "重点 = 全量测评 {t} 个任务里的 {kn} 个，每个任务只取一个主分数，分 {kt} 个标签页：{parts}。其余 {rt} 个任务，以及这 {kn} 个任务"
                         "的其它分数（共 {rm} 个指标），只在“全量测评结果”里。模型行、颜色和曲线的画法两章相同。",
           "focus_part": "{name} {n} 个", "focus_sep": "、",
           "pending": "数学 prompt 修正后的重测还没出分", "running": "评测还在跑，这一项还没出分",
           "stopped": "评测中途停了（作业被取消或失败），这一项没有分数",
           "definition": "这一项是用另一版任务定义测的，不可比，已撤下",
           "s_h": "重点指标随训练步数的变化", "s_run": "{run}（一个点一个 checkpoint）", "s_step": "iter {s}",
           "head": ["模型", "大小"], "post": "后训练",
           "flag": "- = 没有有效分数（不是 0 分），悬停看原因：该任务有 ≥ 20% 的回答是空的（分数无效，已撤下；悬停看比例和原始分），或者还没出分。"
                   "带虚线下划线的分数 = 空回答占 5–20%，分数保留，悬停看比例。未作答率 / not_attempted 是比例，不是分数，始终显示。",
           "empty": "空回答 {p:.0f}%，分数无效（原始分 {v}）", "warn": "空回答 {p:.0f}%（低于 20%，分数保留）", "missing": "没有分数"},
    "en": {"h_scope": "Full evaluation scope", "h_focus": "Full evaluation: key benchmarks", "h_res": "Full evaluation results",
           "note": "Full evaluation = the task list <code>nvbench/cosmos_ultra_ladder.yml</code> of the unified eval harness "
                   "lm-evaluation-harness (branch ultra-prod): {t} tasks, {m} metrics, replacing nemo6's 40 tasks and 75 metrics. Prompt, few-shot "
                   "examples, stops and answer reading of each task follow the benchmark's official implementation. Key evaluation = {kn} of "
                   "these tasks, one headline metric each (marked \"key\" in the tree below, headline metric in bold). Scores come from "
                   "<code>runs/cosmos_eval_downstream</code> (since 11-12); the five math tasks (MATH-500 / Minerva / AIME 24 / AIME 25 / HMMT) from the "
                   "re-run with the corrected math prompt; MiMo-V2.5-Pro-Base from the complete re-run with the end-of-document stop token. "
                   "Scores of the earlier protocol (nemo6) are not comparable with this version and are no longer in the report.",
           "root": "cosmos_ultra_ladder", "stats": "{k} parts · {c} categories · {t} tasks · {b} benchmarks · {m} metrics · {kn} key tasks",
           "cnt": "{t} tasks · {m} metrics", "shot": "{n}-shot",
           "chips": {"greedy": "greedy", "sampled": "{r} samples", "judge": "LLM judge", "exec": "code exec", "key": "key"},
           "legend": "key = a task of the chapter \"Full evaluation: key benchmarks\"; its headline metric there is in bold. Generation modes: "
                     "greedy = one greedy decode; 32 samples = 32 samples per item, pass@1 (mean of the 32) and pass@32 (any correct); sampled "
                     "GPQA reports pass@1 and the majority vote (the most frequent answer of the 32). Code exec = the generated code runs the "
                     "test cases. acc (ties shared) = an exact tie at the top counts 1/k.",
           "focus_note": "Key = {kn} of the {t} tasks of the full evaluation, one headline metric per task, in {kt} tabs: {parts}. The other "
                         "{rt} tasks, and the other scores of these {kn} tasks ({rm} metrics in all), are in \"Full evaluation results\" only. "
                         "Model rows, colours and curves are the same in both chapters.",
           "focus_part": "{name} {n}", "focus_sep": ", ",
           "pending": "the re-run with the corrected math prompt has no score yet", "running": "the evaluation is still running, no score for this task yet",
           "stopped": "the evaluation stopped before this task (job cancelled or failed), no score",
           "definition": "scored with another version of the task definition: not comparable, withdrawn",
           "s_h": "Key metrics vs. training step", "s_run": "{run} (one point per checkpoint)", "s_step": "iter {s}",
           "head": ["model", "size"], "post": "post-trained",
           "flag": "- = no valid score (not a zero), hover for the reason: >= 20 % of the task's answers are empty (score invalid, withdrawn; "
                   "hover for the share and the raw score), or the score is not in yet. A score with a dotted underline = 5-20 % empty answers, "
                   "kept; hover for the share. The no-answer / not_attempted rates are rates, not scores, and always shown.",
           "empty": "{p:.0f}% empty answers, score invalid (raw score {v})", "warn": "{p:.0f}% empty answers (below 20 %, score kept)",
           "missing": "no score"},
}
RCSS = ("<style>.nt-tab th{white-space:normal;vertical-align:bottom;min-width:64px}.nt-tab td.nt-flag{color:var(--muted);font-style:italic}"
        ".nt-tab td.nt-warn{text-decoration:underline dotted;text-underline-offset:3px}.nt-fn{margin:8px 0 0}.nt-tab{margin-top:10px}"
        ".nt-part+.nt-part{margin-top:26px;padding-top:14px;border-top:1px solid var(--grid)}"
        ".nt-c.key{background:color-mix(in srgb,var(--text-primary) 11%,transparent);color:var(--text-primary);font-weight:600}"
        ".nt-m b{color:var(--text-primary);font-weight:600}"
        ".tk-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(232px,1fr));gap:14px 18px;margin:10px 0 4px}"
        ".tk-cell .tk-t{font-size:12.5px;font-weight:600;line-height:1.3}.tk-cell .tk-t span{font-weight:400;color:var(--text-secondary)}"
        ".tk-leg{display:flex;flex-wrap:wrap;gap:4px 18px;font-size:12.5px;color:var(--text-secondary);margin:8px 0 2px}"
        ".tk-leg span{display:inline-flex;align-items:center;gap:6px;white-space:nowrap}"
        + SHADES + "</style>")
KEY = {alias: metric for _, _, _, keys in FOCUS_TABS for alias, metric in keys}   # task (= alias of its headline metric) -> headline metric


def _cfg():
    cfg = yaml.safe_load(open(CFG))["tasks"]
    by = {t["task_name"]: t for t in cfg}
    listed = [tn for b in BENCH for tn, _, _ in b[4]]
    assert sorted(listed) == sorted(by), f"cosmos_eval: BENCH out of sync with the task list: {set(by) ^ set(listed)}"
    for t in cfg:
        mets = {rm["metric"].split(",")[0] for rm in t["report_metric"]}
        assert (t["task_name"] in NONGEN) == (mets <= {"acc", "acc_norm", "acc_ties_shared"}), f"cosmos_eval: NONGEN out of sync for {t['task_name']}"
    placed = [n for _, _, parts in SECTIONS for _, _, names in parts for n in names]
    assert sorted(placed) == sorted(b[1] for b in BENCH), f"cosmos_eval: SECTIONS out of sync with BENCH: {set(placed) ^ {b[1] for b in BENCH}}"
    keys = [k for _, _, _, ks in FOCUS_TABS for k in ks]
    assert len(KEY) == len(keys), "cosmos_eval: FOCUS_TABS names a task twice"
    for alias, metric in keys:                             # a key entry = the task itself (not a subgroup) and one of its report metrics
        assert alias in by and any((rm["alias"], rm["metric"]) == (alias, metric) for rm in by[alias]["report_metric"]), \
            f"cosmos_eval: FOCUS_TABS entry {alias} {metric} is not a report metric of the task list"
    return cfg, by


def _kind(b):
    ks = {"ll" if tn in NONGEN else "gen" for tn, _, _ in b[4]}
    assert len(ks) == 1, f"cosmos_eval: mixed kinds in {b[1]}"
    return ks.pop()


def _mname(alias, metric, lang):
    i = 0 if lang == "zh" else 1
    if (alias, metric) in LABEL:
        return LABEL[(alias, metric)][i]
    n = re.sub(r"\[\d+\]$", "", metric.split(",")[0])
    return GENERIC.get(n, (n, n))[i]


def _load():
    out = {}
    if os.path.isdir(RESULTS):
        for f in sorted(os.listdir(RESULTS)):
            if f.endswith(".json"):
                try:
                    d = json.load(open(os.path.join(RESULTS, f)))
                    out[d["name"]] = d
                except Exception:
                    pass
    return out


def ce_results():
    """name -> the collected JSON of every evaluated model (results_xx/cosmos_eval); "ours": true = one of our checkpoints."""
    return _load()


def ce_counts():
    """(tasks, metrics, key tasks) of the task list."""
    cfg, _ = _cfg()
    return len(cfg), sum(len(t["report_metric"]) for t in cfg), len(KEY)


def _chips(t):
    if t["task_name"] in NONGEN:
        return []
    reps = int((t.get("overrides") or {}).get("repeats", 1))
    mets = {rm["metric"].split(",")[0] for rm in t["report_metric"]}
    chips = [("exec", None)] if t["task_group"] == "code" else []
    if "judge_accuracy" in mets:
        chips.append(("judge", None))
    chips.append(("sampled", reps) if reps > 1 else ("greedy", None))
    return chips


def ce_chapter(lang, esc=_e, more=""):
    """HTML of "全量测评范围": the protocol note (+ more: further note html of the tab) + the benchmark tree of the task list with the key
    tasks marked. Empty string without the config or any result."""
    res = _load()
    if not os.path.exists(CFG) or not res:
        return ""
    tx, zh = TX[lang], lang == "zh"
    cfg, by = _cfg()
    pilot = {t["task"]: t for ts in (res.get(PILOT) or {}).get("groups", {}).values() for t in ts}

    def shots(tn):
        s = (pilot.get(tn) or {}).get("n_shot")
        if isinstance(s, int) and not (s == 0 and tn in ZERO_SHOT):
            return s
        if isinstance(by[tn].get("num_fewshot"), int):
            return by[tn]["num_fewshot"]
        return 0 if tn in ZERO_SHOT else None

    def row(tn, label):
        t = by[tn]
        s = shots(tn)
        names, bold = [], set()
        for rm in t["report_metric"]:
            n = _mname(rm["alias"], rm["metric"], lang)
            if n not in names:
                names.append(n)
            if KEY.get(rm["alias"]) == rm["metric"]:       # the headline metric of a key task
                bold.add(n)
        chips = "".join(f'<span class="nt-c {k}">{esc(tx["chips"][k].format(r=r))}</span>' for k, r in ([("key", None)] if tn in KEY else []) + _chips(t))
        bits = [b for b in (label, tx["shot"].format(n=s) if s is not None else "") if b]
        mets = " · ".join(f"<b>{esc(n)}</b>" if n in bold else esc(n) for n in names)
        return f'<span class="nt-d">{esc(" · ".join(bits))}</span>{chips} <span class="nt-m">→ {mets}</span>'

    def card(kind, key, name, how):
        bs = [b for b in BENCH if b[0] == key and _kind(b) == kind]
        tasks = [tn for b in bs for tn, _, _ in b[4]]
        mets = sum(len(by[tn]["report_metric"]) for tn in tasks)
        items = []
        for _, bname, dzh, den, variants in bs:
            d = dzh if zh else den
            head = f'<span class="nt-b">{esc(bname)}</span>' + (f' <span class="nt-d">{esc(d)}</span>' if d else "")
            if len(variants) == 1:
                items.append(f"<li>{head} {row(variants[0][0], '')}</li>")
            else:
                items.append(f"<li>{head}<ul>" + "".join(f"<li>{row(tn, vz if zh else ve)}</li>" for tn, vz, ve in variants) + "</ul></li>")
        return (f'<div class="card"><div class="nt-h"><b>{esc(name)}</b><span class="nt-s">{esc(key)} · {esc(tx["cnt"].format(t=len(tasks), m=mets))}</span></div>'
                f'<div class="nt-how">{esc(how)}</div><ul class="nt-tree">{"".join(items)}</ul></div>')

    assert {(_kind(b), b[0]) for b in BENCH} <= {c[:2] for c in CATS}, "cosmos_eval: a benchmark has no (kind, task_group) category"
    parts = []
    for kind, kzh, ken, lzh, len_ in KINDS:
        ktasks = [tn for b in BENCH if _kind(b) == kind for tn, _, _ in b[4]]
        kmets = sum(len(by[tn]["report_metric"]) for tn in ktasks)
        cards = [card(kind, key, czh if zh else cen, hzh if zh else hen) for k, key, czh, cen, hzh, hen in CATS if k == kind]
        parts.append(f'<h3 class="nt-k">{esc(kzh if zh else ken)} <span class="nt-s">{esc(tx["cnt"].format(t=len(ktasks), m=kmets))}</span></h3>'
                     f'<p class="note">{esc(lzh if zh else len_)}</p><div class="nt-grid">{"".join(cards)}</div>')
    nm = sum(len(t["report_metric"]) for t in cfg)
    stats = tx["stats"].format(k=len(KINDS), c=len(CATS), t=len(cfg), b=len(BENCH), m=nm, kn=len(KEY))
    return (f'{CSS}{KCSS}{RCSS}<h2>{esc(tx["h_scope"])}</h2><p class="note">{tx["note"].format(t=len(cfg), m=nm, kn=len(KEY))}</p>{more}'
            f'<div class="card nt-root"><b>{esc(tx["root"])}</b><span class="nt-s">{esc(stats)}</span>'
            f'<span class="nt-c key">{esc(tx["chips"]["key"])}</span><span class="nt-c greedy">greedy</span>'
            f'<span class="nt-c sampled">{esc(tx["chips"]["sampled"].format(r=32))}</span><span class="nt-c exec">{esc(tx["chips"]["exec"])}</span>'
            f'<span class="nt-c judge">{esc(tx["chips"]["judge"])}</span></div>{"".join(parts)}<p class="nt-leg">{esc(tx["legend"])}</p>')


def ce_scores(lang, esc, ref_models, ref_size, focus=False, colors=None, legend=True):
    """HTML of "全量测评重点" (focus=True: one card, a tab per FOCUS_TABS entry) or "全量测评结果" (a tab per SECTIONS entry). Rows = the models
    of ref_models [(code, display name, _)] that have a results_xx/cosmos_eval JSON, in that order.
    colors = {code: css variable}: the downstream tab's form. Every model has its own colour, and its lines, points, hover rows and table
    rows carry data-dm=<code> for the show / hide boxes of "实验列表"; legend=False leaves the legend to that table."""
    res = _load()
    models = [(c, n) for c, n, _ in ref_models if c in res]
    if not os.path.exists(CFG) or not models:
        return ""
    tx, zh = TX[lang], lang == "zh"
    ctx = {**_NSTXT[lang], **tx}                           # the curve's own strings (axis, family legend) + ours
    cfg, by = _cfg()
    cells, empty, pending = {}, {}, {}
    for c, _ in models:
        for ts in res[c]["groups"].values():
            for t in ts:
                empty[(c, t["task"])] = t.get("empty_frac")
                if t.get("pending"):
                    pending[(c, t["task"])] = t["pending"]
                for m in t["metrics"]:
                    cells[(c, m["alias"], m["metric"])] = m["value"]

    def col(bname, tn, v, rm):
        """one report metric -> (header html, getter(code) -> (value | None, reason), warn(code) -> text | None, x label 1, x label 2, drawn)"""
        alias, metric = rm["alias"], rm["metric"]
        base = re.sub(r"\[\d+\]$", "", metric.split(",")[0])
        sub = _mname(alias, metric, lang)

        def g(c):
            val, e = cells.get((c, alias, metric)), empty.get((c, tn))
            if not isinstance(val, (int, float)):
                return None, tx.get({"mathfix": "pending"}.get(pending.get((c, tn)), pending.get((c, tn))), tx["missing"])
            if base not in RATES and isinstance(e, (int, float)) and e >= EMPTY_FLAG:
                return None, tx["empty"].format(p=e * 100, v=f"{val * 100:.1f}")
            return val, None

        def warn(c):
            e = empty.get((c, tn))
            return tx["warn"].format(p=e * 100) if base not in RATES and isinstance(e, (int, float)) and EMPTY_NOTE <= e < EMPTY_FLAG else None
        top = esc(bname) + (f' <span class="nt-d">{esc(v)}</span>' if v else "")
        return (f'{top}<br><span class="nt-d">{esc(sub)}</span>', g, warn, SHORT.get(bname, bname) + (f" {v}" if v else ""), sub, base not in NO_CURVE)

    def cols_of(names):
        out = []
        for b in BENCH:
            if b[1] in names:
                for tn, vz, ve in b[4]:
                    out += [((rm["alias"], rm["metric"]), col(b[1], tn, vz if zh else ve, rm)) for rm in by[tn]["report_metric"]]
        order = {n: i for i, n in enumerate(names)}        # columns in the order the section lists its benchmarks
        bn = {(rm["alias"], rm["metric"]): b[1] for b in BENCH for tn, _, _ in b[4] for rm in by[tn]["report_metric"]}
        return sorted(out, key=lambda x: order[bn[x[0]]])

    def table(cols):
        head = (f'<tr><th class="l">{esc(tx["head"][0])}</th><th class="l">{esc(tx["head"][1])}</th>' + "".join(f"<th>{c[0]}</th>" for c in cols) + "</tr>")
        body, flagged = [], False
        for c, n in models:
            tag = f' <span class="nt-d">{esc(tx["post"])}</span>' if res[c].get("post_trained") else ""
            tds = []
            for _, g, warn, *_ in cols:
                v, why = g(c)
                w = warn(c)
                if v is None:                              # withdrawn or missing = "-", never 0 (user 11-06); reason on hover
                    tds.append(f'<td class="nt-flag" data-v="" title="{esc(why)}">-</td>')
                else:
                    tds.append(f'<td data-v="{v}"' + (f' class="nt-warn" title="{esc(w)}"' if w else "") + f">{v * 100:.1f}</td>")
                flagged = flagged or v is None or bool(w)
            dm = f' data-dm="{esc(c)}"' if colors else ""
            body.append(f'<tr data-s="1"{dm}><td class="l">{esc(n)}{tag}</td><td class="l nt-d">{esc(ref_size.get(c, ""))}</td>{"".join(tds)}</tr>')
        return f'<table class="rsort nt-tab" data-hi="1"><thead>{head}</thead><tbody>{"".join(body)}</tbody></table>', flagged

    def pane(cols, title=""):
        tbl, flagged = table(cols)
        tbl = '<div style="overflow-x:auto">' + tbl.replace('<table class="rsort nt-tab"', '<table class="rsort nt-tab" style="display:table;width:100%"', 1) + "</div>"
        chart = _curve([(l1, l2, g) for _, g, _, l1, l2, ok in cols if ok], models, esc, ctx, colors=colors, legend=legend)
        return f"{title}{chart}{tbl}" + (f'<p class="note nt-fn">{esc(tx["flag"])}</p>' if flagged else "")

    if focus:
        allc = dict(cols_of([b[1] for b in BENCH]))
        tabs = [(k, nz if zh else ne, pane([allc[key] for key in keys])) for k, nz, ne, keys in FOCUS_TABS]
        pre, h = "cf", tx["h_focus"]
        nt, nm = len(cfg), sum(len(t["report_metric"]) for t in cfg)
        parts = tx["focus_sep"].join(tx["focus_part"].format(name=nz if zh else ne, n=len(keys)) for _, nz, ne, keys in FOCUS_TABS)
        note = f'<p class="note">{esc(tx["focus_note"].format(t=nt, kn=len(KEY), kt=len(FOCUS_TABS), parts=parts, rt=nt - len(KEY), rm=nm - len(KEY)))}</p>'
    else:
        tabs = []
        for i, (nz, ne, parts) in enumerate(SECTIONS):
            cats = []                                      # the scoring note of every category in the section, in CATS order
            for kind, key, czh, cen, hzh, hen in CATS:
                if any(b[1] in names and (_kind(b), b[0]) == (kind, key) for _, _, names in parts for b in BENCH):
                    cats.append((czh, hzh) if zh else (cen, hen))
            how = ("；" if zh else "; ").join(f"{a}{'：' if zh else ': '}{b}" for a, b in cats) if len(cats) > 1 else cats[0][1]
            body = f'<p class="note">{esc(how)}</p>'
            for ptz, pte, names in parts:
                title = f'<div class="ct">{esc(ptz if zh else pte)}</div>' if ptz else ""
                body += f'<div class="nt-part">{pane([c for _, c in cols_of(names)], title)}</div>'
            tabs.append((str(i), nz if zh else ne, body))
        pre, h, note = "cr", tx["h_res"], ""
    btns = "".join(f'<button data-r="{pre}_{k}" class="{"on" if i == 0 else ""}">{esc(lab)}</button>' for i, (k, lab, _) in enumerate(tabs))
    panes = "".join(f'<div data-ref="{pre}_{k}"{"" if i == 0 else " hidden"}>{body}</div>' for i, (k, _, body) in enumerate(tabs))
    return RCSS + KCSS + f'<h2>{esc(h)}</h2>{note}<div class="card"><div class="rtoggle" style="flex-wrap:wrap">{btns}</div>{panes}</div>'


def _ticks(lo, hi, n=3):
    """nice y ticks covering [lo, hi]: (first tick, step, count)."""
    span = max(hi - lo, 1e-9)
    step = next((c for c in (0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50) if span / c <= n), 50)
    a = math.floor(lo / step + 1e-9) * step
    k = int(math.ceil((hi - a) / step - 1e-9))
    return a, step, max(k, 1)


def _mini(esc, title, sub, pts, refs, xmax, tipx, w=300, h=178):
    """one small chart: pts = [(step, value 0-100, colour var of the checkpoint)] of the run, joined by a neutral line; refs = [(name,
    value, colour)] as dashed horizontal lines."""
    ml, mr, mt, mb = 30, 14, 10, 22
    vals = [v for _, v, _ in pts] + [v for _, v, _ in refs]
    lo, hi = min(vals), max(vals)
    pad = max((hi - lo) * 0.12, 0.6)
    a, st, k = _ticks(lo - pad, hi + pad)
    y0, y1 = a, a + st * k
    X = lambda x: ml + x / xmax * (w - ml - mr)
    Y = lambda v: mt + (y1 - v) / (y1 - y0) * (h - mt - mb)
    dec = 1 if st < 1 else 0
    o = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="{esc(title + " " + sub)}">']
    for i in range(k + 1):                                 # recessive grid + y tick labels
        yv = y0 + st * i
        o.append(f'<line x1="{ml}" x2="{w - mr}" y1="{Y(yv):.1f}" y2="{Y(yv):.1f}" stroke="var(--grid)" stroke-width="1"/>'
                 f'<text x="{ml - 5}" y="{Y(yv) + 3.5:.1f}" text-anchor="end" style="fill:var(--muted);font-size:10px">{yv:.{dec}f}</text>')
    tick = next(c for c in (2000, 5000, 10000, 20000, 50000) if xmax / c <= 4)
    for x in range(0, xmax + 1, tick):                     # x axis = training step
        o.append(f'<text x="{X(x):.1f}" y="{h - mb + 14}" text-anchor="middle" style="fill:var(--muted);font-size:10px">{f"{x // 1000}k" if x else "0"}</text>')
    o.append(f'<line x1="{ml}" x2="{w - mr}" y1="{h - mb}" y2="{h - mb}" stroke="var(--axis)"/>')
    for _, v, c in refs:                                   # references: dashed, as in the prod iso-doc tab
        o.append(f'<line x1="{ml}" x2="{w - mr}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="var({c})" stroke-width="1.6" stroke-dasharray="5 4"/>')
    if len(pts) > 1:
        d = "M" + " L".join(f"{X(x):.1f} {Y(v):.1f}" for x, v, _ in pts)
        o.append(f'<path d="{d}" fill="none" stroke="var(--muted)" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
    for x, v, c in pts:                                    # each checkpoint in its own grey of "实验列表"
        o.append(f'<circle cx="{X(x):.1f}" cy="{Y(v):.1f}" r="4.5" fill="var({c})" stroke="var(--surface-1)" stroke-width="2"/>')
    x, v, _ = pts[-1]                                      # direct label: the latest value only, in text ink
    left = X(x) > w - mr - 34
    o.append(f'<text x="{X(x) + (-8 if left else 8):.1f}" y="{Y(v) + (-8 if left else 4):.1f}" text-anchor="{"end" if left else "start"}" '
             f'style="fill:var(--text-primary);font-size:11.5px;font-weight:600">{v:.1f}</text>')
    xs = [x for x, _, _ in pts]
    for i, (x, v, c) in enumerate(pts):                    # hover band per checkpoint
        a_ = ml if i == 0 else (X(xs[i - 1]) + X(x)) / 2
        b_ = w - mr if i == len(pts) - 1 else (X(x) + X(xs[i + 1])) / 2
        tip = (f"<b>{esc(title)} · {esc(sub)}</b><br><span style=\"color:var({c})\">●</span> {esc(tipx(x))}: {v:.1f}"
               + "".join(f'<br><span style="color:var({rc})">●</span> {esc(n)}: {rv:.1f}' for n, rv, rc in refs))
        o.append(f'<rect class="hit" x="{a_:.1f}" y="{mt}" width="{b_ - a_:.1f}" height="{h - mt - mb}" data-tip="{esc(tip)}"/>')
    o.append("</svg>")
    return "".join(o)


def ce_steps(lang, esc, run, steps, refs, colors):
    """HTML of the block above "实验列表" (user 11-13: "上面还是把一个 blocked chart，是重点指标，根据 x-axis 训练时间变化标注出来"): one card, a tab
    per FOCUS_TABS entry, in each a grid of small charts: the headline score of one key task against the training step, one point per
    evaluated checkpoint of the tracked run in that checkpoint's colour, the baselines as dashed lines. Charts only, no notes and no table:
    the numbers are the rows of "全量测评重点" below, so the grid is .nomd (not repeated in the md mirror).
    steps = [(code, step)], refs = [(code, name, colour var)], colors = {code: colour var}."""
    res = _load()
    steps = sorted(((c, s) for c, s in steps if c in res), key=lambda x: x[1])
    refs = [(c, n, k) for c, n, k in refs if c in res]
    if not os.path.exists(CFG) or not steps:
        return ""
    tx, zh = TX[lang], lang == "zh"
    total = (res[steps[0][0]].get("track") or {}).get("total_steps") or max(s for _, s in steps)
    xmax = max(10000, math.ceil(max(s for _, s in steps) * 1.06 / 5000) * 5000)   # the prod iso-doc tab's x range
    empty, cells = {}, {}
    for c in [c for c, _ in steps] + [c for c, _, _ in refs]:
        for ts in res[c]["groups"].values():
            for t in ts:
                empty[(c, t["task"])] = t.get("empty_frac")
                for m in t["metrics"]:
                    cells[(c, m["alias"], m["metric"])] = m["value"]

    def get(c, alias, metric):                             # score x100, or None: no score, or withdrawn for empty answers (as the tables)
        v, e = cells.get((c, alias, metric)), empty.get((c, alias))
        if not isinstance(v, (int, float)) or (isinstance(e, (int, float)) and e >= EMPTY_FLAG):
            return None
        return v * 100

    bname = {tn: (b[1], (vz if zh else ve)) for b in BENCH for tn, vz, ve in b[4]}
    tipx = lambda s: f'{tx["s_step"].format(s=f"{s:,}")} ({s / total:.0%})'
    tabs = []
    for key, nz, ne, keys in FOCUS_TABS:
        grid = []
        for alias, metric in keys:
            bn, var = bname[alias]
            title, sub = bn + (f" {var}" if var else ""), _mname(alias, metric, lang)
            pts = [(s, get(c, alias, metric), colors.get(c, "--text-primary")) for c, s in steps]
            pts = [p for p in pts if p[1] is not None]
            rf = [(n, get(c, alias, metric), k) for c, n, k in refs]
            rf = [r for r in rf if r[1] is not None]
            svg = _mini(esc, title, sub, pts, rf, xmax, tipx) if pts else ""
            grid.append(f'<div class="tk-cell"><div class="tk-t">{esc(title)} <span>{esc(sub)}</span></div>{svg}</div>')
        tabs.append((key, nz if zh else ne, f'<div class="tk-grid nomd">{"".join(grid)}</div>'))
    last = colors.get(steps[-1][0], "--text-primary")
    leg = (f'<span><svg width="30" height="10"><line x1="1" x2="29" y1="5" y2="5" stroke="var(--muted)" stroke-width="2"/>'
           f'<circle cx="15" cy="5" r="4" fill="var({last})" stroke="var(--surface-1)" stroke-width="2"/></svg>{esc(tx["s_run"].format(run=run))}</span>'
           + "".join(f'<span><svg width="30" height="10"><line x1="1" x2="29" y1="5" y2="5" stroke="var({k})" stroke-width="1.6" stroke-dasharray="5 4"/></svg>'
                     f'{esc(n)}</span>' for _, n, k in refs))
    btns = "".join(f'<button data-r="cs_{k}" class="{"on" if i == 0 else ""}">{esc(lab)}</button>' for i, (k, lab, _) in enumerate(tabs))
    panes = "".join(f'<div data-ref="cs_{k}"{"" if i == 0 else " hidden"}>{b}</div>' for i, (k, _, b) in enumerate(tabs))
    return (RCSS + KCSS + f'<h2>{esc(tx["s_h"])}</h2><div class="card"><div class="rtoggle" style="flex-wrap:wrap">{btns}</div>'
            f'<div class="tk-leg nomd">{leg}</div>{panes}</div>')
