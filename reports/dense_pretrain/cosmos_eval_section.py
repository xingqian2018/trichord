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
import itertools
import json
import math
import os
import re

import yaml

from nemo6_section import CSS, KCSS, SHADES, SHORT as _NSHORT, STXT as _NSTXT, _curve, _e

HERE = os.path.dirname(os.path.abspath(__file__))
# One colour per model in BOTH prod tabs (user 11-16 "色系要能统一"; "我们的线用白色就好，不用区分颜色"): prod downstream (gen_eval_html.py
# DOWN_BASE) and prod iso-doc (prod_isodoc_section.py REFS) take their colours from here and both emit PROD_FAM_CSS.
#   our production run: --text-primary (white in dark, black in light; prod downstream: its checkpoints grey toward it, the latest = it)
#   Qwen3-30B-A3B Base / Instruct-2507 / Thinking-2507: blues --c1 / --db2 / --db3 (user 11-16 "三个baseline（蓝色系）"), Base strongest
#   Qwen3.8-27B / Qwen3.8-Flash-Next / Qwen3.5-397B-A17B / Qwen3.8-Max: orange --dq1..4 by total size, strongest = largest
#   MiMo-V2.5-Pro-Base (the KD teacher): violet --dt; ladder references (iso-doc only): aqua --c3 / --g07 / --g09
# The orange ramp was searched for the largest colour-blind distance to every other family (>= 8.3 dE under protan / deutan / tritan in both
# modes); --ordinal passes in both modes. The teacher's violet: >= 9.5 (light) / 6.9 (dark). Weakest pair left: dark-mode KD q3 aqua vs
# Thinking-2507 blue (3.8).
# 11-17 (Doc-isolate测试, for the user's "nemo 3.5 nano" = NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Base): Nemotron = magenta --dn (OKLCH hue 340,
# chroma 0.14), searched the same way: >= 8.8 (light) / 8.3 (dark) to every colour above under normal / protan / deutan / tritan vision; contrast
# 3.97 / 5.42 on the surface. (NVIDIA green was not usable: 2.8 / 0.6 from the oranges under deutan, and the ladder references are green already.)
PROD_FAM = {"QWEN3_30B_A3B_BASE": "--c1", "QWEN3_30B_A3B_INSTRUCT_2507": "--db2", "QWEN3_30B_A3B_THINKING_2507": "--db3",
            "QWEN38_27B": "--dq1", "QWEN38_FLASH_NEXT": "--dq2", "QWEN35_397B_A17B": "--dq3", "QWEN38_MAX": "--dq4", "MIMO_V25_PRO_BASE": "--dt",
            "NEMOTRON35_LIGHTNING_30B_BASE": "--dn",
            "NEMOTRON35_LIGHTNING_30B_A3B_BASE": "--dn"}   # the same model under its downstream run code (job 76598, Downstream测试); unused until listed
PROD_FAM_CSS = ('<style>.viz-root{--db2:#4b96f5;--db3:#72affe;--dq1:#ff7e2d;--dq2:#e56806;--dq3:#af4e03;--dq4:#893b01;--dt:#5341b1;--dn:#b95e9d}'
                '@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) '
                '.viz-root{--db2:#1a69c4;--db3:#0250a0;--dq1:#964101;--dq2:#d86105;--dq3:#f4741e;--dq4:#ff9f6d;--dt:#9c95fe;--dn:#cd70b0}}</style>')
# y-range variants of a chart whose references follow tick boxes (prod val loss and prod downstream, 11-16 review): a chart block .rk-wrap
# holds one drawing per y range and data-rv = its reference values in tick-box order; the page script shows the drawing whose data-rk holds
# "lo,hi" = the indices of the lowest / highest ticked value (first index on ties), "e" = nothing ticked. O(k^2) drawings instead of one per
# subset of the k boxes (2^k). The hover rects do not depend on the y range: one transparent overlay svg (.rk-hit) per chart block.
RK_CSS = "<style>.rk-wrap{position:relative}.rk-v svg{display:block}.rk-hit{position:absolute;left:0;top:0;width:100%;height:100%}</style>"


def rk_variants(rv):
    """rv = reference values in tick-box order (None = no value) -> {key: (lo, hi) or None}, every key the page script can compute."""
    out = {"e": None}
    idx = [i for i, v in enumerate(rv) if v is not None]
    for i in idx:
        for j in idx:
            if i == j or rv[i] < rv[j]:
                out[f"{i},{j}"] = (rv[i], rv[j])
    return out


def rk_key(rv, on):
    """the key of the ticked set `on` (booleans in tick-box order), as the page script computes it."""
    lo = hi = -1
    for i, (v, o) in enumerate(zip(rv, on)):
        if v is None or not o:
            continue
        if lo < 0 or v < rv[lo]:
            lo = i
        if hi < 0 or v > rv[hi]:
            hi = i
    return "e" if lo < 0 else f"{lo},{hi}"


def rk_block(esc, kind, rv, on, draw, hits):
    """the chart block: draw(fit) -> svg for the y range that holds `fit`; hits = the overlay svg; kind = whose tick boxes (pi / dm)."""
    drawn, by = {}, {}
    for key, rg in rk_variants(rv).items():
        if rg not in drawn:
            drawn[rg] = draw([] if rg is None else list(rg))
        by.setdefault(drawn[rg], []).append(key)
    k0 = rk_key(rv, on)
    return (f'<div class="rk-wrap" data-rkc="{kind}" data-rv="{esc(json.dumps(rv))}">'
            + "".join(f'<div class="rk-v" data-rk="{" ".join(ks)}"{"" if k0 in ks else " hidden"}>{svg}</div>' for svg, ks in by.items())
            + hits + "</div>")


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
# prod downstream tab only (Runlong 11-16 16:5x, session "Pretrain Benchmark Track": "prod downstream 页面，把所有 bench 都放完，分类，按照你觉得的
# 重要性排序"): ALL 36 tasks, one headline metric each, in six categories; categories and the tasks inside them in order of importance for a
# pretraining base model (low noise and decision value first; saturated / near-zero tasks last). The 外部参考 tab keeps FOCUS_TABS.
#   know: MMLU-Pro (12k CoT questions) first, the MMLU family, then GPQA (198 q, sampled before greedy), AGIEval, SuperGPQA
#   math: MATH-500 / Minerva (5,000 q, low noise) before GSM8K (near ceiling), AIME / HMMT last (base models score 0-5 %)
#   code: LiveCodeBench greedy (release v6, least contaminated) first, then HumanEval, MBPP, LCB sampled
#   fact: TriviaQA / NQ / PopQA (large, stable) before the small low-score SimpleQA / SimpleQA-Verified / FACTS
#   zh: Global-MMLU-zh, INCLUDE-zh (Chinese knowledge), MGSM-zh (250 q)
#   cs: commonsense / reading: HellaSwag, WinoGrande, ARC first; OpenBookQA (500 q), Social IQa, TruthfulQA (noisy) last
PROD_TABS = [
    ("know", "知识 / 推理考试", "Knowledge / reasoning exams",
     [("mmlu_pro_5_shot_base", "exact_match_official,tigerlab-extract"), ("mmlu", "acc,none"), ("mmlu_redux", "acc,none"),
      ("gpqa_diamond_cot_n_shot_sampled", "pass@1[32],flexible-extract"), ("gpqa_diamond_cot_n_shot", "exact_match,flexible-extract"),
      ("agieval_en_cot", "macro-exact_match,custom-extract"), ("supergpqa_main_cot_n_shot", "exact_match,flexible-extract")]),
    ("math", "数学", "Math",
     [("math_500_sampled", "pass@1[32],math_verify"), ("minerva_math_nemo", "exact_match,none"), ("gsm8k_cot_nemo", "exact_match,none"),
      ("gsm8k_cot", "exact_match,strict-match"), ("aime24", "pass@1[32],math_verify"), ("aime25", "pass@1[32],math_verify"),
      ("hmmt_sampled", "pass@1[32],math_verify")]),
    ("code", "代码", "Code",
     [("lcb_codegen", "pass@1[1],score"), ("humaneval_sampled", "pass@1[32],evalplus_sanitized"),
      ("mbpp_sanitized_3shot_sampled", "pass@1[32],evalplus_sanitized"), ("lcb_codegen_sampled", "pass@1[32],score")]),
    ("fact", "世界知识问答", "World-knowledge QA",
     [("trivia_qa", "exact_match,remove_whitespace"), ("naturalqs_5shot", "exact_match,remove_whitespace"),
      ("popqa_15shot", "acc_substring,remove_whitespace"), ("simpleqa_5shot", "f1,none"), ("simpleqa_verified_25shot", "judge_accuracy,none"),
      ("facts_parametric_25shot", "judge_accuracy,none")]),
    ("zh", "中文", "Chinese",
     [("global_mmlu_zh", "acc,none"), ("include_base_44_chinese", "acc,none"), ("mgsm_native_cot_zh", "exact_match,symbolic-match")]),
    ("cs", "常识 / 阅读", "Commonsense / reading",
     [("hellaswag", "acc_norm,none"), ("winogrande", "acc,none"), ("arc_challenge_llama", "acc_norm,none"), ("commonsense_qa", "acc,none"),
      ("piqa", "acc_norm,none"), ("race_high_all", "acc,none"), ("openbookqa", "acc_norm,none"), ("social_iqa", "acc,none"),
      ("truthfulqa_mc2", "acc,none")]),
]
# the results chapter of the prod downstream tab in the same categories and order (every metric of every benchmark; SECTIONS stays for 外部参考)
PROD_SECTIONS = [
    ("知识 / 推理考试", "Knowledge / reasoning exams", [(None, None, ["MMLU-Pro", "MMLU", "MMLU-Redux", "GPQA-Diamond", "AGIEval-en", "SuperGPQA"])]),
    ("数学", "Math", [(None, None, ["MATH-500", "Minerva MATH", "GSM8K", "AIME 2024", "AIME 2025", "HMMT Feb 2025"])]),
    ("代码", "Code", [(None, None, ["LiveCodeBench", "HumanEval", "MBPP"])]),
    ("世界知识问答", "World-knowledge QA",
     [(None, None, ["TriviaQA", "Natural Questions", "PopQA", "SimpleQA", "SimpleQA-Verified", "FACTS-parametric"])]),
    ("中文", "Chinese", [(None, None, ["Global-MMLU-zh", "INCLUDE-zh", "MGSM-zh"])]),
    ("常识 / 阅读", "Commonsense / reading",
     [(None, None, ["HellaSwag", "WinoGrande", "ARC-Challenge", "CommonsenseQA", "PIQA", "RACE-high", "OpenBookQA", "Social IQa", "TruthfulQA"])]),
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
           "s_h": "重点指标随训练步数的变化", "s_run": "{run}（一个点一个 checkpoint）", "s_step": "第 {s} 步（{p}）",
           "s_h_all": "全部指标随训练步数的变化（{t} 个任务，{kt} 类，类内按重要性排序）",
           "h_all": "全部 benchmark（分类，按重要性排序）",
           "all_note": "全量测评的全部 {t} 个任务，每个任务取一个主分数，分 {kt} 类：{parts}；类别和类内的顺序 = 对预训练 base 模型的重要性（题量大、噪声小、"
                       "对决策有用的在前，已饱和或接近 0 分的在后）。各任务的其它分数（共 {rm} 个指标）在下面的“全量测评结果”里，同样的分类和顺序。",
           "head": ["模型", "大小"], "post": "后训练",
           "flag": "- = 没有有效分数（不是 0 分），悬停看原因：该任务有 ≥ 20% 的回答是空的（分数无效，已撤下；悬停看比例和原始分），或者还没出分。"
                   "带虚线下划线的分数 = 空回答占 5–20%，分数保留，悬停看比例。未作答率 / not_attempted 是比例，不是分数，始终显示。",
           "empty": "空回答 {p:.0f}%，分数无效（原始分 {v}）", "warn": "空回答 {p:.0f}%（低于 20%，分数保留）", "missing": "没有分数",
           # the scoreboard (ce_board) and the folded chapters, 11-16 review
           "b_h": "最新成绩 vs {base}", "b_ht": "各任务最新成绩 vs {base}（表）",
           "b_note": "每个任务取 59B 最新已出分的 checkpoint（若不是表里最新的一步，分数旁注明步数）。Δ = 59B − {base}，分数 ×100。▲ / ▼ = 差距超过 2 倍标准误"
                     "（两边 stderr 合成：2·√(se₁² + se₂²)；lm_eval 没给 stderr 的 SimpleQA / SimpleQA-Verified / FACTS 按二项分布 √(p(1−p)/n) 估），≈ = 在噪声内；"
                     "悬停 Δ 看这个阈值。条形：向右 = 59B 更高；实色 = 超出噪声，灰色 = 噪声内。“历史最好”是带噪声读数里的最大值，偏高。",
           "b_head": ["任务", "指标", "59B 最新", "Δ vs Base", "", "历史最好"],
           "b_at": "第 {s} 步（{p}）", "t_ck": "最新出齐的 checkpoint", "t_ck_s": "第 {s} 步（{p}）：{n} / {t} 个任务有分（其余为无效分）",
           "t_ck_run": "。评测中：{runs}", "t_ck_runi": "第 {s} 步（{p}）已出 {n} / {t}", "t_ck_none": "还没有出齐的 checkpoint",
           "t_win": "超出噪声地领先 / 落后 {base}", "t_win_s": "其余 {eq} 个任务在 ±2σ 内；只看正负号：{pos} / {n} 个高于 Base。各任务取自己最新的 checkpoint（{lo}–{hi} 步）",
           "t_cat": "平均 Δ vs {base}（{n} 个任务）", "fold_h": "▸ ", "steps_h": "各任务随训练步数的变化",
           "s_note": "一个点一个 checkpoint（越新的点颜色越接近文字色、越醒目），虚线 = 实验列表里勾选的 baseline；竖虚线 = {p2}换成 phase 2 数据。标题里的数 = 最新一点的分数和它与 {base} 的差。"},
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
           "s_h": "Key metrics vs. training step", "s_run": "{run} (one point per checkpoint)", "s_step": "step {s} ({p})",
           "s_h_all": "All metrics vs. training step ({t} tasks, {kt} categories, by importance inside each)",
           "h_all": "All benchmarks (by category, in order of importance)",
           "all_note": "All {t} tasks of the full evaluation, one headline metric each, in {kt} categories: {parts}. Categories and the tasks "
                       "inside them are ordered by importance for a pretraining base model (large, low-noise, decision-relevant first; saturated or "
                       "near-zero last). The other scores of the tasks ({rm} metrics in all) are in \"Full evaluation results\" below, same order.",
           "head": ["model", "size"], "post": "post-trained",
           "flag": "- = no valid score (not a zero), hover for the reason: >= 20 % of the task's answers are empty (score invalid, withdrawn; "
                   "hover for the share and the raw score), or the score is not in yet. A score with a dotted underline = 5-20 % empty answers, "
                   "kept; hover for the share. The no-answer / not_attempted rates are rates, not scores, and always shown.",
           "empty": "{p:.0f}% empty answers, score invalid (raw score {v})", "warn": "{p:.0f}% empty answers (below 20 %, score kept)",
           "missing": "no score",
           "b_h": "Latest scores vs {base}", "b_ht": "Latest score per task vs {base} (table)",
           "b_note": "Each task at the newest 59B checkpoint that has its score (the step is noted when it is not the newest one). Δ = 59B − {base}, "
                     "scores ×100. ▲ / ▼ = the gap exceeds 2 standard errors of the difference (2·√(se₁² + se₂²) from both stderrs), ≈ = within noise; "
                     "SimpleQA / SimpleQA-Verified / FACTS have no stderr from lm_eval: binomial √(p(1−p)/n). Hover the Δ for the threshold. Bar: right = 59B "
                     "higher; solid = beyond noise, grey = within noise. \"Best so far\" is the maximum of noisy readings, biased upward.",
           "b_head": ["task", "metric", "59B latest", "Δ vs Base", "", "best so far"],
           "b_at": "step {s} ({p})", "t_ck": "Latest complete checkpoint", "t_ck_s": "step {s} ({p}): {n} / {t} tasks with a valid score",
           "t_ck_run": ". Being evaluated: {runs}", "t_ck_runi": "step {s} ({p}) {n} / {t}", "t_ck_none": "no complete checkpoint yet",
           "t_win": "Ahead / behind {base} beyond noise", "t_win_s": "the other {eq} tasks within ±2σ; by sign only: {pos} / {n} above Base. Each task at its own latest checkpoint (steps {lo}–{hi})",
           "t_cat": "Mean Δ vs {base} ({n} tasks)", "fold_h": "▸ ", "steps_h": "Every task vs. training step",
           "s_note": "One point per checkpoint (newer = closer to the text colour, more prominent); dashed = the baselines ticked in the experiment list; the vertical dotted line = the "
                     "switch to the phase-2 data at {p2}. The number in each title = the latest score and its gap to {base}."},
}
RCSS = ("<style>.nt-tab th{white-space:normal;vertical-align:bottom;min-width:64px}.nt-tab td.nt-flag{color:var(--muted);font-style:italic}"
        ".nt-tab td.nt-warn{text-decoration:underline dotted;text-underline-offset:3px}.nt-fn{margin:8px 0 0}.nt-tab{margin-top:10px}"
        ".nt-part+.nt-part{margin-top:26px;padding-top:14px;border-top:1px solid var(--grid)}"
        ".nt-c.key{background:color-mix(in srgb,var(--text-primary) 11%,transparent);color:var(--text-primary);font-weight:600}"
        ".nt-m b{color:var(--text-primary);font-weight:600}"
        ".tk-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px 18px;margin:10px 0 4px}"   # user 11-15: "改成每行4个吧，这样能放大一点"
        "@media (max-width:900px){.tk-grid{grid-template-columns:repeat(2,minmax(0,1fr))}}@media (max-width:560px){.tk-grid{grid-template-columns:1fr}}"
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
    pkeys = [k for _, _, _, ks in PROD_TABS for k in ks]   # prod downstream: every task exactly once, with one of its report metrics
    assert sorted(a for a, _ in pkeys) == sorted(by), f"cosmos_eval: PROD_TABS out of sync with the task list: {set(by) ^ {a for a, _ in pkeys}}"
    for alias, metric in pkeys:
        assert any((rm["alias"], rm["metric"]) == (alias, metric) for rm in by[alias]["report_metric"]), \
            f"cosmos_eval: PROD_TABS entry {alias} {metric} is not a report metric of the task list"
    pplaced = [n for _, _, parts in PROD_SECTIONS for _, _, names in parts for n in names]
    assert sorted(pplaced) == sorted(b[1] for b in BENCH), f"cosmos_eval: PROD_SECTIONS out of sync with BENCH: {set(pplaced) ^ {b[1] for b in BENCH}}"
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


def ce_scores(lang, esc, ref_models, ref_size, focus=False, colors=None, legend=True, opt=None, tabs=None, sections=None, curve=True, fold=False):
    """HTML of "全量测评重点" (focus=True: one card, a tab per FOCUS_TABS entry) or "全量测评结果" (a tab per SECTIONS entry). Rows = the models
    of ref_models [(code, display name, _)] that have a results_xx/cosmos_eval JSON, in that order.
    curve=False: tables only (prod downstream since the 11-16 review: its scoreboard and step charts carry the picture); fold=True: the
    chapter is a closed <details> (its h2 is the summary; the md mirror still reads every table).
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
        cc = [(l1, l2, g) for _, g, _, l1, l2, ok in cols if ok]
        if not curve:
            return f"{title}{tbl}" + (f'<p class="note nt-fn">{esc(tx["flag"])}</p>' if flagged else "")
        chart = _curve(cc, models, esc, ctx, colors=colors, legend=legend)
        # opt = codes of the optional (unticked by default) baselines: the min-max scale of the default view leaves them out (user 11-16
        # "可选的，不是default的指标"), so the curve is drawn twice: data-cv="0" without them (shown while none of them is ticked), data-cv="1"
        # over every model (shown once one is ticked: the page's CSS, :has() on the tick boxes that carry data-opt)
        if opt and any(c in opt for c, _ in models):
            chart = (f'<div class="cv-v" data-cv="0">{_curve(cc, [m for m in models if m[0] not in opt], esc, ctx, colors=colors, legend=legend)}</div>'
                     f'<div class="cv-v" data-cv="1">{chart}</div>')
        return f"{title}{chart}{tbl}" + (f'<p class="note nt-fn">{esc(tx["flag"])}</p>' if flagged else "")

    # tabs / sections (prod downstream, 11-16): every task in PROD_TABS / PROD_SECTIONS order instead of the key tier / SECTIONS
    ftabs, secs = tabs or FOCUS_TABS, sections or SECTIONS
    if focus:
        allc = dict(cols_of([b[1] for b in BENCH]))
        tabs = [(k, nz if zh else ne, pane([allc[key] for key in keys])) for k, nz, ne, keys in ftabs]
        nt, nm = len(cfg), sum(len(t["report_metric"]) for t in cfg)
        parts = tx["focus_sep"].join(tx["focus_part"].format(name=nz if zh else ne, n=len(keys)) for _, nz, ne, keys in ftabs)
        if ftabs is FOCUS_TABS:
            pre, h = "cf", tx["h_focus"]
            note = f'<p class="note">{esc(tx["focus_note"].format(t=nt, kn=len(KEY), kt=len(FOCUS_TABS), parts=parts, rt=nt - len(KEY), rm=nm - len(KEY)))}</p>'
        else:
            pre, h = "cf", tx["h_all"]
            note = f'<p class="note">{esc(tx["all_note"].format(t=nt, kt=len(ftabs), parts=parts, rm=nm - nt))}</p>'
    else:
        tabs = []
        for i, (nz, ne, parts) in enumerate(secs):
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
    if fold:
        return (RCSS + KCSS + BCSS + f'<details class="fold"><summary><h2>{esc(h)}</h2></summary>{note}'
                f'<div class="card"><div class="rtoggle" style="flex-wrap:wrap">{btns}</div>{panes}</div></details>')
    return RCSS + KCSS + f'<h2>{esc(h)}</h2>{note}<div class="card"><div class="rtoggle" style="flex-wrap:wrap">{btns}</div>{panes}</div>'


BCSS = ("<style>.b-tab td,.b-tab th{padding:4px 9px}.b-tab tr.b-cat td{font-weight:600;color:var(--text-primary);padding-top:14px;"
        "border-bottom:1px solid var(--axis)}.b-tab td.b-d{font-weight:600}.b-tab td.b-q{color:var(--text-secondary)}.b-tab .b-s{color:var(--muted);"
        "font-size:11px;margin-left:5px;font-weight:400}.b-tab td.b-bar{padding:0 6px;width:130px}.b-tab td.b-bar svg{display:block}"
        ".b-tab td.b-n{color:var(--muted);font-weight:400}.tile .dl{font-size:13px;font-weight:600;margin:2px 0}"
        ".tile .dl span{font-weight:400;color:var(--text-secondary)}.lead{font-size:15px;margin:4px 0 12px}"
        "details.fold>summary{cursor:pointer;list-style:none}details.fold>summary::-webkit-details-marker{display:none}"
        "details.fold>summary h2{display:inline-block}details.fold>summary h2::before{content:'▸ ';color:var(--muted)}"
        "details.fold[open]>summary h2::before{content:'▾ '}</style>")


def ce_board(lang, esc, steps, base, base_name, extra=(), names=None, split=False):
    """HTML of the first block of the prod downstream tab (11-16 review of "主要两个可视化就是 val loss 和 downstream 放在最前面"): three tiles
    and one table, a row per task of PROD_TABS (by category, in its order of importance): the tracked run's latest score (the newest
    checkpoint with a valid score of that task), Δ vs the baseline `base` with a noise call (2 standard errors of the difference, from
    the stderr lm_eval writes), a diverging bar of Δ, the best score so far, and the baselines base + extra as columns.
    steps = [(code, step)] of the tracked run; names = {code: display name} of the baseline columns.
    split=True (user 11-16 18:1x "overview 大字总结放在图的前面 表格还是放后面"): returns (tiles html, table html) to place apart."""
    res = _load()
    steps = sorted(((c, s) for c, s in steps if c in res), key=lambda x: x[1])
    if not os.path.exists(CFG) or not steps or base not in res:
        return ""
    tx, zh = TX[lang], lang == "zh"
    _cfg()
    names = names or {}
    total = (res[steps[0][0]].get("track") or {}).get("total_steps") or steps[-1][1]
    cols = [base] + [c for c in extra if c in res and c != base]
    val, se, empty, ndocs, pend = {}, {}, {}, {}, {}
    for c in [c for c, _ in steps] + cols:
        for ts in res[c]["groups"].values():
            for t in ts:
                empty[(c, t["task"])] = t.get("empty_frac")
                ndocs[(c, t["task"])] = t.get("n_docs")
                if t.get("pending"):
                    pend[(c, t["task"])] = t["pending"]
                for m in t["metrics"]:
                    val[(c, m["alias"], m["metric"])] = m["value"]
                    se[(c, m["alias"], m["metric"])] = m.get("stderr")

    def get(c, alias, metric):                             # (score x100, stderr x100) or (None, None): no score, or withdrawn (as the tables)
        v, e = val.get((c, alias, metric)), empty.get((c, alias))
        if not isinstance(v, (int, float)) or (isinstance(e, (int, float)) and e >= EMPTY_FLAG):
            return None, None
        s, n = se.get((c, alias, metric)), ndocs.get((c, alias))
        if not isinstance(s, (int, float)) and isinstance(n, int) and n > 0 and 0 <= v <= 1:
            s = math.sqrt(v * (1 - v) / n)                 # no stderr from lm_eval (f1, judge_accuracy): binomial, an upper bound for f1
        return v * 100, (s * 100 if isinstance(s, (int, float)) else None)

    def why(c, alias, metric):                             # (title, warn class) of a baseline cell, as the tables mark them
        v, e = val.get((c, alias, metric)), empty.get((c, alias))
        if not isinstance(v, (int, float)):
            return tx.get({"mathfix": "pending"}.get(pend.get((c, alias)), pend.get((c, alias))), tx["missing"]), ""
        if isinstance(e, (int, float)) and e >= EMPTY_FLAG:
            return tx["empty"].format(p=e * 100, v=f"{v * 100:.1f}"), ""
        if isinstance(e, (int, float)) and EMPTY_NOTE <= e < EMPTY_FLAG:
            return tx["warn"].format(p=e * 100), " nt-warn"
        return "", ""

    pct = lambda s: f"{100 * s / total:.1f} %"
    at = lambda s: tx["b_at"].format(s=f"{s:,}", p=f"{100 * s / total:.1f} %")
    bname = {tn: (b[1], (vz if zh else ve)) for b in BENCH for tn, vz, ve in b[4]}
    newest = steps[-1][1]
    data, cats = [], []                                    # per row: (category, name, metric, latest step, value, delta, sig, thr, best)
    for key, nz, ne, keys in PROD_TABS:
        ds = []
        for alias, metric in keys:
            hist = [(s, *get(c, alias, metric)) for c, s in steps]
            hist = [h for h in hist if h[1] is not None]
            bn, var = bname[alias]
            bv, bse = get(base, alias, metric)
            row = {"cat": nz if zh else ne, "name": bn + (f" {var}" if var else ""), "metric": _mname(alias, metric, lang),
                   "base": [(get(c, alias, metric)[0], *why(c, alias, metric)) for c in cols], "s": None}
            if hist:
                s, v, vse = hist[-1]
                best = max(hist, key=lambda h: (h[1], h[0]))       # ties: the newest step
                d = v - bv if bv is not None else None
                thr = 2 * math.hypot(vse, bse) if d is not None and vse is not None and bse is not None else None
                row.update(s=s, v=v, d=d, thr=thr, sig=(abs(d) > thr) if thr is not None else None, best=best[:2])
                if d is not None:
                    ds.append(d)
            data.append(row)
        cats.append((nz if zh else ne, sum(ds) / len(ds) if ds else None))
    rows_d = [r for r in data if r["s"] is not None and r["d"] is not None]
    newest = max((r["s"] for r in data if r["s"] is not None), default=newest)   # the newest step that has a score (11-16 verify review)
    scale = max([abs(r["d"]) for r in rows_d] + [1.0])
    W, H, C = 120, 14, 60

    def bar(r):
        if r["s"] is None or r["d"] is None:
            return '<td class="b-bar"></td>'
        x = r["d"] / scale * (C - 2)
        ink = "var(--text-primary)" if r["sig"] else "var(--muted)"
        return (f'<td class="b-bar"><svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" aria-hidden="true">'
                f'<line x1="{C}" x2="{C}" y1="0" y2="{H}" stroke="var(--axis)"/>'
                f'<rect x="{min(C, C + x):.1f}" y="3" width="{max(abs(x), 1):.1f}" height="{H - 6}" rx="2" fill="{ink}"/></svg></td>')

    head = [esc(h) for h in tx["b_head"]] + [esc(names.get(c, c)) for c in cols]
    ncol = len(head)
    body, cur = [], None
    for r in data:
        if r["cat"] != cur:
            cur = r["cat"]
            body.append(f'<tr class="b-cat"><td class="l" colspan="{ncol}">{esc(cur)}</td></tr>')
        bcells = "".join(f'<td class="b-q{wc}"{f" title={chr(34)}{esc(ti)}{chr(34)}" if ti else ""}>{x:.1f}</td>' if x is not None
                         else f'<td class="b-q nt-flag"{f" title={chr(34)}{esc(ti)}{chr(34)}" if ti else ""}>-</td>' for x, ti, wc in r["base"])
        if r["s"] is None:
            body.append(f'<tr><td class="l">{esc(r["name"])}</td><td class="l nt-d">{esc(r["metric"])}</td><td>-</td><td>-</td>'
                        f'<td class="b-bar"></td><td>-</td>{bcells}</tr>')
            continue
        sv = f'{r["v"]:.1f}' + (f'<span class="b-s"> {esc(at(r["s"]))}</span>' if r["s"] != newest else "")
        if r["d"] is None:
            dc = "<td>-</td>"
        else:
            mark = " ?" if r["sig"] is None else (" ▲" if r["sig"] and r["d"] > 0 else " ▼" if r["sig"] else " ≈")
            tip = f' title="2σ = {r["thr"]:.1f}"' if r["thr"] is not None else ""
            dc = f'<td class="b-d{"" if r["sig"] else " b-n"}"{tip}>{r["d"]:+.1f}{mark}</td>'
        bs, bv_ = r["best"]
        best = f"{bv_:.1f}" + (f'<span class="b-s"> {esc(at(bs))}</span>' if bs != r["s"] else "")
        body.append(f'<tr><td class="l">{esc(r["name"])}</td><td class="l nt-d">{esc(r["metric"])}</td><td>{sv}</td>{dc}{bar(r)}<td>{best}</td>{bcells}</tr>')
    table = (f'<div style="overflow-x:auto"><table class="b-tab" style="display:table;width:100%"><thead><tr>'
             + "".join(f'<th class="l">{h}</th>' if i < 2 else f"<th>{h}</th>" for i, h in enumerate(head)) + f'</tr></thead><tbody>{"".join(body)}</tbody></table></div>')
    # tiles: the latest complete checkpoint (<= 4 of the 66 metrics missing), how many tasks lead the baseline, the mean gap by category
    nt = sum(len(ks) for _, _, _, ks in PROD_TABS)
    # complete = no headline metric still pending (a withdrawn score counts as done); the others are listed as being evaluated
    heads = {f"{a} {m}" for _, _, _, ks in PROD_TABS for a, m in ks}
    nsc = lambda c: sum(get(c, a, m)[0] is not None for _, _, _, ks in PROD_TABS for a, m in ks)
    done = [(c, s) for c, s in steps if not heads & set(res[c].get("missing") or [])]
    k1 = tx["t_ck_s"].format(s=f"{done[-1][1]:,}", p=pct(done[-1][1]), n=nsc(done[-1][0]), t=nt) if done else tx["t_ck_none"]
    runs = [(c, s) for c, s in steps if (not done or s > done[-1][1]) and heads & set(res[c].get("missing") or [])]
    if runs:
        k1 += tx["t_ck_run"].format(runs=" · ".join(tx["t_ck_runi"].format(s=f"{s:,}", p=pct(s), n=nsc(c), t=nt) for c, s in runs))
    up = sum(1 for r in rows_d if r["sig"] and r["d"] > 0)
    dn = sum(1 for r in rows_d if r["sig"] and r["d"] < 0)
    eq = sum(1 for r in rows_d if r["sig"] is False)
    pos = sum(1 for r in rows_d if r["d"] > 0)
    mean = sum(r["d"] for r in rows_d) / len(rows_d) if rows_d else 0.0
    ss = [r["s"] for r in rows_d]
    sep = " · "
    tiles = [(tx["t_ck"], pct(done[-1][1]) if done else "-", k1),
             (tx["t_win"].format(base=base_name), f"{up} ▲ · {dn} ▼",
              tx["t_win_s"].format(eq=eq, pos=pos, n=len(rows_d), lo=f"{min(ss):,}" if ss else "-", hi=f"{max(ss):,}" if ss else "-")
              + ("" if eq + up + dn == len(rows_d) else f" (? {len(rows_d) - eq - up - dn})")),
             (tx["t_cat"].format(base=base_name, n=len(rows_d)), f"{mean:+.1f}", sep.join(f"{n} {m:+.1f}" for n, m in cats if m is not None))]
    th = '<div class="tiles">' + "".join(f'<div class="tile"><div class="k">{esc(k)}</div><div class="v">{esc(v)}</div><div class="s">{esc(s)}</div></div>'
                                         for k, v, s in tiles) + "</div>"
    tbl = f'<div class="card"><p class="note">{esc(tx["b_note"].format(base=base_name))}</p>{table}</div>'
    if split:
        return (RCSS + BCSS + f'<h2>{esc(tx["b_h"].format(base=base_name))}</h2>{th}',
                RCSS + BCSS + f'<h2>{esc(tx["b_ht"].format(base=base_name))}</h2>{tbl}')
    return RCSS + BCSS + f'<h2>{esc(tx["b_h"].format(base=base_name))}</h2>{th}{tbl}'


# the 59B checkpoint averages (collect_cosmos_eval.py: MERGE59_<wave>_<recipe>_iter<newest step>): a chapter of their own in the prod
# downstream tab, compared with the same baseline as the run's own scoreboard (user 11-17: "能否把 [59B avg ...] 在 prod downstream 里面，
# 但开一个 section，baseline 还是 qwen3-30b-base")
AVG_ORDER = ["U2", "U3", "U5", "U8", "LIN5", "EXP6", "EXPr05", "POLYH", "POLY", "DU3", "DU5", "DU10", "DLIN5", "DEXPT500", "DEXPT1000"]
AVG_CSS = ("<style>.b-tab td.av-c{line-height:1.25;vertical-align:top}.b-tab td.av-c .b-s{display:block;margin-left:0}"
           ".b-tab td.av-c .av-sig{color:var(--text-primary);font-weight:600}.b-tab td.av-top{font-weight:600}"
           ".b-tab tr.av-plain td{border-bottom:1px solid var(--axis)}</style>")
AVG_TX = {
    "zh": {"h": "59B checkpoint 平均 vs {base}",
           "note": "把 59B 生产 run 的几个 checkpoint 按权重平均成一个模型，再用同一套评测。每一组用它最新的一步命名（M94 = 最新的是第 94,000 步，"
                   "M97 = 最终的第 97,330 步）。分数 ×100，每个任务取主指标（和上面成绩表相同的 {t} 个）。Δ = 平均模型 − {base}；▲ / ▼ = 差距超过 2 倍"
                   "标准误，≈ = 在噪声以内。评测还没跑完的模型，平均 Δ 只算它已出分的任务，和别的行不能直接比（灰色）；“vs 未平均”只算两边都有分的任务。",
           "wave": "{w}：最新 checkpoint 第 {s} 步", "plain": "第 {s} 步（未平均）", "plain_d": "这一组最新的 checkpoint 本身，作对照",
           "head": ["模型", "平均方式", "已出分", "有分任务", "平均 Δ vs Base", "▲", "▼", "平均 Δ vs 未平均"],
           "run": "评测还在跑", "own": "只算它已出分的 {n} 个任务", "pair": "只算两边都有分的 {n} 个任务",
           "h_tab": "各任务成绩：每格上面是分数，下面是 Δ vs Base；每行最高分加粗", "task": "任务", "metric": "指标", "raw": "未平均"},
    "en": {"h": "59B checkpoint averages vs {base}",
           "note": "Several checkpoints of the 59B production run averaged into one model with the given weights, evaluated on the same task "
                   "list. A group is named after its newest step (M94 = newest checkpoint 94,000, M97 = the final 97,330). Scores x100, the "
                   "headline metric of each task (the same {t} as the scoreboard above). Δ = averaged model − {base}; ▲ / ▼ = beyond 2 standard "
                   "errors, ≈ = within noise. For a model still being evaluated the mean Δ covers only its scored tasks and is not comparable "
                   "with the other rows (grey); \"vs un-averaged\" covers the tasks both have.",
           "wave": "{w}: newest checkpoint step {s}", "plain": "step {s} (not averaged)", "plain_d": "the newest checkpoint of the group itself, for reference",
           "head": ["model", "averaging", "scored", "tasks scored", "mean Δ vs Base", "▲", "▼", "mean Δ vs un-averaged"],
           "run": "the evaluation is still running", "own": "over its {n} scored tasks only", "pair": "over the {n} tasks both have",
           "h_tab": "Score per task: the score on top, Δ vs Base below; the best score of each row in bold", "task": "task", "metric": "metric", "raw": "not averaged"}}


def ce_avg(lang, esc, groups, base, base_name):
    """HTML of the chapter "59B checkpoint 平均 vs <base>" of the prod downstream tab: a summary table (one row per averaged model: metrics
    scored, mean Δ vs the baseline over the headline metrics of PROD_TABS, how many tasks are beyond 2 standard errors, mean Δ vs the
    un-averaged checkpoint of the same step) and a table of every task per group.
    groups = [(wave, newest step, code of the un-averaged checkpoint of that step or None, [(code, recipe, description)])].
    Scores are read as ce_board reads them (withdrawn at >= EMPTY_FLAG empty answers, binomial stderr where lm_eval wrote none)."""
    res = _load()
    groups = [(w, s, p if p in res else None, sorted(((c, r, d) for c, r, d in ms if c in res),
                                                        key=lambda m: (AVG_ORDER.index(m[1]) if m[1] in AVG_ORDER else len(AVG_ORDER), m[1])))
              for w, s, p, ms in groups]
    groups = [g for g in groups if g[3]]
    if not os.path.exists(CFG) or not groups or base not in res:
        return ""
    tx, ax, zh = TX[lang], AVG_TX[lang], lang == "zh"
    _cfg()
    val, se, empty, ndocs, pend = {}, {}, {}, {}, {}
    for c in {base} | {c for _, _, p, ms in groups for c in ([p] if p else []) + [m[0] for m in ms]}:
        for ts in res[c]["groups"].values():
            for t in ts:
                empty[(c, t["task"])] = t.get("empty_frac")
                ndocs[(c, t["task"])] = t.get("n_docs")
                if t.get("pending"):
                    pend[(c, t["task"])] = t["pending"]
                for m in t["metrics"]:
                    val[(c, m["alias"], m["metric"])] = m["value"]
                    se[(c, m["alias"], m["metric"])] = m.get("stderr")

    def get(c, alias, metric):                             # (score x100, stderr x100) or (None, None): no score, or withdrawn
        v, e = val.get((c, alias, metric)), empty.get((c, alias))
        if not isinstance(v, (int, float)) or (isinstance(e, (int, float)) and e >= EMPTY_FLAG):
            return None, None
        s, n = se.get((c, alias, metric)), ndocs.get((c, alias))
        if not isinstance(s, (int, float)) and isinstance(n, int) and n > 0 and 0 <= v <= 1:
            s = math.sqrt(v * (1 - v) / n)
        return v * 100, (s * 100 if isinstance(s, (int, float)) else None)

    def why(c, alias, metric):                             # hover text of a cell without a score / with a marked score
        v, e = val.get((c, alias, metric)), empty.get((c, alias))
        if not isinstance(v, (int, float)):
            return tx.get({"mathfix": "pending"}.get(pend.get((c, alias)), pend.get((c, alias))), tx["missing"])
        if isinstance(e, (int, float)) and e >= EMPTY_FLAG:
            return tx["empty"].format(p=e * 100, v=f"{v * 100:.1f}")
        if isinstance(e, (int, float)) and EMPTY_NOTE <= e < EMPTY_FLAG:
            return tx["warn"].format(p=e * 100)
        return ""

    cats = [(nz if zh else ne, ks) for _, nz, ne, ks in PROD_TABS]
    keys = [k for _, ks in cats for k in ks]
    nt = len(keys)
    bname = {tn: (b[1], (vz if zh else ve)) for b in BENCH for tn, vz, ve in b[4]}

    def delta(c, ref, alias, metric):                      # (Δ, beyond 2 SE: True / False / None = no stderr) or None when one side has no score
        (v, vs), (b, bs) = get(c, alias, metric), get(ref, alias, metric)
        if v is None or b is None:
            return None
        d = v - b
        return d, ((abs(d) > 2 * math.hypot(vs, bs)) if vs is not None and bs is not None else None)

    def stat(c, ref):                                      # over the tasks both have: (n, mean Δ, tasks above / below beyond 2 SE)
        ds = [x for x in (delta(c, ref, a, m) for a, m in keys) if x is not None]
        return (len(ds), sum(d for d, _ in ds) / len(ds) if ds else None,
                sum(1 for d, sig in ds if sig and d > 0), sum(1 for d, sig in ds if sig and d < 0))

    def scored(c):
        d = res[c]
        n = sum(isinstance(m.get("value"), (int, float)) for ts in d["groups"].values() for t in ts for m in t["metrics"])
        run = f' title="{esc(ax["run"])}" style="text-decoration:underline dotted;text-underline-offset:3px"' if d.get("running") else ""
        return f"<td{run}>{n} / {d.get('n_metrics') or '-'}</td>"

    ncol = len(ax["head"])
    srows, panes = [], []
    for w, s, plain, ms in groups:
        srows.append(f'<tr class="b-cat"><td class="l" colspan="{ncol}">{esc(ax["wave"].format(w=w, s=f"{s:,}"))}</td></tr>')
        members = ([(plain, ax["plain"].format(s=f"{s:,}"), ax["plain_d"], True)] if plain else []) + [(c, f"{w}_{r}", d, False) for c, r, d in ms]
        for c, label, desc, is_plain in members:
            k, mean, up, dn = stat(c, base)
            part = k < nt
            mcell = "<td>-</td>" if mean is None else (f'<td class="b-d{" b-n" if part else ""}"' + (f' title="{esc(ax["own"].format(n=k))}"' if part else "")
                                                       + f">{mean:+.2f}</td>")
            if is_plain or not plain:
                pcell = "<td></td>" if is_plain else "<td>-</td>"
            else:
                pk, pmean, pup, pdn = stat(c, plain)
                pcell = "<td>-</td>" if pmean is None else (f'<td class="b-d"' + (f' title="{esc(ax["pair"].format(n=pk))}"' if pk < nt else "")
                                                             + f'>{pmean:+.2f}<span class="b-s">{pup} ▲ · {pdn} ▼' + (f" · n {pk}" if pk < nt else "") + "</span></td>")
            srows.append(f'<tr{" class=av-plain" if is_plain else ""}><td class="l" style="white-space:nowrap">{esc(label)}</td>'
                         f'<td class="l nt-d" style="white-space:normal">{esc(desc)}</td>{scored(c)}<td{" class=b-n" if part else ""}>{k} / {nt}</td>'
                         f'{mcell}<td>{up}</td><td>{dn}</td>{pcell}</tr>')
        # every task of the group: baseline, the un-averaged checkpoint, then the averages
        cols = ([(plain, ax["raw"], ax["plain"].format(s=f"{s:,}"))] if plain else []) + [(c, r, d) for c, r, d in ms]
        head = (f'<th class="l">{esc(ax["task"])}</th><th class="l">{esc(ax["metric"])}</th><th>{esc(base_name)}</th>'
                + "".join(f'<th title="{esc(d)}">{esc(n)}</th>' for _, n, d in cols))
        body = []
        for cat, ks in cats:
            body.append(f'<tr class="b-cat"><td class="l" colspan="{3 + len(cols)}">{esc(cat)}</td></tr>')
            for alias, metric in ks:
                bn, var = bname[alias]
                bv = get(base, alias, metric)[0]
                bt = why(base, alias, metric)
                tds = [(f'<td class="b-q"' + (f' title="{esc(bt)}"' if bt else "") + f">{bv:.1f}</td>") if bv is not None
                       else f'<td class="b-q nt-flag" title="{esc(bt)}">-</td>']
                vals = [get(c, alias, metric)[0] for c, _, _ in cols]
                top = max((v for v in vals if v is not None), default=None)
                for (c, _, _), v in zip(cols, vals):
                    ti = why(c, alias, metric)
                    if v is None:
                        tds.append(f'<td class="nt-flag" title="{esc(ti)}">-</td>')
                        continue
                    x = delta(c, base, alias, metric)
                    if x is None:
                        sub = ""
                    else:
                        d, sig = x
                        mark = " ?" if sig is None else (" ▲" if sig and d > 0 else " ▼" if sig else " ≈")
                        sub = f'<span class="b-s{" av-sig" if sig else ""}">{d:+.1f}{mark}</span>'
                    tds.append(f'<td class="av-c{" av-top" if top is not None and v == top else ""}"' + (f' title="{esc(ti)}"' if ti else "") + f">{v:.1f}{sub}</td>")
                body.append(f'<tr><td class="l">{esc(bn + (f" {var}" if var else ""))}</td><td class="l nt-d">{esc(_mname(alias, metric, lang))}</td>{"".join(tds)}</tr>')
        panes.append((w, ax["wave"].format(w=w, s=f"{s:,}"),
                      f'<div style="overflow-x:auto"><table class="b-tab" style="display:table;width:100%"><thead><tr>{head}</tr></thead>'
                      f'<tbody>{"".join(body)}</tbody></table></div>'))
    shead = "".join(f'<th class="l">{esc(h)}</th>' if i < 2 else f"<th>{esc(h)}</th>" for i, h in enumerate(ax["head"]))
    summary = (f'<div class="card"><div style="overflow-x:auto"><table class="b-tab" style="display:table;width:100%"><thead><tr>{shead}</tr></thead>'
               f'<tbody>{"".join(srows)}</tbody></table></div></div>')
    btns = ('<div class="rtoggle" style="flex-wrap:wrap">' + "".join(f'<button data-r="av_{esc(w)}" class="{"on" if i == 0 else ""}">{esc(lab)}</button>'
                                                                    for i, (w, lab, _) in enumerate(panes)) + "</div>") if len(panes) > 1 else ""
    detail = (f'<div class="card"><p class="note" style="margin-top:0">{esc(ax["h_tab"])}</p>{btns}'
              + "".join(f'<div data-ref="av_{esc(w)}"{"" if i == 0 else " hidden"}>{tbl}</div>' for i, (w, _, tbl) in enumerate(panes)) + "</div>")
    return (RCSS + BCSS + AVG_CSS + f'<h2>{esc(ax["h"].format(base=base_name))}</h2>'
            f'<p class="note">{esc(ax["note"].format(base=base_name, t=nt))}</p>{summary}{detail}')


def _ticks(lo, hi, n=3):
    """nice y ticks covering [lo, hi]: (first tick, step, count)."""
    span = max(hi - lo, 1e-9)
    step = next((c for c in (0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50) if span / c <= n), 50)
    a = math.floor(lo / step + 1e-9) * step
    k = int(math.ceil((hi - a) / step - 1e-9))
    return a, step, max(k, 1)


MINI_M = (30, 14, 10, 22)                                  # margins of the small step charts (left, right, top, bottom)


def _mini(esc, title, sub, pts, refs, xmax, tipx, w=300, h=178, fit=None, phase2=None):
    """one small chart: pts = [(step, value 0-100, colour var of the checkpoint)] of the run, joined by a neutral line; refs = [(name,
    value, colour, code)] as dashed horizontal lines (data-dm = code: the tick boxes of "实验列表" show / hide them). fit = the reference
    values the y range must hold (default: all); a reference outside the range is not drawn. phase2 = the step where the data blend
    switches (a dotted vertical rule). No hover layer: _mini_hits draws it once for every y-range drawing of the chart."""
    ml, mr, mt, mb = MINI_M
    vals = [v for _, v, _ in pts] + ([v for _, v, _, _ in refs] if fit is None else list(fit))
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
    tick = next(c for c in (2000, 5000, 10000, 20000, 50000) if xmax / c <= 5)
    for x in range(0, xmax + 1, tick):                     # x axis = training step, the whole run (fixed domain since the 11-16 review)
        o.append(f'<text x="{X(x):.1f}" y="{h - mb + 14}" text-anchor="middle" style="fill:var(--muted);font-size:10px">{f"{x // 1000}k" if x else "0"}</text>')
    o.append(f'<line x1="{ml}" x2="{w - mr}" y1="{h - mb}" y2="{h - mb}" stroke="var(--axis)"/>')
    if phase2 and 0 < phase2 < xmax:
        o.append(f'<line x1="{X(phase2):.1f}" x2="{X(phase2):.1f}" y1="{mt}" y2="{h - mb}" stroke="var(--muted)" stroke-width="1" stroke-dasharray="2 3"/>')
    refs = [r for r in refs if y0 <= r[1] <= y1]
    for _, v, c, rc in refs:                               # references: dashed, as in the prod val loss tab
        o.append(f'<line x1="{ml}" x2="{w - mr}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="var({c})" stroke-width="1.6" stroke-dasharray="5 4" data-dm="{esc(rc)}"/>')
    if len(pts) > 1:
        d = "M" + " L".join(f"{X(x):.1f} {Y(v):.1f}" for x, v, _ in pts)
        o.append(f'<path d="{d}" fill="none" stroke="var(--muted)" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
    for x, v, c in pts:                                    # each checkpoint in its own grey of "实验列表"
        o.append(f'<circle cx="{X(x):.1f}" cy="{Y(v):.1f}" r="4.5" fill="var({c})" stroke="var(--surface-1)" stroke-width="2"/>')
    x, v, _ = pts[-1]                                      # direct label: the latest value only, in text ink
    left = X(x) > w - mr - 34
    o.append(f'<text x="{X(x) + (-8 if left else 8):.1f}" y="{Y(v) + (-8 if left else 4):.1f}" text-anchor="{"end" if left else "start"}" '
             f'style="fill:var(--text-primary);font-size:11.5px;font-weight:600">{v:.1f}</text>')
    o.append("</svg>")
    return "".join(o)


def _mini_hits(esc, title, sub, pts, refs, xmax, tipx, w=300, h=178):
    """the hover layer of a small chart: one band per checkpoint, its tooltip lists every reference (data-dm rows follow the boxes)."""
    ml, mr, mt, mb = MINI_M
    X = lambda x: ml + x / xmax * (w - ml - mr)
    xs = [x for x, _, _ in pts]
    o = [f'<svg class="rk-hit" viewBox="0 0 {w} {h}" aria-hidden="true">']
    for i, (x, v, c) in enumerate(pts):
        a_ = ml if i == 0 else (X(xs[i - 1]) + X(x)) / 2
        b_ = w - mr if i == len(pts) - 1 else (X(x) + X(xs[i + 1])) / 2
        tip = (f"<b>{esc(title)} · {esc(sub)}</b><br><span style=\"color:var({c})\">●</span> {esc(tipx(x))}: {v:.1f}"
               + "".join(f'<span data-dm="{esc(code)}"><br><span style="color:var({rc})">●</span> {esc(n)}: {rv:.1f}</span>' for n, rv, rc, code in refs))
        o.append(f'<rect class="hit" x="{a_:.1f}" y="{mt}" width="{b_ - a_:.1f}" height="{h - mt - mb}" data-tip="{esc(tip)}"/>')
    o.append("</svg>")
    return "".join(o)


def ce_steps(lang, esc, run, steps, refs, colors, on=None, tabs=None):
    """HTML of the step-chart block (user 11-13: "上面还是把一个 blocked chart，是重点指标，根据 x-axis 训练时间变化标注出来"): one card, a tab
    per FOCUS_TABS / PROD_TABS entry, in each a grid of small charts: the headline score of one task against the training step, one point
    per evaluated checkpoint of the tracked run in that checkpoint's colour, the baselines as dashed lines. Charts only: the numbers are
    the rows of the tables below, so the grid is .nomd (not repeated in the md mirror).
    steps = [(code, step)], refs = [(code, name, colour var)] (the first = the default comparison: its gap is in each chart's title),
    colors = {code: colour var}. The baselines follow the tick boxes of "实验列表" (user 11-16: "用户自己选，default可以是base"; on = ticked by
    default): each chart is drawn once per y range a set of ticked baselines can give (rk_block), the page script shows the one of the
    current set, so a ticked baseline is always inside the chart; the hover layer is drawn once per chart."""
    res = _load()
    steps = sorted(((c, s) for c, s in steps if c in res), key=lambda x: x[1])
    refs = [(c, n, k) for c, n, k in refs if c in res]
    if not os.path.exists(CFG) or not steps:
        return ""
    tx, zh = TX[lang], lang == "zh"
    trk = res[steps[0][0]].get("track") or {}
    total = trk.get("total_steps") or max(s for _, s in steps)
    xmax = total                                           # the whole run (11-16 review: fixed x domain, the remaining steps visible)
    p2 = trk.get("phase1_end")
    empty, cells, ses = {}, {}, {}
    for c in [c for c, _ in steps] + [c for c, _, _ in refs]:
        for ts in res[c]["groups"].values():
            for t in ts:
                empty[(c, t["task"])] = t.get("empty_frac")
                for m in t["metrics"]:
                    cells[(c, m["alias"], m["metric"])] = m["value"]
                    ses[(c, m["alias"], m["metric"])] = m.get("stderr")

    def get(c, alias, metric):                             # score x100, or None: no score, or withdrawn for empty answers (as the tables)
        v, e = cells.get((c, alias, metric)), empty.get((c, alias))
        if not isinstance(v, (int, float)) or (isinstance(e, (int, float)) and e >= EMPTY_FLAG):
            return None
        return round(v * 100, 4)

    bname = {tn: (b[1], (vz if zh else ve)) for b in BENCH for tn, vz, ve in b[4]}
    on = set(on if on is not None else [c for c, _, _ in refs])
    onl = [c in on for c, _, _ in refs]
    tipx = lambda s: tx["s_step"].format(s=f"{s:,}", p=f"{100 * s / total:.1f} %")
    ftabs = tabs or FOCUS_TABS                             # prod downstream (11-16): PROD_TABS, every task
    tabs = []
    for key, nz, ne, keys in ftabs:
        grid = []
        for alias, metric in keys:
            bn, var = bname[alias]
            title, sub = bn + (f" {var}" if var else ""), _mname(alias, metric, lang)
            sc = [(c, s, get(c, alias, metric)) for c, s in steps]
            pts = [(s, v, colors.get(c, "--text-primary")) for c, s, v in sc if v is not None]
            rv = [get(c, alias, metric) for c, _, _ in refs]
            rf = [(n, rv[i], k, c) for i, (c, n, k) in enumerate(refs) if rv[i] is not None]
            head = ""
            if pts:                                        # the title carries the latest score and its gap to the first baseline (11-16 review)
                lc = [c for c, s, v in sc if v is not None][-1]
                head = f' <b style="margin-left:4px">{pts[-1][1]:.1f}</b>'
                if refs and rv[0] is not None:
                    d = pts[-1][1] - rv[0]
                    s1, s2 = ses.get((lc, alias, metric)), ses.get((refs[0][0], alias, metric))
                    thr = 2 * 100 * math.hypot(s1, s2) if isinstance(s1, (int, float)) and isinstance(s2, (int, float)) else None
                    sig = thr is not None and abs(d) > thr
                    mark = "" if thr is None else (" ▲" if sig and d > 0 else " ▼" if sig else " ≈")
                    head += (f' <span{"" if thr is None else f" title={chr(34)}2σ = {thr:.1f}{chr(34)}"} style="font-weight:{600 if sig else 400};'
                             f'color:var({"--text-primary" if sig else "--text-secondary"})">Δ {d:+.1f}{mark}</span>')
                svg = rk_block(esc, "dm", rv, onl, lambda fit: _mini(esc, title, sub, pts, rf, xmax, tipx, fit=fit, phase2=p2),
                               _mini_hits(esc, title, sub, pts, rf, xmax, tipx))
            else:
                svg = ""
            grid.append(f'<div class="tk-cell"><div class="tk-t">{esc(title)} <span>{esc(sub)}</span>{head}</div>{svg}</div>')
        tabs.append((key, nz if zh else ne, f'<div class="tk-grid nomd">{"".join(grid)}</div>'))
    last = colors.get(steps[-1][0], "--text-primary")
    leg = (f'<span><svg width="30" height="10"><line x1="1" x2="29" y1="5" y2="5" stroke="var(--muted)" stroke-width="2"/>'
           f'<circle cx="15" cy="5" r="4" fill="var({last})" stroke="var(--surface-1)" stroke-width="2"/></svg>{esc(tx["s_run"].format(run=run))}</span>'
           + "".join(f'<span data-dm="{esc(c)}"><svg width="30" height="10"><line x1="1" x2="29" y1="5" y2="5" stroke="var({k})" stroke-width="1.6" stroke-dasharray="5 4"/></svg>'
                     f'{esc(n)}</span>' for c, n, k in refs))
    btns = "".join(f'<button data-r="cs_{k}" class="{"on" if i == 0 else ""}">{esc(lab)}</button>' for i, (k, lab, _) in enumerate(tabs))
    panes = "".join(f'<div data-ref="cs_{k}"{"" if i == 0 else " hidden"}>{b}</div>' for i, (k, _, b) in enumerate(tabs))
    if ftabs is FOCUS_TABS:
        sh, note = tx["s_h"], ""
    else:
        sh = tx["steps_h"]
        note = f'<p class="note">{esc(tx["s_note"].format(p2=f"第 {p2:,} 步（{100 * p2 / total:.1f} %）" if p2 and zh else f"step {p2:,} ({100 * p2 / total:.1f} %)" if p2 else "-", base=refs[0][1] if refs else "-"))}</p>'
    return (RCSS + KCSS + RK_CSS + f'<h2>{esc(sh)}</h2>{note}<div class="card"><div class="rtoggle" style="flex-wrap:wrap">{btns}</div>'
            f'<div class="tk-leg nomd">{leg}</div><div class="tk-wrap">{panes}</div></div>')
