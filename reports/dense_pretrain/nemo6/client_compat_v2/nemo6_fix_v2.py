"""nemo6 client fixes v2 (2026-11-08, Runlong "发合理的代码测评"): three client-side defects of OUR pipeline (local-completions against a
served model, ADLR harness frozen @17abcb7) that Boxiang's in-process pipeline does not have. Found by the 11-08 read of his nemo6 code.
Active only through client_compat_v2/sitecustomize.py, i.e. for jobs launched with NEMO6_FIX=v2; client_compat/ (v1) is unchanged, so
runs started before 11-08 keep their behaviour.

1. Few-shot seed. This harness revision never calls Task.set_fewshot_seed (simple_evaluate takes fewshot_random_seed=1234 but only
   reports it), so every task that samples its few-shot examples drew them from an unseeded random.Random(): each run, and so each
   model, saw other examples (arc, commonsense_qa, Global-MMLU, hellaswag, INCLUDE, MBPP, MGSM order, NQ, PopQA, TriviaQA, WinoGrande;
   Boxiang measured popqa_5shot moving 1-3 points between two runs of one checkpoint). Seeded per task exactly as his
   nemo6_driver.py:149-167 does (NEMO6_FEWSHOT_SEED, default 1234; "none" = the harness's behaviour).
2. Generation cap. local-completions defaults max_gen_toks to 256 (openai_completions.py:77); tasks whose yaml sets no cap (MGSM x6, NQ,
   TriviaQA, PopQA) were cut at 256 tokens (our KD q3 MGSM answers end at exactly 256). Boxiang's default is 4096. NEMO6_MAX_GEN_TOKS
   (default 4096) becomes the default when the model_args give none; a task's own generation_kwargs.max_gen_toks still wins.
3. Log-likelihood context. With no max_length, _DEFAULT_MAX_LENGTH = 2048 silently left-truncates context+continuation (:188-193):
   INCLUDE-ja 14 %, Global-MMLU de/ko 4 %, ja 2.8 %, ARC 0.3 % of requests. NEMO6_MAX_LENGTH (the server's max sequence) becomes the
   default max_length. Generation prompts are not truncated by max_length in this class (raw contexts are sent), so only LL changes.
None of this changes a prompt template, a sampling setting a task sets, or a scoring rule.
"""
import os
import sys

import nemo6_patches  # v1 patches (client_compat/ is on PYTHONPATH after client_compat_v2/); provides the post-import hook table

FEWSHOT_SEED = os.environ.get("NEMO6_FEWSHOT_SEED", "1234")
MAX_GEN_TOKS = int(os.environ.get("NEMO6_MAX_GEN_TOKS", "4096"))
MAX_LENGTH = os.environ.get("NEMO6_MAX_LENGTH", "")


def _say(msg):
    print(f"[nemo6_fix_v2] {msg}", file=sys.stderr, flush=True)


def _patch_task(m):
    if FEWSHOT_SEED.lower() in ("", "none", "off"):
        return
    cls = m.ConfigurableTask
    if getattr(cls.__init__, "_nemo6_v2", False):
        return
    orig = cls.__init__

    def __init__(self, *a, **kw):
        orig(self, *a, **kw)
        if getattr(self, "fewshot_rnd", None) is not None:
            self.set_fewshot_seed(int(FEWSHOT_SEED))

    __init__._nemo6_v2 = True
    cls.__init__ = __init__
    _say(f"few-shot seed {FEWSHOT_SEED} for every task")


def _patch_completions(m):
    cls = m.OpenaiCompletionsLM
    if getattr(cls.__init__, "_nemo6_v2", False):
        return
    orig = cls.__init__

    def __init__(self, *a, **kw):
        kw.setdefault("max_gen_toks", MAX_GEN_TOKS)
        if MAX_LENGTH and kw.get("max_length") in (None, "", 0):
            kw["max_length"] = int(MAX_LENGTH)
        orig(self, *a, **kw)
        _say(f"local-completions max_gen_toks default {self._max_gen_toks}, max_length {self.max_length}")

    __init__._nemo6_v2 = True
    cls.__init__ = __init__


def _chain(name, fn):
    prev = nemo6_patches._POST.get(name)
    nemo6_patches._POST[name] = fn if prev is None else (lambda mod, p=prev: (p(mod), fn(mod)))
    mod = sys.modules.get(name)
    if mod is not None:  # already imported (should not happen from sitecustomize): patch now
        fn(mod)


# 4. (11-09) Short-answer QA cap + request timeout, both env-gated (set by nemo6_ladder.sbatch for jobs submitted after 11-09 07:3x):
#    NEMO6_SHORTQA_MAX_GEN=256: requests whose `until` is exactly ["\n", ".", ","] (trivia_qa / popqa_5shot / naturalqs_5shot) and that carry
#    no task cap get max_gen_toks 256 instead of the 4096 default. The answer ends at the first of those strings (a few words), so the text
#    is the same; but with 4096 the Megatron dynamic server admitted ~10-15x fewer concurrent requests (v2 trivia 100-218 min vs 3-20 min
#    with the v1 256 default, popqa likewise), and on the slower 512-expert models the batches outran the client timeout.
#    NEMO6_REQ_TIMEOUT=<s>: per-request HTTP timeout; the SDK default (600 s) made long batches time out and be re-sent while the server was
#    still generating them (2-4k "Retrying request" lines per task on LATENT2_E2X q3/q4, 11-09), i.e. duplicate load that never caught up.
SHORTQA_UNTIL = ["\n", ".", ","]
SHORTQA_MAX_GEN = int(os.environ.get("NEMO6_SHORTQA_MAX_GEN", "0") or 0)
REQ_TIMEOUT = float(os.environ.get("NEMO6_REQ_TIMEOUT", "0") or 0)


def _patch_completions_gen(m):
    cls = m.OpenaiCompletionsLM
    if not SHORTQA_MAX_GEN or getattr(cls.generate_until, "_nemo6_v2q", False):
        return
    orig = cls.generate_until

    def generate_until(self, requests, *a, **kw):
        n = 0
        for req in requests:
            gk = req.args[1] if len(req.args) > 1 and isinstance(req.args[1], dict) else None
            if gk is not None and "max_gen_toks" not in gk:
                until = gk.get("until")
                until = [until] if isinstance(until, str) else list(until or [])
                if until == SHORTQA_UNTIL:
                    gk["max_gen_toks"] = SHORTQA_MAX_GEN
                    n += 1
        if n:
            _say(f"short-answer QA: max_gen_toks {SHORTQA_MAX_GEN} on {n} requests (until {SHORTQA_UNTIL!r})")
        return orig(self, requests, *a, **kw)

    generate_until._nemo6_v2q = True
    cls.generate_until = generate_until


def _patch_openai_timeout(m):
    if not REQ_TIMEOUT:
        return
    for c in (m.AsyncCompletions, m.Completions):
        orig = c.create
        if getattr(orig, "_nemo6_v2t", False):
            continue

        def make(orig):
            def create(self, *a, **kw):
                kw.setdefault("timeout", REQ_TIMEOUT)
                return orig(self, *a, **kw)
            create._nemo6_v2t = True
            create._nemo6 = getattr(orig, "_nemo6", False)
            return create

        c.create = make(orig)
    _say(f"request timeout {REQ_TIMEOUT:.0f} s")


_chain("lm_eval.api.task", _patch_task)
_chain("lm_eval.models.openai_completions", _patch_completions)
_chain("lm_eval.models.openai_completions", _patch_completions_gen)
_chain("openai.resources.completions", _patch_openai_timeout)
