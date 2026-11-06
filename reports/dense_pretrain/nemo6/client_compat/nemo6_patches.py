"""Client-side operational patches for the nemo6 external-reference runs (reports_xx/nemo6; imported by sitecustomize.py).
None of them changes a prompt, a sampling setting or a scoring rule:

1. `stop` <= 4 strings per request. vLLM v0.31's OpenAI schema rejects longer lists (400 Bad Request: "List should have at most 4
   items"), which several nemo6 tasks send (7 `until` strings; ADLR's older vLLM images accepted them). The harness already cuts every
   completion at the first occurrence of ANY `until` string client-side (OpenaiCompletionsLM.generate_until), so only the first 4 go to
   the server: the text after the client's cut is identical, the server just generates a little further when a later string hits first.
2. LiveCodeBench code execution: scoring runs os.cpu_count() (= 144 on Grace) generated programs at once with no memory limit; the
   Qwen3-30B pilot (job 39048) OOM-killed the whole client step (915 GB RSS). Cap: NEMO6_LCB_WORKERS (default 32) workers and an
   RLIMIT_DATA of NEMO6_EXEC_DATA_GB (default 16) GB per executed program (contest solutions need <= ~1 GB).
"""
import importlib.abc
import os
import sys

MAX_STOP = 4


def _patch_openai(m):
    """post-import hook of openai.resources.completions (lazy: generated-code subprocesses never import openai)."""
    for cls in (m.AsyncCompletions, m.Completions):
        orig = cls.create
        if getattr(orig, "_nemo6", False):
            continue

        def make(orig):
            def create(self, *a, **kw):
                st = kw.get("stop")
                if isinstance(st, (list, tuple)) and len(st) > MAX_STOP:
                    kw["stop"] = list(st)[:MAX_STOP]
                return orig(self, *a, **kw)
            create._nemo6 = True
            return create

        cls.create = make(orig)


def _patch_lcb_metrics(m):
    n = int(os.environ.get("NEMO6_LCB_WORKERS", "32"))
    orig = m.codegen_metrics

    def codegen_metrics(*a, **kw):
        if int(kw.get("num_process_evaluate", 16)) > n:
            kw["num_process_evaluate"] = n
        return orig(*a, **kw)

    m.codegen_metrics = codegen_metrics


def _patch_lcb_exec(m):
    gb = float(os.environ.get("NEMO6_EXEC_DATA_GB", "16"))
    orig = m.reliability_guard

    def reliability_guard(maximum_memory_bytes=None):
        if maximum_memory_bytes is None and gb > 0:
            import resource
            lim = int(gb * 2 ** 30)
            resource.setrlimit(resource.RLIMIT_DATA, (lim, lim))
        return orig(maximum_memory_bytes)

    m.reliability_guard = reliability_guard


_POST = {"lm_eval.tasks.livecodebench.codegen.metrics": _patch_lcb_metrics,
         "lm_eval.tasks.livecodebench.codegen.exectute": _patch_lcb_exec}


class _PostImportHook(importlib.abc.MetaPathFinder):
    """Run _POST[name](module) right after the module executes (before any `from ... import name` binds the original)."""

    def find_spec(self, name, path, target=None):
        if name not in _POST:
            return None
        for f in sys.meta_path:
            if f is self or not hasattr(f, "find_spec"):
                continue
            spec = f.find_spec(name, path, target)
            if spec is not None:
                break
        else:
            return None
        orig_exec, post = spec.loader.exec_module, _POST[name]

        def exec_module(module):
            orig_exec(module)
            post(module)

        spec.loader.exec_module = exec_module
        return spec


def _patch_auto_tokenizer(_m):
    """Client tokenizer = the server's tokenizer (checked by tok_check.py before any task runs):
    NEMO6_TRUST_REMOTE_CODE=1: AutoTokenizer.from_pretrained defaults to trust_remote_code=True (Kimi-K3's tiktoken tokenizer is remote
      code; the harness's local-completions client does not pass the flag).
    NEMO6_ADD_BOS=1: encode() prepends bos_token_id when add_special_tokens (default True) and it is missing. Mistral-Large-4: the server
      (tokenizer-mode mistral, tekken) starts every prompt with BOS 1, the HF tokenizer.json does not (job 39795 tok_check); generation
      requests are tokenized by the server, likelihood requests by the client, so both must agree.
    Lazy (post-import hook of transformers.models.auto.tokenization_auto): an eager `import transformers` here made EVERY python start
    (the harness runs each generated program in a fresh interpreter) take seconds -> all of Mistral's HumanEval / MBPP / BigCodeBench
    programs hit the execution timeout (job 39853, 11-05), score 0."""
    trc, bos = os.environ.get("NEMO6_TRUST_REMOTE_CODE") == "1", os.environ.get("NEMO6_ADD_BOS") == "1"
    m = _m
    orig = m.AutoTokenizer.from_pretrained.__func__

    def from_pretrained(cls, *a, **kw):
        if trc:
            kw.setdefault("trust_remote_code", True)
        tok = orig(cls, *a, **kw)
        try:                                   # the harness client reads tokenizer.vocab; Kimi's TikTokenTokenizer has none (job 39844)
            tok.vocab
        except AttributeError:
            try:
                object.__setattr__(tok, "vocab", tok.get_vocab())
            except Exception:
                pass
        if bos and tok.bos_token_id is not None:
            enc = tok.encode

            def encode(text, *ea, **ekw):
                ids = list(enc(text, *ea, **ekw))
                if ekw.get("add_special_tokens", True) and (not ids or ids[0] != tok.bos_token_id):
                    ids = [tok.bos_token_id] + ids
                return ids

            tok.encode = encode
        return tok

    m.AutoTokenizer.from_pretrained = classmethod(from_pretrained)


def _patch_judge(m):
    """LLM judge (simpleqa_verified / facts_parametric): the gateway's gpt-oss-120b RPM limit (1000) is shared with other users; 6 quick
    openai retries do not outlast a 1-minute window -> "N judge calls failed" (DeepSeek-V4-Flash job 39841, 11-05). Raise the openai
    client's max_retries (exponential backoff, honours Retry-After) to NEMO6_JUDGE_RETRIES (30). Grading is unchanged."""
    orig = m.AsyncOpenAIJudge.__init__
    if getattr(orig, "_nemo6", False):
        return

    def __init__(self, *a, **kw):
        kw["max_retries"] = max(int(kw.get("max_retries", 6)), int(os.environ.get("NEMO6_JUDGE_RETRIES", "30")))
        return orig(self, *a, **kw)

    __init__._nemo6 = True
    m.AsyncOpenAIJudge.__init__ = __init__


_POST["lm_eval.api.judge"] = _patch_judge
_POST["openai.resources.completions"] = _patch_openai
if os.environ.get("NEMO6_TRUST_REMOTE_CODE") == "1" or os.environ.get("NEMO6_ADD_BOS") == "1":
    _POST["transformers.models.auto.tokenization_auto"] = _patch_auto_tokenizer
if not any(isinstance(f, _PostImportHook) for f in sys.meta_path):
    sys.meta_path.insert(0, _PostImportHook())
