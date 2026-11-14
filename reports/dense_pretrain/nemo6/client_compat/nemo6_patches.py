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
    """post-import hook of openai.resources.completions (lazy: generated-code subprocesses never import openai).
    Besides the stop cap (1.), two switches for OUR ladder checkpoints behind Megatron's dynamic text-generation server
    (nemo6_ladder.sbatch; both unset for the vLLM external references):
      NEMO6_STOP_TOKEN_IDS=151643: every request also carries stop_token_ids. The ladder data ends each document with <|endoftext|>
        151643, while Megatron's HuggingFaceTokenizer EOD for the Qwen3 tokenizer is <|im_end|> 151645: without it a generation would
        run past the document end into a new "document" until a stop string or max_tokens (vLLM stops at the base model's EOS itself).
      NEMO6_DROP_SEED=1: requests go without the client's constant seed=-1 (local-completions asserts seed == -1). Megatron rejects
        any negative seed (400 "seed must be a nonnegative int64 or null", job 43373), and a fixed seed would repeat samples anyway:
        Megatron derives per-request seeds seed + index inside one HTTP request, so the same prompt sent in two batches
        (hmmt_sampled: repeats 32 > batch 16) would get identical samples; unseeded requests draw from the engine's RNG stream.
        Greedy (temperature 0 -> top_k 1) and likelihood requests do not depend on the seed.
    The lm_eval batch-cache key is computed from the client kwargs before this call, so resumability is unaffected."""
    stop_ids = [int(x) for x in os.environ.get("NEMO6_STOP_TOKEN_IDS", "").split(",") if x.strip()]
    drop_seed = os.environ.get("NEMO6_DROP_SEED") == "1"
    for cls in (m.AsyncCompletions, m.Completions):
        orig = cls.create
        if getattr(orig, "_nemo6", False):
            continue

        def make(orig):
            def create(self, *a, **kw):
                st = kw.get("stop")
                if isinstance(st, (list, tuple)) and len(st) > MAX_STOP:
                    kw["stop"] = list(st)[:MAX_STOP]
                if stop_ids:
                    eb = dict(kw.get("extra_body") or {})
                    eb.setdefault("stop_token_ids", stop_ids)
                    kw["extra_body"] = eb
                if drop_seed:
                    kw.pop("seed", None)
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


_BCB_PYTHON_DEFAULT = "/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/boxiangw/envs/bcb_eval/bin/python"


def _patch_bcb_checker(m):
    """BigCodeBench judged by the OFFICIAL checker (11-06, user: "Bigcodebench 我们的实验的数字异常的高"). This harness revision writes
    candidate + tests to a file and runs python on it: the unittest TestCases are defined but never run, so a program "passes" iff it
    exits 0 (empty answers pass). Same fix as Boxiang's nemo6_driver.py (his rows in the downstream sheet): every BigCodeBench program
    goes to his bcb_official/bcb_check.py (official untrusted_check: official sandbox, TestCases run, any failure / error = fail) in his
    BigCodeBench venv (NEMO6_BCB_PYTHON). Sanitizing, prompts and pass@k stay the harness's. At most NEMO6_BCB_WORKERS (32) checks run at
    once (the harness pool is 128 threads). ON BY DEFAULT (user 11-06: "这个harness不对要修复，不能直接放一个不对的分数"): NEMO6_BCB_PYTHON
    defaults to Boxiang's venv; if it is missing the BigCodeBench check raises (task fails, no number) - never the old exit-code scoring.
    NEMO6_BCB_PYTHON=0 restores the harness runner (debugging only)."""
    import json as _json
    import subprocess as _subprocess
    import tempfile as _tempfile
    import threading as _threading

    py = os.environ.get("NEMO6_BCB_PYTHON", _BCB_PYTHON_DEFAULT)
    check = os.environ.get("NEMO6_BCB_CHECK", "/scratch/fsw/portfolios/cosmos/projects/cosmos_base_data/users/boxiangw/cosmos_ultra_pre/"
                                              "evals/nemo6/bcb_official/bcb_check.py")
    sem = _threading.Semaphore(int(os.environ.get("NEMO6_BCB_WORKERS", "32")))
    harness_check = m.check_correctness
    if getattr(harness_check, "_nemo6", False):
        return

    def check_correctness(check_program, timeout, task_id, completion_id):
        if not str(task_id).startswith("BigCodeBench/"):
            return harness_check(check_program, timeout, task_id, completion_id)
        if not os.access(py, os.X_OK):
            raise RuntimeError(f"BigCodeBench needs the official test environment, {py} missing (NEMO6_BCB_PYTHON)")
        # 11-10: Boxiang's evals/nemo6/ (bcb_check.py) vanished from disk; `python -I <missing script>` prints nothing, which scored every
        # program "fail" (BigCodeBench = 0.0 written as a valid number). A missing checker must fail the task ("-"), never score 0.
        if not os.path.isfile(check):
            raise RuntimeError(f"BigCodeBench checker {check} missing (NEMO6_BCB_CHECK): task fails instead of scoring 0")
        with _tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            _json.dump({"program": check_program}, f)
        try:
            with sem:
                r = _subprocess.run([py, "-I", check, f.name], capture_output=True, text=True, timeout=600)
            lines = [x for x in r.stdout.strip().splitlines() if x.startswith("{")]
            out = _json.loads(lines[-1]) if lines else {"status": "fail", "details": {"checker": r.stderr[-1000:]}}
        except _subprocess.TimeoutExpired:
            out = {"status": "timeout", "details": {}}
        finally:
            os.unlink(f.name)
        return {"program": check_program, "status": out["status"], "passed": out["status"] == "pass", "result": out.get("details"),
                "task_id": task_id, "completion_id": completion_id}

    check_correctness._nemo6 = True
    m.check_correctness = check_correctness


def _patch_replay(m):
    """NEMO6_REPLAY_DIR=<dir with the run's old samples_*.jsonl>: re-score greedy generate_until tasks WITHOUT a server and without a
    request cache (11-06: Qwen3-30B-A3B-Base / Qwen3.8-27B ran BigCodeBench before --use_cache existed). generate_until returns, per
    request, the response logged for the same (task, prompt) in the newest samples file of that task; a request with no logged response,
    or a sampled request, raises (the task fails, no number) - never a substitute answer. Scoring then runs unchanged in the harness."""
    import glob as _glob
    import json as _json

    rdir = os.environ["NEMO6_REPLAY_DIR"]
    cls = m.OpenaiCompletionsLM
    if getattr(cls.generate_until, "_nemo6", False):
        return
    logged = {}

    def responses(task):
        if task not in logged:
            fs = sorted(_glob.glob(f"{rdir}/**/samples_{task}_20*.jsonl", recursive=True), key=os.path.getmtime)
            if not fs:
                raise RuntimeError(f"replay: no samples for {task} under {rdir}")
            mp = {}
            for line in open(fs[-1]):
                x = _json.loads(line)
                a, r = x["arguments"]["gen_args_0"], x["resps"][0]
                mp[a["arg_0"]] = r[0] if isinstance(r, list) else r
            logged[task] = mp
        return logged[task]

    def generate_until(self, requests, disable_tqdm=False):
        out = []
        for req in requests:
            ctx, kw = req.args[0], req.args[1]
            if kw.get("do_sample"):
                raise RuntimeError("replay: sampled request, only greedy tasks can be replayed")
            mp = responses(req.task_name)
            if ctx not in mp:
                raise RuntimeError(f"replay: no logged response for a {req.task_name} request (doc {req.doc_id})")
            out.append(mp[ctx])
        return out

    generate_until._nemo6 = True
    cls.generate_until = generate_until


if os.environ.get("NEMO6_REPLAY_DIR"):
    _POST["lm_eval.models.openai_completions"] = _patch_replay
_POST["lm_eval.api.judge"] = _patch_judge
_POST["openai.resources.completions"] = _patch_openai
if os.environ.get("NEMO6_BCB_PYTHON", _BCB_PYTHON_DEFAULT) not in ("", "0", "none"):
    _POST["lm_eval.metrics.code_eval"] = _patch_bcb_checker
if os.environ.get("NEMO6_TRUST_REMOTE_CODE") == "1" or os.environ.get("NEMO6_ADD_BOS") == "1":
    _POST["transformers.models.auto.tokenization_auto"] = _patch_auto_tokenizer
if not any(isinstance(f, _PostImportHook) for f in sys.meta_path):
    sys.meta_path.insert(0, _PostImportHook())
