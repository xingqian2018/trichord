#!/usr/bin/env python3
"""reports_xx/nemo6/megatron_api_check.py - preflight of a Megatron dynamic text-generation server (nemo6_ladder.sbatch) before any nemo6
task runs against one of OUR ladder checkpoints. The lm_eval client is the external references' `local-completions` (likelihood requests:
client-tokenized ids, echo=True, logprobs=1, max_tokens=0, score = sum(token_logprobs[ctxlen:]); generation requests: text, server-
tokenized). Every assumption that score relies on is checked against the live server:

  1. tokenizer: client AutoTokenizer(T).encode(text) == the server's prompt_token_ids for the same text (generation path);
  2. likelihood response shape: max_tokens=0 + echo + logprobs on an id prompt -> token_logprobs == [None] + one value per prompt
     token after the first, tokens / top_logprobs aligned (else sum(token_logprobs[ctxlen:]) would include a generated token);
  3. PARITY with the doc eval: the whole validation split --parity-prefix (Megatron .bin/.idx) is re-scored through the server with the
     doc eval's protocol (code/scaling_ladder/doc_eval.py: one document per sample, chunks of <= --parity-seq predictions, label == EOD
     151643 not counted, no cross-document context): token count must equal --parity-tokens exactly and the token-weighted mean NLL
     must match --parity-loss within --tol (bf16 training-path vs inference-path kernels: expect ~1e-3). This is the check that the
     inference path (flash attention + per-layer SWA window + NoPE layers + learnable-sink softmax correction, EP all-to-all) computes
     the trained model;
  4. greedy generation runs and stops; stop_token_ids is accepted;
  5. sampling: identical prompts at temperature 1 with no seed give different samples (the client drops seed -1 for sampled requests,
     nemo6_patches NEMO6_DROP_SEED: the server would otherwise derive seed -1 + index per request, repeating samples across batches).

Exit 0 = all pass, 4 = a check failed (no task should run), 5 = could not run. Standard library + numpy + transformers."""
import argparse
import concurrent.futures as cf
import json
import math
import sys
import time
import urllib.error
import urllib.request

import numpy as np

SAMPLES = [
    "Question: Which planet is known as the Red Planet?\nAnswer: Mars\n\nQuestion: What is 17 * 23?\nAnswer:",
    "def fib(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n",
    "The quick brown fox jumps over the lazy dog. It's 3.14159 -- isn't it?   Tabs\tand  spaces.",
    "中文测试：人工智能正在改变世界。请计算 12 + 30 = ？",
]
_DTYPES = {1: np.uint8, 2: np.int8, 3: np.int16, 4: np.int32, 5: np.int64, 6: np.float64, 7: np.float32, 8: np.uint16}


def post(url, body, timeout=600):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def read_split(prefix):
    """(dtype, sizes, pointers) of a Megatron MMIDIDX v1 index + the token memmap (same reader as doc_eval.read_index)."""
    with open(prefix + ".idx", "rb") as f:
        assert f.read(9) == b"MMIDIDX\x00\x00", prefix
        assert int(np.frombuffer(f.read(8), "<u8")[0]) == 1
        dtype = np.dtype(_DTYPES[f.read(1)[0]])
        n = int(np.frombuffer(f.read(8), "<u8")[0])
        f.read(8)
        sizes = np.frombuffer(f.read(4 * n), "<i4").astype(np.int64)
        pointers = np.frombuffer(f.read(8 * n), "<i8").astype(np.int64)
    return dtype, sizes, pointers, np.memmap(prefix + ".bin", dtype=dtype, mode="r")


def parity(url, prefix, seq, eod, batch, workers):
    """Doc eval protocol through the server: per document d (L tokens incl. its EOD), predictions d[1..L-1] in chunks of <= seq, chunk =
    prompt d[s .. s+n] (n predictions, its own context only); labels == eod (the data EOD = Megatron's pad) are not counted."""
    dtype, sizes, pointers, data = read_split(prefix)
    prompts = []
    for i in range(sizes.size):
        d = np.asarray(data[pointers[i] // dtype.itemsize: pointers[i] // dtype.itemsize + sizes[i]], dtype=np.int64)
        preds = max(int(sizes[i]) - 1, 0)
        s = 0
        while s < preds:
            n = min(seq, preds - s)
            prompts.append(d[s: s + n + 1].tolist())
            s += n
    chunks = [prompts[k: k + batch] for k in range(0, len(prompts), batch)]

    def score(ch):
        r = post(url, {"model": "x", "prompt": ch, "max_tokens": 0, "echo": True, "logprobs": 1, "temperature": 0.0})
        nll, cnt = 0.0, 0
        for p, c in zip(ch, r["choices"]):
            lp = c["logprobs"]["token_logprobs"]
            if len(lp) != len(p) or lp[0] is not None:
                raise RuntimeError(f"logprob length {len(lp)} != prompt {len(p)} (or first not None)")
            for tok, v in zip(p[1:], lp[1:]):
                if tok != eod:
                    nll -= v
                    cnt += 1
        return nll, cnt

    t0 = time.time()
    tot, cnt = 0.0, 0
    with cf.ThreadPoolExecutor(workers) as ex:
        for nll, c in ex.map(score, chunks):
            tot += nll
            cnt += c
    return tot / max(cnt, 1), cnt, len(prompts), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--stop-token-ids", default="151643")
    ap.add_argument("--parity-prefix", required=True)
    ap.add_argument("--parity-loss", type=float, required=True)
    ap.add_argument("--parity-tokens", type=int, required=True)
    ap.add_argument("--parity-seq", type=int, default=8192)
    ap.add_argument("--eod", type=int, default=151643)
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    url = f"http://{a.host}:{a.port}/v1/completions"
    stops = [int(x) for x in a.stop_token_ids.split(",") if x]
    rep, bad = {}, []
    try:
        import transformers
        tok = transformers.AutoTokenizer.from_pretrained(a.tokenizer)

        # 1. tokenizer
        for s in SAMPLES:
            c = list(tok.encode(s))
            v = post(url, {"model": "x", "prompt": s, "max_tokens": 1, "temperature": 0.0})["choices"][0]["prompt_token_ids"]
            ok = c == list(v)
            print(f"[1] tokenizer {'ok      ' if ok else 'MISMATCH'} client {len(c)} / server {len(v)} ids | {s[:40]!r}")
            if not ok:
                bad.append("tokenizer")
        rep["tokenizer"] = "tokenizer" not in bad

        # 2. likelihood response shape (the exact payload of local-completions loglikelihood)
        ids = tok.encode(SAMPLES[0])
        # = the patched client's payload: local-completions sends seed=-1, NEMO6_DROP_SEED removes it (Megatron rejects negative seeds)
        ch = post(url, {"model": "x", "prompt": [ids], "max_tokens": 0, "echo": True, "logprobs": 1, "temperature": 0.0})["choices"][0]
        lp = ch["logprobs"]
        shape_ok = (len(lp["token_logprobs"]) == len(ids) and lp["token_logprobs"][0] is None and len(lp["tokens"]) == len(ids)
                    and lp["top_logprobs"] is not None and isinstance(lp["top_logprobs"][1], dict)
                    and all(v is not None and v <= 0 for v in lp["token_logprobs"][1:]))
        print(f"[2] likelihood shape {'ok' if shape_ok else 'BAD'}: prompt {len(ids)} ids -> token_logprobs {len(lp['token_logprobs'])}, "
              f"first {lp['token_logprobs'][0]!r}, generated ids {ch.get('generation_token_ids')}, top_logprobs[1] {str(lp['top_logprobs'][1])[:80]}")
        if not shape_ok:
            bad.append("likelihood_shape")
        rep["likelihood_shape"] = shape_ok

        # 3. parity with the doc eval
        loss, n_tok, n_prompts, sec = parity(url, a.parity_prefix, a.parity_seq, a.eod, batch=32, workers=16)
        d = loss - a.parity_loss
        par_ok = n_tok == a.parity_tokens and abs(d) <= a.tol
        print(f"[3] parity {'ok' if par_ok else 'FAILED'}: {a.parity_prefix.split('/')[-1]} {n_prompts} samples, {n_tok} predicted tokens "
              f"(doc eval {a.parity_tokens}); server NLL {loss:.5f} vs doc eval {a.parity_loss:.5f} (diff {d:+.5f}, tol {a.tol}); {sec:.0f} s")
        if not par_ok:
            bad.append("parity")
        rep["parity"] = {"split": a.parity_prefix, "server_loss": loss, "doc_eval_loss": a.parity_loss, "diff": d,
                         "tokens": n_tok, "doc_eval_tokens": a.parity_tokens, "seconds": round(sec, 1), "ok": par_ok}

        # 4. greedy generation + stop_token_ids
        g = post(url, {"model": "x", "prompt": "The capital of France is", "max_tokens": 24, "temperature": 0.0,
                       "stop_token_ids": stops})["choices"][0]
        print(f"[4] greedy: {g['text']!r} (finish {g.get('finish_reason')}, {len(g['generation_token_ids'])} tokens)")
        rep["greedy"] = g["text"]

        # 5. sampling diversity (no seed, as the patched client sends sampled requests)
        r = post(url, {"model": "x", "prompt": ["Once upon a time"] * 8, "max_tokens": 24, "temperature": 1.0, "top_p": 1.0,
                       "stop_token_ids": stops})
        texts = [c["text"] for c in r["choices"]]
        distinct = len(set(texts))
        print(f"[5] sampling: {distinct}/8 distinct; e.g. {texts[0]!r}")
        if distinct < 6:
            bad.append("sampling")
        rep["sampling_distinct"] = distinct
    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        body = e.read().decode()[:500] if isinstance(e, urllib.error.HTTPError) else ""
        print(f"APICHECK ERROR: {type(e).__name__}: {e} {body}")
        return 5
    except Exception as e:
        print(f"APICHECK ERROR: {type(e).__name__}: {e}")
        return 5
    rep["passed"] = not bad
    if a.out:
        with open(a.out, "w") as f:
            json.dump(rep, f, indent=1)
    print(f"APICHECK {'PASSED' if not bad else 'FAILED ' + ','.join(bad)}")
    return 0 if not bad else 4


if __name__ == "__main__":
    sys.exit(main())
