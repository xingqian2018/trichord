#!/usr/bin/env python3
"""reports_xx/nemo6/tok_check.py --tokenizer T --host H --port P --model NAME - preflight: the client's tokenizer must give the same ids
as the server's /tokenize. lm_eval's local-completions client tokenizes loglikelihood requests itself and sends token ids, while
generation requests go as text and are tokenized by the server: if the two disagree (own tokenizers: Kimi tiktoken, Mistral tekken,
Ripple GPT2Tokenizer under transformers 5.x), likelihood scores silently go wrong. Exit 0 = identical, 4 = mismatch, 5 = check failed.
The client side loads the tokenizer exactly as the harness does: AutoTokenizer.from_pretrained(T) (+ nemo6_patches), encode(text)."""
import argparse
import json
import sys
import urllib.request

SAMPLES = [
    "Question: Which planet is known as the Red Planet?\nAnswer: Mars\n\nQuestion: What is 17 * 23?\nAnswer:",
    "def fib(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n",
    "The quick brown fox jumps over the lazy dog. It's 3.14159 -- isn't it?   Tabs\tand  spaces.",
    "中文测试：人工智能正在改变世界。请计算 12 + 30 = ？",
]


def server_ids(host, port, model, text):
    body = json.dumps({"model": model, "prompt": text}).encode()
    req = urllib.request.Request(f"http://{host}:{port}/tokenize", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)["tokens"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", required=True)
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    try:
        import transformers
        tok = transformers.AutoTokenizer.from_pretrained(a.tokenizer)
    except Exception as e:
        print(f"TOKCHECK FAILED: client tokenizer did not load: {type(e).__name__}: {e}")
        return 5
    bad = 0
    for s in SAMPLES:
        try:
            c, v = tok.encode(s), server_ids(a.host, a.port, a.model, s)
        except Exception as e:
            print(f"TOKCHECK FAILED: {type(e).__name__}: {e}")
            return 5
        ok = list(c) == list(v)
        bad += not ok
        print(f"TOKCHECK {'ok      ' if ok else 'MISMATCH'} client {len(c)} ids / server {len(v)} ids | {s[:40]!r}")
        if not ok:
            i = next((k for k, (x, y) in enumerate(zip(c, v)) if x != y), min(len(c), len(v)))
            print(f"   first diff at {i}: client {list(c)[max(0, i - 2):i + 6]} vs server {list(v)[max(0, i - 2):i + 6]}")
    print(f"TOKCHECK {'PASSED' if not bad else 'MISMATCH'} ({len(SAMPLES) - bad}/{len(SAMPLES)} identical), tokenizer {type(tok).__name__}")
    return 0 if not bad else 4


if __name__ == "__main__":
    sys.exit(main())
