#!/usr/bin/env python3
"""reports_xx/mimo_doc_eval.py - the ladder's document-isolated per-split eval for an HF model served by vLLM (MiMo-V2.5-Pro-Base,
MiMo-V2.6-Pro-RL): same 115 splits, same scoring as code/scaling_ladder/doc_eval.py, so the losses sit next to our ladder JSONs.

  - Splits = the prefixes of a reference ladder eval (--ref <eval dir>: its doc_eval_stats.json lists the 115 prefixes with their
    predicted-token counts; its eval/val_splits_*_doc_isolated.json gives the suite -> split order).
  - One sample per document chunk (doc_eval.chunk_plan, seq 8192): prompt = d[start .. start + n] (n + 1 tokens, the chunk starts
    WITHOUT the previous chunk's context), targets d[start + 1 .. start + n]; targets == pad id (151643 = <|endoftext|> = EOD) carry
    no loss - Megatron's pad rule, so L - 2 predictions per document. No padding / packing: each chunk is its own vLLM prompt.
  - vLLM prompt_logprobs=0 = log p(actual token) under the model's full (native) vocabulary; fp32 LM head as in the teacher cache.
  - Loss per split = token-weighted mean NLL; its predicted-token count MUST equal the reference's (else the split is an error).
  - Resumable: every finished split is written to <out>/splits/<prefix basename>.json at once; done splits are skipped.

Other tokenizers (DeepSeek-V4, Kimi-K3; --text-dir ~/data/eval_ppl_decon): the ORIGINAL text of the same documents (line i of
<text-dir>/<dir>/<split>.jsonl = document i of the token file) is re-tokenised with the model's own tokenizer (special-token
strings encoded as plain text, no EOD) and scored with the SAME rule: the first token of a document is never predicted,
chunks of <= 8192 predictions without the previous chunk's context; --bos puts the model's BOS in front of every chunk as
context only. Loss per split is then per OWN token (not comparable across tokenizers); each split also records its NLL sum and
UTF-8 bytes (== the suite yaml's n_bytes), so BPB = nll / (bytes ln 2) compares across tokenizers.

Writes <out>/eval/val_splits_iter000000000_doc_isolated.json (the ladder's format) + <out>/model.txt; collect_doc_eval.py turns it
into <NAME>_ref.json. Run inside the vLLM container (reports_xx/mimo_doc_eval.sbatch).
"""
import argparse
import glob
import math
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code", "scaling_ladder"))
from doc_eval import chunk_plan, read_index  # noqa: E402


def split_chunks(prefix, seq_len):
    """[(prompt token ids int64, number of predictions)] for every chunk of every document of a split, in document order."""
    dtype, sizes, pointers = read_index(prefix)
    data = np.memmap(prefix + ".bin", dtype=dtype, mode="r")
    doc, start, length = chunk_plan(sizes, seq_len)
    first = pointers // dtype.itemsize
    out = []
    for d, s, n in zip(doc, start, length):
        a = int(first[d]) + int(s)
        out.append(np.asarray(data[a:a + int(n) + 1], dtype=np.int64))
    return out


TOK_DIR = "/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/cosmos3_llm_datasets/pretrain/qwen3_tokenized/val_set/eval_ppl_decon"


def encode_texts(tok, texts):
    """Own-tokenizer ids of each text: no BOS / EOS, special-token strings in the text encoded as ordinary text."""
    if hasattr(tok, "_encode_text_piece"):                # Kimi's TikTokenTokenizer
        return [tok.encode(t, allow_special_tokens=False) for t in texts]
    return tok(texts, add_special_tokens=False, split_special_tokens=True)["input_ids"]


def text_chunks(prefix, seq_len, tok, text_dir, bos):
    """[(prompt ids, number of scored positions at its end)] + split bytes + round-trip mismatches (first 50 docs)."""
    man = json.load(open(os.path.join(text_dir, "manifest.json")))["splits"]      # token prefix -> aligned text file
    path = os.path.join(text_dir, next(v["text"] for v in man.values() if v["qwen3_prefix"] == prefix))
    texts = [json.loads(line)["text"] for line in open(path, encoding="utf-8")]
    n_docs = read_index(prefix)[1].size
    assert len(texts) == n_docs, f"{path}: {len(texts)} docs, token file {n_docs}"
    ids = encode_texts(tok, texts)
    rt_bad = sum(1 for t, d in zip(texts[:50], ids[:50]) if tok.decode(d) != t)
    head = [tok.bos_token_id] if bos else []
    out = []
    for d in ids:
        preds = len(d) - 1
        for s in range(0, max(preds, 0), seq_len):
            n = min(seq_len, preds - s)
            out.append((np.asarray(head + list(d[s:s + n + 1]), dtype=np.int64), n))
    return out, sum(len(t.encode("utf-8")) for t in texts), rt_bad


def score(llm, sp, prompts):
    """Per prompt: log p of tokens 1..T (float64)."""
    outs = llm.generate([{"prompt_token_ids": p.tolist()} for p in prompts], sp, use_tqdm=False)
    res = []
    for p, o in zip(prompts, outs):
        T = p.size - 1
        plp = o.prompt_logprobs
        if hasattr(plp, "token_ids"):                     # FlatLogprobs: W entries per position, column 0 = the actual token
            tok = np.asarray(plp.token_ids, dtype=np.int64); lp = np.asarray(plp.logprobs, dtype=np.float64)
            W = tok.size // T
            tok, lp = tok.reshape(T, W)[:, 0], lp.reshape(T, W)[:, 0]
        else:                                             # list of dicts (first entry None)
            tok = np.empty(T, np.int64); lp = np.empty(T, np.float64)
            for i, dct in enumerate(plp[1:]):
                t = int(p[i + 1]); tok[i] = t; lp[i] = dct[t].logprob
        if not np.array_equal(tok, p[1:]):
            raise RuntimeError("prompt logprobs are not aligned with the prompt tokens")
        res.append(lp)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--ref", required=True, help="a ladder doc-isolated eval dir (doc_eval_stats.json + eval/val_splits_*.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--pp", type=int, default=1)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--batch", type=int, default=512, help="chunks per llm.generate call")
    ap.add_argument("--nnodes", type=int, default=1, help="multi-node (mp backend): this process = node rank 0; the other nodes run\n"
                    "`vllm serve --headless --node-rank k` with the same parallel / model settings (mimo_doc_eval.sbatch)")
    ap.add_argument("--master-addr", default="127.0.0.1")
    ap.add_argument("--master-port", type=int, default=29501)
    ap.add_argument("--tok-dir", default="", help="another tokenisation of the same sets laid out like the Qwen3 dir, e.g.\n"
                    "cosmos3_llm_datasets/pretrain/kimik3_tokenized/val_set/eval_ppl_decon (its manifest.json: EOD id, bytes)")
    ap.add_argument("--text-dir", default="", help="re-tokenise the original text (other tokenizers); see the module docstring")
    ap.add_argument("--bos", action="store_true", help="with --text-dir: the model's BOS in front of every chunk (context only)")
    ap.add_argument("--hf-overrides", default='{"head_dtype": "float32"}', help="JSON; MiMo: fp32 LM head as in the teacher cache")
    ap.add_argument("--only", default="", help="comma-separated split basenames (smoke test)")
    ap.add_argument("--max-chunks", type=int, default=0, help="smoke test: first N chunks of each split, no count check")
    a = ap.parse_args()

    stats = json.load(open(os.path.join(a.ref, "doc_eval_stats.json")))
    seq, splits = int(stats["seq_len"]), stats["splits"]
    ref = json.load(open(sorted(glob.glob(os.path.join(a.ref, "eval", "val_splits_iter*_doc_isolated.json")))[-1]))
    if a.only:
        keep = set(a.only.split(","))
        splits = [s for s in splits if os.path.basename(s["prefix"]) in keep]
    os.makedirs(os.path.join(a.out, "splits"), exist_ok=True); os.makedirs(os.path.join(a.out, "eval"), exist_ok=True)
    open(os.path.join(a.out, "model.txt"), "w").write(os.path.realpath(a.model) + "\n")

    from vllm import LLM, SamplingParams
    llm = LLM(model=a.model, tensor_parallel_size=a.tp, pipeline_parallel_size=a.pp, trust_remote_code=True, dtype="bfloat16",
              max_model_len=seq + 8, gpu_memory_utilization=a.gpu_mem, max_logprobs=1, enable_prefix_caching=False, seed=0,
              hf_overrides=json.loads(a.hf_overrides), **({"nnodes": a.nnodes, "node_rank": 0, "master_addr": a.master_addr, "master_port": a.master_port} if a.nnodes > 1 else {}))
    sp = SamplingParams(max_tokens=1, temperature=1.0, prompt_logprobs=0, detokenize=False)
    tok = tman = bos_id = None
    if a.tok_dir:                                         # other tokenisation, same layout / documents / chunking as the ladder
        tman = json.load(open(os.path.join(a.tok_dir, "manifest.json")))
        if a.bos:
            from transformers import AutoTokenizer
            bos_id = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True).bos_token_id
            assert bos_id is not None, "--bos: the tokenizer has no BOS"
    if a.text_dir:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
        assert not a.bos or tok.bos_token_id is not None, "--bos: the tokenizer has no BOS"

    t_all = time.time()
    for k, s in enumerate(splits):
        name = os.path.basename(s["prefix"])
        dst = os.path.join(a.out, "splits", f"{s['prefix'].replace('/', '__')}.json")
        if os.path.exists(dst) and not a.max_chunks:
            continue
        t0 = time.time()
        n_bytes = rt_bad = expected = None
        if tman is not None:                              # the same documents in another tokenisation; its EOD is never a target
            rel = os.path.relpath(s["prefix"], TOK_DIR)
            ent, pad = tman["splits"][rel], int(tman["tokenizer"]["eod_id"])
            chunks = [(c, None) for c in split_chunks(os.path.join(a.tok_dir, rel), seq)]
            n_bytes, expected = ent["n_bytes"], ent["n_tokens"] - 2 * ent["n_docs"]
        elif tok is None:                                 # the ladder's own token files (MiMo == Qwen3 ids)
            pad = int(s["pad_id"])
            chunks = [(c, None) for c in split_chunks(s["prefix"], seq)]
            expected = int(s["predicted_tokens"])
        else:
            chunks, n_bytes, rt_bad = text_chunks(s["prefix"], seq, tok, a.text_dir, a.bos)
        if a.max_chunks:
            chunks = chunks[:a.max_chunks]
        nll, n = 0.0, 0
        order = np.argsort([-c.size for c, _ in chunks], kind="stable")   # long chunks together: fewer ragged batches
        for b in range(0, len(order), a.batch):
            batch = [chunks[i] for i in order[b:b + a.batch]]
            prompts = [np.concatenate([[bos_id], c]) if bos_id is not None else c for c, _ in batch]   # BOS = context only
            for (p, nk), lp in zip(batch, score(llm, sp, prompts)):
                if bos_id is not None:
                    lp = lp[1:]                               # drop the prediction of the chunk's first token from BOS
                if nk is None:
                    m = p[1:] != pad
                    nll -= float(lp[m].sum()); n += int(m.sum())
                else:                                     # the last k positions = the document's tokens after its first
                    nll -= float(lp[len(lp) - nk:].sum()); n += nk
        if expected is not None and not a.max_chunks and n != expected:
            raise SystemExit(f"{name}: {n} predicted tokens, expected {expected} (n_tokens - 2 n_docs)")
        rec = {"prefix": s["prefix"], "loss": nll / n, "tokens": n, "nll": nll, "seconds": round(time.time() - t0, 1)}
        if n_bytes is not None:
            rec.update(bytes=n_bytes, bpb=nll / (n_bytes * math.log(2)))
            if rt_bad is not None:
                rec["roundtrip_mismatch_first50"] = rt_bad
        print(f"[mimo_doc_eval] {k + 1}/{len(splits)} {name:40s} loss {rec['loss']:.4f}{'  bpb %.4f' % rec['bpb'] if 'bpb' in rec else ''}  tokens {n:>10d}  {n / (time.time() - t0):,.0f} tok/s", flush=True)
        if a.max_chunks:
            continue
        with open(dst + ".tmp", "w") as f:
            json.dump(rec, f)
        os.replace(dst + ".tmp", dst)
    if a.only or a.max_chunks:
        return 0

    done = {json.load(open(p))["prefix"]: json.load(open(p)) for p in glob.glob(os.path.join(a.out, "splits", "*.json"))}
    by_prefix = {s["prefix"]: s for s in stats["splits"]}
    suites, i = {}, 0
    for sname, srec in ref["suites"].items():            # the ladder's suite / split order; splits are listed in the same order
        labels = srec["labels"]
        pre = [stats["splits"][i + j]["prefix"] for j in range(len(labels))]; i += len(labels)
        assert all(by_prefix[p]["predicted_tokens"] == t for p, t in zip(pre, srec["tokens"])), f"{sname}: split order mismatch"
        suites[sname] = {"labels": labels, "loss": [done[p]["loss"] for p in pre], "tokens": [done[p]["tokens"] for p in pre]}
        if tok is not None or tman is not None:           # other tokenizer: loss / tokens are per OWN token; bytes for BPB
            suites[sname].update(nll=[done[p]["nll"] for p in pre], bytes=[done[p]["bytes"] for p in pre], bpb=[done[p]["bpb"] for p in pre])
    out = {"iteration": 0, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "protocol": "doc_isolated", "suites": suites,
           "model": os.path.realpath(a.model), "scorer": "vLLM prompt_logprobs (reports_xx/mimo_doc_eval.py)",
           "tokenizer": (f"{tman['tokenizer'].get('name')} token files {a.tok_dir}, bos {a.bos}" if tman is not None else
                         "qwen3 (the ladder's token files)" if tok is None else f"own ({type(tok).__name__}), text {a.text_dir}, bos {a.bos}")}
    with open(os.path.join(a.out, "eval", "val_splits_iter000000000_doc_isolated.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"[mimo_doc_eval] done: {len(done)} splits in {(time.time() - t_all) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
