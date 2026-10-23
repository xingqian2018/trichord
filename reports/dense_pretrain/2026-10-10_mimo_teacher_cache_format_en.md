# MiMo teacher cache format (top-128 + off-top-k samples)

Status: format settled (2026-10-10), code not yet implemented. Teacher: `mimo-v2.6-pro-rl`; corpus: `kd_corpus/mimo_v2_v1/` (byte-identical to `qwen3_v1/`).

## 1. Differences from Runlong's existing format (`dpl.teacher_cache/1`)

| | Existing format (Qwen3.8 / DeepSeek caches) | New format |
|---|---|---|
| top-k | k=64 (32 for DeepSeek) | k=128 |
| Probability of the actual next token | not stored (vLLM returns it, but it is only used for statistics) | stored in column 0 of `offtopk_*` when it is not in the top-128 |
| Information inside the tail | none, only the total mass | 7 to 8 random samples per position from the teacher's tail distribution |
| `offsets.bin` | yes (int64, for cross-tokenizer alignment) | dropped (not needed with identity alignment) |
| `token_ids.bin` | yes | kept (per-token alignment check) |

## 2. Directory layout

Same as the existing format, one directory per prefix:

```
<teacher_cache_root>/mimo-v2.6-pro-rl/<blend_hash>/<prefix_key>/
    meta.json
    docs.idx
    token_ids.bin
    topk_ids.bin
    topk_logp.bin
    tail_logp.bin
    offtopk_ids.bin
    offtopk_logp.bin
```

`<prefix_key>` is the corpus directory name; `<blend_hash>` comes from the blend JSON written by `finalize-kd-corpus`.

## 3. File definitions

Let N be the total number of rows over all cached documents of the prefix. Every `.bin` is a headerless, little-endian, row-major fixed-size array and can be read directly with `np.memmap`.

| File | Shape | dtype | Content |
|---|---|---|---|
| `token_ids.bin` | [N] | int32 | input token of row t (equals the corpus token at that position; used for checking) |
| `topk_ids.bin` | [N, 128] | int32 | ids of the 128 tokens the teacher considers most likely as the next token, in descending order of probability |
| `topk_logp.bin` | [N, 128] | float16 | their log probabilities (natural log, T=1), descending |
| `tail_logp.bin` | [N] | float16 | log(1 − Σ top-128 probabilities), computed in float32 or higher, then cast to float16 |
| `offtopk_ids.bin` | [N, 8] | int32 | 8 tokens outside the top-128, rules in section 4 |
| `offtopk_logp.bin` | [N, 8] | float16 | teacher log probabilities of these 8 tokens (natural log, T=1, not conditional probabilities) |
| `docs.idx` | [documents, 3] | int64 | one `(doc_idx, row_start, n_rows)` per row, sorted by `doc_idx` |
| `meta.json` | — | JSON | format version, teacher, k=128, m=8, temperature, sampling random-seed rule, blend_hash, row count, etc. |

Row semantics: a document d of L tokens (including its final EOD) has L−1 rows. Row t (t = 0..L−2) is the teacher's prediction of token t+1 after seeing the first t+1 tokens of the document. Row t of document d is row `row_start(d) + t` in every `.bin`.

Ids are `int32`: the MiMo vocabulary of 152,576 exceeds the range of int16 / uint16.

## 4. Rules for `offtopk_*`

Let y be the actual next token at row t (token t+1 of the corpus), and let the teacher's conditional tail distribution be q(i) = p_i / P_tail (i not in the top-128).

- **y not in the top-128**: column 0 stores y and log p_y; columns 1 to 7 are 7 tokens sampled **with replacement** from q.
- **y in the top-128**: all 8 columns are tokens sampled with replacement from q.

No extra flag is needed to tell the two cases apart: check whether y appears in `topk_ids[t]`. Sampled tokens may coincide with y or with each other; this is expected with sampling with replacement.

Sampling must be done on the GPU from the teacher's full distribution according to q, not from an approximation of the distribution outside the top-128.

## 5. Use in training (KD loss)

Let s be the student's full log probabilities at position t (log_softmax over the whole vocabulary).

The top-128 part is computed exactly:

```
KL_top = Σ_{k=1..128} p_k · (log p_k − s[topk_ids[t,k]])
```

The tail part is estimated without bias from the random samples (only the sampled columns, not y in column 0):

```
Let R be the set of sampled columns (columns 1..7 if y is not in the top-128, otherwise columns 0..7), M = |R|
KL_tail ≈ P_tail · (1/M) · Σ_{m∈R} ( offtopk_logp[t,m] − s[offtopk_ids[t,m]] )
```

Total KL = KL_top + KL_tail, loss = (1 − α)·CE + α·T²·KL (current setting α=0.75, T=1). Because q can be computed directly from the stored log p_i and P_tail, the sampling proposal probabilities do not need to be stored separately.

y in column 0 (when not in the top-128) was not sampled from q and must not be included in the average above, otherwise the estimate is biased. It is there to diagnose how well the teacher fits the actual token, and for possible other estimators later.

## 6. Size estimate

The MiMo corpus has 46,658,149,267 tokens in 63,035,592 documents, so N = 46,595,113,675 rows.

| File | Bytes per row | Total size |
|---|---|---|
| `token_ids.bin` | 4 | 0.19 TB |
| `topk_ids.bin` | 512 | 23.86 TB |
| `topk_logp.bin` | 256 | 11.93 TB |
| `tail_logp.bin` | 2 | 0.09 TB |
| `offtopk_ids.bin` | 32 | 1.49 TB |
| `offtopk_logp.bin` | 16 | 0.75 TB |
| **Total** | **822** | **about 38.3 TB** |

`docs.idx` is about 1.5 GB, negligible. About 62% of the space is `topk_ids.bin`; if compression is needed later, packing ids into 18 bits (152,576 < 2^18) is the biggest win and brings the total down to about 27 TB, but the reading code has to change.

## 7. Relation to the existing training caches

- The top-64 cache used in training can be derived from this format: take the first 64 columns, `tail_64 = log(exp(tail_128) + Σ_{k=65..128} p_k)`. Columns 65 to 128 have small probabilities, so float16 rounding has limited effect on tail_64. This way MiMo only needs to run once.
- This format has two more files (`offtopk_*`) and one fewer (`offsets.bin`) than the existing format, so Runlong's `TeacherCacheReader` cannot read it directly; the reader has to be extended or a separate reader written.

## 8. Requirements on the producer side and open issues

- vLLM: `prompt_logprobs=128`, `flat_logprobs=True`; column 0 already carries the log probability of the actual token, so y's probability needs no extra computation.
- **Tail sampling**: vLLM's `prompt_logprobs` does not expose the full distribution and cannot sample from the conditional tail distribution. A GPU sampling hook has to be added where the prompt logits are computed, returning only the 8 samples. **Whether this works on vLLM 0.30.0 + MiMo (tp=4, custom model) has not been verified**; the whole plan depends on it.
- **Reproducibility**: suggest determining the random seed by `(blend_hash, prefix_key, doc_idx, row number)` and writing the rule to `meta.json`, so that resuming after an interruption or recomputing a single document gives identical results.
- **Precision**: log probabilities and the tail are computed in float32 or higher and cast to float16 only at the end.
- **Engine**: the MiMo engine has not yet got past the GPU kernel caching step (several processes download FlashInfer kernels at the same time and segfault); this has to be fixed first.

## 9. To confirm

1. When y is in the top-128, all 8 columns are random samples; is this as intended?
2. Should the sampling random seed follow the suggestion in section 8?
3. Pack ids into 18 bits, or keep int32 for simplicity for now?
4. Which directory holds the 38 TB, and is it within quota?
