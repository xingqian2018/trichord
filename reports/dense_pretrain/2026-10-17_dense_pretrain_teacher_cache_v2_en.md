# Dense pretrain teacher cache format v2 (top-128 + next-token rank + two off-top-k sample sets)

Status: format agreed (2026-10-17), code not yet implemented. This format applies to any teacher used for dense pretraining (the first one is `mimo-v2.6-pro-rl`). The target corpus is about 600B tokens of pretraining data.

Revision (2026-10-17, Runlong): (1) new file `off128_tail_stats.bin` with three tail statistics (section 3); (2) the next token's log probability is now a dense per-row array `next_token_logp.bin`, replacing the sparse pickle, which removes the ≤ 100M-rows-per-unit limit (section 4); (3) the total tail mass is now explicitly computed with logsumexp (section 3); (4) added the implementation requirements for `exact_k` = 64: combining the two sample sets and per-row random seeds (section 5). Row size 686 → 694 bytes.

## 1. Overview

- **top-K main data**: ids (uint24) and log probabilities (float16) of the sorted top-128; the id of the actual next token together with its rank (`next_token_id_and_rank`), and its log probability (`next_token_logp`, stored for every row).
- **off-top-K data**: the total tail mass outside the top-128 and three tail statistics; plus two tail sample sets, each with 4 ids and log probabilities: off64 (sampled from the renormalised distribution outside the top-64, as if only the top-64 were stored in full) and off128 (sampled from the renormalised distribution outside the top-128).
- Why two sample sets: whether training uses `exact_k` = 64 or 128, an unbiased tail estimate is available, so top-64 experiments do not need a second teacher pass. With `exact_k` = 64, the 4 off64 samples plus 4 samples built from off128 give 8 independent samples (section 5).
- Why tail statistics: they give the teacher's exact entropy, ‖p‖², and the tail mass at temperatures T = 0.5 and T = 2. These are sums over the full vocabulary, so they can only be computed while the teacher scores; afterwards they could only be roughly estimated from the 4 samples.
- 694 bytes per row; about 416 TB in total for 600B tokens of pretraining data.

## 2. Units and directory layout

Units follow the existing mechanism: one directory per prefix; large prefixes are cut at document boundaries into `parts/<lo>-<hi>/` by the producer's `plan_units` (current implementation: `tools/distill/mimo_offtopk_producer.py`).

**Shared storage directory**: `/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/cosmos3_llm_datasets/pretrain/tc/<model>/`, where `<model>` is the teacher name (e.g. `mimo-v2.6-pro-rl`).

```
/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/cosmos3_llm_datasets/pretrain/tc/<model>/<blend_hash>/<prefix_key>/      # main directory or parts/<lo>-<hi>/
    meta.json
    docs.idx
    top128_id.bin             top128_logp.bin
    next_token_id_and_rank.bin    next_token_logp.bin
    off128_tail_logp.bin          off128_tail_stats.bin
    off64_sample_id.bin       off64_sample_logp.bin
    off128_sample_id.bin      off128_sample_logp.bin
```

- **Unit**: the main directory or one `parts/<lo>-<hi>/` directory. Each unit is a complete, self-contained cache with its own `meta.json` and `docs.idx`; `row_start` is a row number local to the unit.
- **Cut points**: strictly at document boundaries. A document never spans two units, so recomputing one unit never affects its neighbours.
- **Unit size**: set by `--part-tokens`.
- **Reading**: `ShardedOfftopkReader` reads unit by unit; parts never need to be merged.
- **Concurrency**: one writer per directory (`writer.lock`); jobs are split by unit across a Slurm array.

## 3. File definitions

Let n be the total number of rows over all cached documents in the unit. Every `.bin` is a headerless, little-endian, row-major fixed-size array and can be read with `np.memmap`.

| File | Shape | dtype | Content |
|---|---|---|---|
| `top128_id.bin` | [n,128,3] | uint8 | ids of the 128 tokens the teacher considers most likely as the next token (uint24, little-endian), in descending order of probability |
| `top128_logp.bin` | [n,128] | float16 | their log probabilities (natural log, T=1), descending |
| `next_token_id_and_rank.bin` | [n] | uint32 | `id<<8 \| rank`: id is the actual next token (uint24); rank is its position 0..127 in the top-128, or 255 if it is not in the top-128 |
| `next_token_logp.bin` | [n] | float16 | teacher log probability of the actual next token (T=1), stored for every row; equals `top128_logp[rank]` when rank < 128 |
| `off128_tail_logp.bin` | [n] | float16 | log Σ_{i∉top-128} p_i: logsumexp over the log probabilities outside the top-128, computed in float32, then cast to float16 (this is what the existing `dpl_vllm_offtopk` hook does). Do not use log(1 − Σ top-128 probabilities): the subtraction loses precision when the tail is small |
| `off128_tail_stats.bin` | [n,3] | float16 | logs of three sums over the tokens outside the top-128: [:,0] = log(−Σ p log p), [:,1] = log Σ p², [:,2] = log Σ √p (definition and use below) |
| `off64_sample_id.bin` | [n,4,3] | uint8 | 4 token ids (uint24) sampled with replacement from the (renormalised) tail distribution outside the top-64 |
| `off64_sample_logp.bin` | [n,4] | float16 | teacher log probabilities of these 4 tokens (T=1, not conditional probabilities) |
| `off128_sample_id.bin` | [n,4,3] | uint8 | 4 token ids (uint24) sampled with replacement from the (renormalised) tail distribution outside the top-128 |
| `off128_sample_logp.bin` | [n,4] | float16 | same as above, for off128 |
| `docs.idx` | [documents,3] | int64 | one `(doc_idx, row_start, n_rows)` per row, sorted by `doc_idx` |
| `meta.json` | — | JSON | format version, teacher, k=128, m=4 for both sample sets, temperature, random-seed rule, definitions of the three tail statistics, blend_hash, row count, and the byte size and checksum of every file |

Row semantics: a document of L tokens (including its final EOD) has L−1 rows. Row t is the teacher's prediction of token t+1 after seeing the first t+1 tokens; its row number in every file is `row_start + t`.

**Vocabulary restriction: the teacher distribution is computed over Qwen3's 151,669 tokens only (ids 0–151668). Logits with id ≥ 151669 (MiMo-V2's 6 extra special tokens 151669–151674, and the output-layer padding rows up to 152575) are set to −∞ before normalisation: lp_i = z_i − log Σ_{j<151669} exp(z_j). Every id in every file is therefore < 151669 and compatible with the Qwen3 vocabulary; `meta.json` records `vocab_limit = 151669`.**

**Reading and writing uint24**: ids are below 2^24 (the teacher vocabulary must be ≤ 16,777,216; all current teachers qualify). Each id takes the 3 uint8 bytes `a[..., 0..2]` of the last dimension, little-endian (low byte first):

```
byte:   a[...,0]          a[...,1]              a[...,2]
        bit 7..0          bit 15..8             bit 23..16
write:  id & 0xFF         (id >> 8) & 0xFF      (id >> 16) & 0xFF
read:   id = a[...,0] | (a[...,1] << 8) | (a[...,2] << 16)      # cast to int32 before shifting
```

- That is, `a[...,k] = (id >> 8k) & 0xFF` (k = 0, 1, 2), i.e. **`id = a[0] + a[1]·2^8 + a[2]·2^16`** (`a[0]` is the lowest byte).
- Example: id = 1000 = 0x0003E8 → `[0xE8, 0x03, 0x00]` = `[232, 3, 0]`; decoding gives 232 + 3×256 + 0×65536 = 1000.

**Reading and writing `next_token_id_and_rank`**: one uint32 whose high 24 bits are the id and low 8 bits the rank.

```
bit:   31 ............... 8 | 7 ...... 0
       id (24 bits)         | rank (8 bits)
```

- Pack: `v = (id << 8) | rank`
- Unpack: `id = v >> 8`, `rank = v & 0xFF` (the low 8 bits)
- Example: id = 1000, rank = 3 → v = 1000×256 + 3 = 256003. Stored little-endian, byte 0 is the rank and bytes 1–3 are the id.

**Computing `off128_tail_logp`**: p_i is the teacher probability of token i (T = 1), lp_i = log p_i.

```
off128_tail_logp = log Σ_{i∉top-128} p_i = logsumexp_{i∉top-128} lp_i
```

That is, the log of the summed probability of every token outside the top-128. Sum the tail directly with logsumexp rather than computing log(1 − Σ_{top-128} p_i): the subtraction loses precision when the tail is small.

**`off64_tail_logp` is not stored**: `off64_tail_logp = log(exp(off128_tail_logp) + Σ_{k=65..128} exp(top128_logp[k]))`, derived from the stored data.

**Definition of `off128_tail_stats`**: every sum runs over the tokens outside the top-128; p is the full T=1 distribution.

```
[:,0] = log( −Σ p_i log p_i )    the tail's contribution to the teacher entropy
[:,1] = log( Σ p_i² )            the tail's contribution to ‖p‖², also Σ p_i^(1/T) at T = 0.5
[:,2] = log( Σ √p_i )            Σ p_i^(1/T) at T = 2
```

- **Use**: teacher entropy H = −Σ_{top-128} p log p + exp([:,0]), for entropy weighting or skipping low-entropy positions. ‖p‖² = Σ_{top-128} p² + exp([:,1]); the ‖·‖² of the tail shape p/P_tail128 is exp([:,1]) / P_tail128², which sets the variance of tail sampling, i.e. how many samples are enough. At temperature T the teacher distribution is ∝ p^(1/T), with normaliser Σ_{top-128} p^(1/T) + Σ_{tail} p^(1/T): the second term is exp([:,1]) at T = 0.5 and exp([:,2]) at T = 2. For T between 0.8 and 1.33 the second term can be estimated from the 4 samples (P_tail128 · the sample mean of p^(1/T−1)).
- **Computation**: as for the total tail mass, logsumexp in float32 over the log probabilities lp outside the top-128: [:,0] = logsumexp(lp + log(−lp)), [:,1] = logsumexp(2·lp), [:,2] = logsumexp(lp / 2). Drop tokens with p = 0 (lp = −inf) first, otherwise [:,0] becomes NaN. Rows with zero tail mass have −inf in all three columns.
- **With `exact_k` = 64**: like `off64_tail_logp`, not stored separately; add back the terms of ranks 65–128, e.g. Σ_{i∉top-64} p² = exp([:,1]) + Σ_{k=65..128} exp(2·top128_logp[k]).

**No `token_ids.bin`**: the id in `next_token_id_and_rank` is token t+1 of the corpus, so each row can be checked against the corpus directly; no separate input-token file is needed.

## 4. Next-token logp and consistency checks

The next token's log probability is stored for every row (`next_token_logp.bin`, dense float16), whether or not the token is in the top-128. Like every other `.bin`, the dense array is row-aligned, so there is no risk of misalignment from implicitly matching entries to the order in which `rank==255` rows occur, and it can be `np.memmap`-ed instead of loaded whole.

- **Size**: 2 bytes per row, about 1.2 TB for 600B rows, about 0.3% of the total. For comparison, a sparse pickle is smaller on disk, but each 100M-row unit takes about 120 MB of memory when loaded, once per dataloader worker, and cannot be memory-mapped.
- **Checks on read**: for rows with rank < 128, `next_token_logp` must equal `top128_logp[rank]` (both come from the same float32 value, so they must be bit-identical after the float16 cast); for rows with rank == 255, `next_token_logp` must be ≤ `top128_logp[127]` and ≤ `off128_tail_logp`. Any mismatch is an error.
- **Commit order**: as before, "write the data files first, then write `docs.idx` and `meta.json` atomically". `next_token_logp.bin` and `off128_tail_stats.bin` are data files and are written before `docs.idx` is committed (tmp + `os.replace`).
- **Blast radius of corruption**: `meta.json` records a checksum for every file, so corruption only requires recomputing one unit.

## 5. Sampling rules

Let p be the teacher's full distribution at row t.

- **off128**: q128(i) = p_i / P_tail128 (i not in the top-128); 4 samples with replacement.
- **off64**: q64(i) = p_i / P_tail64 (i not in the top-64, so ranks 65 to 128 are included); 4 samples with replacement.
- **Combining the two sets when `exact_k` = 64**: the off128 samples plus ranks 65–128 can produce samples with the same distribution as off64. For each slot, with probability P_mid / P_tail64 draw one of ranks 65–128 according to p (P_mid = Σ_{k=65..128} p_k); otherwise use that slot's off128 sample. `offtopk_forward_kl` in `dpl/megatron/dp.py` already does this. These 4 are independent of the 4 off64 samples, giving 8 independent samples from q64 and halving the variance of the tail estimate. With `exact_k` = 128 only off128 is used.
- Sampling must be done on the GPU from the teacher's full distribution, not from an approximation of the distribution outside the top-128.
- Sampled tokens may coincide with the next token and with each other; this is expected with sampling with replacement.
- The two sample sets are independent and use separate random streams.
- **Reproducibility**: the random seed is determined by `(blend_hash, prefix_key, doc_idx, row number, sample-set name)`, and the rule is written to `meta.json`, so resuming after an interruption or recomputing a single document gives identical results. The existing `dpl_vllm_offtopk` hook uses one `torch.Generator` per process, which is reproducible only under identical batching and call order and does not meet this requirement. It must switch to per-row deterministic random numbers, e.g. a uniform generated from a hash of (seed, row, slot), followed by inverse-transform sampling on the tail's cumulative distribution.
- **Precision**: log probabilities, total tail mass and the three tail statistics are computed with logsumexp in float32 or higher, and cast to float16 only at the end.
- **Feasibility**: the existing MiMo top-128 cache was produced with the `dpl_vllm_offtopk` hook (vLLM 0.30.0 + MiMo tp=4), so the tail-sampling hook works in that environment; other teachers need to be re-validated on their own models.

## 6. Size estimate

Estimated for 600B tokens of pretraining data: N ≈ 600B rows (one fewer per document, negligible).

| File | Bytes per row | Total size |
|---|---|---|
| `top128_id.bin` | 384 | 230.4 TB |
| `top128_logp.bin` | 256 | 153.6 TB |
| `next_token_id_and_rank.bin` | 4 | 2.4 TB |
| `next_token_logp.bin` | 2 | 1.2 TB |
| `off128_tail_logp.bin` | 2 | 1.2 TB |
| `off128_tail_stats.bin` | 6 | 3.6 TB |
| `off64_sample_id.bin` + `off64_sample_logp.bin` | 12 + 8 | 12.0 TB |
| `off128_sample_id.bin` + `off128_sample_logp.bin` | 12 + 8 | 12.0 TB |
| **Total** | **694** | **about 416.4 TB** |

`docs.idx`: about 800M documents and about 20 GB at the current corpus's average document length; negligible.
