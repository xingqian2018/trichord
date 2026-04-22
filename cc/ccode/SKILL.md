---
name: ccode
description: Provides coding priors for writing parallelized ML pipelines. Read the relevant method file, then apply the pattern.
user_invocable: true
---

# ccode — Parallelization Coding Skill

When the user asks to write or edit code:
1. **Ask the user which scheme they want** — present the options from the Decision Guide below and wait for their choice before proceeding
2. **Ask whether Gateway API calls are needed** — if yes, read `UnifiedGatewayCalls.md` and copy-paste the relevant class (`UnifiedGatewayVLM` or `UnifiedGatewayLLM`) verbatim into the implementation
3. **Ask whether S3 data operations are needed** — if yes, read `MSCBatchS3DataOp.md` and copy-paste the relevant utilities (`setup_msc`, `msc_download_many`, `msc_upload_many`, `reformat_path_s3_to_msc`) verbatim into the implementation
4. Read the corresponding method file for the pseudo-code pattern
5. Implement it following the universal rules

---

## Decision Guide

| Method | Scenario | File |
|---|---|---|
| A | 1 node, 1 process — simple batch process & save (e.g. VLM/LLM call → save output) | `MethodA_1N_1P_batch_process.md` |
| B | 1 node, 1 process — gather per-sample result files into one consolidated output (post Method A or multi-process jobs) | `MethodB_1N_1P_batch_process_result_gather.md` |
| C | N nodes, M processes (usually 1 process per GPU) — distributed batch process & save across machines | `MethodC_nN_mP_batch_process.md` |
| D | N nodes, M processes (usually 1 process per GPU) — distributed batch process where each rank buffers results locally (`result_buffer_sharded`), then all shards are gathered to rank 0 after the loop for final consolidation and save | `MethodD_nN_mP_batch_process_result_gather.md` |

---

## Common Structure (all methods)

```
1. Gather all samples -> todo_list (each sample is a dict with at least an ID/name)
2. Set sample['try_num'] = 0 for each
3. Filter out already-done samples -> todo_list
4. Sort todo_list by name or ID (deterministic)
5. Run the parallelized loop (see method file)
6. On failure: increment try_num; drop if try_num >= max_retry
```

---

## Universal Rules

- Always implement skip logic before the loop (check if output already exists)
- Always track `try_num`; never silently drop failures without counting them
- Always sort `todo_list` before sharding so sharding is deterministic
- Never write to the same output file from multiple workers without locking
- Save results incrementally per batch, not only at the end
- No comments in code — use self-documenting variable and function names
- Minimize try/except — only wrap external calls (API, I/O), never logic blocks

---

## Optional Integration: Gateway API (`UnifiedGatewayCalls.md`)

Read this file and copy-paste verbatim when the task requires VLM or LLM inference calls.

| Class | Use when |
|---|---|
| `UnifiedGatewayVLM` | Video/image + text input to a vision-language model |
| `UnifiedGatewayLLM` | Text-only input to a language model |

**Key points to preserve exactly:**
- Constructor spins up a dedicated asyncio event loop in a background thread — do not change this pattern
- `query()` is the only public entry point; it handles the semaphore and tqdm internally
- `build_request()` handles Gemini vs. non-Gemini message layout differences
- Model names resolved through `MODEL_CHOICE` dict; response format controlled by `RESPONSE_FORMAT`

---

## Optional Integration: S3 / MSC Data Operations (`MSCBatchS3DataOp.md`)

Read this file and copy-paste verbatim when the task reads from or writes to S3-compatible storage.

| Utility | Use when |
|---|---|
| `reformat_path_s3_to_msc` | Converting `s3://` URIs to MSC profile-prefixed paths |
| `setup_msc` | Initializing MSC credentials at job startup (call once on all ranks) |
| `msc_download_many` | Batch-downloading files from S3 concurrently |
| `msc_upload_many` | Batch-uploading byte payloads to S3 concurrently |

**Key points to preserve exactly:**
- `setup_msc` must be called before any MSC filesystem operations; only rank 0 writes the config
- `msc_download_many` and `msc_upload_many` are `async` — call them inside an async context or via `asyncio.run()`
- Failed downloads return `None`; failed uploads return `False` — callers must handle sparse results
- Concurrency is capped by `MSC_MAX_WORKERS` env var (default 32), set inside `setup_msc`
