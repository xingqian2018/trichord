---
name: reorg_slurm_log
description: Move slurm log files in ~/log/slurm/ into per-job-family subfolders, grouping related variants (e.g. all `golden_caption_*`) together. Skips files whose job is still running.
user_invocable: true
---

Reorganizes `~/log/slurm/` from a flat pile into per-job-family subfolders. Original filenames are preserved; only the containing folder changes.

## Steps

1. **Active PIDs** — `squeue -u xingqianx -h -o '%i'`. Never move files whose pid is in this set.
2. **Existing families** — `find ~/log/slurm/ -maxdepth 1 -mindepth 1 -type d -printf '%f\n'`. These are anchors.
3. **Scan files** — `find ~/log/slurm/ -maxdepth 1 -type f -printf '%f\n'`. Parse each with `^(?P<prefix>.+)\.(?P<pid>[0-9]+)\.(?P<ext>e|o|err|out)$`. Files not matching → UNMATCHED.
4. **Assign family** for each prefix:
   - **Pass A (anchor to existing):** if `prefix_tokens[:len(F)] == F_tokens` for some existing family `F` (tokens split on `_`), use `F`. On multiple matches, pick the **longest**.
   - **Pass B (cluster leftovers):** build a token-trie over remaining prefixes; family = deepest trie node with ≥2 descendant leaves (joined by `_`). Stop at the first token that diverges across members. Leaves with no peers are singletons named by their full prefix. Don't form families on a single generic token alone (`run`, `job`, `test`, `tmp`). **The family cutoff is NOT fixed at 2nd/3rd underscore — use agent judgment.** Pick the depth that looks semantically right for the actual prefixes: one family may be 2 tokens (`golden_caption`), another 3 (`t2i_mot_expDAITR002`), another 5. Go deeper when the extra tokens are clearly stable identifiers shared by all members; stop shallower when the next token already differs or looks like a variant marker (versions, batch IDs, numeric suffixes).
5. **Plan** — group by family, print `<src> → <family>/<src>`. Also print SKIPPED (active), UNMATCHED, and a summary.
6. **Apply** — dry-run by default; ask for confirmation. With arg `apply`, execute immediately. For each move: `mkdir -p ~/log/slurm/<family>/ && mv -n ~/log/slurm/<file> ~/log/slurm/<family>/<file>`. Conflicts (destination exists) → listed, skipped.

## Examples

- `golden_caption_v1`, `golden_caption_v2_big` → family `golden_caption`.
- `imagegen_sst_batch1`, `imagegen_sst_batch2` → family `imagegen_sst`.
- `t2i_mot_expDAITR002_000_...`, `t2i_mot_expDAITR002_003_...` → family `t2i_mot_expDAITR002`.
- Existing `golden_caption/` + new `golden_caption_v3_tiny.*` → routed into `golden_caption/` (Pass A).

## Arguments

- `apply` — skip confirmation.
- `family=<name>` — only operate on files that resolve to `<name>`.

## Constraints

- Only top-level files; never recurse into subfolders.
- Never move a file whose pid is in `squeue`.
- `mv -n` only — never overwrite or delete.
- Preserve the original filename; only the parent folder changes.
