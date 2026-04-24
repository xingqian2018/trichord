---
name: ccodesync
description: Two stages. S1 — collect git status for ~/Project/{imaginaire4,imaginaire4_alt,imaginaire4_sila,bashrc,trichord} across n0, awscode, gcpcode, render one table, ask to sync. S2 — on explicit yes, push+pull any unambiguously-drifted repo, then re-render the S1 table as proof.
user_invocable: true
---

## Hosts

Local is `xingqian-AIZ` or `xingqianx-NVDesktop0` (`n0`). SSH reachable: `awscode`, `gcpcode`. **AIZ→n0 is one-way** — from n0, AIZ is unreachable; mark as "not reachable from n0".

## Stage 1 — collect & report

Run in parallel (awscode's login shell is tcsh, so pipe the script via stdin, don't `bash -lc`):

```
bash ~/.claude/skills/ccodesync/check_repos.sh n0
ssh awscode 'bash -s awscode' < ~/.claude/skills/ccodesync/check_repos.sh
ssh gcpcode 'bash -s gcpcode' < ~/.claude/skills/ccodesync/check_repos.sh
```

Script emits `<machine>|<repo>|<branch>|<yes/no>|<sync>`. Sync values: `-` · `behind` · `lead` · `diverged` · `untracked`. Missing repos emit no row.

**Reply = the table only, then one prompt line. No commentary, no readouts, no anomaly notes.**

Columns: `machine | repo | branch | unstaged | sync`. Sort by machine (n0, awscode, gcpcode, then unreachable); keep the script's repo order within each block. Render `unstaged`: `yes`→`⚠️`, `no`→`-`.

**Table format: wrap the whole table in a fenced code block (triple backticks, no language tag) so it renders as monospace and columns line up.** Inside the fence, pad each column to the width of its widest value + 1 space, separate columns with ` | `, and include a header row plus a `---` separator row. Example shape:

````
```
machine | repo             | branch                | unstaged | sync
------- | ---------------- | --------------------- | -------- | ----
n0      | imaginaire4      | xingqianx/cosmos3_aid | -        | behind
n0      | bashrc           | master                | ⚠️        | -
awscode | imaginaire4      | xingqianx/cosmos3_aid | -        | -
```
````

Because the table is inside a code fence, markdown bold (`**...**`) won't render — rely on the literal word (`behind`, `lead`, `diverged`, `untracked`) to stand out against the sea of `-`.

Last line (outside the fence): `Sync the eligible repos? (yes/no)` — then stop.

## Stage 2 — sync (only on explicit user "yes")

Never initiate on your own. Eligibility is **per-repo, independent** — multiple repos can qualify in the same run. A repo is eligible iff **all**:

1. Exactly ONE row for the repo is non-clean (`unstaged=yes` and/or `sync=lead`) → call it the leader.
2. Leader's drift is only `unstaged` and/or `lead` — `behind`/`diverged`/`untracked` on the leader always blocks.
3. Every OTHER row for the **same repo** has `unstaged=no` AND `sync ∈ {-, behind}`. `behind` on the other machines is fine — we're about to `pull` them anyway. `lead`/`diverged`/`untracked` anywhere else blocks.

Drift in unrelated repos does NOT block this repo's sync.

For each eligible repo:
- On leader: `git add -A && git commit -m "auto code sync"` (if dirty), then `git push`.
- On every other machine holding the repo: `git pull --ff-only`.

Remote commands go through `ssh <host> 'bash -lc "cd ~/Project/<repo> && <cmd>"'`.

**Result output, in this order:**

1. A short "acted" list — one line per eligible repo: `<repo>: pushed from <leader>, pulled on <others>`.
2. A short "blocked" list — one line per repo skipped: `<repo>: <one-line reason>` (e.g. "leader behind on n0", "diverged on gcpcode", "multiple leads (n0, gcpcode)", "untracked everywhere"). Omit the list entirely if nothing was blocked.
3. Then re-run Stage 1 and render the refreshed table in the **same fenced/padded format** as Stage 1.

No other commentary.

## Extending

Add a repo → append to `REPOS=( … )` in `check_repos.sh`. Add a host → another `ssh <host> 'bash -s <label>' < check_repos.sh`. Never inline the repo list into the SSH command.
