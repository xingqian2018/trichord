---
name: ccodesync
description: Two stages. S1 — collect git status for ~/Project/{imaginaire4,imaginaire4_alt,imaginaire4_sila,bashrc,trichord} across the tracked machines (AIZ + n0 + awscode + gcpcode when launched from AIZ; n0 + awscode + gcpcode when launched from n0), render one table, ask to sync. S2 — on explicit yes, push+pull any unambiguously-drifted repo, then re-render the S1 table as proof.
user_invocable: true
---

## Hosts

**First, determine which machine you are on by running `hostname`.** The local machine is itself a tracked repo host — its row comes from running the script locally, never from SSH-to-self. Reachability differs by host:

- `xingqian-AIZ` (AIZ workstation) — tracked set is **4 machines**: {AIZ, n0, awscode, gcpcode}. AIZ row comes from local invocation; the other three via SSH. Local row is labeled `AIZ` (not `n0`).
- `xingqianx-NVDesktop0` — this machine **is** `n0`. Tracked set is **3 machines**: {n0, awscode, gcpcode}. n0 row comes from local invocation; awscode/gcpcode via SSH. AIZ is **out of scope** from n0 — don't try to reach it, don't mention it, don't add a row for it.

Pick the Stage 1 command set that matches your hostname.

## Stage 1 — collect & report

**Issue every call in a single parallel tool message — never one-by-one.** Fan out the local `bash` and all SSH calls together in one message; serial execution is needlessly slow and never warranted here. (awscode's login shell is tcsh, so pipe the script via stdin, don't `bash -lc`.)

**If on `xingqian-AIZ`** — local invocation gives the AIZ row, plus three SSH calls (4 machines total):

```
bash ~/.claude/skills/ccodesync/check_repos.sh AIZ
ssh n0      'bash -s n0'      < ~/.claude/skills/ccodesync/check_repos.sh
ssh awscode 'bash -s awscode' < ~/.claude/skills/ccodesync/check_repos.sh
ssh gcpcode 'bash -s gcpcode' < ~/.claude/skills/ccodesync/check_repos.sh
```

**If on `xingqianx-NVDesktop0` (n0)** — local invocation gives the n0 row, plus two SSH calls (3 machines total). Do NOT add an AIZ call:

```
bash ~/.claude/skills/ccodesync/check_repos.sh n0
ssh awscode 'bash -s awscode' < ~/.claude/skills/ccodesync/check_repos.sh
ssh gcpcode 'bash -s gcpcode' < ~/.claude/skills/ccodesync/check_repos.sh
```

Script emits `<machine>|<repo>|<branch>|<yes/no>|<sync>`. Sync values: `-` · `behind` · `lead` · `diverged` · `untracked`. Missing repos emit no row.

**Reply = the table only, then one prompt line. No commentary, no readouts, no anomaly notes.**

Columns: `machine | repo | branch | unstaged | sync`. Sort by machine in fixed order: `AIZ, n0, awscode, gcpcode` — skip any machine that wasn't queried (e.g. when launched from n0 there's no AIZ block). Keep the script's repo order within each block. Render `unstaged`: `yes`→`⚠️`, `no`→`-`.

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

Never initiate on your own. Eligibility is **per-repo, independent** — multiple repos can qualify in the same run. Two sync modes, pick the first that matches:

### Mode A — push & pull (there's a leader)

Iff **all** hold:

1. Exactly ONE row for the repo is `unstaged=yes` and/or `sync=lead` → call it the leader.
2. Leader's sync is only `-` or `lead` (never `behind`/`diverged`/`untracked`).
3. Every OTHER row for the same repo has `unstaged=no` AND `sync ∈ {-, behind}`. `behind` is fine — we're about to pull it anyway.

Action:
- On leader: `git add -A && git commit -m "auto code sync"` (if dirty), then `git push`.
- On every other machine holding the repo: `git pull --ff-only`.

### Mode B — pull-only (no leader, just stragglers)

Iff **all** hold:

1. No row for the repo is `unstaged=yes` or `sync=lead` anywhere.
2. At least one row is `sync=behind`.
3. Every row has `sync ∈ {-, behind}` (no `diverged`/`untracked`).

Action: on each `behind` machine for this repo, `git pull --ff-only`. No commit, no push.

### Common

Drift in unrelated repos does NOT block this repo's sync.

**Same parallelism rule as Stage 1: fan out every git command — local pulls, remote pulls, remote pushes — in a single tool message.** Do not serialize per-repo or per-host. (Pull-then-push on the same machine should still be one chained shell command, not two separate tool calls.)

Remote commands go through `ssh <host> 'bash -lc "cd ~/Project/<repo> && <cmd>"'`. The local machine runs commands without SSH (`cd ~/Project/<repo> && <cmd>` directly). From `xingqian-AIZ`: AIZ is local, {n0, awscode, gcpcode} are remote. From `xingqianx-NVDesktop0` (n0): n0 is local, {awscode, gcpcode} are remote — AIZ is out of scope, ignore it entirely.

**Result output, in this order:**

1. A short "acted" list — one line per eligible repo. Mode A: `<repo>: pushed from <leader>, pulled on <others>`. Mode B: `<repo>: pulled on <machines>`.
2. A short "blocked" list — one line per repo skipped: `<repo>: <one-line reason>` (e.g. "leader behind on n0", "diverged on gcpcode", "multiple leads (n0, gcpcode)", "untracked everywhere"). Omit the list entirely if nothing was blocked.
3. Then re-run Stage 1 and render the refreshed table in the **same fenced/padded format** as Stage 1.

No other commentary.

## Extending

Add a repo → append to `REPOS=( … )` in `check_repos.sh`. Add a host → another `ssh <host> 'bash -s <label>' < check_repos.sh`. Never inline the repo list into the SSH command.
