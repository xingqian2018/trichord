---
name: ssh_run
description: Run a command on a remote cluster over SSH. Default hosts are `awscode` and `gcpcode`. The skill wraps the user's command in `ssh <host> '<cmd>'`, reports that it launched, and — if the remote invocation produces a Slurm job id — points at where the Slurm log lives on the remote.
user_invocable: true
---

# Step 1: Read the Rules! ⚠️

## ⚠️ Don't Directly `ssh ...` or `scp ...`. It will not Work

Use `/usr/bin/ssh ...` and `/usr/bin/scp ...`

## Rule for `scancel` on ANY Slurm Job

- NEVER run `scancel` on any Slurm job without first asking the user and getting an explicit "yes" for the specific job id(s).** This applies to *every* invocation, including:
- "relaunch" / "re-run" / "fix and re-submit" requests — even if the user implied the prior submission is broken, still ask. Phrase it as: *"This will require `scancel <jobids>` — confirm?"*
- Cancelling **your own** previous submissions in the same thread — still ask.
- Duplicates found by the dedupe check — still ask.
- "Obviously wrong" jobs (wrong cluster, wrong cred, wrong nodes) — still ask.

**Never assume the user wants the prior job killed.** A re-launch may be intended to coexist. Always confirm. Paste the *exact* job ids so the user can sanity-check.

**Why this matters.** Killing the wrong job costs hours-to-days of compute. A typo or wrong assumption about scope can destroy a run with no recovery. "Relaunch" ≠ "scancel and relaunch" unless the user spells it out.

## Hosts

- `awscode` — AWS cluster head node
- `gcpcode` — GCP cluster head node

Pick from the user's phrasing: "aws" / "awscode" → `awscode`; "gcp" / "gcpcode" → `gcpcode`; ambiguous → ask.

## `slurm` / `slaunch` Pre-flight Dedupe Check Required

**Before any Slurm submission, check whether the same job is already running on the target host.**

```bash
/usr/bin/ssh <host> 'squeue -u $USER -o "%i %j %T %R" | grep <job_name>'
```

- Match by job name and/or identifying args (`--signature`, output path, etc.).
- For multi-variant batches, check **each** variant — don't check one and assume the rest are clear.
- If a duplicate is found, **stop and ask**. Acceptable resolutions: skip / cancel-and-rerun / submit-anyway.


# Step 2: How to Run SSH! (There Are Customized Things So Please READ! ⚠️)

## For Simple One Line Command

Default shape for any simple "run X on cluster" request.

```bash
/usr/bin/ssh <host> '<remote command>'
```

- Quote remote commands with **single quotes** so the local shell doesn't expand `$` / `` ` ``.
- If the remote command contains single quotes, escape with `'"'"'`.
- Chain a `cd` when the script needs a specific working directory: `cd <dir> && <cmd>`.

## For `slurm` and `slaunch` and Other Long Multi-line Command

- Scratch a shell at a local temporary folder `~/tmp/<name_at_your_choice>.sh`
- `~/tmp/` is persistent, quota-tracked, and easy to inspect: `/usr/bin/ssh <host> 'ls ~/tmp/'`.
- Use `/usr/bin/scp ...` copy local `~/tmp/<name_at_your_choice>.sh` to remote `~/tmp/<name_at_your_choice>.sh`.
  - i.e. `/usr/bin/scp ~/tmp/sshrun/<name_at_your_choice>.sh <host>:~/tmp/<name_at_your_choice>.sh`
- Create if needed: `/urs/bin/ssh <host> 'mkdir -p ~/tmp'`. Leave files in place after use.
- DON'T use `/tmp/` as your temporary folder.

### Heredoc ban (absolute)

**Never use heredoc syntax.** No `<<EOF`, `<<-EOF`, `<<'EOF'`, `<<"EOF"`, no custom terminators — anywhere, ever.

When the remote command has complex quoting, use **write-locally → scp → ssh-execute**:

- Write the script locally to `/tmp/sshrun/<name>.sh` using the Write tool.
- `/usr/bin/scp /tmp/sshrun/<name>.sh <host>:~/tmp/<name>.sh`
- `/usr/bin/ssh <host> 'bash ~/tmp/<name>.sh'`

**Why the ban is absolute.** The pattern `/usr/bin/ssh host "cat > foo.sh <<'EOF' ... EOF; bash foo.sh"` causes two failure modes together:

- The heredoc body is a double-quoted local arg. Backslash-newline continuations get eaten by the remote shell's tokenizer before heredoc collection completes, collapsing lines.
- With lines collapsed the `EOF` terminator isn't recognized, so `cat` slurps the trailing `bash foo.sh` into the script body — making it self-recursive. The script then re-invokes itself in a runaway loop.

---

## Composing with other Skills

`ssh_run` can carry another skill's work onto the remote host. Trigger: the user names another skill alongside `ssh_run` (e.g. "ssh_run meow on awscode").

- Read the inner skill's SKILL.md from `cc/<skill_name>/SKILL.md` to get the exact shell command.
- Run that command on the remote via `/usr/bin/ssh <host> '<cmd>'` — do **not** invoke the inner skill locally.
- Slurm job id reporting still applies.
- If the inner skill's steps are non-trivial to translate into a single remote command, say so and ask.

Counter-case: if the user invokes the other skill without mentioning `ssh_run`, run it locally as normal.

---

# Step 3: Reporting

After the ssh command returns:

- **Announce launch:** one short line — `Launched on <host>: <short description>`.
- **Do NOT stream or summarize remote stdout** beyond extracting a Slurm job id.
- **If a Slurm job id is found** (look for `Submitted batch job <N>` or a bare numeric id), report:

```
Launched on <host>: <job name or command>
Slurm job id: <jobid>
Remote log:   ~/log/slurm/*.<jobid>.e  (stderr)
               ~/log/slurm/*.<jobid>.o  (stdout)
To tail:      ssh <host> 'tail -f ~/log/slurm/*.<jobid>.e'
```

The parsed number is the **Slurm job id** — not the short-lived OS PID of the `slaunch`/`sbatch` wrapper. Always label it "Slurm job id". Log files are on the **remote** machine.

- **If ssh exits non-zero**, report the exit code and last few lines of stderr. Do not retry automatically.

## What this skill does NOT do

- Does not edit files locally or remotely.
- Does not poll job status — for status, the user invokes `/checkrun` or asks explicitly.
- Does not install or modify crontab entries without the user spelling out the exact schedule and command; if vague, ask first.
- Does not choose between `awscode` and `gcpcode` when ambiguous — ask.
