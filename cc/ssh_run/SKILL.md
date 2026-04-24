---
name: ssh_run
description: Run a command on a remote cluster over SSH. Default hosts are `awscode` and `gcpcode`. The skill wraps the user's command in `ssh <host> '<cmd>'`, reports that it launched, and — if the remote invocation produces a Slurm job id — points at where the Slurm log lives on the remote.
user_invocable: true
---

## Hosts

Two SSH aliases are pre-configured in `~/.ssh/config`:

- `awscode` — AWS cluster head node
- `gcpcode` — GCP cluster head node

Pick the host from the user's phrasing:
- Mentions "aws", "on aws", "awscode" → `awscode`
- Mentions "gcp", "on gcp", "gcpcode" → `gcpcode`
- Ambiguous → ask the user which one; do not guess.

## Default behavior — plain SSH

For ANY request of the form "run X on the cluster" / "kick off X" / "submit X", default to plain SSH. Do not invent wrappers. The shape is always:

```
ssh <host> '<remote command>'
```

Examples of what "plain SSH" covers:
- One-shot shell commands: `ssh awscode 'ls ~/log/slurm | head'`
- Slurm submissions via `slaunch` or `sbatch`: `ssh awscode 'cd ~/Project/... && slaunch small 1 <job_name> <script> ...'`
- Cron-related work (installing/listing/removing crontab entries, status checks): `ssh gcpcode 'crontab -l'`

Quote the remote command with **single quotes** so the local shell doesn't expand `$` / `` ` ``. If the remote command itself contains single quotes, escape using `'"'"'` or switch to a heredoc via `ssh <host> bash -s <<'EOF' ... EOF`.

If the remote command needs a specific working directory, chain it: `cd <dir> && <cmd>`. Do NOT assume `$HOME` is the right cwd for Slurm submissions.

**Load the user profile by default.** Most remote commands rely on aliases, shell functions, `$PATH` entries, conda/venv activation, or env vars that only exist once `~/.bashrc` / `~/.profile` is sourced. Non-interactive SSH does **not** source these by default, so the safe default is to wrap the remote command in a login shell:

```
ssh <host> 'bash -lc "cd <dir> && <cmd>"'
# or
ssh <host> 'source ~/.bashrc && cd <dir> && <cmd>'
```

This matters especially for `slaunch` — it is a shell function / alias defined in `~/.bashrc` (via `bashrc.sh`), not a binary on `$PATH`, so a plain `ssh <host> 'slaunch ...'` will fail with "command not found". The same pitfall hits conda envs, `s3_omni.py` wrappers, custom `PATH` additions, and any other bashrc-defined tooling.

Skip the profile load only for trivial built-ins where you are certain nothing custom is needed (e.g. `ls`, `cat`, `crontab -l`).

### Tmp-file trick for complex commands

When the remote command has awkward quoting (nested quotes, multi-line scripts, long pipelines, heredocs inside the command, etc.), avoid fighting the escaping. Instead:

1. Write the command to a temporary script under `~/tmp/` on the **remote** host (e.g. `~/tmp/ssh_run_<timestamp>.sh`).
2. `chmod +x` it and run it: `ssh <host> 'bash ~/tmp/ssh_run_<timestamp>.sh'`.
3. **Clean up on success** — remove the tmp file once the command completes successfully (`rm ~/tmp/ssh_run_<timestamp>.sh`). On failure, leave it so the user can inspect.

Use `~/tmp/` specifically (not `/tmp/`) so the scratch file lives under the user's home and is easy to find. Create the directory if missing (`mkdir -p ~/tmp`).

## Composing with other skills

`ssh_run` is a **wrapper** — it can carry any other skill's work onto the remote host.

Trigger: the user's message names another skill alongside `ssh_run` (e.g. "ssh_run meow on awscode", "/ssh_run /meow gcp", "run meow on the cluster via ssh_run"). In that case:

1. Resolve what the named skill would execute **locally** — read its SKILL.md from `cc/<skill_name>/SKILL.md` to determine the exact shell command(s) it runs.
2. Run that same command on the remote host via `ssh <host> '<cmd>'` instead of locally. Pass through any args the user gave.
3. Do NOT invoke the inner skill's local execution path. The inner skill's SKILL.md is read as a **recipe**, not executed in this environment.
4. Report using the rules below (Slurm job id parsing still applies).

Counter-case — if the user invokes the other skill **without** mentioning `ssh_run` (e.g. just `/meow`), run it locally as that skill normally would. `ssh_run` only engages when the user explicitly names it.

Example:
- User: `/meow` → run meow locally, cat on local console.
- User: `ssh_run meow on awscode` → read `cc/meow/SKILL.md`, find the shell command it runs, then `ssh awscode '<that command>'`.

If the inner skill's SKILL.md describes multiple steps or is non-trivial to translate into a single remote command, say so and ask the user how to proceed rather than guessing.

## Reporting

After running the ssh command:

1. **Announce it launched.** One short line: `Launched on <host>: <short description>`.
2. **Do NOT stream or summarize remote stdout** beyond what's needed to extract a Slurm job id. The user will check status separately.
3. **If the output contains a Slurm job id** (look for `Submitted batch job <N>` or a bare numeric id returned by `slaunch`/`sbatch`), report:

   ```
   Launched on <host>: <job name or command>
   Slurm job id: <jobid>
   Remote log: ~/log/slurm/*.<jobid>.e  (stderr)
               ~/log/slurm/*.<jobid>.o  (stdout)
   To tail:    ssh <host> 'tail -f ~/log/slurm/*.<jobid>.e'
   ```

   The number parsed out is the **Slurm job id** (the long-lived scheduler-assigned id that persists for the whole run and names the log files). It is **not** the short-lived OS PID of the `slaunch`/`sbatch` wrapper process — that PID exits as soon as submission completes and is useless for tailing logs or checking status. Always label it as "Slurm job id", never as "pid".

   The log files are on the **remote** machine, not local. Do not try to read them from the local filesystem.

4. If ssh exits non-zero, report the exit code and the last few lines of stderr so the user can diagnose. Do not retry automatically.

## What this skill does NOT do

- Does not edit files locally or remotely.
- Does not poll job status in a loop — for status, the user invokes `/checkrun` (local) or asks explicitly.
- Does not install or modify crontab entries without the user spelling out the exact schedule and command; if the user is vague, ask before `crontab -l | ... | crontab -`.
- Does not choose between `awscode` and `gcpcode` when the request is ambiguous — ask.
