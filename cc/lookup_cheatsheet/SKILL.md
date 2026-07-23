---
name: lookup_cheatsheet
description: Look up a named cheatsheet and show its contents. Each cheatsheet is a sibling .md file in this skill's folder describing a specific recipe or workflow. Invoke as /lookup_cheatsheet <name>. This skill only reads and displays — it does not run anything.
user_invocable: true
---

# lookup_cheatsheet

Sibling `.md` files (except `SKILL.md`) are cheatsheets. Filename without `.md` is the name.

**Lookup-only.** Display contents; never execute, launch, or modify anything.

## Steps

1. Parse the argument as `<name>` → `<name>.md`.
2. If no argument or no match, list available cheatsheets (`ls` the folder) and ask the user to pick. No guessing, no fuzzy-matching.
3. Read and show the matching file.
4. **Keep the template's indented multi-line form** (backslash continuations, `VAR=... \` prefixes, per-line args). Substitute placeholders the user gave you; do not flatten. Only produce a one-liner if the user explicitly asks.
5. **Resolve credentials locally** (env var, `credentials/*.json`, `~/.aws/credentials`, `~/.netrc`, etc.) and inline them into the command. If unresolvable, leave the `<placeholder>` and say where to set it.
6. Leave any unresolved placeholders as `<angle-bracket>` and call them out. Never invent values.

## Local-run conversion (cross-cutting)

If the user asks to run any cheatsheet's `slaunch ...` command **locally** (on `n0`, no slurm), or to convert a remote/GCP/AWS submission into a docker-local invocation — **always** consult the **`slaunch_to_local_docker`** cheatsheet for the canonical recipe (tmux + `docker exec`, credential handling, watching, pitfalls). Apply that recipe on top of the pipeline-specific args from the cheatsheet the user named. Do **not** improvise the local form.

This rule applies to every current and future cheatsheet — pipeline-specific files do not need to repeat the local-run instructions.

## Remote-run via `ssh_run` (cross-cutting)

> **Important — read before launching anything.**
>
> Most cheatsheet commands (`slaunch ...`, `slurm` submissions, file writes that land in a project repo, etc.) must execute on a **remote cluster** (GCP or AWS), **not** on `n0`. Whenever the user asks to **launch** a command, **submit** a job, or **write** a file that belongs on the remote repo:
>
> 1. **Always consult the `ssh_run` skill first** — it documents the canonical remote-launch recipe (host selection, command wrapping, slurm-id capture, log path resolution).
> 2. **Ask the user which cluster** (`awscode` / `gcpcode` / `gb300`) if they have not already specified one. Do not guess.
> 3. Use `ssh_run` to dispatch the command — never paste a raw `ssh <host> '...'` command unless `ssh_run` cannot express it.
>
> This rule applies to every current and future cheatsheet — pipeline-specific files do not need to repeat the remote-launch instructions.

## Conversion to `.vscode/launch.json` for debug purpose

If the user asks to set up a VSCode debug config for one of the cheatsheet runs (e.g. golden caption, version15 stage3, …), defer to the **`create_vscode_launch_json`** skill — it owns the canonical recipe (symlink convention, mothball-and-append workflow, skeleton, repo lookup table). Do **not** improvise the launch.json edit here.

