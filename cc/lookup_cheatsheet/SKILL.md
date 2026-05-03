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
> 2. **Ask the user which cluster** (`awscode` / `gcpcode`) if they have not already specified one. Do not guess.
> 3. Use `ssh_run` to dispatch the command — never paste a raw `ssh <host> '...'` command unless `ssh_run` cannot express it.
>
> This rule applies to every current and future cheatsheet — pipeline-specific files do not need to repeat the remote-launch instructions.

## Conversion to `.vscode/launch.json` for debug purpose

If the user asks to set up a VSCode debug config for one of the cheatsheet runs (e.g. golden caption, version15 stage3, …):

- The target is `<project_repo>/.vscode/launch.json`, which is a symlink to `<project_repo>/.vscode/launch_gsb.json`. Writing through the symlink is fine — both paths refer to the same file.
- If `<project_repo>/.vscode/launch.json` is not a symlink to `<project_repo>/.vscode/launch_gsb.json`, you need to symlink it yourself.
- This file lives on a **remote cluster** (GCP or AWS), not on `n0` — see the **Remote-run via `ssh_run`** section above. Ask the user which cluster, then use `ssh_run` to read/write the file on the chosen host.
- **Append, do not overwrite.** The file accumulates a history of recent debug sessions so the user can switch back to an earlier config in one click. Procedure:
  1. **Read** the existing `launch_gsb.json` first (via `ssh <host> 'cat ...'`). If the file is empty or missing, start from the skeleton below.
  2. **Comment out only the currently-live entry/entries.** Wrap each live `{ ... }` object in a `/* ... */` block comment. Leave the `"version"` field, the array brackets, and **all already-commented entries** untouched — never re-format, merge, or clean up the existing mothball blocks. They pipeline up: each successive update just appends one more `/* ===== superseded ===== */` block on top of the existing ones. VSCode's `launch.json` parser tolerates `/* ... */` the same way it tolerates the `//` dividers, so an arbitrarily long stack is fine.
  3. **Append the new entry** as the last (live) element of `"configurations"`, in the format below. Place any trailing comma **inside** the block comment so the live array stays well-formed regardless of how many prior entries got mothballed.
  4. **Write** the merged file back via the `ssh_run` write-locally → scp pattern (no heredocs).
- **An update always introduces a new live entry.** Commenting out the prior config without adding a new live one is not a valid outcome — if there is nothing new to add, the user is not asking for an update. Ask them what the new config should be before touching the file.

Skeleton — what the file should look like after one append on top of one prior run:

```JSON
{
    "version": "0.2.0",
    "configurations": [

        /* ===== superseded =====
        {
            "name": "<old_debugname>",
            ...prior config body...
        },
        */

        {
            "name": "<debugname>",
            "type": "debugpy",
            "request": "launch",
            "module": "torch.distributed.run",
            "cwd": "${workspaceFolder}",
            "console": "integratedTerminal",
            "justMyCode": false,
            "env": {
                "PYTHONPATH": "${workspaceFolder}",
                <other_specified_env_variables>
            },

            /////////////////
            // <debugname> //
            /////////////////

            "args": [
                "--nproc_per_node=1",
                "--master_port=<random_between_20000_to_30000>",
                "<code_path_relative_to_project_repo>",
                <double_quoted_comma_delimited_arguments>
            ]
        }
    ]
}
```

Field guide:
- `<debugname>` — short name for the run; reuse it in both `"name"` and the `// <debugname> //` banner.
- `<other_specified_env_variables>` — additional `"KEY": "value"` entries, comma-separated. Note the trailing comma after `"${workspaceFolder}"`. If there are none, omit this line and drop that comma.
- `<code_path_relative_to_project_repo>` — path to the script entry point, relative to the repo root, double-quoted.
- `<double_quoted_comma_delimited_arguments>` — each CLI flag and value as its own quoted string, comma-separated. Translate the cheatsheet's shell command into this form (drop backslash continuations; keep one token per array entry).
- `<random_between_20000_to_30000>` — pick any free port in that range.

The `///` and `/* ... */` markers are non-standard JSON, but VSCode's `launch.json` parser tolerates both. Use `///` as a visual divider for the live entry; use `/* ... */` to mothball superseded entries.

Repo lookup table:

| Repo | Branch | Purpose |
|---|---|---|
| `~/Project/imaginaire4` | `xingqianx/cosmos3_aid` | golden caption, image caption evaluation |
| `~/Project/imaginaire4_sila` | `xingqianx/pipe_text_render` | wedds sharding, dataset creation |

