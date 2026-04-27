---
name: cheatsheet_lookup
description: Look up a named cheatsheet and show its contents. Each cheatsheet is a sibling .md file in this skill's folder describing a specific recipe or workflow. Invoke as /cheatsheet_lookup <name>. This skill only reads and displays — it does not run anything.
user_invocable: true
---

# cheatsheet_lookup

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

## Conversion to `.vscode/launch.json` for debug purpose

If the user asks to set up a VSCode debug config for one of the cheatsheet runs (e.g. golden caption, version15 stage3, …):

- The target is `<project_repo>/.vscode/launch.json`, which is a symlink to `<project_repo>/.vscode/launch_gsb.json`. Writing through the symlink is fine — both paths refer to the same file.
- If `<project_repo>/.vscode/launch.json` is not a symlink to `<project_repo>/.vscode/launch_gsb.json`, you need to symlink it yourself.
- This file lives on a **remote cluster** (GCP or AWS), not on `n0`. The user must specify which one; use the `ssh_run` skill to write the file on the chosen host.
- **Rewrite the entire file** in the format below. Do not merge with prior contents.

```JSON
{
    "version": "0.2.0",
    "configurations": [
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

The `///` lines are non-standard JSON, but VSCode's `launch.json` parser tolerates them. Keep them as a visual divider.

Repo lookup table:

| Repo | Branch | Purpose |
|---|---|---|
| `~/Project/imaginaire4` | `xingqianx/cosmos3_aid` | golden caption, image caption evaluation |
| `~/Project/imaginaire4_sila` | `xingqianx/pipe_text_render` | wedds sharding, dataset creation |

