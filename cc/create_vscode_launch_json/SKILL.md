---
name: create_vscode_launch_json
description: Convert a cheatsheet / shell run command into a VSCode debug entry on a remote cluster. Writes a per-run `.vscode/launch_<run_name>.json` and points the `launch.json` symlink at it. Reusing the same `<run_name>` overwrites in place — history lives as sibling files, not as in-file comments. Invoke as /create_vscode_launch_json.
user_invocable: true
---

# create_vscode_launch_json

Set up (or update) a VSCode debug config for a cheatsheet-style run — e.g. golden caption, version15 stage3, a unigenbench eval, etc. The target lives in `<project_repo>/.vscode/` on the **remote cluster** (GCP or AWS), not on `n0`.

## Preconditions

The user must specify (or you must ask):

1. **Which cluster** — `awscode` or `gcpcode`. Do not guess.
2. **Which project repo** — `~/Project/imaginaire4`, `~/Project/imaginaire4_sila`, etc. See the lookup table at the bottom.
3. **The run name** — short label, used as the filename suffix, the `"name"` field, and the `// <run_name> //` banner. Be **concise**: lowercase, words joined by `_`, ideally one or two tokens (e.g. `gsb`, `golden_caption`, `ugb1170L`, `v15_stage3`). No spaces, no dates, no full sentences.
4. **The script entry point** — path relative to the repo root (e.g. `projects/cosmos3/vfm/evaluation/text_to_image/inference_unigenbench_distributed.py`).
5. **The CLI args** — typically the body of a `slaunch ...` command from a cheatsheet.
6. **Any extra env vars** — `PYTHONPATH=${workspaceFolder}` is always set; ask if more are needed.

All remote file reads/writes go through the **`ssh_run`** skill — never paste raw `ssh <host> '...'` here unless `ssh_run` cannot express it. In particular: **no heredocs**. Use the `ssh_run` write-locally → `scp` → ssh-execute pattern.

## File layout

`<project_repo>/.vscode/` accumulates one JSON file per run:

```
.vscode/
├── launch.json                       # symlink → launch_<active_run>.json
├── launch_gsb.json
├── launch_golden_caption.json
├── launch_ugb1170L.json
└── ...
```

- `launch.json` is **always a symlink**. Never write to `launch.json` directly.
- Each `launch_<run_name>.json` is a self-contained, single-entry config — no `/* ... */` mothball blocks, no `///` banners around superseded entries, no historical layering inside the file.
- Switching active config = repoint the symlink. Done.

## Procedure

1. **Inspect** the current state:

   ```
   ssh <host> 'ls -la <repo>/.vscode/'
   ```

   Note the current symlink target and which `launch_<run_name>.json` files already exist.

2. **Pick `<run_name>`.** If the user gave one, use it verbatim (after concision check). Otherwise propose a short one and confirm.

3. **Write `launch_<run_name>.json`** using the skeleton below.

   - If a file with the same name **already exists**, overwrite it. Same name = same purpose; we don't version inside the file. Don't read-and-merge, don't preserve old args, just write the fresh single-entry config.
   - Use the `ssh_run` write-locally → `scp` pattern: write to a local temp file, scp to `<host>:<repo>/.vscode/launch_<run_name>.json`. No heredocs.

4. **Repoint the symlink** to the new file:

   ```
   ssh <host> 'cd <repo>/.vscode && ln -sfn launch_<run_name>.json launch.json'
   ```

   `ln -sfn` is force + no-deref: it replaces an existing symlink without following it into the target. Always use this exact form — plain `ln -sf` will write *through* an existing symlink and clobber the file it points at.

5. **Verify:**

   ```
   ssh <host> 'ls -la <repo>/.vscode/launch.json'
   ```

   Confirm the symlink resolves to `launch_<run_name>.json`.

## Skeleton — what `launch_<run_name>.json` should contain

```JSON
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "<run_name>",
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
            // <run_name> //
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

Single live entry, no superseded blocks. The `///` banner is just a visual divider inside the live entry — VSCode's `launch.json` parser tolerates `//` line comments.

## Field guide

- `<run_name>` — concise, lowercase, `_`-separated. Reused as the filename suffix, the `"name"` field, and the `// <run_name> //` banner.
- `<other_specified_env_variables>` — additional `"KEY": "value"` entries, comma-separated. Note the trailing comma after `"${workspaceFolder}"`. If there are none, omit this line and drop that comma.
- `<code_path_relative_to_project_repo>` — path to the script entry point, relative to the repo root, double-quoted.
- `<double_quoted_comma_delimited_arguments>` — each CLI flag and value as its own quoted string, comma-separated. Translate the cheatsheet's shell command into this form (drop backslash continuations; keep one token per array entry).
- `<random_between_20000_to_30000>` — pick any free port in that range.

## Repo lookup table

| Repo                         | Branch                       | Purpose                                  |
|------------------------------|------------------------------|------------------------------------------|
| `~/Project/imaginaire4`      | `xingqianx/cosmos3_aid`      | golden caption, image caption evaluation |
| `~/Project/imaginaire4_sila` | `xingqianx/pipe_text_render` | wedds sharding, dataset creation         |

## What this skill does NOT do

- Does not run the debug config — it only writes the JSON and repoints the symlink. The user launches it from VSCode.
- Does not write to `launch.json` directly — `launch.json` is always a symlink, and writing through it would clobber whichever `launch_<run_name>.json` it currently points at.
- Does not version configs *inside* a file (no `/* ... */` mothball blocks, no stacked superseded entries). History lives as sibling files; switching = `ln -sfn`.
- Does not delete old `launch_<run_name>.json` files. They stick around as a one-symlink-flip away from being live again.
- Does not pick the cluster, repo, or `<run_name>` when ambiguous — ask the user.
- Does not bypass `ssh_run` for the remote read/write — heredocs are banned (see `ssh_run` for the gory details).
