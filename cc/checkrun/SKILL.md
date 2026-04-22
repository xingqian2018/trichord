---
name: checkrun
description: Check all active Slurm jobs for user xingqianx, then report the latest status from each job's log files.
user_invocable: true
---

## Constraints

- **Read-only**: Do NOT edit, write, or modify any file. Only read and run commands.
- **No confirmation prompts**: Execute all steps immediately without asking the user for permission to proceed.

## Step 1 — Get active Slurm jobs

Run:
```
squeue --format '%.10i %.9P %.24j %.9u %.2t %.12M %.12L %.6D %R' | grep 'xingqianx'
```
(alias: `sls`)

This returns a table of running jobs. From this output:
- Extract each job's **PID** (the numeric job ID in the first column).
- **Skip** any job whose name starts with `1N@debug_interactive` — those are interactive debug nodes, not real training runs.

## Step 2 — Find the log files

Log files live in `~/log/slurm/`. Each job has two files:
- `*.<pid>.e` — stderr (training progress, errors)
- `*.<pid>.o` — stdout

For each PID collected in Step 1, find the matching `.e` and `.o` files:
```
ls ~/log/slurm/ | grep '<pid>'
```

## Step 3 — Report job status (smart per job type)

Determine the job type from its name, then apply the appropriate reporting strategy.

### Job type: `imagegen_sst_*`

These jobs generate data files sequentially. The `.e` log contains lines like:
```
2026-03-17 21:11:48.400 | INFO | __main__:main:265 - Processing 000000161.json - 1000/1000 samples missing
2026-03-17 21:22:40.490 | INFO | __main__:main:307 - 000000161.json is completed
```

Read the last 5 lines of the `.e` file:
```
tail -5 ~/log/slurm/*.<pid>.e
```

Scan those lines for any `*.json` filename (pattern: `[0-9]+\.json`). Report the most recent filename found and its status:
- If the last line mentioning a filename contains `is completed` → that file is done.
- If it contains `Processing` → the job is actively working on that file.
- If neither keyword appears, just report the filename and the raw line for context.

Report: the filename, its status, and the timestamp of that log line.

### Job type: all others (default)

Fall back to showing the raw tail of the `.e` file:
```
tail -20 ~/log/slurm/*.<pid>.e
```

## Output format

For each active job (excluding `1N@debug_interactive`), print:

```
=== Job <pid>: <job name> ===
[imagegen_sst] Latest file: 000000161.json — completed (2026-03-17 21:22:40)
   -- or --
[imagegen_sst] Latest file: 000000161.json — IN PROGRESS (2026-03-17 21:11:48)
   -- or (default) --
<last 20 lines of .e>
```