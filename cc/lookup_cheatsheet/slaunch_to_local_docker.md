# slaunch → Local Docker (tmux) Conversion

When the GCP / AWS cluster has no available nodes, or you simply want to run a job on `n0` instead of going through Slurm, this is the recipe to convert any `slaunch ...` command from another cheatsheet into an equivalent **tmux + docker exec** invocation that runs inside the already-up `i4` container on the local machine.

This file is a **generic adapter** — it does not know which pipeline you're running. Pair it with the original cheatsheet (e.g. `golden_caption.md`, `image_caption_eval.md`) to get the actual flags/args.

---

## When to use

- GCP `squeue` shows your jobs stuck `PENDING (Priority)` for hours.
- AWS cluster is down / quota exhausted.
- You want a single quick rerun of a few stragglers without the slurm queue overhead.
- You're iterating on the script and want fast turnaround.

If the cluster is fine, just use `slaunch` — it parallelizes across the queue and survives forever.

---

## The mapping

Given a slurm command like:

```bash
CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4 \
LEPTON_API_QWEN3_VL_235B=<credential> \
slaunch cpu 1x1 <slurm_job_name> \
    <python_relative_code_path> \
    <args...>
```

substitute the parts as follows:

| slurm element | local-docker equivalent |
|---|---|
| `CONTAINER_WORKDIR=...` (env in shell) | `-e CONTAINER_WORKDIR=...` flag on `docker exec` |
| `LEPTON_API_QWEN3_VL_235B=<cred>` | `-e LEPTON_API_QWEN3_VL_235B=<resolved-credential>` flag on `docker exec` |
| `slaunch cpu 1x1 <slurm_job_name>` | `tmux new-session -d -s <TAG> "docker exec -w <workdir> -e ... i4 ..."` |
| `<python_relative_code_path>` | `.venv/bin/python <python_relative_code_path>` (inside docker) |
| `<args...>` | unchanged |
| (no equivalent) | `2>&1 | tee $LOG` to keep a host-side log after tmux closes |

`<TAG>` is a short identifier (e.g. `gc_v13s4`) — used as both the tmux session name and (optionally) for the log filename. Reuse the slurm job name with `_` instead of `-`/spaces.

Resolve credentials from `~/Project/trichord/credentials/gateway.json` and inline them as plain text; do **not** leave `<credential>` in the rendered command.

---

## Full skeleton

```bash
LOG=~/log/local_<TAG>_$(date +%Y%m%d_%H%M%S).log
echo "LOG=$LOG"
tmux new-session -d -s <TAG> "docker exec \
    -w /home/xingqianx/Project/imaginaire4 \
    -e CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4 \
    -e LEPTON_API_QWEN3_VL_235B=<resolved-credential> \
    i4 \
    .venv/bin/python \
        <python_relative_code_path> \
        <args...> \
        2>&1 | tee $LOG"
```

Key points:
- `tmux new-session -d -s <TAG>` — detached session named `<TAG>`. Survives ssh disconnect / shell exit.
- `docker exec -w <workdir> -e <env> i4 <cmd>` — runs `<cmd>` non-interactively inside the already-up `i4` container. **Do not** use `dockerrun` here; that alias does `docker exec -it i4 bash` (interactive) which can't be driven non-interactively.
- `-e VAR=value` mirrors the env-prefix-on-shell-command form used by slurm. Add as many `-e` flags as the original command had env vars.
- `i4` is the container name. Confirm with `docker ps` — should show `Up`. If not running, run `dockerrun` once interactively (it calls `docker start i4` and drops you in a shell), exit (Ctrl-D), then proceed.
- `2>&1 | tee $LOG` — keeps a host-side log even after the python process exits and the tmux session closes. The log file is your source of truth; tmux history is ephemeral.

---

## Pre-flight checklist

1. **Container is up.** `docker ps | grep i4` → should say `Up`. If missing or `Exited`, run `dockerrun` once.
2. **Venv exists.** `ls /home/xingqianx/Project/imaginaire4/.venv/bin/python` should resolve.
3. **Credentials resolvable.** `~/Project/trichord/credentials/gateway.json` has the keys you need (e.g. `LEPTON_API_QWEN3_VL_235B`, `NVIDIA_GATEWAY`).
4. **No tag collision.** `tmux ls | grep <TAG>` must be empty. Pick a fresh tag if it isn't.
5. **`~/log/` exists.** `mkdir -p ~/log` is cheap insurance.

---

## Worked example — converting a `golden_caption_v13s4` slurm command

**Original (from `golden_caption.md`):**

```bash
CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4 \
LEPTON_API_QWEN3_VL_235B=<credential> \
slaunch cpu 1x1 golden_caption_v13s4 \
    projects/cosmos3/vfm/evaluation/captioning/golden_caption/stage4_camera_and_style.py \
    --input_folder s3://nv-00-10206-vfm/debug/xingqianx/evaluation/CosCapBenchImage/V1/ \
    --input_credential credentials/gcs.secret \
    --input_json_folder s3://nv-00-10206-vfm/debug/xingqianx/evaluation/CosCapBenchImage/golden_caption_v13_q235bg3p/stage3/ \
    --input_json_credential credentials/gcs.secret \
    --output_folder s3://nv-00-10206-vfm/debug/xingqianx/evaluation/CosCapBenchImage/golden_caption_v13_q235bg3p/stage4/ \
    --output_credential credentials/gcs.secret \
    --num_concurrency 32 \
    --batch_size 100 \
    --max_retry 3 \
    --force_gen_model qwen3-vl-235b-a22b-instruct
```

**Converted (tmux + docker, credential inlined, `<TAG>` = `gc_v13s4`):**

```bash
LOG=~/log/local_v13s4_$(date +%Y%m%d_%H%M%S).log
tmux new-session -d -s gc_v13s4 "docker exec \
    -w /home/xingqianx/Project/imaginaire4 \
    -e CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4 \
    -e LEPTON_API_QWEN3_VL_235B=4pdJp2M7ejcS1sqLjS4lhAFWS5j1nKRw \
    i4 \
    .venv/bin/python \
        projects/cosmos3/vfm/evaluation/captioning/golden_caption/stage4_camera_and_style.py \
        --input_folder s3://nv-00-10206-vfm/debug/xingqianx/evaluation/CosCapBenchImage/V1/ \
        --input_credential credentials/gcs.secret \
        --input_json_folder s3://nv-00-10206-vfm/debug/xingqianx/evaluation/CosCapBenchImage/golden_caption_v13_q235bg3p/stage3/ \
        --input_json_credential credentials/gcs.secret \
        --output_folder s3://nv-00-10206-vfm/debug/xingqianx/evaluation/CosCapBenchImage/golden_caption_v13_q235bg3p/stage4/ \
        --output_credential credentials/gcs.secret \
        --num_concurrency 32 \
        --batch_size 100 \
        --max_retry 3 \
        --force_gen_model qwen3-vl-235b-a22b-instruct \
        2>&1 | tee $LOG"
```

Diff at a glance:
- `CONTAINER_WORKDIR=... ` (shell prefix) → `-e CONTAINER_WORKDIR=...` (docker exec flag)
- `LEPTON_API_QWEN3_VL_235B=<credential>` → `-e LEPTON_API_QWEN3_VL_235B=4pdJp2M7ejcS1sqLjS4lhAFWS5j1nKRw` (resolved)
- `slaunch cpu 1x1 golden_caption_v13s4` → `tmux new-session -d -s gc_v13s4 "docker exec -w ... -e ... i4`
- `projects/.../stage4_camera_and_style.py` → `.venv/bin/python projects/.../stage4_camera_and_style.py`
- Same args after that
- Added `2>&1 | tee $LOG` and the closing `"`

---

## Sequential vs parallel

**Parallel** — multiple jobs at once, each its own tmux session inside the same `i4` container. Good for unrelated jobs that don't compete for the same VLM gateway:

```bash
tmux new-session -d -s gc_v13s3 "docker exec ... stage3_... | tee $LOG_A"
tmux new-session -d -s gc_v13s4 "docker exec ... stage4_... | tee $LOG_B"
```

**Sequential** — chain jobs in one tmux session via a host-side wrapper script. Good when downstream stages need upstream outputs, or when you want to be polite to a rate-limited VLM:

1. Write a wrapper shell script under `~/tmp/` that runs each step in order.
2. Tmux launches `docker exec ... bash <wrapper.sh>`.
3. Each step is `.venv/bin/python … 2>&1 | tee <step-specific-log>`.

Skeleton wrapper:

```bash
#!/bin/bash
set -x
cd /home/xingqianx/Project/imaginaire4

.venv/bin/python <step1.py> <args> 2>&1 | tee /home/xingqianx/log/local_step1.log
.venv/bin/python <step2.py> <args> 2>&1 | tee /home/xingqianx/log/local_step2.log
.venv/bin/python <step3.py> <args> 2>&1 | tee /home/xingqianx/log/local_step3.log
echo "=== ALL STEPS COMPLETE at $(date) ==="
```

Then:

```bash
tmux new-session -d -s gc_seq "docker exec -w /home/xingqianx/Project/imaginaire4 -e CONTAINER_WORKDIR=... -e LEPTON_API_QWEN3_VL_235B=... i4 bash /home/xingqianx/tmp/<wrapper>.sh 2>&1 | tee ~/log/local_seq_master.log"
```

The host home is volume-mounted into `i4`, so `~/tmp/<wrapper>.sh` on the host is the same file as `/home/xingqianx/tmp/<wrapper>.sh` inside the container.

---

## Watching / sanity-checking

```bash
tmux ls                                # session still alive?
tmux attach -t <TAG>                   # interactive view (Ctrl-b d to detach)
tail -f ~/log/local_<TAG>_*.log        # passive tail
docker exec i4 ps -ef | grep python    # confirm the python pid is alive inside the container
```

When the python process exits, tmux closes the session automatically — `tmux ls` no longer listing `<TAG>` is the **"done"** signal, **not** a stuck signal. The log file is the source of truth for whether it succeeded.

---

## Common pitfalls

- **"Looks stuck."** Usually it just finished. Verify with `tmux ls` (session gone) and the log's final `Summary:` block. The most-recent stdout line in the log doesn't update once the process exits — this can mislead you into thinking it hung.
- **`429 Too Many Requests`.** LLM gateway rate-limit. Re-run the same command — most pipeline scripts skip already-complete files, so a re-run is idempotent. If 429s persist, drop `--num_concurrency` (e.g. 32 → 8) or wait a few minutes.
- **VLM timeout (`exceeded 100s` / `exceeded 400s`).** Same gateway pressure — bump `--max_retry`, lower concurrency, or switch judge/gen model.
- **`slaunch: command not found`.** You're on `n0` (no slurm). Expected — that's why you're using this recipe.
- **Container not running.** `docker ps` doesn't show `i4`. Run `dockerrun` once to `docker start` and drop you in a shell, exit, then re-run the tmux command.
- **Persistent content failures (NED mismatch, "not a valid YAML list", "expected N items, got M").** These are model-behavior issues, not transient. Retrying the same command won't help. Try a different `--force_gen_model` / `--force_judge_model` or accept the partial result.
- **Credential placeholder leaked.** If you forget to resolve `<credential>` from `gateway.json`, the python script will raise an auth error in the first few seconds. Check the log head, kill the tmux session, fix the env var, re-launch.

---

## Listing / cleaning up sessions

```bash
tmux ls                          # all sessions
tmux kill-session -t <TAG>       # stop a specific session early
tmux kill-server                 # stop EVERYTHING (use with care)
```

`docker exec` processes outlive their `tmux` session if and only if the wrapper detaches them. The pattern in this doc keeps them tied — kill tmux, the python process dies. That's usually what you want.
