#!/usr/bin/env python3
"""reports_xx/nemo6/ladder_server_argv.py - the argv of Megatron's dynamic text-generation server (tools/run_dynamic_text_generation_server.py,
OpenAI /v1/completions) for ONE of our ladder checkpoints, so the nemo6 suite (nemo6_ladder.sbatch) can run on it with the same lm_eval
client as the external references (user 11-05: "全量测评" on KD 2.5 · PROD1_DPSK · 新数据 q3).

    python3 ladder_server_argv.py --run-dir <run dir> [--iter N] [--max-seq 16384] [--buffer-gb 80] [--out argv.txt]

Starts from the run's exact training argv (<run dir>/argv.txt, the same source as tools/eval_argv.py / the doc eval), so every model flag
(per-layer SWA window, NoPE layers, QK-norm, MoE router / shared expert, tokenizer) is the trained one. Dropped, with their values:
  * flags of our own entry points that Megatron's parser does not know: --hyperp-*, --kd-*, --pack-plan / --packplan-*, --doc-*,
    --val-*, --eval-at-iterations;
  * output / logging side effects of a training run: --save, --load (re-set), --tensorboard-dir, --wandb-*, --log-*-to-tensorboard,
    --exit-duration-in-mins, --check-weight-hash-across-dp-replicas-interval.
Added: --load <run>/checkpoints [--ckpt-step N], --no-load-optim --no-load-rng, --inference-max-seq-length (16384 = the refs' vLLM
max-model-len), --attention-backend flash (generation path validated on GB300 for the SWA/NoPE ladder in swa/eval/megatron_inproc_lm.py),
the dynamic-batching KV buffer (the tool's default is 2 GB), --dist-ckpt-strictness log_unexpected. Parallelism stays as trained
(TP 1; EP as trained, clamped to the server's 4 ranks on one node: q4 / q5 train with EP 8). Standard library only."""
from __future__ import annotations

import argparse
import os
import sys

DROP_PREFIX = ("--hyperp-", "--kd-", "--pack-plan", "--packplan-", "--doc-", "--val-", "--wandb-")
DROP_EXACT = {"--save", "--load", "--tensorboard-dir", "--eval-at-iterations", "--exit-duration-in-mins",
              "--check-weight-hash-across-dp-replicas-interval", "--log-validation-ppl-to-tensorboard", "--log-timers-to-tensorboard",
              "--attention-backend", "--inference-max-seq-length", "--no-load-optim", "--no-load-rng", "--ckpt-step",
              "--skip-train", "--disable-gloo-process-groups"}   # gloo: the inference engine's CPU-side collectives may need it


def render(train_argv: list[str], run_dir: str, it: int | None, max_seq: int, buffer_gb: float, max_tokens: int,
           world: int = 4) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(train_argv):
        t = train_argv[i]
        j = i + 1
        while j < len(train_argv) and not train_argv[j].startswith("--"):   # the flag's values (nargs: one, several or none)
            j += 1
        if t.startswith("--") and (t in DROP_EXACT or t.startswith(DROP_PREFIX)):
            i = j
            continue
        out.extend(train_argv[i:j])
        i = j
    # q4 / q5 train with EP 8 on many nodes; the server is 4 ranks on one node. torch_dist reshards expert weights on load (no optimizer
    # state loaded), so clamp EP to the largest divisor of the world size that is <= the training EP (= tools/eval_argv.py, the doc eval).
    if "--expert-model-parallel-size" in out:
        k = out.index("--expert-model-parallel-size")
        ep_train = int(out[k + 1])
        out[k + 1] = str(max(d for d in range(1, min(ep_train, world) + 1) if world % d == 0 and ep_train % d == 0))
    out += ["--load", os.path.join(run_dir, "checkpoints"), "--no-load-optim", "--no-load-rng",
            "--attention-backend", "flash", "--inference-max-seq-length", str(max_seq),
            "--inference-dynamic-batching-buffer-size-gb", str(buffer_gb),
            "--inference-dynamic-batching-max-tokens", str(max_tokens),
            "--dist-ckpt-strictness", "log_unexpected",
            # decoding defaults for fields a request omits = vLLM's under the references' `--generation-config vllm`: the task's own
            # settings only. Unset, the server would apply the TOKENIZER dir's generation_config.json (Qwen3-32B instruct:
            # temperature 0.6, top_p 0.95, top_k 20) to every sampled task that does not set top_p / top_k itself.
            "--default-temperature", "1.0", "--default-top-p", "1.0", "--default-top-k", "0"]
    if it is not None:
        out += ["--ckpt-step", str(it)]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--iter", type=int, default=None, help="checkpoint iteration (default: latest_checkpointed_iteration.txt)")
    ap.add_argument("--max-seq", type=int, default=16384)
    ap.add_argument("--buffer-gb", type=float, default=80.0)
    ap.add_argument("--max-tokens", type=int, default=32768, help="tokens per engine step (largest prompt prefilled in one step)")
    ap.add_argument("--out")
    a = ap.parse_args()
    run = os.path.abspath(a.run_dir)
    argv = open(os.path.join(run, "argv.txt")).read().split()
    ck = os.path.join(run, "checkpoints")
    it = a.iter if a.iter is not None else int(open(os.path.join(ck, "latest_checkpointed_iteration.txt")).read().strip())
    d = os.path.join(ck, f"iter_{it:07d}")
    if not os.path.isfile(os.path.join(d, ".metadata")):
        print(f"no torch_dist checkpoint at {d}", file=sys.stderr)
        return 2
    out = render(argv, run, it, a.max_seq, a.buffer_gb, a.max_tokens)
    text = "\n".join(out) + "\n"
    if a.out:
        open(a.out, "w").write(text)
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
