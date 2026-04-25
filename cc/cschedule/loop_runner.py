#!/usr/bin/env python3
"""cschedule Backend C — a self-contained loop runner.

Spawned with `nohup python loop_runner.py <args> &`. It sleeps until the next
period boundary, runs the configured shell command, logs the result, and loops.
Survives the spawning shell so long as the host kernel keeps the process alive.

Why this exists: in some sandboxed environments BOTH the Claude harness's REPL
CronCreate AND the OS user crontab get periodically wiped. A user-space loop
process bypasses both, since it's just a regular long-running process that ticks
on its own internal clock.

Single-instance contract: only one runner per --tag. On startup the runner tries
to acquire an exclusive flock on `/tmp/cschedule_<tag>.pid`. If another runner
already holds it, this one exits immediately so we don't double-fire.

Cancel a runner with: `kill $(cat /tmp/cschedule_<tag>.pid)`.
"""

import argparse
import datetime as dt
import fcntl
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True, help="Unique tag for this runner")
    p.add_argument(
        "--period-seconds",
        type=int,
        required=True,
        help="Seconds between fires; the runner aligns to wall-clock multiples",
    )
    p.add_argument(
        "--cmd",
        required=True,
        help="Shell command to run on each fire (passed to /bin/sh -c)",
    )
    p.add_argument(
        "--log",
        default=None,
        help="Log file path; defaults to /tmp/cschedule_<tag>.log",
    )
    return p.parse_args()


def acquire_singleton_lock(tag: str) -> int:
    """Acquire an exclusive lock on /tmp/cschedule_<tag>.pid; return its fd.

    If another runner holds the lock, exit 0 (it's not an error — we just
    don't want to double-spawn).
    """
    pid_path = Path(f"/tmp/cschedule_{tag}.pid")
    fd = os.open(str(pid_path), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        # Already running — silently exit so re-invoking is idempotent.
        os.close(fd)
        sys.exit(0)
    os.ftruncate(fd, 0)
    os.write(fd, f"{os.getpid()}\n".encode())
    os.fsync(fd)
    return fd


def sleep_to_next_boundary(period_seconds: int) -> None:
    """Sleep until the next wall-clock multiple of period_seconds."""
    now = time.time()
    wait = period_seconds - (now % period_seconds)
    if wait <= 0:
        wait += period_seconds
    time.sleep(wait)


def main() -> None:
    args = parse_args()
    log_path = Path(args.log or f"/tmp/cschedule_{args.tag}.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    acquire_singleton_lock(args.tag)  # holds for life of process

    # Re-open stdout/stderr to the log so anything we print goes there too.
    log_fd = os.open(str(log_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(log_fd, 1)
    os.dup2(log_fd, 2)

    print(
        f"--- {dt.datetime.now().isoformat(timespec='seconds')} runner started "
        f"tag={args.tag} period={args.period_seconds}s pid={os.getpid()} ---",
        flush=True,
    )

    while True:
        sleep_to_next_boundary(args.period_seconds)
        ts = dt.datetime.now().isoformat(timespec="seconds")
        print(f"--- {ts} cron-fire (tag={args.tag}) ---", flush=True)
        try:
            # Shell out so the user can pass an arbitrary pipeline.
            subprocess.run(args.cmd, shell=True, check=False)
        except Exception as e:  # pragma: no cover — defensive
            print(f"!!! runner exception: {e}", flush=True)


if __name__ == "__main__":
    main()
