"""Find live Kimi-K3 vLLM endpoints on Slurm (stdlib only, so it runs on the host outside any container).

    python3 k3_endpoints.py --out /path/k3_endpoints.json                 # once
    python3 k3_endpoints.py --out /path/k3_endpoints.json --every 300     # keep refreshing
"""

import argparse
import json
import os
import shutil
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

K3_MODEL = "moonshotai/Kimi-K3"


def http_json(url: str, body: Optional[dict] = None, timeout: float = 5) -> Any:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def endpoint_alive(base_url: str, timeout: float = 5) -> bool:
    try:
        return any(m.get("id") == K3_MODEL for m in http_json(f"{base_url}/models", timeout=timeout).get("data", []))
    except Exception:
        return False


def discover_slurm_endpoints(user: str = "sdharur", job_prefix: str = "k3-scale2048", port_base: int = 20000) -> list[str]:
    """Running jobs of `user` named `job_prefix*`; head node = first host, port = port_base + jobid % 20000."""
    if shutil.which("squeue") is None:
        return []
    out = subprocess.run(["squeue", "-h", "-u", user, "-t", "RUNNING", "-o", "%i %j %N"], capture_output=True, text=True, check=True).stdout
    candidates = []
    for line in out.splitlines():
        job_id, name, nodelist = line.split(maxsplit=2)
        if not name.startswith(job_prefix):
            continue
        head = subprocess.run(["scontrol", "show", "hostnames", nodelist], capture_output=True, text=True, check=True).stdout.split()[0]
        candidates.append(f"http://{head}:{port_base + int(job_id) % 20000}/v1")
    with ThreadPoolExecutor(max_workers=16) as pool:
        alive = list(pool.map(endpoint_alive, candidates))
    return [url for url, ok in zip(candidates, alive) if ok]


def load_endpoints() -> list[str]:
    """K3_ENDPOINTS (comma list) > K3_ENDPOINTS_FILE (json list) > squeue discovery."""
    if os.environ.get("K3_ENDPOINTS"):
        return [u.strip() for u in os.environ["K3_ENDPOINTS"].split(",") if u.strip()]
    path = os.environ.get("K3_ENDPOINTS_FILE")
    if path and Path(path).is_file():
        return json.loads(Path(path).read_text())
    return discover_slurm_endpoints()


def write_endpoints(out: str, user: str, job_prefix: str) -> int:
    urls = discover_slurm_endpoints(user, job_prefix)
    if urls:  # never overwrite a good list with an empty one (squeue hiccup)
        tmp = f"{out}.tmp.{os.getpid()}"
        Path(tmp).write_text(json.dumps(urls, indent=1))
        os.replace(tmp, out)
    return len(urls)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--user", default="sdharur")
    p.add_argument("--job-prefix", default="k3-scale2048")
    p.add_argument("--every", type=float, default=0, help="seconds between refreshes; 0 = run once")
    args = p.parse_args()
    while True:
        n = write_endpoints(args.out, args.user, args.job_prefix)
        print(f"{time.strftime('%H:%M:%S')} {n} live endpoint(s) -> {args.out}", flush=True)
        if args.every <= 0:
            break
        time.sleep(args.every)
