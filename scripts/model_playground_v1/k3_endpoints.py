"""Find live Kimi-K3 vLLM endpoints on Slurm (stdlib only, so it runs on the host outside any container).

    python3 k3_endpoints.py --out /path/k3_endpoints.json                 # once
    python3 k3_endpoints.py --out /path/k3_endpoints.json --every 300     # keep refreshing

Any other served model works the same way, e.g. the self-hosted Cosmos3 T2I replicas:
    python3 k3_endpoints.py --out /path/c3_endpoints.json --user xingqianx --job-prefix c3-xqx --model LeptonTR/Cosmos3-Super-Text2Image \
        --api-key-file ~/.config/cosmos3_serving/api_key
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


def http_json(url: str, body: Optional[dict] = None, timeout: float = 5, api_key: Optional[str] = None) -> Any:
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json", **({"Authorization": f"Bearer {api_key}"} if api_key else {})}
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def endpoint_alive(base_url: str, timeout: float = 5, model: str = K3_MODEL, api_key: Optional[str] = None) -> bool:
    try:
        return any(m.get("id") == model for m in http_json(f"{base_url}/models", timeout=timeout, api_key=api_key).get("data", []))
    except Exception:
        return False


def discover_slurm_endpoints(user: str = "sdharur", job_prefix: str = "k3-scale2048", port_base: int = 20000, model: str = K3_MODEL, api_key: Optional[str] = None) -> list[str]:
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
        alive = list(pool.map(lambda url: endpoint_alive(url, model=model, api_key=api_key), candidates))
    return [url for url, ok in zip(candidates, alive) if ok]


def load_endpoints() -> list[str]:
    """K3_ENDPOINTS (comma list) > K3_ENDPOINTS_FILE (json list) > squeue discovery."""
    if os.environ.get("K3_ENDPOINTS"):
        return [u.strip() for u in os.environ["K3_ENDPOINTS"].split(",") if u.strip()]
    path = os.environ.get("K3_ENDPOINTS_FILE")
    if path and Path(path).is_file():
        return json.loads(Path(path).read_text())
    return discover_slurm_endpoints()


def write_endpoints(out: str, user: str, job_prefix: str, model: str = K3_MODEL, api_key: Optional[str] = None) -> int:
    urls = discover_slurm_endpoints(user, job_prefix, model=model, api_key=api_key)
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
    p.add_argument("--model", default=K3_MODEL, help="served model id that /v1/models must list")
    p.add_argument("--api-key-file", default=None, help="file holding the bearer key of endpoints served with --api-key")
    p.add_argument("--every", type=float, default=0, help="seconds between refreshes; 0 = run once")
    args = p.parse_args()
    api_key = Path(args.api_key_file).expanduser().read_text().strip() if args.api_key_file else None
    while True:
        n = write_endpoints(args.out, args.user, args.job_prefix, args.model, api_key)
        print(f"{time.strftime('%H:%M:%S')} {n} live endpoint(s) -> {args.out}", flush=True)
        if args.every <= 0:
            raise SystemExit(0 if n else 1)
        time.sleep(args.every)
