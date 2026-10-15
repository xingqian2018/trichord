"""Worker sidecar for serving/broker.py (standard library only; runs on the replica's host, next to vLLM).

Keeps --max-inflight requests in flight: each slot long-polls the broker for one job of --model, forwards it to the
local vLLM (--backend, e.g. http://host:port) with the model's bearer key, and posts the response back. A heartbeat
renews the leases of the jobs in flight. If the backend is unreachable the job is released back to the queue.
The broker URL is re-read from BROKER_URL_FILE on every failure, so the broker can move to another node; with no
broker running the sidecar just waits, and the replica stays usable directly.
"""

import argparse
import base64
import json
import os
import socket
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

BROKER_URL_FILE = Path(os.environ.get("BROKER_URL_FILE", "/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/run/broker/broker_url"))
CREDENTIAL_FILE = Path.home() / "Project/trichord/credentials/gateway.json"
HEARTBEAT_SECONDS = 15
BACKEND_TIMEOUT = 3600


def http(method: str, url: str, body: dict | None, key: str, timeout: float) -> tuple[int, str, bytes]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.headers.get("Content-Type", "application/json"), r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Content-Type", "application/json"), e.read()


class Worker:
    def __init__(self, args):
        self.model, self.backend, self.max_inflight = args.model, args.backend.rstrip("/"), args.max_inflight
        self.key = json.loads(CREDENTIAL_FILE.read_text())[args.api_key_name].strip()
        self.worker_id = f"{socket.gethostname()}:{self.backend.rsplit(':', 1)[-1]}:{uuid.uuid4().hex[:6]}"
        self.held: set[str] = set()
        self.lock = threading.Lock()

    def broker(self) -> str | None:
        return BROKER_URL_FILE.read_text().strip().rstrip("/") if BROKER_URL_FILE.is_file() else None

    def call_broker(self, path: str, body: dict, timeout: float = 30) -> dict | None:
        url = self.broker()
        if url is None:
            return None
        try:
            status, _, payload = http("POST", url + path, {"worker_id": self.worker_id, "model": self.model, **body}, self.key, timeout)
            return json.loads(payload) if status == 200 else None
        except (urllib.error.URLError, OSError, json.JSONDecodeError):
            return None

    def heartbeat(self) -> None:
        while True:
            with self.lock:
                ids = list(self.held)
            self.call_broker("/worker/heartbeat", {"ids": ids})
            time.sleep(HEARTBEAT_SECONDS)

    def slot(self) -> None:
        while True:
            reply = self.call_broker("/worker/pull", {"max": 1}, timeout=60)
            if reply is None:
                time.sleep(10)
                continue
            for job in reply["jobs"]:
                with self.lock:
                    self.held.add(job["id"])
                try:
                    self.run(job)
                finally:
                    with self.lock:
                        self.held.discard(job["id"])

    def run(self, job: dict) -> None:
        try:
            status, content_type, payload = http("POST", self.backend + job["path"], job["body"], self.key, BACKEND_TIMEOUT)
        except (urllib.error.URLError, OSError) as e:
            print(f"backend unreachable ({type(e).__name__}: {e}), releasing {job['id']}", flush=True)
            self.call_broker("/worker/release", {"id": job["id"]})
            time.sleep(30)
            return
        self.call_broker("/worker/result", {"id": job["id"], "status": status, "content_type": content_type, "body_b64": base64.b64encode(payload).decode()}, timeout=120)

    def start(self) -> None:
        print(f"worker {self.worker_id}: model={self.model} backend={self.backend} max_inflight={self.max_inflight}", flush=True)
        threading.Thread(target=self.heartbeat, daemon=True).start()
        slots = [threading.Thread(target=self.slot, daemon=True) for _ in range(self.max_inflight)]
        for t in slots:
            t.start()
        for t in slots:
            t.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="served model name, e.g. Qwen/Qwen-Image-2.1")
    parser.add_argument("--backend", required=True, help="local vLLM root URL without /v1, e.g. http://host:port")
    parser.add_argument("--api-key-name", required=True, help="credentials/gateway.json entry of the model's key")
    parser.add_argument("--max-inflight", type=int, default=4)
    Worker(parser.parse_args()).start()
