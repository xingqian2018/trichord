"""Pull-based request broker for the self-hosted model replicas (standard library only).

Clients talk to ONE stable URL instead of a list of replica endpoints:
    POST <broker>/v1/...        any OpenAI-style JSON request (non-streaming); "model" picks the queue
    GET  <broker>/v1/models     models that currently have live workers
GPU replicas run serving/broker_worker.py next to vLLM; each worker keeps a fixed number of requests in flight
by pulling from the queue of its model, so idle replicas take work and busy ones do not get more. Replicas can
come and go (Slurm requeue) without any endpoint list:
    POST /worker/pull       {worker_id, model, max}       long-polls up to 20 s, returns jobs and leases them
    POST /worker/heartbeat  {worker_id, model, ids}       renews the leases of the jobs a worker still holds
    POST /worker/result     {worker_id, id, status, content_type, body_b64}
    POST /worker/release    {worker_id, id}               puts a job back (worker cannot reach its backend)
    GET  /stats                                           queue depth, in-flight and live workers per model
A job whose lease expires (worker died) is requeued at the front, up to MAX_ATTEMPTS times.
Auth: every request carries the bearer key of its model (the SLURM_*_API_KEY entries of credentials/gateway.json),
for clients and workers alike.
"""

import argparse
import base64
import collections
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

CREDENTIAL_FILE = Path.home() / "Project/trichord/credentials/gateway.json"
MODEL_KEYS = {
    "moonshotai/Kimi-K3": "SLURM_KIMI_K3_API_KEY",
    "mimo-v2.6-pro-rl": "SLURM_MIMO_API_KEY",
    "LeptonTR/Cosmos3-Super-Text2Image": "SLURM_COSMOS3_API_KEY",
    "Qwen/Qwen-Image-2.1": "SLURM_QWEN_IMAGE_API_KEY",
    "Qwen/Qwen-Image-2.1-PE-T2I": "SLURM_QWEN_IMAGE_PE_API_KEY",
}
JOB_TIMEOUT = 3600
LEASE_SECONDS = 60
WORKER_ALIVE_SECONDS = 60
PULL_WAIT_SECONDS = 20
MAX_ATTEMPTS = 3


class Job:
    def __init__(self, model: str, path: str, body: dict):
        self.id = uuid.uuid4().hex
        self.model, self.path, self.body = model, path, body
        self.created = time.time()
        self.attempts = 0
        self.worker = None
        self.lease_until = 0.0
        self.done = threading.Event()
        self.status, self.content_type, self.payload = 504, "application/json", b""

    def finish(self, status: int, content_type: str, payload: bytes) -> None:
        self.status, self.content_type, self.payload = status, content_type, payload
        self.done.set()


class Broker:
    def __init__(self):
        creds = json.loads(CREDENTIAL_FILE.read_text())
        self.key_models: dict[str, set[str]] = collections.defaultdict(set)
        for model, name in MODEL_KEYS.items():
            if creds.get(name, "").strip():
                self.key_models[creds[name].strip()].add(model)
        self.cond = threading.Condition()
        self.queues: dict[str, collections.deque] = collections.defaultdict(collections.deque)
        self.inflight: dict[str, Job] = {}
        self.workers: dict[str, dict] = {}
        self.served = collections.Counter()
        self.failed = collections.Counter()
        threading.Thread(target=self.reap, daemon=True).start()

    def allowed(self, key: str, model: str) -> bool:
        return model in self.key_models.get(key, ())

    def submit(self, job: Job) -> None:
        with self.cond:
            self.queues[job.model].append(job)
            self.cond.notify_all()

    def drop_waiting(self, job: Job) -> None:
        with self.cond:
            if job in self.queues[job.model]:
                self.queues[job.model].remove(job)
            self.inflight.pop(job.id, None)

    def pull(self, worker_id: str, model: str, limit: int) -> list[Job]:
        deadline = time.time() + PULL_WAIT_SECONDS
        with self.cond:
            self.touch(worker_id, model)
            while not self.queues[model] and time.time() < deadline:
                self.cond.wait(timeout=max(0.0, deadline - time.time()))
            jobs = []
            while self.queues[model] and len(jobs) < limit:
                job = self.queues[model].popleft()
                if job.done.is_set():
                    continue
                job.worker, job.lease_until = worker_id, time.time() + LEASE_SECONDS
                job.attempts += 1
                self.inflight[job.id] = job
                jobs.append(job)
            self.touch(worker_id, model)
            return jobs

    def touch(self, worker_id: str, model: str) -> None:
        self.workers[worker_id] = {"model": model, "seen": time.time()}

    def heartbeat(self, worker_id: str, model: str, ids: list[str]) -> None:
        with self.cond:
            self.touch(worker_id, model)
            for job_id in ids:
                job = self.inflight.get(job_id)
                if job is not None and job.worker == worker_id:
                    job.lease_until = time.time() + LEASE_SECONDS

    def result(self, worker_id: str, job_id: str, status: int, content_type: str, payload: bytes) -> None:
        with self.cond:
            job = self.inflight.pop(job_id, None)
        if job is not None and job.worker == worker_id:
            (self.served if status < 500 else self.failed)[job.model] += 1
            job.finish(status, content_type, payload)

    def release(self, worker_id: str, job_id: str) -> None:
        with self.cond:
            job = self.inflight.pop(job_id, None)
            if job is not None and job.worker == worker_id and not job.done.is_set():
                job.attempts -= 1
                self.queues[job.model].appendleft(job)
                self.cond.notify_all()

    def reap(self) -> None:
        while True:
            time.sleep(5)
            now = time.time()
            with self.cond:
                for job_id, job in list(self.inflight.items()):
                    if job.lease_until >= now:
                        continue
                    del self.inflight[job_id]
                    if job.attempts >= MAX_ATTEMPTS:
                        self.failed[job.model] += 1
                        job.finish(503, "application/json", json.dumps({"error": f"worker lost the job {MAX_ATTEMPTS} times"}).encode())
                    else:
                        self.queues[job.model].appendleft(job)
                        self.cond.notify_all()
                for worker_id, info in list(self.workers.items()):
                    if now - info["seen"] > 10 * WORKER_ALIVE_SECONDS:
                        del self.workers[worker_id]

    def live_models(self) -> dict[str, int]:
        now = time.time()
        counts = collections.Counter(info["model"] for info in self.workers.values() if now - info["seen"] <= WORKER_ALIVE_SECONDS)
        return dict(counts)

    def stats(self) -> dict:
        live = self.live_models()
        with self.cond:
            models = set(MODEL_KEYS) | set(self.queues)
            return {m: {"queued": len(self.queues[m]), "in_flight": sum(j.model == m for j in self.inflight.values()),
                        "live_workers": live.get(m, 0), "served": self.served[m], "failed": self.failed[m]} for m in sorted(models)}


class Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 1024


class Handler(BaseHTTPRequestHandler):
    broker: Broker
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def reply(self, status: int, payload: bytes, content_type: str = "application/json") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def reply_json(self, status: int, obj) -> None:
        self.reply(status, json.dumps(obj).encode())

    def bearer(self) -> str:
        return self.headers.get("Authorization", "").removeprefix("Bearer ").strip()

    def body(self) -> dict:
        return json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")

    def do_GET(self):
        if self.path == "/stats":
            return self.reply_json(200, self.broker.stats())
        if self.path == "/health":
            return self.reply_json(200, {"ok": True})
        if self.path.rstrip("/") == "/v1/models":
            allowed = self.broker.key_models.get(self.bearer())
            if not allowed:
                return self.reply_json(401, {"error": "invalid api key"})
            live = self.broker.live_models()
            return self.reply_json(200, {"object": "list", "data": [{"id": m, "object": "model", "live_workers": live[m]} for m in sorted(allowed) if live.get(m)]})
        self.reply_json(404, {"error": f"unknown path {self.path}"})

    def do_POST(self):
        try:
            body = self.body()
        except (json.JSONDecodeError, ValueError):
            return self.reply_json(400, {"error": "body must be JSON"})
        if self.path.startswith("/worker/"):
            return self.worker_call(body)
        model = body.get("model", "")
        if not self.broker.allowed(self.bearer(), model):
            return self.reply_json(401, {"error": f"invalid api key for model {model!r}"})
        if body.get("stream"):
            return self.reply_json(400, {"error": "the broker does not support streaming; send stream=false"})
        job = Job(model, self.path, body)
        self.broker.submit(job)
        if not job.done.wait(timeout=JOB_TIMEOUT):
            self.broker.drop_waiting(job)
            return self.reply_json(504, {"error": f"no worker finished the request within {JOB_TIMEOUT}s"})
        self.reply(job.status, job.payload, job.content_type)

    def worker_call(self, body: dict):
        model, worker_id = body.get("model", ""), body.get("worker_id", "")
        if self.path in ("/worker/pull", "/worker/heartbeat") and not self.broker.allowed(self.bearer(), model):
            return self.reply_json(401, {"error": "invalid api key"})
        if self.path in ("/worker/result", "/worker/release") and not self.broker.key_models.get(self.bearer()):
            return self.reply_json(401, {"error": "invalid api key"})
        if self.path == "/worker/pull":
            jobs = self.broker.pull(worker_id, model, int(body.get("max", 1)))
            return self.reply_json(200, {"jobs": [{"id": j.id, "path": j.path, "body": j.body} for j in jobs]})
        if self.path == "/worker/heartbeat":
            self.broker.heartbeat(worker_id, model, body.get("ids", []))
            return self.reply_json(200, {"ok": True})
        if self.path == "/worker/result":
            self.broker.result(worker_id, body["id"], int(body["status"]), body.get("content_type", "application/json"), base64.b64decode(body.get("body_b64", "")))
            return self.reply_json(200, {"ok": True})
        if self.path == "/worker/release":
            self.broker.release(worker_id, body["id"])
            return self.reply_json(200, {"ok": True})
        self.reply_json(404, {"error": f"unknown path {self.path}"})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    Handler.broker = Broker()
    server = Server((args.host, args.port), Handler)
    print(f"broker listening on {args.host}:{args.port}, models: {sorted(MODEL_KEYS)}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
