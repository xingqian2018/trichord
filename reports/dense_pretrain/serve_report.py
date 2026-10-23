"""Static file server for the report: gzip when the client accepts it, conditional GET (304), and one log line per
request with the bytes sent and the seconds the send took (to tell a slow network from a slow client).

    python3 reports_xx/serve_report.py PORT reports_xx
    -> http://<host>:PORT/qwen3_scaling_ladder_progress_en.html   (no index.html here, so "/" is a 404)

Copied verbatim from runlongl's Dense-Pretraining-LMs tools/analysis/serve_report.py (83518b2, 2026-10-16).
"""
import gzip
import http.server
import os
import socket
import sys
import time
from email.utils import formatdate, parsedate_to_datetime

PORT, ROOT = int(sys.argv[1]), os.path.abspath(sys.argv[2])
_cache = {}


def payload(path):
    st = os.stat(path)
    key = (path, st.st_mtime_ns, st.st_size)
    if key not in _cache:
        raw = open(path, "rb").read()
        _cache.clear()
        _cache[key] = (raw, gzip.compress(raw, 9), st.st_mtime)
    return _cache[key]


class H(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_HEAD(self):
        self.do_GET(head=True)

    def do_GET(self, head=False):
        rel = self.path.split("?", 1)[0].lstrip("/") or "index.html"
        path = os.path.realpath(os.path.join(ROOT, rel))
        if not path.startswith(ROOT + os.sep) or not os.path.isfile(path):
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        raw, gz, mtime = payload(path)
        ims = self.headers.get("If-Modified-Since")
        if ims:
            try:
                if int(mtime) <= parsedate_to_datetime(ims).timestamp():
                    self.send_response(304)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
            except (TypeError, ValueError):
                pass
        use_gz = "gzip" in self.headers.get("Accept-Encoding", "")
        body = gz if use_gz else raw
        t0 = time.time()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8" if path.endswith(".html") else "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Last-Modified", formatdate(mtime, usegmt=True))
        self.send_header("Cache-Control", "no-cache")
        if use_gz:
            self.send_header("Content-Encoding", "gzip")
        self.send_header("Vary", "Accept-Encoding")
        self.end_headers()
        if head:
            return
        self.wfile.write(body)
        self.wfile.flush()
        self.log_message('sent %s %d bytes%s in %.2f s', rel, len(body), " (gzip)" if use_gz else "", time.time() - t0)


class S(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def server_bind(self):
        # smaller segments than the 9000-byte node MTU; the path to remote clients runs through tunnels with ~1300 MTU
        self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_MAXSEG, 1200)
        super().server_bind()


S(("0.0.0.0", PORT), H).serve_forever()
