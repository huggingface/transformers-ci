# Copyright 2026 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""HTTP front of the status service.

``POST /webhook`` is the only route meant to be public: it verifies the
signature over the raw body BEFORE parsing, and answers 2xx only after the
state is committed, so GitHub's delivery log is an honest record of what was
stored. ``GET /metrics`` and ``GET /healthz`` are for the cluster.

No GitHub API call happens on the request path; enrichment and repair belong to
the reconciliation worker.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import metrics
from .store import Store
from .webhook import Filters, Ignored, Rejected, parse_delivery, verify_signature

# GitHub caps payloads at 25 MB; workflow_run/workflow_job are a few KB. Anything
# near the cap is not a CI event and is refused before it is read.
MAX_BODY_BYTES = 5 * 1024 * 1024


@dataclass
class Service:
    store: Store
    secret: bytes
    filters: Filters
    completed_window_seconds: float
    deliveries: dict[str, int] = field(default_factory=dict)
    processing_seconds: float = 0.0
    last_delivery_at: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)
    # Set when reconciliation is on; its health joins the payload.
    reconciler: object | None = None

    def _count(self, outcome: str, started: float) -> None:
        with self.lock:
            self.deliveries[outcome] = self.deliveries.get(outcome, 0) + 1
            self.processing_seconds += time.monotonic() - started
            if outcome == "accepted":
                self.last_delivery_at = time.time()

    def handle_delivery(
        self, event: str, delivery_id: str, signature: str | None, body: bytes
    ) -> tuple[int, str]:
        """Status code and a short reason for one delivery."""
        started = time.monotonic()
        if not verify_signature(self.secret, body, signature):
            self._count("rejected", started)
            return 401, "invalid signature"
        if event == "ping":
            self._count("ignored", started)
            return 200, "pong"
        try:
            payload = json.loads(body)
            update = parse_delivery(event, payload, self.filters)
        except Ignored as reason:
            self._count("ignored", started)
            return 202, str(reason)
        except (Rejected, ValueError) as reason:
            self._count("rejected", started)
            return 400, str(reason)
        try:
            applied = self.store.apply([update], delivery_id=delivery_id or None)
        except Exception as error:  # the commit failed: tell GitHub, keep serving
            self._count("error", started)
            print(
                f"[ci-github-status] store error: {error!r}",
                file=sys.stderr,
                flush=True,
            )
            return 500, "store error"
        self._count("accepted" if applied else "duplicate", started)
        return 200, "stored" if applied else "duplicate delivery"

    def render_metrics(self, now: float | None = None) -> str:
        now = time.time() if now is None else now
        since = now - self.completed_window_seconds
        counts = self.store.counts()
        with self.lock:
            service = {
                "deliveries": dict(self.deliveries),
                "processing_seconds_total": self.processing_seconds,
                "last_delivery_timestamp_seconds": self.last_delivery_at,
            }
        service.update(
            runs_active=counts["runs_active"],
            jobs_active=counts["jobs_active"],
            runs_stored=counts["runs_active"] + counts["runs_completed"],
            jobs_stored=counts["jobs_active"] + counts["jobs_completed"],
            needs_lookup=counts["needs_lookup"],
            publication_timestamp_seconds=now,
        )
        body = metrics.render(
            self.store.runs(completed_since=since),
            self.store.jobs(completed_since=since),
            service=service,
        )
        snapshot = getattr(self.reconciler, "snapshot", None)
        return body + metrics.render_reconcile(snapshot(now) if snapshot else None)


def make_handler(service: Service) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "ci-github-status"

        def _reply(
            self, status: int, body: str, content_type: str = "text/plain"
        ) -> None:
            data = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", f"{content_type}; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self) -> None:  # noqa: N802 - http.server API
            if self.path != "/webhook":
                self._reply(404, "not found\n")
                return
            try:
                length = int(self.headers.get("Content-Length", ""))
            except ValueError:
                self._reply(411, "length required\n")
                return
            if length < 0 or length > MAX_BODY_BYTES:
                self._reply(413, "payload too large\n")
                return
            body = self.rfile.read(length)
            status, reason = service.handle_delivery(
                self.headers.get("X-GitHub-Event", ""),
                self.headers.get("X-GitHub-Delivery", ""),
                self.headers.get("X-Hub-Signature-256"),
                body,
            )
            self._reply(status, reason + "\n")

        def do_GET(self) -> None:  # noqa: N802 - http.server API
            if self.path == "/metrics":
                self._reply(200, service.render_metrics(), "text/plain; version=0.0.4")
            elif self.path == "/healthz":
                service.store.counts()
                self._reply(200, "ok\n")
            else:
                self._reply(404, "not found\n")

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            return  # one line per scrape and delivery is noise; outcomes are metrics

    return Handler


def serve(service: Service, host: str, port: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), make_handler(service))
    server.daemon_threads = True
    return server
