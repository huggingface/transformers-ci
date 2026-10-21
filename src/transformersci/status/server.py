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

import ctypes
import json
import sys
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import metrics
from .store import Store
from .webhook import Filters, Ignored, Rejected, parse_delivery, verify_signature

# One /metrics body serves every scrape for this long. Rendering it is the
# costly part: 7.6 MB for 7.5k jobs, ~130 MiB peak RSS for one render, about
# twice that on a busy day. A scrape that times out keeps its thread rendering,
# so at a 5s scrape interval renders also stacked; the pod OOMKilled at 256Mi
# every ~25 min (2026-10-21). Single flight keeps it to one render at a time.
METRICS_MAX_AGE_SECONDS = 2.0

# How long a scrape waits for the render thread before serving the previous
# body: under Prometheus' 10s scrape timeout, so a slow render still answers.
METRICS_WAIT_SECONDS = 8.0


def _release_free_memory() -> None:
    """Hand free heap pages back to the OS (glibc only; a no-op elsewhere).

    A render frees ~80 MiB of objects, but glibc keeps freed pages in the
    arena that served them. With 256Mi and then 512Mi the pod still grew in
    50-60 MiB steps (one render each) until OOMKilled (2026-10-21).
    """
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


# GitHub caps payloads at 25 MB; workflow_run/workflow_job are a few KB. Anything
# near the cap is not a CI event and is refused before it is read.
MAX_BODY_BYTES = 5 * 1024 * 1024

# Label values for ci_github_status_events_total. Anything else is counted as
# "other": the header and the action come from the request, and a label value
# must never be something a caller can mint at will.
KNOWN_EVENTS = frozenset({"workflow_run", "workflow_job", "ping"})
KNOWN_ACTIONS = frozenset(
    {"requested", "queued", "waiting", "in_progress", "completed"}
)


def event_labels(event: str, payload: object) -> tuple[str, str]:
    """(event, action) for a verified delivery, clamped to known values."""
    name = event if event in KNOWN_EVENTS else "other"
    action = payload.get("action") if isinstance(payload, dict) else None
    if name == "ping":
        return name, "none"
    return name, action if action in KNOWN_ACTIONS else "other"


@dataclass
class Service:
    store: Store
    secret: bytes
    filters: Filters
    completed_window_seconds: float
    deliveries: dict[str, int] = field(default_factory=dict)
    # Signature-verified deliveries by (event, action), whatever happened next.
    events: dict[tuple[str, str], int] = field(default_factory=dict)
    processing_seconds: float = 0.0
    last_delivery_at: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)
    # Set when reconciliation is on; its health joins the payload.
    reconciler: object | None = None
    # The last /metrics body and when it was rendered (monotonic): see
    # metrics_body. The render lock makes one render at a time, ever.
    metrics_max_age: float = METRICS_MAX_AGE_SECONDS
    metrics_wait: float = METRICS_WAIT_SECONDS
    _metrics: tuple[float, bytes] = (float("-inf"), b"")
    _render_lock: threading.Lock = field(default_factory=threading.Lock)
    # Set by render_loop: scrapes then ask its thread to render instead of
    # rendering on their own short-lived request threads.
    _render_wanted: threading.Condition = field(default_factory=threading.Condition)
    _render_requested: bool = False
    _rendering: bool = False
    _render_failed_at: float = float("-inf")
    _render_worker: bool = False

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
        try:
            payload = json.loads(body)
        except ValueError:
            payload = None
        key = event_labels(event, payload)
        with self.lock:
            self.events[key] = self.events.get(key, 0) + 1
        if event == "ping":
            self._count("ignored", started)
            return 200, "pong"
        try:
            if payload is None:
                raise Rejected("body is not JSON")
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

    def metrics_body(self) -> bytes:
        """The encoded /metrics payload, rendered at most once per
        ``metrics_max_age``.

        Single flight: a scrape that arrives while another renders waits for
        that render and serves its bytes, so concurrent scrapes never hold two
        renders in memory at once. With ``render_loop`` running, the render
        happens on its thread, never on a request thread.
        """
        if not self._render_worker:
            with self._render_lock:
                rendered_at, body = self._metrics
                if time.monotonic() - rendered_at < self.metrics_max_age:
                    return body
                body = self.render_metrics().encode("utf-8")
                self._metrics = (time.monotonic(), body)
                return body
        with self._render_wanted:
            rendered_at, body = self._metrics
            if time.monotonic() - rendered_at < self.metrics_max_age:
                return body
            asked = time.monotonic()
            if not self._rendering:  # else the render in flight answers this scrape
                self._render_requested = True
                self._render_wanted.notify_all()
            self._render_wanted.wait_for(
                lambda: (
                    self._metrics[0] > rendered_at or self._render_failed_at >= asked
                ),
                timeout=self.metrics_wait,
            )
            body = self._metrics[1]
        if not body:
            raise RuntimeError("no /metrics body rendered yet")
        return body  # fresh, or the previous one if the render failed or is slow

    def render_loop(self, stop: threading.Event) -> None:
        """Render /metrics on this one long-lived thread, when a scrape asks.

        Every render then allocates from the same glibc arena. Rendered on
        each request's own thread, a render could land in a new arena that
        kept its ~55 MiB after the render ended, and the pod OOMKilled every
        ~1.5h at 512Mi (2026-10-21).
        """
        self._render_worker = True
        while not stop.is_set():
            with self._render_wanted:
                if not self._render_wanted.wait_for(
                    lambda: self._render_requested or stop.is_set(), timeout=1.0
                ):
                    continue
                if stop.is_set():
                    break
                self._render_requested = False
                self._rendering = True
            try:
                body = self.render_metrics().encode("utf-8")
            except Exception as error:  # keep serving the previous body
                print(
                    f"[ci-github-status] /metrics render failed: {error!r}",
                    file=sys.stderr,
                    flush=True,
                )
                with self._render_wanted:
                    self._rendering = False
                    self._render_failed_at = time.monotonic()
                    self._render_wanted.notify_all()
                continue
            with self._render_wanted:
                self._rendering = False
                self._metrics = (time.monotonic(), body)
                self._render_wanted.notify_all()
            del body
            _release_free_memory()

    def render_metrics(self, now: float | None = None) -> str:
        now = time.time() if now is None else now
        since = now - self.completed_window_seconds
        counts = self.store.counts()
        with self.lock:
            service = {
                "deliveries": dict(self.deliveries),
                "events": dict(self.events),
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
            self, status: int, body: str | bytes, content_type: str = "text/plain"
        ) -> None:
            data = body if isinstance(body, bytes) else body.encode("utf-8")
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
                self._reply(200, service.metrics_body(), "text/plain; version=0.0.4")
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
