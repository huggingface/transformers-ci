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
"""Repair what the webhooks missed, from the GitHub Actions API.

GitHub does not redeliver a failed webhook by itself, and the receiver can be
down, so every cycle (default 60s, and once at start) the reconciler:

1. **discovers** runs of the followed workflows created since a persisted
   watermark (minus an overlap), so a run whose every event was lost — even a
   short one that already finished — still appears;
2. **refreshes** each stored run that is not settled (see
   :meth:`Store.runs_needing_reconcile`): the attempt-specific run, then its
   jobs, applied as *authoritative* so a disputed conclusion is settled;
3. **enriches** a PR run GitHub sent without its PR (a PR from a fork) by
   asking which PR has that fork branch as its head.

It is bounded: a request budget per cycle (least recently refreshed first),
one request at a time, conditional requests (a 304 costs no rate limit), and a
pause until the reset when the remaining budget falls below a floor or GitHub
answers 403/429. A failed or skipped cycle does not advance
``last_success``, which is what the stale flag reads; nothing here ever marks a
run or job completed without GitHub saying so.
"""

from __future__ import annotations

import http.client
import json
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field

from .reducer import COMPLETED, JobUpdate, RunUpdate
from .store import Store, _pr_unknown
from .webhook import Filters, Ignored, Rejected, parse_workflow_job, parse_workflow_run

API = "https://api.github.com"
WATERMARK_KEY = "discovery_watermark"


# Failures of a single GET worth retrying: the request is idempotent, and on
# 2026-09-24 the first production cycle died on one ~480 KB jobs page whose body
# was cut off (IncompleteRead) — the whole cycle, for one dropped connection.
TRANSIENT = (
    http.client.IncompleteRead,
    http.client.RemoteDisconnected,
    ConnectionError,
    TimeoutError,
    json.JSONDecodeError,
)


def _transient(error: BaseException) -> bool:
    if isinstance(error, urllib.error.HTTPError):
        return error.code >= 500
    if isinstance(error, urllib.error.URLError):
        return True
    return isinstance(error, TRANSIENT)


class Throttled(Exception):
    """GitHub asked us to stop until ``resume_at``."""

    def __init__(self, resume_at: float) -> None:
        super().__init__(f"throttled until {resume_at:.0f}")
        self.resume_at = resume_at


@dataclass
class Response:
    status: int
    body: object
    headers: dict[str, str]


# Bound on the raw JSON bytes the ETag cache keeps (parsed, about 2.3-3.3x that).
# The working set is one discovery listing per workflow plus the unsettled runs'
# attempt and jobs pages, a few MB at peak. Unbounded, it kept every page ever
# fetched: discovery's 10-minute watermark bucket mints a new listing URL 144 times
# a day, and every run adds its jobs pages, so the pod hit its 256Mi limit in ~24h.
ETAG_CACHE_BYTES = 16 * 1024 * 1024


class GitHubClient:
    """Minimal authenticated GET with ETag reuse and rate-limit bookkeeping."""

    def __init__(
        self,
        token: str,
        *,
        api: str = API,
        timeout: float = 20.0,
        opener: Callable[..., object] = urllib.request.urlopen,
        retries: int = 2,
        sleep: Callable[[float], None] = time.sleep,
        etag_cache_bytes: int = ETAG_CACHE_BYTES,
    ) -> None:
        self._token, self._api, self._timeout, self._open = token, api, timeout, opener
        self._retries, self._sleep = retries, sleep
        # url -> (etag, parsed body, raw size), least recently used first
        self._etags: OrderedDict[str, tuple[str, object, int]] = OrderedDict()
        self._etag_bytes, self._etag_cache_bytes = 0, etag_cache_bytes
        self.rate: dict[str, float] = {}
        self.requests: dict[str, int] = {}

    def _count(self, outcome: str) -> None:
        self.requests[outcome] = self.requests.get(outcome, 0) + 1

    def _record_rate(self, headers: dict[str, str]) -> None:
        for key, header in (
            ("limit", "x-ratelimit-limit"),
            ("remaining", "x-ratelimit-remaining"),
            ("reset", "x-ratelimit-reset"),
        ):
            try:
                self.rate[key] = float(headers[header])
            except (KeyError, ValueError):
                continue

    def get(self, path: str, params: dict[str, str | int] | None = None) -> object:
        """GET ``path``; transient failures are retried with backoff before the
        last one is raised. A throttle is never retried here: it pauses the loop."""
        for attempt in range(self._retries + 1):
            try:
                return self._get_once(path, params)
            except Throttled:
                raise
            except Exception as error:
                if attempt == self._retries or not _transient(error):
                    raise
                self._count("retried")
                self._sleep(2.0**attempt)
        raise AssertionError("unreachable")

    def _get_once(self, path: str, params: dict[str, str | int] | None) -> object:
        url = f"{self._api}{path}"
        if params:
            url += "?" + urllib.parse.urlencode(params)
        request = urllib.request.Request(url)
        request.add_header("Accept", "application/vnd.github+json")
        request.add_header("X-GitHub-Api-Version", "2022-11-28")
        request.add_header("User-Agent", "transformersci-ci-github-status")
        if self._token:
            request.add_header("Authorization", f"Bearer {self._token}")
        cached = self._etags.get(url)
        if cached:
            self._etags.move_to_end(url)
            request.add_header("If-None-Match", cached[0])
        try:
            with self._open(request, timeout=self._timeout) as response:
                headers = {k.lower(): v for k, v in response.headers.items()}
                raw = response.read()
                body = json.loads(raw)
        except urllib.error.HTTPError as error:
            headers = {k.lower(): v for k, v in (error.headers or {}).items()}
            self._record_rate(headers)
            if error.code == 304 and cached:
                self._count("not_modified")
                return cached[1]
            if error.code in (403, 429) and (
                "retry-after" in headers or headers.get("x-ratelimit-remaining") == "0"
            ):
                self._count("throttled")
                raise Throttled(_resume_at(headers)) from error
            self._count("error")
            raise
        except Exception:
            self._count("error")
            raise
        self._record_rate(headers)
        self._count("ok")
        if headers.get("etag"):
            self._remember(url, headers["etag"], body, len(raw))
        return body

    def _remember(self, url: str, etag: str, body: object, size: int) -> None:
        old = self._etags.pop(url, None)
        if old:
            self._etag_bytes -= old[2]
        if size > self._etag_cache_bytes:
            return
        self._etags[url] = (etag, body, size)
        self._etag_bytes += size
        while self._etag_bytes > self._etag_cache_bytes:
            _url, (_etag, _body, evicted) = self._etags.popitem(last=False)
            self._etag_bytes -= evicted

    @property
    def etag_cache(self) -> tuple[int, int]:
        """(entries, raw bytes) held by the ETag cache."""
        return len(self._etags), self._etag_bytes


def _resume_at(headers: dict[str, str]) -> float:
    now = time.time()
    try:
        return now + float(headers["retry-after"])
    except (KeyError, ValueError):
        pass
    try:
        return float(headers["x-ratelimit-reset"])
    except (KeyError, ValueError):
        return now + 60.0


def _iso(ts: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


@dataclass
class Settings:
    interval_seconds: float = 60.0
    # First start: how far back discovery looks. Later: the persisted watermark.
    initial_lookback_seconds: float = 6 * 3600.0
    # Re-list this much before the newest run already seen, so a run created
    # just before it but indexed late is not skipped. Rounded to 10 minutes so
    # the listing URL is stable across cycles and its ETag can answer 304.
    overlap_seconds: float = 900.0
    max_list_pages: int = 10
    max_job_pages: int = 5
    requests_per_cycle: int = 200
    # Stop spending when fewer requests than this remain in the window.
    rate_floor: int = 300
    # Runs untouched for this long are no longer polled (GitHub lost them).
    active_horizon_seconds: float = 24 * 3600.0
    # The stale flag rises when no cycle succeeded for this long.
    stale_after_seconds: float = 180.0


@dataclass
class Stats:
    cycles: dict[str, int] = field(default_factory=dict)
    repairs: dict[str, int] = field(default_factory=dict)
    last_attempt: float = 0.0
    last_success: float = 0.0
    last_duration: float = 0.0
    last_requests: int = 0
    last_error: str = ""
    discovery_truncated: int = 0
    run_errors: int = 0
    paused_until: float = 0.0
    started: float = field(default_factory=time.time)


class _Budget(Exception):
    """The per-cycle request budget is spent; the rest waits for next cycle."""


class _Partial(Exception):
    """Some runs could not be refreshed; the others were. Not a success."""


class Reconciler:
    def __init__(
        self,
        store: Store,
        client: GitHubClient,
        filters: Filters,
        settings: Settings | None = None,
    ) -> None:
        self.store, self.client, self.filters = store, client, filters
        self.settings = settings or Settings()
        self.stats = Stats()
        self._lock = threading.Lock()
        self._workflow_ids: dict[str, list[int]] = {}
        self._last_refreshed: dict[tuple[str, int, int], float] = {}
        self._spent = 0

    # -- plumbing -------------------------------------------------------------

    def _get(self, path: str, params: dict[str, str | int] | None = None) -> object:
        if self._spent >= self.settings.requests_per_cycle:
            raise _Budget
        remaining = self.client.rate.get("remaining")
        if remaining is not None and remaining < self.settings.rate_floor:
            raise Throttled(self.client.rate.get("reset", time.time() + 60.0))
        self._spent += 1
        return self.client.get(path, params)

    def _repaired(self, kind: str, count: int) -> None:
        if count:
            self.stats.repairs[kind] = self.stats.repairs.get(kind, 0) + count

    def _paged(
        self, path: str, key: str, max_pages: int, params: dict
    ) -> tuple[list[dict], bool]:
        """Items of a paginated list and whether the listing is complete."""
        items: list[dict] = []
        for page in range(1, max_pages + 1):
            body = self._get(path, {**params, "per_page": 100, "page": page})
            batch = body.get(key) if isinstance(body, dict) else None
            if not isinstance(batch, list):
                return items, False
            items.extend(item for item in batch if isinstance(item, dict))
            total = body.get("total_count")
            if len(batch) < 100 or (isinstance(total, int) and len(items) >= total):
                return items, True
        return items, False

    # -- one cycle ------------------------------------------------------------

    def run_once(self, now: float | None = None) -> bool:
        """One reconciliation pass. Returns whether it completed; a partial
        pass (budget, throttle, error) keeps what it applied."""
        now = time.time() if now is None else now
        with self._lock:
            self.stats.last_attempt = now
            self._spent = 0
            started = time.monotonic()
            outcome = "ok"
            try:
                if now < self.stats.paused_until:
                    outcome = "paused"
                    return False
                for repository in sorted(self.filters.repositories):
                    self._discover(repository, now)
                self._refresh(now)
                self.stats.last_success = now
                self.stats.last_error = ""
                return True
            except _Budget:
                outcome = "budget"
                return False
            except _Partial as partial:
                outcome = "partial"
                self.stats.last_error = str(partial)
                return False
            except Throttled as throttle:
                outcome = "throttled"
                self.stats.paused_until = throttle.resume_at
                return False
            except Exception as error:
                outcome = "error"
                self.stats.last_error = type(error).__name__
                print(
                    f"[ci-github-status] reconcile failed: {error!r}",
                    file=sys.stderr,
                    flush=True,
                )
                return False
            finally:
                self.stats.cycles[outcome] = self.stats.cycles.get(outcome, 0) + 1
                self.stats.last_duration = time.monotonic() - started
                self.stats.last_requests = self._spent

    def _workflows(self, repository: str) -> list[int]:
        if repository not in self._workflow_ids:
            items, _complete = self._paged(
                f"/repos/{repository}/actions/workflows", "workflows", 5, {}
            )
            self._workflow_ids[repository] = sorted(
                int(w["id"]) for w in items if w.get("name") in self.filters.workflows
            )
        return self._workflow_ids[repository]

    def _discover(self, repository: str, now: float) -> None:
        key = f"{WATERMARK_KEY}:{repository}"
        stored = self.store.get_meta(key)
        watermark = (
            float(stored) if stored else now - self.settings.initial_lookback_seconds
        )
        since = watermark - self.settings.overlap_seconds
        since -= since % 600
        newest = watermark
        for workflow_id in self._workflows(repository):
            runs, complete = self._paged(
                f"/repos/{repository}/actions/workflows/{workflow_id}/runs",
                "workflow_runs",
                self.settings.max_list_pages,
                {"created": f">={_iso(since)}"},
            )
            if not complete:
                self.stats.discovery_truncated += 1
            updates = []
            for run in runs:
                try:
                    update = parse_workflow_run(
                        {"workflow_run": run, "repository": _public(repository, run)},
                        self.filters,
                    )
                except (Ignored, Rejected):
                    continue
                updates.append(update)
                if update.created_at:
                    newest = max(newest, update.created_at)
            runs_changed, _ = self.store.reconcile(updates, now=now)
            self._repaired("runs", runs_changed)
        self.store.set_meta(key, repr(newest))

    def _refresh(self, now: float) -> None:
        due = self.store.runs_needing_reconcile(
            touched_since=now - self.settings.active_horizon_seconds
        )
        due.sort(key=lambda run: self._last_refreshed.get(_key(run), 0.0))
        failed: list[str] = []
        for run in due:
            try:
                self._refresh_run(run, now)
            except (_Budget, Throttled):
                raise
            except urllib.error.HTTPError as error:
                if error.code not in (404, 410):
                    failed.append(f"{_key(run)[1]}: HTTP {error.code}")
                else:
                    # GitHub no longer has it (deleted run): stop asking. It is
                    # not marked completed, because GitHub never said so.
                    self.store.mark_run(_key(run), gone=True)
            except Exception as error:
                failed.append(f"{_key(run)[1]}: {type(error).__name__}")
            # Tried either way, so a run that keeps failing goes to the back.
            self._last_refreshed[_key(run)] = now
        if failed:
            self.stats.run_errors += len(failed)
            print(
                f"[ci-github-status] {len(failed)} run(s) not refreshed: {failed[:5]}",
                file=sys.stderr,
                flush=True,
            )
            raise _Partial(f"{len(failed)} run(s) not refreshed")

    def _refresh_run(self, stored: dict, now: float) -> None:
        repository, run_id, attempt = _key(stored)
        body = self._get(
            f"/repos/{repository}/actions/runs/{run_id}/attempts/{attempt}"
        )
        if not isinstance(body, dict):
            return
        try:
            update = parse_workflow_run(
                {"workflow_run": body, "repository": _public(repository, body)},
                self.filters,
            )
        except (Ignored, Rejected):
            return
        changed, _ = self.store.reconcile([update], now=now)
        self._repaired("runs", changed)

        jobs, complete = self._paged(
            f"/repos/{repository}/actions/runs/{run_id}/attempts/{attempt}/jobs",
            "jobs",
            self.settings.max_job_pages,
            {},
        )
        job_updates: list[JobUpdate] = []
        for job in jobs:
            job.setdefault("workflow_name", update.workflow)
            try:
                job_updates.append(
                    parse_workflow_job(
                        {"workflow_job": job, "repository": _public(repository, job)},
                        self.filters,
                    )
                )
            except (Ignored, Rejected):
                continue
        _, jobs_changed = self.store.reconcile(job_updates, now=now)
        self._repaired("jobs", jobs_changed)
        if complete and update.status == COMPLETED:
            self.store.mark_run(update.key, jobs_synced=True)

        merged = {
            **stored,
            "prs": update.prs,
            "event": update.event or stored.get("event"),
        }
        if _pr_unknown(merged) and not stored.get("pr_lookup_done"):
            self._find_pr(update, now)

    def _find_pr(self, run: RunUpdate, now: float) -> None:
        """The PR whose head is the fork branch this run built. Only a PR whose
        head commit is the run's own counts; otherwise the PR stays unknown."""
        owner = run.head_repository.split("/", 1)[0]
        if not owner or not run.head_branch:
            self.store.mark_run(run.key, pr_lookup_done=True)
            return
        body = self._get(
            f"/repos/{run.repository}/pulls",
            {"head": f"{owner}:{run.head_branch}", "state": "all", "per_page": 10},
        )
        numbers = tuple(
            sorted(
                int(pr["number"])
                for pr in (body if isinstance(body, list) else [])
                if isinstance(pr, dict)
                and isinstance(pr.get("number"), int)
                and (pr.get("head") or {}).get("sha") == run.head_sha
            )
        )
        if numbers:
            changed, _ = self.store.reconcile(
                [
                    RunUpdate(
                        run.repository, run.run_id, run.attempt, run.status, prs=numbers
                    )
                ],
                now=now,
            )
            self._repaired("prs", changed)
        self.store.mark_run(run.key, pr_lookup_done=True)

    # -- loop and health ------------------------------------------------------

    def loop(self, stop: threading.Event) -> None:
        while True:
            self.run_once()
            if stop.wait(self.settings.interval_seconds):
                return

    def snapshot(self, now: float | None = None) -> dict:
        now = time.time() if now is None else now
        stats = self.stats
        reference = stats.last_success or stats.started
        return {
            "cycles": dict(stats.cycles),
            "repairs": dict(stats.repairs),
            "requests": dict(self.client.requests),
            "rate": dict(self.client.rate),
            "last_attempt": stats.last_attempt,
            "last_success": stats.last_success,
            "last_duration": stats.last_duration,
            "last_requests": stats.last_requests,
            "discovery_truncated": stats.discovery_truncated,
            "run_errors": stats.run_errors,
            "paused_until": stats.paused_until,
            "etag_cache": self.client.etag_cache,
            "stale": 1 if now - reference > self.settings.stale_after_seconds else 0,
        }


def _key(run: dict) -> tuple[str, int, int]:
    return (run["repository"], int(run["run_id"]), int(run["attempt"]))


def _public(repository: str, item: dict) -> dict:
    """The ``repository`` block the webhook parser checks. API run objects carry
    the repository (with its visibility); job objects do not, and inherit the
    run's, which was checked when the run was."""
    block = item.get("repository")
    if isinstance(block, dict) and "private" in block:
        return block
    return {"full_name": repository, "private": False}
