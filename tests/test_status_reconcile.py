from __future__ import annotations

import io
import json
import urllib.error
import urllib.parse

import pytest

from transformersci.status import metrics, webhook
from transformersci.status.reconcile import GitHubClient, Reconciler, Settings
from transformersci.status.reducer import JobUpdate, RunUpdate
from transformersci.status.store import Store

REPO = "huggingface/transformers"
FILTERS = webhook.Filters(
    repositories=frozenset({REPO}), workflows=frozenset({"PR CI"})
)
NOW = 1_790_236_800.0  # 2026-09-24T08:00:00Z


def iso(ts: float) -> str:
    import time

    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


def api_run(
    run_id: int = 900,
    *,
    status: str = "completed",
    conclusion: str | None = "success",
    attempt: int = 1,
    prs: list[int] | None = None,
    created: float = NOW - 600,
    private: bool = False,
    head_repository: str = REPO,
    head_branch: str = "fix-bug",
) -> dict:
    return {
        "id": run_id,
        "name": "PR CI",
        "path": ".github/workflows/pr-ci-caller.yml",
        "run_attempt": attempt,
        "event": "pull_request",
        "status": status,
        "conclusion": conclusion,
        "head_sha": "a" * 40,
        "head_branch": head_branch,
        "head_repository": {"full_name": head_repository},
        "pull_requests": [{"number": n} for n in (prs if prs is not None else [4321])],
        "created_at": iso(created),
        "run_started_at": iso(created + 5),
        "updated_at": iso(created + 300),
        "repository": {"full_name": REPO, "private": private},
    }


def api_job(
    job_id: int,
    *,
    status: str = "completed",
    conclusion: str | None = "success",
    run_id: int = 900,
) -> dict:
    return {
        "id": job_id,
        "run_id": run_id,
        "run_attempt": 1,
        "workflow_name": "PR CI",
        "name": f"pr-ci / job {job_id}",
        "status": status,
        "conclusion": conclusion if status == "completed" else None,
        "head_sha": "a" * 40,
        "head_branch": "fix-bug",
        "created_at": iso(NOW - 590),
        "started_at": iso(NOW - 580),
        "completed_at": iso(NOW - 400) if status == "completed" else None,
    }


class FakeGitHub:
    """Routes a request by path to a handler returning (status, body, headers)."""

    def __init__(self, routes: dict, *, remaining: int = 4000) -> None:
        self.routes, self.remaining = routes, remaining
        self.calls: list[tuple[str, dict, dict]] = []

    def __call__(self, request, timeout=None):
        parsed = urllib.parse.urlparse(request.full_url)
        params = dict(urllib.parse.parse_qsl(parsed.query))
        headers = dict(request.header_items())
        self.calls.append((parsed.path, params, headers))
        handler = self.routes.get(parsed.path)
        if handler is None:
            raise urllib.error.HTTPError(request.full_url, 404, "not found", {}, None)
        result = handler(params, headers) if callable(handler) else handler
        status, body, extra = result if len(result) == 3 else (*result, {})
        response_headers = {
            "X-RateLimit-Remaining": str(self.remaining),
            "X-RateLimit-Limit": "5000",
            "X-RateLimit-Reset": str(int(NOW + 3600)),
            **extra,
        }
        if status != 200:
            raise urllib.error.HTTPError(
                request.full_url, status, "error", response_headers, io.BytesIO(b"{}")
            )
        return _Response(json.dumps(body).encode(), response_headers)

    def paths(self) -> list[str]:
        return [path for path, _params, _headers in self.calls]


class _Response(io.BytesIO):
    def __init__(self, body: bytes, headers: dict) -> None:
        super().__init__(body)
        self.headers = headers

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def workflows() -> tuple:
    return (
        200,
        {
            "total_count": 2,
            "workflows": [
                {"id": 11, "name": "PR CI"},
                {"id": 12, "name": "Doc builder"},
            ],
        },
    )


def setup(tmp_path, routes, *, settings: Settings | None = None, remaining: int = 4000):
    fake = FakeGitHub(
        {f"/repos/{REPO}/actions/workflows": workflows(), **routes}, remaining=remaining
    )
    store = Store(tmp_path / "status.db")
    reconciler = Reconciler(
        store, GitHubClient("token", opener=fake), FILTERS, settings or Settings()
    )
    return store, reconciler, fake


def listing(*runs: dict) -> tuple:
    return (200, {"total_count": len(runs), "workflow_runs": list(runs)})


def jobs(*items: dict) -> tuple:
    return (200, {"total_count": len(items), "jobs": list(items)})


RUNS = f"/repos/{REPO}/actions/workflows/11/runs"
ATTEMPT = f"/repos/{REPO}/actions/runs/900/attempts/1"


def test_a_dropped_completion_is_repaired(tmp_path) -> None:
    # The webhooks said queued/in progress; the completion never arrived.
    store, reconciler, _fake = setup(
        tmp_path,
        {
            RUNS: listing(),
            ATTEMPT: (200, api_run()),
            ATTEMPT + "/jobs": jobs(api_job(1), api_job(2, conclusion="failure")),
        },
    )
    store.apply(
        [
            RunUpdate(
                REPO,
                900,
                1,
                "in_progress",
                workflow="PR CI",
                event="pull_request",
                prs=(4321,),
            )
        ]
    )
    store.apply(
        [
            JobUpdate(REPO, 900, 1, 1, "in_progress"),
            JobUpdate(REPO, 900, 1, 2, "in_progress"),
        ]
    )

    assert reconciler.run_once(now=NOW) is True
    assert {j["job_id"]: (j["status"], j["conclusion"]) for j in store.jobs()} == {
        1: ("completed", "success"),
        2: ("completed", "failure"),
    }
    (run,) = store.runs()
    assert (run["status"], run["conclusion"], run["jobs_synced"]) == (
        "completed",
        "success",
        True,
    )
    assert reconciler.stats.repairs == {"runs": 1, "jobs": 2}
    # Settled: the next cycle does not read it again.
    assert store.runs_needing_reconcile(touched_since=0) == []


def test_a_run_whose_every_event_was_lost_is_discovered(tmp_path) -> None:
    store, reconciler, fake = setup(
        tmp_path,
        {
            RUNS: listing(api_run()),
            ATTEMPT: (200, api_run()),
            ATTEMPT + "/jobs": jobs(api_job(1)),
        },
    )
    assert reconciler.run_once(now=NOW) is True
    assert [r["run_id"] for r in store.runs()] == [900]
    assert [j["job_id"] for j in store.jobs()] == [1]
    # Only the followed workflow is listed, from the first-start lookback,
    # rounded so the URL (and its ETag) is stable between cycles.
    (listing_params,) = [p for path, p, _h in fake.calls if path == RUNS]
    since = NOW - Settings().initial_lookback_seconds - Settings().overlap_seconds
    assert listing_params["created"] == ">=" + iso(since - since % 600)
    assert store.get_meta(f"discovery_watermark:{REPO}") == repr(NOW - 600)


def test_private_and_other_workflow_runs_are_never_stored(tmp_path) -> None:
    other = {**api_run(901), "name": "Doc builder"}
    store, reconciler, _fake = setup(
        tmp_path, {RUNS: listing(api_run(900, private=True), other)}
    )
    assert reconciler.run_once(now=NOW) is True
    assert store.runs() == []


def test_an_api_outage_raises_the_stale_flag_and_completes_nothing(tmp_path) -> None:
    def down(_params, _headers):
        raise urllib.error.URLError("connection refused")

    store, reconciler, _fake = setup(tmp_path, {RUNS: down})
    store.apply([JobUpdate(REPO, 900, 1, 1, "in_progress")])
    reconciler.stats.started = NOW - 600
    assert reconciler.run_once(now=NOW) is False
    snapshot = reconciler.snapshot(now=NOW)
    assert snapshot["stale"] == 1
    assert snapshot["cycles"] == {"error": 1}
    assert snapshot["last_success"] == 0.0
    assert store.jobs()[0]["status"] == "in_progress"  # no false completion
    text = metrics.render_reconcile(snapshot)
    assert "ci_github_status_stale 1" in text
    assert 'ci_github_status_reconcile_cycles_total{outcome="error"} 1' in text


def test_fresh_after_a_successful_cycle(tmp_path) -> None:
    _store, reconciler, _fake = setup(tmp_path, {RUNS: listing()})
    assert reconciler.run_once(now=NOW) is True
    assert reconciler.snapshot(now=NOW + 60)["stale"] == 0
    assert reconciler.snapshot(now=NOW + 600)["stale"] == 1


def test_a_throttle_pauses_until_githubs_reset(tmp_path) -> None:
    throttled = (
        403,
        {},
        {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": str(int(NOW + 900))},
    )
    _store, reconciler, fake = setup(tmp_path, {RUNS: throttled})
    assert reconciler.run_once(now=NOW) is False
    assert reconciler.stats.paused_until == NOW + 900
    calls = len(fake.calls)
    assert reconciler.run_once(now=NOW + 60) is False
    assert len(fake.calls) == calls  # paused: not a single request
    assert reconciler.stats.cycles == {"throttled": 1, "paused": 1}


def test_below_the_rate_floor_nothing_is_spent(tmp_path) -> None:
    _store, reconciler, fake = setup(tmp_path, {RUNS: listing()}, remaining=100)
    assert reconciler.run_once(now=NOW) is False  # the workflows call reveals the floor
    assert len(fake.calls) == 1
    assert reconciler.stats.cycles == {"throttled": 1}


def test_unchanged_listings_are_served_by_etag(tmp_path) -> None:
    def etagged(_params, headers):
        if headers.get("If-none-match") == '"v1"':
            return (304, {}, {"ETag": '"v1"'})
        return (200, {"total_count": 0, "workflow_runs": []}, {"ETag": '"v1"'})

    _store, reconciler, _fake = setup(tmp_path, {RUNS: etagged})
    assert reconciler.run_once(now=NOW) is True
    assert reconciler.run_once(now=NOW + 60) is True
    assert reconciler.client.requests["not_modified"] == 1


def test_the_api_settles_a_disputed_conclusion(tmp_path) -> None:
    store, reconciler, _fake = setup(
        tmp_path,
        {
            RUNS: listing(),
            ATTEMPT: (200, api_run()),
            ATTEMPT + "/jobs": jobs(api_job(1, conclusion="failure")),
        },
    )
    store.apply([JobUpdate(REPO, 900, 1, 1, "completed", conclusion="success")])
    store.apply([JobUpdate(REPO, 900, 1, 1, "completed", conclusion="failure")])
    assert store.jobs()[0]["needs_lookup"] is True
    assert reconciler.run_once(now=NOW) is True
    (job,) = store.jobs()
    assert (job["conclusion"], job["needs_lookup"]) == ("failure", False)


@pytest.mark.parametrize(
    ("head_sha", "expected"), [("a" * 40, (48976,)), ("b" * 40, ())]
)
def test_a_fork_pr_is_found_by_its_head_branch(tmp_path, head_sha, expected) -> None:
    fork_run = api_run(
        prs=[], head_repository="contributor/transformers", head_branch="patch-1"
    )
    pulls = f"/repos/{REPO}/pulls"
    store, reconciler, fake = setup(
        tmp_path,
        {
            RUNS: listing(),
            ATTEMPT: (200, fork_run),
            ATTEMPT + "/jobs": jobs(),
            pulls: (200, [{"number": 48976, "head": {"sha": head_sha}}]),
        },
    )
    store.apply(
        [
            webhook.parse_workflow_run(
                {"workflow_run": fork_run, "repository": fork_run["repository"]},
                FILTERS,
            )
        ]
    )
    assert reconciler.run_once(now=NOW) is True
    (run,) = store.runs()
    assert run["prs"] == expected  # a PR whose head is another commit is not this run's
    assert run["pr_lookup_done"] is True
    (params,) = [p for path, p, _h in fake.calls if path == pulls]
    assert params["head"] == "contributor:patch-1"
    # Asked once, whatever the answer.
    reconciler.run_once(now=NOW + 60)
    assert fake.paths().count(pulls) == 1


def test_the_request_budget_rotates_through_runs(tmp_path) -> None:
    routes = {RUNS: listing()}
    for run_id in (900, 901):
        path = f"/repos/{REPO}/actions/runs/{run_id}/attempts/1"
        routes[path] = (200, api_run(run_id, status="in_progress", conclusion=None))
        routes[path + "/jobs"] = jobs(
            api_job(run_id * 10, status="in_progress", run_id=run_id)
        )
    store, reconciler, fake = setup(
        tmp_path, routes, settings=Settings(requests_per_cycle=4)
    )
    for run_id in (900, 901):
        store.apply(
            [
                RunUpdate(
                    REPO,
                    run_id,
                    1,
                    "in_progress",
                    workflow="PR CI",
                    event="pull_request",
                    prs=(1,),
                )
            ]
        )
    # workflows + listing + one run's attempt + its jobs = 4: the budget.
    assert reconciler.run_once(now=NOW) is False
    assert reconciler.stats.cycles == {"budget": 1}
    first = [p for p in fake.paths() if p.endswith("/attempts/1")]
    reconciler.run_once(now=NOW + 60)
    second = [p for p in fake.paths() if p.endswith("/attempts/1")][len(first) :]
    assert first and second and first != second  # the other run goes first next time
