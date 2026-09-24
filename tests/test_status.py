from __future__ import annotations

import hashlib
import hmac
import json
import threading
import urllib.error
import urllib.request

import pytest

from transformersci.status import cli, metrics, webhook
from transformersci.status.reducer import JobUpdate, RunUpdate, merge_job, merge_run
from transformersci.status.server import Service, serve
from transformersci.status.store import Store

SECRET = b"test-secret"
REPO = "huggingface/transformers"
FILTERS = webhook.Filters(
    repositories=frozenset({REPO}), workflows=frozenset({"PR CI"})
)


def _repository(private: bool = False, name: str = REPO) -> dict:
    return {"full_name": name, "private": private}


def run_payload(
    *,
    action: str = "requested",
    status: str = "queued",
    conclusion: str | None = None,
    run_id: int = 900,
    attempt: int = 1,
    prs: list[int] | None = None,
    event: str = "pull_request",
    head_branch: str = "fix-bug",
    workflow: str = "PR CI",
    updated_at: str = "2026-09-24T07:30:00Z",
    private: bool = False,
    repository: str = REPO,
) -> dict:
    return {
        "action": action,
        "workflow_run": {
            "id": run_id,
            "name": workflow,
            "path": ".github/workflows/pr-ci-caller.yml",
            "run_attempt": attempt,
            "event": event,
            "status": status,
            "conclusion": conclusion,
            "head_sha": "a" * 40,
            "head_branch": head_branch,
            "pull_requests": [
                {"number": n} for n in (prs if prs is not None else [4321])
            ],
            "created_at": "2026-09-24T07:29:00Z",
            "run_started_at": "2026-09-24T07:29:05Z",
            "updated_at": updated_at,
            "html_url": f"https://github.com/{repository}/actions/runs/{run_id}",
        },
        "repository": _repository(private, repository),
    }


def job_payload(
    *,
    action: str = "queued",
    status: str = "queued",
    conclusion: str | None = None,
    job_id: int = 7001,
    run_id: int = 900,
    attempt: int = 1,
    name: str = "pr-ci / Check repository consistency",
    workflow: str = "PR CI",
    started_at: str | None = None,
    completed_at: str | None = None,
) -> dict:
    return {
        "action": action,
        "workflow_job": {
            "id": job_id,
            "run_id": run_id,
            "run_attempt": attempt,
            "workflow_name": workflow,
            "name": name,
            "status": status,
            "conclusion": conclusion,
            "head_sha": "a" * 40,
            "head_branch": "fix-bug",
            "created_at": "2026-09-24T07:29:10Z",
            "started_at": started_at,
            "completed_at": completed_at,
            "html_url": f"https://github.com/{REPO}/actions/runs/{run_id}/job/{job_id}",
        },
        "repository": _repository(),
    }


def sign(body: bytes, secret: bytes = SECRET) -> str:
    return "sha256=" + hmac.new(secret, body, hashlib.sha256).hexdigest()


def job(status: str, conclusion: str = "", **kwargs) -> JobUpdate:
    return JobUpdate(REPO, 900, 1, 7001, status, conclusion=conclusion, **kwargs)


def run(status: str, conclusion: str = "", **kwargs) -> RunUpdate:
    return RunUpdate(REPO, 900, 1, status, conclusion=conclusion, **kwargs)


# -- reducer ------------------------------------------------------------------


def test_status_only_moves_forward() -> None:
    state = merge_job(None, job("completed", "failure", completed_at=30.0))
    late = merge_job(state, job("in_progress", started_at=10.0))
    assert late["status"] == "completed"
    assert late["conclusion"] == "failure"
    # ...but a late event still fills a field the state lacked.
    assert late["started_at"] == 10.0
    assert merge_job(late, job("queued"))["status"] == "completed"


@pytest.mark.parametrize(
    "conclusion",
    [
        "success",
        "failure",
        "cancelled",
        "skipped",
        "timed_out",
        "neutral",
        "action_required",
        "stale",
    ],
)
def test_every_conclusion_is_kept_verbatim(conclusion: str) -> None:
    state = merge_job(merge_job(None, job("in_progress")), job("completed", conclusion))
    assert state["conclusion"] == conclusion


def test_an_undocumented_status_never_overrides_a_known_one() -> None:
    state = merge_job(None, job("in_progress"))
    assert merge_job(state, job("unheard_of"))["status"] == "in_progress"


def test_disagreeing_conclusions_keep_the_first_and_ask_for_a_lookup() -> None:
    state = merge_job(None, job("completed", "success"))
    conflicted = merge_job(state, job("completed", "failure"))
    assert conflicted["conclusion"] == "success"
    assert conflicted["needs_lookup"] is True
    # A repeat of the first answer does not clear it; only the API does.
    assert merge_job(conflicted, job("completed", "success"))["needs_lookup"] is True
    resolved = merge_job(conflicted, job("completed", "failure"), authoritative=True)
    assert (resolved["conclusion"], resolved["needs_lookup"]) == ("failure", False)


def test_run_prs_accumulate_and_timestamps_only_advance() -> None:
    state = merge_run(None, run("in_progress", prs=(), updated_at=20.0))
    state = merge_run(state, run("queued", prs=(4321,), updated_at=10.0))
    assert state["status"] == "in_progress"
    assert state["prs"] == (4321,)
    assert state["updated_at"] == 20.0


def test_a_finished_job_does_not_finish_its_run() -> None:
    assert job("completed", "success").run_stub().status == "in_progress"
    assert job("queued").run_stub().status == "queued"


# -- webhook ------------------------------------------------------------------


def test_signature_is_checked_on_the_raw_body() -> None:
    body = json.dumps(job_payload()).encode()
    assert webhook.verify_signature(SECRET, body, sign(body))
    assert not webhook.verify_signature(SECRET, body + b" ", sign(body))
    assert not webhook.verify_signature(SECRET, body, sign(body, b"other"))
    assert not webhook.verify_signature(SECRET, body, None)
    assert not webhook.verify_signature(SECRET, body, "sha1=" + "0" * 40)
    assert not webhook.verify_signature(b"", body, sign(body, b""))


def test_only_allow_listed_public_repositories_are_stored() -> None:
    with pytest.raises(webhook.Ignored, match="allow-list"):
        webhook.parse_delivery(
            "workflow_run", run_payload(repository="huggingface/other"), FILTERS
        )
    # On the list but private: still refused. The dashboard is anonymous.
    with pytest.raises(webhook.Ignored, match="not public"):
        webhook.parse_delivery("workflow_run", run_payload(private=True), FILTERS)


def test_other_workflows_and_events_are_ignored() -> None:
    with pytest.raises(webhook.Ignored, match="workflow"):
        webhook.parse_delivery(
            "workflow_run", run_payload(workflow="Doc builder"), FILTERS
        )
    with pytest.raises(webhook.Ignored, match="workflow"):
        webhook.parse_delivery(
            "workflow_job", job_payload(workflow="Doc builder"), FILTERS
        )
    with pytest.raises(webhook.Ignored, match="event"):
        webhook.parse_delivery("push", {}, FILTERS)


def test_malformed_payloads_are_rejected() -> None:
    payload = job_payload()
    payload["workflow_job"]["id"] = "7001"
    with pytest.raises(webhook.Rejected):
        webhook.parse_delivery("workflow_job", payload, FILTERS)
    with pytest.raises(webhook.Rejected):
        webhook.parse_delivery("workflow_job", [], FILTERS)


def test_a_fork_pr_without_pull_requests_is_kept_as_unknown() -> None:
    update = webhook.parse_delivery("workflow_run", run_payload(prs=[]), FILTERS)
    assert update.prs == ()
    assert metrics.pr_label({"prs": update.prs, "event": update.event}) == ""


def test_merge_group_and_branch_runs_get_the_exporters_pr_label() -> None:
    queued = webhook.parse_delivery(
        "workflow_run",
        run_payload(
            prs=[],
            event="merge_group",
            head_branch="gh-readonly-queue/main/pr-48976-0123abcd",
        ),
        FILTERS,
    )
    assert queued.prs == (48976,)
    push = webhook.parse_delivery(
        "workflow_run", run_payload(prs=[], event="push", head_branch="main"), FILTERS
    )
    assert (
        metrics.pr_label({"prs": push.prs, "event": push.event, "head_branch": "main"})
        == "main"
    )


def test_rerun_attempts_are_separate_runs() -> None:
    first = webhook.parse_delivery("workflow_run", run_payload(attempt=1), FILTERS)
    second = webhook.parse_delivery("workflow_run", run_payload(attempt=2), FILTERS)
    assert first.key != second.key


# -- store --------------------------------------------------------------------


def test_duplicate_delivery_is_a_no_op(tmp_path) -> None:
    store = Store(tmp_path / "status.db")
    update = webhook.parse_delivery("workflow_job", job_payload(), FILTERS)
    assert store.apply([update], delivery_id="d-1") is True
    assert store.apply([update], delivery_id="d-1") is False
    assert len(store.jobs()) == 1


def test_a_job_before_its_run_still_publishes_both(tmp_path) -> None:
    store = Store(tmp_path / "status.db")
    store.apply([webhook.parse_delivery("workflow_job", job_payload(), FILTERS)])
    (only_run,) = store.runs()
    assert (only_run["run_id"], only_run["status"], only_run["prs"]) == (
        900,
        "queued",
        (),
    )
    # The run event then fills in what the job could not say.
    store.apply([webhook.parse_delivery("workflow_run", run_payload(), FILTERS)])
    (only_run,) = store.runs()
    assert only_run["prs"] == (4321,)
    assert only_run["event"] == "pull_request"


def test_state_survives_a_restart(tmp_path) -> None:
    path = tmp_path / "status.db"
    store = Store(path)
    store.apply(
        [
            webhook.parse_delivery(
                "workflow_job", job_payload(status="in_progress"), FILTERS
            )
        ],
        delivery_id="d-1",
    )
    store.close()
    reopened = Store(path)
    assert reopened.jobs()[0]["status"] == "in_progress"
    # Delivery ids are durable too: a redelivery after the restart is caught.
    assert reopened.apply([], delivery_id="d-1") is False


def test_prune_forgets_old_completed_state_but_never_active_state(tmp_path) -> None:
    store = Store(tmp_path / "status.db")
    store.apply(
        [job("completed", "success", completed_at=1_000.0)],
        delivery_id="old",
        now=1_000.0,
    )
    store.apply(
        [JobUpdate(REPO, 901, 1, 7002, "in_progress")],
        delivery_id="active",
        now=1_000.0,
    )
    removed = store.prune(now=1_000.0 + 8 * 86400, retention_seconds=7 * 86400)
    assert removed == 1  # the completed job; its run stub is still in progress
    assert [j["job_id"] for j in store.jobs()] == [7002]
    assert store.apply([], delivery_id="old", now=0) is True  # its delivery id went too


def test_completed_records_leave_the_publication_window(tmp_path) -> None:
    store = Store(tmp_path / "status.db")
    store.apply([job("completed", "success", completed_at=100.0)])
    assert store.jobs(completed_since=50.0)
    assert not store.jobs(completed_since=150.0)


# -- service ------------------------------------------------------------------


@pytest.fixture
def running_service(tmp_path):
    def start(path=tmp_path / "status.db"):
        service = Service(
            store=Store(path),
            secret=SECRET,
            filters=FILTERS,
            completed_window_seconds=6 * 3600,
        )
        server = serve(service, "127.0.0.1", 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        started.append((server, service))
        return f"http://127.0.0.1:{server.server_address[1]}", service

    started: list = []
    yield start
    for server, service in started:
        server.shutdown()
        server.server_close()
        service.store.close()


def post(
    base: str, event: str, payload: dict, *, delivery: str, signature: str | None = None
) -> int:
    body = json.dumps(payload).encode()
    request = urllib.request.Request(f"{base}/webhook", data=body, method="POST")
    request.add_header("X-GitHub-Event", event)
    request.add_header("X-GitHub-Delivery", delivery)
    request.add_header("X-Hub-Signature-256", signature or sign(body))
    request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status
    except urllib.error.HTTPError as error:
        return error.code


def scrape(base: str) -> str:
    with urllib.request.urlopen(f"{base}/metrics", timeout=5) as response:
        return response.read().decode()


def test_a_run_and_job_are_published_without_any_trace(running_service) -> None:
    base, _service = running_service()
    identity = 'repository="huggingface/transformers",run_id="900:1"'
    job_identity = identity + ',job_id="7001"'

    assert post(base, "workflow_job", job_payload(), delivery="d-1") == 200
    out = scrape(base)
    assert f"ci_github_run_status{{{identity}}} 1" in out
    assert f"ci_github_job_status{{{job_identity}}} 1" in out
    assert "ci_github_job_conclusion_info" not in out  # queued: no conclusion yet

    assert (
        post(
            base,
            "workflow_run",
            run_payload(action="in_progress", status="in_progress"),
            delivery="d-2",
        )
        == 200
    )
    assert (
        post(
            base,
            "workflow_job",
            job_payload(
                action="in_progress",
                status="in_progress",
                started_at="2026-09-24T07:29:30Z",
            ),
            delivery="d-3",
        )
        == 200
    )
    assert (
        post(
            base,
            "workflow_job",
            job_payload(
                action="completed",
                status="completed",
                conclusion="cancelled",
                started_at="2026-09-24T07:29:30Z",
                completed_at="2026-09-24T07:30:30Z",
            ),
            delivery="d-4",
        )
        == 200
    )
    out = scrape(base)
    assert f"ci_github_job_status{{{job_identity}}} 3" in out
    assert (
        f'ci_github_job_conclusion_info{{{job_identity},conclusion="cancelled"}} 1'
        in out
    )
    assert (
        f"ci_github_run_status{{{identity}}} 2" in out
    )  # the job finished, not the run
    assert (
        f'ci_github_run_info{{{identity},workflow="PR CI",event="pull_request",pr="4321"}} 1'
        in out
    )
    assert (
        f'ci_github_job_info{{{job_identity},pr="4321",name="pr-ci / Check repository consistency"}} 1'
        in out
    )
    assert (
        f"ci_github_job_completed_timestamp_seconds{{{job_identity}}} 1790235030.000"
        in out
    )
    assert 'ci_github_status_deliveries_total{outcome="accepted"} 4' in out


def test_bad_signatures_store_nothing(running_service) -> None:
    base, service = running_service()
    assert (
        post(
            base,
            "workflow_job",
            job_payload(),
            delivery="d-1",
            signature="sha256=" + "0" * 64,
        )
        == 401
    )
    assert service.store.jobs() == []
    assert 'ci_github_status_deliveries_total{outcome="rejected"} 1' in scrape(base)


def test_ignored_and_duplicate_deliveries_are_acknowledged(running_service) -> None:
    base, _service = running_service()
    assert post(base, "ping", {"zen": "hi"}, delivery="p-1") == 200
    assert (
        post(base, "workflow_run", run_payload(workflow="Doc builder"), delivery="d-0")
        == 202
    )
    assert post(base, "workflow_job", job_payload(), delivery="d-1") == 200
    assert post(base, "workflow_job", job_payload(), delivery="d-1") == 200
    out = scrape(base)
    assert 'ci_github_status_deliveries_total{outcome="duplicate"} 1' in out
    assert 'ci_github_status_deliveries_total{outcome="ignored"} 2' in out


def test_the_service_serves_the_same_state_after_a_restart(
    running_service, tmp_path
) -> None:
    path = tmp_path / "restart.db"
    base, _service = running_service(path)
    post(
        base,
        "workflow_job",
        job_payload(status="in_progress", action="in_progress"),
        delivery="d-1",
    )
    before = [
        line
        for line in scrape(base).splitlines()
        if line.startswith("ci_github_job_status")
    ]
    base_again, _other = running_service(path)
    after = [
        line
        for line in scrape(base_again).splitlines()
        if line.startswith("ci_github_job_status")
    ]
    assert before == after and before


def test_cli_refuses_to_start_without_a_secret(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.delenv("GITHUB_WEBHOOK_SECRET", raising=False)
    code = cli.main(
        [
            "serve",
            "--db",
            str(tmp_path / "s.db"),
            "--repository",
            REPO,
            "--workflow",
            "PR CI",
        ]
    )
    assert code == 2
    assert "refusing" in capsys.readouterr().err


def test_duration_parsing() -> None:
    assert cli.duration("90s") == 90
    assert cli.duration("6h") == 21600
    assert cli.duration("7d") == 604800
    assert cli.duration("30") == 30


def test_a_push_run_is_filed_under_its_branch_whatever_github_lists() -> None:
    # GitHub attached PR #1 (head branch "main" in some fork) to pushes to main.
    push = webhook.parse_delivery(
        "workflow_run", run_payload(prs=[1], event="push", head_branch="main"), FILTERS
    )
    state = {"prs": push.prs, "event": push.event, "head_branch": push.head_branch}
    assert metrics.pr_label(state) == "main"
