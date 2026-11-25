import copy
import json
from email.message import Message
from io import BytesIO
from urllib.error import HTTPError

import pytest

from transformersci.benchmarks import bisect, page

GOOD, BAD, LATER = "a" * 40, "b" * 40, "c" * 40
SERIES = {
    "key": "gemma-fa2",
    "group": "Generation",
    "model": "google/gemma-3-1b-it",
    "scenario": "flash_attention_2 · b1 · 448+128 tokens",
    "gpu": "NVIDIA A10G #4",
    "bisect": {
        "bench_args": "-w 5 -i 6 -b 1 -s 448 -n 128",
        "config": {"attn_implementation": "flash_attention_2"},
    },
    "points": [
        {"date": "2026-11-20T03:00:00", "commit": GOOD, "values": {"e2e": [5450.0, 5400.0, 5500.0]}, "runs": 3, "failed": 0},
        {"date": "2026-11-21T03:00:00", "commit": BAD, "values": {"e2e": [5790.0, 5700.0, 5800.0]}, "runs": 3, "failed": 0},
        {"date": "2026-11-22T03:00:00", "commit": LATER, "values": {"e2e": [5800.0, 5700.0, 5900.0]}, "runs": 3, "failed": 0},
    ],
}  # fmt: skip
SNAPSHOT = {
    "pages": {"main.html": {"series": [SERIES], "metrics": {}, "series_metrics": {}}}
}
MEMBER = {
    "id": 1,
    "user_id": 42,
    "kind": "browser",
    "scope": "browser",
    "login": "member",
}


class Handler:
    def __init__(self, path, body=None, auth=None):
        self.path = path
        self.headers = Message()
        payload = json.dumps(body).encode() if body is not None else b""
        if auth:
            self.headers["Authorization"] = auth
        self.headers["Content-Type"] = "application/json"
        self.headers["Content-Length"] = str(len(payload))
        self.rfile = BytesIO(payload)
        self.reply = None

    def _reply_json(self, status, body):
        self.reply = status, body

    def _send(self, status, content_type, payload, **kwargs):
        self.reply = status, payload


@pytest.fixture
def service(tmp_path, monkeypatch):
    monkeypatch.setenv("TCI_BENCHMARK_BISECT_ENABLED", "1")
    monkeypatch.setenv("TCI_BENCHMARK_BISECT_TOKEN", "worker-secret")
    monkeypatch.setenv("TCI_BENCHMARK_BISECT_SLACK_CHANNEL", "C123")
    monkeypatch.setenv("PYTEST_TRACE_EXPORTER_RUN_STORE", str(tmp_path))
    monkeypatch.setattr(bisect, "_stores", {})
    monkeypatch.setattr(bisect, "snapshot", lambda: SNAPSHOT)
    monkeypatch.setattr(bisect, "requester", lambda handler: MEMBER)
    monkeypatch.setattr(bisect, "still_member", lambda user: None)
    monkeypatch.setattr(bisect.auth, "config", lambda: None)
    posts = []

    def message(channel, text, blocks=None, *, thread_ts=None):
        posts.append((channel, text, thread_ts))
        return {"ok": True, "ts": "1.0"}

    monkeypatch.setattr(bisect.slack_api, "message", message)
    return posts


def submit(good=GOOD, bad=BAD):
    h = Handler("/benchmarks/bisect", {"series": "gemma-fa2", "good": good, "bad": bad})
    assert bisect.handle(h, "POST")
    return h.reply


def test_the_job_comes_from_the_snapshot_not_the_request():
    job = bisect.build_job(SNAPSHOT, "gemma-fa2", GOOD, BAD)
    assert job["model"] == "google/gemma-3-1b-it"
    assert job["scenario"] == {"attn_implementation": "flash_attention_2"}
    assert job["bench_args"] == "-w 5 -i 6 -b 1 -s 448 -n 128"
    assert job["expected"] == pytest.approx(6.24)
    assert (job["good_date"], job["bad_date"]) == ("2026-11-20", "2026-11-21")


@pytest.mark.parametrize(
    ("series", "good", "bad", "error"),
    [("nope", GOOD, BAD, "unknown_series"), ("gemma-fa2", GOOD, LATER, "not_adjacent_daily_points"),
     ("gemma-fa2", BAD, GOOD, "not_adjacent_daily_points"), ("gemma-fa2", "d" * 40, BAD, "not_adjacent_daily_points")],
)  # fmt: skip
def test_only_adjacent_daily_points_of_a_known_series(series, good, bad, error):
    with pytest.raises(bisect.Error) as exc:
        bisect.build_job(SNAPSHOT, series, good, bad)
    assert exc.value.error == error


def test_a_series_without_a_bisect_spec_cannot_be_bisected():
    snap = {"pages": {"main.html": {"series": [{**SERIES, "bisect": None}]}}}
    with pytest.raises(bisect.Error) as exc:
        bisect.build_job(snap, "gemma-fa2", GOOD, BAD)
    assert exc.value.status == 422


def test_submit_claim_report_and_slack_thread(service):
    status, body = submit()
    assert status == 201 and body["url"] == f"/benchmarks/bisect/{body['id']}"
    assert (
        service[0][0] == "C123"
        and "requested by member" in service[0][1]
        and service[0][2] is None
    )

    worker = "Bearer worker-secret"
    h = Handler("/benchmarks/bisect/next", auth=worker)
    bisect.handle(h, "GET")
    status, job = h.reply
    assert status == 200 and job["id"] == body["id"] and job["state"] == "running"
    assert (
        job["scenario"] == {"attn_implementation": "flash_attention_2"}
        and "slack_ts" not in job
    )

    h = Handler("/benchmarks/bisect/next", auth=worker)
    bisect.handle(h, "GET")
    assert h.reply[0] == 204  # nothing else queued

    step = {"role": "reproduce", "base": GOOD, "head": BAD, "verdict": "not_reproduced", "change": 0.78,
            "ci": [-2.9, 4.59], "ab_dir": ""}  # fmt: skip
    h = Handler(
        f"/benchmarks/bisect/{job['id']}/report",
        {"kind": "step", "step": step},
        auth=worker,
    )
    bisect.handle(h, "POST")
    assert h.reply[0] == 200
    result = {"state": "not_reproduced", "reason": "rules out +6.24%", "steps": [step]}
    h = Handler(
        f"/benchmarks/bisect/{job['id']}/report",
        {"kind": "done", "result": result},
        auth=worker,
    )
    bisect.handle(h, "POST")
    assert h.reply[0] == 200

    threaded = [p for p in service if p[2] == "1.0"]
    assert [p[1].split()[0] for p in threaded] == ["Claimed", "reproduce", "*Bisect"]
    assert "not a transformers change" in threaded[-1][1]

    h = Handler(f"/benchmarks/bisect/{job['id']}?format=json")
    bisect.handle(h, "GET")
    assert h.reply[1]["state"] == "done" and h.reply[1]["steps"] == [step]
    h = Handler(f"/benchmarks/bisect/{job['id']}")
    bisect.handle(h, "GET")
    assert h.reply[0] == 200 and b"did not reproduce" in h.reply[1]


def test_a_finished_job_takes_no_more_reports(service):
    _, body = submit()
    worker = "Bearer worker-secret"
    bisect.handle(Handler("/benchmarks/bisect/next", auth=worker), "GET")
    done = {"kind": "done", "result": {"state": "failed", "reason": "x"}}
    bisect.handle(
        Handler(f"/benchmarks/bisect/{body['id']}/report", done, auth=worker), "POST"
    )
    h = Handler(f"/benchmarks/bisect/{body['id']}/report", done, auth=worker)
    bisect.handle(h, "POST")
    assert h.reply == (409, {"error": "job_not_running"})


def test_worker_endpoints_need_the_worker_token(service):
    for auth in (None, "Bearer wrong", "Bearer session-of-a-member"):
        h = Handler("/benchmarks/bisect/next", auth=auth)
        bisect.handle(h, "GET")
        assert h.reply[0] == 401


def test_one_job_per_series_and_a_bounded_queue(service, monkeypatch):
    assert submit()[0] == 201
    assert submit(BAD, LATER) == (409, {"error": "series_already_bisecting"})
    monkeypatch.setattr(bisect, "MAX_PENDING", 1)
    other = {**SERIES, "key": "other"}
    monkeypatch.setattr(
        bisect,
        "snapshot",
        lambda: {"pages": {"main.html": {"series": [SERIES, other]}}},
    )
    h = Handler("/benchmarks/bisect", {"series": "other", "good": GOOD, "bad": BAD})
    bisect.handle(h, "POST")
    assert h.reply == (429, {"error": "bisect_queue_full"})


def test_an_ex_member_is_refused_before_anything_is_queued(service, monkeypatch):
    def refuse(user):
        raise bisect.Error(403, "account_not_allowed")

    monkeypatch.setattr(bisect, "still_member", refuse)
    assert submit() == (403, {"error": "account_not_allowed"})
    assert bisect.store().recent() == [] and service == []


def test_still_member_asks_github_through_the_app(monkeypatch):
    monkeypatch.setattr(bisect.auth, "config", lambda: None)
    monkeypatch.setattr(bisect.github_app, "app_token", lambda: "app")
    seen = []

    def github_json(url, token=None):
        seen.append((url, token))
        if "member" in url and "gone" in url:
            raise HTTPError(url, 404, "Not Found", None, None)
        return {"state": "pending" if "invited" in url else "active"}

    monkeypatch.setattr(bisect.auth, "github_json", github_json)
    bisect.still_member({"user_id": 1, "login": "member"})
    assert seen == [
        ("https://api.github.com/orgs/huggingface/memberships/member", "app")
    ]
    for login in ("gone", "invited"):
        with pytest.raises(bisect.Error) as exc:
            bisect.still_member({"user_id": 2, "login": login})
        assert exc.value.error == "account_not_allowed"
    monkeypatch.setattr(bisect.github_app, "app_token", lambda: "")
    with pytest.raises(bisect.Error) as exc:  # fail closed
        bisect.still_member({"user_id": 1, "login": "member"})
    assert exc.value.status == 503


def test_still_member_relies_on_the_login_check_when_the_app_cannot_read_members(
    monkeypatch,
):
    monkeypatch.setattr(bisect.auth, "config", lambda: None)
    monkeypatch.setattr(bisect.github_app, "app_token", lambda: "app")

    def github_json(url, token=None):
        raise HTTPError(url, 403, "Resource not accessible by integration", None, None)

    monkeypatch.setattr(bisect.auth, "github_json", github_json)
    bisect.still_member({"user_id": 1, "login": "member"})  # no raise

    def down(url, token=None):
        raise HTTPError(url, 502, "Bad Gateway", None, None)

    monkeypatch.setattr(bisect.auth, "github_json", down)
    with pytest.raises(bisect.Error) as exc:  # anything else still fails closed
        bisect.still_member({"user_id": 1, "login": "member"})
    assert exc.value.status == 503


def test_cli_tokens_need_the_bench_submit_scope(monkeypatch):
    monkeypatch.setattr(
        bisect.auth,
        "identity",
        lambda *a, **kw: {**MEMBER, "kind": "cli", "scope": "mic:submit"},
    )
    with pytest.raises(bisect.Error) as exc:
        bisect.requester(Handler("/benchmarks/bisect", {}, auth="Bearer cli"))
    assert exc.value.error == "insufficient_scope"
    monkeypatch.setattr(
        bisect.auth,
        "identity",
        lambda *a, **kw: {**MEMBER, "kind": "cli", "scope": "bench:submit"},
    )
    assert (
        bisect.requester(Handler("/benchmarks/bisect", {}, auth="Bearer cli"))["login"]
        == "member"
    )


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("TCI_BENCHMARK_BISECT_ENABLED", raising=False)
    assert not bisect.handle(Handler("/benchmarks/bisect/next"), "GET")


def test_lost_jobs_fail(service, monkeypatch):
    _, body = submit()
    bisect.handle(
        Handler("/benchmarks/bisect/next", auth="Bearer worker-secret"), "GET"
    )
    monkeypatch.setattr(bisect, "LOST_SECONDS", -1)
    bisect.store().expire_lost()
    job = bisect.store().get(body["id"])
    assert (
        job["state"] == "done"
        and job["result"]["reason"] == "the worker stopped reporting"
    )


def test_daily_page_shows_bisect_buttons_only_when_enabled(monkeypatch):
    data = {"series": [SERIES], "metrics": {"e2e": {"label": "End-to-end latency", "unit": "ms"}},
            "series_metrics": {}}  # fmt: skip
    monkeypatch.delenv("TCI_BENCHMARK_BISECT_ENABLED", raising=False)
    assert 'class="bisect"' not in page.daily(data, {})
    monkeypatch.setenv("TCI_BENCHMARK_BISECT_ENABLED", "1")
    html = page.daily(data, {})
    assert (
        f'data-good="{GOOD}" data-bad="{BAD}"' in html
    )  # the largest step, and its table row
    assert f'data-good="{BAD}" data-bad="{LATER}"' in html
    assert (
        "Largest day-over-day e2e step (shaded on the chart): 2026-11-20 → 2026-11-21, +6.24%"
        in html
    )
    assert html.count('class="step"') == 1 and 'class="step-line"' in html
    assert "/benchmarks/bisect" in html and "X-TCI-Action" in html


def test_no_bisect_this_step_under_two_percent(monkeypatch):
    monkeypatch.setenv("TCI_BENCHMARK_BISECT_ENABLED", "1")
    flat = copy.deepcopy(SERIES)
    for p in flat["points"]:
        p["values"]["e2e"] = [1000.0, 990.0, 1010.0]
    flat["points"][-1]["values"]["e2e"] = [1019.0, 1010.0, 1030.0]  # +1.9%
    data = {"series": [flat], "metrics": {"e2e": {"label": "End-to-end latency", "unit": "ms"}},
            "series_metrics": {}}  # fmt: skip
    html = page.daily(data, {})
    assert "Bisect this step" not in html and 'class="step"' not in html
    assert page.largest_step(flat["points"]) is None
    assert page.largest_step(flat["points"], minimum=1.0)[2] > 1.8


HEAD = "e" * 40
COMPILED = {**SERIES, "key": "gemma-compiled", "scenario": "compiled",
            "bisect": {"bench_args": "-b 1", "config": {"attn_implementation": "sdpa", "compile_kwargs": {"mode": "default"}}}}  # fmt: skip


@pytest.fixture
def pr_service(service, monkeypatch):
    monkeypatch.setattr(
        bisect,
        "snapshot",
        lambda: {"pages": {"main.html": {"series": [SERIES, COMPILED]}}},
    )
    monkeypatch.setattr(
        bisect, "pull_request", lambda pr: {"head": HEAD, "title": "Faster attention"}
    )
    return service


def submit_pr(pr=123):
    h = Handler("/benchmarks/pr-bench", {"pr": pr})
    assert bisect.handle(h, "POST")
    return h.reply


def test_pr_bench_runs_the_uncompiled_daily_scenarios(pr_service):
    status, reply = submit_pr()
    assert status == 201 and reply["url"] == f"/benchmarks/pr-bench/{reply['id']}"
    job = bisect.store().get(reply["id"])
    assert job["kind"] == "pr" and job["pr"] == 123 and job["head"] == HEAD
    assert [s["series"] for s in job["scenarios"]] == ["gemma-fa2"]
    assert job["scenarios"][0]["bench_args"] == "-w 5 -i 6 -b 1 -s 448 -n 128"
    assert "PR bench" in pr_service[0][1] and "#123" in pr_service[0][1]
    assert submit_pr() == (409, {"error": "pr_already_queued"})
    assert submit_pr("12; rm") == (400, {"error": "invalid_request"})


def test_pr_bench_reports_one_step_per_scenario_and_links_the_results(pr_service):
    _, reply = submit_pr()
    claim = Handler("/benchmarks/bisect/next", auth="Bearer worker-secret")
    bisect.handle(claim, "GET")
    assert claim.reply[1]["kind"] == "pr"
    step = {"model": SERIES["model"], "scenario": SERIES["scenario"], "base": GOOD, "head": HEAD,
            "change": -3.1, "ci": [-4.0, -2.2], "verdict": "faster"}  # fmt: skip
    for kind, payload in (
        ("step", {"step": step}),
        ("done", {"result": {"state": "done"}}),
    ):
        h = Handler(f"/benchmarks/bisect/{reply['id']}/report", {"kind": kind, **payload},
                    auth="Bearer worker-secret")  # fmt: skip
        bisect.handle(h, "POST")
        assert h.reply[0] == 200
    texts = [t for _, t, _ in pr_service]
    assert any("-3.10% [-4.00, -2.20] → faster" in t for t in texts)
    assert f"/benchmarks/pr-bench/{reply['id']}|results>" in texts[-1]
    page_ = Handler(f"/benchmarks/pr-bench/{reply['id']}")
    bisect.handle(page_, "GET")
    assert b"-3.10%" in page_.reply[1] and b"Faster attention" in page_.reply[1]
    listing = Handler("/benchmarks/bisect")
    bisect.handle(listing, "GET")
    assert b"PR #123" in listing.reply[1]


def test_a_closed_pr_is_not_benched(service, monkeypatch):
    monkeypatch.setattr(bisect.github_app, "app_token", lambda: "app")
    monkeypatch.setattr(
        bisect.auth, "github_json", lambda url, token=None: {"state": "closed"}
    )
    assert submit_pr() == (422, {"error": "pr_not_open"})


def test_slack_bench_queues_for_the_linked_login(pr_service):
    from transformersci.otel import slack_commands

    reply = slack_commands.bench("123", "member", "https://dash")
    assert (
        "PR bench queued for #123" in reply
        and "https://dash/benchmarks/pr-bench/" in reply
    )
    assert bisect.store().recent()[0]["requester"] == "member"
    assert "already has a bench" in slack_commands.bench(
        "123", "member", "https://dash"
    )


def test_a_running_bisect_turns_the_button_into_a_link_to_its_page(service):
    data = {"series": [SERIES], "metrics": {"e2e": {"label": "End-to-end latency", "unit": "ms"}},
            "series_metrics": {}}  # fmt: skip
    assert "Bisect this step" in page.daily(data, {})
    _, reply = submit()
    html = page.daily(data, {})
    assert "Bisect this step" not in html and 'class="bisect"' not in html
    assert (
        f'href="/benchmarks/bisect/{reply["id"]}"' in html and "Bisect queued" in html
    )
    claim = Handler("/benchmarks/bisect/next", auth="Bearer worker-secret")
    bisect.handle(claim, "GET")
    assert "Bisect in progress" in page.daily(data, {})


def test_job_pages_update_themselves_until_done(service):
    _, reply = submit()
    h = Handler(f"/benchmarks/bisect/{reply['id']}")
    bisect.handle(h, "GET")
    html = h.reply[1].decode()
    assert (
        'id="job"' in html
        and "DOMParser" in html
        and "data-done" not in html.split("<script>")[0]
    )
    assert "reload to update" not in html


def test_bench_queue_is_for_signed_in_members_only(pr_service, monkeypatch):
    _, first = submit()
    _, second = submit_pr()
    monkeypatch.setattr(bisect.auth, "identity", lambda headers, cli=False: None)
    h = Handler("/benchmarks/queue")
    assert bisect.handle(h, "GET")
    html = h.reply[1].decode()
    assert (
        "Sign in with GitHub" in html
        and first["id"] not in html
        and second["id"] not in html
    )
    assert "/auth/github?next=/benchmarks/queue" in html


def test_bench_queue_shows_running_queued_and_the_last_ten(pr_service, monkeypatch):
    monkeypatch.setattr(bisect.auth, "identity", lambda headers, cli=False: MEMBER)
    db = bisect.store()
    for i in range(12):  # twelve finished PR benches: only the last ten are listed
        job = db.create({**bisect.build_pr_job(bisect.snapshot(), str(1000 + i), {"head": HEAD, "title": f"t{i}"})},
                        "member", 42)  # fmt: skip
        db.claim()
        db.finish(job["id"], {"state": "done"})
    _, running = submit()
    bisect.handle(
        Handler("/benchmarks/bisect/next", auth="Bearer worker-secret"), "GET"
    )
    _, queued = submit_pr()
    h = Handler("/benchmarks/queue")
    bisect.handle(h, "GET")
    html = h.reply[1].decode()
    running_part, rest = html.split("<h2>Queued</h2>")
    queued_part, finished_part = rest.split("<h2>Last 10 finished</h2>")
    assert running["id"] in running_part and "job-progress" in running_part
    assert queued["id"] in queued_part and "PR #123" in queued_part
    assert (
        "PR #1011" in finished_part
        and "PR #1002" in finished_part
        and "PR #1001 " not in finished_part
    )
    assert 'id="job"' in html and "DOMParser" in html


def test_bench_queue_tab_has_a_lock_and_shows_only_when_enabled(monkeypatch):
    monkeypatch.delenv("TCI_BENCHMARK_BISECT_ENABLED", raising=False)
    assert "Bench queue" not in page.tabs("main")
    monkeypatch.setenv("TCI_BENCHMARK_BISECT_ENABLED", "1")
    nav = page.tabs("queue")
    assert (
        'href="/benchmarks/queue" aria-current="page">Bench queue<svg class="tci-lock"'
        in nav
    )


def test_bench_queue_tab_pulses_while_a_bench_runs(service):
    assert "queue-busy" not in page.tabs("main")
    submit()
    assert "queue-busy" not in page.tabs("main")  # queued is not running
    bisect.handle(
        Handler("/benchmarks/bisect/next", auth="Bearer worker-secret"), "GET"
    )
    assert 'class="queue-busy"' in page.tabs("main")
