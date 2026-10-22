import json
import threading
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen

from transformersci.otel import merge_queue as mq

NOW = 1792160000
NODE = "tests/models/muse_spark/test_processing_muse_spark.py::MuseSparkProcessingIntegrationTest::test_apply_chat_template_audio"


def span(span_id, nodeid, pr, run_id, start, status="ok", exc=None):
    """A test span as Tempo's /api/search returns it with our select()."""
    attributes = [
        {
            "key": "vcs.ref.head.name",
            "value": {"stringValue": f"gh-readonly-queue/main/pr-{pr}-abc"},
        },
        {"key": "vcs.change.id", "value": {"stringValue": str(pr)}},
        {"key": "pytest.nodeid", "value": {"stringValue": nodeid}},
        {"key": "cicd.pipeline.run.id", "value": {"stringValue": run_id}},
        {"key": "status", "value": {"stringValue": status}},
    ]
    if exc:
        attributes += [
            {"key": "exception.type", "value": {"stringValue": exc[0]}},
            {"key": "exception.message", "value": {"stringValue": exc[1]}},
        ]
    return {
        "spanID": span_id,
        "startTimeUnixNano": str(start * 10**9),
        "durationNanos": "48408444",
        "attributes": attributes,
    }


def payload(*spans):
    return {
        "traces": [
            {
                "traceID": f"{i:032x}",
                "spanSet": {"spans": [s]},
                "spanSets": [{"spans": [s]}],
            }
            for i, s in enumerate(spans)
        ]
    }


def runs(pattern, prs):
    """Runs newest first: f failed, r rerun, p passed."""
    names = {"f": "failed", "r": "rerun", "p": "passed"}
    return [{"status": names[c], "pr": pr} for c, pr in zip(pattern, prs)]


def test_parse_search_reads_selected_attributes():
    rows = mq.parse_search(
        payload(
            span(
                "a",
                NODE,
                49428,
                "38861508042:1",
                NOW - 60,
                exc=("HfHubHTTPError", "401 Unauthorized"),
            ),
            span(
                "b",
                NODE,
                49557,
                "38892094058:1",
                NOW - 30,
                status="error",
                exc=("RuntimeError", "read-only"),
            ),
        )
    )
    assert [(r["pr"], r["run_id"], r["status"], r["exception_type"]) for r in rows] == [
        ("49428", "38861508042:1", "ok", "HfHubHTTPError"),
        ("49557", "38892094058:1", "error", "RuntimeError"),
    ]
    assert rows[0]["start"] == NOW - 60


def test_parse_search_skips_spans_without_identity():
    bad = span("a", NODE, "", "1:1", NOW)
    assert mq.parse_search(payload(bad)) == []
    assert mq.parse_search({"traces": None}) == [] and mq.parse_search("x") == []


def test_outcome_reads_ok_with_exception_by_rerun_fix_date():
    before = {
        "status": "ok",
        "exception_type": "OSError",
        "start": mq.RERUN_STATUS_FIXED_AT - 1,
    }
    after = dict(before, start=mq.RERUN_STATUS_FIXED_AT)
    assert mq.outcome(before) == "failed"  # every attempt may have failed
    assert mq.outcome(after) == "rerun"  # a failure would have been ERROR
    assert mq.outcome({"status": "error", "exception_type": "", "start": 0}) == "failed"
    assert mq.outcome({"status": "ok", "exception_type": "", "start": 0}) == "passed"


def test_verdicts():
    others = [str(n) for n in range(20)]
    assert mq.verdict(runs("p" * 20, others))[0] == "pr"
    assert mq.verdict(runs("pppp", others))[0] == "unknown"  # too few runs
    assert mq.verdict(runs("f" + "p" * 19, others))[0] == "unknown"  # one other PR
    assert mq.verdict(runs("pfpppfpppppppfpppppp", others))[0] == "flaky"
    assert mq.verdict(runs("prpppfpppp", others))[0] == "flaky"  # reruns count
    assert mq.verdict(runs("ffffffff", ["1", "2"] * 4))[0] == "others"
    assert mq.verdict(runs("ffffffff", ["1", "2", "3", "4"] * 2))[0] == "main"
    assert mq.verdict([])[0] == "none"


def test_fresh_breakage_is_not_flaky():
    older = ["9"] * 17
    assert mq.verdict(runs("fff" + "p" * 17, ["1", "2", "3"] + older))[0] == "main"
    kind, reason = mq.verdict(runs("fff" + "p" * 17, ["1", "2", "1"] + older))
    assert kind == "others" and "Started failing 3 merge-queue runs ago" in reason
    # One PR failing three times in a row is that PR, not main.
    assert mq.verdict(runs("fff" + "p" * 17, ["1", "1", "1"] + older))[0] == "unknown"


def test_history_excludes_this_pr_and_keeps_one_entry_per_run():
    rows = mq.parse_search(
        payload(
            span(
                "a", NODE, 49566, "1:1", NOW - 10, status="error", exc=("E", "")
            ),  # this PR
            span("b", NODE, 49428, "2:1", NOW - 20),  # passed in one job...
            span(
                "c", NODE, 49428, "2:1", NOW - 20, status="error", exc=("E", "x")
            ),  # ...failed in another
            span("d", NODE, 49557, "3:1", NOW - 3000),
        )
    )
    entry = mq.history(rows, [NODE, "tests/x.py::T::test_y"], "49566", now=NOW)
    test = entry[NODE]
    assert [(r["pr"], r["status"]) for r in test["runs"]] == [
        ("49428", "failed"),
        ("49557", "passed"),
    ]
    assert (
        test["ran_count"] == 2
        and test["fail_count"] == 1
        and test["distinct_prs_failed"] == 1
    )
    assert test["runs"][0]["log_url"].endswith("/actions/runs/2")
    assert entry["tests/x.py::T::test_y"]["verdict"] == "none"


def test_history_window_is_newest_first_and_capped():
    rows = [
        {
            "nodeid": NODE,
            "run_id": f"{n}:1",
            "pr": str(n),
            "status": "ok",
            "exception_type": "",
            "exception_message": "",
            "start": NOW - n * 60,
        }
        for n in range(30)
    ]
    test = mq.history(rows, [NODE], "1", now=NOW)[NODE]
    assert test["ran_count"] == mq.WINDOW
    ages = [r["age_seconds"] for r in test["runs"]]
    assert ages == sorted(ages)


def test_queries_escape_and_chunk():
    nodeids = [f'tests/a.py::T::test_{n}[x"y]' for n in range(mq.NODEIDS_PER_QUERY + 1)]
    found = mq.queries(nodeids, "pytest-observability", "huggingface/transformers")
    assert len(found) == 2
    assert 'resource.vcs.ref.head.name =~ "gh-readonly-queue/.*"' in found[0]
    assert 'resource.vcs.repository.name = "huggingface/transformers"' in found[0]
    assert (
        '\\"' in found[0] and "test_0\\\\[" in found[0]
    )  # regex escape survives the string escape


def test_fetch_rows_searches_each_day_and_flags_full_or_failed_searches():
    calls = []

    def search(query, start, end):
        calls.append((start, end))
        if len(calls) == 2:
            raise OSError("tempo down")
        return payload(span("a", NODE, 1, "1:1", end - 5))

    rows, incomplete = mq.fetch_rows(
        [NODE], search, service_name="s", repository="r", now=NOW
    )
    assert (
        len(calls) == mq.LOOKBACK_DAYS
        and incomplete
        and len(rows) == mq.LOOKBACK_DAYS - 1
    )
    assert all(end - start == 86400 for start, end in calls)

    full = {"traces": [{"traceID": "1", "spanSet": {"spans": []}}] * mq.SEARCH_LIMIT}
    assert mq.fetch_rows(
        [NODE], lambda *_: full, service_name="s", repository="r", now=NOW
    )[1]


def test_queue_history_endpoint(monkeypatch):
    from transformersci.otel import trace_exporter as te

    queries = []

    def fake_get(url, timeout=None, upstream="tempo"):
        queries.append(url)
        return payload(span("a", NODE, 49428, "2:1", NOW - 60))

    monkeypatch.setattr(te, "_http_get_json", fake_get)
    te._swr_cache.clear()
    server = ThreadingHTTPServer(("127.0.0.1", 0), te.MetricsHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}/pr-ci-view/queue-history?"
    try:
        with urlopen(base + urlencode([("pr", "49566"), ("t", NODE)])) as reply:
            body = json.load(reply)
        assert body["tests"][NODE]["runs"][0]["pr"] == "49428"
        assert body["incomplete"] is False and body["window"] == mq.WINDOW
        searched = len(queries)
        assert searched == mq.LOOKBACK_DAYS and "gh-readonly-queue" in queries[0]
        with urlopen(base + urlencode([("pr", "49566"), ("t", NODE)])):
            pass
        assert len(queries) == searched  # served from the cache
        for bad in (
            [("pr", "49566")],
            [("pr", "x"), ("t", NODE)],
            [("pr", "1"), ("t", "no-separator")],
            [("pr", "1")] + [("t", f"a.py::t{n}") for n in range(mq.MAX_NODEIDS + 1)],
        ):
            try:
                urlopen(base + urlencode(bad))
            except HTTPError as error:
                assert error.code == 400
            else:
                raise AssertionError(f"accepted {bad}")
    finally:
        server.shutdown()
        server.server_close()


def test_incomplete_queue_history_is_rebuilt_on_the_next_view(monkeypatch):
    from transformersci.otel import trace_exporter as te

    monkeypatch.setattr(
        te.merge_queue, "fetch_rows", lambda *args, **kwargs: ([], True)
    )
    te._swr_cache.clear()
    server = ThreadingHTTPServer(("127.0.0.1", 0), te.MetricsHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}/pr-ci-view/queue-history?"
    try:
        with urlopen(base + urlencode([("pr", "49566"), ("t", NODE)])) as reply:
            assert json.load(reply)["incomplete"] is True
    finally:
        server.shutdown()
        server.server_close()
    # Kept only as already stale: the next view gets it and rebuilds it.
    assert [
        built_at
        for key, (built_at, _value) in te._swr_cache.items()
        if key.startswith("queue-history:")
    ] == [float("-inf")]
    te._swr_cache.clear()
