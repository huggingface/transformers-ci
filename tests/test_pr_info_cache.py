import threading

import pytest

from transformersci.otel import trace_exporter


@pytest.fixture
def clock(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(trace_exporter.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(trace_exporter, "_cached_pr_info", {})
    monkeypatch.setattr(trace_exporter, "_pr_info_refreshing", set())
    monkeypatch.delenv("PYTEST_TRACE_EXPORTER_GITHUB_CACHE_SECONDS", raising=False)
    return now


def info(title: str, state: str = "open", merged: str = "false") -> dict[str, str]:
    return {"title": title, "state": state, "merged": merged, "labels_json": "[]"}


def settle() -> None:
    for _ in range(500):
        if not trace_exporter._pr_info_refreshing:
            return
        threading.Event().wait(0.01)
    raise AssertionError("refresh did not finish")


def test_page_requests_get_an_expired_value_at_once_and_one_refresh(
    clock, monkeypatch
) -> None:
    reads: list[str] = []
    release = threading.Event()

    def fetch(repository: str, pr: str) -> dict[str, str]:
        reads.append(pr)
        if len(reads) > 1:
            assert release.wait(5)  # a slow GitHub must not hold a request
        return info(f"v{len(reads)}")

    monkeypatch.setattr(trace_exporter, "fetch_github_pr_info", fetch)
    swr = trace_exporter.fetch_github_pr_info_swr
    assert swr("o/r", "1")["title"] == "v1"  # never read: waits on GitHub
    clock[0] += 60
    assert swr("o/r", "1")["title"] == "v1" and reads == ["1"]  # fresh
    clock[0] += 600  # past 5 min: old value now, refreshed behind it
    assert [swr("o/r", "1")["title"] for _ in range(3)] == ["v1"] * 3
    release.set()
    settle()
    assert reads == ["1", "1"]  # three stale requests, one refresh
    assert swr("o/r", "1")["title"] == "v2"


def test_a_failed_refresh_keeps_the_last_good_value(clock, monkeypatch) -> None:
    # Each request on a stale value retries: GitHub is down for both retries.
    results = [info("good"), RuntimeError("GitHub 502"), RuntimeError("GitHub 502")]

    def fetch(repository: str, pr: str) -> dict[str, str]:
        result = results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(trace_exporter, "fetch_github_pr_info", fetch)
    trace_exporter.fetch_github_pr_info_swr("o/r", "2")
    clock[0] += 600
    assert trace_exporter.fetch_github_pr_info_swr("o/r", "2")["title"] == "good"
    settle()
    assert trace_exporter.fetch_github_pr_info_swr("o/r", "2")["title"] == "good"
    settle()
    assert results == []


def test_the_fallback_after_a_failed_read_is_retried_in_the_request(
    clock, monkeypatch
) -> None:
    results = [RuntimeError("GitHub 502"), info("recovered")]

    def fetch(repository: str, pr: str) -> dict[str, str]:
        result = results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(trace_exporter, "fetch_github_pr_info", fetch)
    assert trace_exporter.fetch_github_pr_info_swr("o/r", "3")["state"] == ""
    clock[0] += 600
    assert trace_exporter.fetch_github_pr_info_swr("o/r", "3")["title"] == "recovered"


@pytest.mark.parametrize(
    "state,merged,refetched",
    [("open", "false", True), ("closed", "false", False), ("closed", "true", False)],
)
def test_closed_and_merged_prs_stay_fresh_for_an_hour(
    clock, monkeypatch, state, merged, refetched
) -> None:
    reads: list[str] = []

    def fetch(repository: str, pr: str) -> dict[str, str]:
        reads.append(pr)
        return info("t", state, merged)

    monkeypatch.setattr(trace_exporter, "fetch_github_pr_info", fetch)
    trace_exporter.fetch_github_pr_info_cached("o/r", "4")
    clock[0] += 1800
    trace_exporter.fetch_github_pr_info_cached("o/r", "4")
    assert len(reads) == (2 if refetched else 1)
    clock[0] += 1900  # past an hour: even a merged PR is read again
    trace_exporter.fetch_github_pr_info_cached("o/r", "4")
    assert len(reads) == (3 if refetched else 2)


def test_a_disabled_cache_reads_every_time(clock, monkeypatch) -> None:
    monkeypatch.setenv("PYTEST_TRACE_EXPORTER_GITHUB_CACHE_SECONDS", "0")
    reads: list[str] = []
    monkeypatch.setattr(
        trace_exporter,
        "fetch_github_pr_info",
        lambda repository, pr: reads.append(pr) or info("t", "closed", "true"),
    )
    for _ in range(2):
        trace_exporter.fetch_github_pr_info_swr("o/r", "5")
    assert reads == ["5", "5"]
