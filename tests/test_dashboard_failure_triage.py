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

import unittest
from unittest.mock import patch

from transformersci.agentic import dashboard_failure_triage as dft


class _Resp:
    """urllib.request.urlopen context-manager stub (mirrors the itf tests)."""

    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _otlp_trace(spans):
    """Wrap ``[(nodeid, exc_type, exc_msg, exc_stack, has_exc)]`` into an OTLP
    trace dict shaped like Tempo's ``/api/traces/{id}`` response."""
    out_spans = []
    for nodeid, etype, emsg, estack, has_exc in spans:
        span = {
            "name": nodeid,
            "attributes": [
                {"key": "pytest.nodeid", "value": {"stringValue": nodeid}},
                {"key": "pytest.span_type", "value": {"stringValue": "test"}},
            ],
            "events": [],
        }
        if has_exc:
            span["events"].append(
                {
                    "name": "exception",
                    "attributes": [
                        {"key": "exception.type", "value": {"stringValue": etype}},
                        {"key": "exception.message", "value": {"stringValue": emsg}},
                        {
                            "key": "exception.stacktrace",
                            "value": {"stringValue": estack},
                        },
                    ],
                }
            )
        out_spans.append(span)
    return {"batches": [{"resource": {}, "scopeSpans": [{"spans": out_spans}]}]}


_URL = (
    "https://transformers-ci.lor-e.huggingface.cool/d/pytest-test/test?"
    "var-trace_id=abc123&"
    "var-test_nodeid=tests/models/foo/test_modeling_foo.py::FooTest::test_x&"
    "var-exception_type=AttributeError&var-test_job=run_tests_gpu&"
    "var-pr=555&var-run_id=42&var-test_function=$__all"
)


class ParseDashboardUrlTest(unittest.TestCase):
    def test_extracts_wanted_vars(self):
        p = dft.parse_dashboard_url(_URL)
        self.assertEqual(p["trace_id"], "abc123")
        self.assertEqual(
            p["test_nodeid"], "tests/models/foo/test_modeling_foo.py::FooTest::test_x"
        )
        self.assertEqual(p["exception_type"], "AttributeError")
        self.assertEqual(p["test_job"], "run_tests_gpu")
        self.assertEqual(p["pr"], "555")
        self.assertEqual(p["run_id"], "42")

    def test_drops_all_sentinel_and_empty(self):
        # `$__all` and blank values are treated as absent.
        self.assertIsNone(dft.parse_dashboard_url(_URL)["test_function"])
        p = dft.parse_dashboard_url("https://x/d/pytest-test/test?var-trace_id=")
        self.assertIsNone(p["trace_id"])

    def test_missing_vars_are_none(self):
        p = dft.parse_dashboard_url("https://x/d/pytest-test/test")
        self.assertTrue(all(v is None for v in p.values()))


class ExtractFailureTest(unittest.TestCase):
    def test_matches_nodeid_among_many(self):
        trace = _otlp_trace(
            [
                ("a::t1", "ValueError", "boom", "stack1", True),
                ("a::t2", "AttributeError", "no attr", "stack2", True),
            ]
        )
        f = dft.extract_failure(trace, nodeid="a::t2")
        self.assertEqual(f["nodeid"], "a::t2")
        self.assertEqual(f["exception_type"], "AttributeError")
        self.assertEqual(f["exception_message"], "no attr")

    def test_falls_back_to_first_exception_without_nodeid(self):
        trace = _otlp_trace(
            [
                ("a::t1", "", "", "", False),  # no exception event → skipped
                ("a::t2", "ValueError", "boom", "s", True),
            ]
        )
        f = dft.extract_failure(trace)
        self.assertEqual(f["nodeid"], "a::t2")

    def test_none_when_no_exception(self):
        trace = _otlp_trace([("a::t1", "", "", "", False)])
        self.assertIsNone(dft.extract_failure(trace))


class FetchTraceTest(unittest.TestCase):
    def test_parses_json_body(self):
        trace = _otlp_trace([("a::t", "ValueError", "boom", "s", True)])
        with patch.object(
            dft.urllib.request,
            "urlopen",
            return_value=_Resp(dft.json.dumps(trace).encode()),
        ):
            got = dft.fetch_trace("abc")
        self.assertEqual(got["batches"][0]["scopeSpans"][0]["spans"][0]["name"], "a::t")

    def test_raises_on_too_large(self):
        with patch.object(
            dft.urllib.request,
            "urlopen",
            return_value=_Resp(b"response larger than the max"),
        ):
            with self.assertRaises(dft.TraceFetchError):
                dft.fetch_trace("abc")

    def test_raises_on_http_error(self):
        def boom(*a, **k):
            raise dft.urllib.error.URLError("down")

        with patch.object(dft.urllib.request, "urlopen", side_effect=boom):
            with self.assertRaises(dft.TraceFetchError):
                dft.fetch_trace("abc")


class FingerprintTest(unittest.TestCase):
    def test_stable_across_addresses_and_ints(self):
        # Hex addresses / integers in the message are normalized away, so the
        # same logical failure fingerprints identically across runs.
        a = dft.failure_fingerprint(
            "a::t", "AttributeError", "obj at 0x7f1c061fdf30 line 42"
        )
        b = dft.failure_fingerprint(
            "a::t", "AttributeError", "obj at 0x9999999999 line 99"
        )
        self.assertEqual(a, b)

    def test_distinguishes_nodeid_and_type(self):
        base = dft.failure_fingerprint("a::t", "AttributeError", "x")
        self.assertNotEqual(
            base, dft.failure_fingerprint("a::t2", "AttributeError", "x")
        )
        self.assertNotEqual(base, dft.failure_fingerprint("a::t", "ValueError", "x"))

    def test_marker_and_branch(self):
        fp = dft.failure_fingerprint("a::t", "AttributeError", "x")
        self.assertEqual(
            dft.fingerprint_marker(fp), f"<!-- serge-dashboard-fix:sha256:{fp} -->"
        )
        self.assertEqual(dft.task_branch_prefix(fp), f"serge/fix/dash-{fp[:12]}")

    def test_symbol(self):
        self.assertEqual(dft.test_symbol("a/b.py::Cls::test_x"), "test_x")
        self.assertEqual(dft.test_symbol(""), "")


class SearchExistingWorkTest(unittest.TestCase):
    def test_filters_own_automation_and_annotates_error(self):
        items = [
            {
                "number": 1,
                "title": "Fix test_x AttributeError",
                "body": "human PR",
                "pull_request": {"url": "u1/pulls/1"},
                "html_url": "u1",
            },
            {
                "number": 2,
                "title": "serge fix",
                "body": "<!-- serge-dashboard-fix:sha256:z -->",
                "html_url": "u2",
            },
            {"number": 3, "title": "unrelated", "body": "nothing", "html_url": "u3"},
        ]
        with patch.object(dft, "search_issues", return_value=items):
            hits = dft.search_existing_work(
                "o/r", "a::test_x", "test_x", "AttributeError", "tok"
            )
        nums = [h["number"] for h in hits]
        self.assertEqual(nums, [1, 3])  # #2 (our marker) filtered out
        self.assertTrue(hits[0]["is_pr"])
        self.assertTrue(hits[0]["mentions_error"])
        self.assertFalse(hits[1]["mentions_error"])


class CheckLockTest(unittest.TestCase):
    def _patches(self, *, pulls=None, issue=None, search=None):
        return (
            patch.object(dft, "list_open_pulls", return_value=pulls or []),
            patch.object(dft, "find_open_issue_by_marker", return_value=issue),
            patch.object(dft, "search_issues", return_value=search or []),
        )

    def test_clear(self):
        for pchr in self._patches():
            pchr.start()
        self.addCleanup(patch.stopall)
        lock = dft.check_lock("o/r", "fp" * 32, "a::t", "t", "E", "tok")
        self.assertFalse(lock["locked"])

    def test_locked_by_open_pr(self):
        fp = dft.failure_fingerprint("a::t", "E", "m")
        marker = dft.fingerprint_marker(fp)
        pulls = [{"number": 7, "body": marker, "head": {"ref": "x"}}]
        for pchr in self._patches(pulls=pulls):
            pchr.start()
        self.addCleanup(patch.stopall)
        lock = dft.check_lock("o/r", fp, "a::t", "t", "E", "tok")
        self.assertTrue(lock["locked"])
        self.assertEqual(lock["existing_pr"], 7)

    def test_locked_by_existing_issue(self):
        for pchr in self._patches(issue=99):
            pchr.start()
        self.addCleanup(patch.stopall)
        lock = dft.check_lock("o/r", "fp" * 32, "a::t", "t", "E", "tok")
        self.assertTrue(lock["locked"])
        self.assertEqual(lock["existing_issue"], 99)

    def test_locked_by_human_work(self):
        search = [
            {"number": 3, "title": "human fixing a::t", "body": "wip", "html_url": "u"}
        ]
        for pchr in self._patches(search=search):
            pchr.start()
        self.addCleanup(patch.stopall)
        lock = dft.check_lock("o/r", "fp" * 32, "a::t", "t", "E", "tok")
        self.assertTrue(lock["locked"])
        self.assertEqual(len(lock["human_work"]), 1)


class BuildPayloadTest(unittest.TestCase):
    def test_new_pr_payload(self):
        fp = dft.failure_fingerprint("a::t", "E", "m")
        payload = dft.build_dashboard_payload(
            "huggingface/transformers", "main", "ctx", "the title", fingerprint=fp
        )
        self.assertEqual(payload["repo"], "huggingface/transformers")
        self.assertEqual(payload["output"]["mode"], "new_pr")
        self.assertEqual(payload["output"]["branch_prefix"], dft.task_branch_prefix(fp))
        self.assertEqual(payload["output"]["title"], "the title")
        self.assertIn("minimal patch", payload["instruction"])
        self.assertNotIn("tracking_issue", payload)

    def test_existing_pr_and_tracking_issue(self):
        fp = dft.failure_fingerprint("a::t", "E", "m")
        payload = dft.build_dashboard_payload(
            "o/r",
            "main",
            "ctx",
            None,
            fingerprint=fp,
            existing_pr=12,
            tracking_issue=34,
        )
        self.assertEqual(payload["output"], {"mode": "existing_pr", "pr_number": 12})
        self.assertEqual(payload["tracking_issue"], 34)


class RenderContextTest(unittest.TestCase):
    def test_carries_marker_and_relates_to(self):
        failure = {
            "nodeid": "a::t",
            "exception_type": "AttributeError",
            "exception_message": "no attr",
            "exception_stacktrace": "line1\nline2",
        }
        parsed = dft.parse_dashboard_url(_URL)
        fp = dft.failure_fingerprint("a::t", "AttributeError", "no attr")
        ctx = dft.render_serge_context(
            failure, parsed, fp, "https://g", issue_number=77
        )
        self.assertIn(dft.fingerprint_marker(fp), ctx)
        self.assertIn("Relates to #77", ctx)
        self.assertIn("AttributeError", ctx)
        self.assertIn("no attr", ctx)


class EnsureIssueTest(unittest.TestCase):
    def test_updates_when_issue_exists(self):
        with (
            patch.object(dft, "find_open_issue_by_marker", return_value=5),
            patch.object(dft, "update_issue_body", return_value=True) as upd,
            patch.object(dft, "create_issue") as crt,
        ):
            n = dft.ensure_failure_issue("o/r", "fp", "t", "body", "tok")
        self.assertEqual(n, 5)
        upd.assert_called_once()
        crt.assert_not_called()

    def test_creates_when_absent(self):
        with (
            patch.object(dft, "find_open_issue_by_marker", return_value=None),
            patch.object(dft, "create_issue", return_value=8) as crt,
        ):
            n = dft.ensure_failure_issue("o/r", "fp", "t", "body", "tok")
        self.assertEqual(n, 8)
        crt.assert_called_once()


_FAILURE = {
    "nodeid": "tests/models/foo/test_modeling_foo.py::FooTest::test_x",
    "exception_type": "AttributeError",
    "exception_message": "no attr",
    "exception_stacktrace": "line1\nline2",
}


class MainProposeVsDispatchTest(unittest.TestCase):
    def test_propose_only_writes_nothing(self):
        with (
            patch.object(dft, "fetch_trace_failure", return_value=dict(_FAILURE)),
            patch.object(dft, "list_open_pulls", return_value=[]),
            patch.object(dft, "find_open_issue_by_marker", return_value=None),
            patch.object(dft, "search_issues", return_value=[]),
            patch.object(dft, "dispatch_to_serge") as disp,
            patch.object(dft, "create_issue") as crt,
        ):
            rc = dft.main([_URL])
        self.assertEqual(rc, 0)
        disp.assert_not_called()
        crt.assert_not_called()

    def test_dispatch_refused_when_locked(self):
        with (
            patch.object(dft, "fetch_trace_failure", return_value=dict(_FAILURE)),
            patch.object(dft, "list_open_pulls", return_value=[]),
            patch.object(
                dft, "find_open_issue_by_marker", return_value=42
            ),  # lock held
            patch.object(dft, "search_issues", return_value=[]),
            patch.object(dft, "dispatch_to_serge") as disp,
            patch.dict(dft.os.environ, {"SERGE_OIDC_TOKEN": "tok"}, clear=False),
        ):
            rc = dft.main([_URL, "--dispatch", "--serge-url", "http://s"])
        self.assertEqual(rc, 0)
        disp.assert_not_called()

    def test_dispatch_fires_when_clear(self):
        with (
            patch.object(dft, "fetch_trace_failure", return_value=dict(_FAILURE)),
            patch.object(dft, "list_open_pulls", return_value=[]),
            patch.object(dft, "find_open_issue_by_marker", return_value=None),
            patch.object(dft, "search_issues", return_value=[]),
            patch.object(dft, "create_issue", return_value=100),
            patch.object(dft, "update_issue_body", return_value=True),
            patch.object(
                dft,
                "dispatch_to_serge",
                return_value={
                    "id": "job1",
                    "url": "/tasks/huggingface/transformers/job1",
                },
            ) as disp,
            patch.object(dft, "reconcile_failure_issue", return_value={}) as rec,
            patch.dict(dft.os.environ, {"SERGE_OIDC_TOKEN": "tok"}, clear=False),
        ):
            rc = dft.main([_URL, "--dispatch", "--serge-url", "http://s"])
        self.assertEqual(rc, 0)
        disp.assert_called_once()
        rec.assert_called_once()
        # the dispatched payload should be a new_pr task for the right repo
        payload = disp.call_args.args[2]
        self.assertEqual(payload["output"]["mode"], "new_pr")
        self.assertEqual(payload["tracking_issue"], 100)

    def test_forced_dispatch_overrides_lock(self):
        with (
            patch.object(dft, "fetch_trace_failure", return_value=dict(_FAILURE)),
            patch.object(dft, "list_open_pulls", return_value=[]),
            patch.object(dft, "find_open_issue_by_marker", return_value=42),  # locked
            patch.object(dft, "search_issues", return_value=[]),
            patch.object(dft, "create_issue", return_value=100),
            patch.object(dft, "update_issue_body", return_value=True),
            patch.object(
                dft,
                "dispatch_to_serge",
                return_value={"id": "j", "url": "/tasks/o/r/j"},
            ) as disp,
            patch.object(dft, "reconcile_failure_issue", return_value={}),
            patch.dict(dft.os.environ, {"SERGE_OIDC_TOKEN": "tok"}, clear=False),
        ):
            rc = dft.main([_URL, "--dispatch", "--force", "--serge-url", "http://s"])
        self.assertEqual(rc, 0)
        disp.assert_called_once()


class ReconcileTest(unittest.TestCase):
    def test_resolves_when_pr_appears(self):
        fp = dft.failure_fingerprint("a::t", "E", "m")
        marker = dft.fingerprint_marker(fp)
        pulls = [{"number": 21, "body": marker, "head": {"ref": "x"}}]
        with (
            patch.object(dft, "list_open_pulls", return_value=pulls),
            patch.object(dft, "update_issue_body", return_value=True) as upd,
            patch.object(dft.time, "sleep", lambda _s: None),
        ):
            out = dft.reconcile_failure_issue(
                "o/r",
                fp,
                dict(_FAILURE),
                {},
                "https://g",
                issue_number=9,
                job_id="job1",
                serge_task_url="http://s/tasks/o/r/job1",
                token="tok",
                serge_url="http://s",
                serge_token="tok",
                timeout_seconds=300,
                poll_seconds=1,
            )
        self.assertEqual(out["pr_number"], 21)
        upd.assert_called()

    def test_noop_without_issue(self):
        out = dft.reconcile_failure_issue(
            "o/r",
            "fp",
            {},
            {},
            "g",
            issue_number=None,
            job_id=None,
            serge_task_url=None,
            token="t",
            serge_url=None,
            serge_token=None,
            timeout_seconds=300,
        )
        self.assertEqual(out, {"pr_number": None, "status": None})


if __name__ == "__main__":
    unittest.main()
