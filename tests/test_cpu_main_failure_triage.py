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

from transformersci.agentic import cpu_main_failure_triage as cmf


class PromqlTest(unittest.TestCase):
    def test_recent_failures_query_matches_dashboard_shape_and_excludes_gpu(self):
        query = cmf.recent_failures_query(limit=10, lookback="90d")
        self.assertIn("topk(10, topk by (test_job, test_nodeid)", query)
        self.assertIn('pytest_test_last_failure_info{pr="main"', query)
        self.assertIn('test_job!~".*[gG][pP][uU].*"', query)
        self.assertNotIn("pytest_test_duration_seconds", query)
        self.assertNotIn("[90d]", query)

    def test_persistence_query_pins_candidate_and_main_branch(self):
        query = cmf.persistence_query(
            {"test_job": "tests_torch", "test_nodeid": "tests/a.py::test_x"}
        )
        self.assertIn("pytest_test_last_failure_info", query)
        self.assertIn('pr="main"', query)
        self.assertIn('test_job="tests_torch"', query)
        self.assertIn('test_nodeid="tests/a.py::test_x"', query)
        self.assertNotIn("test_job!~", query)
        self.assertIn("[24h]", query)


class ParseRecentFailuresTest(unittest.TestCase):
    def test_keeps_test_identity_and_sorts_by_last_failure(self):
        parsed = cmf.parse_recent_failures(
            [
                {
                    "metric": {"test_job": "tests_a", "test_nodeid": "a::test_old"},
                    "value": [1, "10"],
                },
                {
                    "metric": {"test_job": "tests_b", "test_nodeid": "b::test_new"},
                    "value": [1, "20"],
                },
                {"metric": {"test_job": "missing"}, "value": [1, "30"]},
            ]
        )
        self.assertEqual([c["test_nodeid"] for c in parsed], ["b::test_new", "a::test_old"])

    def test_daily_failure_count_deduplicates_series_at_same_step(self):
        count = cmf.daily_failure_count(
            [
                {"values": [[100, "1"], [200, "1"], [300, "0"]]},
                {"values": [[100, "1"], [400, "1"]]},
            ]
        )
        self.assertEqual(count, 3)

    def test_has_failure_evidence_checks_instant_vector_values(self):
        self.assertTrue(cmf.has_failure_evidence([{"value": [100, "1"]}]))
        self.assertFalse(cmf.has_failure_evidence([{"value": [100, "0"]}]))


class CollectPersistentFailuresTest(unittest.TestCase):
    def test_keeps_only_failures_seen_for_every_daily_window(self):
        recent = [
            {
                "metric": {"test_job": "tests_a", "test_nodeid": "a::test"},
                "value": [1, "30"],
            },
            {
                "metric": {"test_job": "tests_b", "test_nodeid": "b::test"},
                "value": [1, "20"],
            },
        ]
        detail = [
            {
                "metric": {
                    "test_job": "tests_a",
                    "test_nodeid": "a::test",
                    "trace_id": "trace-a",
                    "exception_type": "AssertionError",
                },
                "value": [1, "30"],
            }
        ]

        with patch.object(
            cmf,
            "prom_query",
            side_effect=[
                recent,
                [{"value": [300, "1"]}],
                [{"value": [300 - 86400, "1"]}],
                [{"value": [300 - 2 * 86400, "1"]}],
                detail,
                [{"value": [300, "1"]}],
                [],
                [{"value": [300 - 2 * 86400, "1"]}],
            ],
        ):
            out = cmf.collect_persistent_failures(
                grafana_url="https://g",
                datasource_uid="prometheus",
                days=3,
                end=300,
                select=3,
            )

        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["test_nodeid"], "a::test")
        self.assertEqual(out[0]["trace_id"], "trace-a")
        self.assertEqual(out[0]["daily_windows_seen"], 3)


class PayloadTest(unittest.TestCase):
    def test_context_and_payload_use_cpu_main_marker_and_branch(self):
        candidate = {
            "test_job": "tests_torch",
            "test_nodeid": "tests/a.py::test_x",
            "daily_windows_seen": 7,
            "trace_id": "abc",
            "last_failed_ms": 1_700_000_000_000,
        }
        failure = {
            "nodeid": "tests/a.py::test_x",
            "exception_type": "AssertionError",
            "exception_message": "bad",
            "exception_stacktrace": "stack",
        }
        fp = cmf.failure_fingerprint(candidate, failure)
        context = cmf.render_context(candidate, failure, fp, "https://grafana")
        payload = cmf.build_cpu_main_payload(
            "huggingface/transformers",
            "main",
            candidate,
            failure,
            fp,
            context,
        )

        self.assertIn(cmf.fingerprint_marker(fp), context)
        self.assertEqual(payload["output"]["branch_prefix"], cmf.task_branch_prefix(fp))
        self.assertIn("CPU test", payload["output"]["title"])
        self.assertIn("last 7 daily windows", payload["context"])

    def test_context_links_tracking_issue_when_present(self):
        candidate = {
            "test_job": "tests_torch",
            "test_nodeid": "tests/a.py::test_x",
            "daily_windows_seen": 7,
            "required_daily_windows": 7,
        }
        failure = {"nodeid": "tests/a.py::test_x", "exception_type": "AssertionError"}
        fp = cmf.failure_fingerprint(candidate, failure)
        context = cmf.render_context(
            candidate, failure, fp, "https://grafana", issue_number=123
        )

        self.assertIn("Relates to #123", context)
        self.assertIn("- `tests/a.py::test_x` [cpu]", context)

    def test_tracking_issue_body_has_daily_marker_task_and_pr_columns(self):
        candidate = {
            "test_job": "tests_torch",
            "test_nodeid": "tests/a.py::test_x",
            "daily_windows_seen": 7,
            "required_daily_windows": 7,
        }
        failure = {"nodeid": "tests/a.py::test_x", "exception_type": "AssertionError"}
        fp = cmf.failure_fingerprint(candidate, failure)
        body = cmf.render_tracking_issue_body(
            [{"candidate": candidate, "failure": failure, "fingerprint": fp}],
            "2026-07-27",
            statuses={fp: "running"},
            task_urls={fp: "https://serge/tasks/1"},
            pr_numbers={fp: 456},
        )

        self.assertIn(cmf.tracking_issue_marker("2026-07-27"), body)
        self.assertIn("| Job | Test | Error | Seen | Serge task | PR |", body)
        self.assertIn("[task](https://serge/tasks/1)", body)
        self.assertIn("#456", body)


if __name__ == "__main__":
    unittest.main()
