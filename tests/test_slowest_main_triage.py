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

from transformersci.agentic import slowest_main_triage as smt


class PromqlTest(unittest.TestCase):
    def test_slowest_query_uses_recording_metric(self):
        query = smt.slowest_query(limit=10)
        self.assertEqual(
            query,
            "sort_desc(topk(10, last_over_time(pytest_test:duration_avg_main:top20_7d[8d])))",
        )

    def test_persistence_query_pins_candidate(self):
        query = smt.persistence_query(
            {"test_job": "tests_torch", "test_nodeid": "tests/a.py::test_x"}
        )
        self.assertIn("pytest_test:duration_avg_main:top20_7d", query)
        self.assertIn("last_over_time(", query)
        self.assertIn("[24h]", query)
        self.assertIn('test_job="tests_torch"', query)
        self.assertIn('test_nodeid="tests/a.py::test_x"', query)


class CollectSlowestTest(unittest.TestCase):
    def test_keeps_only_tests_seen_for_each_daily_window(self):
        recent = [
            {
                "metric": {
                    "test_job": "tests_a",
                    "test_nodeid": "a::test",
                    "test_module": "a.py",
                    "test_function": "test",
                },
                "value": [300, "30"],
            },
            {
                "metric": {"test_job": "tests_b", "test_nodeid": "b::test"},
                "value": [300, "20"],
            },
        ]
        with patch.object(
            smt,
            "prom_query",
            side_effect=[
                recent,
                [{"value": [300, "30"]}],
                [{"value": [300 - 86400, "31"]}],
                [{"value": [300 - 2 * 86400, "32"]}],
                [{"value": [300, "20"]}],
                [],
                [{"value": [300 - 2 * 86400, "22"]}],
            ],
        ):
            out = smt.collect_persistent_slowest(
                grafana_url="https://g",
                datasource_uid="prometheus",
                days=3,
                end=300,
                select=3,
                nodeid_prefix="",
            )

        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["test_nodeid"], "a::test")
        self.assertEqual(out[0]["daily_windows_seen"], 3)
        self.assertEqual(out[0]["max_duration_seconds"], 32)

    def test_defaults_to_real_tests_and_can_pin_exact_nodeid(self):
        recent = [
            {
                "metric": {
                    "test_job": "check_repository_consistency",
                    "test_nodeid": "utils/checkers.py::modular_conversion",
                },
                "value": [300, "165"],
            },
            {
                "metric": {
                    "test_job": "tests_generate",
                    "test_nodeid": "tests/models/a/test_modeling_a.py::A::test_slow",
                },
                "value": [300, "100"],
            },
            {
                "metric": {
                    "test_job": "tests_generate",
                    "test_nodeid": "tests/models/b/test_modeling_b.py::B::test_slow",
                },
                "value": [300, "90"],
            },
        ]
        with patch.object(
            smt,
            "prom_query",
            side_effect=[
                recent,
                [{"value": [300, "90"]}],
            ],
        ):
            out = smt.collect_persistent_slowest(
                grafana_url="https://g",
                datasource_uid="prometheus",
                days=1,
                end=300,
                select=3,
                test_nodeid="tests/models/b/test_modeling_b.py::B::test_slow",
            )

        self.assertEqual(len(out), 1)
        self.assertEqual(
            out[0]["test_nodeid"],
            "tests/models/b/test_modeling_b.py::B::test_slow",
        )


class PayloadTest(unittest.TestCase):
    def test_context_and_payload_are_parseable_for_serge_reproduce_verify(self):
        candidate = {
            "test_job": "tests_torch",
            "test_nodeid": "tests/a.py::test_x",
            "avg_duration_seconds": 12.5,
            "max_duration_seconds": 14.0,
            "daily_windows_seen": 7,
            "required_daily_windows": 7,
        }
        fp = smt.failure_fingerprint(candidate)
        context = smt.render_context(candidate, fp, "https://grafana", issue_number=123)
        payload = smt.build_slowest_payload(
            "huggingface/transformers", "main", candidate, fp, context
        )

        self.assertIn(smt.fingerprint_marker(fp), context)
        self.assertIn("Relates to #123", context)
        self.assertIn("- `tests/a.py::test_x` [cpu]", context)
        self.assertEqual(payload["output"]["branch_prefix"], smt.task_branch_prefix(fp))
        self.assertIn("Investigate slow main test", payload["output"]["title"])
        self.assertIn("A slow test is not automatically a bug", payload["instruction"])
        self.assertIn("quadratic or worse work", payload["instruction"])
        self.assertIn("Do not treat expected framework overhead", payload["instruction"])
        self.assertIn("Do not loop over the same tradeoff", payload["instruction"])
        self.assertIn("immediately return final JSON with an empty patch", payload["instruction"])
        self.assertIn("If the local tool environment cannot execute shell commands", payload["instruction"])
        self.assertIn("A skip is a coverage deletion", context)
        self.assertIn("Stop rule", context)

    def test_tracking_issue_body_has_daily_marker_task_and_pr_columns(self):
        candidate = {
            "test_job": "tests_torch",
            "test_nodeid": "tests/a.py::test_x",
            "avg_duration_seconds": 12.5,
            "daily_windows_seen": 7,
            "required_daily_windows": 7,
        }
        fp = smt.failure_fingerprint(candidate)
        body = smt.render_tracking_issue_body(
            [{"candidate": candidate, "fingerprint": fp}],
            "2026-07-27",
            statuses={fp: "running"},
            task_urls={fp: "https://serge/tasks/1"},
            pr_numbers={fp: 456},
        )

        self.assertIn(smt.tracking_issue_marker("2026-07-27"), body)
        self.assertIn("| Job | Test | Avg duration | Seen | Serge task | PR |", body)
        self.assertIn("[task](https://serge/tasks/1)", body)
        self.assertIn("#456", body)


if __name__ == "__main__":
    unittest.main()
