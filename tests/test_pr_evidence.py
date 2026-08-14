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

from transformersci.agentic import pr_evidence

GRAFANA = "https://transformers-ci.lor-e.huggingface.cool"
NODE = "tests/models/gemma3/test_modeling_gemma3.py::Gemma3IntegrationTest::test_x"
OTHER = "tests/models/foo/test_modeling_foo.py::FooTest::test_a"


class GrafanaTestUrlTest(unittest.TestCase):
    def test_url_carries_the_node_id_and_the_daily_job(self):
        url = pr_evidence.grafana_test_url(GRAFANA, NODE)
        self.assertTrue(url.startswith(f"{GRAFANA}/d/pytest-test/test?"))
        self.assertIn("var-test_nodeid=tests%2Fmodels%2Fgemma3", url)
        self.assertIn("var-test_job=run_models_gpu", url)
        self.assertIn("var-pr=main", url)
        # No trace to point at unless the caller has one.
        self.assertNotIn("var-trace_id", url)

    def test_trailing_slash_does_not_double_up(self):
        self.assertIn(
            "cool/d/pytest-test", pr_evidence.grafana_test_url(f"{GRAFANA}/", NODE)
        )

    def test_a_caller_with_a_trace_id_populates_the_traceback_panel(self):
        url = pr_evidence.grafana_test_url(GRAFANA, NODE, trace_id="abc123", pr="47281")
        self.assertIn("var-trace_id=abc123", url)
        self.assertIn("var-pr=47281", url)

    def test_missing_input_yields_no_url(self):
        self.assertEqual(pr_evidence.grafana_test_url("", NODE), "")
        self.assertEqual(pr_evidence.grafana_test_url(GRAFANA, ""), "")


class TestLinksPayloadTest(unittest.TestCase):
    def test_one_entry_per_test(self):
        links = pr_evidence.test_links(GRAFANA, [NODE, OTHER])
        self.assertEqual(sorted(links), sorted([NODE, OTHER]))
        self.assertEqual(links[NODE][0]["label"], "Test dashboard")

    def test_duplicates_collapse(self):
        self.assertEqual(len(pr_evidence.test_links(GRAFANA, [NODE, NODE])), 1)

    def test_unconfigured_grafana_yields_no_links(self):
        # The dispatcher then omits the field entirely and Serge renders nothing —
        # the pre-existing behaviour.
        self.assertEqual(pr_evidence.test_links("", [NODE]), {})
        self.assertEqual(pr_evidence.test_links(GRAFANA, [""]), {})

    def test_trace_ids_are_applied_per_test(self):
        links = pr_evidence.test_links(GRAFANA, [NODE, OTHER], trace_ids={NODE: "t1"})
        self.assertIn("var-trace_id=t1", links[NODE][0]["url"])
        self.assertNotIn("var-trace_id", links[OTHER][0]["url"])


if __name__ == "__main__":
    unittest.main()
