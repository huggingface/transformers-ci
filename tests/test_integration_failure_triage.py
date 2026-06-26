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

from transformersci.agentic import integration_failure_triage as itf


def _failure(model, gpu, test, trace, mode="output_mismatch", days=6):
    return {
        "model": model,
        "gpu": gpu,
        "test": test,
        "trace": trace,
        "latest_trace": trace,
        "days_seen": days,
        "failure_mode": mode,
    }


class FailureSignatureTest(unittest.TestCase):
    def test_known_symptoms(self):
        self.assertEqual(
            itf.failure_signature("AssertionError: Tensor-likes are not close!"),
            "tensor values differ",
        )
        self.assertEqual(
            itf.failure_signature("AssertionError: Tensor-likes are not equal!"),
            "tensor values differ",
        )
        self.assertEqual(
            itf.failure_signature("AssertionError: Lists differ: [1] != [2]"),
            "list output differs",
        )

    def test_fallback_to_exception_type(self):
        self.assertEqual(itf.failure_signature("ValueError: bad thing"), "ValueError")

    def test_empty_trace(self):
        self.assertEqual(itf.failure_signature(""), "unknown")


class PickTargetsGroupingTest(unittest.TestCase):
    def _report(self, unpinned, clusters=None, flaky=None):
        return {
            "clusters": clusters or {},
            "flaky": flaky or [],
            "unpinned": unpinned,
            "totals": {"total": len(unpinned)},
        }

    def test_groups_by_model_not_one_bucket(self):
        # The old behavior lumped all `output_mismatch` failures (across many
        # unrelated models) into a single unfixable bucket. Each model must now
        # be its own coherent group.
        unpinned = [
            _failure(
                "dac",
                "single",
                "tests/models/dac/t.py::DacIntegrationTest::a",
                "Tensor-likes are not close!",
            ),
            _failure(
                "dac",
                "multi",
                "tests/models/dac/t.py::DacIntegrationTest::b",
                "Tensor-likes are not close!",
            ),
            _failure(
                "whisper",
                "single",
                "tests/models/whisper/t.py::WhisperIntegrationTest::c",
                "Lists differ: [1] != [2]",
            ),
        ]
        targets = itf.pick_targets(self._report(unpinned))

        self.assertEqual(len(targets), 2)
        self.assertTrue(all(t["kind"] == "model_failures" for t in targets))
        models = {t["model"] for t in targets}
        self.assertEqual(models, {"dac", "whisper"})
        # No cross-model group leaks more than one model's failures.
        for t in targets:
            self.assertEqual({f["model"] for f in t["failures"]}, {t["model"]})

    def test_largest_group_first(self):
        unpinned = [
            _failure(
                "solo",
                "single",
                "tests/models/solo/t.py::SoloIntegrationTest::a",
                "Tensor-likes are not close!",
            ),
            _failure(
                "big",
                "single",
                "tests/models/big/t.py::BigIntegrationTest::a",
                "Tensor-likes are not close!",
            ),
            _failure(
                "big",
                "multi",
                "tests/models/big/t.py::BigIntegrationTest::b",
                "Tensor-likes are not close!",
            ),
        ]
        targets = itf.pick_targets(self._report(unpinned))
        self.assertEqual(targets[0]["model"], "big")
        self.assertEqual(len(targets[0]["failures"]), 2)

    def test_distinct_failure_modes_stay_separate(self):
        unpinned = [
            _failure(
                "m",
                "single",
                "tests/models/m/t.py::MIntegrationTest::a",
                "Tensor-likes are not close!",
            ),
            _failure(
                "m",
                "single",
                "tests/models/m/t.py::MIntegrationTest::b",
                "CUDA out of memory",
                mode="OOM",
            ),
        ]
        targets = itf.pick_targets(self._report(unpinned))
        self.assertEqual(len(targets), 2)
        self.assertEqual(
            {t["failure_mode"] for t in targets}, {"output_mismatch", "OOM"}
        )

    def test_clusters_rank_before_model_groups(self):
        unpinned = [
            _failure(
                "m",
                "single",
                "tests/models/m/t.py::MIntegrationTest::a",
                "Tensor-likes are not close!",
            ),
        ]
        clusters = {
            "deadbeef" * 5: {
                "bad_commit": "deadbeef" * 5,
                "pr_number": 123,
                "author": "octocat",
                "failures": [
                    _failure(
                        "x",
                        "single",
                        "tests/models/x/t.py::XIntegrationTest::a",
                        "Tensor-likes are not close!",
                    ),
                ],
            }
        }
        targets = itf.pick_targets(self._report(unpinned, clusters=clusters))
        self.assertEqual(targets[0]["kind"], "cluster")
        self.assertEqual(targets[1]["kind"], "model_failures")

    def test_label_mentions_model_mode_and_signature(self):
        unpinned = [
            _failure(
                "dac",
                "single",
                "tests/models/dac/t.py::DacIntegrationTest::a",
                "Tensor-likes are not close!",
            ),
        ]
        label = itf.pick_targets(self._report(unpinned))[0]["label"]
        self.assertIn("`dac`", label)
        self.assertIn("output_mismatch", label)
        self.assertIn("tensor values differ", label)

    def test_fingerprints_differ_per_group(self):
        unpinned = [
            _failure(
                "a",
                "single",
                "tests/models/a/t.py::AIntegrationTest::a",
                "Tensor-likes are not close!",
            ),
            _failure(
                "b",
                "single",
                "tests/models/b/t.py::BIntegrationTest::a",
                "Tensor-likes are not close!",
            ),
        ]
        targets = itf.pick_targets(self._report(unpinned))
        fps = {itf.target_fingerprint(t) for t in targets}
        self.assertEqual(len(fps), 2)


class TraceExcerptTest(unittest.TestCase):
    def test_short_trace_kept_whole(self):
        trace = "line1\nAssertionError: Lists differ: ['a'] != ['b']"
        self.assertEqual(itf.trace_excerpt(trace, 2500), trace)

    def test_long_trace_keeps_tail_with_ellipsis(self):
        trace = (
            "HEAD\n"
            + ("filler\n" * 300)
            + "AssertionError: not close\nMismatched: 3/10"
        )
        ex = itf.trace_excerpt(trace, 120)
        self.assertTrue(ex.startswith("…\n"))
        self.assertIn("Mismatched: 3/10", ex)  # the meaningful tail survives
        self.assertLessEqual(len(ex), 122)

    def test_empty(self):
        self.assertEqual(itf.trace_excerpt(""), "")

    def test_context_embeds_full_trace_block(self):
        trace = (
            "boom\n"
            + ("x\n" * 200)
            + "AssertionError: Tensor-likes are not close!\nMismatched elements: 5"
        )
        target = {
            "kind": "model_failures",
            "label": "g",
            "model": "dac",
            "failure_mode": "output_mismatch",
            "cluster": None,
            "failures": [
                {
                    "model": "dac",
                    "gpu": "single",
                    "test": "t::DacIntegrationTest::a",
                    "trace": trace,
                    "latest_trace": trace,
                    "days_seen": 6,
                    "failure_mode": "output_mismatch",
                }
            ],
        }
        ctx = itf.render_serge_context(
            [target], ["2026-06-13", "2026-06-19"], trace_chars=2500
        )
        self.assertIn("```", ctx)
        self.assertIn("Mismatched elements: 5", ctx)


class MatchExistingPrTest(unittest.TestCase):
    def test_matches_by_fingerprint_marker(self):
        fp = "a" * 64
        pulls = [
            {
                "number": 5,
                "body": "stuff\n" + itf.fingerprint_marker(fp),
                "head": {"ref": "x"},
            }
        ]
        self.assertEqual(itf.match_existing_pr(pulls, fp), 5)

    def test_matches_by_branch_prefix(self):
        fp = "b" * 64
        pulls = [
            {
                "number": 9,
                "body": "",
                "head": {"ref": itf.task_branch_prefix(fp) + "-2"},
            }
        ]
        self.assertEqual(itf.match_existing_pr(pulls, fp), 9)

    def test_no_match(self):
        pulls = [{"number": 1, "body": "unrelated", "head": {"ref": "feature/x"}}]
        self.assertIsNone(itf.match_existing_pr(pulls, "c" * 64))


class DispatchTargetsTest(unittest.TestCase):
    def _targets(self):
        return [
            {
                "kind": "model_failures",
                "label": "g1",
                "model": "a",
                "failure_mode": "output_mismatch",
                "cluster": None,
                "failures": [
                    {
                        "model": "a",
                        "gpu": "single",
                        "test": "t::AIntegrationTest::a",
                        "trace": "x",
                        "latest_trace": "x",
                        "days_seen": 6,
                        "failure_mode": "output_mismatch",
                    }
                ],
            },
            {
                "kind": "model_failures",
                "label": "g2",
                "model": "b",
                "failure_mode": "OOM",
                "cluster": None,
                "failures": [
                    {
                        "model": "b",
                        "gpu": "single",
                        "test": "t::BIntegrationTest::a",
                        "trace": "y",
                        "latest_trace": "y",
                        "days_seen": 6,
                        "failure_mode": "OOM",
                    }
                ],
            },
        ]

    def test_one_task_per_group(self):
        sent = []

        def fake_dispatch(serge_url, token, payload, timeout=240):
            sent.append(payload)
            return {"id": f"job{len(sent)}", "url": f"/tasks/o/r/job{len(sent)}"}

        with (
            patch.object(itf, "list_open_pulls", return_value=[]),
            patch.object(itf, "dispatch_to_serge", side_effect=fake_dispatch),
        ):
            accepted, failed = itf.dispatch_targets(
                self._targets(),
                repo="o/r",
                base_ref="main",
                serge_url="http://s",
                token="tok",
                window=["2026-06-19"],
                timeout=10,
                github_token=None,
            )

        self.assertEqual((accepted, failed), (2, 0))
        self.assertEqual(len(sent), 2)
        # Each task is a new_pr with its own fingerprint-derived branch.
        branches = {p["output"]["branch_prefix"] for p in sent}
        self.assertEqual(len(branches), 2)
        self.assertTrue(all(p["output"]["mode"] == "new_pr" for p in sent))
        self.assertTrue(all(p["output"]["title"].startswith("[serge] ") for p in sent))

    def test_task_finished_notification_payload(self):
        sent = []

        def fake_dispatch(serge_url, token, payload, timeout=240):
            sent.append(payload)
            return {"id": "job1", "url": "/tasks/o/r/job1"}

        with (
            patch.object(itf, "list_open_pulls", return_value=[]),
            patch.object(itf, "dispatch_to_serge", side_effect=fake_dispatch),
        ):
            itf.dispatch_targets(
                self._targets()[:1],
                repo="o/r",
                base_ref="main",
                serge_url="http://s",
                token="tok",
                window=["2026-06-19"],
                timeout=10,
                github_token=None,
                slack_channel="#dynamic-ci",
                notify_task_finished=True,
            )

        self.assertEqual(
            sent[0]["notifications"],
            {
                "pr_created": True,
                "task_finished": True,
                "slack_channel": "#dynamic-ci",
            },
        )

    def test_existing_pr_becomes_followup(self):
        targets = self._targets()
        fp0 = itf.target_fingerprint(targets[0])
        pulls = [
            {"number": 42, "body": itf.fingerprint_marker(fp0), "head": {"ref": "z"}}
        ]

        sent = []

        def fake_dispatch(serge_url, token, payload, timeout=240):
            sent.append(payload)
            return {"id": "j", "url": "/tasks/o/r/j"}

        with (
            patch.object(itf, "list_open_pulls", return_value=pulls),
            patch.object(itf, "dispatch_to_serge", side_effect=fake_dispatch),
        ):
            itf.dispatch_targets(
                targets,
                repo="o/r",
                base_ref="main",
                serge_url="http://s",
                token="tok",
                window=["2026-06-19"],
                timeout=10,
                github_token=None,
            )

        self.assertEqual(
            sent[0]["output"],
            {"mode": "existing_pr", "pr_number": 42, "title": "[serge] Fix g1"},
        )
        self.assertEqual(sent[1]["output"]["mode"], "new_pr")

    def test_one_failure_does_not_abort_the_rest(self):
        calls = []

        def fake_dispatch(serge_url, token, payload, timeout=240):
            calls.append(payload)
            if len(calls) == 1:
                raise itf.SergeDispatchError("boom")
            return {"id": "j", "url": "/tasks/o/r/j"}

        with (
            patch.object(itf, "list_open_pulls", return_value=[]),
            patch.object(itf, "dispatch_to_serge", side_effect=fake_dispatch),
        ):
            accepted, failed = itf.dispatch_targets(
                self._targets(),
                repo="o/r",
                base_ref="main",
                serge_url="http://s",
                token="tok",
                window=["2026-06-19"],
                timeout=10,
                github_token=None,
            )

        self.assertEqual((accepted, failed), (1, 1))
        self.assertEqual(len(calls), 2)  # second group still attempted


class TrackingIssueTest(unittest.TestCase):
    def _target(self, label="g1", model="a"):
        return {
            "kind": "model_failures",
            "label": label,
            "model": model,
            "failure_mode": "output_mismatch",
            "cluster": None,
            "failures": [
                {
                    "model": model,
                    "gpu": "single",
                    "test": f"t::{model}IntegrationTest::a",
                    "trace": "boom",
                    "latest_trace": "boom",
                    "days_seen": 6,
                    "failure_mode": "output_mismatch",
                }
            ],
        }

    def test_marker_omitted_without_issue(self):
        ctx = itf.add_state_marker("body", "f" * 64)
        self.assertNotIn("Relates to #", ctx)

    def test_marker_includes_relates_to(self):
        ctx = itf.add_state_marker("body", "f" * 64, issue_number=77)
        self.assertIn("Relates to #77", ctx)

    def test_issue_body_renders_table(self):
        targets = [self._target("g1", "alpha"), self._target("g2", "beta")]
        body = itf.render_tracking_issue_body(
            targets, ["2026-06-13", "2026-06-19"], "2026-06-19"
        )
        self.assertIn(itf.tracking_issue_marker("2026-06-19"), body)
        self.assertIn("generated by AI-assisted automation", body)
        self.assertIn("can be incomplete or misleading", body)
        self.assertIn("| Model | Error | Occurrences | PR |", body)
        self.assertIn("| --- | --- | --- | --- |", body)
        self.assertIn("`alpha`", body)
        self.assertIn("`beta`", body)
        # pending groups show their branch in the PR column
        for t in targets:
            self.assertIn(itf.task_branch_prefix(itf.target_fingerprint(t)), body)

    def test_issue_body_links_existing_pr_inline(self):
        targets = [self._target("g1", "a"), self._target("g2", "b")]
        fp0 = itf.target_fingerprint(targets[0])
        fp1 = itf.target_fingerprint(targets[1])
        body = itf.render_tracking_issue_body(
            targets, ["2026-06-19"], "2026-06-19", existing_prs={fp0: 62, fp1: None}
        )
        self.assertIn("#62", body)  # follow-up group links its PR directly
        self.assertIn(itf.task_branch_prefix(fp1), body)  # new-PR group shows branch

    def test_resolve_existing_prs(self):
        targets = [self._target("g1", "a"), self._target("g2", "b")]
        fp0 = itf.target_fingerprint(targets[0])
        pulls = [
            {"number": 62, "body": itf.fingerprint_marker(fp0), "head": {"ref": "x"}}
        ]
        resolved = itf.resolve_existing_prs(targets, pulls)
        self.assertEqual(resolved[fp0], 62)
        self.assertIsNone(resolved[itf.target_fingerprint(targets[1])])

    def test_ensure_issue_noop_without_token(self):
        self.assertIsNone(
            itf.ensure_tracking_issue("o/r", "2026-06-19", "t", "b", None)
        )

    def test_dispatch_injects_issue_backreference(self):
        sent = []

        def fake_dispatch(serge_url, token, payload, timeout=240):
            sent.append(payload)
            return {"id": "j", "url": "/tasks/o/r/j"}

        with (
            patch.object(itf, "list_open_pulls", return_value=[]),
            patch.object(itf, "dispatch_to_serge", side_effect=fake_dispatch),
        ):
            itf.dispatch_targets(
                [self._target()],
                repo="o/r",
                base_ref="main",
                serge_url="http://s",
                token="tok",
                window=["2026-06-19"],
                timeout=10,
                github_token=None,
                issue_number=123,
            )
        self.assertIn("Relates to #123", sent[0]["context"])


if __name__ == "__main__":
    unittest.main()
