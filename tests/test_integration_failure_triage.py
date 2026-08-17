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

import datetime
import os
import unittest
from unittest.mock import patch

from transformersci.agentic import integration_failure_triage as itf
from transformersci.agentic import serge_dispatch as sd


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


# Realistic pytest failure blocks: each ends in a real ``E   <Exc>: ...`` line
# and a ``path:line: <Exc>`` location line, and — crucially — the printed test
# BODY contains ``self.assertEqual(...)``. The classifier must key off the
# terminal exception, not that assert text.
_TRACE_DTYPE_CRASH = """\
    def test_large_generation(self):
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large")
        transcript = processor.batch_decode(generated_ids)
>       self.assertEqual(transcript, EXPECTED_TRANSCRIPT)

    hidden_states = nn.functional.gelu(self.conv1(input_features))
E       RuntimeError: Input type (float) and bias type (c10::Half) should be the same

src/transformers/models/whisper/modeling_whisper.py:905: RuntimeError
"""

_TRACE_UNBOUND_CRASH = """\
    def test_speculative_decoding_distil(self):
        transcription_non_ass = ...
>       self.assertEqual(transcription_ass, transcription_non_ass)

    if is_cross_attention and past_key_values and is_updated:
E       UnboundLocalError: local variable 'is_updated' referenced before assignment

src/transformers/models/whisper/modeling_whisper.py:323: UnboundLocalError
"""

_TRACE_TENSOR_MISMATCH = """\
    def test_small_token_timestamp_generation(self):
>       torch.testing.assert_close(generate_outputs["token_timestamps"].to("cpu"), EXPECTED_OUTPUT)
E       AssertionError: Tensor-likes are not close!
E
E       Mismatched elements: 56 / 164 (34.1%)

tests/models/whisper/test_modeling_whisper.py:2023: AssertionError
"""

_TRACE_LIST_MISMATCH = """\
    def test_tiny_en_generation(self):
>       self.assertListEqual(decoded_all, EXPECTED_TEXT)
E       AssertionError: Lists differ: [' Mr'] != [' Mister']

tests/models/whisper/test_modeling_whisper.py:1399: AssertionError
"""

_TRACE_KEYERROR_CRASH = """\
    def test_tiny_timestamp_generation(self):
>       self.assertEqual(decoded, expected_output)
E       KeyError: 0

tests/models/whisper/test_modeling_whisper.py:1882: KeyError
"""


class ClassifyTest(unittest.TestCase):
    def test_dtype_runtimeerror_is_not_output_mismatch(self):
        # The old classifier saw `self.assertEqual` in the body and mislabeled
        # this crash `output_mismatch`. It must be `cuda_runtime` now.
        self.assertEqual(itf.classify(_TRACE_DTYPE_CRASH), "cuda_runtime")

    def test_unbound_local_crash_is_other_not_mismatch(self):
        self.assertEqual(itf.classify(_TRACE_UNBOUND_CRASH), "other")

    def test_keyerror_crash_is_other(self):
        self.assertEqual(itf.classify(_TRACE_KEYERROR_CRASH), "other")

    def test_genuine_tensor_mismatch_is_output_mismatch(self):
        self.assertEqual(itf.classify(_TRACE_TENSOR_MISMATCH), "output_mismatch")

    def test_genuine_list_mismatch_is_output_mismatch(self):
        self.assertEqual(itf.classify(_TRACE_LIST_MISMATCH), "output_mismatch")

    def test_oom_still_wins(self):
        self.assertEqual(
            itf.classify("E   torch.OutOfMemoryError: CUDA out of memory"), "OOM"
        )

    def test_bare_symptom_without_exception_type_falls_back(self):
        # No `E  <Exc>:` line at all — legacy symptom fallback still applies.
        self.assertEqual(itf.classify("Tensor-likes are not close!"), "output_mismatch")
        self.assertEqual(itf.classify("nothing useful here"), "other")


class TerminalExceptionTest(unittest.TestCase):
    def test_extracts_raised_exception_over_body_asserts(self):
        etype, msg = itf.terminal_exception(_TRACE_DTYPE_CRASH)
        self.assertEqual(etype, "RuntimeError")
        self.assertIn("bias type", msg)

    def test_assertion_error(self):
        etype, _ = itf.terminal_exception(_TRACE_TENSOR_MISMATCH)
        self.assertEqual(etype, "AssertionError")

    def test_none_when_no_exception(self):
        self.assertEqual(itf.terminal_exception("just a plain line"), (None, ""))

    def test_crash_site_is_raising_frame(self):
        self.assertEqual(
            itf.crash_site(_TRACE_UNBOUND_CRASH),
            "src/transformers/models/whisper/modeling_whisper.py:323",
        )


class GroupingSplitsHeterogeneousFailuresTest(unittest.TestCase):
    def _filtered(self, test, trace):
        return {
            "model": "whisper",
            "gpu": "single",
            "test": f"tests/models/whisper/test_modeling_whisper.py::WhisperModelIntegrationTests::{test}",
            "trace": trace,
            "latest_trace": trace,
            "days_seen": 6,
        }

    def test_one_whisper_bucket_splits_by_root_cause(self):
        # Four whisper failures that the old pipeline lumped into ONE
        # `output_mismatch` group (unfixable by a single PR) must now split into
        # coherent per-root-cause groups: dtype crash, is_updated crash, and the
        # true mismatches (tensor + list) grouped together.
        filtered = [
            self._filtered("test_large_generation", _TRACE_DTYPE_CRASH),
            self._filtered("test_speculative_decoding_distil", _TRACE_UNBOUND_CRASH),
            self._filtered(
                "test_small_token_timestamp_generation", _TRACE_TENSOR_MISMATCH
            ),
            self._filtered("test_tiny_en_generation", _TRACE_LIST_MISMATCH),
        ]
        report = itf.cluster_failures(filtered, None)
        targets = itf.pick_targets(report)

        modes = sorted(t["failure_mode"] for t in targets)
        self.assertEqual(modes, ["cuda_runtime", "other", "output_mismatch"])
        # The two genuine mismatches share one "refresh expected values" group.
        mismatch = next(t for t in targets if t["failure_mode"] == "output_mismatch")
        self.assertEqual(len(mismatch["failures"]), 2)
        # The crash groups name their raised exception in the label.
        crash = next(t for t in targets if t["failure_mode"] == "cuda_runtime")
        self.assertIn("RuntimeError", crash["label"])
        # Every dispatched group is a single coherent unit (no mixed exceptions).
        for t in targets:
            excs = {f["terminal_exc"] for f in t["failures"]}
            self.assertEqual(len(excs), 1, f"group {t['label']} mixes {excs}")


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
            accepted, failed, _job_ids = itf.dispatch_targets(
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
            accepted, failed, _job_ids = itf.dispatch_targets(
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

    def test_bounded_dispatch_limits_active_serge_tasks(self):
        targets = self._targets() + [
            {
                "kind": "model_failures",
                "label": "g3",
                "model": "c",
                "failure_mode": "output_mismatch",
                "cluster": None,
                "failures": [
                    {
                        "model": "c",
                        "gpu": "single",
                        "test": "t::CIntegrationTest::a",
                        "trace": "z",
                        "latest_trace": "z",
                        "days_seen": 6,
                        "failure_mode": "output_mismatch",
                    }
                ],
            }
        ]
        live = set()
        max_live = 0
        calls = []

        def fake_dispatch(serge_url, token, payload, timeout=240):
            nonlocal max_live
            job_id = f"job{len(calls) + 1}"
            calls.append(payload)
            live.add(job_id)
            max_live = max(max_live, len(live))
            return {"id": job_id, "url": f"/tasks/o/r/{job_id}"}

        def fake_poll(serge_url, token, repo, job_id):
            live.discard(job_id)
            return {"status": "published", "error": None}

        with (
            patch.object(itf, "list_open_pulls", return_value=[]),
            patch.object(itf, "dispatch_to_serge", side_effect=fake_dispatch),
            patch.object(itf, "poll_serge_task", side_effect=fake_poll),
            patch.object(itf, "mint_serge_oidc_token", return_value=None),
            patch.object(itf.time, "sleep", lambda _s: None),
        ):
            accepted, failed, _job_ids = itf.dispatch_targets(
                targets,
                repo="o/r",
                base_ref="main",
                serge_url="http://s",
                token="tok",
                window=["2026-06-19"],
                timeout=10,
                github_token=None,
                serge_concurrency=2,
                retry_attempts=0,
                poll_seconds=0,
            )

        self.assertEqual((accepted, failed), (3, 0))
        self.assertEqual(len(calls), 3)
        self.assertLessEqual(max_live, 2)

    def test_bounded_dispatch_retries_terminal_rate_limit_error(self):
        target = self._targets()[:1]
        jobs = []

        def fake_dispatch(serge_url, token, payload, timeout=240):
            job_id = f"job{len(jobs) + 1}"
            jobs.append(job_id)
            return {"id": job_id, "url": f"/tasks/o/r/{job_id}"}

        def fake_poll(serge_url, token, repo, job_id):
            if job_id == "job1":
                return {
                    "status": "error",
                    "error": "LLM endpoint returned 429 Too Many Requests: rate limit exceeded",
                }
            return {"status": "published", "error": None}

        with (
            patch.object(itf, "list_open_pulls", return_value=[]),
            patch.object(itf, "dispatch_to_serge", side_effect=fake_dispatch),
            patch.object(itf, "poll_serge_task", side_effect=fake_poll),
            patch.object(itf, "mint_serge_oidc_token", return_value=None),
            patch.object(itf.random, "uniform", return_value=0),
            patch.object(itf.time, "sleep", lambda _s: None),
        ):
            accepted, failed, job_ids = itf.dispatch_targets(
                target,
                repo="o/r",
                base_ref="main",
                serge_url="http://s",
                token="tok",
                window=["2026-06-19"],
                timeout=10,
                github_token=None,
                serge_concurrency=1,
                retry_attempts=1,
                retry_base_seconds=1,
                poll_seconds=0,
            )

        self.assertEqual((accepted, failed), (1, 0))
        self.assertEqual(jobs, ["job1", "job2"])
        self.assertEqual(job_ids[itf.target_fingerprint(target[0])], "job2")


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

    def test_carry_forward_rows_keeps_resolved_drops_pending(self):
        # A prior same-day run: convnextv2 got a PR, oldpending is still pending.
        prior = [self._target("g1", "convnextv2"), self._target("g2", "oldpending")]
        prior_body = itf.render_tracking_issue_body(
            prior,
            ["2026-07-25"],
            "2026-07-25",
            existing_prs={itf.target_fingerprint(prior[0]): 47540},
        )
        # This run picked a different (shuffled) group.
        carried = itf._carry_forward_rows(prior_body, [self._target("g3", "helium")])
        joined = "\n".join(carried)
        self.assertIn("#47540", joined)  # PR'd group kept
        self.assertIn("`convnextv2`", joined)
        self.assertNotIn("oldpending", joined)  # still-(pending) group dropped

    def test_carry_forward_skips_group_in_current_run(self):
        prior = [self._target("g1", "convnextv2")]
        prior_body = itf.render_tracking_issue_body(
            prior,
            ["2026-07-25"],
            "2026-07-25",
            existing_prs={itf.target_fingerprint(prior[0]): 47540},
        )
        # convnextv2 is in THIS run too -> its live row wins, don't duplicate it.
        self.assertEqual(
            itf._carry_forward_rows(prior_body, [self._target("g1", "convnextv2")]), []
        )

    def test_render_appends_carry_rows_to_table(self):
        body = itf.render_tracking_issue_body(
            [self._target("g1", "helium")],
            ["2026-07-25"],
            "2026-07-25",
            carry_rows=["| `convnextv2` | output_mismatch — x | 2 | #47540 |"],
        )
        self.assertIn("`helium`", body)
        self.assertIn("`convnextv2`", body)
        self.assertIn("#47540", body)

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
        # Serge is told the tracking issue so it can comment no_fix/error
        # outcomes there (groups that open no PR).
        self.assertEqual(sent[0]["tracking_issue"], 123)

    def test_reconcile_refreshes_issue_when_pr_appears(self):
        targets = [self._target("g1", "a"), self._target("g2", "b")]
        fp0 = itf.target_fingerprint(targets[0])
        fp1 = itf.target_fingerprint(targets[1])
        both = [
            {"number": 81, "body": itf.fingerprint_marker(fp0), "head": {"ref": "x"}},
            {"number": 82, "body": itf.fingerprint_marker(fp1), "head": {"ref": "y"}},
        ]
        calls = {"n": 0}

        def fake_pulls(repo, github_token):
            # First poll: no PRs yet. Subsequent polls: both PRs have appeared,
            # so the loop terminates via linked >= total.
            calls["n"] += 1
            return [] if calls["n"] == 1 else both

        patched = []

        def fake_update(repo, issue_number, body, github_token):
            patched.append(body)
            return True

        with (
            patch.object(itf, "list_open_pulls", side_effect=fake_pulls),
            patch.object(itf, "update_issue_body", side_effect=fake_update),
            patch.object(itf.time, "sleep", lambda _s: None),
        ):
            resolved = itf.reconcile_tracking_issue(
                targets,
                repo="o/r",
                window=["2026-06-19"],
                run_key="2026-06-19",
                issue_number=42,
                github_token="tok",
                timeout_seconds=300,
                poll_seconds=1,
            )
        self.assertEqual(resolved[fp0], 81)
        self.assertEqual(resolved[fp1], 82)
        # Re-rendered twice: the empty first poll, then once the PRs showed up.
        self.assertEqual(len(patched), 2)
        self.assertIn("#81", patched[-1])
        self.assertIn("#82", patched[-1])

    def test_dispatch_returns_job_ids(self):
        def fake_dispatch(serge_url, token, payload, timeout=240):
            return {"id": "job-123", "url": "/tasks/o/r/job-123"}

        t = self._target()
        with (
            patch.object(itf, "list_open_pulls", return_value=[]),
            patch.object(itf, "dispatch_to_serge", side_effect=fake_dispatch),
        ):
            accepted, failed, job_ids = itf.dispatch_targets(
                [t],
                repo="o/r",
                base_ref="main",
                serge_url="http://s",
                token="tok",
                window=["2026-06-19"],
                timeout=10,
                github_token=None,
            )
        self.assertEqual(accepted, 1)
        self.assertEqual(job_ids[itf.target_fingerprint(t)], "job-123")

    def test_render_shows_serge_statuses(self):
        targets = [self._target("g1", "a"), self._target("g2", "b")]
        fp0 = itf.target_fingerprint(targets[0])
        fp1 = itf.target_fingerprint(targets[1])
        body = itf.render_tracking_issue_body(
            targets,
            ["2026-06-19"],
            "2026-06-19",
            existing_prs={},
            statuses={fp0: "no_fix", fp1: "error"},
        )
        self.assertIn("no fix", body)
        self.assertIn("task failed", body)

    def test_render_outcome_recap_reason_and_spend(self):
        targets = [self._target("g1", "gemma"), self._target("g2", "whisper")]
        fp0 = itf.target_fingerprint(targets[0])
        fp1 = itf.target_fingerprint(targets[1])
        details = {
            fp0: {
                "reason": "[not_reproduced] targeted tests passed at base",
                "model": "moonshotai/Kimi-K2.7-Code",
                "prompt_tokens": 12345,
                "completion_tokens": 678,
            },
            fp1: {
                "reason": "no safe change found",
                "model": "moonshotai/Kimi-K2.7-Code",
                "prompt_tokens": 2041248,
                "completion_tokens": 7310,
            },
        }
        body = itf.render_tracking_issue_body(
            targets,
            ["2026-06-19"],
            "2026-06-19",
            existing_prs={},
            statuses={fp0: "no_fix", fp1: "no_fix"},
            details=details,
        )
        self.assertIn("## Outcome recap", body)
        self.assertIn("not_reproduced", body)
        self.assertIn("no safe change found", body)
        self.assertIn("2,041,248", body)  # spend rendered with thousands sep
        self.assertIn("Kimi-K2.7-Code", body)

    def test_recap_skips_groups_with_a_pr(self):
        targets = [self._target("g1", "gemma")]
        fp0 = itf.target_fingerprint(targets[0])
        body = itf.render_tracking_issue_body(
            targets,
            ["2026-06-19"],
            "2026-06-19",
            existing_prs={fp0: 123},  # a PR is the outcome → no recap row
            statuses={},
            details={fp0: {"reason": "x", "prompt_tokens": 1, "completion_tokens": 2}},
        )
        self.assertNotIn("## Outcome recap", body)

    def test_distill_outcome_strips_marker_and_uses_verdict(self):
        detail = {
            "status": "no_fix",
            "result": {
                "message": "<!-- serge-task:foo:sha256:abc -->\nRelates to #47515\n"
                "The 14 GemmaIntegrationTest cases are all output_mismatch.",
                "verify_verdict": "not_reproduced",
            },
            "model": "kimi",
            "prompt_tokens": 100,
            "completion_tokens": 20,
        }
        out = itf._distill_outcome(detail)
        self.assertNotIn("serge-task", out["reason"])
        self.assertNotIn("Relates to", out["reason"])
        self.assertTrue(out["reason"].startswith("[not_reproduced]"))
        self.assertEqual(out["prompt_tokens"], 100)

    def test_distill_outcome_error_uses_error_text(self):
        out = itf._distill_outcome({"status": "error", "error": "boom", "result": None})
        self.assertEqual(out["reason"], "boom")
        self.assertIsNone(out["normalizer_error"])

    def test_distill_outcome_surfaces_failing_checker_on_error_path(self):
        # The 2026-07-29 longcat_flash shape: the terminal error names only the
        # last symptom, while the normalizer failure is what doomed the run.
        detail = {
            "status": "error",
            "error": "LLM returned unparseable output (finish_reason=stop, 62 LLM turns)",
            "result": None,
            "normalizer_error": (
                "Normalizer failed (exit 1) for `bash -lc make style`:\n"
                "Docstring formatting\n"
                "ModuleNotFoundError: No module named "
                "'transformers.models.granitemoe_swa'\n"
                "FAILED (12.46s)\n\n1 failed: docstrings"
            ),
        }
        out = itf._distill_outcome(detail)
        self.assertIn("unparseable output", out["reason"])
        self.assertIn("normalizer: 1 failed: docstrings", out["reason"])
        self.assertIn("granitemoe_swa", out["normalizer_error"])

    def test_distill_outcome_normalizer_falls_back_to_exit_line(self):
        # No checker summary (make style itself died) — still name the failure.
        out = itf._distill_outcome(
            {
                "status": "no_fix",
                "result": {"message": "no PR opened"},
                "normalizer_error": "Normalizer failed (exit 137) for `make style`:\nKilled",
            }
        )
        self.assertIn("normalizer: Normalizer failed (exit 137)", out["reason"])

    def test_recap_renders_normalizer_output_block(self):
        targets = [self._target("g1", "longcat_flash")]
        fp0 = itf.target_fingerprint(targets[0])
        body = itf.render_tracking_issue_body(
            targets,
            ["2026-07-29"],
            "2026-07-29",
            existing_prs={},
            statuses={fp0: "error"},
            details={
                fp0: {
                    "reason": "unparseable — normalizer: 1 failed: docstrings",
                    "normalizer_error": "FAILED\nNo module named 'x.granitemoe_swa'\n1 failed: docstrings",
                    "model": "kimi",
                    "prompt_tokens": 1,
                    "completion_tokens": 2,
                }
            },
        )
        self.assertIn("normalizer output (tail)", body)
        self.assertIn("granitemoe_swa", body)
        # The collapsible must sit after the recap table, not inside it.
        self.assertLess(body.index("| Group | Reason |"), body.index("<details>"))

    def test_recap_normalizer_block_fence_survives_backticks(self):
        output = "checker said:\n```\nboom\n```\n1 failed: docstrings"
        lines = itf._render_normalizer_block("`m`", output)
        block = "\n".join(lines)
        # Our fence must be longer than any run inside, so it can't close early.
        self.assertIn("````", block)
        self.assertIn("boom", block)

    def test_recap_normalizer_block_empty_when_absent(self):
        self.assertEqual(itf._render_normalizer_block("`m`", None), [])
        self.assertEqual(itf._render_normalizer_block("`m`", "   "), [])

    def test_recap_normalizer_block_keeps_the_tail(self):
        output = (
            "HEAD-MARKER\n"
            + ("x" * itf._NORMALIZER_DETAIL_CHARS)
            + "\n1 failed: docstrings"
        )
        block = "\n".join(itf._render_normalizer_block("`m`", output))
        self.assertIn("1 failed: docstrings", block)  # tail survives
        self.assertNotIn("HEAD-MARKER", block)
        self.assertIn("omitted", block)

    def test_poll_serge_status_parses_status(self):
        class _Resp:
            def __init__(self, data):
                self._d = data

            def read(self):
                return self._d

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        # poll_serge_status now lives in the shared serge_dispatch module, so the
        # network is stubbed there rather than on itf's urllib.
        with patch.object(
            sd.urllib.request, "urlopen", return_value=_Resp(b'{"status": "no_fix"}')
        ):
            st = sd.poll_serge_status("http://s", "tok", "o/r", "j1")
        self.assertEqual(st, "no_fix")

    def test_poll_serge_task_returns_error_detail(self):
        class _Resp:
            def __init__(self, data):
                self._d = data

            def read(self):
                return self._d

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        with patch.object(
            sd.urllib.request,
            "urlopen",
            return_value=_Resp(
                b'{"status": "error", "error": "429 Too Many Requests"}'
            ),
        ):
            detail = itf.poll_serge_task("http://s", "tok", "o/r", "j1")
        self.assertEqual(detail["status"], "error")
        self.assertIn("429", detail["error"])

    def test_poll_serge_status_swallows_errors(self):
        def boom(*a, **k):
            raise sd.urllib.error.URLError("down")

        with patch.object(sd.urllib.request, "urlopen", side_effect=boom):
            self.assertIsNone(sd.poll_serge_status("http://s", "tok", "o/r", "j1"))

    def test_reconcile_marks_no_fix_from_serge_status(self):
        # A group that opens no PR but ends no_fix must show on the issue —
        # reconcile polls Serge's status and renders it instead of "(pending)".
        targets = [self._target("g1", "a")]
        fp = itf.target_fingerprint(targets[0])
        patched = []

        def fake_update(repo, issue_number, body, github_token):
            patched.append(body)
            return True

        with (
            patch.object(itf, "list_open_pulls", return_value=[]),
            patch.object(itf, "update_issue_body", side_effect=fake_update),
            patch.object(itf, "mint_serge_oidc_token", return_value=None),
            patch.object(itf, "poll_serge_task", return_value={"status": "no_fix"}),
            patch.object(itf.time, "sleep", lambda _s: None),
        ):
            itf.reconcile_tracking_issue(
                targets,
                repo="o/r",
                window=["2026-06-19"],
                run_key="2026-06-19",
                issue_number=42,
                github_token="tok",
                job_ids={fp: "job-1"},
                serge_url="http://s",
                serge_token="tok",
                timeout_seconds=300,
                poll_seconds=1,
            )
        self.assertTrue(patched)
        self.assertIn("no fix", patched[-1])

    def test_reconcile_noop_without_issue_or_timeout(self):
        targets = [self._target()]
        with patch.object(itf, "list_open_pulls") as lop:
            self.assertEqual(
                itf.reconcile_tracking_issue(
                    targets,
                    repo="o/r",
                    window=["d"],
                    run_key="d",
                    issue_number=None,
                    github_token="tok",
                    timeout_seconds=300,
                ),
                {},
            )
            self.assertEqual(
                itf.reconcile_tracking_issue(
                    targets,
                    repo="o/r",
                    window=["d"],
                    run_key="d",
                    issue_number=42,
                    github_token="tok",
                    timeout_seconds=0,
                ),
                {},
            )
        lop.assert_not_called()


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class TestTrackingIssueLifecycle(unittest.TestCase):
    _MARK = itf._tracking_issue_marker_prefix()

    def test_find_prior_matches_marker_excludes_self_and_prs(self):
        page1 = [
            {"number": 10, "body": self._MARK + "2026-06-18 -->"},
            {"number": 11, "body": self._MARK + "2026-06-19 -->"},  # today (excluded)
            {"number": 12, "body": "unrelated issue"},
            {
                "number": 13,
                "body": self._MARK + "x -->",
                "pull_request": {"url": "https://api.github.com/…/pulls/13"},  # a PR
            },
        ]
        with patch.object(
            itf.urllib.request,
            "urlopen",
            side_effect=[_Resp(itf.json.dumps(page1).encode()), _Resp(b"[]")],
        ):
            found = itf.find_prior_tracking_issues("o/r", "tok", exclude=11)
        self.assertEqual(found, [10])

    def test_find_prior_noop_without_token(self):
        self.assertEqual(itf.find_prior_tracking_issues("o/r", None), [])

    def test_close_superseded_comments_then_closes(self):
        calls = []

        def fake_api(repo, num, tok, *, method, payload):
            calls.append((num, method, payload))
            return True

        with (
            patch.object(itf, "find_prior_tracking_issues", return_value=[10, 20]),
            patch.object(itf, "_issue_api", side_effect=fake_api),
        ):
            closed = itf.close_superseded_tracking_issues("o/r", 30, "tok")
        self.assertEqual(closed, [10, 20])
        # each prior issue: one POST comment naming the superseding issue, then a PATCH close
        self.assertEqual([c[1] for c in calls], ["POST", "PATCH", "POST", "PATCH"])
        self.assertIn("#30", calls[0][2]["body"])
        self.assertEqual(
            calls[1][2], {"state": "closed", "state_reason": "not_planned"}
        )

    def test_close_superseded_noop_without_issue(self):
        with patch.object(itf, "find_prior_tracking_issues") as fp:
            self.assertEqual(
                itf.close_superseded_tracking_issues("o/r", None, "tok"), []
            )
            fp.assert_not_called()

    def test_assign_sets_assignees_and_labels(self):
        seen = {}

        def fake_api(repo, num, tok, *, method, payload):
            seen["method"], seen["payload"] = method, payload
            return True

        with patch.object(itf, "_issue_api", side_effect=fake_api):
            itf.assign_tracking_issue(
                "o/r", 42, "tok", assignees=["alice"], labels=["ci-triage"]
            )
        self.assertEqual(seen["method"], "PATCH")
        self.assertEqual(
            seen["payload"], {"assignees": ["alice"], "labels": ["ci-triage"]}
        )

    def test_assign_noop_when_nothing_to_set(self):
        with patch.object(itf, "_issue_api") as api:
            itf.assign_tracking_issue("o/r", 42, "tok", assignees=[], labels=[])
            api.assert_not_called()


class SelectDispatchTargetsTest(unittest.TestCase):
    def _targets(self, n):
        return [{"id": i} for i in range(n)]

    def test_no_cap_returns_all(self):
        t = self._targets(5)
        self.assertIs(itf.select_dispatch_targets(t, 0, shuffle=True), t)

    def test_under_cap_returns_all(self):
        t = self._targets(3)
        self.assertEqual(itf.select_dispatch_targets(t, 5, shuffle=True), t)

    def test_no_shuffle_takes_top_n(self):
        t = self._targets(10)
        out = itf.select_dispatch_targets(t, 3, shuffle=False)
        self.assertEqual([x["id"] for x in out], [0, 1, 2])

    def test_shuffle_samples_and_preserves_priority_order(self):
        import random

        t = self._targets(10)
        out = itf.select_dispatch_targets(t, 3, shuffle=True, rng=random.Random(42))
        ids = [x["id"] for x in out]
        self.assertEqual(len(ids), 3)
        self.assertEqual(len(set(ids)), 3)  # no dupes
        self.assertEqual(ids, sorted(ids))  # returned in original priority order

    def _modal(self, spec):
        """`spec` = list of (category-shaping mode, terminal_exc) in priority order."""
        return [
            {"id": i, "kind": "model_failures", "failure_mode": m, "terminal_exc": e}
            for i, (m, e) in enumerate(spec)
        ]

    def test_mix_spends_the_cap_on_distinct_categories(self):
        # A realistic pool: mismatches dominate, with one load error and one
        # dependency error buried below them. Priority order alone (and a
        # uniform sample) would spend all 3 slots on mismatches.
        t = self._modal(
            [("output_mismatch", "AssertionError")] * 6
            + [("load_error", "OSError"), ("import_or_config", "AttributeError")]
        )
        out = itf.select_dispatch_targets(t, 3, shuffle=False)
        cats = [itf.dispatch_category(x) for x in out]
        self.assertEqual(len(out), 3)
        self.assertEqual(
            sorted(cats), ["import_or_config", "load_error", "output_mismatch"]
        )
        self.assertEqual([x["id"] for x in out], sorted(x["id"] for x in out))

    def test_mix_falls_back_to_priority_within_a_single_category(self):
        t = self._modal([("output_mismatch", "AssertionError")] * 5)
        out = itf.select_dispatch_targets(t, 3, shuffle=False)
        self.assertEqual([x["id"] for x in out], [0, 1, 2])

    def test_mix_can_be_disabled(self):
        t = self._modal(
            [("output_mismatch", "AssertionError")] * 3 + [("load_error", "OSError")]
        )
        out = itf.select_dispatch_targets(t, 3, shuffle=False, mix_categories=False)
        self.assertEqual([x["id"] for x in out], [0, 1, 2])  # top-N, all mismatch

    def test_assertion_wearing_a_crash_label_counts_as_a_mismatch(self):
        # `other`/`cuda_runtime` + AssertionError already gets _MISMATCH_GUIDANCE;
        # counting it as a crash is what made "3 modes" mean 3 expectation updates.
        self.assertEqual(
            itf.dispatch_category(_target("other", exc="AssertionError")),
            "output_mismatch",
        )
        self.assertEqual(
            itf.dispatch_category(_target("cuda_runtime", exc="RuntimeError")),
            "cuda_runtime",
        )
        self.assertEqual(
            itf.dispatch_category(_target("OOM", kind="cluster")), "cluster"
        )

    def test_mix_still_varies_which_member_of_a_category_runs(self):
        import random

        t = self._modal([("output_mismatch", "AssertionError")] * 10)
        a = [
            x["id"]
            for x in itf.select_dispatch_targets(
                t, 3, shuffle=True, rng=random.Random(1)
            )
        ]
        b = [
            x["id"]
            for x in itf.select_dispatch_targets(
                t, 3, shuffle=True, rng=random.Random(2)
            )
        ]
        self.assertNotEqual(a, b)

    def test_shuffle_varies_across_seeds(self):
        import random

        t = self._targets(20)
        a = [
            x["id"]
            for x in itf.select_dispatch_targets(
                t, 3, shuffle=True, rng=random.Random(1)
            )
        ]
        b = [
            x["id"]
            for x in itf.select_dispatch_targets(
                t, 3, shuffle=True, rng=random.Random(2)
            )
        ]
        # Different seeds should (very probably) pick a different set — proves we
        # are not always returning the top-3.
        self.assertNotEqual(a, b)


def _target(mode, exc="AssertionError", site="", n=1, model="dac", kind=None):
    """A minimal ``pick_targets``-shaped descriptor for the per-category tests."""
    return {
        "kind": kind or "model_failures",
        "label": f"{n} test(s) for `{model}` failing with `{mode}`",
        "failures": [
            _failure(
                model,
                "single",
                f"tests/models/{model}/t.py::T::t{i}",
                "AssertionError: Tensor-likes are not close!",
                mode=mode,
            )
            for i in range(n)
        ],
        "cluster": None,
        "model": model,
        "failure_mode": mode,
        "terminal_exc": exc,
        "crash_site": site,
    }


class InstructionAddendumTest(unittest.TestCase):
    """The failure mode is known before any GPU minute or token is spent; the
    agent's marching orders must reflect it rather than being one constant."""

    def test_mismatch_gets_expectation_guidance(self):
        text = itf.build_instruction(_target("output_mismatch"))
        self.assertIn(itf._INSTRUCTION, text)  # trunk preserved
        self.assertIn("output_mismatch", text)
        # The sibling-model heuristic belongs to assertion failures.
        self.assertIn("sibling architectures", text)
        self.assertIn("prefer correcting the test", text)

    def test_mismatch_forbids_inventing_a_tolerance_but_allows_precedent(self):
        text = itf.build_instruction(_target("output_mismatch"))
        self.assertIn("Do not invent a tolerance", text)
        # A value that already exists in a comparable test is legitimate, cited.
        self.assertIn("ALREADY appears in a comparable test", text)
        self.assertIn("name that precedent", text)
        # …but the escape hatch must not swallow a real regression.
        self.assertIn("far larger than such a precedent would cover", text)

    def test_mismatch_requires_a_plausible_expectation_not_the_observed_output(self):
        # transformers PR #47938 recorded a degenerate A10 completion
        # ("1. The image is a 1.你好!") as the expectation and went green.
        text = itf.build_instruction(_target("output_mismatch"))
        self.assertIn("PLAUSIBLE VARIANT", text)
        self.assertIn("not simply whatever the run produced", text)
        # The two tells from that PR: the output stopped describing the image,
        # and the divergence was excused as a hardware quirk.
        self.assertIn("stops describing the actual input image", text)
        self.assertIn("blaming the hardware", text)
        # A device-keyed entry stays legitimate for small backend divergence.
        self.assertIn("known SMALL divergence between backends", text)

    def test_retention_oom_gets_the_teardown_fix_not_a_shrug(self):
        text = itf.instruction_addendum(_oom_target(_OOM_RETENTION_TRACE))
        self.assertIn("RETAINED-MEMORY", text)
        self.assertIn("cleanup(torch_device, gc_collect=True)", text)
        self.assertIn("tearDown", text)
        # The coverage guard must survive the rewrite.
        self.assertIn("no shrinking the model", text)
        # …and it must not tell the agent this is probably unfixable.
        self.assertNotIn("very often NOT fixable", text)
        # Both non-fixable labels are named, so an ambiguous test is not treated
        # as a teardown bug just because a sibling in the group was one.
        self.assertIn("over capacity", text)
        self.assertIn("unclear", text)

    def test_capacity_oom_keeps_the_conservative_guidance(self):
        text = itf.instruction_addendum(_oom_target(_OOM_CAPACITY_TRACE))
        self.assertIn("very often NOT fixable", text)
        self.assertNotIn("RETAINED-MEMORY", text)

    def test_plausibility_gate_is_absent_from_crash_guidance(self):
        text = itf.build_instruction(_target("cuda_runtime", exc="RuntimeError"))
        self.assertNotIn("PLAUSIBLE VARIANT", text)

    def test_crash_forbids_test_side_edits_and_names_the_site(self):
        text = itf.build_instruction(
            _target("cuda_runtime", exc="RuntimeError", site="src/transformers/x.py:42")
        )
        self.assertIn("src/transformers/x.py:42", text)
        self.assertIn("library/model bug until", text)
        self.assertIn("Do NOT resolve this by editing the test", text)
        # The mismatch-only advice must NOT leak into a crash group.
        self.assertNotIn("prefer correcting the test", text)

    def test_assertion_reported_as_other_still_gets_mismatch_guidance(self):
        # `classify` labels an AssertionError-with-no-marker group `other`; it is
        # still an assertion, so it must not be told "this is a library bug".
        text = itf.build_instruction(_target("other", exc="AssertionError"))
        self.assertIn("Do not invent a tolerance", text)
        self.assertNotIn("library/model bug until", text)

    def test_load_and_import_modes_get_their_own_blocks(self):
        load = itf.build_instruction(_target("load_error", exc="OSError"))
        self.assertIn("gated on the Hub", load)
        imp = itf.build_instruction(_target("import_or_config", exc="AttributeError"))
        self.assertIn("version pin or a dependency bump", imp)

    def test_cluster_and_unknown_get_the_trunk_alone(self):
        # A bad-commit cluster already carries a much stronger signal.
        self.assertEqual(
            itf.build_instruction(_target("output_mismatch", kind="cluster")),
            itf._INSTRUCTION,
        )
        self.assertEqual(itf.build_instruction(None), itf._INSTRUCTION)
        self.assertEqual(itf.build_instruction(_target("")), itf._INSTRUCTION)

    def test_payload_carries_the_per_category_instruction(self):
        payload = itf.build_task_payload(
            "huggingface/transformers",
            "main",
            "ctx",
            "title",
            fingerprint="f" * 64,
            target=_target("cuda_runtime", exc="RuntimeError"),
        )
        self.assertIn("Do NOT resolve this by editing the test", payload["instruction"])
        # Omitting the target keeps today's behaviour exactly.
        bare = itf.build_task_payload(
            "huggingface/transformers", "main", "ctx", "title", fingerprint="f" * 64
        )
        self.assertEqual(bare["instruction"], itf._INSTRUCTION)


class PayloadTestLinksTest(unittest.TestCase):
    """Serge holds no Grafana config: the per-test dashboard links in its PR body
    are built HERE, where the dashboard UID and its variables are defined."""

    def test_payload_links_every_failing_test_in_the_group(self):
        target = _target("output_mismatch", n=2, model="gemma3")
        payload = itf.build_task_payload(
            "huggingface/transformers",
            "main",
            "ctx",
            "title",
            fingerprint="f" * 64,
            target=target,
            grafana_url="https://grafana.example/",
        )
        links = payload["test_links"]
        self.assertEqual(sorted(links), sorted(f["test"] for f in target["failures"]))
        url = links["tests/models/gemma3/t.py::T::t0"][0]["url"]
        self.assertTrue(url.startswith("https://grafana.example/d/pytest-test/test?"))
        self.assertIn("var-test_nodeid=tests%2Fmodels%2Fgemma3", url)
        self.assertIn("var-test_job=run_models_gpu", url)
        self.assertIn("var-pr=main", url)

    def test_no_grafana_url_omits_the_field(self):
        # An unconfigured deployment dispatches exactly as before: no field, and
        # Serge renders no link section.
        payload = itf.build_task_payload(
            "huggingface/transformers",
            "main",
            "ctx",
            "title",
            fingerprint="f" * 64,
            target=_target("output_mismatch"),
            grafana_url="",
        )
        self.assertNotIn("test_links", payload)

    def test_grafana_url_falls_back_to_the_environment(self):
        with patch.dict(os.environ, {"ITF_GRAFANA_URL": "https://env.example"}):
            payload = itf.build_task_payload(
                "huggingface/transformers",
                "main",
                "ctx",
                "title",
                fingerprint="f" * 64,
                target=_target("output_mismatch"),
            )
        self.assertIn("https://env.example/d/", str(payload["test_links"]))


# Real CUDA OOM messages from the 2026-08-14 daily run (see
# docs/plans/serge-oom-retention.md). The numbers are what make these two the
# same label and different bugs.
_OOM_RETENTION_TRACE = (
    "(line 134)  torch.OutOfMemoryError: CUDA out of memory. Tried to allocate "
    "14.00 MiB. GPU 0 has a total capacity of 22.30 GiB of which 4.69 MiB is free. "
    "Process 860120 has 22.29 GiB memory in use. Of the allocated memory 21.92 GiB "
    "is allocated by PyTorch, with 22.00 MiB allocated in private pools (e.g., CUDA "
    "Graphs), and 16.47 MiB is reserved by PyTorch but unallocated."
)
_OOM_CAPACITY_TRACE = (
    "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 21.10 GiB. GPU 0 "
    "has a total capacity of 22.30 GiB of which 50.00 MiB is free. Process 189021 has "
    "22.28 GiB memory in use. Of the allocated memory 20.94 GiB is allocated by "
    "PyTorch, with 22.00 MiB allocated in private pools (e.g., CUDA Graphs)."
)
# mamba2's batched test: the request alone fits on an empty card, but we cannot
# tell how much of `held` is its own model, so it must NOT be called fixable.
_OOM_AMBIGUOUS_TRACE = (
    "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 12.00 GiB. GPU 0 "
    "has a total capacity of 22.30 GiB of which 8.01 GiB is free. Process 440984 has "
    "14.29 GiB memory in use. Of the allocated memory 13.91 GiB is allocated by "
    "PyTorch, with 22.00 MiB allocated in private pools (e.g., CUDA Graphs)."
)


def _oom_target(*traces, model="mamba2"):
    """An OOM group whose failures carry the given CUDA OOM messages."""
    return {
        "kind": "model_failures",
        "label": f"{len(traces)} test(s) for `{model}` failing with `OOM`",
        "failures": [
            _failure(
                model, "single", f"tests/models/{model}/t.py::T::t{i}", t, mode="OOM"
            )
            for i, t in enumerate(traces)
        ],
        "cluster": None,
        "model": model,
        "failure_mode": "OOM",
        "terminal_exc": "OutOfMemoryError",
        "crash_site": "",
    }


class OomShapeTest(unittest.TestCase):
    """`OOM` is two different bugs wearing one label. Telling them apart is what
    decides whether a group is dispatched or deferred."""

    def test_trivial_request_on_a_full_card_is_retention(self):
        shape, nums = itf.oom_shape(_OOM_RETENTION_TRACE)
        self.assertEqual(shape, itf.OOM_RETENTION)
        self.assertAlmostEqual(nums["want"], 14 / 1024, places=4)
        self.assertAlmostEqual(nums["held"], 21.92, places=2)
        self.assertAlmostEqual(nums["capacity"], 22.30, places=2)

    def test_one_allocation_near_the_whole_card_is_capacity(self):
        shape, nums = itf.oom_shape(_OOM_CAPACITY_TRACE)
        self.assertEqual(shape, itf.OOM_CAPACITY)
        self.assertAlmostEqual(nums["want"], 21.10, places=2)

    def test_a_big_request_on_a_half_full_card_stays_unknown(self):
        # 12 GiB would fit on an empty 22.3 GiB card, but 13.91 GiB is already
        # held and we cannot attribute it — claiming "fixable" here would send the
        # agent after a teardown that does not exist.
        shape, _ = itf.oom_shape(_OOM_AMBIGUOUS_TRACE)
        self.assertEqual(shape, itf.OOM_UNKNOWN)

    def test_an_unparseable_message_is_unknown_with_no_numbers(self):
        shape, nums = itf.oom_shape("torch.OutOfMemoryError: CUDA out of memory.")
        self.assertEqual(shape, itf.OOM_UNKNOWN)
        self.assertEqual(nums, {})
        self.assertEqual(itf.oom_shape(""), (itf.OOM_UNKNOWN, {}))


class OomEvidenceTest(unittest.TestCase):
    def test_evidence_labels_each_test_and_keeps_the_numbers(self):
        target = _oom_target(_OOM_RETENTION_TRACE, _OOM_CAPACITY_TRACE)
        text = "\n".join(itf.oom_evidence_lines(target))
        self.assertIn("retained memory (fixable)", text)
        self.assertIn("over capacity (do not patch)", text)
        self.assertIn("already held 21.92 GiB of 22.30 GiB", text)

    def test_the_same_test_on_two_runners_keeps_the_conservative_shape(self):
        # Real case: test_deepseek_v2_lite was retention-shaped on multi-gpu and
        # capacity-shaped on single-gpu. Calling it fixable would send the agent
        # after a teardown that cannot make the single-gpu runner fit it.
        target = {
            "kind": "model_failures",
            "label": "1 test for `deepseek_v2` failing with `OOM`",
            "failure_mode": "OOM",
            "model": "deepseek_v2",
            "cluster": None,
            "terminal_exc": "OutOfMemoryError",
            "crash_site": "",
            "failures": [
                _failure(
                    "deepseek_v2",
                    "multi",
                    "t.py::T::lite",
                    _OOM_RETENTION_TRACE,
                    mode="OOM",
                ),
                _failure(
                    "deepseek_v2",
                    "single",
                    "t.py::T::lite",
                    _OOM_CAPACITY_TRACE,
                    mode="OOM",
                ),
            ],
        }
        shapes = itf.oom_shapes(target)
        self.assertEqual(shapes["t.py::T::lite"][0], itf.OOM_CAPACITY)
        # …and the order the runners appear in must not change the verdict.
        target["failures"].reverse()
        self.assertEqual(itf.oom_shapes(target)["t.py::T::lite"][0], itf.OOM_CAPACITY)

    def test_no_parseable_message_renders_nothing(self):
        self.assertEqual(itf.oom_evidence_lines(_oom_target("OutOfMemoryError")), [])

    def test_evidence_reaches_the_serge_context(self):
        context = itf.render_serge_context(
            [_oom_target(_OOM_RETENTION_TRACE)], ["2026-08-14"]
        )
        self.assertIn("Device-memory shape per failing test", context)
        self.assertIn("retained memory (fixable)", context)


class PartitionTargetsTest(unittest.TestCase):
    """Groups no minimal patch can fix must not spend a GPU reproduce run, an
    investigation, or a --max-groups slot."""

    def test_retention_shaped_oom_is_dispatched_not_deferred(self):
        # 26 of the 54 persistent OOMs on 2026-08-14 had this shape and were being
        # deferred as "needs runner capacity" — they are missing-tearDown bugs.
        oom = _oom_target(_OOM_RETENTION_TRACE)
        dispatch, deferred = itf.partition_targets([oom])
        self.assertEqual(len(dispatch), 1)
        self.assertEqual(deferred, [])

    def test_capacity_shaped_oom_is_still_deferred(self):
        oom = _oom_target(_OOM_CAPACITY_TRACE)
        dispatch, deferred = itf.partition_targets([oom])
        self.assertEqual(dispatch, [])
        self.assertIn("device memory", deferred[0]["defer_reason"])

    def test_a_mixed_group_is_dispatched_for_the_fixable_test(self):
        oom = _oom_target(_OOM_CAPACITY_TRACE, _OOM_RETENTION_TRACE)
        dispatch, _ = itf.partition_targets([oom])
        self.assertEqual(len(dispatch), 1)

    def test_ambiguous_oom_stays_deferred(self):
        dispatch, deferred = itf.partition_targets([_oom_target(_OOM_AMBIGUOUS_TRACE)])
        self.assertEqual(dispatch, [])
        self.assertEqual(len(deferred), 1)

    def test_oom_is_deferred_with_a_reason(self):
        oom = _target("OOM", exc="OutOfMemoryError", model="big")
        mismatch = _target("output_mismatch", model="dac")
        dispatch, deferred = itf.partition_targets([oom, mismatch])
        self.assertEqual([t["model"] for t in dispatch], ["dac"])
        self.assertEqual([t["model"] for t in deferred], ["big"])
        self.assertIn("device memory", deferred[0]["defer_reason"])

    def test_missing_dependency_is_deferred_but_other_config_errors_are_not(self):
        dep = _target("import_or_config", exc="ModuleNotFoundError")
        api = _target("import_or_config", exc="AttributeError")
        dispatch, deferred = itf.partition_targets([dep, api])
        self.assertEqual(len(deferred), 1)
        self.assertEqual(deferred[0]["terminal_exc"], "ModuleNotFoundError")
        self.assertEqual(dispatch[0]["terminal_exc"], "AttributeError")

    def test_clusters_are_never_deferred(self):
        # An attributed bad commit is a stronger signal than its members' modes.
        cluster = _target("OOM", exc="OutOfMemoryError", kind="cluster")
        dispatch, deferred = itf.partition_targets([cluster])
        self.assertEqual(len(dispatch), 1)
        self.assertEqual(deferred, [])

    def test_opt_out_dispatches_everything(self):
        oom = _target("OOM", exc="OutOfMemoryError")
        with patch.dict("os.environ", {"ITF_DEFER_ENV_GROUPS": "0"}):
            dispatch, deferred = itf.partition_targets([oom])
        self.assertEqual(len(dispatch), 1)
        self.assertEqual(deferred, [])

    def test_deferred_groups_are_reported_in_the_tracking_issue(self):
        oom = _target("OOM", exc="OutOfMemoryError", model="big")
        dispatch, deferred = itf.partition_targets([oom, _target("output_mismatch")])
        body = itf.render_tracking_issue_body(
            dispatch, ["2026-07-20"], "2026-07-20", deferred=deferred
        )
        self.assertIn("Not dispatched — environment / dependency", body)
        self.assertIn("`big`", body)
        # The deferred table must not be mistaken for a dispatched row on a
        # later same-day run (that parser keys off a `| PR |` header).
        carried = itf._carry_forward_rows(body, [])
        self.assertFalse(any("big" in row for row in carried))

    def test_oom_groups_collapse_to_one_line_instead_of_a_table(self):
        # Nobody patches an OOM, and a week can defer 18 of them; the section
        # must not spend a table row each (issue #47936).
        oom = [
            _target("OOM", exc="OutOfMemoryError", model=f"m{i}", n=i + 1)
            for i in range(3)
        ]
        section = "\n".join(itf._render_deferred_section(oom))
        self.assertIn("3 models ran out of device memory", section)
        self.assertIn("6 failures", section)  # 1 + 2 + 3
        self.assertNotIn("| Model |", section)  # no table at all
        # Worst-first, with each model's occurrence count kept inline.
        self.assertIn("`m2` (3), `m1` (2), `m0` (1)", section)

    def test_non_oom_deferred_groups_keep_their_table_row(self):
        dep = _target("import_or_config", exc="ModuleNotFoundError", model="dep")
        _, deferred = itf.partition_targets(
            [_target("OOM", exc="OutOfMemoryError", model="big"), dep]
        )
        section = "\n".join(itf._render_deferred_section(deferred))
        self.assertIn("1 model ran out of device memory", section)
        self.assertIn("| Model |", section)
        rows = [ln for ln in section.splitlines() if ln.startswith("| `")]
        self.assertEqual(len(rows), 1)
        self.assertIn("`dep`", rows[0])


class TraceBudgetByCategoryTest(unittest.TestCase):
    """A crash group is N copies of one traceback (it was keyed by `crash_site`);
    a mismatch group's members each carry different expected values."""

    def test_crash_group_renders_few_full_traces_but_keeps_every_nodeid(self):
        target = _target(
            "cuda_runtime", exc="RuntimeError", site="src/x.py:1", n=10, model="big"
        )
        lines = itf._render_serge_target(target, 7)
        text = "\n".join(lines)
        # Every node-id survives — serge parses them to build the reproduce run.
        for i in range(10):
            self.assertIn(f"::T::t{i}`", text)
        self.assertEqual(text.count("  ```"), 2 * itf._CRASH_TRACE_LIMIT)
        self.assertIn("SAME raising line", text)

    def test_mismatch_group_renders_a_trace_for_every_failure(self):
        target = _target("output_mismatch", n=10, model="dac")
        text = "\n".join(itf._render_serge_target(target, 7))
        self.assertEqual(text.count("  ```"), 20)
        self.assertNotIn("SAME raising line", text)

    def test_crash_traces_get_more_room_each_than_mismatch_traces(self):
        crash = _target("cuda_runtime", exc="RuntimeError", site="src/x.py:1", n=40)
        mismatch = _target("output_mismatch", n=40)
        self.assertTrue(itf._groups_by_crash_site(crash))
        self.assertFalse(itf._groups_by_crash_site(mismatch))
        # Same budget, fewer traces → each crash traceback is allowed to be
        # complete instead of being cut to ~1/40th of the budget.
        self.assertGreater(
            itf._SERGE_TRACE_BUDGET // itf._CRASH_TRACE_LIMIT,
            itf._SERGE_TRACE_BUDGET // itf._FULL_TRACE_LIMIT,
        )


def _attr_day(test, *, status, bad_commit=None, model="foo", gpu="single-gpu"):
    """One day's `new_failures_with_bad_commit_grouped_by_authors.json`."""
    return {
        "someone": {
            model: {gpu: [{"test": test, "status": status, "bad_commit": bad_commit}]}
        }
    }


class AttributionHistoryTests(unittest.TestCase):
    TEST = "tests/models/foo/test_modeling_foo.py::FooIntegrationTest::test_x"
    KEY = ("foo", "single", TEST)

    def test_walks_back_past_the_window_and_stamps_the_day(self):
        idx = itf.index_attribution_history(
            {
                "2026-08-01": _attr_day(
                    self.TEST, status=itf._GOOD_STATUS, bad_commit="deadbeef"
                ),
                "2026-08-10": {},
            }
        )
        self.assertEqual(idx[self.KEY]["bad_commit"], "deadbeef")
        self.assertEqual(idx[self.KEY]["attributed_on"], "2026-08-01")

    def test_a_later_flaky_record_does_not_bury_an_earlier_pin(self):
        # Upstream stops re-bisecting once it converges, so the pin is usually
        # the OLDER record; newest-wins would throw the culprit away.
        idx = itf.index_attribution_history(
            {
                "2026-08-01": _attr_day(
                    self.TEST, status=itf._GOOD_STATUS, bad_commit="deadbeef"
                ),
                "2026-08-09": _attr_day(self.TEST, status="flaky: passed and failed"),
            }
        )
        self.assertEqual(idx[self.KEY]["bad_commit"], "deadbeef")

    def test_newest_wins_between_two_unpinned_records(self):
        idx = itf.index_attribution_history(
            {
                "2026-08-01": _attr_day(self.TEST, status="flaky: old"),
                "2026-08-09": _attr_day(self.TEST, status="flaky: new"),
            }
        )
        self.assertEqual(idx[self.KEY]["status"], "flaky: new")

    def test_cluster_failures_uses_the_history_index(self):
        failure = {
            "model": "foo",
            "gpu": "single",
            "test": self.TEST,
            "trace": "boom",
            "latest_trace": "boom",
        }
        idx = itf.index_attribution_history(
            {
                "2026-08-01": _attr_day(
                    self.TEST, status=itf._GOOD_STATUS, bad_commit="deadbeef"
                )
            }
        )
        # The latest day is empty — exactly the production case — so without the
        # history index this failure is unattributed.
        plain = itf.cluster_failures([failure], {})
        self.assertEqual(plain["totals"]["clusters"], 0)
        withhist = itf.cluster_failures([failure], {}, attribution=idx)
        self.assertEqual(withhist["totals"]["clusters"], 1)
        self.assertIn("deadbeef", withhist["clusters"])


class GroupLabelTests(unittest.TestCase):
    """The issue tables used to show a bare `cluster <sha>` (and `—` in the
    recap), which says nothing about what is broken."""

    def _cluster(self, models, **cluster):
        failures = [
            {"model": m, "gpu": "single", "test": f"tests/models/{m}/t.py::T::t"}
            for m in models
        ]
        return {
            "kind": "cluster",
            "label": "N integration tests regressed by commit deadbeefcafe",
            "model": None,
            "failures": failures,
            "cluster": {"bad_commit": "deadbeefcafe1234", **cluster},
        }

    def test_a_model_group_is_just_its_model(self):
        self.assertEqual(
            itf.group_label({"model": "whisper", "failures": []}), "`whisper`"
        )

    def test_a_cluster_names_the_model_and_the_pr(self):
        label = itf.group_label(self._cluster(["florence2"], pr_number=46556))
        self.assertEqual(label, "`florence2` (regressed by PR #46556)")

    def test_a_cluster_without_a_pr_falls_back_to_the_commit(self):
        self.assertEqual(
            itf.group_label(self._cluster(["florence2"])),
            "`florence2` (regressed by commit deadbeef)",
        )

    def test_many_models_are_capped(self):
        label = itf.group_label(self._cluster(["a", "b", "c", "d", "e"], pr_number=1))
        self.assertEqual(label, "`a`, `b`, `c` +2 more (regressed by PR #1)")

    def test_no_sha_is_never_shown_alone(self):
        for target in (
            self._cluster(["florence2"], pr_number=46556),
            self._cluster(["florence2"]),
        ):
            self.assertNotIn("cluster ", itf.group_label(target))
            self.assertIn("florence2", itf.group_label(target))

    def test_a_group_with_no_model_at_all_still_renders(self):
        self.assertEqual(
            itf.group_label({"kind": "cluster", "failures": [], "cluster": {}}),
            "unknown model",
        )


class CrossGpuAttributionTests(unittest.TestCase):
    TEST = "tests/models/t5/test_modeling_t5.py::T5ModelIntegrationTests::test_x"

    def setUp(self):
        self.attr = {
            ("t5", "single", self.TEST): {
                "status": itf._GOOD_STATUS,
                "bad_commit": "9f66415a",
            }
        }

    def test_the_other_machine_type_inherits_the_pin(self):
        # Upstream bisects one machine type; the daily CI runs both, so the
        # multi-gpu half of the same regression arrives unattributed.
        rec = itf.lookup_attribution(self.attr, ("t5", "multi", self.TEST))
        self.assertEqual(rec["bad_commit"], "9f66415a")
        self.assertEqual(rec["attributed_gpu"], "single")

    def test_the_pinned_machine_type_is_not_stamped(self):
        rec = itf.lookup_attribution(self.attr, ("t5", "single", self.TEST))
        self.assertNotIn("attributed_gpu", rec)

    def test_a_different_test_does_not_inherit(self):
        self.assertIsNone(
            itf.lookup_attribution(self.attr, ("t5", "multi", "other.py::T::t"))
        )

    def test_an_unpinned_exact_record_still_loses_to_a_pin_elsewhere(self):
        self.attr[("t5", "multi", self.TEST)] = {"status": "flaky: x"}
        rec = itf.lookup_attribution(self.attr, ("t5", "multi", self.TEST))
        self.assertEqual(rec["bad_commit"], "9f66415a")

    def test_renders_where_the_pin_came_from(self):
        failure = {
            "model": "t5",
            "gpu": "multi",
            "test": self.TEST,
            "failure_mode": "other",
            "days_seen": 7,
            "latest_trace": "boom",
        }
        target = {
            "kind": "cluster",
            "label": "c",
            "failures": [failure],
            "cluster": {
                "bad_commit": "9f66415a",
                "failures": [failure],
                "attributed_gpu": "single",
                "attributed_on": "2026-08-05",
            },
        }
        text = "\n".join(itf._render_serge_target(target, 7))
        self.assertIn("bisect ran on the single-gpu run", text)
        self.assertIn("2026-08-05", text)


class JobProducedResultsTests(unittest.TestCase):
    def test_a_crashed_job_is_not_a_green_day(self):
        # 2026-08-13 models_gpt_oss: reported nothing, so its 74 red tests
        # simply vanished from the failure list.
        crashed = {
            "success": 0,
            "skipped": 0,
            "errors": 0,
            "error": True,
            "time_spent": [],
        }
        self.assertFalse(itf.model_job_produced_results(crashed))

    def test_a_real_run_counts(self):
        self.assertTrue(
            itf.model_job_produced_results({"success": 141, "error": False})
        )

    def test_a_job_with_no_successes_does_not_count(self):
        self.assertFalse(itf.model_job_produced_results({"success": 0, "error": False}))
        self.assertFalse(itf.model_job_produced_results(None))


def _history(days):
    """days: {date: (models_with_results, {failing keys})}"""
    return {
        "dates": sorted(days),
        "ran": {d: set(v[0]) for d, v in days.items()},
        "failures": {d: set(v[1]) for d, v in days.items()},
    }


class FindFlipTests(unittest.TestCase):
    KEY = ("foo", "single", "tests/models/foo/test_modeling_foo.py::F::test_x")

    def test_finds_the_day_it_went_red(self):
        h = _history(
            {
                "2026-08-01": ({"foo"}, set()),
                "2026-08-02": ({"foo"}, {self.KEY}),
                "2026-08-03": ({"foo"}, {self.KEY}),
            }
        )
        self.assertEqual(itf.find_flip(self.KEY, h), ("2026-08-01", "2026-08-02"))

    def test_a_day_without_results_is_skipped_not_treated_as_green(self):
        h = _history(
            {
                "2026-08-01": ({"foo"}, {self.KEY}),
                "2026-08-02": (set(), set()),  # job crashed
                "2026-08-03": ({"foo"}, {self.KEY}),
            }
        )
        self.assertIsNone(itf.find_flip(self.KEY, h))

    def test_red_for_the_whole_window_has_no_bracket(self):
        h = _history(
            {
                "2026-08-01": ({"foo"}, {self.KEY}),
                "2026-08-02": ({"foo"}, {self.KEY}),
            }
        )
        self.assertIsNone(itf.find_flip(self.KEY, h))

    def test_green_on_the_newest_day_has_no_bracket(self):
        h = _history(
            {
                "2026-08-01": ({"foo"}, {self.KEY}),
                "2026-08-02": ({"foo"}, set()),
            }
        )
        self.assertIsNone(itf.find_flip(self.KEY, h))


class ComputeBracketTests(unittest.TestCase):
    NODEID = "tests/models/foo/test_modeling_foo.py::F::test_x"
    FAILURE = {"model": "foo", "gpu": "single", "test": NODEID}
    TODAY = datetime.date(2026, 8, 3)

    def setUp(self):
        self.history = _history(
            {
                "2026-08-01": ({"foo"}, set()),
                "2026-08-02": ({"foo"}, {("foo", "single", self.NODEID)}),
            }
        )
        self.shas = {
            "2026-08-01": {"single": "good123"},
            "2026-08-02": {"single": "bad456"},
        }

    def _run(self, status="passed", compare={"total_commits": 3}, **kw):
        with (
            patch.object(itf, "collated_test_status", return_value=status),
            patch.object(itf, "compare_commits", return_value=compare),
        ):
            return itf.compute_bracket(
                self.FAILURE,
                self.history,
                self.shas,
                repo="huggingface/transformers",
                today=self.TODAY,
                **kw,
            )

    def test_a_confirmed_pass_produces_a_bracket(self):
        b = self._run()
        self.assertEqual(b["good_sha"], "good123")
        self.assertEqual(b["bad_sha"], "bad456")
        self.assertEqual(b["commits"], 3)

    def test_skipped_is_not_a_pass(self):
        self.assertIsNone(self._run(status="skipped"))

    def test_absent_from_the_report_is_not_a_pass(self):
        self.assertIsNone(self._run(status=itf.STATUS_ABSENT))

    def test_an_unresolvable_sha_drops_the_bracket(self):
        # A daily-CI commit can be force-pushed away (918dbf1 → HTTP 422).
        self.assertIsNone(self._run(compare=None))

    def test_zero_commits_between_is_a_flake_not_a_regression(self):
        self.assertIsNone(self._run(compare={"total_commits": 0}))

    def test_a_green_day_older_than_the_cap_is_ignored(self):
        self.assertIsNone(self._run(max_age_days=0))

    def test_a_missing_day_sha_drops_the_bracket(self):
        self.shas = {"2026-08-01": {"single": "good123"}}
        self.assertIsNone(self._run())


class BracketRenderingTests(unittest.TestCase):
    def _target(self, **bracket):
        base = {
            "good_day": "2026-08-03",
            "good_sha": "b3a3603",
            "bad_day": "2026-08-04",
            "bad_sha": "ff2421c",
            "commits": 14,
            "compare": "https://example/compare",
            "evidence": "collated_reports: `passed` …",
            "subjects": ["Update daily CI Docker image to torch 2.13 (#47738)"],
        }
        base.update(bracket)
        return {
            "kind": "model_failures",
            "label": "foo",
            "failures": [],
            "cluster": None,
            "bracket": base,
        }

    def test_emits_a_machine_readable_block(self):
        text = "\n".join(itf._bracket_lines(self._target()))
        self.assertIn("```serge-bisect", text)
        self.assertIn('"good_sha": "b3a3603"', text)
        self.assertIn("b3a3603", text)
        self.assertIn("torch 2.13", text)

    def test_many_unrelated_models_reads_as_infrastructure(self):
        models = [f"m{i}" for i in range(itf._INFRA_MODEL_THRESHOLD)]
        text = "\n".join(itf._bracket_lines(self._target(shared_models=models)))
        self.assertIn("infrastructure", text)

    def test_one_model_does_not(self):
        text = "\n".join(itf._bracket_lines(self._target(shared_models=["foo"])))
        self.assertNotIn("infrastructure", text)

    def test_no_bracket_renders_nothing(self):
        self.assertEqual(itf._bracket_lines({"bracket": None}), [])


class FlakyHandlingTests(unittest.TestCase):
    def _group(self, flaky, label="g"):
        status = "flaky: test both passed and failed …: 0cdd8a19" if flaky else ""
        return {
            "kind": "model_failures",
            "label": label,
            "model": "foo",
            "failure_mode": "output_mismatch",
            "terminal_exc": "AssertionError",
            "cluster": None,
            "failures": [
                {
                    "model": "foo",
                    "gpu": "single",
                    "test": "tests/models/foo/test_modeling_foo.py::F::test_x",
                    "status": status,
                }
            ],
        }

    def test_flaky_is_warned_about_in_the_task_context(self):
        text = "\n".join(itf._flaky_lines(self._group(True)))
        self.assertIn("Known flaky upstream", text)
        self.assertIn("0cdd8a19", text)
        self.assertIn("does not prove a fix", text)

    def test_non_flaky_group_says_nothing(self):
        self.assertEqual(itf._flaky_lines(self._group(False)), [])

    def test_flaky_groups_lose_the_tie_for_a_dispatch_slot(self):
        targets = [self._group(True, "flaky-a"), self._group(False, "solid-b")]
        picked = itf.select_dispatch_targets(
            targets, 1, shuffle=False, mix_categories=True
        )
        self.assertEqual([t["label"] for t in picked], ["solid-b"])

    def test_flaky_groups_are_not_excluded_when_there_is_room(self):
        targets = [self._group(True, "flaky-a"), self._group(False, "solid-b")]
        picked = itf.select_dispatch_targets(
            targets, 2, shuffle=False, mix_categories=True
        )
        self.assertEqual(len(picked), 2)

    def test_a_flaky_group_is_never_bracketed(self):
        targets = [self._group(True)]
        with patch.object(itf, "compute_bracket") as cb:
            itf.attach_brackets(targets, _history({}), {}, repo="x/y")
        cb.assert_not_called()
        self.assertIsNone(targets[0]["bracket"])


class AttachBracketsTests(unittest.TestCase):
    def _group(self, tests):
        return {
            "kind": "model_failures",
            "label": "g",
            "cluster": None,
            "failures": [
                {"model": "foo", "gpu": "single", "test": t, "status": ""}
                for t in tests
            ],
        }

    def test_members_must_agree_on_one_window(self):
        targets = [self._group(["a", "b"])]
        brackets = [
            {"good_day": "2026-08-01", "bad_day": "2026-08-02"},
            {"good_day": "2026-07-20", "bad_day": "2026-07-21"},
        ]
        with patch.object(itf, "compute_bracket", side_effect=brackets):
            itf.attach_brackets(targets, _history({}), {}, repo="x/y")
        self.assertIsNone(targets[0]["bracket"])

    def test_one_member_without_a_bracket_drops_the_group(self):
        targets = [self._group(["a", "b"])]
        with patch.object(
            itf,
            "compute_bracket",
            side_effect=[{"good_day": "d1", "bad_day": "d2"}, None],
        ):
            itf.attach_brackets(targets, _history({}), {}, repo="x/y")
        self.assertIsNone(targets[0]["bracket"])

    def test_an_already_pinned_group_is_not_bracketed(self):
        # The pin names the culprit exactly; a 20-commit window around it is
        # noise, and deriving it costs a ~25 MB download.
        target = self._group(["a"])
        target["cluster"] = {"bad_commit": "deadbeef"}
        with patch.object(itf, "compute_bracket") as cb:
            itf.attach_brackets([target], _history({}), {}, repo="x/y")
        cb.assert_not_called()
        self.assertIsNone(target["bracket"])

    def test_shared_windows_collect_every_model(self):
        a, b = self._group(["a"]), self._group(["b"])
        b["failures"][0]["model"] = "bar"
        window = {"good_day": "d1", "bad_day": "d2"}
        with patch.object(
            itf, "compute_bracket", side_effect=[dict(window), dict(window)]
        ):
            itf.attach_brackets([a, b], _history({}), {}, repo="x/y")
        self.assertEqual(a["bracket"]["shared_models"], ["bar", "foo"])


if __name__ == "__main__":
    unittest.main()
