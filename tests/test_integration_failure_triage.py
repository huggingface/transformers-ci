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
            st = itf.poll_serge_status("http://s", "tok", "o/r", "j1")
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
            self.assertIsNone(itf.poll_serge_status("http://s", "tok", "o/r", "j1"))

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
            patch.object(itf, "poll_serge_status", return_value="no_fix"),
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


if __name__ == "__main__":
    unittest.main()
