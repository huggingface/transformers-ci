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
import itertools
import json
import os
import unittest
import urllib.error
from unittest.mock import patch

from transformersci.agentic import github_api as gh_api
from transformersci.agentic import integration_failure_triage as itf
from transformersci.agentic import prometheus_api as prom
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


class FingerprintStabilityTest(unittest.TestCase):
    """v1 hashed the traceback excerpt into the group identity, so an unchanged
    failure fingerprinted differently every run and the existing-PR lookup never
    matched. Observed as six mamba2 OOM attempts with six distinct fingerprints."""

    def _target(self, trace):
        return {
            "kind": "model_failures",
            "label": "2 tests for model `mamba2` failing with `OOM`",
            "cluster": None,
            "failures": [
                {
                    "test": "tests/models/mamba2/test_modeling_mamba2.py::T::test_a",
                    "gpu": "single-gpu",
                    "failure_mode": "OOM",
                    "latest_trace": trace,
                }
            ],
        }

    def test_stable_across_differing_tracebacks(self):
        a = self._target("CUDA out of memory. Tried to allocate 20.00 MiB")
        b = self._target("CUDA out of memory. Tried to allocate 44.00 MiB")
        self.assertEqual(itf.target_fingerprint(a), itf.target_fingerprint(b))

    def test_v1_was_unstable(self):
        a = self._target("CUDA out of memory. Tried to allocate 20.00 MiB")
        b = self._target("CUDA out of memory. Tried to allocate 44.00 MiB")
        self.assertNotEqual(
            itf.target_fingerprint(a, version=1),
            itf.target_fingerprint(b, version=1),
        )

    def test_identity_change_still_changes_the_fingerprint(self):
        a = self._target("boom")
        b = self._target("boom")
        b["failures"][0]["test"] = "tests/models/mamba2/test_modeling_mamba2.py::T::z"
        self.assertNotEqual(itf.target_fingerprint(a), itf.target_fingerprint(b))

    def test_candidates_include_the_legacy_scheme(self):
        t = self._target("boom")
        cands = itf.fingerprint_candidates(t)
        self.assertEqual(cands[0], itf.target_fingerprint(t))
        self.assertIn(itf.target_fingerprint(t, version=1), cands)


class PriorAttemptsTest(unittest.TestCase):
    FP = "d" * 64

    def _pr(self, number, state, merged=False, fp=None):
        return {
            "number": number,
            "state": state,
            "merged_at": "2026-08-01T00:00:00Z" if merged else None,
            "body": itf.fingerprint_marker(fp or self.FP),
            "head": {"ref": "x"},
        }

    def test_buckets_open_merged_and_rejected(self):
        pulls = [
            self._pr(1, "closed"),
            self._pr(2, "closed", merged=True),
            self._pr(3, "open"),
            self._pr(4, "closed"),
            {"number": 9, "state": "closed", "body": "other", "head": {"ref": "y"}},
        ]
        prior = itf.classify_prior_attempts(pulls, [self.FP])
        self.assertEqual(prior.open_pr, 3)
        self.assertEqual(prior.merged, (2,))
        self.assertEqual(prior.rejected, (1, 4))
        self.assertEqual(prior.attempts, 4)

    def test_open_only_listing_degrades_safely(self):
        """Given only open PRs (the old ledger) nothing is ever deferred."""
        prior = itf.classify_prior_attempts([self._pr(3, "open")], [self.FP])
        self.assertEqual(prior.rejected, ())
        self.assertEqual(itf.rejected_attempts_reason(prior, 2), "")

    def test_a_pr_without_an_explicit_state_counts_as_open(self):
        """Fail-safe: a phantom rejection would wrongly skip a live group."""
        prior = itf.classify_prior_attempts(
            [{"number": 62, "body": itf.fingerprint_marker(self.FP), "head": {}}],
            [self.FP],
        )
        self.assertEqual(prior.open_pr, 62)
        self.assertEqual(prior.rejected, ())

    def test_matches_a_legacy_fingerprint(self):
        legacy = "e" * 64
        prior = itf.classify_prior_attempts(
            [self._pr(7, "closed", fp=legacy)], [self.FP, legacy]
        )
        self.assertEqual(prior.rejected, (7,))


class RejectedAttemptsReasonTest(unittest.TestCase):
    def test_defers_at_the_threshold(self):
        prior = itf.PriorAttempts(rejected=(11, 12))
        reason = itf.rejected_attempts_reason(prior, 2)
        self.assertIn("#11", reason)
        self.assertIn("#12", reason)

    def test_below_threshold_still_dispatches(self):
        self.assertEqual(
            itf.rejected_attempts_reason(itf.PriorAttempts(rejected=(11,)), 2), ""
        )

    def test_an_open_pr_is_a_follow_up_not_a_block(self):
        prior = itf.PriorAttempts(open_pr=20, rejected=(11, 12))
        self.assertEqual(itf.rejected_attempts_reason(prior, 2), "")

    def test_a_merged_fix_that_did_not_hold_is_new_information(self):
        prior = itf.PriorAttempts(merged=(30,), rejected=(11, 12))
        self.assertEqual(itf.rejected_attempts_reason(prior, 2), "")

    def test_zero_disables(self):
        prior = itf.PriorAttempts(rejected=(11, 12, 13))
        self.assertEqual(itf.rejected_attempts_reason(prior, 0), "")


class PartitionTargetsPriorAttemptsTest(unittest.TestCase):
    def _target(self):
        return {
            "kind": "model_failures",
            "label": "2 tests for model `generation` failing with `output_mismatch`",
            "model": "generation",
            "failure_mode": "output_mismatch",
            "cluster": None,
            "failures": [
                {
                    "test": "tests/generation/test_utils.py::T::test_a",
                    "gpu": "single-gpu",
                    "failure_mode": "output_mismatch",
                    "latest_trace": "AssertionError: Tensor-likes are not close",
                }
            ],
        }

    def test_repeatedly_rejected_group_is_deferred(self):
        t = self._target()
        priors = {itf.target_fingerprint(t): itf.PriorAttempts(rejected=(1, 2))}
        dispatch, deferred = partition = itf.partition_targets(
            [t], priors, max_rejected=2
        )
        self.assertEqual(dispatch, [])
        self.assertEqual(len(deferred), 1)
        self.assertIn("closed unmerged", deferred[0]["defer_reason"])
        del partition

    def test_without_priors_behaviour_is_unchanged(self):
        t = self._target()
        dispatch, deferred = itf.partition_targets([t])
        self.assertEqual(dispatch, [t])
        self.assertEqual(deferred, [])


def _prom_series(pairs):
    """`[({labels}, value), ...]` -> a Prometheus instant-query result list."""
    return [{"metric": m, "value": [0, str(v)]} for m, v in pairs]


class PrometheusApiTest(unittest.TestCase):
    def test_parses_a_successful_response(self):
        body = json.dumps(
            {
                "status": "success",
                "data": {"result": [{"metric": {"a": "b"}, "value": [1, "2"]}]},
            }
        )
        out = prom.instant_query("up", fetch=lambda url: body)
        self.assertEqual(out[0]["metric"]["a"], "b")

    def test_query_is_url_encoded_into_the_proxy_path(self):
        seen = {}

        def fetch(url):
            seen["url"] = url
            return json.dumps({"status": "success", "data": {"result": []}})

        prom.instant_query('x{y="z"}', base_url="https://g.example", fetch=fetch)
        self.assertTrue(seen["url"].startswith("https://g.example" + prom.QUERY_PATH))
        self.assertIn("y%3D%22z%22", seen["url"])

    def test_every_failure_mode_degrades_to_no_data(self):
        # unattended nightly: "no data" must never become a crashed run
        for body in (
            '{"status": "error"}',
            "not json",
            '{"data": {"result": {}}}',
            '{"status": "success"}',
            "[]",
        ):
            self.assertEqual(prom.instant_query("up", fetch=lambda url: body), [])

        def boom(url):
            raise urllib.error.URLError("unreachable")

        self.assertEqual(prom.instant_query("up", fetch=boom), [])

    def test_scalars_by_label_skips_unparseable_samples(self):
        series = _prom_series([({"run_id": "a"}, 1.5), ({"run_id": "b"}, "nan-ish")])
        series.append({"metric": {}, "value": [0, "3"]})
        series.append({"metric": {"run_id": "c"}, "value": None})
        out = prom.scalars_by_label(series, "run_id")
        self.assertEqual(out, {"a": 1.5})


class LivenessTest(unittest.TestCase):
    # Two daily runs a day: one ~12.8k-test run and one ~162k full suite.
    TOTALS = {
        "big1": 162000,
        "small1": 12800,
        "big2": 160000,
        "big3": 161000,
        "small2": 12900,
        "push1": 10400,
    }
    STARTS = {
        "big1": 100,
        "small1": 200,
        "big2": 300,
        "big3": 400,
        "small2": 500,
        "push1": 450,
    }
    EVENTS = {
        "big1": "daily",
        "small1": "daily",
        "big2": "daily",
        "big3": "daily",
        "small2": "daily",
        "push1": "push",
    }

    def _query(self, failing, totals=None, starts=None):
        totals = self.TOTALS if totals is None else totals
        starts = self.STARTS if starts is None else starts

        def q(expr):
            if "pytest_run_job_total_tests" in expr:
                return _prom_series(
                    [
                        ({"run_id": k, "ci_event": self.EVENTS.get(k, "daily")}, v)
                        for k, v in totals.items()
                    ]
                )
            if "pytest_run_start_time_seconds" in expr:
                return _prom_series([({"run_id": k}, v) for k, v in starts.items()])
            if "pytest_test_last_failure_info" in expr:
                return _prom_series(
                    [
                        ({"test_nodeid": n, "run_id": r}, 1)
                        for n, rs in failing.items()
                        for r in rs
                    ]
                )
            raise AssertionError(f"unexpected query: {expr}")

        return q

    def _target(self, *tests, kind="model_failures"):
        return {
            "kind": kind,
            "label": "group",
            "model": "m",
            "failures": [{"test": t, "gpu": "single-gpu"} for t in tests],
        }

    def test_baseline_is_full_suite_runs_only_newest_first(self):
        live = itf.fetch_liveness(runs=3, query=self._query({"t": ["big3"]}))
        # small2 is the newest run of all, and must NOT be a baseline run: a test
        # absent from a 12.8k-test run is absent, not green.
        self.assertEqual(live.runs, ("big3", "big2", "big1"))
        self.assertNotIn("small2", live.runs)
        self.assertTrue(live.usable)

    def test_recent_spans_every_run_since_the_baseline_started(self):
        live = itf.fetch_liveness(runs=3, query=self._query({"t": ["big3"]}))
        # presence counts everywhere: partial and push runs are in `recent` ...
        self.assertEqual(live.recent, frozenset(self.STARTS))
        # ... while absence is only trusted across the full-suite baseline.
        self.assertEqual(len(live.runs), 3)

    def test_still_failing_group_dispatches(self):
        live = itf.fetch_liveness(runs=3, query=self._query({"t": ["big3"]}))
        self.assertEqual(itf.settled_reason(self._target("t"), live), "")

    def test_group_that_only_failed_in_a_PARTIAL_run_still_dispatches(self):
        # The trap this gate exists to avoid, from the other side: `small2` is a
        # 12.9k-test run, so it is no baseline — but a failure there is a real,
        # current failure and must not be skipped.
        live = itf.fetch_liveness(runs=3, query=self._query({"t": ["small2"]}))
        self.assertEqual(itf.settled_reason(self._target("t"), live), "")

    def test_group_that_only_failed_in_a_push_run_still_dispatches(self):
        live = itf.fetch_liveness(runs=3, query=self._query({"t": ["push1"]}))
        self.assertEqual(itf.settled_reason(self._target("t"), live), "")

    def test_settled_group_is_held_back_with_a_reason(self):
        # failed only before the baseline window opened
        live = itf.fetch_liveness(runs=2, query=self._query({"t": ["big1", "small1"]}))
        self.assertEqual(live.runs, ("big3", "big2"))
        reason = itf.settled_reason(self._target("t"), live)
        self.assertIn("stopped failing", reason)
        self.assertIn("full-suite run_models_gpu", reason)

    def test_unknown_nodeids_fail_SAFE_and_dispatch(self):
        # A node-id format drift between the dataset and the exporter looks
        # exactly like "all green". It must never skip the pool.
        live = itf.fetch_liveness(runs=3, query=self._query({"other": ["big1"]}))
        self.assertTrue(live.usable)
        self.assertEqual(itf.settled_reason(self._target("t"), live), "")

    def test_one_still_failing_member_keeps_the_whole_group(self):
        live = itf.fetch_liveness(
            runs=3, query=self._query({"a": ["big1"], "b": ["big3"]})
        )
        self.assertEqual(itf.settled_reason(self._target("a", "b"), live), "")

    def test_bad_commit_clusters_are_gated_too(self):
        live = itf.fetch_liveness(runs=2, query=self._query({"t": ["big1"]}))
        reason = itf.settled_reason(self._target("t", kind="cluster"), live)
        self.assertIn("stopped failing", reason)

    def test_thin_or_missing_data_dispatches(self):
        # no failures observed at all -> indistinguishable from a broken query
        live = itf.fetch_liveness(runs=3, query=self._query({}))
        self.assertFalse(live.usable)
        self.assertEqual(itf.settled_reason(self._target("t"), live), "")
        # empty responses
        empty = itf.fetch_liveness(runs=3, query=lambda expr: [])
        self.assertFalse(empty.usable)
        self.assertEqual(itf.settled_reason(self._target("t"), empty), "")
        self.assertEqual(itf.settled_reason(self._target("t"), None), "")

    def test_no_daily_run_means_no_baseline(self):
        live = itf.fetch_liveness(
            runs=3,
            query=self._query(
                {"t": ["push1"]}, totals={"push1": 10400}, starts={"push1": 450}
            ),
        )
        self.assertEqual(live.runs, ())
        self.assertFalse(live.usable)

    def test_default_runs_stays_within_the_green_days_the_window_allows(self):
        # A candidate must fail on >= --min-days of --window days to reach this
        # gate, so it can only have been green for (window - min_days) days. A
        # baseline wider than that can never be failure-free, so the gate would
        # silently never fire. Measured 2026-09-03 against prod: 0 of 155 groups
        # settled at runs>=2, 1 (`phimoe`) at runs=1. Pinned so that
        # "hardening" the default cannot quietly turn the check off.
        allowed_green_days = 7 - 5  # --window / --min-days defaults
        self.assertLessEqual(itf.DEFAULT_LIVENESS_RUNS, allowed_green_days)
        self.assertLessEqual(itf.MIN_LIVENESS_RUNS, allowed_green_days)
        # and in practice the dataset yields 6 usable days, not 7
        self.assertLessEqual(itf.MIN_LIVENESS_RUNS, 6 - 5)

    def test_one_full_suite_run_is_enough_to_gate_on(self):
        live = itf.fetch_liveness(
            runs=1, query=self._query({"t": ["big1"], "u": ["big3"]})
        )
        self.assertEqual(live.runs, ("big3",))
        self.assertTrue(live.usable)
        # `u` failed in that run -> dispatch; `t` did not -> settled
        self.assertEqual(itf.settled_reason(self._target("u"), live), "")
        self.assertIn("stopped failing", itf.settled_reason(self._target("t"), live))


class DropSettledTargetsTest(LivenessTest):
    def test_settled_groups_are_split_out_and_annotated(self):
        live = itf.fetch_liveness(runs=2, query=self._query({"t": ["big1"]}))
        keep, settled = itf.drop_settled_targets([self._target("t")], live)
        self.assertEqual(keep, [])
        self.assertEqual(len(settled), 1)
        self.assertIn("stopped failing", settled[0]["settled_reason"])

    def test_env_flag_disables_the_gate(self):
        live = itf.fetch_liveness(runs=2, query=self._query({"t": ["big1"]}))
        t = self._target("t")
        with patch.dict(os.environ, {"ITF_LIVENESS_CHECK": "0"}):
            keep, settled = itf.drop_settled_targets([t], live)
        self.assertEqual(keep, [t])
        self.assertEqual(settled, [])

    def test_sentence_names_the_groups_and_carries_no_pr_column(self):
        # `| PR |` in anything rendered here would let _carry_forward_rows adopt
        # it as a dispatched row on a same-day re-run.
        sentence = itf.settled_sentence(
            [{"model": f"m{i}", "label": "l"} for i in range(6)]
        )
        self.assertIn("6 group(s) skipped", sentence)
        self.assertIn("`m0`", sentence)
        self.assertIn("+2 more", sentence)
        self.assertNotIn("| PR |", sentence)


class PriorFeedbackTest(unittest.TestCase):
    """Carrying the *reason* a previous attempt was rejected, not just the count.

    Both fixtures are real: transformers #48223 (a CHANGES_REQUESTED on an
    expectation rewrite) and #48322 (a working patch at the wrong altitude)."""

    C_48223 = {
        "pr": 48223,
        "kind": "inline",
        "author": "zucchini-nlp",
        "created_at": "2026-08-26T07:03:47Z",
        "state": "",
        "path": "tests/models/recurrent_gemma/test_modeling_recurrent_gemma.py",
        "line": 282,
        "body": (
            "the linked commit deleted a `partial_rotary_factor` so I think this "
            "is valid, we need to the fix model."
        ),
    }
    V_48223 = {
        "pr": 48223,
        "kind": "review",
        "author": "zucchini-nlp",
        "created_at": "2026-08-26T07:03:50Z",
        "state": "CHANGES_REQUESTED",
        "path": None,
        "line": None,
        "body": "",
    }
    RUN_SLOW = {
        "pr": 48223,
        "kind": "comment",
        "author": "a-maintainer",
        "created_at": "2026-08-22T23:00:00Z",
        "state": "",
        "path": None,
        "line": None,
        "body": "run-slow: recurrent_gemma",
    }

    def _target(self):
        return {
            "kind": "model_failures",
            "label": "2 integration tests regressed by commit 6034e90c7d1b",
            "model": "recurrent_gemma",
            "failure_mode": "output_mismatch",
            "cluster": None,
            "failures": [
                {
                    "test": (
                        "tests/models/recurrent_gemma/"
                        "test_modeling_recurrent_gemma.py::T::test_long_context"
                    ),
                    "model": "recurrent_gemma",
                    "gpu": "single-gpu",
                    "failure_mode": "output_mismatch",
                    "trace": "AssertionError: Lists differ",
                    "latest_trace": "AssertionError: Lists differ",
                    "days_seen": 3,
                }
            ],
        }

    # ── collection ──────────────────────────────────────────────────────────

    def test_reads_only_the_rejected_attempts(self):
        asked = []

        def fetch(repo, number, token):
            asked.append(number)
            return [{**self.C_48223, "pr": number}]

        prior = itf.PriorAttempts(open_pr=99, merged=(30,), rejected=(11, 12))
        items = itf.collect_prior_feedback(
            prior, "huggingface/transformers", None, max_prs=2, fetch=fetch
        )
        # Neither the open follow-up nor the merged fix is an objection to avoid.
        self.assertEqual(sorted(asked), [11, 12])
        self.assertEqual(len(items), 2)

    def test_caps_the_number_of_prs_read_newest_first(self):
        asked = []

        def fetch(repo, number, token):
            asked.append(number)
            return []

        prior = itf.PriorAttempts(rejected=(11, 12, 13))
        itf.collect_prior_feedback(
            prior, "huggingface/transformers", None, max_prs=2, fetch=fetch
        )
        self.assertEqual(asked, [13, 12])

    def test_a_group_with_no_rejection_makes_no_api_call(self):
        def fetch(repo, number, token):  # pragma: no cover - must not run
            raise AssertionError("fetched feedback for a group with no rejection")

        for prior in (None, itf.PriorAttempts(), itf.PriorAttempts(open_pr=7)):
            self.assertEqual(
                itf.collect_prior_feedback(
                    prior, "huggingface/transformers", None, fetch=fetch
                ),
                [],
            )

    def test_zero_disables(self):
        def fetch(repo, number, token):  # pragma: no cover - must not run
            raise AssertionError("fetched feedback with --feedback-prs 0")

        self.assertEqual(
            itf.collect_prior_feedback(
                itf.PriorAttempts(rejected=(11,)),
                "huggingface/transformers",
                None,
                max_prs=0,
                fetch=fetch,
            ),
            [],
        )

    # ── rendering ───────────────────────────────────────────────────────────

    def test_quotes_the_reviewer_and_names_the_pr(self):
        block = "\n".join(itf.prior_feedback_lines([self.V_48223, self.C_48223]))
        self.assertIn("/pull/48223", block)
        self.assertIn("zucchini-nlp", block)
        self.assertIn("partial_rotary_factor", block)
        self.assertIn("test_modeling_recurrent_gemma.py`:282", block)
        self.assertIn("requested changes", block)
        self.assertIn("untrusted", block)
        self.assertIn("Do not re-send", block)

    def test_a_bare_run_slow_comment_is_not_feedback(self):
        self.assertEqual(itf.prior_feedback_lines([self.RUN_SLOW]), [])
        self.assertTrue(itf.is_boilerplate_feedback("run-slow: recurrent_gemma"))
        self.assertTrue(itf.is_boilerplate_feedback("  run-slow: a\nrun-slow: b\n"))
        # A directive with an opinion attached is an opinion.
        self.assertFalse(itf.is_boilerplate_feedback("run-slow: a, and also fix X"))
        self.assertTrue(itf.is_boilerplate_feedback("run-slow: gemma, llama"))

    def test_an_empty_changes_requested_still_counts(self):
        """The verdict and the sentence are separate GitHub objects."""
        self.assertTrue(itf.carries_reviewer_signal(self.V_48223))
        self.assertFalse(itf.carries_reviewer_signal(self.RUN_SLOW))
        block = "\n".join(itf.prior_feedback_lines([self.V_48223]))
        self.assertIn("requested changes", block)

    def test_a_long_comment_is_truncated_not_dropped(self):
        long = {**self.C_48223, "body": "x" * 2000}
        block = "\n".join(itf.prior_feedback_lines([long]))
        self.assertIn("[…]", block)
        self.assertLess(len(block), 1200)

    def test_nothing_to_say_renders_nothing(self):
        self.assertEqual(itf.prior_feedback_lines([]), [])

    # ── attach + context ────────────────────────────────────────────────────

    def test_the_failure_report_carries_the_objection(self):
        t = self._target()
        priors = {itf.target_fingerprint(t): itf.PriorAttempts(rejected=(48223,))}
        out = itf.attach_prior_feedback(
            [t],
            priors,
            "huggingface/transformers",
            None,
            fetch=lambda repo, number, token: [self.C_48223, self.V_48223],
        )
        self.assertEqual(len(out[0]["prior_feedback"]), 2)
        context = itf.render_serge_context(out, ["2026-08-24", "2026-08-26"])
        self.assertIn("partial_rotary_factor", context)
        # The objection has to arrive before the tracebacks, or it reads as a
        # footnote to a wall of trace.
        self.assertLess(
            context.index("partial_rotary_factor"), context.index("Failing tests")
        )

    def test_attaching_cannot_change_a_group_identity(self):
        t = self._target()
        before = itf.target_fingerprint(t)
        priors = {before: itf.PriorAttempts(rejected=(48223,))}
        out = itf.attach_prior_feedback(
            [t],
            priors,
            "huggingface/transformers",
            None,
            fetch=lambda repo, number, token: [self.C_48223],
        )
        self.assertEqual(itf.target_fingerprint(out[0]), before)

    def test_a_group_with_only_boilerplate_is_left_alone(self):
        t = self._target()
        priors = {itf.target_fingerprint(t): itf.PriorAttempts(rejected=(48223,))}
        out = itf.attach_prior_feedback(
            [t],
            priors,
            "huggingface/transformers",
            None,
            fetch=lambda repo, number, token: [self.RUN_SLOW],
        )
        self.assertNotIn("prior_feedback", out[0])

    def test_no_priors_is_a_no_op(self):
        t = self._target()
        self.assertEqual(
            itf.attach_prior_feedback([t], None, "huggingface/transformers", None), [t]
        )


class ListPrReviewFeedbackTest(unittest.TestCase):
    """The three-endpoint merge in ``github_api.list_pr_review_feedback``.

    The objection, the verdict, and the close reason live in three different
    GitHub collections, and none of them is a superset of the others."""

    INLINE = [
        {
            "user": {"login": "sergereview[bot]"},
            "created_at": "2026-08-25T22:32:00Z",
            "body": "my own finding",
            "path": "a.py",
            "line": 1,
        },
        {
            "user": {"login": "zucchini-nlp"},
            "created_at": "2026-08-26T07:32:53Z",
            "body": "why don't we pop in `EdgeTamModel` right in the beginning",
            "path": "src/transformers/models/edgetam/modeling_edgetam.py",
            "line": 482,
        },
    ]
    REVIEWS = [
        {
            "user": {"login": "zucchini-nlp"},
            "state": "commented",
            "submitted_at": "2026-08-26T07:32:53Z",
            "body": "",
        },
        {
            "user": {"login": "zucchini-nlp"},
            "state": "CHANGES_REQUESTED",
            "submitted_at": "2026-08-26T07:32:55Z",
            "body": "",
        },
    ]
    CONVO = [
        {
            "user": {"login": "github-actions[bot]"},
            "created_at": "2026-08-25T22:33:47Z",
            "body": "run-slow: edgetam",
        },
        {
            "user": {"login": "HuggingFaceDocBuilderDev"},
            "created_at": "2026-08-25T22:42:22Z",
            "body": "The docs for this PR live here",
        },
        {
            "user": {"login": "a-human"},
            "created_at": "2026-08-26T09:00:00Z",
            "body": "closing, the fix is wrong",
        },
    ]

    def _fake(self):
        class _Resp:
            def __init__(self, data):
                self._d = data

            def read(self):
                return self._d

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def urlopen(req, timeout=None):
            url = req.full_url
            if "/pulls/48322/comments" in url:
                payload = self.INLINE
            elif "/pulls/48322/reviews" in url:
                payload = self.REVIEWS
            elif "/issues/48322/comments" in url:
                payload = self.CONVO
            else:  # pragma: no cover - an unexpected endpoint is a bug
                raise AssertionError(f"unexpected GET {url}")
            return _Resp(json.dumps(payload).encode())

        return urlopen

    def _fetch(self, **kw):
        with patch.object(gh_api.urllib.request, "urlopen", self._fake()):
            return gh_api.list_pr_review_feedback(
                "huggingface/transformers", 48322, None, **kw
            )

    def test_bots_are_dropped_and_humans_kept(self):
        items = self._fetch()
        authors = {i["author"] for i in items}
        self.assertEqual(authors, {"zucchini-nlp", "a-human"})

    def test_newest_first(self):
        items = self._fetch()
        stamps = [i["created_at"] for i in items]
        self.assertEqual(stamps, sorted(stamps, reverse=True))

    def test_an_empty_bodied_review_is_kept_only_when_it_blocks(self):
        states = [i["state"] for i in self._fetch() if i["kind"] == "review"]
        self.assertEqual(states, ["CHANGES_REQUESTED"])

    def test_the_inline_comment_keeps_its_location(self):
        inline = [i for i in self._fetch() if i["kind"] == "inline"]
        self.assertEqual(len(inline), 1)
        self.assertEqual(inline[0]["line"], 482)
        self.assertTrue(inline[0]["path"].endswith("modeling_edgetam.py"))

    def test_capped(self):
        self.assertEqual(len(self._fetch(max_items=2)), 2)

    def test_an_api_error_degrades_to_fewer_items(self):
        def boom(req, timeout=None):
            raise urllib.error.URLError("no network")

        with patch.object(gh_api.urllib.request, "urlopen", boom):
            self.assertEqual(
                gh_api.list_pr_review_feedback("huggingface/transformers", 1, None), []
            )

    def test_a_malformed_repo_makes_no_call(self):
        def boom(req, timeout=None):  # pragma: no cover - must not run
            raise AssertionError("called GitHub for a repo with no owner")

        with patch.object(gh_api.urllib.request, "urlopen", boom):
            self.assertEqual(
                gh_api.list_pr_review_feedback("transformers", 1, None), []
            )


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
            accepted, failed, _job_ids, _errors = itf.dispatch_targets(
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
            accepted, failed, _job_ids, _errors = itf.dispatch_targets(
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
            accepted, failed, _job_ids, _errors = itf.dispatch_targets(
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
            accepted, failed, job_ids, _errors = itf.dispatch_targets(
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

    def test_bounded_dispatch_mints_a_fresh_bearer_for_the_retry_post(self):
        """A GitHub Actions OIDC token lives minutes; the 429 backoff sleeps for
        minutes before the retry POST. Dispatching with the bearer this loop
        started with is what lost the 2026-08-18 `deepseek_vl` group to
        ``401 ... Signature has expired``, so every POST re-mints."""
        target = self._targets()[:1]
        bearers = []
        minted = itertools.count(1)

        def fake_dispatch(serge_url, token, payload, timeout=240):
            bearers.append(token)
            job_id = f"job{len(bearers)}"
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
            patch.object(
                itf,
                "mint_serge_oidc_token",
                side_effect=lambda: f"fresh{next(minted)}",
            ),
            patch.object(itf.random, "uniform", return_value=0),
            patch.object(itf.time, "sleep", lambda _s: None),
        ):
            accepted, failed, _job_ids, _errors = itf.dispatch_targets(
                target,
                repo="o/r",
                base_ref="main",
                serge_url="http://s",
                token="stale",
                window=["2026-06-19"],
                timeout=10,
                github_token=None,
                serge_concurrency=1,
                retry_attempts=1,
                retry_base_seconds=1,
                poll_seconds=0,
            )

        self.assertEqual((accepted, failed), (1, 0))
        self.assertEqual(len(bearers), 2)
        self.assertNotIn("stale", bearers)

    def test_a_failed_post_is_reported_against_its_fingerprint(self):
        """`failed` counts; it does not say *which* group or *why*. Reconcile
        needs both to mark the group terminal instead of leaving it pending."""
        target = self._targets()[:1]
        err = sd.SergeDispatchError(
            "Serge POST /tasks failed: 401 Unauthorized\n"
            '{"detail": "oidc_verification_failed: invalid OIDC token: '
            'Signature has expired"}',
            status=401,
        )
        for concurrency, attempts in ((0, 0), (1, 1)):
            with self.subTest(bounded=bool(concurrency or attempts)):
                with (
                    patch.object(itf, "list_open_pulls", return_value=[]),
                    patch.object(itf, "dispatch_to_serge", side_effect=err),
                    patch.object(itf, "mint_serge_oidc_token", return_value=None),
                    patch.object(itf.time, "sleep", lambda _s: None),
                ):
                    accepted, failed, job_ids, errors = itf.dispatch_targets(
                        target,
                        repo="o/r",
                        base_ref="main",
                        serge_url="http://s",
                        token="tok",
                        window=["2026-06-19"],
                        timeout=10,
                        github_token=None,
                        serge_concurrency=concurrency,
                        retry_attempts=attempts,
                        retry_base_seconds=1,
                        poll_seconds=0,
                    )

                fp = itf.target_fingerprint(target[0])
                self.assertEqual((accepted, failed), (0, 1))
                self.assertEqual(job_ids, {})
                self.assertIn("Signature has expired", errors[fp])

    def test_bounded_dispatch_keeps_its_bearer_outside_actions(self):
        """Outside Actions there is no token service to ask, so
        ``mint_serge_oidc_token`` returns ``None`` and the initial bearer must
        survive rather than being replaced by nothing."""
        target = self._targets()[:1]
        bearers = []

        def fake_dispatch(serge_url, token, payload, timeout=240):
            bearers.append(token)
            return {"id": "job1", "url": "/tasks/o/r/job1"}

        with (
            patch.object(itf, "list_open_pulls", return_value=[]),
            patch.object(itf, "dispatch_to_serge", side_effect=fake_dispatch),
            patch.object(itf, "poll_serge_task", return_value={"status": "published"}),
            patch.object(itf, "mint_serge_oidc_token", return_value=None),
            patch.object(itf.time, "sleep", lambda _s: None),
        ):
            itf.dispatch_targets(
                target,
                repo="o/r",
                base_ref="main",
                serge_url="http://s",
                token="tok",
                window=["2026-06-19"],
                timeout=10,
                github_token=None,
                serge_concurrency=1,
                retry_attempts=0,
                poll_seconds=0,
            )

        self.assertEqual(bearers, ["tok"])


class DispatchFailureReasonTest(unittest.TestCase):
    def test_lifts_the_detail_out_of_serges_json_body(self):
        err = sd.SergeDispatchError(
            "Serge POST /tasks failed: 401 Unauthorized\n"
            '{"detail": "oidc_verification_failed: invalid OIDC token: '
            'Signature has expired", "serge": {"version": "0.1.0"}}',
            status=401,
        )
        reason = itf.dispatch_failure_reason(err)
        self.assertIn("401 Unauthorized", reason)
        self.assertIn("Signature has expired", reason)
        self.assertNotIn("\n", reason)

    def test_keeps_a_non_json_body_as_text(self):
        err = sd.SergeDispatchError("Serge POST /tasks failed: 502\n<html>bad</html>")
        self.assertEqual(
            itf.dispatch_failure_reason(err),
            "Serge POST /tasks failed: 502 — <html>bad</html>",
        )

    def test_a_single_line_error_needs_no_detail(self):
        err = sd.SergeDispatchError(
            "could not reach Serge at http://s/tasks: timed out"
        )
        self.assertEqual(
            itf.dispatch_failure_reason(err),
            "could not reach Serge at http://s/tasks: timed out",
        )


class DispatchTokenReplayTest(unittest.TestCase):
    """``dispatch_to_serge`` re-mints and replays once on a 401 — the safety net
    under the per-POST re-mint, and the only protection the other callers
    (``dashboard_failure_triage``) get."""

    def _expired(self):
        return sd.SergeDispatchError(
            "Serge POST /tasks failed: 401 Unauthorized\n"
            '{"detail": "oidc_verification_failed: invalid OIDC token: '
            'Signature has expired"}',
            status=401,
        )

    def test_replays_once_with_a_freshly_minted_bearer(self):
        seen = []

        def fake_post(serge_url, token, payload, timeout):
            seen.append(token)
            if token == "stale":
                raise self._expired()
            return {"id": "job1"}

        with (
            patch.object(sd, "_post_task", side_effect=fake_post),
            patch.object(sd, "mint_serge_oidc_token", return_value="fresh"),
        ):
            resp = sd.dispatch_to_serge("http://s", "stale", {})

        self.assertEqual(resp, {"id": "job1"})
        self.assertEqual(seen, ["stale", "fresh"])

    def test_a_401_that_re_mints_to_the_same_token_is_not_replayed(self):
        """A 401 holding a *valid* bearer is a real authorization failure (wrong
        audience, untrusted repo). Replaying it would only double the load."""
        seen = []

        def fake_post(serge_url, token, payload, timeout):
            seen.append(token)
            raise self._expired()

        with (
            patch.object(sd, "_post_task", side_effect=fake_post),
            patch.object(sd, "mint_serge_oidc_token", return_value="tok"),
        ):
            with self.assertRaises(sd.SergeDispatchError):
                sd.dispatch_to_serge("http://s", "tok", {})

        self.assertEqual(seen, ["tok"])

    def test_a_non_401_failure_is_not_replayed(self):
        seen = []

        def fake_post(serge_url, token, payload, timeout):
            seen.append(token)
            raise sd.SergeDispatchError("Serge POST /tasks failed: 422", status=422)

        with (
            patch.object(sd, "_post_task", side_effect=fake_post),
            patch.object(sd, "mint_serge_oidc_token", return_value="fresh"),
        ):
            with self.assertRaises(sd.SergeDispatchError):
                sd.dispatch_to_serge("http://s", "tok", {})

        self.assertEqual(seen, ["tok"])

    def test_an_unreachable_serge_carries_no_status(self):
        """``URLError`` means Serge never answered, so there is no status to
        confuse with a 401."""
        err = sd.SergeDispatchError("could not reach Serge at http://s/tasks: nope")
        self.assertIsNone(err.status)


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
            accepted, failed, job_ids, _errors = itf.dispatch_targets(
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

    def test_dispatch_failure_lands_in_the_table_and_the_recap(self):
        """A group whose POST never landed has no job to poll. Left alone it sits
        `(pending)` for the whole reconcile window and then reads as "Serge is
        still working" — when Serge never received it. Seeded as terminal, the
        dispatcher's own error becomes the group's reason."""
        targets = [self._target("g1", "deepseek_vl")]
        fp = itf.target_fingerprint(targets[0])
        patched = []

        with (
            patch.object(itf, "list_open_pulls", return_value=[]),
            patch.object(
                itf,
                "update_issue_body",
                side_effect=lambda r, n, body, t: patched.append(body) or True,
            ),
            patch.object(itf.time, "sleep", lambda _s: None),
        ):
            itf.reconcile_tracking_issue(
                targets,
                repo="o/r",
                window=["2026-08-18"],
                run_key="2026-08-18",
                issue_number=48050,
                github_token="tok",
                timeout_seconds=300,
                poll_seconds=1,
                dispatch_errors={
                    fp: "Serge POST /tasks failed: 401 Unauthorized — "
                    "oidc_verification_failed: invalid OIDC token: Signature has expired"
                },
            )

        self.assertEqual(len(patched), 1)
        body = patched[-1]
        row = next(line for line in body.splitlines() if "| `deepseek_vl` |" in line)
        self.assertIn("⚠️ task failed", row)
        self.assertNotIn("(pending)", row)
        self.assertIn("## Outcome recap", body)
        self.assertIn("Signature has expired", body)

    def test_recap_rows_carry_forward_with_their_marker(self):
        """The table carries a resolved group's 🚫/⚠️ cell into a later run; the
        recap is re-rendered from this run's polls only. Without carrying the
        recap row too, the marker survives and the reason behind it does not."""
        prior = "\n".join(
            [
                "| Group | Reason | LLM | Tokens (in / out) | Failing tests |",
                "| --- | --- | --- | --- | --- |",
                "| `deepseek_vl` | dispatch failed: 401 expired bearer | `k2` | 1 / 2 | [t](u) |",
                "| `kimi_k25` (regressed by PR #47573) | [not_fixed] still red | `k2` | 3 / 4 | [t](u) |",
                "",
            ]
        )
        targets = [self._target("g1", "kimi_k25")]
        rows = itf._carry_forward_recap_rows(prior, targets)

        # deepseek_vl is not in this run, so its reason has to survive; kimi_k25
        # is, and will be re-rendered from this run's own poll details.
        self.assertEqual(len(rows), 1)
        self.assertIn("deepseek_vl", rows[0])

    def test_carried_recap_rows_render_without_a_current_recap(self):
        targets = [self._target("g1", "alpha")]
        carried = "| `deepseek_vl` | dispatch failed: 401 | `k2` | 1 / 2 | — |"
        body = itf.render_tracking_issue_body(
            targets,
            ["2026-08-18"],
            "2026-08-18",
            existing_prs={},
            carry_recap_rows=[carried],
        )
        self.assertIn("## Outcome recap", body)
        self.assertIn(carried, body)

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

    def test_mismatch_carries_the_maintainer_convention(self):
        """The rule that decides the patch is in no transformers doc.

        `docs/source/en/testing.md` explains what `Expectations` *is* — the SM
        keys, the fallback — and never says what to do when a value goes stale.
        The maintainers' answer is only visible in their diffs: PR #48198
        (`table_transformer`, ydshieh, merged 2026-08-21) demoted the existing
        `("cuda", None)` entry to `(None, None)` and ADDED `("cuda", 8)`. serge
        cannot read that from the checkout, so it is stated here as fact.
        """
        text = itf.build_instruction(_target("output_mismatch"))
        self.assertIn("#48198", text)
        self.assertIn("Add a device key", text)
        self.assertIn("do not overwrite the value that is there", text)
        # The shape itself, so the agent copies a form rather than a slogan.
        self.assertIn("(None, None)", text)
        self.assertIn('("cuda", 8)', text)
        # A plain literal has to become an Expectations block, not be replaced.
        self.assertIn("convert it", text)

    def test_mismatch_calibrates_drift_against_a_real_accepted_one(self):
        """ "Far larger than a precedent would cover" has no number in it.

        Two groups on 2026-08-31 (`ministral` ~8.5% relative, `smollm3` ~9.36
        absolute) were correctly refused. The convention above, handed over
        WITHOUT this calibration, would have turned both into recorded
        expectations — the exact failure it is meant to prevent. The accepted
        drift in #48198 was ~0.5%, so the two live an order of magnitude apart.
        """
        text = itf.build_instruction(_target("output_mismatch"))
        self.assertIn("~0.5% relative", text)
        self.assertIn("one or two orders of magnitude past that is NOT drift", text)
        self.assertIn("~9.36 absolute", text)
        # Refusing has to be named as a success, or the agent optimises for a PR.
        self.assertIn("Producing no patch is a correct outcome", text)

    def test_the_convention_stays_out_of_a_crash_group(self):
        """A crash never got as far as comparing values, so "add a device key"
        is the last thing it should hear."""
        text = itf.build_instruction(_target("cuda_runtime", exc="RuntimeError"))
        self.assertNotIn("#48198", text)
        self.assertNotIn("Add a device key", text)

    def test_the_trunk_forbids_fixing_ci_in_a_shared_loading_path(self):
        """serge patched `conversion_mapping.py` to make one runner pass.

        That file is reached by every `from_pretrained` call, so the flag it set
        would slow loading for every user of the model, for ever, to fix a CI
        runner condition (transformers#48426). The rule belongs in the TRUNK, not
        in the OOM block: that group was a bad-commit cluster whose CI message
        said "issues during automatic conversion", so it was classified `other`
        and `instruction_addendum` gave it an EMPTY addendum. Only the trunk
        reaches a group like that.
        """
        text = itf.build_instruction(None)  # trunk alone — what that group got
        self.assertIn("Blast radius", text)
        self.assertIn("conversion_mapping.py", text)
        self.assertIn("core_model_loading.py", text)
        self.assertIn("that is a signal you have the wrong cause", text)

    def test_load_oom_bans_force_cpu_and_gives_the_threshold(self):
        """The flag's author: it is for EXTREME single-parameter cases only —
        qwen4's ngram embedding, ~50B params in one tensor. serge copied the
        mechanism from that one precedent without the magnitude that justifies
        it, which is the same failure as taking the `Expectations` device-key
        convention without its drift threshold."""
        text = itf.instruction_addendum(_oom_target(_OOM_LOAD_TRACE))
        self.assertIn("force_cpu=True", text)
        self.assertIn("ONE tensor", text)
        self.assertIn("no matter how closely the traceback resembles", text)

    def test_load_oom_names_the_test_side_causes_first(self):
        """Cyril's actual diagnosis: the test's own `device_map` plus its
        `cap_psutil_cpu_memory(...)` budget. Both are test-side and fixable, and
        they must be checked before anything in a shared loading path."""
        text = itf.instruction_addendum(_oom_target(_OOM_LOAD_TRACE))
        self.assertIn("cap_psutil_cpu_memory", text)
        self.assertIn("HF_DEACTIVATE_ASYNC_LOAD=true", text)
        # …and a transient must not be patched at all.
        self.assertIn("A transient is not something to patch", text)
        self.assertLess(
            text.index("the test's own placement"), text.index("async loading")
        )

    _CONV_TRACE = (
        "RuntimeError: We encountered some issues during automatic conversion of "
        "the weights. For details look at the `CONVERSION` entries of the above "
        "report!"
    )

    def _conv_target(self, mode="other", kind="cluster", exc=None):
        t = {
            "kind": kind,
            "label": "1 integration tests regressed by commit bd9509355c8a",
            "model": "phimoe",
            "failure_mode": mode,
            "failures": [
                {
                    "test": "tests/models/phimoe/test_modeling_phimoe.py::PhimoeIntegrationTest::test_phimoe_instruct_generation",
                    "gpu": "multi-gpu",
                    "failure_mode": mode,
                    "latest_trace": self._CONV_TRACE,
                }
            ],
        }
        if exc:
            t["terminal_exc"] = exc
        return t

    def test_a_conversion_failure_reaches_the_load_guidance(self):
        """The phimoe group behind transformers#48426 got an EMPTY addendum.

        Its CI message is a wrapper — "issues during automatic conversion of the
        weights" — which names no cause, so `classify` filed it `other`; and a
        bad-commit cluster gets no per-category block unless every failure is
        already OOM-labelled. With nothing to go on it patched
        `conversion_mapping.py`, a production loading path, to make one runner
        pass. A crash inside the conversion step is a loading problem.
        """
        text = itf.instruction_addendum(self._conv_target())
        self.assertIn("size the checkpoint", text)
        self.assertIn("max_memory", text)
        self.assertIn("force_cpu=True", text)

    def test_it_also_covers_a_plain_model_group(self):
        text = itf.instruction_addendum(self._conv_target(kind="model_failures"))
        self.assertIn("size the checkpoint", text)

    def test_it_does_not_hijack_a_mode_with_its_own_block(self):
        """`core_model_loading.py` appears in plenty of tracebacks. A group whose
        mode has specific guidance must keep it."""
        t = self._conv_target(mode="output_mismatch", kind="model_failures")
        t["failures"][0]["latest_trace"] = "core_model_loading.py:1727 in convert"
        self.assertIn("PLAUSIBLE VARIANT", itf.instruction_addendum(t))

    def test_an_assertion_still_wins_over_the_conversion_route(self):
        text = itf.instruction_addendum(
            self._conv_target(kind="model_failures", exc="AssertionError")
        )
        self.assertIn("PLAUSIBLE VARIANT", text)
        self.assertNotIn("size the checkpoint", text)

    def test_a_plain_crash_still_gets_crash_guidance(self):
        t = self._conv_target(mode="cuda_runtime", kind="model_failures")
        t["failures"][0]["latest_trace"] = "RuntimeError: expected device cuda:0"
        self.assertIn("library/model bug until", itf.instruction_addendum(t))

    def test_routing_does_not_change_the_group_identity(self):
        """Routed at guidance time on purpose: `failure_mode` is in
        `target_fingerprint`'s basis, so relabelling the mode would orphan the
        group's open PR and re-dispatch it as new work."""
        t = self._conv_target(kind="model_failures")
        before = itf.target_fingerprint(t)
        itf.instruction_addendum(t)
        self.assertEqual(itf.target_fingerprint(t), before)
        self.assertEqual(t["failure_mode"], "other")  # untouched

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


class RegressionReviewerTest(unittest.TestCase):
    """When CI's bisect pinned the regression to a commit, its author is the one
    person who knows what that change meant to do — so the fix PR asks them for
    the review instead of waiting to be noticed."""

    def _clustered(self, author, pr_number=46556):
        target = _target("output_mismatch", model="florence2")
        target["cluster"] = {
            "bad_commit": "254e4b6e7cd9",
            "pr_number": pr_number,
            "author": author,
            "merged_by": "someone-else",
        }
        return target

    def test_payload_requests_the_blamed_commits_author(self):
        payload = itf.build_task_payload(
            "huggingface/transformers",
            "main",
            "ctx",
            "title",
            fingerprint="f" * 64,
            target=self._clustered("ArthurZucker"),
            grafana_url="",
        )
        self.assertEqual(payload["reviewers"], ["ArthurZucker"])

    def test_unattributed_groups_request_nobody(self):
        # A `model_failures` group has no cluster, and an unconverged bisect
        # leaves the author null — the PR then keeps whatever the repo's own
        # reviewer routing decides.
        for target in (
            _target("output_mismatch"),
            self._clustered(None),
            self._clustered(""),
        ):
            payload = itf.build_task_payload(
                "huggingface/transformers",
                "main",
                "ctx",
                "title",
                fingerprint="f" * 64,
                target=target,
                grafana_url="",
            )
            self.assertNotIn("reviewers", payload)

    def test_bot_authors_are_dropped(self):
        # A bot cannot review, and GitHub 422s the whole request when one login
        # is invalid — which would take any real reviewer down with it.
        self.assertEqual(
            itf.regression_reviewers(self._clustered("dependabot[bot]")), []
        )

    def test_no_target_is_safe(self):
        self.assertEqual(itf.regression_reviewers(None), [])


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


# muse_glimmer, 2026-08-24: a 30B checkpoint pinned to one 24 GiB A10 by
# `device_map=torch_device`. The numbers alone are retention-shaped -- a 254 MiB
# request on a card PyTorch already fills -- but the frame is `from_pretrained`
# still materializing weights, so there is nothing retained to free.
_OOM_LOAD_TRACE = (
    "src/transformers/core_model_loading.py:1219: in _materialize_copy\n"
    "    tensor = tensor.to(device=device, dtype=dtype)\n"
    "E   torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 254.00 MiB. "
    "GPU 0 has a total capacity of 22.30 GiB of which 18.00 MiB is free. Process "
    "12345 has 22.28 GiB memory in use. Of the allocated memory 21.86 GiB is "
    "allocated by PyTorch, with 22.00 MiB allocated in private pools (e.g., CUDA "
    "Graphs)."
)


def _oom_cluster(*traces, model="muse_glimmer"):
    """A bad-commit cluster whose member failures carry the given traces."""
    return {
        "kind": "cluster",
        "label": f"{len(traces)} integration tests regressed by commit fe95f5423d65",
        "failures": [
            _failure(
                model, "single", f"tests/models/{model}/t.py::T::t{i}", t, mode="OOM"
            )
            for i, t in enumerate(traces)
        ],
        "cluster": {"author": "someone", "pr_number": 47867},
        "model": None,
        "failure_mode": None,
    }


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


class UnresolvedGroupEvidenceLinksTest(unittest.TestCase):
    """The groups that end WITHOUT a PR are the ones a human picks up, so the
    issue must link their failing tests rather than only naming them
    (transformers#48050: `gpt_oss` and `phimoe` were classified as environment
    issues with no way to reach the failure)."""

    GRAFANA = "https://grafana.example/d/abc/tests"

    def test_outcome_recap_links_each_failing_test(self):
        target = _target("import_or_config", model="gpt_oss", n=2)
        fp = itf.target_fingerprint(target)
        body = itf.render_tracking_issue_body(
            [target],
            ["2026-08-18"],
            "2026-08-18",
            existing_prs={},
            statuses={fp: "no_fix"},
            details={
                fp: {
                    "reason": "[reproduced] ENVIRONMENT issue: `kernels` not installed",
                    "prompt_tokens": 1,
                    "completion_tokens": 2,
                }
            },
            grafana_url=self.GRAFANA,
        )
        recap = body.split("## Outcome recap", 1)[1]
        # One link per failing test, labelled by the node-id's last segment.
        self.assertIn("[t0](https://grafana.example/", recap)
        self.assertIn("[t1](https://grafana.example/", recap)
        # The node-id itself is what the link resolves — not a bare dashboard URL.
        self.assertIn("gpt_oss", recap)

    def test_deferred_rows_and_oom_models_link_too(self):
        dep = _target("import_or_config", exc="ModuleNotFoundError", model="dep")
        oom = _target("OOM", exc="OutOfMemoryError", model="big")
        _, deferred = itf.partition_targets([oom, dep])
        section = "\n".join(itf._render_deferred_section(deferred, self.GRAFANA))
        # The collapsed OOM sentence links the model name…
        self.assertIn("[`big`](https://grafana.example/", section)
        # …and the table row links the test.
        self.assertIn("[t0](https://grafana.example/", section)
        self.assertIn("| Failing tests |", section)

    def test_links_are_capped_with_a_remainder(self):
        target = _target("output_mismatch", model="m", n=5)
        cell = itf._evidence_cell(target, self.GRAFANA)
        self.assertEqual(cell.count("https://grafana.example/"), 3)
        self.assertTrue(cell.endswith("+2 more"))

    def test_unconfigured_grafana_renders_no_links(self):
        target = _target("output_mismatch", model="m", n=2)
        self.assertEqual(itf._evidence_cell(target, ""), "—")
        # …and the OOM sentence keeps its plain model names rather than an
        # empty link, so an unconfigured deployment reads exactly as before.
        oom = [_target("OOM", exc="OutOfMemoryError", model="big")]
        self.assertIn("`big` (1)", itf._oom_sentence(oom))
        self.assertNotIn("](", itf._oom_sentence(oom))

    def test_evidence_columns_do_not_confuse_carry_forward(self):
        # _carry_forward_rows keys off a `| PR |` header; the recap/deferred
        # tables gaining a column must not start matching it.
        target = _target("import_or_config", model="gpt_oss")
        oom = _target("OOM", exc="OutOfMemoryError", model="big")
        dispatch, deferred = itf.partition_targets([oom, target])
        fp = itf.target_fingerprint(dispatch[0])
        body = itf.render_tracking_issue_body(
            dispatch,
            ["2026-08-18"],
            "2026-08-18",
            existing_prs={},
            statuses={fp: "no_fix"},
            details={fp: {"reason": "env", "prompt_tokens": 1, "completion_tokens": 2}},
            deferred=deferred,
            grafana_url=self.GRAFANA,
        )
        carried = itf._carry_forward_rows(body, [])
        self.assertFalse(any("grafana.example" in row for row in carried))

    def test_grafana_url_defaults_to_the_environment(self):
        target = _target("output_mismatch", model="m")
        fp = itf.target_fingerprint(target)
        with patch.dict(os.environ, {"ITF_GRAFANA_URL": self.GRAFANA}):
            body = itf.render_tracking_issue_body(
                [target],
                ["2026-08-18"],
                "2026-08-18",
                existing_prs={},
                statuses={fp: "no_fix"},
                details={
                    fp: {"reason": "r", "prompt_tokens": 1, "completion_tokens": 2}
                },
            )
        self.assertIn("grafana.example", body)


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


class OomLoadShapeTest(unittest.TestCase):
    """A checkpoint too big for the card fills it one tensor at a time, so the
    failing request is small and the card is full -- byte-for-byte a retention
    bug. Only the frame separates them, and getting it wrong sends the agent
    after a `tearDown` that cannot free anything."""

    def test_an_oom_inside_the_loading_path_is_load_shaped_not_retention(self):
        shape, nums = itf.oom_shape(_OOM_LOAD_TRACE)
        self.assertEqual(shape, itf.OOM_LOAD)
        # The arithmetic on its own says retention: a trivial request on a card
        # PyTorch already holds 98% of. This is exactly the trap.
        self.assertLess(nums["want"], itf._OOM_TRIVIAL_WANT_GIB)
        self.assertGreater(nums["held"], nums["capacity"] * itf._OOM_HELD_SHARE)

    def test_the_same_numbers_off_the_loading_path_stay_retention(self):
        self.assertEqual(itf.oom_shape(_OOM_RETENTION_TRACE)[0], itf.OOM_RETENTION)

    def test_one_allocation_over_the_whole_card_is_still_capacity(self):
        # Capacity wins even on the loading path: no device map makes a single
        # allocation larger than the card fit.
        trace = (
            "core_model_loading.py:1219: in _materialize_copy\n" + _OOM_CAPACITY_TRACE
        )
        self.assertEqual(itf.oom_shape(trace)[0], itf.OOM_CAPACITY)

    def test_an_unparseable_load_oom_is_still_recognised(self):
        trace = "core_model_loading.py: torch.OutOfMemoryError: CUDA out of memory."
        self.assertEqual(itf.oom_shape(trace), (itf.OOM_LOAD, {}))


class OomLoadGuidanceTest(unittest.TestCase):
    def test_a_load_oom_is_pointed_at_the_device_map_not_a_teardown(self):
        text = itf.instruction_addendum(_oom_target(_OOM_LOAD_TRACE))
        self.assertIn('device_map="auto"', text)
        # The lazy idiom the maintainers asked for, not an eager loading setUpClass.
        self.assertIn("def get_model(cls):", text)
        self.assertIn("cls.model = None", text)
        # The retention block's fix must not be what this group is told to do.
        self.assertNotIn("def tearDown(self)", text)

    def test_a_load_oom_permits_trimming_tokens_but_not_the_assertion(self):
        text = itf.instruction_addendum(_oom_target(_OOM_LOAD_TRACE))
        self.assertIn("max_new_tokens", text)
        self.assertIn("assertion must still hold unchanged", text)

    def test_a_load_oom_dispatches_instead_of_waiting_for_a_bigger_runner(self):
        self.assertEqual(itf.env_only_reason(_oom_target(_OOM_LOAD_TRACE)), "")

    def test_a_pure_capacity_group_is_still_deferred(self):
        self.assertIn(
            "runner capacity", itf.env_only_reason(_oom_target(_OOM_CAPACITY_TRACE))
        )

    def test_a_retention_group_keeps_its_teardown_guidance(self):
        text = itf.instruction_addendum(_oom_target(_OOM_RETENTION_TRACE))
        self.assertIn("def tearDown(self)", text)


class OomClusterGuidanceTest(unittest.TestCase):
    """The muse_glimmer regression: a bad-commit cluster that was uniformly OOM
    got the trunk alone, so the agent never saw any memory guidance."""

    def test_an_all_oom_cluster_now_gets_the_memory_guidance(self):
        text = itf.instruction_addendum(_oom_cluster(_OOM_LOAD_TRACE, _OOM_LOAD_TRACE))
        self.assertIn('device_map="auto"', text)

    def test_a_mixed_mode_cluster_still_gets_the_trunk_alone(self):
        cluster = _oom_cluster(_OOM_LOAD_TRACE)
        cluster["failures"].append(
            _failure("muse_glimmer", "single", "t.py::T::x", "E   AssertionError: nope")
        )
        self.assertEqual(itf.instruction_addendum(cluster), "")

    def test_an_empty_cluster_does_not_claim_to_be_oom(self):
        cluster = _oom_cluster()
        self.assertEqual(itf.instruction_addendum(cluster), "")


class ModularSourceParseTests(unittest.TestCase):
    """Items 5 + 6: the facts read out of one `modular_*.py`."""

    SRC = (
        "from ..auto import CONFIG_MAPPING, AutoConfig\n"
        "from ..sam2.modeling_sam2 import Sam2VisionModel, Sam2Attention\n"
        "from ..sam2.configuration_sam2 import Sam2Config\n"
        "from .configuration_edgetam import EdgeTamVisionConfig\n"
        "\n"
        "class EdgeTamVisionModel(Sam2VisionModel):\n"
        "    pass\n"
        "\n"
        "class EdgeTamAttention(Sam2Attention, nn.Module):\n"
        "    pass\n"
        "\n"
        "class EdgeTamConfig:\n"
        "    pass\n"
        "\n"
        "    class NotTopLevel(Nope):\n"
        "        pass\n"
    )

    def test_defined_classes_and_bases(self) -> None:
        info = itf.parse_modular_source(self.SRC)
        self.assertEqual(
            info["defined"],
            [
                ("EdgeTamVisionModel", ["Sam2VisionModel"]),
                ("EdgeTamAttention", ["Sam2Attention", "nn.Module"]),
                ("EdgeTamConfig", []),
            ],
        )

    def test_indented_class_is_not_module_level(self) -> None:
        names = [n for n, _ in itf.parse_modular_source(self.SRC)["defined"]]
        self.assertNotIn("NotTopLevel", names)

    def test_parents_come_from_two_dot_imports_only(self) -> None:
        parents = itf.parse_modular_source(self.SRC)["parents"]
        # `..sam2` is lineage; `.configuration_edgetam` is this model's own
        # package and says nothing about a parent.
        self.assertEqual(list(parents), ["sam2"])
        self.assertIn("Sam2VisionModel", parents["sam2"])
        self.assertIn("Sam2Config", parents["sam2"])

    def test_auto_registry_is_not_lineage(self) -> None:
        # Naming `auto` would send the agent reading models/auto/ -- the exact
        # wasted browsing this block exists to prevent.
        self.assertNotIn("auto", itf.parse_modular_source(self.SRC)["parents"])

    def test_keyword_bases_are_dropped(self) -> None:
        info = itf.parse_modular_source("class Foo(Base, metaclass=Meta):\n    pass\n")
        self.assertEqual(info["defined"], [("Foo", ["Base"])])

    def test_empty_source_yields_nothing(self) -> None:
        self.assertEqual(itf.parse_modular_source(""), {"defined": [], "parents": {}})


class ModularContextRenderTests(unittest.TestCase):
    def _info(self):
        return itf.parse_modular_source(ModularSourceParseTests.SRC)

    def test_block_states_lineage_and_definitions(self) -> None:
        text = "\n".join(itf.modular_context_lines("edgetam", self._info()))
        self.assertIn("ported from: `sam2`", text)
        self.assertIn("`EdgeTamVisionModel` ← `Sam2VisionModel`", text)
        self.assertIn("modular_edgetam.py", text)

    def test_block_says_what_absence_means(self) -> None:
        # The #48322 cost was not knowing that a class missing from the modular
        # file is inherited -- so the block must say it explicitly.
        text = "\n".join(itf.modular_context_lines("edgetam", self._info()))
        self.assertIn("NOT in the list above", text)
        self.assertIn("make fix-repo", text)

    def test_no_info_renders_nothing(self) -> None:
        self.assertEqual(itf.modular_context_lines("bert", None), [])
        self.assertEqual(
            itf.modular_context_lines("bert", {"defined": [], "parents": {}}), []
        )

    def test_class_list_is_bounded(self) -> None:
        many = "".join(f"class C{i}(B):\n    pass\n\n" for i in range(60))
        text = "\n".join(
            itf.modular_context_lines("big", itf.parse_modular_source(many))
        )
        self.assertIn(f"and {60 - itf._MODULAR_MAX_CLASSES} more", text)
        self.assertEqual(text.count("    - `C"), itf._MODULAR_MAX_CLASSES)


class AttachModularContextTests(unittest.TestCase):
    def _target(self, model="edgetam", kind="model_failures"):
        return {"kind": kind, "model": model, "label": "x", "failures": []}

    def test_attaches_for_model_group(self) -> None:
        calls = []

        def fetch(repo, path, token):
            calls.append(path)
            return ModularSourceParseTests.SRC

        out = itf.attach_modular_context(
            [self._target()], "huggingface/transformers", None, fetch=fetch
        )
        self.assertEqual(calls, ["src/transformers/models/edgetam/modular_edgetam.py"])
        self.assertEqual(list(out[0]["modular"]["parents"]), ["sam2"])

    def test_missing_modular_file_leaves_target_untouched(self) -> None:
        t = self._target("bert")
        out = itf.attach_modular_context(
            [t], "huggingface/transformers", None, fetch=lambda *a: None
        )
        self.assertNotIn("modular", out[0])
        self.assertIs(out[0], t)

    def test_one_fetch_per_distinct_model(self) -> None:
        calls = []

        def fetch(repo, path, token):
            calls.append(path)
            return ModularSourceParseTests.SRC

        itf.attach_modular_context(
            [self._target(), self._target(), self._target("blt")],
            "huggingface/transformers",
            None,
            fetch=fetch,
        )
        self.assertEqual(len(calls), 2)

    def test_cluster_targets_are_skipped(self) -> None:
        t = self._target(kind="cluster")
        calls = []
        out = itf.attach_modular_context(
            [t], "r", None, fetch=lambda *a: calls.append(1)
        )
        self.assertEqual(calls, [])
        self.assertIs(out[0], t)

    def test_attaching_does_not_change_the_fingerprint(self) -> None:
        # Same contract as prior_feedback: an additive key must not re-identify
        # a group, or every enriched group looks new to the dedupe.
        t = {
            "kind": "model_failures",
            "model": "edgetam",
            "label": "x",
            "failures": [],
            "failure_mode": "output_mismatch",
        }
        before = itf.target_fingerprint(t)
        out = itf.attach_modular_context(
            [t], "r", None, fetch=lambda *a: ModularSourceParseTests.SRC
        )
        self.assertEqual(itf.target_fingerprint(out[0]), before)


def _commit_file(path, patch="@@ -1 +1 @@\n-a\n+b", adds=1, dels=1):
    return {"filename": path, "patch": patch, "additions": adds, "deletions": dels}


def _cluster_target(sha="6034e90c7d1b", model="recurrent_gemma"):
    return {
        "kind": "cluster",
        "label": f"2 integration tests regressed by commit {sha}",
        "model": None,
        "failure_mode": None,
        "cluster": {
            "bad_commit": sha,
            "pr_number": 47598,
            "failure_excerpt": "",
            "failures": [{"model": model, "failure_mode": "output_mismatch"}],
        },
        "failures": [],
    }


class BadCommitDiffSelectionTests(unittest.TestCase):
    """The attribution block renders a SHA the agent cannot dereference — it has
    no git tool. PR #48223's reviewer answered the whole question from that
    commit's diff ("the linked commit deleted a `partial_rotary_factor`")."""

    def test_only_the_failing_models_files_survive(self) -> None:
        # The real 6034e90c7d1b ("Simplify all Rotary modules") touches 186
        # files; a raw diff is not an option.
        files = [
            _commit_file("src/transformers/models/llama/modeling_llama.py"),
            _commit_file(
                "src/transformers/models/recurrent_gemma/modeling_recurrent_gemma.py"
            ),
            _commit_file("docs/source/en/index.md"),
        ]
        info = itf.select_bad_commit_diff(files, {"recurrent_gemma"})
        self.assertEqual(
            [f["path"] for f in info["files"]],
            ["src/transformers/models/recurrent_gemma/modeling_recurrent_gemma.py"],
        )
        self.assertEqual(info["total_changed"], 3)

    def test_a_generated_sibling_is_dropped_for_its_modular_source(self) -> None:
        # kimi_k25's modeling_*.py and modular_*.py carried the same change
        # twice (+93/-75 and +94/-76, 27,056 chars together).
        files = [
            _commit_file("src/transformers/models/kimi_k25/modeling_kimi_k25.py"),
            _commit_file("src/transformers/models/kimi_k25/modular_kimi_k25.py"),
        ]
        info = itf.select_bad_commit_diff(files, {"kimi_k25"})
        self.assertEqual(
            [f["path"] for f in info["files"]],
            ["src/transformers/models/kimi_k25/modular_kimi_k25.py"],
        )

    def test_the_models_own_files_are_separated_from_shared_code(self) -> None:
        files = [
            _commit_file("src/transformers/generation/utils.py"),
            _commit_file("src/transformers/modeling_utils.py"),
            _commit_file("tests/models/phimoe/test_modeling_phimoe.py"),
            _commit_file("src/transformers/models/phimoe/modeling_phimoe.py"),
        ]
        info = itf.select_bad_commit_diff(files, {"phimoe"})
        self.assertEqual(
            [f["path"] for f in info["files"]],
            [
                "src/transformers/models/phimoe/modeling_phimoe.py",
                "tests/models/phimoe/test_modeling_phimoe.py",
            ],
        )
        self.assertEqual(
            sorted(f["path"] for f in info["shared_files"]),
            [
                "src/transformers/generation/utils.py",
                "src/transformers/modeling_utils.py",
            ],
        )

    def test_shared_code_below_the_top_level_is_kept(self) -> None:
        """Caught by a real dry run, not by a fixture. `9f66415aec04` was
        attributed to a `t5` failure; a top-level-only rule kept
        `src/transformers/_typing.py` (+1/-3, a type annotation) and dropped
        `src/transformers/generation/utils.py` (+33/-53), which is the code a
        failing `test_compile_static_cache` actually runs through."""
        files = [
            _commit_file("src/transformers/_typing.py", adds=1, dels=3),
            _commit_file("src/transformers/generation/utils.py", adds=33, dels=53),
            _commit_file("tests/generation/test_utils.py", adds=56, dels=0),
        ]
        info = itf.select_bad_commit_diff(files, {"t5"})
        self.assertEqual(info["files"], [])
        paths = [f["path"] for f in info["shared_files"]]
        self.assertIn("src/transformers/generation/utils.py", paths)
        # ...and the substantive change leads, so a tight budget keeps it.
        self.assertEqual(paths[0], "src/transformers/generation/utils.py")

    def test_another_models_files_are_never_shared_code(self) -> None:
        info = itf.select_bad_commit_diff(
            [_commit_file("src/transformers/models/llama/modeling_llama.py")], {"t5"}
        )
        self.assertEqual(info["files"], [])
        self.assertEqual(info["shared_files"], [])

    def test_the_patch_budget_is_enforced(self) -> None:
        big = _commit_file(
            "src/transformers/models/blt/modeling_blt.py", patch="x" * 50_000
        )
        info = itf.select_bad_commit_diff([big], {"blt"})
        self.assertLessEqual(
            sum(len(f["patch"]) for f in info["files"]), itf._BAD_COMMIT_DIFF_CHARS
        )
        self.assertTrue(info["files"][0]["truncated"])

    def test_a_file_with_no_patch_key_does_not_crash(self) -> None:
        # GitHub omits `patch` for binary and very large files.
        info = itf.select_bad_commit_diff(
            [{"filename": "src/transformers/models/blt/modeling_blt.py"}], {"blt"}
        )
        self.assertEqual(info["files"][0]["patch"], "")


class BadCommitMisattributionTests(unittest.TestCase):
    """bd9509355c8a was blamed for a `phimoe` weight-conversion RuntimeError and
    changes exactly one file — an unrelated model's test. Rendering nothing would
    leave the agent trusting a bare SHA."""

    def test_an_unrelated_commit_is_reported_not_silently_empty(self) -> None:
        files = [_commit_file("tests/models/inkling/test_modeling_inkling.py")]
        info = itf.select_bad_commit_diff(files, {"phimoe"})
        self.assertEqual(info["files"], [])
        self.assertEqual(info["shared_files"], [])
        text = "\n".join(itf.bad_commit_diff_lines(info))
        self.assertIn("nothing that can reach the failing model", text)
        self.assertIn("tests/models/inkling/test_modeling_inkling.py", text)
        self.assertIn("unconfirmed", text)

    def test_shared_only_is_never_labelled_as_the_models_files(self) -> None:
        """The heading must not claim a shared file belongs to the model — the
        agent may go and edit what it is told is the model's."""
        info = itf.select_bad_commit_diff(
            [_commit_file("src/transformers/generation/utils.py")], {"t5"}
        )
        text = "\n".join(itf.bad_commit_diff_lines(info))
        self.assertIn("none of the failing model's own files", text)
        self.assertIn("does not belong in the model directory", text)
        self.assertNotIn("changed in the failing model's own files", text)

    def test_no_info_renders_nothing(self) -> None:
        self.assertEqual(itf.bad_commit_diff_lines(None), [])


class AttachBadCommitDiffTests(unittest.TestCase):
    def test_one_fetch_per_distinct_commit(self) -> None:
        asked = []

        def fetch(repo, sha, token):
            asked.append(sha)
            return [_commit_file("src/transformers/models/x/modeling_x.py")]

        itf.attach_bad_commit_diff(
            [
                _cluster_target("aaa", "x"),
                _cluster_target("aaa", "x"),
                _cluster_target("bbb", "x"),
            ],
            "huggingface/transformers",
            None,
            fetch=fetch,
        )
        self.assertEqual(asked, ["aaa", "bbb"])

    def test_a_model_group_makes_no_api_call(self) -> None:
        def fetch(*a):  # pragma: no cover — must not run
            raise AssertionError("a model group has no bad commit to fetch")

        t = {"kind": "model_failures", "model": "blt", "label": "x", "failures": []}
        out = itf.attach_bad_commit_diff([t], "r", None, fetch=fetch)
        self.assertIs(out[0], t)

    def test_a_failed_fetch_leaves_the_target_untouched(self) -> None:
        t = _cluster_target()
        out = itf.attach_bad_commit_diff([t], "r", None, fetch=lambda *a: None)
        self.assertIs(out[0], t)

    def test_attaching_does_not_change_the_fingerprint(self) -> None:
        # Same contract as prior_feedback and modular: an additive key must not
        # re-identify a group, or every enriched group looks new to the dedupe.
        t = _cluster_target()
        before = itf.target_fingerprint(t)
        out = itf.attach_bad_commit_diff(
            [t],
            "r",
            None,
            fetch=lambda *a: [
                _commit_file(
                    "src/transformers/models/recurrent_gemma/"
                    "modeling_recurrent_gemma.py"
                )
            ],
        )
        self.assertIn("bad_commit_diff", out[0])
        self.assertEqual(itf.target_fingerprint(out[0]), before)

    def test_the_diff_renders_under_the_sha_and_before_the_tracebacks(self) -> None:
        out = itf.attach_bad_commit_diff(
            [_cluster_target()],
            "r",
            None,
            fetch=lambda *a: [
                _commit_file(
                    "src/transformers/models/recurrent_gemma/"
                    "modeling_recurrent_gemma.py",
                    patch="-        partial_rotary_factor=0.5,",
                )
            ],
        )
        text = "\n".join(itf._render_serge_target(out[0], 7))
        self.assertIn("partial_rotary_factor", text)
        self.assertLess(text.index("bad commit:"), text.index("partial_rotary_factor"))
        self.assertLess(
            text.index("partial_rotary_factor"), text.index("Failure-mode mix")
        )


class UnverifiedBranchInRecapTests(unittest.TestCase):
    """When the GPU gate never judges a patch — no runner picked the job up, the
    dispatch failed, no artifact came back — serge keeps the candidate branch
    instead of tearing it down (serge#110) and opens no PR. Nothing else points
    at that branch: 5 of 79 serge fix branches were orphaned that way, invisible
    unless someone listed the refs. The recap is where a human finds it, so the
    link is the point, not decoration."""

    BRANCH = "serge/fix/itf-71ca76cde2a9-b228e033"

    def _detail(self, **result):
        base = {
            "message": "A candidate patch was committed but GPU verification "
            "could not run (`timeout`), so it is UNVERIFIED.",
            "verify_verdict": "timeout",
        }
        base.update(result)
        return {"status": "no_fix", "result": base, "model": "kimi"}

    def test_distill_carries_the_kept_branch(self) -> None:
        out = itf._distill_outcome(self._detail(branch=self.BRANCH))
        self.assertEqual(out["branch"], self.BRANCH)

    def test_no_branch_when_serge_did_not_keep_one(self) -> None:
        self.assertIsNone(itf._distill_outcome(self._detail())["branch"])
        self.assertIsNone(
            itf._distill_outcome({"status": "error", "error": "boom", "result": None})[
                "branch"
            ]
        )

    def test_the_reason_cell_links_the_branch(self) -> None:
        cell = itf._reason_cell(itf._distill_outcome(self._detail(branch=self.BRANCH)))
        self.assertIn(f"[`{self.BRANCH}`]", cell)
        self.assertIn(f"/tree/{self.BRANCH}", cell)
        self.assertIn("unverified patch kept on", cell)

    def test_the_reason_cell_is_unchanged_without_a_branch(self) -> None:
        distilled = itf._distill_outcome(self._detail())
        self.assertEqual(itf._reason_cell(distilled), distilled["reason"])
        self.assertNotIn("unverified", itf._reason_cell(distilled))

    def test_the_recap_table_renders_the_link(self) -> None:
        target = _target("cwm", "output_mismatch")
        fp = itf.target_fingerprint(target)
        lines = itf._render_outcome_recap(
            [target],
            {fp: None},
            {fp: itf._distill_outcome(self._detail(branch=self.BRANCH))},
        )
        body = "\n".join(lines)
        self.assertIn("## Outcome recap", body)
        self.assertIn(f"/tree/{self.BRANCH}", body)

    def test_a_group_with_a_pr_is_still_skipped(self) -> None:
        """A PR is the outcome; a kept branch must not resurrect a recap row."""
        target = _target("cwm", "output_mismatch")
        fp = itf.target_fingerprint(target)
        self.assertEqual(
            itf._render_outcome_recap(
                [target],
                {fp: 123},
                {fp: itf._distill_outcome(self._detail(branch=self.BRANCH))},
            ),
            [],
        )


class UnverifiedBranchCommandTests(unittest.TestCase):
    """The recap told a reader to "re-run verification against it" and gave them
    no way to do it: dispatching serge-verify-caller.yml needs a base sha, a
    candidate sha, the node-ids and the model folder, and this renderer is the
    only place all four exist at once.

    It also has to say the cheaper thing first. On 2026-09-01 both timed-out
    groups' verify runs finished AFTER Serge stopped waiting — the verdicts were
    computed and never read — so reading the existing run's artifact is free
    where a re-run costs ~40 GPU-minutes.
    """

    BRANCH = "serge/fix/itf-4a72b9af302d-9a266db2"
    COMMIT = "5fbcc9804e6704dacbb22cff8c952868c2f3987e"

    def _detail(self, **over):
        base = {
            "status": "no_fix",
            "result": {
                "message": "GPU verification could not run (`timeout`).",
                "verify_verdict": "timeout",
                "branch": self.BRANCH,
                "commit_sha": self.COMMIT,
            },
            "model": "kimi",
        }
        base["result"].update(over)
        return base

    def _body(self, detail=None):
        target = _target("output_mismatch")
        fp = itf.target_fingerprint(target)
        return "\n".join(
            itf._render_outcome_recap(
                [target],
                {fp: None},
                {fp: itf._distill_outcome(detail or self._detail())},
            )
        )

    def test_the_commit_sha_survives_distillation(self):
        self.assertEqual(
            itf._distill_outcome(self._detail())["commit_sha"], self.COMMIT
        )

    def test_the_command_carries_all_four_dispatch_inputs(self):
        body = self._body()
        self.assertIn("gh workflow run serge-verify-caller.yml", body)
        self.assertIn(f"-f commit_sha={self.COMMIT}", body)
        self.assertIn("-f base_sha=$(gh api", body)  # parent resolved, not guessed
        self.assertIn("-f test_nodeids='tests/", body)
        self.assertIn("-f model=", body)

    def test_it_says_to_read_the_existing_run_first(self):
        """A re-run recomputes a verdict that usually already exists."""
        body = self._body()
        self.assertIn("gh run download", body)
        self.assertIn("Only if no verdict exists", body)
        self.assertLess(body.index("gh run download"), body.index("gh workflow run"))

    def test_no_section_without_a_kept_branch(self):
        body = self._body(self._detail(branch=None, commit_sha=None))
        self.assertNotIn("Unverified branches", body)
        self.assertNotIn("gh workflow run", body)

    def test_no_section_when_the_commit_sha_is_missing(self):
        """An older Serge build returns a branch and no sha; a command with a
        blank commit_sha would dispatch a verify against nothing."""
        body = self._body(self._detail(commit_sha=None))
        self.assertNotIn("gh workflow run", body)

    def test_a_group_with_a_pr_gets_no_command(self):
        target = _target("output_mismatch")
        fp = itf.target_fingerprint(target)
        body = "\n".join(
            itf._render_outcome_recap(
                [target], {fp: 123}, {fp: itf._distill_outcome(self._detail())}
            )
        )
        self.assertNotIn("gh workflow run", body)
