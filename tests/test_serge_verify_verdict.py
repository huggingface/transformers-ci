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

import json
import unittest

from transformersci.agentic import serge_verify_verdict as v

WHISPER = "tests/models/whisper/test_modeling_whisper.py"
CLS = "WhisperModelIntegrationTests"
TS = f"{WHISPER}::{CLS}::test_small_token_timestamp_generation"
GEN = f"{WHISPER}::{CLS}::test_tiny_generation"


def _suite(cases: dict[str, tuple[str, str]]) -> str:
    """Build a pytest-style JUnit XML. `cases` maps method -> (outcome, detail);
    outcome in {green, failed, error, skipped}."""
    parts = ['<?xml version="1.0" encoding="utf-8"?>', "<testsuites><testsuite>"]
    for method, (outcome, detail) in cases.items():
        cn = f"tests.models.whisper.test_modeling_whisper.{CLS}"
        if outcome == "green":
            parts.append(f'<testcase classname="{cn}" name="{method}"/>')
        elif outcome == "skipped":
            parts.append(
                f'<testcase classname="{cn}" name="{method}"><skipped/></testcase>'
            )
        else:  # failed / error
            tag = "failure" if outcome == "failed" else "error"
            parts.append(
                f'<testcase classname="{cn}" name="{method}">'
                f'<{tag} message="{detail}">{detail}</{tag}></testcase>'
            )
    parts.append("</testsuite></testsuites>")
    return "\n".join(parts)


def _write(tmp, name, cases):
    p = tmp / name
    p.write_text(_suite(cases))
    return str(p)


class NodeidKeyTest(unittest.TestCase):
    def test_class_and_method(self):
        self.assertEqual(
            v.nodeid_key(TS), (CLS, "test_small_token_timestamp_generation")
        )

    def test_parametrized(self):
        self.assertEqual(
            v.nodeid_key(f"{WHISPER}::{CLS}::test_x[fp16]"), (CLS, "test_x[fp16]")
        )

    def test_function_level(self):
        self.assertEqual(v.nodeid_key(f"{WHISPER}::test_x"), ("", "test_x"))


class BuildVerdictTest(unittest.TestCase):
    def _rep(self, mapping):
        # mapping: method -> (outcome, detail)
        return {(CLS, m): {"outcome": o, "detail": d} for m, (o, d) in mapping.items()}

    def test_fixed_red_then_green(self):
        baseline = self._rep(
            {"test_small_token_timestamp_generation": ("failed", "boom")}
        )
        patched = self._rep({"test_small_token_timestamp_generation": ("green", "")})
        out = v.build_verdict([TS], baseline, patched)
        self.assertEqual(out["verdict"], "fixed")
        self.assertEqual(
            out["targeted"][0], {"nodeid": TS, "baseline": "failed", "patched": "green"}
        )

    def test_not_fixed_still_red_returns_tracebacks(self):
        baseline = self._rep(
            {"test_small_token_timestamp_generation": ("failed", "boom")}
        )
        patched = self._rep(
            {"test_small_token_timestamp_generation": ("failed", "still boom")}
        )
        out = v.build_verdict([TS], baseline, patched)
        self.assertEqual(out["verdict"], "not_fixed")
        self.assertIn("still boom", out["tracebacks"][TS])

    def test_already_passing_when_baseline_green(self):
        # Baseline-red guard: a target that was green before the patch means the
        # test self-healed / is flaky. Never claim a fix.
        baseline = self._rep({"test_small_token_timestamp_generation": ("green", "")})
        patched = self._rep({"test_small_token_timestamp_generation": ("green", "")})
        out = v.build_verdict([TS], baseline, patched)
        self.assertEqual(out["verdict"], "already_passing")

    def test_error_when_target_missing_from_patched(self):
        baseline = self._rep(
            {"test_small_token_timestamp_generation": ("failed", "boom")}
        )
        out = v.build_verdict([TS], baseline, {})  # patched report has no such test
        self.assertEqual(out["verdict"], "error")

    def test_error_when_skipped(self):
        baseline = self._rep(
            {"test_small_token_timestamp_generation": ("failed", "boom")}
        )
        patched = self._rep({"test_small_token_timestamp_generation": ("skipped", "")})
        out = v.build_verdict([TS], baseline, patched)
        self.assertEqual(out["verdict"], "error")

    def test_multiple_targets_all_must_flip(self):
        baseline = self._rep(
            {
                "test_small_token_timestamp_generation": ("failed", "a"),
                "test_tiny_generation": ("failed", "b"),
            }
        )
        patched = self._rep(
            {
                "test_small_token_timestamp_generation": ("green", ""),
                "test_tiny_generation": ("failed", "b2"),
            }
        )
        out = v.build_verdict([TS, GEN], baseline, patched)
        self.assertEqual(out["verdict"], "not_fixed")

    def test_broke_others_needs_collateral_baseline(self):
        baseline = self._rep({"test_small_token_timestamp_generation": ("failed", "a")})
        patched = self._rep({"test_small_token_timestamp_generation": ("green", "")})
        # A neighbour that was green before but fails after the patch.
        neigh = (CLS, "test_neighbour")
        coll = {neigh: {"outcome": "failed", "detail": "regressed"}}
        coll_base = {neigh: {"outcome": "green", "detail": ""}}
        out = v.build_verdict(
            [TS], baseline, patched, collateral=coll, collateral_baseline=coll_base
        )
        self.assertEqual(out["verdict"], "broke_others")
        self.assertEqual(out["collateral_new_failures"], [f"{CLS}::test_neighbour"])

    def test_collateral_without_baseline_is_advisory_only(self):
        baseline = self._rep({"test_small_token_timestamp_generation": ("failed", "a")})
        patched = self._rep({"test_small_token_timestamp_generation": ("green", "")})
        neigh = (CLS, "test_neighbour")
        coll = {neigh: {"outcome": "failed", "detail": "pre-existing?"}}
        out = v.build_verdict(
            [TS], baseline, patched, collateral=coll, collateral_baseline=None
        )
        # No baseline to compare → can't call it a regression → stays fixed.
        self.assertEqual(out["verdict"], "fixed")
        self.assertEqual(out["collateral_new_failures"], [])


class BuildReproduceVerdictTest(unittest.TestCase):
    def _rep(self, mapping):
        return {(CLS, m): {"outcome": o, "detail": d} for m, (o, d) in mapping.items()}

    def test_reproduced_when_baseline_red(self):
        baseline = self._rep(
            {"test_small_token_timestamp_generation": ("failed", "boom")}
        )
        out = v.build_reproduce_verdict([TS], baseline)
        self.assertEqual(out["mode"], "reproduce")
        self.assertEqual(out["verdict"], "reproduced")
        self.assertIn("boom", out["tracebacks"][TS])
        self.assertEqual(out["targeted"][0], {"nodeid": TS, "baseline": "failed"})
        self.assertNotIn("patched", out["targeted"][0])

    def test_reproduced_from_error_outcome(self):
        baseline = self._rep(
            {"test_small_token_timestamp_generation": ("error", "RuntimeError")}
        )
        out = v.build_reproduce_verdict([TS], baseline)
        self.assertEqual(out["verdict"], "reproduced")
        self.assertIn("RuntimeError", out["tracebacks"][TS])

    def test_not_reproduced_when_baseline_green(self):
        # The failure self-healed / is flaky at base → serge must NOT investigate.
        baseline = self._rep({"test_small_token_timestamp_generation": ("green", "")})
        out = v.build_reproduce_verdict([TS], baseline)
        self.assertEqual(out["verdict"], "not_reproduced")

    def test_error_when_target_missing(self):
        out = v.build_reproduce_verdict([TS], {})
        self.assertEqual(out["verdict"], "error")

    def test_error_when_skipped(self):
        baseline = self._rep({"test_small_token_timestamp_generation": ("skipped", "")})
        out = v.build_reproduce_verdict([TS], baseline)
        self.assertEqual(out["verdict"], "error")

    def test_all_targets_must_be_red(self):
        # One green among the group is enough to bail — mirrors the verify
        # baseline-red guard's conservatism.
        baseline = self._rep(
            {
                "test_small_token_timestamp_generation": ("failed", "a"),
                "test_tiny_generation": ("green", ""),
            }
        )
        out = v.build_reproduce_verdict([TS, GEN], baseline)
        self.assertEqual(out["verdict"], "not_reproduced")


class ParseJunitTest(unittest.TestCase):
    def setUp(self):
        import tempfile
        from pathlib import Path

        self._dir = tempfile.TemporaryDirectory()
        self.tmp = Path(self._dir.name)

    def tearDown(self):
        self._dir.cleanup()

    def test_round_trip_outcomes(self):
        path = _write(
            self.tmp,
            "r.xml",
            {
                "test_small_token_timestamp_generation": (
                    "failed",
                    "Tensor-likes are not close!",
                ),
                "test_tiny_generation": ("green", ""),
            },
        )
        rep = v.parse_junit(path)
        self.assertEqual(
            rep[(CLS, "test_small_token_timestamp_generation")]["outcome"], "failed"
        )
        self.assertIn(
            "Tensor-likes",
            rep[(CLS, "test_small_token_timestamp_generation")]["detail"],
        )
        self.assertEqual(rep[(CLS, "test_tiny_generation")]["outcome"], "green")

    def test_missing_file_is_empty(self):
        self.assertEqual(v.parse_junit("/nope/does-not-exist.xml"), {})
        self.assertEqual(v.parse_junit(None), {})

    def test_main_end_to_end(self):
        baseline = _write(
            self.tmp,
            "b.xml",
            {"test_small_token_timestamp_generation": ("failed", "x")},
        )
        patched = _write(
            self.tmp, "p.xml", {"test_small_token_timestamp_generation": ("green", "")}
        )
        out = self.tmp / "verdict.json"
        rc = v.main(
            [
                "--nodeids",
                TS,
                "--baseline",
                baseline,
                "--patched",
                patched,
                "--out",
                str(out),
            ]
        )
        self.assertEqual(rc, 0)
        data = json.loads(out.read_text())
        self.assertEqual(data["verdict"], "fixed")

    def test_main_reproduce_mode_without_patched(self):
        # mode=reproduce needs only --baseline; a red baseline → rc 0, reproduced.
        baseline = _write(
            self.tmp,
            "b.xml",
            {"test_small_token_timestamp_generation": ("failed", "boom")},
        )
        out = self.tmp / "verdict.json"
        rc = v.main(
            [
                "--mode",
                "reproduce",
                "--nodeids",
                TS,
                "--baseline",
                baseline,
                "--out",
                str(out),
            ]
        )
        self.assertEqual(rc, 0)
        data = json.loads(out.read_text())
        self.assertEqual(data["mode"], "reproduce")
        self.assertEqual(data["verdict"], "reproduced")
        self.assertIn("boom", data["tracebacks"][TS])

    def test_main_reproduce_mode_not_reproduced_is_nonzero(self):
        baseline = _write(
            self.tmp,
            "b.xml",
            {"test_small_token_timestamp_generation": ("green", "")},
        )
        out = self.tmp / "verdict.json"
        rc = v.main(
            [
                "--mode",
                "reproduce",
                "--nodeids",
                TS,
                "--baseline",
                baseline,
                "--out",
                str(out),
            ]
        )
        self.assertEqual(rc, 1)
        self.assertEqual(json.loads(out.read_text())["verdict"], "not_reproduced")


if __name__ == "__main__":
    unittest.main()
