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


class MultiRunVerdictTest(unittest.TestCase):
    """The targeted tests are run several times (fresh process each) so a flaky
    pass/fail can't drive the verdict. Each run is a separate parsed report;
    ``baseline``/``patched`` are the LIST of those reports and the guards must
    hold across every run."""

    METHOD = "test_small_token_timestamp_generation"

    def _runs(self, per_run: list[tuple[str, str]]):
        """One parsed report per run: ``per_run[i]`` = (outcome, detail)."""
        return [{(CLS, self.METHOD): {"outcome": o, "detail": d}} for o, d in per_run]

    def test_collect_gathers_every_run(self):
        runs = v._collect(self._runs([("failed", "a"), ("failed", "b")]), TS)
        self.assertEqual([r["outcome"] for r in runs], ["failed", "failed"])

    def test_collect_missing_when_absent_from_all_runs(self):
        # One "missing" per run when the node-id is in none of them...
        self.assertEqual(
            [r["outcome"] for r in v._collect([{}, {}], TS)], ["missing"] * 2
        )
        # ...and a single "missing" when there were no runs at all.
        self.assertEqual(v._collect([], TS), [{"outcome": "missing", "detail": ""}])

    def test_single_dict_coerced_to_one_run(self):
        rep = {(CLS, self.METHOD): {"outcome": "failed", "detail": "x"}}
        self.assertEqual(len(v._collect(rep, TS)), 1)

    def test_fixed_only_when_every_run_flips(self):
        baseline = self._runs([("failed", "x")] * 5)
        patched = self._runs([("green", "")] * 5)
        out = v.build_verdict([TS], baseline, patched)
        self.assertEqual(out["verdict"], "fixed")
        self.assertEqual(out["runs"], 5)

    def test_not_fixed_when_one_patched_run_red(self):
        baseline = self._runs([("failed", "x")] * 5)
        patched = self._runs(
            [
                ("green", ""),
                ("green", ""),
                ("failed", "flaky"),
                ("green", ""),
                ("green", ""),
            ]
        )
        out = v.build_verdict([TS], baseline, patched)
        self.assertEqual(out["verdict"], "not_fixed")
        self.assertIn("flaky", out["tracebacks"][TS])

    def test_already_passing_when_one_baseline_run_green(self):
        baseline = self._runs([("failed", "x"), ("green", ""), ("failed", "x")])
        patched = self._runs([("green", "")] * 3)
        out = v.build_verdict([TS], baseline, patched)
        self.assertEqual(out["verdict"], "already_passing")

    def test_reproduce_requires_every_run_red(self):
        out = v.build_reproduce_verdict([TS], self._runs([("failed", "boom")] * 5))
        self.assertEqual(out["verdict"], "reproduced")
        self.assertEqual(out["runs"], 5)
        self.assertIn("boom", out["tracebacks"][TS])

    def test_reproduce_not_reproduced_when_one_run_green(self):
        out = v.build_reproduce_verdict(
            [TS], self._runs([("failed", "boom"), ("green", ""), ("failed", "boom")])
        )
        self.assertEqual(out["verdict"], "not_reproduced")

    def test_main_aggregates_multiple_xml_files(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            # 3 baseline + 3 patched XMLs (one per run), all red -> all green.
            base_files, patch_files = [], []
            for i in range(3):
                b = tmp / f"baseline_{i}.xml"
                p = tmp / f"patched_{i}.xml"
                b.write_text(_suite({self.METHOD: ("failed", "x")}))
                p.write_text(_suite({self.METHOD: ("green", "")}))
                base_files.append(str(b))
                patch_files.append(str(p))
            out = tmp / "v.json"
            rc = v.main(
                ["--nodeids", TS, "--baseline", *base_files, "--patched", *patch_files]
                + ["--out", str(out)]
            )
            self.assertEqual(rc, 0)
            data = json.loads(out.read_text())
            self.assertEqual(data["verdict"], "fixed")
            self.assertEqual(data["runs"], 3)

    def test_main_not_fixed_when_one_of_many_patched_files_red(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            base_files, patch_files = [], []
            for i in range(3):
                b = tmp / f"baseline_{i}.xml"
                b.write_text(_suite({self.METHOD: ("failed", "x")}))
                base_files.append(str(b))
                p = tmp / f"patched_{i}.xml"
                # middle run flaky-fails
                p.write_text(
                    _suite(
                        {self.METHOD: ("failed", "flake") if i == 1 else ("green", "")}
                    )
                )
                patch_files.append(str(p))
            out = tmp / "v.json"
            rc = v.main(
                ["--nodeids", TS, "--baseline", *base_files, "--patched", *patch_files]
                + ["--out", str(out)]
            )
            self.assertEqual(rc, 1)
            self.assertEqual(json.loads(out.read_text())["verdict"], "not_fixed")


class DistillFailureTest(unittest.TestCase):
    """The distiller caps big tensor/array literals so the asserted slice + diff
    survive instead of being buried under KB-scale --showlocals dumps."""

    def _huge_tensor(self, elems=4000):
        return "tensor([" + ", ".join(f"{i * 0.001:.6f}" for i in range(elems)) + "])"

    def test_small_tensor_untouched(self):
        text = "expected = tensor([[0.0679, 0.0422, 0.1347]])"
        self.assertEqual(v.distill_failure(text), text)

    def test_large_tensor_elided_header_kept(self):
        header = (
            "AssertionError: Tensor-likes are not close!\nMismatched elements: 6 / 9\n"
        )
        small = "expected_slice = tensor([[0.0679, 0.0422]])\n"
        huge = "image_embeds = " + self._huge_tensor() + "\n"
        out = v.distill_failure(header + small + huge)
        self.assertIn("Mismatched elements: 6 / 9", out)
        self.assertIn("expected_slice = tensor([[0.0679, 0.0422]])", out)  # kept whole
        self.assertIn("chars elided", out)  # huge one elided
        self.assertLess(len(out), len(header + small + huge) // 2)

    def test_nested_field_inside_modeloutput(self):
        # actual asserted tensor is a FIELD inside a big ModelOutput repr, right
        # after a huge sibling field — must survive.
        huge = self._huge_tensor()
        text = f"outputs = OwlViTOutput(image_embeds={huge}, pred_boxes=tensor([[0.11, 0.22, 0.33]]))"
        out = v.distill_failure(text)
        self.assertIn(
            "pred_boxes=tensor([[0.11, 0.22, 0.33]])", out
        )  # asserted slice intact
        self.assertIn("chars elided", out)  # the huge sibling elided

    def test_numpy_array_capped(self):
        huge = "array([" + ", ".join(str(i) for i in range(4000)) + "])"
        out = v.distill_failure(f"logits = {huge}")
        self.assertIn("chars elided", out)
        self.assertLess(len(out), len(huge) // 2)

    def test_moderate_asserted_output_kept_whole(self):
        # The asserted ACTUAL value can itself be a moderately large tensor the
        # model must copy verbatim (e.g. a generated token sequence — instructblip).
        # At the generous cap it must survive intact, not be elided.
        seq = "tensor([[" + ", ".join(str(i) for i in range(300)) + "]])"  # ~1.2 KB
        self.assertLess(len(seq), 2500)
        out = v.distill_failure(f"outputs = {seq}")
        self.assertNotIn("chars elided", out)
        self.assertIn(seq, out)

    def test_no_literals_unchanged(self):
        text = "RuntimeError: CUDA out of memory.\n  File 'x.py', line 5, in f\n"
        self.assertEqual(v.distill_failure(text), text)

    def test_detail_distills_from_junit(self):
        huge = self._huge_tensor()
        xml = (
            '<?xml version="1.0"?><testsuites><testsuite>'
            f'<testcase classname="tests.models.owlvit.test_modeling_owlvit.{CLS}" '
            'name="test_x"><failure>AssertionError: Tensor-likes are not close!\n'
            "Mismatched elements: 6 / 9\n"
            "expected_slice = tensor([[0.0679]])\n"
            f"image_embeds = {huge}\n"
            "</failure></testcase></testsuite></testsuites>"
        )
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "r.xml"
            p.write_text(xml)
            rep = v.parse_junit(str(p))
        detail = rep[(CLS, "test_x")]["detail"]
        self.assertIn("Mismatched elements: 6 / 9", detail)
        self.assertIn("expected_slice = tensor([[0.0679]])", detail)
        self.assertIn("chars elided", detail)
        self.assertLess(len(detail), len(huge))  # noise gone


if __name__ == "__main__":
    unittest.main()
