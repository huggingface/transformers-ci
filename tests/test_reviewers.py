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

from transformersci.reviewers import resolver
from transformersci.reviewers import assign


CODEOWNERS = """
* @catchall
*tokenization* @tokenizer-owner
/src/transformers/models/*/image_processing* @image-owner

@@modality/text @text-owner
@@modality/vision @vision-owner
@@modality/nonsense @nobody

/src/transformers/models/bert/ @bert-owner
utils/dummy*
""".splitlines(keepends=True)

TOCTREE = """
- sections:
  - title: Text models
    sections:
    - local: model_doc/bert
    - local: model_doc/gpt2
  - title: Vision models
    sections:
    - local: model_doc/dinov3
"""


def fake_reader(files):
    """A `read_file` over an in-memory tree, as the resolver's callers inject."""

    def read(path):
        return files.get(path, "")

    return read


def reader(**model_files):
    files = {resolver.TOCTREE_PATH: TOCTREE}
    files.update(model_files)
    return fake_reader(files)


class ResolutionOrderTest(unittest.TestCase):
    def owners(self, path, read_file=None):
        return resolver.owners_for_file(path, CODEOWNERS, read_file or reader())

    def test_an_explicit_rule_beats_the_modality(self):
        # bert is a text model, but it has a rule of its own.
        self.assertEqual(
            self.owners("src/transformers/models/bert/modeling_bert.py"), ["bert-owner"]
        )

    def test_a_model_with_no_rule_falls_to_its_modality(self):
        self.assertEqual(
            self.owners("src/transformers/models/gpt2/modeling_gpt2.py"), ["text-owner"]
        )
        self.assertEqual(
            self.owners("src/transformers/models/dinov3/modeling_dinov3.py"),
            ["vision-owner"],
        )

    def test_a_variant_directory_inherits_the_modality_of_the_model_it_extends(self):
        # No doc page of its own: `dinov3_vit` is a variant of `dinov3`.
        self.assertEqual(
            self.owners("src/transformers/models/dinov3_vit/modeling_dinov3_vit.py"),
            ["vision-owner"],
        )

    def test_an_in_file_tag_beats_everything(self):
        read = reader(
            **{
                "src/transformers/models/bert/modular_bert.py": (
                    "# coding=utf-8\n# Reviewers: @tagged @second\nimport torch\n"
                )
            }
        )
        self.assertEqual(
            self.owners("src/transformers/models/bert/modeling_bert.py", read),
            ["tagged", "second"],
        )

    def test_a_tag_below_the_header_is_not_a_tag(self):
        # The header ends at the first statement; a `# Reviewers:` line after it is just a comment.
        read = reader(
            **{
                "src/transformers/models/gpt2/modular_gpt2.py": (
                    "import torch\n\n# Reviewers: @sneaky\n"
                )
            }
        )
        self.assertEqual(
            self.owners("src/transformers/models/gpt2/modeling_gpt2.py", read),
            ["text-owner"],
        )

    def test_a_file_that_is_not_a_model_uses_its_rule(self):
        self.assertEqual(
            self.owners("src/transformers/models/gpt2/tokenization_gpt2.py"),
            ["tokenizer-owner"],
        )

    def test_a_file_nothing_claims_falls_to_the_catch_all(self):
        self.assertEqual(self.owners("setup.py"), ["catchall"])

    def test_resolution_source_names_which_of_the_four_answered(self):
        source = resolver.resolution_source
        self.assertEqual(
            source(
                "src/transformers/models/bert/modeling_bert.py", CODEOWNERS, reader()
            ),
            "rule",
        )
        self.assertEqual(
            source(
                "src/transformers/models/gpt2/modeling_gpt2.py", CODEOWNERS, reader()
            ),
            "modality",
        )
        self.assertEqual(source("setup.py", CODEOWNERS, reader()), "catch-all")


class CodeownersFileTest(unittest.TestCase):
    def test_a_modality_rule_is_not_matched_as_a_path(self):
        # `@@modality/...` is a rule about a doc section, not a glob -- it must never match a file.
        pattern, _ = resolver.match_codeowners("@@modality/text", CODEOWNERS)
        self.assertEqual(pattern, resolver.CATCH_ALL_PATTERN)

    def test_a_modality_the_resolver_does_not_know_is_reported_not_silently_dropped(
        self,
    ):
        self.assertEqual(resolver.unknown_modality_slugs(CODEOWNERS), ["nonsense"])

    def test_a_rule_with_no_owner_marks_a_path_deliberately_unowned(self):
        self.assertEqual(
            resolver.get_file_owners("utils/dummy_pt_objects.py", CODEOWNERS), []
        )

    def test_pr_author_is_in_hf_ignores_spelling(self):
        self.assertTrue(resolver.pr_author_is_in_hf("BERT-Owner", CODEOWNERS))
        self.assertFalse(resolver.pr_author_is_in_hf("outsider", CODEOWNERS))


class FakePR:
    def __init__(self):
        self.requested = []

    def create_review_request(self, logins):
        self.requested.extend(logins)


class FakeRepo:
    full_name = "huggingface/transformers"

    def __init__(self, collaborators):
        self.collaborators = collaborators

    def has_in_collaborators(self, login):
        return login in self.collaborators


class RequestReviewsTest(unittest.TestCase):
    def test_a_departed_owner_costs_one_slot_not_all_of_them(self):
        # The bug this whole thing exists for: GitHub rejects a multi-name request outright, so one
        # stale entry used to take every valid co-owner down with it.
        repo = FakeRepo({"present", "also-present"})
        pr = FakePR()
        requested = assign.request_reviews(
            repo, pr, ["gone", "present", "also-present"]
        )
        self.assertEqual(requested, ["present", "also-present"])
        self.assertEqual(pr.requested, ["present", "also-present"])

    def test_the_next_ranked_owner_takes_a_skipped_slot(self):
        repo = FakeRepo({"third"})
        pr = FakePR()
        self.assertEqual(
            assign.request_reviews(repo, pr, ["first", "second", "third"]), ["third"]
        )

    def test_it_stops_at_the_limit(self):
        repo = FakeRepo({"a", "b", "c"})
        pr = FakePR()
        self.assertEqual(
            assign.request_reviews(repo, pr, ["a", "b", "c"], limit=2), ["a", "b"]
        )


if __name__ == "__main__":
    unittest.main()
