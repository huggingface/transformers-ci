"""One probable-fix rule for the Test page and Project health: a recent open or
merged PR that changes the test file and names its topic, judged by when the
test last failed on main. The Test page used to show #49147 as the probable fix
of gemma3n's test_flash_attn_2_equivalence: merged 2026-10-08, and the test
failed every day after it."""

from __future__ import annotations

from transformersci.otel import probable_fix as pf
from transformersci.otel import project_health as ph

NODE = "tests/models/gemma3n/test_modeling_gemma3n.py::Gemma3nTextModelTest::test_fa2"
FILE = NODE.split("::")[0]
MERGED = "2026-10-08T11:52:46Z"
MERGED_TS = 1791460366.0


def node(number, title, files, merged_at=None, draft=False):
    return {
        "number": number,
        "title": title,
        "isDraft": draft,
        "mergedAt": merged_at,
        "files": {"nodes": [{"path": path} for path in files]},
    }


def pages(*replies):
    """A GraphQL stub answering each search with its pages in turn."""
    queries = []
    script = {search: list(page) for search, page in replies}

    def graphql(query):
        queries.append(query)
        search = next(key for key in script if key in query)
        nodes, more = script[search].pop(0)
        return {
            "search": {
                "nodes": nodes,
                "pageInfo": {"hasNextPage": more, "endCursor": f"c{len(queries)}"},
            }
        }

    return graphql, queries


def test_index_maps_test_files_to_open_and_merged_prs():
    graphql, queries = pages(
        (
            "is:open",
            [
                (
                    [
                        node(2, "Gemma3n: fix fa2", [FILE, "src/x.py"]),
                        node(4, "dummy: trigger CI on gemma3n", [FILE], draft=True),
                    ],
                    True,
                ),
                ([node(3, "Docs", ["docs/a.md"])], False),
            ],
        ),
        ("is:merged", [([node(1, "Fix Gemma3n fp16 loading", [FILE], MERGED)], False)]),
    )
    by_file = pf.build_index(graphql, now=MERGED_TS + 86400)
    assert sorted(by_file) == [FILE]
    assert [(pr["number"], pr["state"]) for pr in by_file[FILE]] == [
        (2, "open"),
        (1, "merged"),
    ]
    assert by_file[FILE][1]["merged_at"] == MERGED_TS
    assert len(queries) == 3 and 'after: "c1"' in queries[1]
    assert "created:>=2026-08-25" in queries[0] and "merged:>=2026-08-25" in queries[2]


def index(*prs):
    return {
        FILE: [
            {
                "number": number,
                "title": title,
                "url": f"https://github.com/huggingface/transformers/pull/{number}",
                "state": "merged" if merged_at else "open",
                "merged_at": merged_at,
            }
            for number, title, merged_at in prs
        ]
    }


def test_a_pr_merged_before_the_last_failure_did_not_fix_it():
    by_file = index((49147, "This PR fixes Gemma3n fp16 loading", MERGED_TS))
    still = pf.for_test(NODE, MERGED_TS + 9 * 86400, by_file)
    assert (still["number"], still["verdict"]) == (49147, "merged_before")
    fixed = pf.for_test(NODE, MERGED_TS - 3600, by_file)
    assert fixed["verdict"] == "merged_after"
    # No failure on main known: the merge is not contradicted.
    assert pf.for_test(NODE, None, by_file)["verdict"] == "merged_after"


def test_open_beats_merged_and_the_title_must_name_the_topic():
    by_file = index(
        (1, "Fix Gemma3n fp16 loading", MERGED_TS),
        (2, "Gemma3n: fix fa2 equivalence", None),
        (3, "Refactor every test file", None),
    )
    best = pf.for_test(NODE, MERGED_TS + 86400, by_file)
    assert (best["number"], best["verdict"], best["others"]) == (2, "open", 1)
    assert (
        pf.for_test("tests/models/bert/test_modeling_bert.py::T::t", 0, by_file) is None
    )
    assert pf.for_test("../etc/passwd", 0, by_file) is None


def test_last_failure_is_the_latest_failing_run_start():
    seen = []

    def query(expr):
        seen.append(expr)
        if "pytest_test_last_failure_info" in expr:
            return [
                {"metric": {"run_id": "11:1"}, "value": [0, "1"]},
                {"metric": {"run_id": "12:1"}, "value": [0, "1"]},
                {"metric": {"run_id": 'x"}'}, "value": [0, "1"]},  # never in a regex
            ]
        return [
            {"metric": {"run_id": "11:1"}, "value": [0, "100"]},
            {"metric": {"run_id": "12:1"}, "value": [0, "200"]},
        ]

    assert ph.last_failure(query, 'tests/a.py::T::t["q"]') == 200.0
    assert 'test_nodeid="tests/a.py::T::t[\\"q\\"]"' in seen[0]
    assert 'run_id=~"11:1|12:1"' in seen[1]
    assert ph.last_failure(lambda expr: [], NODE) is None


def test_only_fixes_of_this_test_count():
    by_file = index(
        (1, "Add contrastive loss to Gemma3n", None),  # a feature, not a fix
        (2, "Fix Gemma3nVision2TextModelTest::test_model_parallelism split", None),
        (3, "[serge] Fix 2 integration tests for model `gemma3n`", None),
    )
    assert pf.for_test(NODE, 0, by_file)["number"] == 3
    other = (
        "tests/models/gemma3n/test_modeling_gemma3n.py::Gemma3nVision2TextModelTest::"
    )
    assert pf.for_test(other + "test_model_parallelism", 0, by_file)["number"] == 3
    assert pf.for_test(other + "test_model_parallelism", 0, by_file)["others"] == 1
    # The class matches but the named function is another one.
    named = index((2, "Fix Gemma3nVision2TextModelTest::test_model_parallelism", None))
    assert pf.for_test(other + "test_generate", 0, named) is None


FIX = {
    "number": 49147,
    "title": "Fix Gemma3n fp16 loading",
    "url": "https://github.com/huggingface/transformers/pull/49147",
    "verdict": "merged_before",
    "merged_at": MERGED_TS,
    "author": "octocat",
    "others": 0,
}


def test_the_actions_get_the_same_answer_as_the_column():
    from transformersci.otel import new_issue, wdyt, who_can_help

    text = pf.describe(FIX)
    assert "merged on 2026-10-08" in text and "did not fix this failure" in text
    assert "probably fixed" in pf.describe({**FIX, "verdict": "merged_after"})
    assert "Not verified" in pf.describe({**FIX, "verdict": "open"})

    prompt = wdyt.build_prompt(NODE, {}, [], [], FIX)
    assert prompt.endswith("<probable_fix>\n" + text + "\n</probable_fix>")
    assert "<probable_fix>" not in wdyt.build_prompt(NODE, {}, [], [])
    assert "<probable_fix>" in wdyt.SYSTEM_PROMPT
    slow = wdyt.build_slow_prompt(
        NODE, "run_models_gpu", "GPU", [60.0], "2026-10-15", "", FIX
    )
    assert "PR #49147" in slow

    people = [
        {"login": "octocat", "evidence": "reviewed", "url": "u"},
        {"login": "b", "evidence": "e", "url": "u"},
    ]
    ranked = who_can_help.with_fix_author(people, FIX)
    assert [p["login"] for p in ranked] == ["octocat", "b"]
    assert "authored PR #49147" in ranked[0]["evidence"]
    assert who_can_help.with_fix_author(people, None) == people
    assert (
        who_can_help.with_fix_author(people, {**FIX, "author": "bad login"}) == people
    )

    draft = new_issue.slow_draft(
        "alice", NODE, "run_models_gpu", "GPU", [60.0], "2026-10-15", FIX
    )
    assert "Probable fix: [#49147](" in draft["body"]


def test_the_new_issue_model_sees_the_fix(monkeypatch):
    from transformersci.otel import llm, new_issue

    sent = []
    monkeypatch.setattr(
        llm, "config", lambda: {"api_base": "a", "model": "m", "api_key": "k"}
    )
    monkeypatch.setattr(
        llm,
        "chat_json",
        lambda cfg, messages, **kw: (
            sent.append(messages) or {"title": "t", "sections": {}}
        ),
    )
    new_issue.model_draft(NODE, [], {}, [], [], [], [], "", FIX)
    context = sent[0][1]["content"]
    assert (
        '"probable_fix": {"pr": "https://github.com/huggingface/transformers/pull/49147"'
        in context
    )
    assert "probable_fix" in sent[0][0]["content"]
