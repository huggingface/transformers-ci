"""Issue drafts use the repository template and need a signed-in publisher."""

from __future__ import annotations

import base64
import json

from transformersci.otel import new_issue
from transformersci.otel import trace_exporter as te

NODE = "tests/generation/test_streamers.py::StreamerTest::test_stream"
TRACE = "a" * 32
TEMPLATE = """body:
  - type: textarea
    id: system-info
    attributes:
      label: System Info
    validations:
      required: true
  - type: checkboxes
    id: information
    attributes:
      label: Information
      options:
        - label: Official examples
  - type: textarea
    id: reproduction
    attributes:
      label: Reproduction
    validations:
      required: true
  - type: textarea
    id: expected-behavior
    attributes:
      label: Expected behavior
    validations:
      required: true
"""


def setup_function():
    new_issue._drafts.clear()
    new_issue._draft_cache.clear()
    new_issue._asked.clear()


def test_template_and_draft_keep_required_sections_and_trace(monkeypatch):
    urls = []

    def get(url):
        urls.append(url)
        return {
            "encoding": "base64",
            "content": base64.b64encode(TEMPLATE.encode()).decode(),
        }

    fields = new_issue.bug_template(get)
    assert [field["id"] for field in fields] == [
        "system-info",
        "information",
        "reproduction",
        "expected-behavior",
    ]
    assert urls == [
        "https://api.github.com/repos/huggingface/transformers/contents/.github/ISSUE_TEMPLATE/bug-report.yml"
    ]
    monkeypatch.setattr(
        new_issue.who_can_help,
        "source_line",
        lambda *args: ("tests/generation/test_streamers.py", 10),
    )
    monkeypatch.setattr(new_issue.who_can_help, "why", lambda *args: {"number": 42})
    seen_evidence = []
    monkeypatch.setattr(new_issue.related_issues, "lookup", lambda *args: {"hits": []})
    helper = {
        "login": "maintainer",
        "why": "Reviewed the affected line",
        "evidence": "PR #7",
        "url": "https://github.com/huggingface/transformers/pull/7",
    }
    monkeypatch.setattr(
        new_issue.who_can_help,
        "suggest",
        lambda *args, **kwargs: (
            seen_evidence.append(kwargs["evidence"])
            or {"status": "ok", "people": [helper]}
        ),
    )
    captured = []
    monkeypatch.setattr(
        new_issue,
        "model_draft",
        lambda *args: (
            captured.append(args[6])
            or {
                "title": "Streamer fails in CI",
                "sections": {
                    "system-info": "",
                    "reproduction": "Observed in CI",
                    "expected-behavior": "Should pass",
                },
            }
        ),
    )
    events = []
    draft = new_issue.prepare(
        "alice",
        TRACE,
        NODE,
        [{"exception_message": "bad stream"}],
        get,
        emit=events.append,
    )
    assert draft["status"] == "ok"
    assert captured == [[]]
    # The helpers reuse the draft's lookups instead of repeating them.
    assert seen_evidence[0][:2] == (
        ("tests/generation/test_streamers.py", 10),
        {"number": 42},
    )
    assert "- @maintainer" not in draft["body"]  # template has no such field
    assert [event["text"] for event in events if event["event"] == "step"] == [
        "Reading the Transformers bug report template",
        "Reading Relore history for the failing line",
        "Searching related issues and PRs",
        "Reading the failing test's code",
        "Checking recent changes to the test file",
        "Finding people who can help (slow)",
        "Drafting the issue (slow)",
        "Preparing the editable issue form",
    ]
    assert draft["title"] == "Streamer fails in CI"
    assert "### System Info" in draft["body"]
    assert "### Reproduction" in draft["body"]
    assert "### Expected behavior" in draft["body"]
    assert "- [ ] Official examples" in draft["body"]
    assert "bad stream" in draft["body"] and "var-trace_id=" + TRACE in draft["body"]


def test_cached_draft_reuses_model_output_with_distinct_publish_ids(monkeypatch):
    calls = []
    monkeypatch.setattr(
        new_issue,
        "bug_template",
        lambda *args: [
            {
                "id": "reproduction",
                "type": "textarea",
                "label": "Reproduction",
                "required": True,
            }
        ],
    )
    monkeypatch.setattr(new_issue.who_can_help, "source_line", lambda *args: None)
    monkeypatch.setattr(
        new_issue.who_can_help, "frequent_test_modifier", lambda *args: []
    )
    monkeypatch.setattr(new_issue.related_issues, "lookup", lambda *args: {"hits": []})
    monkeypatch.setattr(
        new_issue.who_can_help,
        "suggest",
        lambda *args, **kwargs: {"status": "no_people"},
    )

    def draft(*args):
        calls.append(1)
        return {"title": "Failure", "sections": {"reproduction": "Steps"}}

    monkeypatch.setattr(new_issue, "model_draft", draft)
    first = new_issue.prepare("alice", TRACE, NODE, [{}], lambda url: {})
    events = []
    second = new_issue.prepare("bob", TRACE, NODE, [{}], lambda url: {}, events.append)
    assert first["status"] == second["status"] == "ok"
    assert first["draft_id"] != second["draft_id"]
    assert first["body"] == second["body"]
    assert calls == [1]
    assert events == [
        {"event": "step", "text": "Using an earlier draft for this failure"}
    ]
    assert new_issue._drafts[first["draft_id"]][1] == "alice"
    assert new_issue._drafts[second["draft_id"]][1] == "bob"


def test_draft_uses_recent_test_modifier_when_relore_has_no_people(monkeypatch):
    monkeypatch.setattr(
        new_issue,
        "bug_template",
        lambda *args: [
            {
                "id": "reproduction",
                "type": "textarea",
                "label": "Reproduction",
                "required": True,
            }
        ],
    )
    monkeypatch.setattr(new_issue.who_can_help, "source_line", lambda *args: None)
    monkeypatch.setattr(new_issue.related_issues, "lookup", lambda *args: {"hits": []})
    monkeypatch.setattr(
        new_issue.who_can_help,
        "suggest",
        lambda *args, **kwargs: {"status": "no_people"},
    )
    monkeypatch.setattr(
        new_issue.who_can_help,
        "frequent_test_modifier",
        lambda *args: [
            {
                "login": "alice",
                "evidence": "recent edits",
                "url": "https://github.com/x",
            }
        ],
    )
    seen = []

    def draft(*args):
        seen.extend(args[5])
        return {"title": "Failure", "sections": {"reproduction": "Steps"}}

    monkeypatch.setattr(new_issue, "model_draft", draft)
    result = new_issue.prepare("alice", TRACE, NODE, [{}], lambda url: {})
    assert result["status"] == "ok"
    assert seen[0]["login"] == "alice"


def test_publish_uses_edited_text_once_and_checks_login():
    draft_id = "f" * 32
    new_issue._drafts[draft_id] = (float("inf"), "alice")
    calls = []

    def create(title, body):
        calls.append((title, body))
        return {
            "number": 123,
            "html_url": "https://github.com/huggingface/transformers/issues/123",
        }

    assert new_issue.publish("bob", draft_id, "Edited", "Edited body", create) == {
        "status": "invalid"
    }
    assert new_issue.publish("alice", draft_id, "Edited", "Edited body", create) == {
        "status": "ok",
        "number": 123,
        "url": "https://github.com/huggingface/transformers/issues/123",
    }
    assert calls == [("Edited", "Edited body")]
    assert new_issue.publish("alice", draft_id, "Again", "Again", create) == {
        "status": "invalid"
    }


def test_failed_publish_keeps_draft_for_retry():
    draft_id = "e" * 32
    new_issue._drafts[draft_id] = (float("inf"), "alice")

    def fail(title, body):
        raise OSError("GitHub unavailable")

    assert new_issue.publish("alice", draft_id, "Title", "Body", fail) == {
        "status": "unavailable"
    }
    assert draft_id in new_issue._drafts


def test_publish_uses_write_token_not_read_only_app(monkeypatch):
    monkeypatch.setenv("PYTEST_GITHUB_TOKEN", "write-token")
    monkeypatch.setattr(
        te,
        "github_api_token",
        lambda: (_ for _ in ()).throw(AssertionError("read-only App used")),
    )
    sent = []

    class Reply:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, size):
            return b'{"number": 7}'

    def open_request(request, timeout):
        sent.append(request)
        return Reply()

    monkeypatch.setattr(te, "urlopen", open_request)
    assert te._github_create_issue("Edited", "Edited body") == {"number": 7}
    request = sent[0]
    assert request.get_method() == "POST"
    assert request.get_header("Authorization") == "Bearer write-token"
    assert json.loads(request.data) == {
        "title": "Edited",
        "body": "Edited body",
        "labels": ["bug"],
    }


def _trace(nodeid=NODE):
    tags = {
        "vcs.ref.head.revision": "a" * 40,
        "vcs.ref.head.name": "main",
        "service.version": "b" * 40,
        "process.runtime.name": "cpython",
        "process.runtime.version": "3.11.17",
        "transformers.test.ci_event": "daily",
        "transformers.test.job": "run_models_gpu",
        "transformers.test.hardware": "single-gpu",
        "transformers.test.runner.type": "aws-g5-4xlarge-cache",
        "transformers.test.runner.cpu_model": "AMD EPYC 7R32",
        "transformers.test.runner.vcpus": "16",
        "transformers.test.runner.memory_gib": "62",
        "transformers.test.runner.gpu_vendor": "nvidia",
        "transformers.test.runner.gpu_model": "NVIDIA A10G",
        "transformers.test.runner.gpu_count": "1",
        "transformers.test.runner.gpu_memory_gib": "22",
        "vcs.repository.name": "huggingface/transformers",
    }
    return {
        "processes": {
            "p1": {"tags": [{"key": k, "value": v} for k, v in tags.items()]},
            "p2": {"tags": [{"key": "process.runtime.version", "value": "3.9"}]},
        },
        "spans": [
            {
                "processID": "p2",
                "operationName": "other",
                "tags": [
                    {"key": "pytest.span_type", "value": "test"},
                    {"key": "pytest.nodeid", "value": "tests/other.py::test_x"},
                ],
            },
            {
                "processID": "p1",
                "operationName": nodeid,
                "tags": [
                    {"key": "pytest.span_type", "value": "test"},
                    {"key": "pytest.nodeid", "value": nodeid},
                ],
            },
        ],
    }


def test_trace_environment_reads_the_failing_tests_process():
    env = new_issue.trace_environment(_trace(), NODE)
    assert env["python"] == "3.11.17"
    assert env["commit"] == "a" * 40
    assert env["own_commit"] == "b" * 40
    assert env["gpu_model"] == "NVIDIA A10G"
    assert new_issue.trace_environment(_trace(), "tests/missing.py::t") == {}


def test_system_info_and_command_come_from_the_trace_and_footer_says_ai():
    env = new_issue.trace_environment(_trace(), NODE)
    env.pop("own_commit")
    env["torch"] = "2.14.0"
    env["job_url"] = "https://github.com/huggingface/transformers/actions/runs/1/job/2"
    fields = [
        {
            "id": "system-info",
            "type": "textarea",
            "label": "System Info",
            "required": True,
        },
        {
            "id": "reproduction",
            "type": "textarea",
            "label": "Reproduction",
            "required": True,
        },
    ]
    body = new_issue.render_body(
        fields,
        {
            "system-info": "Not provided in the supplied evidence.",
            "reproduction": "snippet",
        },
        TRACE,
        NODE,
        [{"exception_message": "bad"}],
        env,
    )
    assert "Not provided" not in body
    assert "- Python version: 3.11.17 (CPython)" in body
    assert "- PyTorch version: 2.14.0 (pinned by the GPU CI image)" in body
    assert "- GPU: 1× NVIDIA A10G, 22 GiB" in body
    assert "- CPU: AMD EPYC 7R32, 16 vCPUs, 62 GiB RAM" in body
    assert "`run_models_gpu` (daily) on `aws-g5-4xlarge-cache` [single-gpu]" in body
    assert "actions/runs/1/job/2" in body
    assert f"git checkout {'a' * 40}\nRUN_SLOW=1 python -m pytest '{NODE}'" in body
    assert body.index("snippet") < body.index("RUN_SLOW=1")
    assert body.rstrip().endswith("Please check it before relying on it._")


def test_system_info_falls_back_without_trace_facts():
    fields = [
        {
            "id": "system-info",
            "type": "textarea",
            "label": "System Info",
            "required": True,
        }
    ]
    body = new_issue.render_body(fields, {}, TRACE, NODE, [], {})
    assert "exact versions were not recorded" in body


def test_gpu_image_torch_reads_the_dockerfile_at_the_commit():
    urls = []

    def get(url):
        urls.append(url)
        text = "FROM x\nARG PYTORCH='2.14.0'\n"
        return {
            "encoding": "base64",
            "content": base64.b64encode(text.encode()).decode(),
        }

    assert new_issue.gpu_image_torch("c" * 40, get) == "2.14.0"
    assert urls[0].endswith(
        "/contents/docker/transformers-all-latest-gpu/Dockerfile?ref=" + "c" * 40
    )
    assert new_issue.gpu_image_torch("main", get) == ""


def test_failing_test_source_returns_the_enclosing_test_function():
    urls = []
    text = "\n".join(
        [
            "class T:",
            "    @slow",
            "    def test_a(self):",
            "        x = load(",
            "            'ckpt',",
            "        )",
            "        assert x == 1",
            "",
            "    def test_b(self):",
            "        pass",
        ]
    )

    def get(url):
        urls.append(url)
        return {
            "encoding": "base64",
            "content": base64.b64encode(text.encode()).decode(),
        }

    details = [
        {
            "exception_stacktrace": "tests/a/test_b.py:7: \n"
            "src/transformers/x.py:3: in load\nsrc/transformers/y.py:9: OSError"
        }
    ]
    source = new_issue.failing_test_source(
        "tests/a/test_b.py::T::test_a", details, "d" * 40, get
    )
    assert source.splitlines()[0] == "    @slow"
    assert "assert x == 1" in source and "test_b" not in source
    assert urls == [
        "https://api.github.com/repos/huggingface/transformers/contents/tests/a/test_b.py?ref="
        + "d" * 40
    ]


def test_issue_environment_uses_the_checked_out_commit_for_run_slow():
    daily = te.issue_environment(_trace(), NODE)
    assert daily["commit"] == "a" * 40 and daily["branch"] == "main"
    assert "own_commit" not in daily
    trace = _trace()
    for tag in trace["processes"]["p1"]["tags"]:
        if tag["key"] == "transformers.test.ci_event":
            tag["value"] = te.RUN_SLOW_CI_EVENT
    run_slow = te.issue_environment(trace, NODE)
    assert run_slow["commit"] == "b" * 40 and "branch" not in run_slow


def test_slow_steps_run_in_parallel_keep_alive_and_ask_for_patience(monkeypatch):
    import threading

    monkeypatch.setattr(new_issue, "KEEPALIVE_SECONDS", 0.05)
    monkeypatch.setattr(new_issue, "PATIENCE_SECONDS", 0.1)
    both_running = threading.Barrier(2, timeout=5)
    release = threading.Event()
    events = []

    def people():
        both_running.wait()
        return {"status": "ok"}

    def draft():
        both_running.wait()
        release.wait(5)
        raise ValueError("bad draft")

    def emit(event):
        events.append(event)
        if event["event"] == "step_text":
            release.set()

    found, drafted = new_issue._in_parallel(
        [("People (slow)", people), ("Draft (slow)", draft)], emit
    )
    assert found == {"status": "ok"}
    assert isinstance(drafted, ValueError)
    assert events[:2] == [
        {"event": "step", "text": "People (slow)", "parallel": False},
        {"event": "step", "text": "Draft (slow)", "parallel": True},
    ]
    assert {"event": "step_done", "text": "People (slow)"} in events
    assert {
        "event": "step_text",
        "text": "Draft (slow)",
        "to": "Draft (bear with us!)",
    } in events
    assert events[-1] == {"event": "step_done", "text": "Draft (bear with us!)"}
    assert {"event": "keepalive"} in events


def test_who_can_help_comes_from_the_helper_findings():
    fields = [
        {
            "id": "who-can-help",
            "type": "textarea",
            "label": "Who can help?",
            "required": False,
        }
    ]
    helpers = [
        {
            "login": "maintainer",
            "why": "Reviewed the affected line",
            "url": "https://github.com/huggingface/transformers/pull/7",
        },
        {"login": "bad login!", "why": "x"},
    ]
    body = new_issue.render_body(
        fields, {"who-can-help": "model text"}, TRACE, NODE, [], {}, helpers
    )
    assert (
        "- @maintainer: Reviewed the affected line "
        "([evidence](https://github.com/huggingface/transformers/pull/7))"
    ) in body
    assert "model text" not in body and "bad login" not in body
    body = new_issue.render_body(
        fields, {"who-can-help": "model text"}, TRACE, NODE, [], {}, []
    )
    assert "model text" in body


WORK_FIX = {
    "number": 49147,
    "url": "https://github.com/huggingface/transformers/pull/49147",
    "title": "Fix the streamer for @someone",
    "author": "octocat",
    "verdict": "open",
}


def test_work_on_it_lists_related_threads_and_names_no_one():
    fields = [
        {
            "id": "who-can-help",
            "type": "textarea",
            "label": "Who can help?",
            "required": False,
        },
        {
            "id": "reproduction",
            "type": "textarea",
            "label": "Reproduction",
            "required": True,
        },
        {"id": "expected", "type": "textarea", "label": "Expected", "required": False},
    ]
    hits = [
        {
            "number": 7,
            "type": "issue",
            "url": "https://github.com/huggingface/transformers/issues/7",
            "title": "Streamer hangs",
        },
        {
            "number": 49147,
            "type": "pr",
            "url": WORK_FIX["url"],
            "title": "dup of the fix",
        },
    ]
    history = {
        "number": 12,
        "thread": {"title": "Refactor streamers", "author": "author"},
    }
    related = new_issue.related_text(hits, history, WORK_FIX)
    assert related.splitlines() == [
        "- [#49147](https://github.com/huggingface/transformers/pull/49147) Fix the streamer for `@someone` (probable fix, open)",
        "- [#12](https://github.com/huggingface/transformers/pull/12) Refactor streamers (last changed the failing line)",
        "- [#7](https://github.com/huggingface/transformers/issues/7) Streamer hangs (issue)",
    ]
    body = new_issue.render_body(
        fields,
        {"who-can-help": "ask @maintainer", "expected": "cc @x, mail a@b.co"},
        TRACE,
        NODE,
        [],
        {},
        related=related,
        take="It looks like a race. @octocat knows.",
    )
    assert "### Who can help?" not in body and "### Related\n\n- [#49147]" in body
    assert "ask" not in body and "octocat" not in body.replace("`@octocat`", "")
    assert "cc `@x`, mail a@b.co" in body
    # Serge's take sits right after Reproduction.
    assert (
        body.index("### Reproduction")
        < body.index("### What Serge thinks\n\nIt looks like a race.")
        < body.index("### Expected")
    )
    empty = new_issue.render_body(fields, {}, TRACE, NODE, [], {}, related="")
    assert (
        "_No related issues or PRs found._" in empty
        and "What Serge thinks" not in empty
    )


def test_work_on_it_slow_draft():
    plain = new_issue.slow_draft(
        "alice", NODE, "run_models_gpu", "GPU", [600.0], "2026-10-15", WORK_FIX
    )
    assert (
        "### Who can help?" in plain["body"]
        and "What Serge thinks" not in plain["body"]
    )
    work = new_issue.slow_draft(
        "alice",
        NODE,
        "run_models_gpu",
        "GPU",
        [600.0],
        "2026-10-15",
        WORK_FIX,
        "Use a tiny config, says @x.",
    )["body"]
    assert "### Who can help?" not in work and "### Related\n\n- [#49147]" in work
    assert "octocat" not in work and "says `@x`." in work
    assert (
        work.index("--durations=0")
        < work.index("### What Serge thinks")
        < work.index("### Expected behavior")
    )


def test_work_on_it_prepare_adds_serge_take_and_skips_people(monkeypatch):
    fields = [
        {
            "id": "who-can-help",
            "type": "textarea",
            "label": "Who can help?",
            "required": False,
        },
        {
            "id": "reproduction",
            "type": "textarea",
            "label": "Reproduction",
            "required": True,
        },
    ]
    monkeypatch.setattr(new_issue, "bug_template", lambda *args: fields)
    monkeypatch.setattr(new_issue.who_can_help, "source_line", lambda *args: None)
    hit = {
        "number": 7,
        "type": "issue",
        "url": "https://github.com/huggingface/transformers/issues/7",
        "title": "Streamer hangs",
        "author": "someone",
        "trust": "authoritative",
    }
    monkeypatch.setattr(
        new_issue.related_issues, "lookup", lambda *args: {"hits": [hit]}
    )

    def no_people(*args, **kwargs):
        raise AssertionError("Work on it looks up no people")

    monkeypatch.setattr(new_issue.who_can_help, "suggest", no_people)
    monkeypatch.setattr(new_issue.who_can_help, "frequent_test_modifier", no_people)
    asked = []
    monkeypatch.setattr(
        new_issue.wdyt,
        "answer",
        lambda login, trace_id, nodeid, make_prompt, *rest, **kw: (
            asked.append(make_prompt(lambda event: None))
            or {"status": "ok", "answer": "A race in the streamer."}
        ),
    )
    drafted = []
    monkeypatch.setattr(
        new_issue,
        "model_draft",
        lambda *args, **kwargs: (
            drafted.append((args, kwargs))
            or {
                "title": "Streamer fails",
                "sections": {"reproduction": "Observed in CI"},
            }
        ),
    )
    events = []
    draft = new_issue.prepare(
        "alice",
        "c" * 32,
        NODE,
        [{"exception_message": "bad stream"}],
        lambda url: {},
        events.append,
        fix=WORK_FIX,
        work=True,
    )
    assert draft["status"] == "ok"
    body = draft["body"]
    assert "### What Serge thinks\n\nA race in the streamer." in body
    assert "### Related\n\n- [#49147]" in body and "- [#7]" in body
    assert "@" not in body.replace("`@someone`", "")
    # WDYT? saw the same related threads, plus the fix PR.
    assert [thread["number"] for thread in asked[0][1]] == [7, 49147]
    args, kwargs = drafted[0]
    assert kwargs == {"contacts": False} and args[8]["author"] == ""
    steps = [event["text"] for event in events if event["event"] == "step"]
    assert "Asking Serge what it thinks (slow)" in steps
    assert "Finding people who can help (slow)" not in steps
