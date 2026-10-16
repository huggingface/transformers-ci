"""@slow / @is_flaky labels come from the test's source at the commit that ran."""

from __future__ import annotations

import base64

from transformersci.otel import test_markers
from transformersci.otel import test_page

SHA = "a" * 40
FILES = {
    "tests/models/foo/test_modeling_foo.py": """
from transformers.testing_utils import is_flaky, slow
from ...test_modeling_common import ModelTesterMixin


class FooModelTest(ModelTesterMixin, unittest.TestCase):
    @is_flaky(max_attempts=3)
    def test_batching(self):
        pass

    def test_plain(self):
        pass


@slow
class FooIntegrationTest(unittest.TestCase):
    @pytest.mark.flaky
    def test_generate(self):
        pass


@slow
def test_module_level():
    pass
""",
    "tests/test_modeling_common.py": """
class ModelTesterMixin:
    @slow
    def test_model_from_pretrained(self):
        pass
""",
}


def fake_get(urls):
    def get(url):
        urls.append(url)
        path = url.split("/contents/", 1)[1].split("?", 1)[0]
        if path not in FILES:
            raise OSError("404")
        return {
            "encoding": "base64",
            "content": base64.b64encode(FILES[path].encode()).decode(),
        }

    return get


def setup_function():
    test_markers._files.clear()


def labels(node, commit=SHA, urls=None):
    return test_markers.markers(node, commit, fake_get([] if urls is None else urls))


def test_method_class_and_module_level_decorators():
    base = "tests/models/foo/test_modeling_foo.py"
    assert labels(base + "::FooModelTest::test_batching") == ["flaky"]
    assert labels(base + "::FooModelTest::test_plain") == []
    assert labels(base + "::FooIntegrationTest::test_generate") == ["slow", "flaky"]
    assert labels(base + "::test_module_level") == ["slow"]


def test_inherited_method_is_read_from_the_imported_mixin_at_the_same_commit():
    urls = []
    node = "tests/models/foo/test_modeling_foo.py::FooModelTest::test_model_from_pretrained[x]"
    assert labels(node, urls=urls) == ["slow"]
    assert urls[-1].endswith("/contents/tests/test_modeling_common.py?ref=" + SHA)


def test_files_are_cached_and_main_is_the_fallback_ref():
    urls = []
    node = "tests/models/foo/test_modeling_foo.py::FooModelTest::test_batching"
    labels(node, commit="", urls=urls)
    labels(node, commit="", urls=urls)
    assert len(urls) == 1 and urls[0].endswith("?ref=main")


def test_bad_or_missing_input_gives_no_labels():
    assert labels("src/transformers/x.py::test_a") == []
    assert labels("tests/../etc/x.py::test_a") == []
    assert labels("tests/missing.py::T::test_a") == []


def test_page_has_a_markers_box_and_passes_the_commit():
    html = test_page.render_test_page(
        {"test_nodeid": "tests/a.py::T::test_x", "commit": SHA, "details": []}
    )
    assert "id='markers'" in html and "/failure/markers?" in html
    assert f'"commit": "{SHA}"' in html


def test_markers_endpoint(monkeypatch):
    import json
    import threading
    from http.server import ThreadingHTTPServer
    from urllib.error import HTTPError
    from urllib.parse import urlencode
    from urllib.request import urlopen

    from transformersci.otel import trace_exporter as te

    monkeypatch.setattr(te, "_github_api_get", fake_get([]))
    server = ThreadingHTTPServer(("127.0.0.1", 0), te.MetricsHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}/failure/markers?"
    try:
        node = (
            "tests/models/foo/test_modeling_foo.py::FooIntegrationTest::test_generate"
        )
        with urlopen(base + urlencode({"test_nodeid": node, "commit": SHA})) as reply:
            assert json.load(reply) == {"markers": ["slow", "flaky"]}
            assert "max-age=86400" in reply.headers["Cache-Control"]
        try:
            urlopen(base + urlencode({"test_nodeid": node, "commit": "main;x"}))
        except HTTPError as error:
            assert error.code == 400
        else:
            raise AssertionError("bad commit accepted")
    finally:
        server.shutdown()
        server.server_close()
