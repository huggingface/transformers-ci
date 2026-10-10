#!/usr/bin/env python3
"""Serve patch routes from the working tree and proxy local Grafana.

Run with PYTHONPATH=src .venv/bin/python deploy/scripts/preview-patch-view.py.
The Grafana container should listen on 127.0.0.1:3301.
"""

from http.server import ThreadingHTTPServer
import argparse
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

from transformersci.otel.trace_exporter import MetricsHandler
from transformersci.otel import patch_view


class PreviewHandler(MetricsHandler):
    fixture_dir: Path | None = None

    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)
        if self.fixture_dir and parsed.path in {
            "/patch-view/data",
            "/patch-view/comments",
        }:
            params = parse_qs(parsed.query)
            repository = (params.get("repository") or ["huggingface/transformers"])[0]
            pr = (params.get("pr") or [""])[0]
            if (
                repository == "huggingface/transformers"
                and pr.isascii()
                and pr.isdigit()
            ):
                name = "pr.diff" if parsed.path.endswith("/data") else "comments.json"
                fixture = self.fixture_dir / pr / name
                if fixture.is_file():
                    payload = fixture.read_bytes()
                    self.send_response(200)
                    self.send_header(
                        "Content-Type",
                        "text/plain; charset=utf-8"
                        if name == "pr.diff"
                        else "application/json; charset=utf-8",
                    )
                    self.send_header("Content-Length", str(len(payload)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(payload)
                    return
        if self.path.startswith("/patch-view"):
            # Read assets on each request so browser refresh picks up edits.
            patch_view.PAGE_HTML = (
                Path(patch_view.__file__).with_suffix(".html").read_text()
            )
            patch_view.SCRIPT = Path(patch_view.__file__).with_suffix(".js").read_text()
            return super().do_GET()
        request = Request(
            "http://127.0.0.1:3301" + self.path,
            headers={"Host": self.headers.get("Host", "localhost:8765")},
        )
        try:
            response = urlopen(request, timeout=20)
        except HTTPError as exc:
            response = exc
        except URLError:
            self.send_error(502, "Local Grafana is starting; refresh shortly.")
            return
        with response:
            body = response.read()
            self.send_response(response.status)
            for name, value in response.headers.items():
                if name.lower() not in {
                    "transfer-encoding",
                    "connection",
                    "content-length",
                }:
                    self.send_header(name, value)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--fixture-dir",
        type=Path,
        help="Serve downloaded PR snapshots before querying GitHub",
    )
    args = parser.parse_args()
    PreviewHandler.fixture_dir = args.fixture_dir
    print(
        f"Local preview: http://localhost:{args.port}/d/patch-preview/pr-diff",
        flush=True,
    )
    ThreadingHTTPServer(("127.0.0.1", args.port), PreviewHandler).serve_forever()
