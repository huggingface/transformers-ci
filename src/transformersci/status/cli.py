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
"""``ci-github-status serve``: the webhook receiver and its /metrics.

The webhook secret is read from the environment (``--secret-env``, default
``GITHUB_WEBHOOK_SECRET``), never from the command line, and the service
refuses to start without one: an unsigned receiver would store anything anyone
posts to a public route.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from collections.abc import Sequence

from .server import Service, serve
from .store import Store
from .webhook import Filters

_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400}


def duration(text: str) -> float:
    """``90s`` / ``6h`` / ``7d``, or bare seconds."""
    text = text.strip()
    try:
        if text and text[-1] in _UNITS:
            return float(text[:-1]) * _UNITS[text[-1]]
        return float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a duration: {text!r}") from None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ci-github-status", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("serve", help="receive webhooks and serve /metrics")
    run.add_argument("--db", required=True, help="SQLite file (on the PVC)")
    run.add_argument("--host", default="0.0.0.0")
    run.add_argument("--port", type=int, default=8080)
    run.add_argument(
        "--repository",
        action="append",
        required=True,
        help="public repository to accept, owner/name (repeatable)",
    )
    run.add_argument(
        "--workflow",
        action="append",
        required=True,
        help="workflow name to follow, e.g. 'PR CI' (repeatable)",
    )
    run.add_argument("--secret-env", default="GITHUB_WEBHOOK_SECRET")
    run.add_argument(
        "--publish-completed",
        type=duration,
        default=duration("6h"),
        help="how long a completed run/job stays in /metrics (default 6h)",
    )
    run.add_argument(
        "--retention",
        type=duration,
        default=duration("7d"),
        help="how long completed state and delivery ids are kept (default 7d)",
    )
    return parser


def _prune_loop(store: Store, retention: float, stop: threading.Event) -> None:
    while not stop.wait(3600):
        try:
            store.prune(retention_seconds=retention)
        except Exception as error:  # keep serving; the next hour retries
            print(
                f"[ci-github-status] prune failed: {error!r}",
                file=sys.stderr,
                flush=True,
            )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    secret = os.environ.get(args.secret_env, "")
    if not secret:
        print(
            f"ci-github-status: ${args.secret_env} is empty; refusing to accept unsigned webhooks",
            file=sys.stderr,
        )
        return 2
    store = Store(args.db)
    store.prune(retention_seconds=args.retention)
    service = Service(
        store=store,
        secret=secret.encode("utf-8"),
        filters=Filters(
            repositories=frozenset(args.repository),
            workflows=frozenset(args.workflow),
        ),
        completed_window_seconds=args.publish_completed,
    )
    stop = threading.Event()
    threading.Thread(
        target=_prune_loop,
        args=(store, args.retention, stop),
        daemon=True,
        name="prune",
    ).start()
    server = serve(service, args.host, args.port)
    print(
        f"[ci-github-status] serving on {args.host}:{args.port} "
        f"(repositories={sorted(args.repository)}, workflows={sorted(args.workflow)}) "
        f"at {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        file=sys.stderr,
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        server.server_close()
        store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
