#!/usr/bin/env python3
"""Query the transformers-ci Tempo (and Prometheus) datasources from the CLI.

Everything goes through the public, read-only Grafana datasource proxy, so no
kubectl / cluster access is needed. Vanilla Python 3.10 + stdlib only.

Why this exists: a single sharded pytest run can emit a trace larger than the
16 MB the Grafana proxy will return whole ("response larger than the max"). The
``spans`` / ``status`` / ``count`` subcommands use TraceQL span-selection and
metrics, so you can inspect an arbitrarily large trace without ever fetching it
in full.

Examples
--------
  # Which shard traces exist for a workflow run?
  deploy/scripts/tempo.py search '{ resource.cicd.pipeline.run.id =~ "28642749564.*" }'

  # Did a specific test ever emit an ERROR span? (the dashboard only counts those)
  deploy/scripts/tempo.py status '{ span.pytest.nodeid =~ ".*regnet.*can_load_ignoring_mismatched.*" }' --since 7d

  # Inspect one test's span status inside a >16 MB trace without fetching it:
  deploy/scripts/tempo.py spans '{ resource.cicd.pipeline.run.id =~ "28642749564.*" && span.pytest.nodeid =~ ".*ignoring_mismatched.*" }'

  # The exact number the dashboard uses for the Fail column:
  deploy/scripts/tempo.py promql 'max by (test_job) (last_over_time(pytest_run_job_failed_tests{run_id=~"28642749564.*"}[120d]))'

Env vars
--------
  TRANSFORMERS_CI_GRAFANA_URL   Grafana base URL (default: prod)
  TEMPO_DATASOURCE_UID          Tempo datasource uid (default: tempo)
  PROM_DATASOURCE_UID           Prometheus datasource uid (default: prometheus)
  GRAFANA_TOKEN                 Bearer token, for non-public endpoints
  GRAFANA_USER / GRAFANA_PASSWORD   Basic-auth fallback
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_BASE = os.environ.get(
    "TRANSFORMERS_CI_GRAFANA_URL", "https://transformers-ci.lor-e.huggingface.cool"
)
TEMPO_UID = os.environ.get("TEMPO_DATASOURCE_UID", "tempo")
PROM_UID = os.environ.get("PROM_DATASOURCE_UID", "prometheus")

_UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}


def duration_to_seconds(text: str) -> int:
    """Parse '30m' / '24h' / '7d' (or bare seconds) into an int of seconds."""
    text = text.strip()
    if text and text[-1] in _UNIT_SECONDS:
        return int(float(text[:-1]) * _UNIT_SECONDS[text[-1]])
    return int(float(text))


def auth_header() -> dict[str, str]:
    token = os.environ.get("GRAFANA_TOKEN", "")
    if token:
        return {"Authorization": f"Bearer {token}"}
    user = os.environ.get("GRAFANA_USER", "")
    if user:
        raw = f"{user}:{os.environ.get('GRAFANA_PASSWORD', '')}".encode()
        return {"Authorization": "Basic " + base64.b64encode(raw).decode()}
    return {}


def proxy_get(base: str, uid: str, path: str, params: dict | None = None) -> str:
    """GET a datasource-proxy path and return the raw response body as text."""
    url = f"{base}/api/datasources/proxy/uid/{uid}{path}"
    if params:
        # Drop None values so callers can pass optional params uniformly.
        clean = {k: v for k, v in params.items() if v is not None}
        url = f"{url}?{urllib.parse.urlencode(clean)}"
    req = urllib.request.Request(
        url, headers={"Accept": "application/json", **auth_header()}
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            return resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        raise SystemExit(f"HTTP {exc.code} for {url}\n{body}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"request failed for {url}: {exc.reason}") from exc


def proxy_get_json(base: str, uid: str, path: str, params: dict | None = None) -> dict:
    body = proxy_get(base, uid, path, params)
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        # Tempo returns plain-text errors (e.g. the >16 MB refusal) with 200.
        raise SystemExit(body.strip() or "empty / non-JSON response")


def window(args) -> tuple[int, int]:
    """Resolve (start, end) unix seconds from --from/--to/--since."""
    end = args.to if args.to is not None else int(time.time())
    if getattr(args, "from_ts", None) is not None:
        start = args.from_ts
    else:
        start = end - duration_to_seconds(args.since)
    return start, end


# Tempo's search API rejects ranges wider than 26h (HTTP 400). Clamp to the cap
# but keep the requested end, so an old run is still reachable via --to.
TEMPO_SEARCH_MAX_SECONDS = 26 * 3600 - 60


def tempo_window(args, max_seconds: int = TEMPO_SEARCH_MAX_SECONDS) -> tuple[int, int]:
    start, end = window(args)
    if end - start > max_seconds:
        start = end - max_seconds
        hours = max_seconds // 3600
        print(
            f"note: Tempo caps this query at {hours}h; clamped window to "
            f"[{start}, {end}]. Target an older run with --to <unix-seconds>.",
            file=sys.stderr,
        )
    return start, end


def attr_value(value):
    """Flatten an OTLP AnyValue dict to a scalar."""
    if not isinstance(value, dict):
        return value
    for key in ("stringValue", "intValue", "boolValue", "doubleValue"):
        if key in value:
            return value[key]
    return value.get("value")


# ---------------------------------------------------------------------------
# subcommands
# ---------------------------------------------------------------------------
def cmd_search(args) -> None:
    start, end = tempo_window(args)
    data = proxy_get_json(
        args.base,
        TEMPO_UID,
        "/api/search",
        {"q": args.query, "limit": args.limit, "start": start, "end": end},
    )
    if args.json:
        print(json.dumps(data, indent=2))
        return
    traces = data.get("traces") or []
    meta = data.get("metrics") or {}
    print(f"{len(traces)} trace(s)  |  inspectedBytes={meta.get('inspectedBytes')}")
    for t in traces:
        print(
            f"  {t.get('traceID')}  {t.get('rootServiceName', '-')}  "
            f"{t.get('rootTraceName', '-')}  {t.get('durationMs')}ms"
        )


def _search_spans(
    args, extra_select: str = "status, span.pytest.nodeid, name"
) -> tuple[list[dict], int]:
    """Run a TraceQL search and return (matched spans, trace count).

    We tally client-side from the search results rather than using Tempo's
    TraceQL metrics endpoint: this deployment has no metrics-generator, so
    ``count_over_time()`` returns "empty ring". Counts are bounded by --limit
    traces; raise it if you need exhaustive totals.
    """
    start, end = tempo_window(args)
    query = args.query
    if extra_select and "select(" not in query:
        query = f"{query} | select({extra_select})"
    data = proxy_get_json(
        args.base,
        TEMPO_UID,
        "/api/search",
        {"q": query, "limit": args.limit, "start": start, "end": end},
    )
    if getattr(args, "json", False):
        print(json.dumps(data, indent=2))
        return [], 0
    traces = data.get("traces") or []
    spans: list[dict] = []
    for t in traces:
        spansets = t.get("spanSets") or ([t["spanSet"]] if t.get("spanSet") else [])
        for sset in spansets:
            for s in sset.get("spans") or []:
                attrs = {
                    a.get("key"): attr_value(a.get("value") or {})
                    for a in s.get("attributes", [])
                }
                spans.append(
                    {
                        "trace_id": str(t.get("traceID") or ""),
                        "status": str(
                            s.get("status") or attrs.get("status") or "unset"
                        ),
                        "label": attrs.get("span.pytest.nodeid") or s.get("name", "-"),
                    }
                )
    return spans, len(traces)


def cmd_spans(args) -> None:
    spans, n_traces = _search_spans(args)
    if getattr(args, "json", False):
        return  # already printed by _search_spans
    for s in spans:
        print(f"  {s['trace_id'][:16]}  status={s['status']:14} {s['label']}")
    print(f"{len(spans)} span(s) across {n_traces} trace(s)")


def cmd_status(args) -> None:
    spans, n_traces = _search_spans(args)
    if getattr(args, "json", False):
        return
    if not spans:
        print("no matching spans in window")
        return
    counts: dict[str, int] = {}
    for s in spans:
        counts[s["status"]] = counts.get(s["status"], 0) + 1
    print(f"span count by status ({len(spans)} spans across {n_traces} traces):")
    for status, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {status:14} {count}")


def cmd_count(args) -> None:
    spans, n_traces = _search_spans(args)
    if getattr(args, "json", False):
        return
    print(f"{len(spans)} span(s) across {n_traces} trace(s)")


def cmd_trace(args) -> None:
    try:
        body = proxy_get(args.base, TEMPO_UID, f"/api/traces/{args.trace_id}")
    except SystemExit as exc:
        # Tempo returns HTTP 500 with this body when a trace exceeds the proxy cap.
        if "larger than the max" in str(exc):
            raise SystemExit(
                f"trace {args.trace_id} is too large for the proxy (>16 MB).\n"
                "Use `spans`/`status` with a span filter instead of fetching it whole."
            ) from exc
        raise
    if "larger than the max" in body:
        raise SystemExit(
            f"trace {args.trace_id} is too large for the proxy (>16 MB).\n"
            "Use `spans`/`status` with a span filter instead of fetching it whole."
        )
    try:
        print(json.dumps(json.loads(body), indent=2))
    except json.JSONDecodeError:
        print(body)


def cmd_tags(args) -> None:
    params = {"scope": args.scope} if args.scope else None
    print(
        json.dumps(
            proxy_get_json(args.base, TEMPO_UID, "/api/v2/search/tags", params),
            indent=2,
        )
    )


def cmd_values(args) -> None:
    data = proxy_get_json(
        args.base,
        TEMPO_UID,
        f"/api/v2/search/tag/{args.tag}/values",
        {"limit": args.limit},
    )
    if args.json:
        print(json.dumps(data, indent=2))
        return
    for entry in data.get("tagValues") or []:
        print(f"  {entry.get('value') if isinstance(entry, dict) else entry}")


def cmd_promql(args) -> None:
    data = proxy_get_json(args.base, PROM_UID, "/api/v1/query", {"query": args.query})
    if args.json:
        print(json.dumps(data, indent=2))
        return
    if data.get("status") != "success":
        raise SystemExit(f"query error: {data.get('error')}")
    result = (data.get("data") or {}).get("result") or []
    for r in result:
        metric = r.get("metric", {})
        value = r.get("value", ["", "-"])[1]
        label = ",".join(f"{k}={v}" for k, v in sorted(metric.items())) or "(no labels)"
        print(f"  {value:>12}  {label}")
    print(f"{len(result)} series")


def cmd_raw(args) -> None:
    which = PROM_UID if args.datasource == "prometheus" else TEMPO_UID
    print(proxy_get(args.base, which, args.path))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tempo.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--base", default=DEFAULT_BASE, help="Grafana base URL")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_window(p):
        p.add_argument(
            "--since",
            default="24h",
            help="window ending now, e.g. 30m/24h/7d (default 24h)",
        )
        p.add_argument(
            "--from",
            dest="from_ts",
            type=int,
            help="window start, unix seconds (overrides --since)",
        )
        p.add_argument("--to", type=int, help="window end, unix seconds (default now)")

    p = sub.add_parser("search", help="list traces matching a TraceQL query")
    p.add_argument("query")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--json", action="store_true")
    add_window(p)
    p.set_defaults(func=cmd_search)

    p = sub.add_parser(
        "spans", help="list matching spans with status + nodeid (for huge traces)"
    )
    p.add_argument("query")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--json", action="store_true")
    add_window(p)
    p.set_defaults(func=cmd_spans)

    p = sub.add_parser("status", help="tally matching spans by status (ok/error/unset)")
    p.add_argument("query")
    p.add_argument(
        "--limit", type=int, default=50, help="max traces scanned (default 50)"
    )
    p.add_argument("--json", action="store_true")
    add_window(p)
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("count", help="count matching spans over the window")
    p.add_argument("query")
    p.add_argument(
        "--limit", type=int, default=50, help="max traces scanned (default 50)"
    )
    p.add_argument("--json", action="store_true")
    add_window(p)
    p.set_defaults(func=cmd_count)

    p = sub.add_parser("trace", help="fetch one whole trace as JSON (fails if >16 MB)")
    p.add_argument("trace_id")
    p.set_defaults(func=cmd_trace)

    p = sub.add_parser(
        "tags", help="list tag names (optional scope: span/resource/intrinsic/event)"
    )
    p.add_argument("scope", nargs="?")
    p.set_defaults(func=cmd_tags)

    p = sub.add_parser("values", help="list values for a tag, e.g. span.pytest.nodeid")
    p.add_argument("tag")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_values)

    p = sub.add_parser("promql", help="instant PromQL query against the CI Prometheus")
    p.add_argument("query")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_promql)

    p = sub.add_parser(
        "raw", help="GET an arbitrary datasource-proxy path (escape hatch)"
    )
    p.add_argument("path", help="e.g. /api/status/buildinfo")
    p.add_argument("--datasource", choices=("tempo", "prometheus"), default="tempo")
    p.set_defaults(func=cmd_raw)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
