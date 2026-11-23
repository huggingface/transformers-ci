"""Server-rendered benchmark pages: HTML forms, SVG charts, no JavaScript.

Snapshot data is built by the pinned tperf dashboard analysis, shared with the
original Space. Only presentation lives here; statistical calculations stay in
transformers-perf.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from functools import lru_cache
from html import escape
from pathlib import Path
from urllib.parse import urlencode

CSS = """
:root{color-scheme:dark;--bg:#111217;--panel:#181b1f;--fg:#d8d9da;--muted:#a4a8af;--line:#343940;--accent:#8bb4ff;--good:#73bf69;--bad:#ff8585}
:root[data-theme=light]{color-scheme:light;--bg:#f4f5f5;--panel:white;--fg:#24292e;--muted:#656d76;--line:#d0d7de;--accent:#1f6feb;--good:#26762c;--bad:#b32828}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 Inter,system-ui,sans-serif}
a{color:var(--accent)}header{border-bottom:1px solid var(--line);padding:10px 16px}nav{display:flex;flex-wrap:wrap;gap:8px;align-items:center}header a,.tabs a{color:inherit;text-decoration:none;border:1px solid var(--line);border-radius:4px;padding:6px 9px}a[aria-current=page]{border-color:var(--accent)}header .home{font-size:16px;font-weight:700;border:0}main{max-width:1440px;margin:auto;padding:24px 20px 64px}h1{font-size:26px;margin:16px 0 4px}h2{margin:28px 0 12px;font-size:20px}h3{margin:0 0 4px;font-size:15px}p{margin:8px 0}.muted,small{color:var(--muted)}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(360px,100%),1fr));gap:16px}.card,.notice{border:1px solid var(--line);border-radius:8px;background:var(--panel);padding:18px;margin:12px 0;min-width:0}.headline{font-size:24px;font-weight:600}.meta{font-size:12px;color:var(--muted);overflow-wrap:anywhere}.filters{display:flex;flex-wrap:wrap;align-items:end;gap:12px;margin:20px 0}label{display:grid;gap:4px}select,input,button{font:inherit;background:var(--panel);color:var(--fg);border:1px solid var(--line);border-radius:4px;padding:7px;max-width:100%}button{cursor:pointer;border-color:var(--accent)}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;vertical-align:top;border-bottom:1px solid var(--line);padding:8px;white-space:nowrap}th{color:var(--muted)}td.wrap{white-space:normal;min-width:150px}summary{cursor:pointer;color:var(--accent);margin:8px 0}.good{color:var(--good)}.bad{color:var(--bad)}.chip{display:inline-block;font-size:12px;border:1px solid var(--line);border-radius:16px;padding:2px 8px;margin:2px}.plot{width:100%;height:auto}.plot text{fill:var(--muted);font:11px system-ui}.plot .gridline{stroke:var(--line)}.plot .line{fill:none;stroke:var(--accent);stroke-width:2}.plot .band{fill:var(--accent);opacity:.15}.plot circle{fill:var(--accent)}code{overflow-wrap:anywhere}footer{margin-top:32px;color:var(--muted);font-size:12px}:focus-visible{outline:2px solid var(--accent);outline-offset:3px}
"""
MENU = (
    ("Activity", "/"),
    ("Fix failures", "/d/pytest-test-health/fix-failures"),
    ("MIC failures", "/d/mic-failures/mic-failures"),
    ("Monitoring", "/d/pytest-observability-daily/monitoring"),
    (
        "CI Health 🔒",
        "/login/github?redirectTo=%2Fd%2Fpytest-observability-stack-health%2Fci-health",
    ),
    ("Benchmarks", "/benchmarks"),
    ("Profile", "/profile"),
)


def esc(value):
    return escape(str(value if value is not None else ""), quote=True)


def link(url, label, current=False):
    return (
        f'<a href="{esc(url)}"'
        + (' aria-current="page"' if current else "")
        + f">{esc(label)}</a>"
    )


def url(page="index", *, theme="dark", **params):
    if theme == "light":
        params["theme"] = "light"
    path = "/benchmarks" + ("/" + page if page != "index" else "")
    return path + (
        "?" + urlencode({k: v for k, v in params.items() if v})
        if any(params.values())
        else ""
    )


def fmt(value, unit=""):
    if not isinstance(value, (float, int)) or not math.isfinite(value):
        return "—"
    digits = 1 if abs(value) >= 100 else 2 if abs(value) >= 10 else 3
    number = f"{value:,.{digits}f}".rstrip("0").rstrip(".")
    return number + (f" {unit}" if unit else "")


def commit(sha):
    if not re.fullmatch(r"[a-fA-F0-9]{7,40}", str(sha or "")):
        return esc(sha)
    return link("https://github.com/huggingface/transformers/commit/" + sha, sha[:10])


def table(headings, rows):
    return (
        '<div class="scroll"><table><thead><tr>'
        + "".join(f"<th>{esc(h)}</th>" for h in headings)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows
        )
        + "</tbody></table></div>"
    )


def document(title, body, theme="dark"):
    nav = link("/", "🤗 Transformers CI") + "".join(
        link(
            url(theme=theme) if name == "Benchmarks" else path,
            name,
            name == "Benchmarks",
        )
        for name, path in MENU
    )
    return f'<!doctype html><html lang="en" data-theme="{theme}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{esc(title)} · Transformers CI</title><style>{CSS}</style></head><body><header><nav aria-label="CI dashboards">{nav}</nav></header><main>{body}</main></body></html>'


def chart(points, metric, unit):
    values = [(p, p["values"][metric]) for p in points if metric in p["values"]]
    if not values:
        return '<p class="muted">No measurements for this metric.</p>'
    # Use timestamps, so gaps in the daily runs remain visible.
    times = [
        datetime.fromisoformat(p["date"].replace("Z", "+00:00")).timestamp()
        for p, _ in values
    ]
    low = min(v[1] for _, v in values)
    high = max(v[2] for _, v in values)
    pad = max((high - low) * 0.12, abs(high) * 0.03, 0.001)
    low, high = max(0, low - pad), high + pad

    def x(i):
        return (
            54 + (times[i] - times[0]) / (times[-1] - times[0]) * 400
            if times[-1] != times[0]
            else 254
        )

    def y(v):
        return 150 - (v - low) / (high - low) * 125

    line = " ".join(f"{x(i):.2f},{y(v[0]):.2f}" for i, (_, v) in enumerate(values))
    band = (
        " ".join(f"{x(i):.2f},{y(v[2]):.2f}" for i, (_, v) in enumerate(values))
        + " "
        + " ".join(
            f"{x(i):.2f},{y(values[i][1][1]):.2f}" for i in reversed(range(len(values)))
        )
    )
    marks = ""
    for i, (point, val) in enumerate(values):
        title = f"{point['date']}: {fmt(val[0], unit)} (min {fmt(val[1])}, max {fmt(val[2])}); {point['commit']}"
        marks += f'<circle tabindex="0" cx="{x(i):.2f}" cy="{y(val[0]):.2f}" r="4"><title>{esc(title)}</title></circle>'
    axes = "".join(
        f'<line class="gridline" x1="54" x2="454" y1="{y(v):.2f}" y2="{y(v):.2f}"/><text x="49" y="{y(v) + 4:.2f}" text-anchor="end">{esc(fmt(v))}</text>'
        for v in (low, (low + high) / 2, high)
    )
    return f'<svg class="plot" viewBox="0 0 480 180" role="img" aria-label="{esc(metric)} history in {esc(unit)}"><title>Daily medians and min–max band; exact values in the data table below</title>{axes}<polygon class="band" points="{band}"/><polyline class="line" points="{line}"/>{marks}<text x="54" y="174">{esc(values[0][0]["date"][:10])}</text><text x="454" y="174" text-anchor="end">{esc(values[-1][0]["date"][:10])}</text></svg>'


def select(name, label, choices, selected):
    options = "".join(
        f'<option value="{esc(key)}"'
        + (" selected" if key == selected else "")
        + f">{esc(text)}</option>"
        for key, text in choices
    )
    return f'<label>{esc(label)}<select name="{esc(name)}">{options}</select></label>'


def daily(data, params, theme="dark"):
    get = lambda name, default="": (params.get(name) or [default])[0]
    series = data["series"]
    metrics = data["metrics"]
    metric = get("metric", next(iter(metrics), "e2e"))
    if metric not in metrics:
        metric = next(iter(metrics), "e2e")
    model = get("model")
    group = get("group")
    proc_metric = get("processor_metric", "time")
    if proc_metric not in ("time", "syncs"):
        proc_metric = "time"
    filters = (
        '<input type="hidden" name="theme" value="light">' if theme == "light" else ""
    )
    filters += select(
        "metric",
        "Generation metric",
        [(k, v["label"]) for k, v in metrics.items()],
        metric,
    )
    filters += select(
        "processor_metric",
        "Processor / encoder metric",
        [("time", "Call time"), ("syncs", "GPU→CPU syncs")],
        proc_metric,
    )
    filters += select(
        "model",
        "Model",
        [("", "All models")] + [(m, m) for m in sorted({s["model"] for s in series})],
        model,
    )
    filters += select(
        "group",
        "Group",
        [("", "All groups")]
        + [(g, g) for g in dict.fromkeys(s["group"] for s in series)],
        group,
    )
    out = '<p class="muted">Daily main runs. Lines show medians of run medians; bands show their min–max range. Only current series and metrics present throughout their last seven daily runs are shown.</p>'
    out += f'<form class="filters" method="get" action="/benchmarks/main">{filters}<button>Apply filters</button>{link(url("main", theme=theme), "Reset")}</form>'
    groups = defaultdict(list)
    for s in series:
        if (not model or s["model"] == model) and (not group or s["group"] == group):
            groups[s["group"]].append(s)
    if not groups:
        return out + '<p class="notice">No daily series match these filters.</p>'
    for name, items in groups.items():
        out += f'<h2>{esc(name)}</h2><div class="grid">'
        for s in items:
            m = metric if name == "Generation" else proc_metric
            meta = (metrics if name == "Generation" else data["series_metrics"]).get(
                m, {}
            )
            unit = meta.get("unit", "")
            out += f'<article class="card"><h3>{esc(s["model"])}</h3><p class="meta">{esc(s["scenario"])}<br>{esc(s["gpu"])}</p>'
            points = [p for p in s["points"] if m in p["values"]]
            if not points:
                out += '<p class="muted">This metric is not available for this series.</p></article>'
                continue
            last = points[-1]["values"][m][0]
            first = points[0]["values"][m][0]
            out += f'<p>{esc(meta.get("label", m))} <span class="headline">{esc(fmt(last, unit))}</span></p>'
            if first and len(points) > 1:
                change = (last / first - 1) * 100
                worse = change < 0 if meta.get("higher_is_better") else change > 0
                out += f'<p class="{"bad" if worse else "good"}">{change:+.2f}% since {esc(points[0]["date"][:10])}</p>'
            out += chart(points, m, unit)
            rows = []
            for p in reversed(points):
                v = p["values"][m]
                rows.append(
                    [
                        esc(p["date"]),
                        commit(p["commit"]),
                        esc(p.get("subject", "")),
                        esc(fmt(v[0], unit)),
                        esc(f"{fmt(v[1])} – {fmt(v[2])}"),
                        esc(p["runs"]),
                        esc(p["failed"]),
                    ]
                )
            out += (
                "<details><summary>Data and commits</summary>"
                + table(
                    [
                        "Date",
                        "Commit",
                        "Subject",
                        "Median",
                        "Min–max",
                        "Runs",
                        "Failed",
                    ],
                    rows,
                )
                + "</details>"
            )
            if points[-1].get("sync_sources"):
                out += (
                    "<details><summary>Latest GPU→CPU sync sources</summary>"
                    + table(
                        ["Source", "Count"],
                        [
                            [esc(k), esc(v)]
                            for k, v in points[-1]["sync_sources"].items()
                        ],
                    )
                    + "</details>"
                )
            if points[-1].get("outputs"):
                out += (
                    "<details><summary>Latest output hashes</summary><code>"
                    + "<br>".join(esc(v) for v in points[-1]["outputs"])
                    + "</code></details>"
                )
            out += "</article>"
        out += "</div>"
    return out


def comparisons(data):
    out = '<p class="muted">Base and head run side by side, swapping GPUs each round. Change fits out the GPU difference; positive means worse (slower or less throughput). Confidence intervals are 95%; verdict threshold is 1%.</p>'
    for run in data["abs"]:
        out += f'<section class="card"><h2>{esc(run["model"])} · {esc(run["date"])}</h2><p>Base {commit(run["base_commit"])} → head {commit(run["head_commit"])}</p><p class="meta">{esc(run["ab_id"])}</p><details><summary>Benchmark arguments</summary><code>{esc(run["bench_args"])}</code></details>'
        for scenario in run["scenarios"]:
            out += f"<h3>{esc(scenario['scenario'])} · {scenario['rounds']} paired rounds</h3>"
            rows = []
            for m in scenario["metrics"]:
                cls = (
                    "bad"
                    if m["verdict"] == "REGRESSION"
                    else "good"
                    if m["verdict"] == "improvement"
                    else "muted"
                )
                lo, hi = m["ci"]
                interval = (
                    f"{fmt(lo)}% to {fmt(hi)}%"
                    if lo is not None and hi is not None
                    else "Unbounded (insufficient rounds)"
                )
                rows.append(
                    [
                        esc(m["name"]),
                        esc(fmt(m["base"])),
                        esc(fmt(m["head"])),
                        esc(fmt(m["change_pct"], "%")),
                        esc(interval),
                        f'<span class="{cls}">{esc(m["verdict"])}</span>',
                    ]
                )
            out += (
                table(["Metric", "Base", "Head", "Change", "95% CI", "Verdict"], rows)
                if rows
                else "<p>No complete measurement pairs.</p>"
            )
            if scenario["counters"]:
                out += (
                    '<p class="meta">Counters: first generation / warmup / measurement</p>'
                    + table(
                        ["Counter", "Base", "Head"],
                        [
                            [esc(k), esc(v["base"]), esc(v["head"])]
                            for k, v in scenario["counters"].items()
                        ],
                    )
                )
        if run["failed"]:
            out += (
                '<p class="bad">Failed scenarios: '
                + esc(", ".join(run["failed"]))
                + "</p>"
            )
        out += "</section>"
    return out or "<p>No comparisons yet.</p>"


def overview(data, theme="dark"):
    main = data["main"]
    out = f'<section class="card"><h2>{link(url("main", theme=theme), "Daily main")}</h2><p><span class="headline">{main["series"]}</span> series across {main["days"]} days</p><p>Latest: {esc(main["last_date"] or "No daily runs yet")} · {commit(main["last_commit"])}</p></section><h2>Pull requests and benchmark changes</h2><p class="muted">Verdicts describe the latest run. Open a comparison to see every run.</p>'
    if not data["prs"]:
        return out + '<p class="notice">No side-by-side runs yet.</p>'
    rows = []
    for pr in data["prs"]:
        verdicts = " ".join(
            f'<span class="chip {"bad" if k == "regression" else "good" if k == "improvement" else ""}">{v} {esc(k)}</span>'
            for k, v in pr["verdicts"].items()
        )
        rows.append(
            [
                link(url(pr["file"].removesuffix(".html"), theme=theme), pr["label"]),
                esc(pr["title"]),
                esc(", ".join(pr["models"])),
                esc(pr["runs"]),
                esc(pr["last_date"]),
                verdicts,
            ]
        )
    return out + table(
        ["Comparison", "Title", "Models", "Runs", "Latest", "Verdicts"], rows
    )


@lru_cache(maxsize=1)
def _read_snapshot(path, mtime_ns):
    return json.loads(Path(path).read_text())


def render(path, params):
    theme = "light" if (params.get("theme") or [""])[0] == "light" else "dark"
    name = path.removeprefix("/benchmarks").strip("/") or "index"
    if not re.fullmatch(r"index|main|benchmark-changes|pr-[1-9][0-9]*", name):
        return 404, document(
            "Not found",
            "<h1>Benchmark page not found</h1>" + link(url(theme=theme), "Benchmarks"),
            theme,
        )
    snapshot = Path(
        os.environ.get("TCI_BENCHMARK_SNAPSHOT", "/benchmarks/snapshot.json")
    )
    try:
        data = _read_snapshot(str(snapshot), snapshot.stat().st_mtime_ns)
    except (OSError, ValueError):
        return 503, document(
            "Benchmarks",
            '<h1>Benchmarks</h1><p class="notice">Benchmark data is temporarily unavailable. Please try again shortly.</p>',
            theme,
        )
    pages = data["pages"]
    page = pages.get(name + ".html")
    if page is None:
        return 404, document(
            "Not found",
            "<h1>Benchmark page not found</h1>" + link(url(theme=theme), "Benchmarks"),
            theme,
        )
    title = "Benchmarks" if name == "index" else page["page"]["heading"]
    body = (
        '<nav class="tabs" aria-label="Benchmarks">'
        + link(url(theme=theme), "Overview", name == "index")
        + link(url("main", theme=theme), "Daily main", name == "main")
        + "</nav>"
    )
    body += f'<h1>{esc(title)}</h1><p class="muted">{esc(page["host"])}</p>'
    if page["page"].get("link"):
        body += (
            "<p>" + link(page["page"]["link"], "View pull request on GitHub") + "</p>"
        )
    if time.time() - data["updated_at"] > 48 * 3600:
        body += '<p class="notice" role="status">These results are more than 48 hours old. Showing the last complete dataset snapshot.</p>'
    if name == "index":
        body += overview(page, theme)
    elif name == "main":
        body += daily(page, params, theme)
    else:
        body += comparisons(page)
    updated = datetime.fromtimestamp(data["updated_at"], timezone.utc).strftime(
        "%Y-%m-%d %H:%M UTC"
    )
    body += (
        "<footer>"
        + link("https://huggingface.co/datasets/" + data["dataset"], data["dataset"])
        + f" · Snapshot {esc(data['revision'][:10])} · Updated {esc(updated)}</footer>"
    )
    return 200, document(title, body, theme)
