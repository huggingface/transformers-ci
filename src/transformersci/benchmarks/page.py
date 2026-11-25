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

from transformersci.otel import dashboard_ui

CSS = (
    dashboard_ui.CSS
    + """
.tabs,.section-menu{display:flex;flex-wrap:wrap;gap:4px;margin:0 0 8px;border-bottom:1px solid var(--line)}
.section-menu{position:sticky;top:0;background:var(--bg);z-index:10}
.tabs a,.section-menu a{color:var(--muted);padding:6px 12px;text-decoration:none;border-bottom:2px solid transparent;font-size:13px}
.tabs a:hover,.section-menu a:hover{color:var(--fg);background:var(--hover)}.tabs a[aria-current=page]{color:var(--fg);border-bottom-color:var(--accent)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(360px,100%),1fr));gap:8px}
.card,.notice,.benchmark-group{background:var(--panel);border:1px solid var(--line);border-radius:3px;padding:10px 12px;min-width:0}
.benchmark-group{margin:12px 0;scroll-margin-top:48px}.benchmark-group>.panel-heading{display:flex;align-items:center;flex-wrap:wrap;gap:8px;justify-content:space-between;margin:0 0 8px}.panel-heading form{margin:0}
.benchmark-group .card{background:var(--bg);padding:8px 10px}.headline{font-size:18px;font-weight:500}.meta{font-size:11px;line-height:16px;color:var(--muted);overflow-wrap:anywhere}.filters{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin:8px 0}.filters label{display:flex;align-items:center;gap:6px;font-size:12px}.filters select,.filters button{font-size:12px}.filters select{max-width:360px}label{display:grid;gap:4px}
.plot{width:100%;height:auto}.plot text{fill:var(--muted);font:11px Inter,system-ui}.plot .gridline{stroke:var(--line)}.plot .line{fill:none;stroke:var(--accent);stroke-width:2}.plot .band{fill:var(--accent);opacity:.15}.plot circle{fill:var(--accent)}
summary{cursor:pointer;color:var(--accent);margin:4px 0;font-size:12px}.good{color:var(--good)}.bad{color:var(--bad)}.chip{display:inline-block;font-size:11px;border:1px solid var(--line);border-radius:3px;padding:0 5px;margin:2px}.metric-value{display:flex;align-items:baseline;gap:8px;margin:4px 0 0}.delta{font-size:11px;margin:0}code{overflow-wrap:anywhere}footer{margin-top:20px;color:var(--muted);font-size:11px}.overview-card{margin:10px 0;padding:12px}.comparison-list{margin-top:12px}.comparison-list>h2{margin-bottom:4px}.comparison-list>.scroll{background:var(--panel);border:1px solid var(--line);border-radius:3px;margin-top:8px}td{white-space:nowrap}td.wrap{white-space:normal;min-width:150px}.card>h2{margin-bottom:6px}.card>h2~h3{margin-top:12px}
"""
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
            "<tr>"
            + "".join(
                f'<td class="{"wrap" if headings[i] in ("Title", "Models", "Subject", "Verdicts") else ""}">{c}</td>'
                for i, c in enumerate(row)
            )
            + "</tr>"
            for row in rows
        )
        + "</tbody></table></div>"
    )


def document(title, body, theme="dark"):
    nav = dashboard_ui.header("Benchmarks", theme)
    return f'<!doctype html><html lang="en" data-theme="{theme}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{esc(title)} · Transformers CI</title><style>{CSS}</style></head><body>{nav}<main>{body}</main></body></html>'


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


MEMORY_PATH = re.compile(
    r"/benchmarks/memory/([A-Za-z0-9][A-Za-z0-9._-]*)/"
    r"(report\.html|time\.html|host-flamegraph\.html|memory-viz\.html|gpu_snapshot\.pickle|python\.folded)"
)
# Pages that run scripts (memray's flamegraph, PyTorch's memory viewer) get an
# opaque origin: their scripts cannot reach the dashboard's cookies or storage
SANDBOX = {
    "Content-Security-Policy": "sandbox allow-scripts allow-popups allow-popups-to-escape-sandbox",
    "X-Content-Type-Options": "nosniff",
}
# PyTorch's memory viewer, pinned to the bench host's torch release (the
# snapshot format follows torch); it builds its own page in <body>
MEMORY_VIZ_JS = (
    "https://cdn.jsdelivr.net/gh/pytorch/pytorch@v2.15.0/torch/utils/viz/MemoryViz.js"
)
MEMORY_VIZ_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><title>GPU allocations · {title}</title></head><body><script type="module">
import {{add_remote_files}} from "{js}";
add_remote_files([{{name: {name}, url: "gpu_snapshot.pickle"}}]);
</script></body></html>"""


def memory_file(path, theme="dark"):
    """A daily memory report copied by the refresher: (status, body, content
    type, headers). The report is the site's page around the fragment rendered
    by the refresher; the flamegraph and the memory viewer run sandboxed."""
    match = MEMORY_PATH.fullmatch(path)
    text = "text/html; charset=utf-8"
    if not match or ".." in match.group(1):
        return 404, b"Not found", "text/plain; charset=utf-8", {}
    group, name = match.groups()
    root = (
        Path(
            os.environ.get("TCI_BENCHMARK_SNAPSHOT", "/benchmarks/snapshot.json")
        ).parent
        / "memory"
        / group
    )
    missing = (
        404,
        b"Memory report not found (reports are kept for 14 days)",
        "text/plain; charset=utf-8",
        {},
    )
    try:
        fragment_file = {"report.html": "report.json", "time.html": "time.json"}.get(
            name
        )
        if fragment_file and (root / fragment_file).is_file():
            fragment = json.loads((root / fragment_file).read_text())
            # an A/B side's profile (<ab id>-base / -head) goes back to its PR's page
            pr = re.search(r"-pr(\d+)-(?:base|head)$", group)
            back = (
                link(url(f"pr-{pr.group(1)}", theme=theme), f"← PR #{pr.group(1)}")
                if pr
                else link(url("main", theme=theme), "← Daily main")
            )
            body = (
                f"<style>{fragment['css']}</style>"
                + f'<p class="meta">{back}</p>'
                + fragment["body"]
            )
            return 200, document(fragment["title"], body, theme).encode(), text, {}
        if name == "time.html":
            return missing
        if name == "memory-viz.html":
            if not (root / "gpu_snapshot.pickle.gz").is_file():
                return missing
            page = MEMORY_VIZ_PAGE.format(
                title=esc(group), js=MEMORY_VIZ_JS, name=json.dumps(group)
            )
            return 200, page.encode(), text, SANDBOX
        if name == "gpu_snapshot.pickle":
            # Stored gzipped; every browser decodes it. Fetched by the sandboxed
            # viewer, whose origin is opaque: hence the CORS header
            return (
                200,
                (root / "gpu_snapshot.pickle.gz").read_bytes(),
                "application/octet-stream",
                {"Content-Encoding": "gzip", "Access-Control-Allow-Origin": "*"},
            )
        if name == "python.folded":
            # py-spy's folded stacks, for speedscope.app or flamegraph.pl: plain text
            return (
                200,
                (root / name).read_bytes(),
                "text/plain; charset=utf-8",
                {**SANDBOX, "Content-Disposition": 'inline; filename="python.folded"'},
            )
        return 200, (root / name).read_bytes(), text, SANDBOX
    except (OSError, ValueError, KeyError):
        return missing


def time_report(memory_url):
    """The time report next to a memory report (/benchmarks/memory/<group>/report.html),
    or None when the refresher rendered none for it."""
    match = MEMORY_PATH.fullmatch(memory_url)
    if not match:
        return None
    root = Path(
        os.environ.get("TCI_BENCHMARK_SNAPSHOT", "/benchmarks/snapshot.json")
    ).parent
    if not (root / "memory" / match.group(1) / "time.json").is_file():
        return None
    return memory_url.removesuffix("report.html") + "time.html"


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
    values = {
        "theme": theme if theme == "light" else "",
        "model": model,
        "group": group,
        "metric": metric,
        "processor_metric": proc_metric,
    }

    def hidden(exclude):
        return "".join(
            f'<input type="hidden" name="{key}" value="{esc(value)}">'
            for key, value in values.items()
            if value and key != exclude
        )

    model_filter = select(
        "model",
        "Model",
        [("", "All models")] + [(m, m) for m in sorted({s["model"] for s in series})],
        model,
    )
    out = '<p class="meta">Daily main · median and min–max across runs · metrics recorded throughout the last seven daily runs</p>'
    out += f'<form class="filters" method="get" action="/benchmarks/main">{hidden("model")}{model_filter}<button>Apply model</button>{link(url("main", theme=theme), "Reset")}</form>'
    groups = defaultdict(list)
    for s in series:
        if (not model or s["model"] == model) and (not group or s["group"] == group):
            groups[s["group"]].append(s)
    if not groups:
        return out + '<p class="notice">No daily series match these filters.</p>'
    sections = {
        "Generation": "generation",
        "Multimodal processors (CPU)": "processors",
        "Multimodal encoders (GPU, bf16)": "encoders",
    }
    out += (
        '<nav class="section-menu" aria-label="Benchmark sections">'
        + "".join(link("#" + sections.get(name, "series"), name) for name in groups)
        + "</nav>"
    )
    for name, items in groups.items():
        anchor = sections.get(name, "series")
        controls = ""
        if name == "Generation":
            controls = select(
                "metric",
                "Metric",
                [(k, v["label"]) for k, v in metrics.items()],
                metric,
            )
            controls = hidden("metric") + controls
        elif name == "Multimodal encoders (GPU, bf16)":
            controls = hidden("processor_metric") + select(
                "processor_metric",
                "Metric",
                [("time", "Call time"), ("syncs", "GPU→CPU syncs")],
                proc_metric,
            )
        if controls:
            controls = f'<form class="filters" method="get" action="/benchmarks/main#{anchor}">{controls}<button>Apply metric</button></form>'
        out += f'<section class="benchmark-group" id="{anchor}"><div class="panel-heading"><h2>{esc(name)}</h2>{controls}</div><div class="grid">'
        for s in items:
            m = (
                metric
                if name == "Generation"
                else "time"
                if name == "Multimodal processors (CPU)"
                else proc_metric
            )
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
            out += f'<p class="metric-value"><span class="meta">{esc(meta.get("label", m))}</span><span class="headline">{esc(fmt(last, unit))}</span></p>'
            if first and len(points) > 1:
                change = (last / first - 1) * 100
                worse = change < 0 if meta.get("higher_is_better") else change > 0
                out += f'<p class="delta {"bad" if worse else "good"}">{change:+.2f}% since {esc(points[0]["date"][:10])}</p>'
            out += chart(points, m, unit)
            reports = [p for p in s["points"] if p.get("memory")]
            theme_query = "?theme=light" if theme == "light" else ""
            if reports:
                memory = "/benchmarks/" + reports[-1]["memory"]
                out += (
                    '<p class="meta">'
                    + link(memory + theme_query, "Memory report")
                    + (
                        " · " + link(time_report(memory) + theme_query, "Time report")
                        if time_report(memory)
                        else ""
                    )
                    + f", {esc(reports[-1]['date'][:10])}"
                    + " · what held GPU and host memory at the peak, and where the time went</p>"
                )
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
                    + (
                        [
                            link("/benchmarks/" + p["memory"] + theme_query, "report")
                            if p.get("memory")
                            else ""
                        ]
                        if reports
                        else []
                    )
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
                    ]
                    + (["Memory"] if reports else []),
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
        out += "</div></section>"
    return out


def profile_links(profiles):
    """The memory and time reports of an A/B's base and head (tperf ab with PROFILE)."""
    sides = []
    for side in ("base", "head"):
        if side not in profiles:
            continue
        memory = "/benchmarks/" + profiles[side]
        time_url = time_report(memory)
        sides.append(
            f"{side.capitalize()}: "
            + link(memory, "memory report")
            + (" · " + link(time_url, "time report") if time_url else "")
        )
    if not sides:
        return ""
    return (
        '<p class="meta">Profiles (a separate, untimed run of each side): '
        + " · ".join(sides)
        + "</p>"
    )


def comparisons(data):
    out = '<p class="muted">Base and head run side by side, swapping GPUs each round. Change fits out the GPU difference; positive means worse (slower or less throughput). Confidence intervals are 95%; verdict threshold is 1%.</p>'
    for run in data["abs"]:
        out += f'<section class="card"><h2>{esc(run["model"])} · {esc(run["date"])}</h2><p>Base {commit(run["base_commit"])} → head {commit(run["head_commit"])}</p><p class="meta">{esc(run["ab_id"])}</p><details><summary>Benchmark arguments</summary><code>{esc(run["bench_args"])}</code></details>'
        out += profile_links(run.get("profiles") or {})
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
    out = f'<section class="card overview-card"><h2>{link(url("main", theme=theme), "Daily main")}</h2><p><span class="headline">{main["series"]}</span> series across {main["days"]} days</p><p>Latest: {esc(main["last_date"] or "No daily runs yet")} · {commit(main["last_commit"])}</p></section><section class="comparison-list"><h2>Pull requests and benchmark changes</h2><p class="muted">Verdicts describe the latest run. Open a comparison to see every run.</p>'
    if not data["prs"]:
        return out + '<p class="notice">No side-by-side runs yet.</p></section>'
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
    return (
        out
        + table(["Comparison", "Title", "Models", "Runs", "Latest", "Verdicts"], rows)
        + "</section>"
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
