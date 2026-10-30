"""Current GitHub PR facts and the complete, evidence-based Relore brief."""

from datetime import datetime, timezone
import html
import json
import os
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from . import related_issues, slack_brief

MAX_PAGES = 10
MAX_BYTES = 512 * 1024
SUPPORTED_BRIEF_VERSIONS = {"0.3.18", "0.3.19"}


def brief(pr):
    base = os.getenv("PYTEST_TRACE_EXPORTER_RELORE_URL", "").strip().rstrip("/")
    if not base:
        return "Relore brief is not configured."
    query = urlencode({"repo": related_issues.REPOSITORY, "compact": "false"})
    url = f"{base}/api/v1/brief/{int(pr)}?{query}"
    version = related_issues.WIRE_VERSION
    for attempt in range(2):
        try:
            request = Request(url, headers={"x-relore-client": version})
            with urlopen(request, timeout=20) as response:
                if response.headers.get("x-relore-version") != version:
                    raise ValueError("Relore version mismatch")
                raw = response.read(MAX_BYTES + 1)
            if len(raw) > MAX_BYTES:
                raise ValueError("Relore response too large")
            data = json.loads(raw)
            head = data.get("thread") or {}
            if (
                head.get("repo") != related_issues.REPOSITORY
                or head.get("number") != int(pr)
                or head.get("type") != "pr"
                or not isinstance(data.get("people"), dict)
            ):
                raise ValueError("Unexpected Relore brief")
            rendered = slack_brief.render(data)
            if len(rendered) > 28000:
                # Cut at a complete line so a Slack link or quote is never split.
                rendered = rendered[:28000].rsplit("\n", 1)[0]
                rendered += f"\nBrief truncated for Slack. Full output: `relore brief {pr} --repo {related_issues.REPOSITORY}`."
            return rendered
        except HTTPError as error:
            server = error.headers.get("x-relore-version")
            if (
                attempt == 0
                and error.code in {400, 409, 426}
                and server in SUPPORTED_BRIEF_VERSIONS
                and server != version
            ):
                version = server
                continue
            if error.code == 404:
                return "Relore brief unavailable: this PR is not indexed yet, or the server does not support brief."
            return "Relore brief temporarily unavailable; the GitHub information above is still current."
        except (OSError, ValueError, TypeError, AttributeError):
            return "Relore brief temporarily unavailable; the GitHub information above is still current."
    return "Relore brief unavailable."


def age(value):
    try:
        created = datetime.fromisoformat(value.replace("Z", "+00:00"))
        seconds = max(0, int((datetime.now(timezone.utc) - created).total_seconds()))
        days, hours = seconds // 86400, (seconds % 86400) // 3600
        return f"{days}d {hours}h" if days else f"{hours}h {(seconds % 3600) // 60}m"
    except (AttributeError, ValueError, TypeError):
        return "unknown"


def checks(sha, github):
    counts = {"passed": 0, "failed": 0, "pending": 0, "other": 0}
    failures, pending, incomplete = [], [], []
    contexts = set()
    for route, key in [("check-runs", "check_runs"), ("status", "statuses")]:
        fetched = 0
        for page in range(1, MAX_PAGES + 1):
            result = github(f"commits/{sha}/{route}?per_page=100&page={page}")
            items = result.get(key, [])
            fetched += len(items)
            for item in items:
                if route == "status":
                    context = item.get("context")
                    if context and context in contexts:
                        continue
                    contexts.add(context)
                    state = {
                        "success": "passed",
                        "failure": "failed",
                        "error": "failed",
                        "pending": "pending",
                    }.get(item.get("state"), "other")
                    name = context or "Commit status"
                else:
                    conclusion = item.get("conclusion")
                    state = (
                        "pending"
                        if item.get("status") != "completed"
                        else "passed"
                        if conclusion == "success"
                        else "failed"
                        if conclusion
                        in {
                            "failure",
                            "timed_out",
                            "cancelled",
                            "action_required",
                            "startup_failure",
                        }
                        else "other"
                    )
                    name = item.get("name") or "Check"
                counts[state] += 1
                if state == "failed":
                    failures.append(str(name))
                if state == "pending":
                    pending.append(str(name))
            if fetched >= result.get("total_count", fetched) or not items:
                if fetched < result.get("total_count", fetched):
                    incomplete.append(route)
                break
        else:
            incomplete.append(route)
    detail = ", ".join(f"{count} {key}" for key, count in counts.items())
    if incomplete:
        detail += " (partial: pagination limit or incomplete response; open GitHub for remaining checks)"
    lines = ["*CI at this head:* " + detail]
    for label, names in [("Failed", failures), ("Pending", pending)]:
        if names:
            text = ", ".join(html.escape(name[:150]) for name in names[:10])
            lines.append(
                f"{label}: {text}"
                + (f" (+{len(names) - 10} more)" if len(names) > 10 else "")
            )
    return "\n".join(lines)


def summary(pr, pull, sha, github, dashboard):
    def escape(value):
        return html.escape(str(value))

    title = escape(pull.get("title") or f"PR #{pr}")
    author = escape((pull.get("user") or {}).get("login") or "unknown")
    state = (
        "merged"
        if pull.get("merged")
        else "draft"
        if pull.get("draft")
        else pull.get("state", "unknown")
    )
    base, head = pull.get("base") or {}, pull.get("head") or {}
    lines = [
        f"*<{dashboard}/pr-ci-view?pr={pr}|Transformers PR #{pr}> — {title}*",
        f"{escape(state)} · @{author} · opened {age(pull.get('created_at'))} ago · updated {age(pull.get('updated_at'))} ago",
        f"{escape(head.get('ref', '?'))} → {escape(base.get('ref', '?'))} · head `{sha[:8]}`",
        f"Changes: {pull.get('changed_files', '?')} files, +{pull.get('additions', '?')} / −{pull.get('deletions', '?')}",
        f"<https://github.com/{related_issues.REPOSITORY}/pull/{pr}|Open PR> · <https://github.com/{related_issues.REPOSITORY}/pull/{pr}/checks|All CI checks>",
    ]
    try:
        lines.append(checks(sha, github))
    except Exception:
        lines.append(
            "CI checks temporarily unavailable; open GitHub for their current status."
        )
    lines.extend(["", brief(pr)])
    return "\n".join(lines)
