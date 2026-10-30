"""Human-readable Slack presentation of Relore's structured evidence, without inference."""

import html
from urllib.parse import urlsplit


def text(value):
    # Flatten user-authored content so it cannot introduce headings, code fences,
    # Slack mentions, or links masquerading as our own presentation.
    return (
        html.escape(" ".join(str(value or "").split()), quote=False)
        .replace("`", "ʼ")
        .replace("*", "∗")
        .replace("|", "¦")
    )


def link(url, label):
    parsed = urlsplit(str(url or ""))
    if (
        parsed.scheme == "https"
        and parsed.netloc == "github.com"
        and not any(c in url for c in "<>|\n\r")
    ):
        return f"<{html.escape(url)}|{text(label)}>"
    return text(label)


def person(name):
    name = str(name or "unknown")
    return link(f"https://github.com/{name}", "@" + name)


def access(trust):
    return {"authoritative": "write access", "machine": "automated account"}.get(
        trust, "contributor"
    )


def thread_lines(threads):
    if not threads:
        return ["No matches in the indexed evidence."]
    lines = []
    for thread in threads:
        label = f"#{thread.get('number')} — {thread.get('title') or 'Untitled'}"
        state = (
            "merged"
            if thread.get("merged")
            else "draft"
            if thread.get("draft")
            else thread.get("state", "unknown")
        )
        details = [text(state)]
        if thread.get("review_decision"):
            details.append(text(thread["review_decision"].replace("_", " ")))
        if thread.get("author"):
            details.append(person(thread["author"]))
        if thread.get("age") not in (None, "?"):
            details.append(text(thread["age"]) + " old")
        lines.append("• " + link(thread.get("url"), label) + " — " + "; ".join(details))
        evidence = thread.get("evidence") or {}
        if evidence.get("snippet"):
            source = str(evidence.get("source_type") or "comment").replace("_", " ")
            attribution = (
                person(evidence.get("author")) + " · " + access(evidence.get("trust"))
            )
            if evidence.get("age") not in (None, "?"):
                attribution += " · " + text(evidence["age"]) + " ago"
            lines.append("  " + attribution + " · " + link(evidence.get("url"), source))
            lines.append("> “" + text(evidence["snippet"]) + "”")
    return lines


def render(data):
    head = data["thread"]
    decision = head.get("review_decision")
    review = {
        "approved": "Approved",
        "changes_requested": "Changes requested",
        "review_required": "Review required",
    }.get(
        decision,
        text(decision) if decision else "No approval or blocking review recorded",
    )
    lines = ["*Relore context*", "*Review:* " + review + "."]
    if head.get("review_decision_by"):
        lines.append(
            "Reviewers: " + ", ".join(person(p) for p in head["review_decision_by"])
        )
    if head.get("requested_reviewers"):
        lines.append(
            "Review requested from: "
            + ", ".join(person(p) for p in head["requested_reviewers"])
        )
    if head.get("labels"):
        lines.append("Labels: " + ", ".join(text(p) for p in head["labels"]))
    opener = data.get("opener")
    if opener:
        lines.extend(
            [
                "",
                "*Author and discussion*",
                f"{person(opener.get('author'))} ({access(opener.get('trust'))}): {opener.get('prs', 0)} other indexed PRs, {opener.get('merged', 0)} merged; {opener.get('issues', 0)} other issues.",
            ]
        )
        if opener.get("first_age"):
            lines.append(
                "First indexed contribution: " + text(opener["first_age"]) + " ago."
            )
    else:
        lines.extend(["", "*Author and discussion*", "Author history unavailable."])
    people = data.get("people") or {}
    listed = people.get("listed") or []
    total = people.get("total", 0)
    lines.append(
        f"{total} other participant{'s' if total != 1 else ''}, {people.get('authoritative', 0)} with write access."
    )
    for p in listed:
        age = (
            f"; last reply {text(p['age'])} ago"
            if p.get("age") not in (None, "?")
            else ""
        )
        count = p.get("comments", 0)
        lines.append(
            f"• {person(p.get('author'))} ({access(p.get('trust'))}): {count} comment{'s' if count != 1 else ''}{age}."
        )
    if people.get("total", 0) > len(listed):
        lines.append(f"{people['total'] - len(listed)} more participants not shown.")
    if people.get("machine_comments"):
        lines.append(f"{people['machine_comments']} automated comments omitted.")

    closes = data.get("closes") or []
    lines.extend(["", "*Issues this PR claims to fix*"])
    if not closes:
        lines.append("No closing links recorded.")
    for entry in closes:
        lines.extend(
            thread_lines([entry["thread"]])
            if entry.get("thread")
            else [f"• #{entry['number']} — not indexed."]
        )
    if data.get("closes_total", 0) > len(closes):
        lines.append(
            f"{data['closes_total'] - len(closes)} more closing links not looked up."
        )
    competing = data.get("competing") or {}
    claims = competing.get("claims") or []
    lines.extend(["", "*Other work on the same issues*"])
    if claims:
        lines.extend(thread_lines(claims))
    elif not closes:
        lines.append("Cannot compare: this PR has no recorded closing links.")
    elif not competing.get("links_indexed"):
        lines.append("Unknown: this repository has no indexed relationship data.")
    else:
        lines.append("No other claims found in the indexed relationships.")
    if competing.get("total", 0) > len(claims):
        lines.append(f"{competing['total'] - len(claims)} more claims not shown.")

    history = data.get("file_history")
    lines.extend(["", "*History on these files*"])
    if history:
        basis = "changed" if history.get("basis") == "changed" else "mentioned"
        lines.append(
            f"Compared {len(history.get('paths_used') or [])} of {history.get('paths_total', 0)} indexed {basis} paths; these are overlapping threads, not necessarily the same problem."
        )
        lines.extend(thread_lines(history.get("threads") or []))
        if history.get("untested"):
            lines.append(
                f"Coverage gap: {history['untested']} PRs have no collected file list and could not be compared."
            )
    else:
        lines.append("No usable file paths in the index.")
    if (
        head.get("files_total") is not None
        and head.get("files_collected") is not None
        and head["files_collected"] < head["files_total"]
    ):
        lines.append(
            f"Relore collected {head['files_collected']} of this PR's {head['files_total']} changed files."
        )
    error = data.get("same_error")
    lines.extend(["", "*Same error*"])
    if error:
        lines.append("Matched error: “" + text(error.get("error")) + "”.")
        lines.extend(thread_lines(error.get("threads") or []))
    else:
        lines.append("No error signature available for comparison.")
    similar = data.get("similar") or {}
    lines.extend(
        [
            "",
            "*Similar discussions*",
            "Matches require every word of the title; threads listed above are excluded.",
        ]
    )
    lines.extend(thread_lines(similar.get("threads") or []))
    lines.extend(
        [
            "",
            "_Evidence indexed "
            + text(head.get("indexed_at") or "at an unknown time")
            + "; may lag GitHub. Quotes are excerpts from linked discussions. Search sections show Relore’s selected matches, not exhaustive history._",
        ]
    )
    return "\n".join(lines)
