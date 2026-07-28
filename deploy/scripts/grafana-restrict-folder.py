#!/usr/bin/env python3
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

"""Strip the `Viewer` role from a provisioned Grafana folder.

The chart only *creates* the restricted folder and organizes dashboards into it
(`grafana.dashboards.restrictedFolder`, see deploy/helm/templates/grafana.yaml).
A freshly provisioned folder inherits the default permissions — `Viewer: View`
and `Editor: Edit` — and this stack serves anonymous users as `Viewer`, so until
that role is removed the "restricted" folder is fully public.

This script removes it: it rewrites the folder's permissions to keep `Editor`
(authenticated huggingface-org members are auto-assigned Editor via OAuth) and
drop `Viewer`. Org admins retain access regardless.

**Re-run after any deploy that recreates the folder** — provisioning resets
permissions, so gating is a post-deploy step, not part of the chart.

Usage:

    export GRAFANA_PASSWORD=...            # the `admin` password (Infisical)
    deploy/scripts/grafana-restrict-folder.py --dry-run
    deploy/scripts/grafana-restrict-folder.py

Verify independently — an anonymous read of a dashboard in the folder must 403:

    curl -s -o /dev/null -w '%{http_code}\\n' \\
      https://transformers-ci.lor-e.huggingface.cool/api/dashboards/uid/<uid>
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_URL = "https://transformers-ci.lor-e.huggingface.cool"
DEFAULT_FOLDER = "Restricted"

# Grafana's legacy folder-permission levels.
VIEW, EDIT, ADMIN = 1, 2, 4
_LEVEL_NAMES = {VIEW: "View", EDIT: "Edit", ADMIN: "Admin"}


class GrafanaError(RuntimeError):
    pass


def _request(
    url: str, auth: str, method: str = "GET", payload: dict | list | None = None
) -> object:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", auth)
    req.add_header("Accept", "application/json")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode() or "{}"
        return json.loads(body)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:400]
        if e.code == 401:
            raise GrafanaError(
                "401 Unauthorized. `GF_SECURITY_ADMIN_PASSWORD` is applied only on "
                "Grafana's FIRST boot and then lives in its SQLite DB, so a later "
                "Infisical rotation leaves the DB password behind. Reset the live "
                "DB (single replica + SQLite):\n"
                "  kubectl --context infra:opensource-aws-use1-prod-54 "
                "-n transformers-ci exec deploy/grafana -- \\\n"
                '    grafana cli admin reset-admin-password "$GRAFANA_PASSWORD"'
            ) from e
        if e.code == 403:
            raise GrafanaError(
                f"403 Forbidden on {url}.\nA failed basic-auth silently falls "
                "through to ANONYMOUS (Viewer), which surfaces as an RBAC 403 "
                "rather than a 401 — so this usually means the same password "
                "drift as above, not a missing role. Confirm with:\n"
                '  curl -s -u "admin:$GRAFANA_PASSWORD" '
                f"{DEFAULT_URL}/api/user\n"
                f"detail: {detail}"
            ) from e
        raise GrafanaError(f"{e.code} on {method} {url}: {detail}") from e
    except urllib.error.URLError as e:
        raise GrafanaError(f"could not reach {url}: {e.reason}") from e


def find_folder(url: str, auth: str, title: str) -> dict:
    folders = _request(f"{url}/api/folders?limit=1000", auth)
    if not isinstance(folders, list):
        raise GrafanaError("unexpected /api/folders response")
    for folder in folders:
        if str(folder.get("title", "")).strip() == title:
            return folder
    known = ", ".join(sorted(repr(f.get("title")) for f in folders)) or "(none)"
    raise GrafanaError(
        f"no folder titled {title!r}. Deploy the chart first so provisioning "
        f"creates it. Folders present: {known}"
    )


def describe(items: list[dict]) -> str:
    if not items:
        return "(none — nobody but org admins)"
    out = []
    for item in items:
        who = (
            item.get("role")
            or item.get("teamId") and f"team:{item['teamId']}"
            or item.get("userId") and f"user:{item['userId']}"
            or "?"
        )
        level = _LEVEL_NAMES.get(item.get("permission"), item.get("permission"))
        out.append(f"{who}={level}")
    return ", ".join(out)


def restricted_items(current: list[dict]) -> list[dict]:
    """Current permissions minus any `Viewer`-role grant.

    Keeps team/user grants and the Editor role untouched, so this is safe to run
    over a folder someone has hand-tuned. Anonymous users are Viewer, so dropping
    that role is what actually closes the folder."""
    keep: list[dict] = []
    for item in current:
        if str(item.get("role") or "").strip().lower() == "viewer":
            continue
        entry: dict = {"permission": item.get("permission", EDIT)}
        for key in ("role", "teamId", "userId"):
            if item.get(key):
                entry[key] = item[key]
        keep.append(entry)
    if not any(str(i.get("role") or "").lower() == "editor" for i in keep):
        # Without an Editor grant an OAuth-authenticated member (auto-assigned
        # Editor) would lose the folder too — that is not "restricted", that is
        # "invisible". Add it back.
        keep.append({"role": "Editor", "permission": EDIT})
    return keep


def anonymous_status(url: str, dashboard_uid: str) -> int:
    """HTTP status an unauthenticated caller gets for a dashboard. 403 = gated."""
    req = urllib.request.Request(f"{url}/api/dashboards/uid/{dashboard_uid}")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        return e.code
    except urllib.error.URLError:
        return -1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default=os.environ.get("GRAFANA_URL", DEFAULT_URL))
    p.add_argument("--folder", default=DEFAULT_FOLDER, help="folder TITLE")
    p.add_argument("--user", default=os.environ.get("GRAFANA_USER", "admin"))
    p.add_argument(
        "--password",
        default=os.environ.get("GRAFANA_PASSWORD"),
        help="admin password (default: $GRAFANA_PASSWORD)",
    )
    p.add_argument(
        "--use-token",
        action="store_true",
        help=(
            "authenticate with $GRAFANA_TOKEN instead of admin basic auth. Off by "
            "default on purpose: a stale, non-admin token yields a confusing RBAC "
            "403 on the permissions API instead of a clean 401."
        ),
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    url = args.url.rstrip("/")
    if args.use_token:
        token = os.environ.get("GRAFANA_TOKEN")
        if not token:
            print("error: --use-token but GRAFANA_TOKEN is unset", file=sys.stderr)
            return 2
        auth = f"Bearer {token}"
    else:
        if not args.password:
            print(
                "error: no admin password. Set GRAFANA_PASSWORD (Infisical: "
                "/transformers-ci grafana-admin-password) or pass --password.",
                file=sys.stderr,
            )
            return 2
        if os.environ.get("GRAFANA_TOKEN"):
            print(
                "note: ignoring $GRAFANA_TOKEN and using admin basic auth "
                "(pass --use-token to override)"
            )
        raw = f"{args.user}:{args.password}".encode()
        auth = "Basic " + base64.b64encode(raw).decode()

    try:
        folder = find_folder(url, auth, args.folder)
        uid = folder["uid"]
        print(f"folder: {args.folder!r} (uid={uid})")

        current = _request(f"{url}/api/folders/{uid}/permissions", auth)
        if not isinstance(current, list):
            raise GrafanaError("unexpected permissions response")
        print(f"  before: {describe(current)}")

        wanted = restricted_items(current)
        if describe(wanted) == describe(current):
            print("  already restricted — nothing to change")
        elif args.dry_run:
            print(f"  would set: {describe(wanted)}   [dry-run, nothing applied]")
        else:
            _request(
                f"{url}/api/folders/{uid}/permissions",
                auth,
                method="POST",
                payload={"items": wanted},
            )
            after = _request(f"{url}/api/folders/{uid}/permissions", auth)
            print(f"  after:  {describe(after if isinstance(after, list) else [])}")

        # Independent check: what an anonymous visitor actually gets.
        dashboards = _request(
            f"{url}/api/search?folderUIDs={urllib.parse.quote(uid)}&type=dash-db", auth
        )
        if isinstance(dashboards, list) and dashboards:
            print("  anonymous access to dashboards in this folder:")
            for dash in dashboards:
                duid = dash.get("uid")
                code = anonymous_status(url, str(duid))
                verdict = {403: "gated ✓", 200: "PUBLIC ✗"}.get(code, f"HTTP {code}")
                print(f"    {dash.get('title')!r} ({duid}): {verdict}")
                if code == 200 and not args.dry_run:
                    print(
                        "      still public — check that anonymous auth is enabled "
                        "with org_role=Viewer, and that no other grant "
                        "(team/user/Editor-on-anon) re-opens it"
                    )
        else:
            print("  (no dashboards in the folder yet — nothing to verify)")
    except GrafanaError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
