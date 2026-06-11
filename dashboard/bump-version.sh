#!/usr/bin/env bash
# Single source of truth for the dashboard version.
#
# Bumps dashboard/VERSION and stamps the same value into the
# `dashboard_version` Grafana template constant in the CI Health dashboard
# (pytest-observability-stack-health.json), which drives the version banner /
# release link. Run this when tagging a release, then commit + tag:
#
#   dashboard/bump-version.sh 0.1.3
#   git add dashboard/VERSION dashboard/pytest-observability-stack-health.json
#   git commit -m "bump dashboard to v0.1.3"
#   git tag v0.1.3
#
# Only the `dashboard_version` templating object is rewritten; the rest of the
# JSON is left byte-for-byte unchanged so the diff stays minimal.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERSION_FILE="${SCRIPT_DIR}/VERSION"
DASHBOARD_JSON="${SCRIPT_DIR}/pytest-observability-stack-health.json"

NEW="${1:-}"
if [[ -z "$NEW" ]]; then
  echo "usage: $(basename "$0") <X.Y.Z>" >&2
  echo "current: $(cat "$VERSION_FILE" 2>/dev/null || echo '?')" >&2
  exit 1
fi
if [[ ! "$NEW" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "version must be semver X.Y.Z, got: $NEW" >&2
  exit 1
fi
[[ -f "$DASHBOARD_JSON" ]] || { echo "missing $DASHBOARD_JSON" >&2; exit 1; }

OLD="$(cat "$VERSION_FILE" 2>/dev/null || true)"

# 1. The plain VERSION file.
printf '%s\n' "$NEW" > "$VERSION_FILE"

# 2. Surgically rewrite only the `dashboard_version` templating constant
#    (query + current.text/value + options[0].text/value), preserving the rest
#    of the file exactly so the git diff is small.
NEW="$NEW" python3 - "$DASHBOARD_JSON" <<'PY'
import os, re, sys

path = sys.argv[1]
new = os.environ["NEW"]
src = open(path).read()

# Locate the dashboard_version object: from its "name" key to the matching
# close brace (the next "}" at the same indentation, before "skipUrlSync"'s
# object ends).
marker = re.search(r'\{\s*"name":\s*"dashboard_version"', src)
if not marker:
    sys.exit("could not find dashboard_version templating object")

# Walk braces from the object's opening "{" to find its end.
start = marker.start()
depth = 0
end = None
for i in range(start, len(src)):
    if src[i] == "{":
        depth += 1
    elif src[i] == "}":
        depth -= 1
        if depth == 0:
            end = i + 1
            break
if end is None:
    sys.exit("unbalanced braces in dashboard_version object")

block = src[start:end]
# Replace any X.Y.Z string literal inside this block with the new version.
new_block = re.sub(r'(")\d+\.\d+\.\d+(")', r'\g<1>%s\g<2>' % new, block)

if new_block == block:
    sys.exit("no version literals found to update in dashboard_version object")

open(path, "w").write(src[:start] + new_block + src[end:])
PY

echo "dashboard version: ${OLD:-?} -> $NEW"
echo "updated:"
echo "  $VERSION_FILE"
echo "  $DASHBOARD_JSON (dashboard_version constant)"
echo
echo "next:"
echo "  git add dashboard/VERSION dashboard/pytest-observability-stack-health.json"
echo "  git commit -m \"bump dashboard to v$NEW\" && git tag v$NEW"
