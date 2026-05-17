#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0

if [[ $# -ge 1 && "$1" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 [--dry-run] <version> [date]" >&2
  echo "Example: $0 0.2.1" >&2
  echo "Example: $0 --dry-run 0.2.1 2026-05-18" >&2
  exit 1
fi

VERSION="$1"
RELEASE_DATE="${2:-$(date +%F)}"

if ! [[ "${VERSION}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "Version must use semantic version form X.Y.Z" >&2
  exit 1
fi

if ! [[ "${RELEASE_DATE}" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
  echo "Date must use YYYY-MM-DD format" >&2
  exit 1
fi

if ! git rev-parse --git-dir >/dev/null 2>&1; then
  echo "Not inside a git repository." >&2
  exit 1
fi

python3 - "$VERSION" "$RELEASE_DATE" "$DRY_RUN" <<'PY'
from __future__ import annotations

import pathlib
import re
import sys

version = sys.argv[1]
release_date = sys.argv[2]
dry_run = sys.argv[3] == "1"

root = pathlib.Path(".")
pyproject_path = root / "pyproject.toml"
changelog_path = root / "CHANGELOG.md"

pyproject_text = pyproject_path.read_text(encoding="utf-8")
match = re.search(r'(?m)^version = "([^"]+)"$', pyproject_text)
if not match:
    raise SystemExit("Could not find project version in pyproject.toml")
old_version = match.group(1)
new_pyproject_text = re.sub(
    r'(?m)^version = "([^"]+)"$',
    f'version = "{version}"',
    pyproject_text,
    count=1,
)

changelog_text = changelog_path.read_text(encoding="utf-8")
header = f"## [{version}] - {release_date}"
if re.search(rf"(?m)^## \[{re.escape(version)}\]\b", changelog_text):
    new_changelog_text = changelog_text
    inserted = False
else:
    skeleton = (
        f"{header}\n\n"
        "### Added\n"
        "- \n\n"
        "### Changed\n"
        "- \n\n"
        "### Fixed\n"
        "- \n\n"
    )
    anchor = "The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).\n\n"
    if anchor not in changelog_text:
        raise SystemExit("Could not find changelog insertion anchor")
    new_changelog_text = changelog_text.replace(anchor, anchor + skeleton, 1)
    inserted = True

if dry_run:
    print(f"Dry run OK")
    print(f"- pyproject.toml: {old_version} -> {version}")
    if inserted:
        print(f"- CHANGELOG.md: would insert {header}")
    else:
        print(f"- CHANGELOG.md: {header} already exists")
else:
    pyproject_path.write_text(new_pyproject_text, encoding="utf-8")
    changelog_path.write_text(new_changelog_text, encoding="utf-8")
    print(f"Updated pyproject.toml: {old_version} -> {version}")
    if inserted:
        print(f"Inserted changelog skeleton: {header}")
    else:
        print(f"Changelog entry already exists: {header}")
PY

cat <<EOF

Next steps:
1. Fill in the CHANGELOG bullets for ${VERSION}
2. Review the diff
3. Run ./scripts/release_tag.sh --check-only ${VERSION}
EOF
