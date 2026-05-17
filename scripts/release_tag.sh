#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <version>" >&2
  echo "Example: $0 0.2.1" >&2
  exit 1
fi

VERSION="$1"
TAG="v${VERSION}"

if ! git rev-parse --git-dir >/dev/null 2>&1; then
  echo "Not inside a git repository." >&2
  exit 1
fi

if [[ -n "$(git status --short)" ]]; then
  echo "Working tree is not clean. Commit or stash changes first." >&2
  git status --short
  exit 1
fi

PYPROJECT_VERSION="$(python3 - <<'PY'
import pathlib, tomllib
data = tomllib.loads(pathlib.Path('pyproject.toml').read_text(encoding='utf-8'))
print(data['project']['version'])
PY
)"

if [[ "${PYPROJECT_VERSION}" != "${VERSION}" ]]; then
  echo "pyproject.toml version is '${PYPROJECT_VERSION}', expected '${VERSION}'." >&2
  exit 1
fi

if ! rg -n "^## \\[${VERSION//./\\.}\\]" CHANGELOG.md >/dev/null 2>&1; then
  echo "CHANGELOG.md does not contain a header for ${VERSION}." >&2
  exit 1
fi

if git rev-parse "${TAG}" >/dev/null 2>&1; then
  echo "Tag ${TAG} already exists locally." >&2
  exit 1
fi

echo "Creating annotated tag ${TAG} from $(git rev-parse --short HEAD)"
git tag -a "${TAG}" -m "Release ${TAG}"

echo "Pushing main"
git push origin main

echo "Pushing tag ${TAG}"
git push origin "${TAG}"

cat <<EOF

Release tag ${TAG} was created and pushed.

Next steps:
1. Open GitHub Releases and publish the release notes for ${TAG}
2. Check the 'Release Assets' workflow
3. Run the 'Publish PyPI' workflow for TestPyPI or PyPI
EOF
