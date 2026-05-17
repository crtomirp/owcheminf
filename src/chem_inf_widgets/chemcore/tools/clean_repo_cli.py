from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path


_DIRECTORIES_TO_REMOVE = (
    "build",
    "dist",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
)
_FILE_GLOBS_TO_REMOVE = ("*.pyc", "*.pyo")


@dataclass(frozen=True)
class CleanupEntry:
    path: Path
    kind: str


def _iter_cleanup_entries(project_root: Path) -> list[CleanupEntry]:
    entries: list[CleanupEntry] = []

    for relative in _DIRECTORIES_TO_REMOVE:
        path = project_root / relative
        if path.exists():
            entries.append(CleanupEntry(path=path, kind="dir"))

    for egg_dir in (project_root / ".setuptools-egg-info").glob("*.egg-info"):
        if egg_dir.is_dir():
            entries.append(CleanupEntry(path=egg_dir, kind="dir"))

    for base in (project_root / "src", project_root / "tests"):
        if not base.exists():
            continue
        for pycache_dir in base.rglob("__pycache__"):
            if pycache_dir.is_dir():
                entries.append(CleanupEntry(path=pycache_dir, kind="dir"))
        for pattern in _FILE_GLOBS_TO_REMOVE:
            for file_path in base.rglob(pattern):
                if file_path.is_file():
                    entries.append(CleanupEntry(path=file_path, kind="file"))

    # Keep the placeholder directory tracked, but remove stray source-side egg-info if present.
    for egg_dir in (project_root / "src").rglob("*.egg-info"):
        if egg_dir.is_dir():
            entries.append(CleanupEntry(path=egg_dir, kind="dir"))

    unique: dict[Path, CleanupEntry] = {}
    for entry in entries:
        unique[entry.path] = entry
    return sorted(unique.values(), key=lambda item: str(item.path))


def cleanup_project_tree(project_root: Path, *, dry_run: bool = False) -> list[str]:
    removed: list[str] = []
    for entry in _iter_cleanup_entries(project_root):
        rel = entry.path.relative_to(project_root).as_posix()
        removed.append(rel)
        if dry_run:
            continue
        if entry.kind == "dir":
            shutil.rmtree(entry.path, ignore_errors=True)
        else:
            try:
                entry.path.unlink()
            except FileNotFoundError:
                pass
    return removed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="owcheminf-clean-repo",
        description="Remove local build/cache artifacts from the chem-inf-widgets repository.",
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Repository root to clean. Defaults to the current working directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be removed without deleting anything.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    project_root = Path(args.project_root).resolve()
    if not project_root.exists():
        parser.error(f"Project root does not exist: {project_root}")
    removed = cleanup_project_tree(project_root, dry_run=bool(args.dry_run))
    action = "Would remove" if args.dry_run else "Removed"
    if removed:
        print(f"{action} {len(removed)} artifact(s):")
        for rel in removed:
            print(f" - {rel}")
    else:
        print("No cleanup artifacts found.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
