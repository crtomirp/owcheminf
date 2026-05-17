from __future__ import annotations

from pathlib import Path

from chem_inf_widgets.chemcore.tools.clean_repo_cli import cleanup_project_tree


def test_cleanup_project_tree_dry_run_lists_expected_artifacts(tmp_path: Path):
    (tmp_path / "build").mkdir()
    (tmp_path / ".setuptools-egg-info" / "chem_inf_widgets.egg-info").mkdir(parents=True)
    (tmp_path / "src" / "pkg" / "__pycache__").mkdir(parents=True)
    (tmp_path / "tests" / "__pycache__").mkdir(parents=True)
    (tmp_path / "src" / "pkg" / "mod.pyc").write_bytes(b"pyc")

    removed = cleanup_project_tree(tmp_path, dry_run=True)

    assert "build" in removed
    assert ".setuptools-egg-info/chem_inf_widgets.egg-info" in removed
    assert "src/pkg/__pycache__" in removed
    assert "tests/__pycache__" in removed
    assert "src/pkg/mod.pyc" in removed


def test_cleanup_project_tree_removes_artifacts_but_not_placeholder(tmp_path: Path):
    tracked_dir = tmp_path / ".setuptools-egg-info"
    tracked_dir.mkdir()
    (tracked_dir / ".gitkeep").write_text("", encoding="utf-8")
    egg_dir = tracked_dir / "chem_inf_widgets.egg-info"
    egg_dir.mkdir()
    (egg_dir / "PKG-INFO").write_text("demo", encoding="utf-8")
    pycache_dir = tmp_path / "src" / "pkg" / "__pycache__"
    pycache_dir.mkdir(parents=True)

    removed = cleanup_project_tree(tmp_path, dry_run=False)

    assert ".setuptools-egg-info/chem_inf_widgets.egg-info" in removed
    assert "src/pkg/__pycache__" in removed
    assert not egg_dir.exists()
    assert not pycache_dir.exists()
    assert tracked_dir.exists()
    assert (tracked_dir / ".gitkeep").exists()
