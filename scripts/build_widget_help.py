#!/usr/bin/env python
"""Generate Orange F1 help pages for the Chemoinformatics widgets.

What this does
--------------
Orange resolves a widget's F1 help through the ``orange.canvas.help`` entry
point (provider ``html-index``) declared in ``pyproject.toml``. That provider:

1. loads ``chem_inf_widgets.widgets.WIDGET_HELP_PATH`` -> ``help/index.html``;
2. builds a map of *link text -> href* from ``<li><a>`` items found under the
   element with ``id="widgets"``;
3. on F1, looks up the widget's ``name`` (lower-cased) in that map and opens the
   matching page.

So the only hard requirement is: **the link text in ``index.html`` must equal
the widget's ``name``**. We therefore read the *canonical* names straight from
the widget descriptions (the same path Orange uses) instead of guessing from
file names, then render one HTML page per ``docs/widgets/*.md`` source.

Usage
-----
    python scripts/build_widget_help.py            # build into the package
    python scripts/build_widget_help.py --check     # fail if anything is unmapped

Re-run it whenever you add a widget or edit a doc, and before building a wheel.
The output (``src/chem_inf_widgets/widgets/help/``) is a generated artifact.
"""
from __future__ import annotations

import argparse
import html
import re
import sys
from importlib import import_module
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs" / "widgets"
OUT_DIR = ROOT / "src" / "chem_inf_widgets" / "widgets" / "help"
GITHUB_BLOB = "https://github.com/crtomirp/owcheminf/blob/main/"

# Widgets whose doc file is not simply ``ow_<x>.py`` -> ``<x>.md``.
DOC_OVERRIDES = {
    "ow_chembl_dataretriever": "chembl_bioactivity_retriever.md",
    "ow_mol_editor": "editors.md",
    "ow_mol_ketcher_editor": "editors.md",
    "ow_reactionviewer": "reaction_viewer.md",
}

# Widgets whose F1 help should open an EXTERNAL webpage instead of a bundled
# page. Key = canonical widget ``name`` (exactly as it appears in Orange),
# value = absolute http(s) URL. Listed widgets get no local page generated; the
# URL is written straight into the index, so F1 opens it in the browser.
EXTERNAL_LINKS = {
    # "Drug Filter": "https://crtomirp.github.io/owcheminf/widgets/drug_filter.html",
    # "ChEMBL Browser": "https://www.ebi.ac.uk/chembl/",
}


# --------------------------------------------------------------------------- #
# 1. Canonical widget names (the F1 lookup keys)
# --------------------------------------------------------------------------- #
def discover_widgets():
    """Yield ``(module, widget_name, category, doc_path_or_None)`` for every widget.

    Names come from the real ``WidgetDescription`` so they always match what F1
    searches for, including widgets that compute their ``name`` dynamically.
    """
    pkg = import_module("chem_inf_widgets.widgets")
    for spec in pkg.get_category_specs("full"):
        category = spec["name"]
        for module_name in spec["modules"]:
            try:
                module = import_module(f"chem_inf_widgets.widgets.{module_name}")
                desc = pkg._widget_desc_from_local_module(module)
            except Exception as exc:  # optional-dependency widgets, etc.
                print(f"  ! skipped {module_name}: {exc}", file=sys.stderr)
                continue
            doc_name = DOC_OVERRIDES.get(module_name, module_name[len("ow_"):] + ".md")
            doc_path = DOCS_DIR / doc_name
            yield module_name, desc.name, category, (doc_path if doc_path.exists() else None)


# --------------------------------------------------------------------------- #
# 2. Markdown -> HTML
# --------------------------------------------------------------------------- #
def md_to_html(text: str) -> str:
    """Convert Markdown to an HTML fragment (requires the ``markdown`` package)."""
    try:
        import markdown
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("This script needs 'markdown': pip install -e .[dev]") from exc
    # tab_length=2: the docs indent nested list items by 2 spaces, not 4.
    return markdown.markdown(text, extensions=["fenced_code", "tables"], tab_length=2)


# --------------------------------------------------------------------------- #
# 3. Link rewriting (so help pages link sensibly from the help viewer)
# --------------------------------------------------------------------------- #
def rewrite_links(body: str) -> str:
    def repl(m):
        href = m.group(2)
        if href.startswith(("http://", "https://", "#", "mailto:")):
            return m.group(0)
        # sibling widget doc: foo.md -> foo.html
        if href.split("#")[0].endswith(".md"):
            base, _, frag = href.partition("#")
            target = Path(base).name[:-3] + ".html"
            return f'{m.group(1)}"{target}{("#" + frag) if frag else ""}"'
        # repo-relative path -> GitHub blob URL
        if href.startswith(("../", "./")):
            rel = (DOCS_DIR / href).resolve().relative_to(ROOT).as_posix()
            return f'{m.group(1)}"{GITHUB_BLOB}{rel}"'
        return m.group(0)

    return re.sub(r'(href=)"([^"]+)"', repl, body)


# --------------------------------------------------------------------------- #
# 4. Page + index templates
# --------------------------------------------------------------------------- #
PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>{title}</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
   max-width:820px;margin:2rem auto;padding:0 1.2rem;line-height:1.55;color:#1f2937}}
 h1,h2,h3{{line-height:1.25}} h1{{border-bottom:2px solid #e5e7eb;padding-bottom:.3rem}}
 code{{background:#f3f4f6;padding:.1em .35em;border-radius:4px;font-size:.9em}}
 pre{{background:#f6f8fa;padding:.9rem;border-radius:6px;overflow:auto}}
 pre code{{background:none;padding:0}} a{{color:#2563eb}}
 table{{border-collapse:collapse;width:100%}} th,td{{border:1px solid #e5e7eb;padding:.4rem .6rem}}
 th{{background:#f9fafb;text-align:left}}
</style></head><body>
{body}
</body></html>
"""

STUB = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>{title}</title></head><body>
<h1>{title}</h1>
<p>Reference documentation for this widget has not been written yet.</p>
<p>See the <a href="{repo}">project repository</a> for the latest information.</p>
</body></html>
"""


def build(check: bool = False) -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # category name -> list of (widget_name, page_filename)
    by_category: dict[str, list[tuple[str, str]]] = {}
    rendered: set[str] = set()
    missing: list[str] = []

    for module_name, name, category, doc_path in discover_widgets():
        if name in EXTERNAL_LINKS:
            # External URL: no local page, write the URL straight into the index.
            page = EXTERNAL_LINKS[name]
        elif doc_path is not None:
            page = doc_path.stem + ".html"
            if page not in rendered:
                body = rewrite_links(md_to_html(doc_path.read_text(encoding="utf-8")))
                (OUT_DIR / page).write_text(PAGE.format(title=html.escape(name), body=body),
                                            encoding="utf-8")
                rendered.add(page)
        else:
            missing.append(name)
            page = module_name[len("ow_"):] + ".html"
            if page not in rendered:
                (OUT_DIR / page).write_text(
                    STUB.format(title=html.escape(name), repo=GITHUB_BLOB.rstrip("/")),
                    encoding="utf-8")
                rendered.add(page)
        by_category.setdefault(category, []).append((name, page))

    # index.html — the link list F1 searches (id="widgets")
    sections = []
    for category in sorted(by_category):
        items = "\n".join(
            f'    <li><a href="{page}">{html.escape(name)}</a></li>'
            for name, page in sorted(by_category[category])
        )
        sections.append(f"  <h2>{html.escape(category)}</h2>\n  <ul>\n{items}\n  </ul>")
    index_body = "<h1>Chemoinformatics widgets</h1>\n" \
                 '<section id="widgets">\n' + "\n".join(sections) + "\n</section>"
    (OUT_DIR / "index.html").write_text(
        PAGE.format(title="Chemoinformatics widgets", body=index_body), encoding="utf-8")

    total = sum(len(v) for v in by_category.values())
    print(f"Wrote {len(rendered)} pages + index.html for {total} widgets -> {OUT_DIR}")
    if missing:
        print(f"  {len(missing)} widget(s) have stub pages (no docs/widgets/*.md): "
              + ", ".join(sorted(missing)))
    if check and missing:
        print("ERROR: --check requested and some widgets are undocumented.", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero if any widget has no markdown doc")
    args = ap.parse_args()
    return build(check=args.check)


if __name__ == "__main__":
    raise SystemExit(main())
