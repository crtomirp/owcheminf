# Implementation Notes — F1 Widget Help

This document records the changes that add **F1 (in-app) help** to the
Chemoinformatics widgets, reusing the existing `docs/widgets/*.md` files. It is a
change log + rationale; the contributor-facing how-to lives in
[`docs/developer/widget_help.md`](docs/developer/widget_help.md).

## 1. Goal

Pressing **F1** on any widget in `orange-canvas` should open that widget's
documentation, built from the Markdown we already maintain in `docs/widgets/`.

## 2. How Orange resolves F1 help (the mechanism we target)

Orange does **not** read a per-widget URL. On F1 it:

1. reads an `orange.canvas.help` **entry point** on the add-on distribution; its
   *name* selects a provider — we use **`html-index`**;
2. that provider loads `WIDGET_HELP_PATH` → `help/index.html`, and builds a map of
   *link text → href* from `<li><a>` items under the element with `id="widgets"`
   (xpath `.//*[@id='widgets']//li/a`);
3. it looks up the widget's **`name`** (lower-cased) in that map and opens the page.

Hard requirement: **index link text must equal the widget's `name`.** Names are
therefore read from the canonical `WidgetDescription`, never guessed from file
names (several widgets compute their name at runtime).

## 3. Files changed

| File | Change |
| --- | --- |
| `src/chem_inf_widgets/widgets/__init__.py` | Added `WIDGET_HELP_PATH`; added `DISTRIBUTION_NAME` and stamped `project_name` on every widget/category (**bug fix**, see §5); fixed a tuple-comma bug (**bug fix**, see §5). |
| `pyproject.toml` | Registered `[project.entry-points."orange.canvas.help"]` → `html-index = …:WIDGET_HELP_PATH`; added `help/*.html` to `package-data`; added `markdown` to the `dev` extra. |
| `scripts/build_widget_help.py` | **New** generator: reads canonical names, converts `docs/widgets/*.md` → HTML, emits `help/index.html`. |
| `tests/test_widget_help.py` | **New** test: every widget resolves to a help page (no Qt/network). |
| `docs/developer/widget_help.md` | **New** contributor how-to (mechanism, build, add-a-widget, external links, the `project_name` gotcha). |
| `docs/developer.md`, `docs/widgets/README.md` | Links/notes pointing at the new doc and build step. |
| `.gitignore` | Ignores the generated `src/chem_inf_widgets/widgets/help/`. |

## 4. The generator — `scripts/build_widget_help.py`

```bash
python scripts/build_widget_help.py          # build help/ into the package
python scripts/build_widget_help.py --check    # CI: non-zero if a widget has no doc
```

- **Names**: `discover_widgets()` walks `get_category_specs("full")` and reads each
  widget's canonical `name` via `_widget_desc_from_local_module` (same path Orange
  uses).
- **Doc mapping**: `ow_<x>.py` → `docs/widgets/<x>.md`, with `DOC_OVERRIDES` for the
  four exceptions (e.g. `ow_mol_editor` → `editors.md`).
- **Markdown → HTML**: uses the `markdown` package (`fenced_code`, `tables`,
  `tab_length=2` so the docs' 2-space nested lists nest correctly).
- **Links**: sibling `*.md` links → `*.html`; repo-relative links → GitHub blob URLs.
- **Stubs**: widgets without a Markdown file get a placeholder page so F1 never
  dead-ends (currently 6: Atom Contribution Map, Audit Trail Viewer, Dataset
  Profiler, Molecule QC Dashboard, RDKit Descriptors, Widget Smoke Tester).
- **Index**: `help/index.html` lists every widget by canonical name under
  `id="widgets"`, grouped by category.
- **External links** (optional): fill the `EXTERNAL_LINKS` map (`{name: url}`) to
  make a widget's F1 open an external page instead of a bundled one.

The output `help/` folder is a **generated artifact** (git-ignored). Rebuild after
editing docs and before building a wheel.

## 5. Bugs found and fixed during the work

1. **`project_name` never stamped → F1 silently did nothing.**
   This add-on registers widgets through a custom `widget_discovery()` function.
   Orange short-circuits that case (`process_category_package` → `process_loader` →
   `widget_discovery(self)`) and **never passes the distribution down**, so
   `desc.project_name` stayed `None`. `HelpManager.search()` only consults a
   provider when `project_name` is set, so help resolution was skipped entirely.
   Fix: set `desc.project_name = DISTRIBUTION_NAME` (`"chem-inf-widgets"`) on every
   widget and category in `__init__.py`.

2. **Missing trailing comma in the Filters category `modules`.**
   `("ow_drug_filter")` is a *string*, not a tuple; discovery would iterate its
   characters and crash that category and every category after it. Fixed to
   `("ow_drug_filter",)`.

## 6. Leanness pass

- Removed the ~90-line hand-rolled Markdown fallback converter from the generator
  and now require the `markdown` package (already a `dev` dependency). Script went
  from 317 → 209 lines, and nested-list/table rendering improved.
- `tests/test_widget_help.py` skips its build test when `markdown` is absent.
- Trimmed placeholder comments in `__init__.py`.

## 7. Use it

```bash
python scripts/build_widget_help.py    # generate help pages
pip install -e .                        # once, to register the help entry point
orange-canvas                           # drop a widget, press F1
```

If F1 still does nothing after a code change, restart the canvas (the registry is
rebuilt on launch); to force a rebuild: `rm -rf ~/.cache/Orange/*/widget-registry*`.

### Add help for a new widget
1. Create `docs/widgets/<x>.md` starting with `# <Widget Name>`.
2. If the doc name isn't `ow_<x>.py` → `<x>.md`, add a line to `DOC_OVERRIDES`.
3. Re-run the generator and press F1.

## 8. Verification performed

- Real Orange discovery loads all 5 categories; `project_name` stamped.
- The full `HelpManager.search()` path (what F1 calls) resolves widgets to their
  pages, e.g. `Drug Filter → drug_filter.html`, `QSAR/QSPR Model Hub →
  qsar_model_hub.html`.
- `tests/test_widget_help.py` (2) and `tests/test_packaging_smoke.py` (11) pass.
