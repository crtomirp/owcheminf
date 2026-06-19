# Widget Help (F1)

How pressing **F1** on a Chemoinformatics widget opens its documentation, and how
to keep that documentation working as you add or rename widgets.

## How Orange resolves F1 help

Orange does **not** read a per-widget `help` URL. Instead it looks up an
[`orange.canvas.help`](https://orange-canvas-core.readthedocs.io/) **entry point**
declared by the add-on distribution. The entry-point *name* selects a provider:

| entry-point name | provider | matches on |
| --- | --- | --- |
| `intersphinx` | Sphinx `objects.inv` | `help_ref` or widget `name` |
| `html-simple` | base URL + `help_ref` | widget `help_ref` |
| `html-index` | an HTML index page | widget **`name`** |

We use **`html-index`** because it reuses the existing `docs/widgets/*.md` files
and needs **no per-widget code changes**. The flow on F1 is:

```
F1  ─▶  HelpManager.get_provider("chem-inf-widgets")
        └─ reads entry point  orange.canvas.help → html-index
           = chem_inf_widgets.widgets:WIDGET_HELP_PATH
        └─ HtmlIndexProvider loads help/index.html
           └─ builds { link-text(lower) : href } from  .//*[@id='widgets']//li/a
        └─ provider.search(desc) → looks up desc.name.lower() → opens that page
```

The one hard rule: **the link text in `index.html` must equal the widget's
`name`.** That is why the index is generated from the *canonical*
`WidgetDescription.name` (some widgets compute their name at runtime), never from
file names.

### Gotcha: `project_name` must be stamped on every widget

`HelpManager.search()` only looks up a provider when `desc.project_name` is set:

```python
if desc.project_name:
    provider = self.get_provider(desc.project_name)   # else provider stays None
```

Orange's *normal* discovery stamps `project_name` from the distribution — but this
add-on registers widgets through a custom `widget_discovery()` function, and Orange
takes a shortcut for that (`process_category_package` → `process_loader` →
`widget_discovery(self)`) that **never passes the distribution**. So we must set it
ourselves in `widgets/__init__.py`:

```python
DISTRIBUTION_NAME = "chem-inf-widgets"
...
desc.project_name = DISTRIBUTION_NAME   # in _iter_widget_descriptions()
```

Without this, `project_name` is `None`, no provider is consulted, and **F1 silently
does nothing** — even though the entry point, `WIDGET_HELP_PATH`, and `index.html`
are all correct.

## The three wiring pieces

1. **`src/chem_inf_widgets/widgets/__init__.py`** — `WIDGET_HELP_PATH`, a tuple of
   `(target, xpathquery)` candidates. The first existing target wins; ours points
   at the bundled `help/index.html` next to the package (works for both
   `pip install -e .` and wheels).

2. **`pyproject.toml`** — registers the provider and ships the pages:

   ```toml
   [project.entry-points."orange.canvas.help"]
   html-index = "chem_inf_widgets.widgets:WIDGET_HELP_PATH"

   [tool.setuptools.package-data]
   "chem_inf_widgets.widgets" = ["help/*.html", ...]
   ```

3. **`scripts/build_widget_help.py`** — the generator (see below).

## The generator: `scripts/build_widget_help.py`

```bash
python scripts/build_widget_help.py          # build help/ into the package
python scripts/build_widget_help.py --check    # CI: fail if a widget has no doc
```

What it does:

1. Imports the add-on and reads every widget's **canonical `name`** via the same
   discovery path Orange uses (`get_category_specs("full")` +
   `_widget_desc_from_local_module`).
2. Maps each widget to a doc: `ow_<x>.py` → `docs/widgets/<x>.md`, with a small
   `DOC_OVERRIDES` table for the exceptions (e.g. `ow_mol_editor` → `editors.md`).
3. Converts each Markdown file to a standalone HTML page (uses the `markdown`
   package if installed, otherwise a built-in fallback — no required extra dep).
4. Writes `help/index.html` containing a `<section id="widgets">` whose `<li><a>`
   link texts are the canonical widget names.

Widgets with no Markdown file get a small **stub** page so F1 never dead-ends; the
script lists them at the end (today: Atom Contribution Map, Audit Trail Viewer,
Dataset Profiler, Molecule QC Dashboard, RDKit Descriptors, Widget Smoke Tester).

The output (`src/chem_inf_widgets/widgets/help/`) is a **generated artifact** and
is git-ignored. Run the generator after editing docs and **before building a
wheel** (`python scripts/build_widget_help.py && python -m build`).

## Point help at an external webpage

`html-index` resolves each href with `inventory.resolved(QUrl(href))`, so an
**absolute `http(s)://` href is used as-is** while bare filenames stay local. Two
ways to use that:

* **Per widget** — add the widget to `EXTERNAL_LINKS` in
  `scripts/build_widget_help.py` (key = canonical widget `name`, value = URL).
  That widget gets no local page; F1 opens the URL. Everything else stays local.

  ```python
  EXTERNAL_LINKS = {
      "ChEMBL Browser": "https://www.ebi.ac.uk/chembl/",
  }
  ```

* **Whole add-on online** — make `WIDGET_HELP_PATH` point at a remote index page
  (it must expose the same `id="widgets"` link list with matching widget names):

  ```python
  WIDGET_HELP_PATH = (
      (os.path.join(_HELP_DIR, "index.html"), None),          # local first (offline)
      ("https://crtomirp.github.io/owcheminf/index.html", None),  # online fallback
  )
  ```

  Candidates are tried in order; a **local** target is used only if the file
  exists, so order it first for offline use, or list only the URL to force online.
  External help needs network access and opens in the help viewer / browser.

## Add help for a new widget

1. Create `docs/widgets/<x>.md` (start the file with `# <Widget Name>`).
2. If the doc file name is not `ow_<x>.py` → `<x>.md`, add one line to
   `DOC_OVERRIDES` in `scripts/build_widget_help.py`.
3. `python scripts/build_widget_help.py` and press **F1** on the widget in Orange.

> Changed an entry point or installed fresh? Re-run `pip install -e .` so the new
> `orange.canvas.help` entry point is written into the distribution metadata.

## Verifying without the GUI

```python
from AnyQt.QtWidgets import QApplication; QApplication([])
from orangecanvas.utils.pkgmeta import get_distribution
from orangecanvas.help.manager import get_help_provider_for_distribution

prov = get_help_provider_for_distribution(get_distribution("chem-inf-widgets"))
class D: name = "Drug Filter"
print(prov.search(D()).toString())   # → …/widgets/help/drug_filter.html
```
