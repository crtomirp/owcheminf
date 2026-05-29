"""Orange widget discovery for the Chemoinformatics add-on.

The public Orange sidebar is intentionally curated so the add-on feels like a
coherent chemoinformatics toolbox rather than a long list of prototype widgets.
Less mature, diagnostic, or highly specialised tools remain available under the
``Cheminf - Development`` category instead of being removed from the code base.
"""

from __future__ import annotations

import inspect
from importlib import import_module

from orangecanvas.registry import CategoryDescription, WidgetDescription
from orangewidget.workflow.discovery import widget_desc_from_module

PACKAGE_NAME = __name__

NAME = "Chemoinformatics"
DESCRIPTION = "Chemoinformatics widgets for Orange Data Mining."
PRIORITY = 1

# Curated categories for the v0.2.0 GitHub-ready layout.
#
# Design rule:
# - Core: robust entry-point widgets used in most workflows.
# - Search & Analysis: broadly useful cheminformatics analysis widgets.
# - Filters & Alerts: rule-based filters and structural alerts.
# - QSAR: compact public modeling workflow; older/overlapping QSAR widgets are
#   kept in Development until they are merged or retired.
# - Reactions: reaction-specific widgets.
# - Development: useful but experimental, diagnostic, optional-dependency-heavy,
#   or overlapping widgets that should not clutter the main user-facing palette.
_CATEGORY_SPECS = (
    {
        "name": "Cheminf - Core",
        "description": "Import, clean, standardize, inspect, and visualize molecular datasets.",
        "icon": "icons/categories/cheminf_data.svg",
        "background": "#EFF6FF",
        "priority": 1000,
        "modules": (
            "ow_molecule_import_hub",
            "ow_molecule_export_hub",
            "ow_sdf_reader",
            "ow_sdf_writer",
            "ow_chembl_browser",
            "ow_chembl_dataretriever",
            "ow_molecule_qc_dashboard",
            "ow_mol_standardizer",
            "ow_mol_editor",
            "ow_mol_ketcher_editor",
            "ow_compound_detail_card",
            "ow_mol_viewer",
            "ow_mol3d_viewer",
        ),
    },
    {
        "name": "Cheminf - Search & Analysis",
        "description": "Search, featurize, filter, cluster, and analyze molecular libraries.",
        "icon": "icons/categories/cheminf_processing.svg",
        "background": "#F0FDF4",
        "priority": 1001,
        "modules": (
            "ow_substructure_search",
            "ow_similarity_search",
            "ow_fingerprint_generator",
            "ow_rdkit_descriptors",
            "ow_mol_descriptor",
            "ow_scaffold_analysis",
            "ow_scaffold_splitter",
            "ow_diversity_picker",
            "ow_activity_cliff_finder",
            "ow_rgroup_decomposition",
            "ow_matched_molecular_pairs",
            "ow_pair_viewer",
        ),
    },
    {
        "name": "Cheminf - Filters & Alerts",
        "description": "Rule-based drug-likeness filters, structural alerts, and library triage tools.",
        "icon": "icons/categories/cheminf_processing.svg",
        "background": "#FEFCE8",
        "priority": 1002,
        "modules": (
            "ow_drug_filter",
        ),
    },
    {
        "name": "Cheminf - QSAR",
        "description": "Build QSAR-ready datasets, audit descriptors, train and validate models, and assemble reports or prediction packages.",
        "icon": "icons/categories/cheminf_modeling.svg",
        "background": "#FFF7ED",
        "priority": 1003,
        "modules": (
            "ow_qsar_dataset_builder",
            "ow_descriptor_explorer",
            "ow_descriptor_filter",
            "ow_qsar_model_hub",
            "ow_qsar_validation_dashboard",
            "ow_applicability_domain",
            "ow_model_explanation",
            "ow_qsar_report_generator",
            "ow_qsar_prediction_packager",
        ),
    },
    {
        "name": "Cheminf - Reactions",
        "description": "Inspect, enumerate, and apply RDKit reaction workflows.",
        "icon": "icons/categories/cheminf_processing.svg",
        "background": "#F5F3FF",
        "priority": 1004,
        "modules": (
            "ow_reactionviewer",
            "ow_reactor",
            "ow_reaction_enumerator",
        ),
    },
    {
        "name": "Cheminf - Development",
        "description": "Experimental, diagnostic, optional-dependency-heavy, and legacy widgets.",
        "icon": "icons/categories/cheminf_modeling.svg",
        "background": "#F8FAFC",
        "priority": 1999,
        "modules": (
            "ow_widget_smoke_tester",
            "ow_audit_trail_viewer",
            "ow_pharmafp_search",
            "ow_cyclic_registry_fingerprint",
            "ow_isida_descriptors",
            "ow_padel_descriptors",
            "ow_ad_workbench",
            "ow_atom_contribution_map",
            "ow_qsar_regression",
            "ow_mlr_model_selection",
        ),
    },
)


def _category_description(spec: dict[str, object]) -> CategoryDescription:
    return CategoryDescription(
        name=spec["name"],
        qualified_name=PACKAGE_NAME,
        package=PACKAGE_NAME,
        description=spec["description"],
        priority=spec["priority"],
        icon=spec["icon"],
        background=spec["background"],
    )


def _iter_widget_descriptions(spec: dict[str, object]):
    category_name = spec["name"]
    for module_name in spec["modules"]:
        module = import_module(f"{PACKAGE_NAME}.{module_name}")
        desc = _widget_desc_from_local_module(module)
        desc.category = category_name
        yield desc


def _widget_desc_from_local_module(module) -> WidgetDescription:
    """Return the widget description for a class defined in ``module``.

    Some widgets import other OWWidget classes at module scope for internal
    workflow smoke tests. Orange's default helper accepts the first class with
    ``get_widget_description`` it encounters, even if that class was merely
    imported from another module. Filtering by ``__module__`` ensures the
    registered widget always matches the current file.
    """
    for _, widget_class in inspect.getmembers(module, inspect.isclass):
        if widget_class.__module__ != module.__name__:
            continue
        if not hasattr(widget_class, "get_widget_description"):
            continue

        description = widget_class.get_widget_description()
        if description is None:
            continue

        desc = WidgetDescription(**description)
        desc.package = module.__package__
        desc.category = widget_class.category
        return desc

    return widget_desc_from_module(module)


def widget_discovery(discovery) -> None:
    """Register the curated Orange categories for this add-on."""
    from chem_inf_widgets.widgets.theme import apply_theme

    apply_theme()

    for spec in _CATEGORY_SPECS:
        discovery.handle_category(_category_description(spec))
        for desc in _iter_widget_descriptions(spec):
            discovery.handle_widget(desc)
