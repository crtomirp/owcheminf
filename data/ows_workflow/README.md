# Teaching Experiment Workflows

This folder contains Orange workflow files for the five teaching experiments
described in [`data/7_teaching_workflows/`](../7_teaching_workflows).

Workflows:

- `01_standardization_and_filtering.ows`
  Use with `01_standardization_smiles.csv`
- `02_diversity_picker.ows`
  Use with `02_diversity_smiles.csv`
- `03_activity_cliffs.ows`
  Use with `03_activity_cliffs_vegfr2_ic50.csv`
- `04_qsar_regression.ows`
  Use with `04_qsar_vegfr2_ic50.csv`
- `05_applicability_domain.ows`
  Use with both `05_ad_reference_vegfr2_ic50.csv` and `05_ad_query_vegfr2_ic50.csv`
- `06_chemical_space_map.ows`
  Use with `04_qsar_vegfr2_ic50.csv`
- `07_similarity_clustering.ows`
  Use with `02_diversity_smiles.csv`
- `08_outlier_hunting.ows`
  Use with `04_qsar_vegfr2_ic50.csv`
- `09_activity_landscape.ows`
  Use with `03_activity_cliffs_vegfr2_ic50.csv`
- `10_cluster_aware_qsar.ows`
  Use with `04_qsar_vegfr2_ic50.csv`
- `11_compound_detail_database_search.ows`
  Use with any SMILES-containing compound library CSV; select one compound in `Compound Detail Card`, choose one or more heterocycles or functional groups, and launch PharmaFP-guided search with `AND/OR` motif logic.
- `12_motif_query_substructure_search.ows`
  Use with any SMILES-containing compound library CSV; inspect one compound, select a motif row in `Motif Queries`, and run exact SMARTS containment search over the library.
- `13_multi_motif_hybrid_search.ows`
  Use with any SMILES-containing compound library CSV; select multiple heterocycles and functional groups in `Compound Detail Card` and run hybrid PharmaFP search with `AND/OR` motif logic.
- `14_mol_editor_smoke_test.ows`
  Use with any SMILES-containing compound library CSV; feeds the same input table into `Mol Editor` and `Mol Ketcher` so you can compare loading, editing, and downstream outputs side by side.
- `15_heterocyclic_fingerprint_teaching.ows`
  Use with `15_heterocyclic_fingerprint_demo.csv` or any SMILES-containing compound library enriched in rings and heterocycles; compare the original molecules with the 4096-bit cyclic registry fingerprint matrix and the `Matched Registry Entries` explanation table.

Notes:

- The `.ows` files intentionally do not hardcode local file paths, so they stay
  portable.
- After opening a workflow, load the matching CSV file(s) in the `File`
  widget(s).
- The activity cliff workflow includes a `Pair Viewer` that shows the selected
  pair side by side and sends both compounds onward as a two-row table.
- The compound detail workflow demonstrates how a single inspected molecule can
  emit `Query Molecule`, `Fragment Queries`, `Scaffold Query`, and
  `Search Profile` outputs for downstream database search.
- The motif workflows demonstrate two complementary search styles:
  exact motif containment via `Substructure Search`, and ranked multi-motif
  retrieval via `PharmaFP Search`.
- The smoke-test workflow is intentionally minimal and is useful after editor
  or WebEngine changes, especially on macOS.
- In the heterocyclic fingerprint workflow, enable `Output Molecules with attached fp schema`
  inside the fingerprint widget if you also want the `Fingerprint Molecules`
  viewer to populate from the widget output.
- The mixed Orange workflows are intentionally light on preset controls, so
  projections, group variables, and box-plot targets can be chosen interactively
  during teaching.
- For the QSAR and Applicability Domain workflows, keep the same descriptor
  generation settings on both branches.
- In the QSAR workflow, set `pChEMBL` as the regression target if Orange does
  not infer it automatically.
