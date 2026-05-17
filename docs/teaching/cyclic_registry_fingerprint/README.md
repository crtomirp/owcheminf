# Teaching materials: Cyclic Registry Fingerprint

These materials support the **Cyclic Registry Fingerprint** Orange widget and its command-line interface. The widget computes a 4096-bit molecular fingerprint that combines a Morgan-like hashed section with interpretable registry-backed sections for heterocycles, carbocycles, functional groups, ring topology, and aromaticity/dehydro-sensitive patterns.

## Intended audience

The worksheets are suitable for undergraduate or MSc-level courses in cheminformatics, computer-aided drug design, medicinal chemistry, QSAR/QSPR, molecular modeling, or data-driven chemistry.

## Recommended workflow in Orange

```text
File → Cyclic Registry Fingerprint → Data Table
                             └──→ Matched Registry Entries → Data Table
```

## Command-line workflow without Orange

```bash
owcheminf-cyclic-registry-fingerprint \
  examples/cyclic_registry_fingerprint/cyclic_registry_training_set.csv \
  --smiles-column smiles \
  --name-column name \
  --out-prefix outputs/training_set_crfp \
  --write-json
```

Outputs:

- `training_set_crfp.active_bits.csv`: one row per valid molecule with active bit IDs and names.
- `training_set_crfp.matches.csv`: interpretable registry matches.
- `training_set_crfp.failed.csv`: invalid or empty input molecules.
- `training_set_crfp.summary.json`: compact provenance and statistics.

## Worksheets

1. `worksheet_01_drug_heterocycles.md` — identifying heterocycles in drug-like molecules.
2. `worksheet_02_aromatic_vs_kekule.md` — aromatic versus kekulized input forms.
3. `worksheet_03_natural_products.md` — cyclic motifs in natural-product-like compounds.
4. `worksheet_04_qsar_qspr.md` — using the fingerprint in QSAR/QSPR models.
5. `worksheet_05_feature_interpretation.md` — linking important model features to registry entries.
6. `worksheet_06_library_comparison.md` — comparing compound libraries.
7. `worksheet_07_docking_results.md` — profiling top docking candidates.
8. `worksheet_08_unwanted_motifs.md` — filtering reactive or undesirable motifs.
9. `worksheet_09_smarts_learning.md` — learning SMARTS and substructure patterns.
10. `worksheet_10_method_development.md` — developing and validating a publishable fingerprint.

## Suggested assessment

Each worksheet can be assessed using four criteria: correct workflow construction, chemically meaningful interpretation, appropriate use of the matched-entry table, and critical reflection on limitations such as bit collisions, tautomerism, protonation state, and registry coverage.
