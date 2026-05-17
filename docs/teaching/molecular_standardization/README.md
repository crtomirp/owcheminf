# Teaching Package A: Molecular Standardization and Data Cleaning

This teaching package supports the **Mol Standardizer** widget in `chem-inf-widgets`.

Widget location:

```text
Cheminf - Processing → Mol Standardizer
```

## Purpose

Molecular standardization is one of the most important preprocessing steps in cheminformatics. The same compound can appear as a salt, mixture, charged form, tautomeric form, aromatic form, kekulized form, or invalid structure. If these differences are not handled carefully, downstream fingerprints, descriptors, QSAR models, similarity searches, and cyclic registry matching can become inconsistent.

This package teaches students how to clean and document molecular structures before modelling.

## Learning outcomes

After completing the activities, students should be able to:

1. explain why molecular standardization is needed before cheminformatics analysis;
2. identify invalid SMILES and sanitization problems;
3. distinguish salts, mixtures, parent structures, and counterions;
4. explain cleanup, normalization, metal disconnection, largest fragment selection, reionization, and uncharging;
5. recognize special cases such as nitro groups, quaternary ammonium compounds, zwitterions, and aromatic/kekulized forms;
6. decide which standardization settings are appropriate for QSAR, fingerprinting, docking poses, and data curation;
7. report standardization choices in a reproducible way.

## Recommended teaching sequence

1. `worksheet_01_why_standardization_matters.md`
2. `worksheet_02_invalid_smiles_and_sanitization.md`
3. `worksheet_03_salts_mixtures_and_largest_fragment.md`
4. `worksheet_04_charge_normalization_and_uncharging.md`
5. `worksheet_05_nitro_zwitterions_quaternary_ammonium.md`
6. `worksheet_06_aromaticity_and_kekulization.md`
7. `worksheet_07_tautomers_and_protonation_states.md`
8. `worksheet_08_standardization_before_fingerprints.md`
9. `worksheet_09_standardization_before_qsar.md`
10. `worksheet_10_reproducible_standardization_report.md`

## Core Orange workflow

```text
File → Mol Standardizer → Data Table
```

Recommended extensions:

```text
File → Mol Standardizer → Fingerprint Generator → Data Table
```

```text
File → Mol Standardizer → Cyclic Registry Fingerprint → Matched Registry Entries → Data Table
```

```text
File → Mol Standardizer → Mol Descriptors 2 → QSAR Regression
```

## Example files

Example data are provided in:

```text
examples/molecular_standardization/standardization_training_set.csv
examples/molecular_standardization/standardization_edge_cases.csv
```

## Important teaching message

Standardization is not a purely technical step. It is a chemical decision. The correct standardization protocol depends on the scientific question.

For example:

- QSAR modelling usually needs consistent parent-like structures.
- Docking pose analysis may require preserving the exact input pose and formal charge state.
- Fingerprint comparison may require canonicalization and normalization.
- Regulatory or provenance-sensitive workflows must preserve original structures and logs.
