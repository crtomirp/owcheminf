# Teaching Package C: Scaffold and Activity Cliff Analysis

This teaching package supports practical lessons on scaffold analysis, scaffold-aware dataset splitting, and activity cliff detection in cheminformatics and QSAR/QSPR workflows.

The package is designed for the following widgets:

- **Scaffold Analysis**
- **Scaffold Splitter**
- **Activity Cliff Finder**
- **Mol Standardizer**
- **Fingerprint Generator**
- **Cyclic Registry Fingerprint**
- **Mol Descriptors 2**
- **QSAR Regression**
- **Applicability Domain**

## Main learning idea

Chemical datasets are not collections of independent random points. Molecules often form scaffold families, analog series, and highly similar pairs. A random train/test split can therefore overestimate model performance. Scaffold analysis and activity cliff detection help students see when a model is learning general chemical principles and when it is mainly memorising close analogues.

## Recommended sequence

1. Worksheet 01 — What molecular scaffolds represent
2. Worksheet 02 — Murcko and generic scaffold analysis
3. Worksheet 03 — Scaffold frequency and chemical series
4. Worksheet 04 — Scaffold-based train/validation/test splitting
5. Worksheet 05 — Random split versus scaffold split
6. Worksheet 06 — Activity cliff concepts
7. Worksheet 07 — Detecting activity cliffs with fingerprints
8. Worksheet 08 — Interpreting cliff pairs medicinally
9. Worksheet 09 — Connecting scaffold, activity cliff, QSAR, and AD results
10. Worksheet 10 — Capstone scaffold and activity cliff project

## Example data

The folder `examples/scaffold_activity_cliff_analysis/` contains small teaching datasets with compound names, SMILES, scaffold family labels, and synthetic pIC50 values. These data are intentionally small so that students can inspect every compound and every detected pair.

## Suggested Orange workflows

### Scaffold analysis

```text
File → Mol Standardizer → Scaffold Analysis → Data Table
                                  ↓
                           Scaffold Summary → Data Table
```

### Scaffold split for QSAR

```text
File → Mol Standardizer → Mol Descriptors 2 → Scaffold Splitter
                                                 ↓
                                      Train / Validation / Test Data
```

### Activity cliff detection

```text
File → Mol Standardizer → Fingerprint Generator → Activity Cliff Finder → Cliff Pairs → Data Table
                                                               ↓
                                                        Scaffold Summary → Data Table
```

## Important teaching note

Activity cliffs are not "errors". They are chemically informative cases where a small structural change causes a large activity change. They often reveal limitations of simple similarity-based reasoning and can explain why QSAR models struggle on some regions of chemical space.
