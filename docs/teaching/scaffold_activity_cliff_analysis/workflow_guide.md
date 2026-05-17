# Workflow Guide: Scaffold and Activity Cliff Analysis

## Minimal scaffold workflow

```text
File → Scaffold Analysis → Scaffold Summary → Data Table
```

Expected outputs:

- `Annotated Data`: original table plus Murcko/generic scaffold annotations.
- `Scaffold Summary`: scaffold counts and representative scaffold strings.
- `Annotated Molecules`: molecule objects with scaffold metadata when molecule input is used.

## Scaffold-aware QSAR workflow

```text
File
 → Mol Standardizer
 → Mol Descriptors 2 / Fingerprint Generator / Cyclic Registry Fingerprint
 → Scaffold Splitter
 → QSAR Regression
 → Applicability Domain
```

Suggested split:

```text
Train: 0.70
Validation: 0.15
Test: 0.15
Scaffold kind: Generic Murcko for broader generalisation tests
```

## Activity cliff workflow

```text
File
 → Mol Standardizer
 → Fingerprint Generator
 → Activity Cliff Finder
 → Cliff Pairs / Cliff Compounds / Scaffold Summary
```

Suggested settings for teaching:

```text
Similarity threshold: 0.60–0.80
Activity fold threshold: 10-fold
Activity scale: Log potency if using pIC50/pKi
Maximum pairs: 250
```

## Reporting checklist

When reporting scaffold or activity cliff results, record:

- input dataset name and version,
- SMILES column,
- standardization protocol,
- scaffold type: exact Murcko or generic Murcko,
- fingerprint type used for similarity,
- similarity threshold,
- activity variable,
- activity scale,
- fold-change threshold,
- number of molecules,
- number of valid structures,
- number of scaffold groups,
- number of detected cliff pairs.
