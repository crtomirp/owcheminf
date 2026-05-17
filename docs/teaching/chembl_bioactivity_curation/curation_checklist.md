# ChEMBL Bioactivity Curation Checklist

Use this checklist before modelling.

## Target selection

- [ ] Target identifier recorded.
- [ ] Target name recorded.
- [ ] Organism recorded.
- [ ] Target type recorded.
- [ ] Target confidence inspected.
- [ ] Query date recorded.

## Bioactivity filters

- [ ] Endpoint type selected, for example IC50, Ki, Kd, EC50.
- [ ] Units inspected and standardized.
- [ ] Activity relation inspected: `=`, `<`, `>`, `<=`, `>=`, `~`.
- [ ] pChEMBL availability checked.
- [ ] Assay type inspected.
- [ ] Data validity comments checked.

## Molecular structures

- [ ] Invalid SMILES removed or corrected.
- [ ] Salts and mixtures handled consistently.
- [ ] Molecules standardized with a documented profile.
- [ ] Duplicates identified.
- [ ] Final canonical SMILES recorded.

## Duplicate handling

- [ ] Repeated measurements grouped by compound.
- [ ] Aggregation rule selected: median, mean, best, most recent, or assay-specific.
- [ ] Number of measurements per compound recorded.
- [ ] Conflicting records flagged.

## Final dataset

- [ ] One row per modelling compound, unless repeated records are intentionally retained.
- [ ] Target variable clearly defined.
- [ ] Descriptor/fingerprint generation documented.
- [ ] Train/test or scaffold split documented.
- [ ] Applicability Domain plan defined.
