# Phase 1–2 Stabilization Notes

This package now includes the first infrastructure needed to turn OWChemInf into a more serious cheminformatics workflow platform.

## Phase 1: core stability

Added shared chemcore primitives:

- `chemcore/result.py`
  - `ServiceIssue`
  - `ServiceResult`
  - `count_issues`
- `chemcore/errors.py`
  - `ChemCoreError`
  - `MoleculeContractError`
  - `MoleculeParsingError`
- `chemcore/services/safe_feature_selection.py`
  - `safe_f_regression()` to prevent repeated sklearn `RuntimeWarning: invalid value encountered in sqrt` warnings from constant or numerically unstable descriptor columns.

## Phase 2: molecule table contract

Added `chemcore/molecule_contract.py` with stable shared field names:

- `mol_id`
- `input_smiles`
- `canonical_smiles`
- `standardized_smiles`
- `inchikey`
- `qc_status`
- `qc_severity`
- `qc_issue_codes`
- `standardization_status`
- `standardization_profile`
- `source_format`
- `source_row_index`

Integrated this contract into:

- Molecule Import Hub service
- Orange table to ChemMol conversion
- ChemMol to Orange table conversion
- Molecule Standardizer service
- Molecule QC service input handling

## Standardization audit

The Mol Standardizer widget now emits a `Standardization Report` output table with:

- row index
- source (`table` or `molecules`)
- ok flag
- input SMILES
- standardized SMILES
- status
- profile
- log

## Regression tests added

Added:

- `tests/test_molecule_contract_phase12.py`

Smoke-tested together with existing import, QC, standardization and table-conversion tests.
