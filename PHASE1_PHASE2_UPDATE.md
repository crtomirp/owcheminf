# Phase 1/2 Update: Import Gatekeeper and Duplicate Handling

This update extends the Phase 1/2 stabilization work with a safer Molecule Import Hub.

## Added

- Import records now include `inchikey`, `mol_id`, duplicate metadata, acceptance status, and rejection reason.
- Import summaries now report accepted/rejected records and duplicate groups/records.
- Molecule Import Hub now exposes additional outputs:
  - `Accepted Data`
  - `Accepted Molecules`
  - `Rejected Records`
- Duplicate structures are detected by InChIKey by default.
- Optional duplicate rejection: keep the first occurrence and reject later duplicate structures.
- Imported ChemMol objects receive audit properties:
  - `IMPORT_ACCEPTED`
  - `IMPORT_REJECTION_REASON`
  - `IMPORT_DUPLICATE_KEY`
  - `IMPORT_DUPLICATE_COUNT`
  - `IMPORT_DUPLICATE_GROUP_INDEX`

## Backward compatibility

- Existing `Data`, `Molecules`, `Import Report`, `Failed Records`, and `Import Summary` outputs remain available.
- By default, duplicate molecules are flagged but not rejected.

## Tests

Validated with:

```bash
PYTHONPATH=src pytest -q \
  tests/test_molecule_import_phase31.py \
  tests/test_molecule_contract_phase12.py \
  tests/test_molecule_qc_phase3.py \
  tests/test_mol_standardizer.py \
  tests/test_from_orange_smoke.py
python -m compileall -q src/chem_inf_widgets
```
