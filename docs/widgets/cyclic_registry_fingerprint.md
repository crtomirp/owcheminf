# Cyclic Registry Fingerprint Widget

The **Cyclic Registry Fingerprint** widget computes a standalone, interpretable
4096-bit molecular fingerprint for Orange workflows.  It is designed for
heterocycle-rich drug discovery and natural-product datasets, where a purely
hashed fingerprint can hide chemically meaningful ring information.

## Inputs

- **Data**: Orange Table containing a SMILES column.
- **Molecules**: list of `ChemMol`, RDKit `Mol`, or SMILES strings. If this input
  is present, it takes precedence over the table input.

## Outputs

- **Fingerprints**: Orange Table with 4096 numeric fingerprint columns.
- **Matched Registry Entries**: explanation table listing matched SMARTS registry
  entries, bit number, section, family, match count, and optionally atom indices.
- **Molecules**: optional list of molecules with a packed `fp` schema attached to
  `ChemMol.props`.

## Bit layout

| Bit range | Meaning |
|---:|---|
| 0-2047 | General Morgan/ECFP-like hashed section |
| 2048-3071 | Heterocycle registry section |
| 3072-3327 | Carbocycle registry section |
| 3328-3711 | Functional-group registry section |
| 3712-3839 | Ring-topology section |
| 3840-3967 | Aromaticity/dehydro-sensitive section |
| 3968-4095 | Reserved for future versions |

The registry-backed sections use stable hashing from registry entry identity to
bit position. Collisions are possible and expected, but the explanation output
keeps the exact registry matches so the fingerprint remains interpretable.

## Recommended use

1. Load an SDF or SMILES table.
2. Standardize molecules with the Molecule Standardizer widget, preferably using
   the `fingerprint_canonical` profile.
3. Run Cyclic Registry Fingerprint.
4. Use the **Fingerprints** output for QSAR, clustering, similarity, or model
   selection.
5. Inspect **Matched Registry Entries** to understand which heterocycles,
   carbocycles, or functional groups activated registry bits.

## Notes for publication

The fingerprint stores a versioned bit layout and the registry version in the
Molecules output. This is important for reproducibility. If the registry is
expanded or curated further, bump the registry version and keep old registry
files for exact reproduction of published models.


## Registry validation and collision report

Phase 2.1 adds a validator for the packaged registry. Run:

```bash
owcheminf-validate-cyclic-registry
```

For metadata-only checks without SMARTS compilation:

```bash
owcheminf-validate-cyclic-registry --no-smarts
```

For manuscript/QA artifacts:

```bash
owcheminf-validate-cyclic-registry --report-json cyclic_registry_report.json --collision-csv cyclic_registry_collisions.csv
```
