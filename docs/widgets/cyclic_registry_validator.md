# Cyclic registry validator

The Cyclic Registry Fingerprint widget uses a packaged SMARTS registry and a
fixed 4096-bit layout.  The validator checks that the registry is internally
consistent and reports how registry entries map into the fingerprint bit ranges.

## Run the validator

From a source checkout:

```bash
python -m chem_inf_widgets.chemcore.tools.validate_cyclic_registry
```

After installation:

```bash
owcheminf-validate-cyclic-registry
```

## Useful options

Skip SMARTS compilation for a quick metadata/collision report:

```bash
owcheminf-validate-cyclic-registry --no-smarts
```

Print JSON:

```bash
owcheminf-validate-cyclic-registry --json
```

Write files for publication supplements or CI artifacts:

```bash
owcheminf-validate-cyclic-registry \
  --report-json cyclic_registry_report.json \
  --collision-csv cyclic_registry_collisions.csv
```

Fail CI if invalid SMARTS, duplicate IDs, or bit-layout errors are found:

```bash
owcheminf-validate-cyclic-registry --fail-on-errors
```

## Report sections

The validator reports:

- number of analyzed registry entries,
- validation errors and warnings,
- counts by registry group,
- collision statistics for every bit section,
- examples of collision bits and the entries mapped to them.

Collisions are not automatically errors.  The fingerprint deliberately uses
fixed-size bit ranges, so collisions are expected.  They become a problem only if
the rate is too high for a given section or if collisions hide chemically
important patterns that should be assigned dedicated bits.
