# Instructor Guide: Molecular Standardization and Data Cleaning

## Target audience

This package is suitable for undergraduate or graduate courses in cheminformatics, medicinal chemistry, pharmaceutical chemistry, computational chemistry, or data-driven chemistry.

## Estimated time

- Short version: 1 × 90 min session
- Full version: 3–4 practical sessions
- Capstone version: 1–2 weeks including a report

## Software setup

```bash
conda activate owcheminf
cd cinf
pip install -e .
orange-canvas
```

Check that the widget appears as:

```text
Cheminf - Processing → Mol Standardizer
```

## Recommended classroom workflow

Start with a deliberately messy dataset. Ask students to inspect the original structures before applying any standardization. Then let them compare the standardized output and the standardization log.

Suggested progression:

1. Load `standardization_training_set.csv`.
2. Run `Mol Standardizer` with default settings.
3. Inspect `SMILES_STD` and `STD_LOG` in `Data Table`.
4. Turn individual options on/off and compare outputs.
5. Discuss which settings are chemically appropriate for different downstream tasks.

## Suggested widget settings for teaching

For general QSAR preparation:

- Cleanup: on
- Normalize: on
- MetalDisconnector: on
- LargestFragmentChooser: on
- Reionizer: on
- Uncharger: on
- Sanitize before: on
- Sanitize after: on
- Keep original as `SMILES_ORIG`: on
- Overwrite `SMILES`: off

For docking pose preservation:

- Use a much more conservative protocol.
- Avoid changing protonation, charge state, or fragment composition unless this is explicitly part of the workflow.

## Assessment ideas

Students can be assessed on:

1. ability to identify problematic inputs;
2. correct choice of standardization settings;
3. interpretation of the standardization log;
4. comparison of descriptor/fingerprint outputs before and after standardization;
5. reproducible reporting of preprocessing decisions.

## Common misconceptions

### “Standardization always makes molecules correct.”

No. It makes molecules consistent according to a selected protocol. It does not guarantee that the chosen chemical form is biologically or experimentally correct.

### “Largest fragment is always the active compound.”

Often true for salts, but not always. Mixtures, prodrugs, solvates, metal complexes, and co-crystals require chemical judgement.

### “Uncharging is always good.”

Not always. Charge state may be crucial for binding, solubility, membrane permeability, or docking.

### “A standardized molecule is enough for reproducibility.”

No. The original input, settings, output SMILES, and logs should be preserved.
