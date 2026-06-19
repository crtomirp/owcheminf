# CmDock Docking

## Status

Default-palette widget under the **Cheminf - Docking** category. Requires an
external CmDock install.

Source:
- [ow_cmdock_docking.py](../../src/chem_inf_widgets/widgets/ow_cmdock_docking.py)
- [cmdock_service.py](../../src/chem_inf_widgets/chemcore/services/cmdock_service.py)

## Purpose

`CmDock Docking` docks ligands against a prepared CmDock receptor and returns one
row per docked pose. Every CmDock SDF property is expanded into its own column, so
the output table is ready for ranking, filtering, and plotting.

CmDock must be installed separately: https://gitlab.com/Jukic/cmdock. The widget
defaults to `/opt/CmD/bin/cmdock` and `/opt/CmD/lib` when present.

## Inputs

- `Molecules` — a list of 3D `ChemMol` objects to dock (e.g. the **Molecules**
  output of `SDF Reader`). This is the primary input; each molecule's existing 3D
  conformer is sent to CmDock.
- `Data` — an optional `Table`, one row per molecule (same order), whose columns
  are attached to every output pose row. When connected together with
  `Molecules` (as `SDF Reader` provides both), it supplies the per-ligand data.

If no `Molecules` are connected, a `Data` table containing an **SDF text** or
**SDF file-path** column can be docked on its own (pick the column and input mode
in the controls).

Ligands must already be 3D — CmDock does not generate conformers.

## Outputs

- `Docked Poses` — one row per pose:
  - **attributes** (numeric): `pose_index`, `rank_by_score`,
    `ranking_score (<field>)` (a copy of the chosen **Ranking score field**, e.g.
    `ranking_score (SCORE.INTER)`, used for sorting and `rank_by_score`),
    `cmdock_return_code`, and every numeric CmDock score property (`SCORE`,
    `SCORE.INTER`, `SCORE.INTER.VDW`, `SCORE.INTRA`, `SCORE.RESTR`, `SCORE.norm`, …).
  - **metas** (text): `ligand_name`, `status`, `pose_sdf` (full pose mol block),
    `error_message`, `properties_json`, the CmDock command and stdout/stderr, plus
    any text SDF properties.
  - the per-ligand `Data`/molecule properties are copied onto every pose row.
- `Molecules` — the docked poses as `ChemMol` objects: each input molecule with its
  conformation **replaced by the docked pose geometry**, carrying the scores and
  per-ligand data as `props`. Feed it to `Mol Viewer`, `SDF Writer`, etc.

Poses are ranked ascending by the selected score field (most negative first).

## Main controls

- **Receptor .prm** — prepared CmDock receptor parameter file. Its matching `.as`
  cavity file must already exist next to it (build it with `cmcavity -r rec.prm -W`).
- **cmdock executable** / **Library directory** — CmDock binary and (optionally)
  its shared-library directory.
- **SDF column** / **Input mode** — which column holds the ligands, and how to read
  it (`Auto`, `SDF text`, `SDF file path`).
- **Protocol (-p)** — CmDock protocol, e.g. `dock.prm`.
- **Runs (-n)** / **Best poses (-b)** — docking runs per ligand and poses retained.
- **Ranking score field** — SDF property used to rank poses (default `SCORE.INTER`).

### Advanced

- Extra `cmdock` flags, keep temporary files, fail on first ligand error, and
  optional GNU Parallel execution with a job count.

## Receptor preparation

A CmDock receptor is **three files that must sit together in one folder**:

| File | What it is |
| --- | --- |
| `rec.prm` | receptor parameter file (you point the widget at this) |
| `rec.as` | the docking **cavity**, generated from the `.prm` |
| the receptor structure | the `.mol2`/`.pdb` named by `RECEPTOR_FILE` inside the `.prm` |

Two requirements trip people up most often:

1. **Build the cavity** once per receptor:

   ```bash
   cmcavity -r rec.prm -W      # writes rec.as next to rec.prm
   ```

   The `.as` file is named after the `.prm` (`rec.prm` → `rec.as`).

2. **`RECEPTOR_FILE` must match the real file name.** Open the `.prm` and check:

   ```text
   RECEPTOR_FILE target.mol2
   ```

   A file with that exact name must exist next to the `.prm`. If your structure is
   called something else (e.g. `my_receptor.mol2`), either copy/rename it to the
   name in `RECEPTOR_FILE`, or edit `RECEPTOR_FILE` to the real name.

**Avoid spaces and shared names.** If many receptors live in one folder and every
`.prm` says `RECEPTOR_FILE target.mol2`, only one `target.mol2` can be present at a
time. Give each receptor its own folder with simple names, for example:

```bash
mkdir -p ~/cmdock/my_target && cd ~/cmdock/my_target
cp /path/to/receptor.prm     rec.prm     # RECEPTOR_FILE inside should read: target.mol2
cp /path/to/receptor.mol2    target.mol2
cmcavity -r rec.prm -W                    # builds rec.as
```

Then point the widget's **Receptor .prm** at `~/cmdock/my_target/rec.prm`.

Before each run the widget preflights the receptor and stops with a clear message
if the `.prm` is missing, the `.as` cavity is missing, or `RECEPTOR_FILE` points at
a file that is not present.

## Typical workflow

1. Prepare the receptor folder (`.prm` + `.as` cavity + the `RECEPTOR_FILE`
   structure) as above.
2. Load 3D ligands (`SDF Reader` → connect both **Molecules** and **Data**).
3. Configure the receptor and executable, then press **Dock**.
4. Inspect / rank poses in a `Data Table`; keep the best pose per ligand with
   `Select Rows` (`rank_by_score = 1`); send **Molecules** to `Mol Viewer` to see
   the docked poses.

## Notes

- Failed ligands produce diagnostic rows (`status = failed`) instead of aborting,
  unless **Fail on first ligand error** is set.
- `pose_sdf` is plain SDF text; pass it to a writer/viewer to render structures.
- The **Log** (right pane) shows the exact `cmdock` command and, on failure,
  CmDock's own stdout/stderr — the quickest way to see what went wrong.

## Troubleshooting

| Symptom (in the Log / `error_message`) | Cause | Fix |
| --- | --- | --- |
| `Cavity file (rec.as) not found … run cmcavity first` | The `.as` cavity was never built. | `cmcavity -r rec.prm -W` |
| `Error opening …/target.mol2` | `RECEPTOR_FILE` in the `.prm` names a file that is not next to it. | Put that exact file there, or fix the `RECEPTOR_FILE` line. |
| `CmDock finished but did not create an output SDF file` | CmDock failed internally — read the CmDock output shown below it in the Log. | Usually one of the two above. |
| Poses produced but geometry looks wrong / 2D | Ligands were not 3D. | Provide 3D ligands; CmDock does not generate conformers. |
| `CmDock executable not found` | The `cmdock` path/PATH is wrong. | Set **cmdock executable**; on a default install it is `/opt/CmD/bin/cmdock`. |

If CmDock reports a missing shared library, set **Library directory** (e.g.
`/opt/CmD/lib`).
