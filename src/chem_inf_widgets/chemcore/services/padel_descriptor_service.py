from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
from shutil import which
import subprocess
import sys
from tempfile import TemporaryDirectory
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem

from chem_inf_widgets.chemcore.mol import ChemMol
from chem_inf_widgets.chemcore.services.rdkit_safe import (
    safe_canonical_smiles,
    safe_mol_from_smiles,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PadelPreset:
    key: str
    label: str
    description: str
    filename: Optional[str] = None
    calculate_2d: bool = True
    calculate_3d: bool = False
    fingerprints: bool = False

    def resolve_path(self) -> Optional[Path]:
        if not self.filename:
            return None
        return _PADEL_PRESET_DIR / self.filename


@dataclass(frozen=True)
class PadelComputeConfig:
    calculate_2d: bool = True
    calculate_3d: bool = False
    fingerprints: bool = False
    convert_3d: bool = False
    remove_salt: bool = False
    detect_aromaticity: bool = False
    standardize_nitro: bool = False
    standardize_tautomers: bool = False
    threads: int = -1          # -1 = PaDEL default / all cores
    timeout: int = 300         # subprocess timeout in seconds
    maxruntime: int = -1       # per-molecule timeout in seconds; -1 = unlimited
    descriptor_types_path: Optional[str] = None


_PADEL_PRESET_DIR = Path(__file__).resolve().parents[1] / "resources" / "padel_presets"

_PADEL_PRESETS: Tuple[PadelPreset, ...] = (
    PadelPreset(
        key="custom",
        label="Custom / no XML preset",
        description="Use the 2D/3D/fingerprint checkboxes and the full catalog exposed by PaDEL.",
    ),
    PadelPreset(
        key="all_2d_descriptors",
        label="All 2D descriptors",
        description="All PaDEL 2D descriptor families, no fingerprints.",
        filename="all_2d_descriptors.xml",
        calculate_2d=True,
        calculate_3d=False,
        fingerprints=False,
    ),
    PadelPreset(
        key="all_3d_descriptors",
        label="All 3D descriptors",
        description="All PaDEL 3D descriptor families only.",
        filename="all_3d_descriptors.xml",
        calculate_2d=False,
        calculate_3d=True,
        fingerprints=False,
    ),
    PadelPreset(
        key="constitutional_counts",
        label="Descriptor family: constitutional and counts",
        description="Counts, atom/bond composition, chains, rings, and basic constitutional properties.",
        filename="constitutional_counts.xml",
        calculate_2d=True,
        calculate_3d=False,
        fingerprints=False,
    ),
    PadelPreset(
        key="topology_connectivity",
        label="Descriptor family: topology and connectivity",
        description="Topological indices, paths, walks, matrices, and connectivity-derived descriptors.",
        filename="topology_connectivity.xml",
        calculate_2d=True,
        calculate_3d=False,
        fingerprints=False,
    ),
    PadelPreset(
        key="physchem_qsar",
        label="Descriptor family: physicochemical / QSAR",
        description="Lipophilicity, H-bond, polar surface, BCUT, and other common QSAR-ready descriptors.",
        filename="physchem_qsar.xml",
        calculate_2d=True,
        calculate_3d=False,
        fingerprints=False,
    ),
    PadelPreset(
        key="fragment_rings_paths",
        label="Descriptor family: fragments, rings, and paths",
        description="Fragment, SMARTS, chain/ring, and path-centric descriptor subset.",
        filename="fragment_rings_paths.xml",
        calculate_2d=True,
        calculate_3d=False,
        fingerprints=False,
    ),
    PadelPreset(
        key="all_fingerprints",
        label="All fingerprints",
        description="Enable all PaDEL fingerprint sets together.",
        filename="all_fingerprints.xml",
        calculate_2d=False,
        calculate_3d=False,
        fingerprints=True,
    ),
    PadelPreset(
        key="fp_cdk",
        label="Fingerprint: CDK",
        description="Standard CDK fingerprint (PaDEL Fingerprinter).",
        filename="fp_cdk.xml",
        calculate_2d=False,
        calculate_3d=False,
        fingerprints=True,
    ),
    PadelPreset(
        key="fp_extended",
        label="Fingerprint: Extended CDK",
        description="Extended CDK fingerprint.",
        filename="fp_extended.xml",
        calculate_2d=False,
        calculate_3d=False,
        fingerprints=True,
    ),
    PadelPreset(
        key="fp_estate",
        label="Fingerprint: EState",
        description="EState fingerprint set.",
        filename="fp_estate.xml",
        calculate_2d=False,
        calculate_3d=False,
        fingerprints=True,
    ),
    PadelPreset(
        key="fp_graph_only",
        label="Fingerprint: Graph-only CDK",
        description="Graph-only fingerprint, ignoring bond orders and aromatic flags.",
        filename="fp_graph_only.xml",
        calculate_2d=False,
        calculate_3d=False,
        fingerprints=True,
    ),
    PadelPreset(
        key="fp_maccs",
        label="Fingerprint: MACCS",
        description="MACCS structural keys.",
        filename="fp_maccs.xml",
        calculate_2d=False,
        calculate_3d=False,
        fingerprints=True,
    ),
    PadelPreset(
        key="fp_pubchem",
        label="Fingerprint: PubChem",
        description="PubChem fingerprint keys.",
        filename="fp_pubchem.xml",
        calculate_2d=False,
        calculate_3d=False,
        fingerprints=True,
    ),
    PadelPreset(
        key="fp_substructure",
        label="Fingerprint: Substructure",
        description="Presence/absence substructure keys.",
        filename="fp_substructure.xml",
        calculate_2d=False,
        calculate_3d=False,
        fingerprints=True,
    ),
    PadelPreset(
        key="fp_substructure_count",
        label="Fingerprint: Substructure count",
        description="Count-based version of the PaDEL substructure keys.",
        filename="fp_substructure_count.xml",
        calculate_2d=False,
        calculate_3d=False,
        fingerprints=True,
    ),
    PadelPreset(
        key="fp_klekota_roth",
        label="Fingerprint: Klekota–Roth",
        description="Klekota–Roth structural key set.",
        filename="fp_klekota_roth.xml",
        calculate_2d=False,
        calculate_3d=False,
        fingerprints=True,
    ),
    PadelPreset(
        key="fp_klekota_roth_count",
        label="Fingerprint: Klekota–Roth count",
        description="Count-based Klekota–Roth keys.",
        filename="fp_klekota_roth_count.xml",
        calculate_2d=False,
        calculate_3d=False,
        fingerprints=True,
    ),
    PadelPreset(
        key="fp_atom_pairs_2d",
        label="Fingerprint: Atom pairs 2D",
        description="Binary 2D atom-pair fingerprint.",
        filename="fp_atom_pairs_2d.xml",
        calculate_2d=False,
        calculate_3d=False,
        fingerprints=True,
    ),
    PadelPreset(
        key="fp_atom_pairs_2d_count",
        label="Fingerprint: Atom pairs 2D count",
        description="Count-based 2D atom-pair fingerprint.",
        filename="fp_atom_pairs_2d_count.xml",
        calculate_2d=False,
        calculate_3d=False,
        fingerprints=True,
    ),
)

_PRESET_MAP: Dict[str, PadelPreset] = {preset.key: preset for preset in _PADEL_PRESETS}


class PadelDescriptorService:
    """
    Chemcore service for computing PaDEL descriptors/fingerprints from SMILES.

    Design notes
    ------------
    - Uses ``padelpy`` so that the PaDEL jar is bundled with the Python package.
    - Accepts only SMILES as the execution input because this keeps row alignment
      simple for Orange ``Table`` inputs and ``ChemMol`` objects.
    - Listing available descriptor names is done by running a tiny probe job on a
      dummy molecule and caching the resulting CSV header for the active config.
    - XML preset support is exposed through ``descriptor_types_path`` so the widget
      can offer descriptor-family and fingerprint-set presets backed by PaDEL's
      native ``-descriptortypes`` option.
    """

    def __init__(self, cfg: Optional[PadelComputeConfig] = None) -> None:
        self.cfg = cfg or PadelComputeConfig()
        self._catalog_cache: Dict[PadelComputeConfig, List[str]] = {}

    @staticmethod
    def _resolve_java_executable() -> Optional[str]:
        candidate_roots = []
        java_home = os.environ.get("JAVA_HOME")
        if java_home:
            candidate_roots.append(Path(java_home))

        conda_prefix = os.environ.get("CONDA_PREFIX")
        if conda_prefix:
            candidate_roots.append(Path(conda_prefix))

        candidate_roots.append(Path(sys.prefix))

        for root in candidate_roots:
            for candidate in (root / "bin" / "java", root / "lib" / "jvm" / "bin" / "java"):
                if candidate.exists():
                    return str(candidate)

        return which("java")

    @staticmethod
    def dependency_status() -> Tuple[bool, str]:
        try:
            from padelpy.wrapper import padeldescriptor  # noqa: F401
        except Exception:
            return (
                False,
                "PaDELPy is not installed. Install 'padelpy' and a Java runtime (JRE/JDK).",
            )

        java_path = PadelDescriptorService._resolve_java_executable()
        if java_path is None:
            return False, "Java was not found on PATH. PaDEL requires a Java runtime."

        try:
            subprocess.run(
                [java_path, "-version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return (
                False,
                "Java is present on PATH but is not runnable. Install a working JRE/JDK "
                "and make sure the Orange Python environment can resolve it.",
            )

        return True, "OK"

    @staticmethod
    def list_presets() -> List[PadelPreset]:
        return list(_PADEL_PRESETS)

    @staticmethod
    def get_preset(key: str) -> PadelPreset:
        return _PRESET_MAP.get(key, _PRESET_MAP["custom"])

    @classmethod
    def config_from_preset(
        cls,
        preset_key: str,
        *,
        convert_3d: bool = False,
        remove_salt: bool = False,
        detect_aromaticity: bool = False,
        standardize_nitro: bool = False,
        standardize_tautomers: bool = False,
        threads: int = -1,
        timeout: int = 300,
        maxruntime: int = -1,
    ) -> PadelComputeConfig:
        preset = cls.get_preset(preset_key)
        path = preset.resolve_path()
        return PadelComputeConfig(
            calculate_2d=preset.calculate_2d,
            calculate_3d=preset.calculate_3d,
            fingerprints=preset.fingerprints,
            convert_3d=convert_3d,
            remove_salt=remove_salt,
            detect_aromaticity=detect_aromaticity,
            standardize_nitro=standardize_nitro,
            standardize_tautomers=standardize_tautomers,
            threads=threads,
            timeout=timeout,
            maxruntime=maxruntime,
            descriptor_types_path=str(path) if path is not None else None,
        )

    def list_descriptor_names(self, *, cfg: Optional[PadelComputeConfig] = None) -> List[str]:
        use_cfg = cfg or self.cfg
        if use_cfg in self._catalog_cache:
            return list(self._catalog_cache[use_cfg])

        df = self._run_padel_smiles(["CC"], use_cfg)
        names = [str(c) for c in df.columns if str(c).strip()]
        self._catalog_cache[use_cfg] = names
        return list(names)

    def compute(
        self,
        smiles: Sequence[str],
        selected_descriptor_names: Sequence[str],
        *,
        cfg: Optional[PadelComputeConfig] = None,
    ) -> pd.DataFrame:
        use_cfg = cfg or self.cfg
        if not smiles:
            return pd.DataFrame()

        valid_smiles: List[str] = []
        valid_idx: List[int] = []
        for idx, smi in enumerate(smiles):
            s = (smi or "").strip()
            if s:
                valid_smiles.append(s)
                valid_idx.append(idx)

        if not valid_smiles:
            return pd.DataFrame(index=range(len(smiles)))

        batch_df = self._run_padel_smiles(valid_smiles, use_cfg)
        if len(batch_df) != len(valid_smiles):
            batch_df = self._compute_one_by_one(valid_smiles, use_cfg)

        for col in batch_df.columns:
            batch_df[col] = pd.to_numeric(batch_df[col], errors="coerce")

        selected = [n for n in selected_descriptor_names if n in batch_df.columns]
        if selected:
            batch_df = batch_df[selected]

        return self.df_to_full_length(batch_df, valid_idx, len(smiles))

    @staticmethod
    def smiles_to_mols(smiles: Sequence[str]) -> Tuple[List[Optional[Chem.Mol]], List[int]]:
        mols_maybe: List[Optional[Chem.Mol]] = []
        valid_idx: List[int] = []
        for i, smi in enumerate(smiles):
            s = (smi or "").strip()
            mol = safe_mol_from_smiles(s, sanitize=True, remove_hs=True).mol if s else None
            mols_maybe.append(mol)
            if mol is not None:
                valid_idx.append(i)
        return mols_maybe, valid_idx

    @staticmethod
    def chemmols_to_smiles(molecules: Sequence[ChemMol]) -> List[str]:
        smiles: List[str] = []
        for chem_mol in molecules:
            smi = ""
            if isinstance(chem_mol, ChemMol) and getattr(chem_mol, "mol", None) is not None:
                try:
                    smi = safe_canonical_smiles(chem_mol.mol, remove_hs=False)
                except RuntimeError:
                    smi = ""
            if not smi and isinstance(chem_mol, ChemMol):
                smi = str(chem_mol.get_prop("SMILES") or "").strip()
            smiles.append(smi)
        return smiles

    @staticmethod
    def numeric_or_none(value: object) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (float, int, np.floating, np.integer)):
            try:
                if np.isnan(value):
                    return None
            except TypeError:
                pass
            return float(value)
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def df_to_full_length(df_valid: pd.DataFrame, valid_idx: List[int], n_total: int) -> pd.DataFrame:
        if df_valid.empty:
            return pd.DataFrame(index=range(n_total))
        df_valid = df_valid.copy()
        df_valid.index = valid_idx
        return df_valid.reindex(range(n_total))

    def _compute_one_by_one(self, smiles: Sequence[str], cfg: PadelComputeConfig) -> pd.DataFrame:
        rows: List[dict] = []
        all_cols: List[str] = []

        for idx, smi in enumerate(smiles):
            try:
                df_one = self._run_padel_smiles([smi], cfg)
                if len(df_one) == 0:
                    rows.append({})
                    continue
                row = df_one.iloc[0].to_dict()
                rows.append(row)
                for key in row.keys():
                    if key not in all_cols:
                        all_cols.append(key)
            except (
                OSError,
                RuntimeError,
                ValueError,
                subprocess.SubprocessError,
                pd.errors.EmptyDataError,
                pd.errors.ParserError,
            ):
                logger.debug("PaDEL failed for molecule at index %s during one-by-one fallback.", idx, exc_info=True)
                rows.append({})

        if not all_cols:
            return pd.DataFrame(index=range(len(smiles)))

        normalized_rows = [{c: row.get(c) for c in all_cols} for row in rows]
        return pd.DataFrame(normalized_rows, columns=all_cols)

    def _run_padel_smiles(self, smiles: Sequence[str], cfg: PadelComputeConfig) -> pd.DataFrame:
        ready, message = self.dependency_status()
        if not ready:
            raise RuntimeError(message)

        if not (cfg.calculate_2d or cfg.calculate_3d or cfg.fingerprints):
            raise RuntimeError("Enable at least one PaDEL output: 2D, 3D, or fingerprints.")

        descriptor_types_path: Optional[Path] = None
        if cfg.descriptor_types_path:
            descriptor_types_path = Path(cfg.descriptor_types_path)
            if not descriptor_types_path.exists():
                raise RuntimeError(f"PaDEL descriptor preset XML not found: {descriptor_types_path}")

        from padelpy.wrapper import padeldescriptor

        with TemporaryDirectory(prefix="padel_") as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            smi_path = tmpdir / "input.smi"
            out_path = tmpdir / "output.csv"

            smi_path.write_text("\n".join((s or "").strip() for s in smiles), encoding="utf-8")

            maxruntime_ms = -1 if cfg.maxruntime in (-1, 0) else int(cfg.maxruntime) * 1000
            timeout = None if cfg.timeout in (-1, 0) else int(cfg.timeout)
            threads = -1 if cfg.threads in (-1, 0) else int(cfg.threads)
            java_path = self._resolve_java_executable()
            java_bin_dir = str(Path(java_path).parent) if java_path else None

            original_path = os.environ.get("PATH", "")
            if java_bin_dir and java_bin_dir not in original_path.split(os.pathsep):
                os.environ["PATH"] = java_bin_dir + os.pathsep + original_path

            try:
                padeldescriptor(
                    mol_dir=str(smi_path),
                    d_file=str(out_path),
                    d_2d=bool(cfg.calculate_2d),
                    d_3d=bool(cfg.calculate_3d),
                    fingerprints=bool(cfg.fingerprints),
                    convert3d=bool(cfg.convert_3d),
                    descriptortypes=str(descriptor_types_path) if descriptor_types_path else None,
                    removesalt=bool(cfg.remove_salt),
                    detectaromaticity=bool(cfg.detect_aromaticity),
                    standardizenitro=bool(cfg.standardize_nitro),
                    standardizetautomers=bool(cfg.standardize_tautomers),
                    retainorder=True,
                    threads=threads,
                    maxruntime=maxruntime_ms,
                    sp_timeout=timeout,
                )
            finally:
                if java_bin_dir:
                    os.environ["PATH"] = original_path

            if not out_path.exists():
                raise RuntimeError("PaDEL did not produce an output CSV file.")

            df = pd.read_csv(out_path)
            if "Name" in df.columns:
                df = df.drop(columns=["Name"])
            return df
