from __future__ import annotations

from dataclasses import dataclass
import copy
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

from chem_inf_widgets.chemcore.mol import ChemMol
from chem_inf_widgets.chemcore.molecule_contract import (
    STANDARDIZATION_CHANGED,
    STANDARDIZATION_INPUT_SMILES,
    STANDARDIZATION_LOG,
    STANDARDIZATION_OUTPUT_SMILES,
    STANDARDIZATION_PROFILE,
    STANDARDIZATION_STATUS,
    STANDARDIZATION_VERSION_FIELD,
    STANDARDIZED_SMILES,
    append_qc_flag,
    append_transform_step,
    ensure_contract_props,
    set_dropped_reason,
)
from chem_inf_widgets.chemcore.services.rdkit_safe import safe_canonical_smiles, safe_mol_from_smiles


STANDARDIZATION_AUDIT_VERSION = "phase2.4"


@dataclass(frozen=True)
class StandardizeConfig:
    cleanup: bool = True
    normalize: bool = True
    metal_disconnect: bool = True
    largest_fragment: bool = True
    reionize: bool = True
    uncharge: bool = True

    sanitize_before: bool = True
    sanitize_after: bool = True

    canonical_smiles: bool = True


@dataclass(frozen=True)
class StandardizeResult:
    ok: bool
    input_smiles: str
    output_smiles: str
    log: str
    mol: Optional[Chem.Mol]


STANDARDIZATION_PRESETS: Dict[str, StandardizeConfig] = {
    # Phase 2.4 user-facing profiles
    "minimal": StandardizeConfig(
        cleanup=True,
        normalize=False,
        metal_disconnect=False,
        largest_fragment=False,
        reionize=False,
        uncharge=False,
    ),
    "qsar_ready": StandardizeConfig(
        cleanup=True,
        normalize=True,
        metal_disconnect=True,
        largest_fragment=True,
        reionize=True,
        uncharge=True,
    ),
    "chembl_like": StandardizeConfig(
        cleanup=True,
        normalize=True,
        metal_disconnect=True,
        largest_fragment=True,
        reionize=True,
        uncharge=True,
    ),
    "docking_ready": StandardizeConfig(
        cleanup=True,
        normalize=True,
        metal_disconnect=True,
        largest_fragment=True,
        reionize=True,
        uncharge=False,
    ),
    # Backward-compatible aliases used by older tests/workflows
    "drug_discovery_default": StandardizeConfig(
        cleanup=True,
        normalize=True,
        metal_disconnect=True,
        largest_fragment=True,
        reionize=True,
        uncharge=True,
    ),
    "preserve_salts": StandardizeConfig(
        cleanup=True,
        normalize=True,
        metal_disconnect=True,
        largest_fragment=False,
        reionize=True,
        uncharge=False,
    ),
    "docking_pose_safe": StandardizeConfig(
        cleanup=False,
        normalize=False,
        metal_disconnect=False,
        largest_fragment=False,
        reionize=False,
        uncharge=False,
        sanitize_before=False,
        sanitize_after=False,
    ),
    "fingerprint_canonical": StandardizeConfig(
        cleanup=True,
        normalize=True,
        metal_disconnect=True,
        largest_fragment=True,
        reionize=True,
        uncharge=False,
    ),
}

PROFILE_ALIASES: Dict[str, str] = {
    "Minimal": "minimal",
    "QSAR-ready": "qsar_ready",
    "ChEMBL-like": "chembl_like",
    "Docking-ready": "docking_ready",
    "Custom": "custom",
}

PROFILE_LABELS: Dict[str, str] = {
    "minimal": "Minimal",
    "qsar_ready": "QSAR-ready",
    "chembl_like": "ChEMBL-like",
    "docking_ready": "Docking-ready",
    "custom": "Custom",
}


def get_standardization_config(profile: Optional[str] = None) -> StandardizeConfig:
    """Return a named standardization profile. Unknown names fall back to the default profile."""
    if not profile:
        return STANDARDIZATION_PRESETS["qsar_ready"]
    return STANDARDIZATION_PRESETS.get(PROFILE_ALIASES.get(str(profile).strip(), str(profile).strip()), STANDARDIZATION_PRESETS["qsar_ready"])


class MolStandardizer:
    """
    Pure chemcore service.

    - Can standardize: SMILES -> RDKit Mol + SMILES
    - Can standardize: ChemMol -> ChemMol (preserving props + adding log fields)
    """

    def __init__(self, config: Optional[StandardizeConfig] = None, profile: Optional[str] = None) -> None:
        canonical_profile = PROFILE_ALIASES.get(str(profile or "").strip(), profile)
        self.profile = canonical_profile or ("custom" if config is not None else "qsar_ready")
        self.config = config or get_standardization_config(canonical_profile)

        # instantiate once (RDKit objects are light but avoid repeated construction)
        self._normalizer = rdMolStandardize.Normalizer()
        self._metal = rdMolStandardize.MetalDisconnector()
        self._largest = rdMolStandardize.LargestFragmentChooser()
        self._reionizer = rdMolStandardize.Reionizer()
        self._uncharger = rdMolStandardize.Uncharger()

    def standardize_smiles(self, smiles: str) -> StandardizeResult:
        in_smi = (smiles or "").strip()
        if not in_smi:
            return StandardizeResult(False, "", "", "Empty SMILES", None)

        log: List[str] = []
        parsed = safe_mol_from_smiles(in_smi, sanitize=False, remove_hs=False)
        mol = parsed.mol
        if mol is None:
            return StandardizeResult(False, in_smi, "", "Invalid SMILES", None)

        try:
            if self.config.sanitize_before:
                Chem.SanitizeMol(mol)
        except Exception as e:
            return StandardizeResult(False, in_smi, "", f"Sanitize (before) failed: {e}", None)

        try:
            mol, log = self._apply_ops(mol, log)
        except Exception as e:
            return StandardizeResult(False, in_smi, "", f"Standardization failed: {e}", None)

        if mol is None:
            return StandardizeResult(False, in_smi, "", "Standardization produced None", None)

        try:
            if self.config.sanitize_after:
                Chem.SanitizeMol(mol)
        except Exception as e:
            return StandardizeResult(False, in_smi, "", f"Sanitize (after) failed: {e}", None)

        out_smi = safe_canonical_smiles(
            mol,
            remove_hs=False,
            canonical=self.config.canonical_smiles,
            isomeric=True,
        )
        if not log:
            log_txt = "No changes"
        else:
            log_txt = "\n".join(log)

        return StandardizeResult(True, in_smi, out_smi, log_txt, mol)

    def standardize_chemmols(
        self,
        mols: Sequence[ChemMol],
        smiles_prop: str = "SMILES",
        out_smiles_prop: str = "SMILES_STD",
        out_log_prop: str = "STD_LOG",
        keep_original_smiles_prop: bool = True,
        overwrite_smiles_prop: bool = False,
    ) -> Tuple[List[ChemMol], List[StandardizeResult]]:
        """
        Standardize ChemMol list.

        - Reads SMILES from ChemMol props (default key = "SMILES"); if absent tries RDKit mol.
        - Writes:
            - out_smiles_prop (default "SMILES_STD")
            - out_log_prop (default "STD_LOG")
        - If overwrite_smiles_prop=True, also sets smiles_prop to standardized SMILES.
        - If keep_original_smiles_prop and overwrite, stores original as "SMILES_ORIG" (only if not already present).
        """
        out_mols: List[ChemMol] = []
        results: List[StandardizeResult] = []

        def _write_standardization_audit(target: ChemMol, input_smiles: str, res: StandardizeResult) -> ChemMol:
            target.set_prop(out_smiles_prop, res.output_smiles if res.ok else "")
            target.set_prop(out_log_prop, res.log)
            target.set_prop("STD_PROFILE", self.profile)
            target.set_prop(STANDARDIZATION_PROFILE, self.profile)
            target.set_prop("STD_OK", bool(res.ok))
            target.set_prop(STANDARDIZATION_STATUS, "ok" if res.ok else "failed")
            target.set_prop("STD_INPUT_SMILES", input_smiles)
            target.set_prop("STD_OUTPUT_SMILES", res.output_smiles if res.ok else "")
            target.set_prop(STANDARDIZATION_INPUT_SMILES, input_smiles)
            target.set_prop(STANDARDIZATION_OUTPUT_SMILES, res.output_smiles if res.ok else "")
            target.set_prop(STANDARDIZATION_LOG, res.log)
            target.set_prop(STANDARDIZATION_CHANGED, bool(res.ok and input_smiles and res.output_smiles and input_smiles != res.output_smiles))
            target.set_prop(STANDARDIZATION_VERSION_FIELD, STANDARDIZATION_AUDIT_VERSION)
            target.set_prop("STD_CHANGED", bool(res.ok and input_smiles and res.output_smiles and input_smiles != res.output_smiles))
            target.set_prop("STD_STEPS", res.log)
            target.set_prop(STANDARDIZED_SMILES, res.output_smiles if res.ok else "")
            if overwrite_smiles_prop:
                if keep_original_smiles_prop and not target.get_prop("SMILES_ORIG"):
                    target.set_prop("SMILES_ORIG", input_smiles)
                target.set_prop(smiles_prop, res.output_smiles if res.ok else input_smiles)
            audited = ensure_contract_props(target, input_smiles=input_smiles)
            append_transform_step(audited, f"standardize_{self.profile}")
            if not res.ok:
                append_qc_flag(audited, "standardization_failed")
                set_dropped_reason(audited, "standardization_failed")
            return audited

        for cm in mols:
            in_smi = ""
            try:
                v = cm.get_prop(smiles_prop)
                if v:
                    in_smi = str(v).strip()
            except Exception:
                in_smi = ""

            if not in_smi:
                # fallback: if ChemMol can convert to rdkit, derive SMILES
                rdm = cm.to_rdkit() if hasattr(cm, "to_rdkit") else None
                if rdm is not None:
                    try:
                        in_smi = safe_canonical_smiles(rdm, remove_hs=False, canonical=True, isomeric=True)
                    except Exception:
                        in_smi = ""

            res = self.standardize_smiles(in_smi)
            results.append(res)

            # build new ChemMol; never mutate the input object in-place if possible
            try:
                new_cm = cm.copy() if hasattr(cm, "copy") else copy.deepcopy(cm)
            except Exception:
                rdm = cm.to_rdkit() if hasattr(cm, "to_rdkit") else None
                new_cm = ChemMol.from_rdkit(rdm, name=getattr(cm, "name", None)) if rdm is not None else cm

            try:
                new_cm = _write_standardization_audit(new_cm, in_smi, res)
            except Exception:
                pass

            # update the structure if possible
            if res.ok and res.mol is not None:
                try:
                    # If ChemMol supports from_rdkit or set_structure, use it
                    if hasattr(ChemMol, "from_rdkit"):
                        new_cm = ChemMol.from_rdkit(res.mol, name=getattr(cm, "name", None) or None)
                        # restore props
                        for k, v in (cm.props or {}).items():
                            new_cm.set_prop(k, v)
                        new_cm = _write_standardization_audit(new_cm, in_smi, res)
                    elif hasattr(new_cm, "set_rdkit"):
                        new_cm.set_rdkit(res.mol)
                except Exception:
                    # keep as-is if structure update fails
                    pass

            out_mols.append(ensure_contract_props(new_cm, input_smiles=in_smi))

        return out_mols, results

    # ----------------- internals -----------------

    def _apply_ops(self, mol: Chem.Mol, log: List[str]) -> Tuple[Chem.Mol, List[str]]:
        mol = Chem.Mol(mol)  # defensive copy
        before = safe_canonical_smiles(mol, remove_hs=False, canonical=True, isomeric=True)

        if self.config.cleanup:
            mol2 = rdMolStandardize.Cleanup(mol)
            mol, log = self._log_change("Cleanup", mol, mol2, log)

        if self.config.normalize:
            mol2 = self._normalizer.normalize(mol)
            mol, log = self._log_change("Normalize", mol, mol2, log)

        if self.config.metal_disconnect:
            mol2 = self._metal.Disconnect(mol)
            mol, log = self._log_change("MetalDisconnector", mol, mol2, log)

        if self.config.largest_fragment:
            mol2 = self._largest.choose(mol)
            mol, log = self._log_change("LargestFragmentChooser", mol, mol2, log)

        if self.config.reionize:
            mol2 = self._reionizer.reionize(mol)
            mol, log = self._log_change("Reionizer", mol, mol2, log)

        if self.config.uncharge:
            mol2 = self._uncharger.uncharge(mol)
            mol, log = self._log_change("Uncharger", mol, mol2, log)

        after = safe_canonical_smiles(mol, remove_hs=False, canonical=True, isomeric=True)
        if before != after and not log:
            log.append(f"Changed: {before} → {after}")

        return mol, log

    @staticmethod
    def _log_change(name: str, mol_before: Chem.Mol, mol_after: Chem.Mol, log: List[str]) -> Tuple[Chem.Mol, List[str]]:
        b = safe_canonical_smiles(mol_before, remove_hs=False, canonical=True, isomeric=True)
        a = safe_canonical_smiles(mol_after, remove_hs=False, canonical=True, isomeric=True)
        if b != a:
            log.append(f"{name}: {b} → {a}")
        return mol_after, log
