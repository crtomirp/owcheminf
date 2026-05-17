from __future__ import annotations

"""Cyclic/heterocycle registry fingerprint service.

This module implements a publication-oriented 4096-bit molecular fingerprint with
an interpretable registry-backed middle section.  It is intentionally independent
from the generic RDKit fingerprint service so it can be exposed as a dedicated
Orange widget and evolved with a versioned registry.
"""

from dataclasses import dataclass, field
from functools import lru_cache
from hashlib import blake2b
import json
from importlib import resources
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from chem_inf_widgets.chemcore.services.rdkit_safe import (
    safe_canonical_smiles,
    safe_mol_from_smiles,
)

try:  # pragma: no cover - import availability is environment-dependent
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
except Exception:  # pragma: no cover
    Chem = None  # type: ignore
    rdFingerprintGenerator = None  # type: ignore


FINGERPRINT_VERSION = "0.2.0"
DEFAULT_N_BITS = 4096

# Stable, documented bit layout.  These ranges are intentionally reserved even
# if a given dataset only activates a subset of them.
BIT_SECTIONS: Dict[str, Tuple[int, int]] = {
    "morgan": (0, 2048),
    "heterocycle_registry": (2048, 3072),
    "carbocycle_registry": (3072, 3328),
    "functional_group_registry": (3328, 3712),
    "ring_topology": (3712, 3840),
    "aromaticity_dehydro": (3840, 3968),
    "reserved": (3968, 4096),
}


@dataclass(frozen=True)
class RegistryEntry:
    """Normalized view of one cyclic registry entry."""

    entry_id: str
    name: str
    smarts: str
    group: str = "cyclic"
    family: str = ""
    superclass: str = ""
    ring_count: int = 0
    ring_atom_count: int = 0
    hetero_ring_atom_count: int = 0
    is_true_heterocycle: bool = False
    aromatic: Optional[bool] = None
    dehydro_variant: bool = False
    source: str = "cyclic_registry.json"
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def section(self) -> str:
        group_l = (self.group or "").lower()
        family_l = (self.family or "").lower()
        name_l = (self.name or "").lower()
        if "functional" in group_l:
            return "functional_group_registry"
        if self.dehydro_variant or "dehydro" in name_l or "dehydro" in family_l:
            return "aromaticity_dehydro"
        if self.is_true_heterocycle or int(self.hetero_ring_atom_count or 0) > 0:
            return "heterocycle_registry"
        if int(self.ring_count or 0) > 0 or "cyclic" in group_l or "carbocycle" in family_l:
            return "carbocycle_registry"
        return "functional_group_registry"


@dataclass(frozen=True)
class RegistryMatch:
    row: int
    bit: int
    entry_id: str
    name: str
    section: str
    family: str
    smarts: str
    match_count: int
    atom_matches: Tuple[Tuple[int, ...], ...] = ()


@dataclass(frozen=True)
class CyclicRegistryFingerprintResult:
    X: np.ndarray
    smiles: List[str]
    valid_indices: List[int]
    failed_indices: List[int]
    bit_names: List[str]
    errors: List[str]
    matches: List[RegistryMatch]
    registry_version: str
    fingerprint_version: str = FINGERPRINT_VERSION
    n_bits: int = DEFAULT_N_BITS
    section_ranges: Dict[str, Tuple[int, int]] = field(default_factory=lambda: dict(BIT_SECTIONS))
    params: Dict[str, Any] = field(default_factory=dict)


def _stable_hash_int(text: str) -> int:
    digest = blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def _bit_for_entry(entry: RegistryEntry) -> int:
    start, end = BIT_SECTIONS[entry.section]
    width = end - start
    key = f"{entry.section}|{entry.entry_id}|{entry.name}|{entry.smarts}"
    return start + (_stable_hash_int(key) % width)


def _topology_bit(name: str) -> int:
    start, end = BIT_SECTIONS["ring_topology"]
    return start + (_stable_hash_int(f"ring_topology|{name}") % (end - start))


def _reserved_bit(name: str) -> int:
    start, end = BIT_SECTIONS["reserved"]
    return start + (_stable_hash_int(f"reserved|{name}") % (end - start))


def _infer_aromatic(smarts: str, name: str = "") -> Optional[bool]:
    # SMARTS with lower-case aromatic atoms usually contains c/n/o/s/p.  This is
    # a heuristic only; exact matching is performed by RDKit.
    text = smarts or ""
    if any(ch in text for ch in ("c", "n", "o", "s", "p")):
        return True
    name_l = (name or "").lower()
    if "aromatic" in name_l:
        return True
    return None


def _normalize_record(record: Dict[str, Any], idx: int) -> Optional[RegistryEntry]:
    smarts = str(record.get("smarts") or record.get("smarts_aromatic") or record.get("smarts_kekule") or "").strip()
    if not smarts:
        return None
    entry_id = str(record.get("id") or record.get("pattern_id") or f"REG_{idx:06d}").strip()
    name = str(record.get("name") or entry_id).strip()
    group = str(record.get("group") or record.get("class") or "cyclic").strip()
    family = str(record.get("heterocycle_family") or record.get("family") or "").strip()
    superclass = str(record.get("heterocycle_superclass") or record.get("superclass") or "").strip()
    dehydro = bool(record.get("dehydro_variant") or ("dehydro" in name.lower()) or ("dehydro" in family.lower()))
    is_het = bool(record.get("is_true_heterocycle") or record.get("true_heterocycle") or False)
    try:
        hetero_ring_atoms = int(record.get("hetero_ring_atom_count") or 0)
    except Exception:
        hetero_ring_atoms = 0
    if hetero_ring_atoms > 0:
        is_het = True
    try:
        ring_count = int(record.get("ring_count") or 0)
    except Exception:
        ring_count = 0
    try:
        ring_atom_count = int(record.get("ring_atom_count") or 0)
    except Exception:
        ring_atom_count = 0
    return RegistryEntry(
        entry_id=entry_id,
        name=name,
        smarts=smarts,
        group=group,
        family=family,
        superclass=superclass,
        ring_count=ring_count,
        ring_atom_count=ring_atom_count,
        hetero_ring_atom_count=hetero_ring_atoms,
        is_true_heterocycle=is_het,
        aromatic=record.get("aromatic") if isinstance(record.get("aromatic"), bool) else _infer_aromatic(smarts, name),
        dehydro_variant=dehydro,
        source=str(record.get("source_csv") or record.get("source") or "cyclic_registry.json"),
        raw=dict(record),
    )


def _load_registry_json() -> Dict[str, Any]:
    with resources.files("chem_inf_widgets.chemcore.data").joinpath("cyclic_registry.json").open("r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def load_registry_entries() -> Tuple[str, Tuple[RegistryEntry, ...]]:
    """Load and normalize the packaged cyclic registry.

    Supports both the current legacy shape (``records``) and the proposed v2
    shape (``entries``).  Invalid/empty SMARTS are skipped at this stage; SMARTS
    compilation is validated lazily so the widget can report bad entries rather
    than fail during import.
    """
    data = _load_registry_json()
    version = str(data.get("registry_version") or data.get("version") or data.get("title") or "legacy-cyclic-registry")
    raw_entries = data.get("entries") or data.get("records") or []
    entries: List[RegistryEntry] = []
    for i, rec in enumerate(raw_entries):
        if not isinstance(rec, dict):
            continue
        e = _normalize_record(rec, i)
        if e is not None:
            entries.append(e)
    return version, tuple(entries)


@lru_cache(maxsize=8192)
def _compiled_smarts(smarts: str):
    if Chem is None:
        raise ImportError("RDKit is required for cyclic registry fingerprints.")
    return Chem.MolFromSmarts(smarts)


def validate_registry_entries(entries: Optional[Sequence[RegistryEntry]] = None, *, limit: Optional[int] = None) -> List[str]:
    """Return human-readable SMARTS validation errors for registry entries."""
    if entries is None:
        _, loaded = load_registry_entries()
        entries = loaded
    errors: List[str] = []
    for i, entry in enumerate(entries):
        if limit is not None and i >= int(limit):
            break
        try:
            patt = _compiled_smarts(entry.smarts)
        except Exception as exc:
            errors.append(f"{entry.entry_id}: SMARTS exception: {exc}")
            continue
        if patt is None:
            errors.append(f"{entry.entry_id}: invalid SMARTS: {entry.smarts}")
    return errors


def _bit_names(entries: Sequence[RegistryEntry], *, include_morgan: bool = True) -> List[str]:
    names = [f"CRFP_{i:04d}" for i in range(DEFAULT_N_BITS)]
    if include_morgan:
        for i in range(*BIT_SECTIONS["morgan"]):
            names[i] = f"CRFP_morgan_{i:04d}"
    labels_by_bit: Dict[int, List[str]] = {}
    for entry in entries:
        b = _bit_for_entry(entry)
        label = f"CRFP_{entry.section}_{b}_{entry.entry_id}"
        labels_by_bit.setdefault(b, []).append(label)
    for bit, labels in labels_by_bit.items():
        if len(labels) == 1:
            names[bit] = labels[0]
            continue
        preview = "__".join(label.rsplit("_", 1)[-1] for label in labels[:3])
        if len(labels) > 3:
            preview = f"{preview}__plus{len(labels) - 3}more"
        names[bit] = f"{labels[0]}__collision{len(labels)}__{preview}"
    for name in [
        "ring_count_ge_1", "ring_count_ge_2", "ring_count_ge_3", "ring_count_ge_4",
        "hetero_ring_count_ge_1", "hetero_ring_count_ge_2",
        "aromatic_ring_count_ge_1", "aromatic_ring_count_ge_2",
        "fused_ring_system", "macrocycle_ge_8", "spiro_candidate",
    ]:
        b = _topology_bit(name)
        names[b] = f"CRFP_topology_{name}"
    return names


def _set_morgan_bits(mol: Any, row: np.ndarray) -> None:
    if rdFingerprintGenerator is None:
        return
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fp = gen.GetFingerprint(mol)
    for bit in fp.GetOnBits():
        if 0 <= int(bit) < 2048:
            row[int(bit)] = 1


def _set_topology_bits(mol: Any, row: np.ndarray) -> None:
    if Chem is None:
        return
    try:
        ring_info = mol.GetRingInfo()
        atom_rings = list(ring_info.AtomRings())
    except Exception:
        atom_rings = []
    ring_count = len(atom_rings)
    aromatic_count = 0
    hetero_count = 0
    macrocycle = False
    for ring in atom_rings:
        atoms = [mol.GetAtomWithIdx(int(i)) for i in ring]
        if all(a.GetIsAromatic() for a in atoms):
            aromatic_count += 1
        if any(a.GetAtomicNum() not in (6, 1) for a in atoms):
            hetero_count += 1
        if len(ring) >= 8:
            macrocycle = True
    thresholds = {
        "ring_count_ge_1": ring_count >= 1,
        "ring_count_ge_2": ring_count >= 2,
        "ring_count_ge_3": ring_count >= 3,
        "ring_count_ge_4": ring_count >= 4,
        "hetero_ring_count_ge_1": hetero_count >= 1,
        "hetero_ring_count_ge_2": hetero_count >= 2,
        "aromatic_ring_count_ge_1": aromatic_count >= 1,
        "aromatic_ring_count_ge_2": aromatic_count >= 2,
        "fused_ring_system": ring_count >= 2 and any(set(a) & set(b) for i, a in enumerate(atom_rings) for b in atom_rings[i + 1:]),
        "macrocycle_ge_8": macrocycle,
        # This is intentionally conservative; detailed spiro detection can be refined later.
        "spiro_candidate": ring_count >= 2 and any(len(set(a) & set(b)) == 1 for i, a in enumerate(atom_rings) for b in atom_rings[i + 1:]),
    }
    for name, ok in thresholds.items():
        if ok:
            row[_topology_bit(name)] = 1


def compute_cyclic_registry_fingerprints_from_smiles(
    smiles: Sequence[str],
    *,
    include_morgan: bool = True,
    sanitize: bool = True,
    max_registry_entries: Optional[int] = None,
    include_atom_matches: bool = True,
    progress_cb: Optional[Callable[[int], None]] = None,
    cancel_cb: Optional[Callable[[], bool]] = None,
) -> CyclicRegistryFingerprintResult:
    """Compute the 4096-bit cyclic registry fingerprint for SMILES strings."""
    if Chem is None:
        raise ImportError("RDKit is required for cyclic registry fingerprints.")

    registry_version, entries_all = load_registry_entries()
    entries = list(entries_all)
    if max_registry_entries is not None and int(max_registry_entries) > 0:
        entries = entries[: int(max_registry_entries)]

    X_rows: List[np.ndarray] = []
    valid_indices: List[int] = []
    failed_indices: List[int] = []
    errors: List[str] = []
    all_matches: List[RegistryMatch] = []
    smiles_out: List[str] = []

    total = max(1, len(smiles))
    for row_i, smi in enumerate(smiles):
        if cancel_cb is not None and cancel_cb():
            break
        smi_str = (smi or "").strip() if isinstance(smi, str) else str(smi or "").strip()
        if not smi_str:
            failed_indices.append(row_i)
            errors.append(f"Row {row_i + 1}: empty SMILES")
            if progress_cb:
                progress_cb(int((row_i + 1) * 100 / total))
            continue
        parse_result = safe_mol_from_smiles(smi_str, sanitize=bool(sanitize), remove_hs=False)
        mol = parse_result.mol
        if mol is None:
            failed_indices.append(row_i)
            errors.append(parse_result.error or f"Row {row_i + 1}: invalid SMILES")
            if progress_cb:
                progress_cb(int((row_i + 1) * 100 / total))
            continue
        canon = safe_canonical_smiles(mol, remove_hs=False, canonical=True, isomeric=True) or smi_str

        bits = np.zeros(DEFAULT_N_BITS, dtype=np.uint8)
        if include_morgan:
            try:
                _set_morgan_bits(mol, bits)
            except Exception as exc:
                errors.append(f"Row {row_i + 1}: Morgan section failed: {exc}")
        try:
            _set_topology_bits(mol, bits)
        except Exception as exc:
            errors.append(f"Row {row_i + 1}: topology section failed: {exc}")

        out_row_idx = len(X_rows)
        for entry in entries:
            try:
                patt = _compiled_smarts(entry.smarts)
            except Exception as exc:
                # Registry errors are recorded but do not invalidate the molecule.
                errors.append(f"Registry {entry.entry_id}: SMARTS exception: {exc}")
                continue
            if patt is None:
                continue
            try:
                atom_matches = mol.GetSubstructMatches(patt, uniquify=True)
            except Exception as exc:
                errors.append(f"Row {row_i + 1}, registry {entry.entry_id}: match failed: {exc}")
                continue
            if not atom_matches:
                continue
            bit = _bit_for_entry(entry)
            bits[bit] = 1
            all_matches.append(
                RegistryMatch(
                    row=out_row_idx,
                    bit=bit,
                    entry_id=entry.entry_id,
                    name=entry.name,
                    section=entry.section,
                    family=entry.family,
                    smarts=entry.smarts,
                    match_count=len(atom_matches),
                    atom_matches=tuple(tuple(int(a) for a in m) for m in atom_matches) if include_atom_matches else (),
                )
            )

        X_rows.append(bits)
        valid_indices.append(row_i)
        smiles_out.append(canon)
        if progress_cb:
            progress_cb(int((row_i + 1) * 100 / total))

    X = np.vstack(X_rows).astype(np.uint8, copy=False) if X_rows else np.zeros((0, DEFAULT_N_BITS), dtype=np.uint8)
    return CyclicRegistryFingerprintResult(
        X=X,
        smiles=smiles_out,
        valid_indices=valid_indices,
        failed_indices=failed_indices,
        bit_names=_bit_names(entries, include_morgan=include_morgan),
        errors=errors,
        matches=all_matches,
        registry_version=registry_version,
        n_bits=DEFAULT_N_BITS,
        params={
            "include_morgan": bool(include_morgan),
            "sanitize": bool(sanitize),
            "max_registry_entries": max_registry_entries,
            "include_atom_matches": bool(include_atom_matches),
            "bit_sections": dict(BIT_SECTIONS),
        },
    )
