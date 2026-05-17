from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

try:  # pragma: no cover - import availability is environment dependent
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
except Exception:  # pragma: no cover
    Chem = None  # type: ignore
    Descriptors = None  # type: ignore
    rdMolDescriptors = None  # type: ignore

from chem_inf_widgets.chemcore.mol import ChemMol
from chem_inf_widgets.chemcore.molecule_contract import (
    CANONICAL_SMILES,
    INCHIKEY,
    INPUT_SMILES,
    QC_DUPLICATE_COUNT,
    QC_DUPLICATE_KEY,
    QC_ISSUE_CODES,
    QC_ISSUES,
    QC_N_ISSUES,
    QC_SEVERITY,
    QC_STATUS,
    QC_VERSION_FIELD,
    append_qc_flag,
    append_transform_step,
    ensure_contract_props,
    set_dropped_reason,
)
from chem_inf_widgets.chemcore.services.rdkit_safe import safe_canonical_smiles, safe_mol_from_smiles


QC_VERSION = "0.1.0"

ORGANIC_ATOMIC_NUMBERS = {5, 6, 7, 8, 9, 15, 16, 17, 35, 53}
COMMON_METAL_ATOMIC_NUMBERS = {
    3, 4, 11, 12, 13, 19, 20, 21, 22, 23, 24, 25, 26,
    27, 28, 29, 30, 31, 37, 38, 39, 40, 41, 42, 43, 44,
    45, 46, 47, 48, 49, 50, 55, 56, 57, 58, 59, 60, 62,
    63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75,
    76, 77, 78, 79, 80, 81, 82, 83,
}


@dataclass(frozen=True)
class MoleculeQCConfig:
    """Configuration for molecule-level quality-control checks."""

    sanitize: bool = True
    duplicate_key: str = "canonical_smiles"  # canonical_smiles or inchikey
    max_mw: float = 900.0
    min_heavy_atoms: int = 3
    max_heavy_atoms: int = 90
    max_fragments: int = 1
    flag_metals: bool = True
    flag_isotopes: bool = True
    flag_radicals: bool = True
    flag_formal_charge: bool = True
    flag_missing_chiral_stereo: bool = True
    flag_missing_double_bond_stereo: bool = True


@dataclass(frozen=True)
class MoleculeQCRecord:
    row_index: int
    name: str
    input_smiles: str
    canonical_smiles: str
    inchikey: str
    ok_parse: bool
    status: str
    severity: str
    issues: List[str] = field(default_factory=list)
    issue_codes: List[str] = field(default_factory=list)
    n_issues: int = 0
    n_fragments: int = 0
    largest_fragment_atoms: int = 0
    heavy_atoms: int = 0
    molecular_weight: float = 0.0
    formal_charge: int = 0
    n_rings: int = 0
    n_hetero_atoms: int = 0
    has_metal: bool = False
    has_isotope: bool = False
    has_radical: bool = False
    possible_chiral_centers: int = 0
    unassigned_chiral_centers: int = 0
    possible_double_bond_stereo: int = 0
    unassigned_double_bond_stereo: int = 0
    duplicate_key: str = ""
    duplicate_count: int = 1
    parse_error: str = ""
    parse_warnings: str = ""

    @property
    def is_clean(self) -> bool:
        return self.ok_parse and self.severity == "OK"


@dataclass(frozen=True)
class MoleculeQCSummary:
    total: int
    clean: int
    problem: int
    invalid: int
    warnings: int
    errors: int
    duplicate_groups: int
    duplicate_records: int
    issue_counts: Dict[str, int]
    version: str = QC_VERSION


@dataclass(frozen=True)
class MoleculeQCResult:
    records: List[MoleculeQCRecord]
    clean_indices: List[int]
    problem_indices: List[int]
    summary: MoleculeQCSummary


def _inchikey(mol: Any) -> str:
    if Chem is None or mol is None:
        return ""
    try:
        return Chem.MolToInchiKey(Chem.Mol(mol)) or ""
    except Exception:
        return ""


def _name_from_chemmol(cm: ChemMol, fallback: str) -> str:
    if cm.name:
        return str(cm.name)
    for key in ("name", "Name", "NAME", "title", "Title", "compound_id", "molecule_id", "id"):
        value = cm.props.get(key)
        if value not in (None, ""):
            return str(value)
    return fallback


def _mol_from_any(obj: Any, row_index: int, config: MoleculeQCConfig) -> tuple[Optional[Any], str, str, str, List[str]]:
    """Return mol, name, input_smiles, parse_error, parse_warnings."""
    if isinstance(obj, ChemMol):
        mol = obj.mol
        ensure_contract_props(obj, row_index=row_index + 1)
        smi = obj.props.get(INPUT_SMILES) or obj.props.get("SMILES") or obj.props.get("smiles") or safe_canonical_smiles(mol)
        return mol, _name_from_chemmol(obj, f"mol_{row_index + 1}"), str(smi or ""), "", []

    if Chem is not None:
        try:
            if isinstance(obj, Chem.Mol):  # type: ignore[arg-type]
                return obj, f"mol_{row_index + 1}", safe_canonical_smiles(obj), "", []
        except Exception:
            pass

    smi = "" if obj is None else str(obj).strip()
    parsed = safe_mol_from_smiles(smi, sanitize=config.sanitize, remove_hs=True)
    return parsed.mol, f"mol_{row_index + 1}", smi, parsed.error or "", list(parsed.warnings or [])


def _fragment_stats(mol: Any) -> tuple[int, int]:
    if Chem is None or mol is None:
        return 0, 0
    try:
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        if not frags:
            return 0, 0
        return len(frags), max(int(f.GetNumHeavyAtoms()) for f in frags)
    except Exception:
        return 0, 0


def _double_bond_stereo_stats(mol: Any) -> tuple[int, int]:
    if Chem is None or mol is None:
        return 0, 0
    possible = 0
    unassigned = 0
    try:
        for bond in mol.GetBonds():
            if bond.GetBondType() != Chem.BondType.DOUBLE:
                continue
            begin = bond.GetBeginAtom()
            end = bond.GetEndAtom()
            # Terminal C=X double bonds usually do not carry E/Z stereo. This is a simple QC heuristic.
            if begin.GetDegree() < 2 or end.GetDegree() < 2:
                continue
            possible += 1
            if bond.GetStereo() in (Chem.BondStereo.STEREONONE, Chem.BondStereo.STEREOANY):
                unassigned += 1
    except Exception:
        return 0, 0
    return possible, unassigned


def _add_issue(issues: List[str], codes: List[str], code: str, text: str) -> None:
    codes.append(code)
    issues.append(text)


def _analyze_single(obj: Any, row_index: int, config: MoleculeQCConfig) -> MoleculeQCRecord:
    mol, name, input_smiles, parse_error, parse_warnings = _mol_from_any(obj, row_index, config)
    issues: List[str] = []
    codes: List[str] = []

    if mol is None:
        _add_issue(issues, codes, "INVALID_STRUCTURE", parse_error or "Invalid or missing molecular structure.")
        return MoleculeQCRecord(
            row_index=row_index,
            name=name,
            input_smiles=input_smiles,
            canonical_smiles="",
            inchikey="",
            ok_parse=False,
            status="Invalid",
            severity="ERROR",
            issues=issues,
            issue_codes=codes,
            n_issues=len(issues),
            parse_error=parse_error or "Invalid structure",
            parse_warnings=" | ".join(parse_warnings),
        )

    canonical = safe_canonical_smiles(mol, remove_hs=True)
    ikey = _inchikey(mol)
    n_fragments, largest_fragment_atoms = _fragment_stats(mol)

    try:
        heavy_atoms = int(mol.GetNumHeavyAtoms())
    except Exception:
        heavy_atoms = 0

    try:
        mw = float(Descriptors.MolWt(mol)) if Descriptors is not None else 0.0
    except Exception:
        mw = 0.0

    try:
        formal_charge = int(sum(atom.GetFormalCharge() for atom in mol.GetAtoms()))
    except Exception:
        formal_charge = 0

    try:
        n_rings = int(rdMolDescriptors.CalcNumRings(mol)) if rdMolDescriptors is not None else 0
    except Exception:
        n_rings = 0

    atoms = list(mol.GetAtoms())
    n_hetero = sum(1 for a in atoms if a.GetAtomicNum() not in (1, 6))
    has_metal = any(a.GetAtomicNum() in COMMON_METAL_ATOMIC_NUMBERS for a in atoms)
    has_isotope = any(a.GetIsotope() not in (0, None) for a in atoms)
    has_radical = any(a.GetNumRadicalElectrons() > 0 for a in atoms)

    try:
        chiral = Chem.FindMolChiralCenters(mol, includeUnassigned=True) if Chem is not None else []
    except Exception:
        chiral = []
    possible_chiral = len(chiral)
    unassigned_chiral = sum(1 for _, label in chiral if label == "?")

    possible_db, unassigned_db = _double_bond_stereo_stats(mol)

    if parse_warnings:
        _add_issue(issues, codes, "PARSE_WARNING", "RDKit parse/sanitize warning present.")
    if n_fragments > config.max_fragments:
        _add_issue(issues, codes, "MULTI_FRAGMENT", f"Molecule has {n_fragments} disconnected fragments.")
    if config.flag_metals and has_metal:
        _add_issue(issues, codes, "METAL_PRESENT", "Molecule contains metal/metalloid atoms that may require special handling.")
    if config.flag_isotopes and has_isotope:
        _add_issue(issues, codes, "ISOTOPE_PRESENT", "Molecule contains isotope labels.")
    if config.flag_radicals and has_radical:
        _add_issue(issues, codes, "RADICAL_PRESENT", "Molecule contains radical electrons.")
    if config.flag_formal_charge and formal_charge != 0:
        _add_issue(issues, codes, "NET_FORMAL_CHARGE", f"Net formal charge is {formal_charge}.")
    if heavy_atoms < config.min_heavy_atoms:
        _add_issue(issues, codes, "TOO_SMALL", f"Heavy atom count is below {config.min_heavy_atoms}.")
    if heavy_atoms > config.max_heavy_atoms:
        _add_issue(issues, codes, "TOO_LARGE", f"Heavy atom count is above {config.max_heavy_atoms}.")
    if mw > config.max_mw:
        _add_issue(issues, codes, "HIGH_MOLECULAR_WEIGHT", f"Molecular weight is above {config.max_mw:.1f} Da.")
    if config.flag_missing_chiral_stereo and unassigned_chiral > 0:
        _add_issue(issues, codes, "UNASSIGNED_CHIRAL_STEREO", f"{unassigned_chiral} possible chiral center(s) have unspecified configuration.")
    if config.flag_missing_double_bond_stereo and unassigned_db > 0:
        _add_issue(issues, codes, "UNASSIGNED_DOUBLE_BOND_STEREO", f"{unassigned_db} possible double bond stereo center(s) are unspecified.")

    severity = "OK" if not issues else "WARNING"
    return MoleculeQCRecord(
        row_index=row_index,
        name=name,
        input_smiles=input_smiles,
        canonical_smiles=canonical,
        inchikey=ikey,
        ok_parse=True,
        status="Clean" if not issues else "Needs review",
        severity=severity,
        issues=issues,
        issue_codes=codes,
        n_issues=len(issues),
        n_fragments=n_fragments,
        largest_fragment_atoms=largest_fragment_atoms,
        heavy_atoms=heavy_atoms,
        molecular_weight=mw,
        formal_charge=formal_charge,
        n_rings=n_rings,
        n_hetero_atoms=n_hetero,
        has_metal=has_metal,
        has_isotope=has_isotope,
        has_radical=has_radical,
        possible_chiral_centers=possible_chiral,
        unassigned_chiral_centers=unassigned_chiral,
        possible_double_bond_stereo=possible_db,
        unassigned_double_bond_stereo=unassigned_db,
        duplicate_key=canonical if config.duplicate_key == "canonical_smiles" else ikey,
        parse_error=parse_error,
        parse_warnings=" | ".join(parse_warnings),
    )


def run_molecule_qc(items: Sequence[Any], config: Optional[MoleculeQCConfig] = None) -> MoleculeQCResult:
    """Run molecule-level quality control on ChemMol/RDKit Mol/SMILES inputs."""
    if Chem is None:
        raise ImportError("RDKit is required for molecule quality control.")
    cfg = config or MoleculeQCConfig()

    records = [_analyze_single(obj, i, cfg) for i, obj in enumerate(items)]

    groups: Dict[str, List[int]] = defaultdict(list)
    for i, rec in enumerate(records):
        key = rec.duplicate_key or ""
        if rec.ok_parse and key:
            groups[key].append(i)

    mutable_records: List[MoleculeQCRecord] = []
    duplicate_record_indices = set()
    for i, rec in enumerate(records):
        group = groups.get(rec.duplicate_key or "", [])
        duplicate_count = len(group) if rec.ok_parse and rec.duplicate_key else 1
        if duplicate_count > 1:
            duplicate_record_indices.add(i)
            issues = list(rec.issues)
            codes = list(rec.issue_codes)
            if "DUPLICATE_STRUCTURE" not in codes:
                _add_issue(issues, codes, "DUPLICATE_STRUCTURE", f"Duplicate group contains {duplicate_count} records.")
            rec = MoleculeQCRecord(
                **{**rec.__dict__, "issues": issues, "issue_codes": codes, "n_issues": len(issues), "severity": "WARNING", "status": "Needs review", "duplicate_count": duplicate_count}
            )
        mutable_records.append(rec)
    records = mutable_records

    clean_indices = [i for i, rec in enumerate(records) if rec.is_clean]
    problem_indices = [i for i, rec in enumerate(records) if not rec.is_clean]

    issue_counts: Counter[str] = Counter()
    for rec in records:
        issue_counts.update(rec.issue_codes)

    duplicate_groups = sum(1 for members in groups.values() if len(members) > 1)
    summary = MoleculeQCSummary(
        total=len(records),
        clean=len(clean_indices),
        problem=len(problem_indices),
        invalid=sum(1 for rec in records if not rec.ok_parse),
        warnings=sum(1 for rec in records if rec.severity == "WARNING"),
        errors=sum(1 for rec in records if rec.severity == "ERROR"),
        duplicate_groups=duplicate_groups,
        duplicate_records=len(duplicate_record_indices),
        issue_counts=dict(sorted(issue_counts.items())),
    )
    return MoleculeQCResult(records=records, clean_indices=clean_indices, problem_indices=problem_indices, summary=summary)



def annotate_chemmols_with_qc(
    mols: Sequence[ChemMol],
    records: Sequence[MoleculeQCRecord],
    *,
    copy_molecules: bool = False,
) -> List[ChemMol]:
    """Attach QC contract fields to ChemMol.props and return the molecules.

    The default mutates the input ChemMol objects because Orange workflows pass
    these wrappers between widgets. Set ``copy_molecules=True`` when a defensive
    copy is needed in tests or standalone services. Records are matched by list
    position, which is the same order produced by ``run_molecule_qc``.
    """

    annotated: List[ChemMol] = []
    for i, cm in enumerate(mols):
        out = cm.copy() if copy_molecules else cm
        ensure_contract_props(out, row_index=i + 1)
        if i >= len(records):
            annotated.append(out)
            continue

        rec = records[i]
        props = out.props if isinstance(out.props, dict) else {}
        out.props = props
        props[QC_STATUS] = rec.status
        props[QC_SEVERITY] = rec.severity
        props[QC_ISSUE_CODES] = ";".join(rec.issue_codes)
        props[QC_ISSUES] = " | ".join(rec.issues)
        props[QC_N_ISSUES] = int(rec.n_issues)
        props[QC_DUPLICATE_KEY] = rec.duplicate_key
        props[QC_DUPLICATE_COUNT] = int(rec.duplicate_count)
        props[QC_VERSION_FIELD] = QC_VERSION
        if rec.canonical_smiles:
            props[CANONICAL_SMILES] = rec.canonical_smiles
            props.setdefault("SMILES", rec.canonical_smiles)
        if rec.inchikey:
            props[INCHIKEY] = rec.inchikey
        if rec.input_smiles:
            props.setdefault(INPUT_SMILES, rec.input_smiles)
        append_transform_step(out, "molecule_qc")
        for code in rec.issue_codes:
            append_qc_flag(out, str(code or "").strip().lower())
        if rec.severity == "ERROR" or not rec.ok_parse:
            reason = str(rec.issue_codes[0]).strip().lower() if rec.issue_codes else "qc_error"
            set_dropped_reason(out, reason)
        annotated.append(out)
    return annotated


def qc_partition_indices(result: MoleculeQCResult) -> Dict[str, List[int]]:
    """Return stable clean/problem/rejected index partitions.

    Clean means no QC issue, problem means warning-level issues that can be
    reviewed or filtered, and rejected means parse/error-level records.
    """

    clean: List[int] = []
    problem: List[int] = []
    rejected: List[int] = []
    for i, rec in enumerate(result.records):
        if rec.severity == "OK":
            clean.append(i)
        elif rec.severity == "ERROR" or not rec.ok_parse:
            rejected.append(i)
        else:
            problem.append(i)
    return {"clean": clean, "problem": problem, "rejected": rejected}

def qc_records_as_dicts(records: Iterable[MoleculeQCRecord]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for rec in records:
        qc_flags = [str(code or "").strip().lower() for code in rec.issue_codes if str(code or "").strip()]
        dropped_reason = qc_flags[0] if (rec.severity == "ERROR" or not rec.ok_parse) and qc_flags else ""
        rows.append(
            {
                "row_index": rec.row_index + 1,
                "name": rec.name,
                "input_smiles": rec.input_smiles,
                "canonical_smiles": rec.canonical_smiles,
                "inchikey": rec.inchikey,
                "status": rec.status,
                "severity": rec.severity,
                "n_issues": rec.n_issues,
                "issue_codes": ";".join(rec.issue_codes),
                "issues": " | ".join(rec.issues),
                "n_fragments": rec.n_fragments,
                "largest_fragment_atoms": rec.largest_fragment_atoms,
                "heavy_atoms": rec.heavy_atoms,
                "molecular_weight": rec.molecular_weight,
                "formal_charge": rec.formal_charge,
                "n_rings": rec.n_rings,
                "n_hetero_atoms": rec.n_hetero_atoms,
                "has_metal": int(rec.has_metal),
                "has_isotope": int(rec.has_isotope),
                "has_radical": int(rec.has_radical),
                "possible_chiral_centers": rec.possible_chiral_centers,
                "unassigned_chiral_centers": rec.unassigned_chiral_centers,
                "possible_double_bond_stereo": rec.possible_double_bond_stereo,
                "unassigned_double_bond_stereo": rec.unassigned_double_bond_stereo,
                "duplicate_key": rec.duplicate_key,
                "duplicate_count": rec.duplicate_count,
                "parse_error": rec.parse_error,
                "parse_warnings": rec.parse_warnings,
                "qc_flags": " | ".join(qc_flags),
                "dropped_reason": dropped_reason,
            }
        )
    return rows


def qc_summary_as_rows(summary: MoleculeQCSummary) -> List[Dict[str, Any]]:
    rows = [
        {"metric": "total_records", "value": summary.total, "description": "All input records."},
        {"metric": "clean_records", "value": summary.clean, "description": "Records without QC issues."},
        {"metric": "problem_records", "value": summary.problem, "description": "Records requiring review."},
        {"metric": "invalid_records", "value": summary.invalid, "description": "Records that could not be parsed."},
        {"metric": "warning_records", "value": summary.warnings, "description": "Records with warning-level issues."},
        {"metric": "error_records", "value": summary.errors, "description": "Records with error-level issues."},
        {"metric": "duplicate_groups", "value": summary.duplicate_groups, "description": "Duplicate groups by selected duplicate key."},
        {"metric": "duplicate_records", "value": summary.duplicate_records, "description": "Records belonging to duplicate groups."},
    ]
    for code, count in summary.issue_counts.items():
        rows.append({"metric": f"issue_{code}", "value": count, "description": f"Records/issues flagged as {code}."})
    return rows
