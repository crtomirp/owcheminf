from __future__ import annotations

"""Shared curation workflow helpers used by import, QC and standardization widgets.

The goal of this module is deliberately modest: it gives all early-stage
molecule-curation widgets the same small vocabulary for workflow status and a
compact summary table that can be inspected in Orange.  It does not replace the
more detailed per-widget reports.
"""

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from typing import TYPE_CHECKING

from chem_inf_widgets.chemcore.mol import ChemMol
from chem_inf_widgets.chemcore.molecule_contract import append_transform_step, ensure_contract_props, set_dropped_reason

if TYPE_CHECKING:  # pragma: no cover
    from Orange.data import Table

CURATION_STAGE = "curation_stage"
CURATION_STATUS = "curation_status"
CURATION_READY_FOR_QSAR = "curation_ready_for_qsar"
CURATION_READY_FOR_DOCKING = "curation_ready_for_docking"
CURATION_BLOCKERS = "curation_blockers"
CURATION_WARNINGS = "curation_warnings"
CURATION_RECOMMENDED_NEXT_STEP = "curation_recommended_next_step"
CURATION_VERSION_FIELD = "curation_version"
CURATION_VERSION = "phase2.5"

QSAR_COMPATIBLE_STANDARDIZATION_PROFILES = {"qsar_ready", "chembl_like"}
DOCKING_COMPATIBLE_STANDARDIZATION_PROFILES = {"docking_ready"}


@dataclass(frozen=True)
class CurationSummary:
    stage: str
    status: str
    total_records: int = 0
    accepted_records: int = 0
    clean_records: int = 0
    problem_records: int = 0
    rejected_records: int = 0
    standardized_records: int = 0
    failed_records: int = 0
    qsar_ready_records: int = 0
    docking_ready_records: int = 0
    warnings: str = ""
    blockers: str = ""
    recommended_next_step: str = ""


def annotate_curation_props(
    mols: Sequence[ChemMol],
    *,
    stage: str,
    status: str,
    ready_for_qsar: bool = False,
    ready_for_docking: bool = False,
    blockers: str = "",
    warnings: str = "",
    recommended_next_step: str = "",
) -> list[ChemMol]:
    """Annotate molecules with shared curation contract fields."""
    out: list[ChemMol] = []
    for i, cm in enumerate(mols, start=1):
        try:
            ensure_contract_props(cm, row_index=i)
            cm.set_prop(CURATION_STAGE, stage)
            cm.set_prop(CURATION_STATUS, status)
            cm.set_prop(CURATION_READY_FOR_QSAR, bool(ready_for_qsar))
            cm.set_prop(CURATION_READY_FOR_DOCKING, bool(ready_for_docking))
            cm.set_prop(CURATION_BLOCKERS, blockers)
            cm.set_prop(CURATION_WARNINGS, warnings)
            cm.set_prop(CURATION_RECOMMENDED_NEXT_STEP, recommended_next_step)
            cm.set_prop(CURATION_VERSION_FIELD, CURATION_VERSION)
            append_transform_step(cm, f"curation_{stage}")
            if str(status or "").strip().lower() == "blocked" and str(blockers or "").strip():
                set_dropped_reason(cm, blockers)
        except Exception:
            # Curation annotation must never break a chemistry workflow.
            pass
        out.append(cm)
    return out


def curation_summary_to_rows(summary: CurationSummary) -> list[dict[str, Any]]:
    """Return a compact metric table representation of a curation summary."""
    return [
        {"metric": "stage", "value": summary.stage, "description": "Current curation workflow stage"},
        {"metric": "status", "value": summary.status, "description": "Overall status at this stage"},
        {"metric": "total_records", "value": summary.total_records, "description": "Input records considered"},
        {"metric": "accepted_records", "value": summary.accepted_records, "description": "Records that passed the current accept/reject gate"},
        {"metric": "clean_records", "value": summary.clean_records, "description": "Records with no QC issues"},
        {"metric": "problem_records", "value": summary.problem_records, "description": "Records with warning-level issues"},
        {"metric": "rejected_records", "value": summary.rejected_records, "description": "Records rejected or blocked"},
        {"metric": "standardized_records", "value": summary.standardized_records, "description": "Records standardized successfully"},
        {"metric": "failed_records", "value": summary.failed_records, "description": "Records that failed at this stage"},
        {"metric": "qsar_ready_records", "value": summary.qsar_ready_records, "description": "Records ready for QSAR descriptor/model widgets"},
        {"metric": "docking_ready_records", "value": summary.docking_ready_records, "description": "Records ready for docking-oriented workflows"},
        {"metric": "warnings", "value": summary.warnings, "description": "Workflow-level warnings"},
        {"metric": "blockers", "value": summary.blockers, "description": "Workflow-level blockers"},
        {"metric": "recommended_next_step", "value": summary.recommended_next_step, "description": "Recommended next widget/stage"},
        {"metric": "curation_version", "value": CURATION_VERSION, "description": "Curation summary schema version"},
    ]


def curation_summary_to_table(summary: CurationSummary) -> "Table | None":
    """Build an Orange table from a curation summary.

    Returns None when Orange is not installed; this keeps chemcore usable in
    command-line tests/environments without Orange while widgets still receive a
    proper table inside Orange.
    """
    try:
        from chem_inf_widgets.chemcore.services.report_table_utils import summary_rows_to_table
    except ImportError:  # pragma: no cover - Orange-free environment
        return None

    return summary_rows_to_table(
        curation_summary_to_rows(summary),
        name="Curation Summary",
    )


def summary_from_import(import_summary: Any) -> CurationSummary:
    total = int(getattr(import_summary, "total_records", 0) or 0)
    accepted = int(getattr(import_summary, "accepted_records", 0) or 0)
    rejected = int(getattr(import_summary, "rejected_records", 0) or 0)
    failed = int(getattr(import_summary, "failed_records", 0) or 0)
    duplicate_records = int(getattr(import_summary, "duplicate_records", 0) or 0)
    warnings = ""
    if duplicate_records:
        warnings = f"{duplicate_records} duplicate record(s) detected"
    status = "accepted" if accepted and not rejected and not failed else ("needs_review" if accepted else "blocked")
    return CurationSummary(
        stage="import",
        status=status,
        total_records=total,
        accepted_records=accepted,
        rejected_records=rejected,
        failed_records=failed,
        warnings=warnings,
        blockers="" if accepted else "No valid accepted molecules",
        recommended_next_step="Molecule QC Dashboard" if accepted else "Fix input file or SMILES column",
    )


def summary_from_qc(qc_summary: Any) -> CurationSummary:
    total = int(getattr(qc_summary, "total", 0) or 0)
    clean = int(getattr(qc_summary, "clean", 0) or 0)
    invalid = int(getattr(qc_summary, "invalid", 0) or 0)
    errors = int(getattr(qc_summary, "errors", 0) or 0)
    warnings_count = int(getattr(qc_summary, "warnings", 0) or 0)
    problem = max(0, total - clean - invalid - errors)
    rejected = invalid + errors
    status = "clean" if clean == total and total else ("needs_review" if clean or problem else "blocked")
    blockers = "" if clean else "No clean molecules available for QSAR-ready standardization"
    warning_text = f"{warnings_count} warning-level issue(s)" if warnings_count else ""
    return CurationSummary(
        stage="qc",
        status=status,
        total_records=total,
        accepted_records=clean + problem,
        clean_records=clean,
        problem_records=problem,
        rejected_records=rejected,
        warnings=warning_text,
        blockers=blockers,
        recommended_next_step="Mol Standardizer with QSAR-ready profile" if clean else "Inspect Problem/Rejected outputs",
    )


def summary_from_standardization_rows(rows: Iterable[Mapping[str, Any]], profile: str) -> CurationSummary:
    row_list = list(rows)
    total = len(row_list)
    ok = sum(1 for r in row_list if bool(r.get("ok")))
    failed = total - ok
    profile_key = str(profile or "").strip()
    qsar_ready = ok if profile_key in QSAR_COMPATIBLE_STANDARDIZATION_PROFILES else 0
    docking_ready = ok if profile_key in DOCKING_COMPATIBLE_STANDARDIZATION_PROFILES else 0
    if qsar_ready:
        status = "qsar_ready"
        next_step = "Descriptors / Fingerprints → QSAR Studio"
    elif docking_ready:
        status = "docking_ready"
        next_step = "Docking/pose workflow"
    elif ok:
        status = "standardized"
        next_step = "Choose QSAR-ready profile for QSAR workflows"
    else:
        status = "blocked"
        next_step = "Inspect Standardization Report and failed molecules"
    return CurationSummary(
        stage="standardization",
        status=status,
        total_records=total,
        accepted_records=ok,
        standardized_records=ok,
        failed_records=failed,
        rejected_records=failed,
        qsar_ready_records=qsar_ready,
        docking_ready_records=docking_ready,
        blockers="" if ok else "No molecules standardized successfully",
        recommended_next_step=next_step,
    )
