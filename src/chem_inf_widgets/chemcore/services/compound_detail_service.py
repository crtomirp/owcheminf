from __future__ import annotations

import base64
import csv
import html
import math
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import Any, Optional
import io

import numpy as np
from Orange.data import ContinuousVariable, Domain, StringVariable, Table, Variable
from rdkit import Chem, rdBase
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

from chem_inf_widgets.chemcore.mol import ChemMol
from chem_inf_widgets.chemcore.services import mol_depict
from chem_inf_widgets.chemcore.services.rdkit_safe import (
    safe_canonical_smiles,
    safe_mol_from_smiles,
    safe_mol_to_inchikey,
)


PHARMAFP_SIZE = 250


@dataclass(frozen=True)
class PharmaFragmentDef:
    fragment_id: int
    category: str
    name: str
    smiles: str
    smarts: str | None
    example_drug: str | None
    primary_target: str | None
    mechanism: str | None
    drug_count: str | None
    frequency: str | None


@dataclass(frozen=True)
class PharmaFragmentHit:
    fragment_id: int
    category: str
    name: str
    smiles: str
    smarts: str | None
    example_drug: str | None
    primary_target: str | None
    mechanism: str | None
    drug_count: str | None
    frequency: str | None
    match_count: int
    matched_atoms: tuple[int, ...]


@dataclass(frozen=True)
class CompoundReference:
    source_index: int
    name: str
    smiles: str
    mol: Chem.Mol


@dataclass(frozen=True)
class SimilarCompoundHit:
    source_index: int
    name: str
    smiles: str
    similarity: float
    shared_fragments: int


@dataclass(frozen=True)
class PharmaSearchHit:
    source_index: int
    name: str
    smiles: str
    pharmafp_similarity: float
    fragment_overlap: float
    shared_fragments: int
    scaffold_match: float
    motif_match_fraction: float
    hybrid_score: float


@dataclass(frozen=True)
class MotifHit:
    key: str
    category: str
    name: str
    smarts: str
    matched_atoms: tuple[int, ...]
    match_count: int
    family: str | None = None
    superclass: str | None = None
    source: str | None = None


@dataclass(frozen=True)
class CompoundProperties:
    smiles: str
    formula: str
    mol_weight: float
    logp: float
    tpsa: float
    hba: int
    hbd: int
    rot_bonds: int
    inchikey: str


@dataclass(frozen=True)
class CompoundDetail:
    name: str
    smiles: str
    properties: CompoundProperties
    fragment_hits: tuple[PharmaFragmentHit, ...]
    coverage: float
    category_count: int
    similar_hits: tuple[SimilarCompoundHit, ...]


@dataclass(frozen=True)
class CompoundDetailOutputs:
    selected_compound: Optional[Table]
    similar_compounds: Optional[Table]
    matched_fragments: Optional[Table]
    detected_motifs: Optional[Table]
    motif_queries: Optional[Table]
    query_molecule: Optional[ChemMol]
    fragment_queries: Optional[Table]
    scaffold_query: Optional[Table]
    search_profile: Optional[Table]


def _round_float(value: float, digits: int = 2) -> float:
    if value is None or not math.isfinite(float(value)):
        return float("nan")
    return round(float(value), digits)


def _safe_inchikey(mol: Chem.Mol) -> str:
    return safe_mol_to_inchikey(mol)


def _safe_query_mol(smarts: str | None, smiles: str) -> Optional[Chem.Mol]:
    blocker = rdBase.BlockLogs()
    try:
        if smarts:
            query = Chem.MolFromSmarts(smarts)
            if query is not None:
                return query
        if smiles:
            parsed = safe_mol_from_smiles(smiles)
            if parsed.mol is not None:
                mol = parsed.mol
                return Chem.MolFromSmarts(Chem.MolToSmarts(mol))
    finally:
        del blocker
    return None


def _safe_smarts_mol(smarts: str) -> Optional[Chem.Mol]:
    blocker = rdBase.BlockLogs()
    try:
        return Chem.MolFromSmarts((smarts or "").strip())
    finally:
        del blocker


def _safe_mol_from_smiles(smiles: str) -> Optional[Chem.Mol]:
    return safe_mol_from_smiles((smiles or "").strip()).mol


def _iter_all_vars(table: Table) -> list[Variable]:
    return list(table.domain.metas) + list(table.domain.attributes) + list(table.domain.class_vars)


def _string_vars(table: Table) -> list[Variable]:
    return [var for var in _iter_all_vars(table) if isinstance(var, StringVariable)]


def _find_var_name(table: Table, candidates: list[str]) -> str:
    normalized = {candidate.strip().lower() for candidate in candidates}
    for var in _string_vars(table):
        if var.name.strip().lower() in normalized:
            return var.name
    return ""


def default_smiles_var_name(table: Table) -> str:
    preferred = _find_var_name(table, ["SMILES", "SMILES_STD", "canonical_smiles", "smile"])
    if preferred:
        return preferred
    string_vars = _string_vars(table)
    return string_vars[0].name if string_vars else ""


def default_name_var_name(table: Table) -> str:
    return _find_var_name(table, ["Name", "name", "Compound", "CHEMBL_ID", "ID"])


def row_text(row, var_name: str) -> str:
    if not var_name:
        return ""
    try:
        value = row[var_name]
    except Exception:
        return ""
    if value is None:
        return ""
    return str(value).strip()


def png_data_uri(mol: Chem.Mol, *, highlight_atoms: Optional[list[int]] = None, size: int = 360) -> str:
    png = mol_depict.render_mol_png(mol, size=size, highlight_atoms=highlight_atoms or [])
    encoded = base64.b64encode(png).decode("ascii")
    return f"data:image/png;base64,{encoded}"


@lru_cache(maxsize=1)
def load_pharmafp_library() -> tuple[PharmaFragmentDef, ...]:
    resource = resources.files("chem_inf_widgets.chemcore.data").joinpath("pharmafp250.json")
    data = resource.read_text(encoding="utf-8")
    rows: list[dict[str, Any]] = __import__("json").loads(data)
    library: list[PharmaFragmentDef] = []
    for row in rows:
        library.append(
            PharmaFragmentDef(
                fragment_id=int(row.get("id") or 0),
                category=str(row.get("category") or "Uncategorized"),
                name=str(row.get("name") or f"Fragment {row.get('id') or '?'}"),
                smiles=str(row.get("smiles") or ""),
                smarts=row.get("smarts"),
                example_drug=row.get("example_drug"),
                primary_target=row.get("primary_target"),
                mechanism=row.get("mechanism"),
                drug_count=row.get("drug_count"),
                frequency=row.get("frequency"),
            )
        )
    return tuple(library)


@lru_cache(maxsize=1)
def load_functional_group_patterns() -> tuple[dict[str, str], ...]:
    resource = resources.files("chem_inf_widgets.chemcore.data").joinpath("patterns.csv")
    text = resource.read_text(encoding="utf-8")
    reader = csv.DictReader(io.StringIO(text))
    rows: list[dict[str, str]] = []
    for row in reader:
        category = str(row.get("category") or "").strip().lower()
        if category not in {"functional_groups", "functional_group"}:
            continue
        smarts = str(row.get("smarts") or "").strip()
        name = str(row.get("name") or "").strip()
        pattern_id = str(row.get("pattern_id") or name.lower().replace(" ", "_")).strip()
        if not smarts or not name:
            continue
        rows.append(
            {
                "pattern_id": pattern_id,
                "name": name,
                "smarts": smarts,
                "category": "functional_group",
                "priority": str(row.get("priority") or ""),
                "source": str(row.get("source") or "faircheckmol_patterns"),
            }
        )
    return tuple(rows)


@lru_cache(maxsize=1)
def load_heterocycle_registry() -> tuple[dict[str, Any], ...]:
    resource = resources.files("chem_inf_widgets.chemcore.data").joinpath("cyclic_registry.json")
    payload = __import__("json").loads(resource.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for row in payload.get("records", []):
        if not row.get("is_true_heterocycle"):
            continue
        smarts = str(row.get("smarts") or "").strip()
        name = str(row.get("name") or "").strip()
        if not smarts or not name:
            continue
        rows.append(row)
    return tuple(rows)


@lru_cache(maxsize=1)
def compiled_pharmafp_patterns() -> tuple[tuple[PharmaFragmentDef, Optional[Chem.Mol]], ...]:
    compiled = []
    for fragment in load_pharmafp_library():
        compiled.append((fragment, _safe_query_mol(fragment.smarts, fragment.smiles)))
    return tuple(compiled)


@lru_cache(maxsize=1)
def compiled_functional_group_patterns() -> tuple[tuple[dict[str, str], Optional[Chem.Mol]], ...]:
    return tuple((row, _safe_smarts_mol(row["smarts"])) for row in load_functional_group_patterns())


@lru_cache(maxsize=1)
def compiled_heterocycle_patterns() -> tuple[tuple[dict[str, Any], Optional[Chem.Mol]], ...]:
    return tuple((row, _safe_smarts_mol(str(row.get("smarts") or ""))) for row in load_heterocycle_registry())


def compute_pharmafp_hits(mol: Chem.Mol) -> tuple[tuple[PharmaFragmentHit, ...], tuple[int, ...]]:
    hits: list[PharmaFragmentHit] = []
    bits: list[int] = []
    for fragment, query in compiled_pharmafp_patterns():
        if query is None:
            bits.append(0)
            continue
        try:
            matches = list(mol.GetSubstructMatches(query, uniquify=True))
        except Exception:
            matches = []
        if matches:
            bits.append(1)
            matched_atoms = tuple(sorted({int(atom) for match in matches for atom in match}))
            hits.append(
                PharmaFragmentHit(
                    fragment_id=fragment.fragment_id,
                    category=fragment.category,
                    name=fragment.name,
                    smiles=fragment.smiles,
                    smarts=fragment.smarts,
                    example_drug=fragment.example_drug,
                    primary_target=fragment.primary_target,
                    mechanism=fragment.mechanism,
                    drug_count=fragment.drug_count,
                    frequency=fragment.frequency,
                    match_count=len(matches),
                    matched_atoms=matched_atoms,
                )
            )
        else:
            bits.append(0)
    hits.sort(key=lambda item: (item.category.lower(), item.name.lower(), item.fragment_id))
    return tuple(hits), tuple(bits)


def compute_motif_hits(
    mol: Chem.Mol,
    *,
    include_heterocycles: bool = True,
    include_functional_groups: bool = True,
) -> tuple[MotifHit, ...]:
    hits: list[MotifHit] = []

    if include_heterocycles:
        for row, query in compiled_heterocycle_patterns():
            if query is None:
                continue
            try:
                matches = list(mol.GetSubstructMatches(query, uniquify=True))
            except Exception:
                matches = []
            if not matches:
                continue
            atoms = tuple(sorted({int(atom) for match in matches for atom in match}))
            hits.append(
                MotifHit(
                    key=str(row.get("pattern_id") or row.get("name") or "heterocycle"),
                    category="heterocycle",
                    name=str(row.get("name") or "Heterocycle"),
                    smarts=str(row.get("smarts") or ""),
                    matched_atoms=atoms,
                    match_count=len(matches),
                    family=row.get("heterocycle_family_normalized") or row.get("heterocycle_family"),
                    superclass=row.get("heterocycle_superclass"),
                    source="cyclic_registry",
                )
            )

    if include_functional_groups:
        for row, query in compiled_functional_group_patterns():
            if query is None:
                continue
            try:
                matches = list(mol.GetSubstructMatches(query, uniquify=True))
            except Exception:
                matches = []
            if not matches:
                continue
            atoms = tuple(sorted({int(atom) for match in matches for atom in match}))
            hits.append(
                MotifHit(
                    key=str(row.get("pattern_id") or row.get("name") or "functional_group"),
                    category="functional_group",
                    name=str(row.get("name") or "Functional Group"),
                    smarts=str(row.get("smarts") or ""),
                    matched_atoms=atoms,
                    match_count=len(matches),
                    source=row.get("source"),
                )
            )

    def sort_key(hit: MotifHit):
        is_hetero = 0 if hit.category == "heterocycle" else 1
        fam_rank = 0 if hit.family else 1
        return (
            is_hetero,
            fam_rank,
            -len(hit.matched_atoms),
            -hit.match_count,
            hit.name.lower(),
            hit.key.lower(),
        )

    deduped: dict[tuple[str, tuple[int, ...]], MotifHit] = {}
    for hit in sorted(hits, key=sort_key):
        deduped.setdefault((hit.category, hit.matched_atoms), hit)
    return tuple(sorted(deduped.values(), key=sort_key))


def tanimoto_bits(bits_a: tuple[int, ...], bits_b: tuple[int, ...]) -> float:
    union = sum(1 for a, b in zip(bits_a, bits_b) if a or b)
    if union == 0:
        return 0.0
    intersection = sum(1 for a, b in zip(bits_a, bits_b) if a and b)
    return float(intersection) / float(union)


def _shared_bits(bits_a: tuple[int, ...], bits_b: tuple[int, ...]) -> int:
    return sum(1 for a, b in zip(bits_a, bits_b) if a and b)


def compute_properties(mol: Chem.Mol) -> CompoundProperties:
    smiles = safe_canonical_smiles(mol, remove_hs=False, canonical=True, isomeric=True)
    return CompoundProperties(
        smiles=smiles,
        formula=rdMolDescriptors.CalcMolFormula(mol),
        mol_weight=_round_float(Descriptors.MolWt(mol), 2),
        logp=_round_float(Crippen.MolLogP(mol), 2),
        tpsa=_round_float(rdMolDescriptors.CalcTPSA(mol), 2),
        hba=int(rdMolDescriptors.CalcNumHBA(mol)),
        hbd=int(rdMolDescriptors.CalcNumHBD(mol)),
        rot_bonds=int(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        inchikey=_safe_inchikey(mol),
    )


def _murcko_scaffold_smiles(mol: Chem.Mol) -> str:
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    except ValueError:
        return ""
    if scaffold is None or scaffold.GetNumAtoms() == 0:
        return ""
    return safe_canonical_smiles(scaffold, remove_hs=False)


def _generic_scaffold_smiles(mol: Chem.Mol) -> str:
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        generic = MurckoScaffold.MakeScaffoldGeneric(scaffold)
    except ValueError:
        return ""
    if generic is None or generic.GetNumAtoms() == 0:
        return ""
    return safe_canonical_smiles(generic, remove_hs=False)


def build_detail(
    mol: Chem.Mol,
    *,
    name: str = "",
    reference: Optional[list[CompoundReference]] = None,
    top_k: int = 5,
    exclude_smiles: Optional[str] = None,
) -> CompoundDetail:
    properties = compute_properties(mol)
    hits, bits = compute_pharmafp_hits(mol)

    similar_hits: list[SimilarCompoundHit] = []
    reference = reference or []
    query_smiles = (exclude_smiles or properties.smiles or "").strip()
    for record in reference:
        ref_smiles = (record.smiles or "").strip()
        if not ref_smiles or (query_smiles and ref_smiles == query_smiles):
            continue
        ref_hits, ref_bits = compute_pharmafp_hits(record.mol)
        score = tanimoto_bits(bits, ref_bits)
        if score <= 0:
            continue
        similar_hits.append(
            SimilarCompoundHit(
                source_index=int(record.source_index),
                name=record.name,
                smiles=record.smiles,
                similarity=round(score, 4),
                shared_fragments=_shared_bits(bits, ref_bits),
            )
        )
    similar_hits.sort(key=lambda item: (-item.similarity, -item.shared_fragments, item.source_index))

    categories = {hit.category for hit in hits}
    coverage = float(sum(bits)) / float(PHARMAFP_SIZE) if bits else 0.0
    return CompoundDetail(
        name=name,
        smiles=properties.smiles,
        properties=properties,
        fragment_hits=tuple(hits),
        coverage=coverage,
        category_count=len(categories),
        similar_hits=tuple(similar_hits[: max(1, int(top_k))]),
    )


def render_summary_html(detail: CompoundDetail) -> str:
    p = detail.properties
    matched = len(detail.fragment_hits)
    metric_cards = [
        ("Formula", html.escape(p.formula)),
        ("MolWt", str(p.mol_weight)),
        ("LogP", str(p.logp)),
        ("TPSA", str(p.tpsa)),
        ("HBA / HBD", f"{p.hba} / {p.hbd}"),
        ("RotB", str(p.rot_bonds)),
    ]
    metric_html = "".join(
        f"""
        <div style="display:inline-block; width:30%; min-width:120px; vertical-align:top; margin:0 8px 8px 0; padding:8px 10px; background:#ffffff; border:1px solid #e2e8f0; border-radius:10px;">
          <div style="font-size:11px; color:#64748b; margin-bottom:3px;">{label}</div>
          <div style="font-size:13px; color:#0f172a; font-weight:600;">{value}</div>
        </div>
        """
        for label, value in metric_cards
    )
    return f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; padding: 2px;">
      <h2 style="margin:0 0 8px 0; font-size:18px; color:#0f172a;">{html.escape(detail.name or 'Compound')}</h2>
      <div style="margin-bottom:12px;">
        <span style="display:inline-block; border:1px solid #cfd6de; border-radius:999px; padding:4px 9px; margin:0 6px 6px 0; background:#ffffff;">{matched} fragments</span>
        <span style="display:inline-block; border:1px solid #cfd6de; border-radius:999px; padding:4px 9px; margin:0 6px 6px 0; background:#ffffff;">{detail.coverage * 100:.1f}% coverage</span>
        <span style="display:inline-block; border:1px solid #cfd6de; border-radius:999px; padding:4px 9px; margin:0 6px 6px 0; background:#ffffff;">{detail.category_count} categories</span>
      </div>
      <div style="font-size:12px; color:#5f6b7a; margin-bottom:6px; font-weight:600;">Canonical SMILES</div>
      <div style="font-family: Menlo, monospace; font-size:12px; line-height:1.45; padding:8px 10px; background:#ffffff; border:1px solid #e2e8f0; border-radius:10px; margin-bottom:12px;">{html.escape(p.smiles)}</div>
      <div style="margin-bottom:8px;">{metric_html}</div>
      <div style="font-size:11px; color:#64748b; margin-top:4px;">
        <b>InChIKey</b>: <span style="font-family: Menlo, monospace;">{html.escape(p.inchikey or '—')}</span>
      </div>
    </div>
    """


def render_fragment_detail_html(selected_hits: tuple[PharmaFragmentHit, ...]) -> str:
    if not selected_hits:
        return "<div style='color:#5f6b7a;'>Select one or more matched fragments to inspect them together.</div>"
    atoms_union = sorted({atom for hit in selected_hits for atom in hit.matched_atoms})
    selected_categories = sorted({hit.category for hit in selected_hits if hit.category})
    blocks = []
    for hit in selected_hits:
        badge = hit.frequency or "Matched"
        atoms = ", ".join(str(atom) for atom in hit.matched_atoms)
        blocks.append(
            f"""
            <div style="margin-bottom:10px; padding-bottom:10px; border-bottom:1px solid #e5e7eb;">
              <div style="font-weight:700;">{html.escape(hit.name)} <span style="font-size:11px; border:1px solid #cfd6de; border-radius:999px; padding:2px 8px;">{html.escape(badge)}</span></div>
              <div style="color:#5f6b7a; margin:4px 0 6px 0;">{html.escape(hit.category)}</div>
              <div style="font-size:12px;"><b>Matched atoms</b>: {html.escape(atoms or '—')}</div>
              <div style="font-size:12px;"><b>SMILES/SMARTS</b>: <span style="font-family: Menlo, monospace;">{html.escape(hit.smarts or hit.smiles)}</span></div>
              <div style="font-size:12px;"><b>Example drug</b>: {html.escape(hit.example_drug or '—')}</div>
              <div style="font-size:12px;"><b>Target</b>: {html.escape(hit.primary_target or '—')}</div>
            </div>
            """
        )
    return f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;">
      <div style="margin-bottom:10px;">
        <b>{len(selected_hits)} selected fragments</b> | atoms highlighted: {len(atoms_union)} | categories: {html.escape(', '.join(selected_categories) if selected_categories else '—')}
      </div>
      {''.join(blocks)}
    </div>
    """


def render_motif_detail_html(selected_hits: tuple[MotifHit, ...], *, motif_logic: str = "or") -> str:
    if not selected_hits:
        return "<div style='color:#5f6b7a;'>Select one or more motifs to inspect them.</div>"
    html_rows = []
    for hit in selected_hits:
        atoms = ", ".join(str(atom) for atom in hit.matched_atoms)
        extra = []
        if hit.family:
            extra.append(f"Family: {html.escape(hit.family)}")
        if hit.superclass:
            extra.append(f"Superclass: {html.escape(hit.superclass)}")
        html_rows.append(
            f"""
            <div style="margin-bottom:10px; padding-bottom:10px; border-bottom:1px solid #e5e7eb;">
              <div style="font-weight:700;">{html.escape(hit.name)} <span style="font-size:11px; color:#5f6b7a;">[{html.escape(hit.category)}]</span></div>
              <div style="font-family: Menlo, monospace; font-size:12px; margin:4px 0;">{html.escape(hit.smarts)}</div>
              <div style="font-size:12px; color:#334155;">Matches: {hit.match_count} | Atoms: {html.escape(atoms)}</div>
              <div style="font-size:12px; color:#475569;">{' | '.join(extra) if extra else 'No extra annotation'}</div>
            </div>
            """
        )
    return f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;">
      <div style="margin-bottom:8px;"><b>{len(selected_hits)} motif queries</b> with <b>{html.escape((motif_logic or 'or').upper())}</b> logic.</div>
      {''.join(html_rows)}
    </div>
    """


def build_detail_outputs(
    detail: CompoundDetail,
    *,
    motif_hits: tuple[MotifHit, ...],
    selected_motif_hits: tuple[MotifHit, ...],
    motif_logic: str = "or",
) -> CompoundDetailOutputs:
    return CompoundDetailOutputs(
        selected_compound=selected_compound_table(detail),
        similar_compounds=similar_hits_table(detail.similar_hits),
        matched_fragments=fragment_hits_table(detail.fragment_hits),
        detected_motifs=motif_hits_table(motif_hits),
        motif_queries=selected_motif_query_table(selected_motif_hits),
        query_molecule=query_molecule_from_detail(detail),
        fragment_queries=fragment_query_table(detail.fragment_hits),
        scaffold_query=scaffold_query_table(detail),
        search_profile=search_profile_table(detail, motif_queries=selected_motif_hits, motif_logic=motif_logic),
    )


def references_from_table(
    data: Optional[Table],
    *,
    smiles_var_name: str = "",
    name_var_name: str = "",
) -> list[CompoundReference]:
    if data is None:
        return []
    smiles_var_name = smiles_var_name or default_smiles_var_name(data)
    name_var_name = name_var_name or default_name_var_name(data)
    refs: list[CompoundReference] = []
    for idx, row in enumerate(data):
        smiles = row_text(row, smiles_var_name)
        mol = _safe_mol_from_smiles(smiles)
        if mol is None:
            continue
        refs.append(
            CompoundReference(
                source_index=idx,
                name=row_text(row, name_var_name) or f"Compound {idx + 1}",
                smiles=smiles,
                mol=mol,
            )
        )
    return refs


def references_from_molecules(molecules: Optional[list[ChemMol]]) -> list[CompoundReference]:
    refs: list[CompoundReference] = []
    for idx, cm in enumerate(molecules or []):
        mol = cm.to_rdkit()
        if mol is None:
            continue
        refs.append(
            CompoundReference(
                source_index=idx,
                name=(cm.name or "").strip() or f"Compound {idx + 1}",
                smiles=cm.canonical_smiles(),
                mol=mol,
            )
        )
    return refs


def fragment_hits_table(hits: tuple[PharmaFragmentHit, ...]) -> Optional[Table]:
    if not hits:
        return None
    domain = Domain(
        [
            ContinuousVariable("Match Count"),
            ContinuousVariable("Matched Atom Count"),
        ],
        metas=[
            StringVariable("Category"),
            StringVariable("Fragment"),
            StringVariable("Frequency"),
            StringVariable("Drug Count"),
            StringVariable("Example Drug"),
            StringVariable("Primary Target"),
            StringVariable("Mechanism"),
            StringVariable("SMILES/SMARTS"),
            StringVariable("Matched Atoms"),
        ],
    )
    X = np.array(
        [[float(hit.match_count), float(len(hit.matched_atoms))] for hit in hits],
        dtype=float,
    )
    metas = np.array(
        [
            [
                hit.category,
                hit.name,
                hit.frequency or "",
                hit.drug_count or "",
                hit.example_drug or "",
                hit.primary_target or "",
                hit.mechanism or "",
                hit.smarts or hit.smiles,
                ", ".join(str(i) for i in hit.matched_atoms),
            ]
            for hit in hits
        ],
        dtype=object,
    )
    return Table.from_numpy(domain, X=X, metas=metas)


def similar_hits_table(hits: tuple[SimilarCompoundHit, ...]) -> Optional[Table]:
    if not hits:
        return None
    domain = Domain(
        [
            ContinuousVariable("PharmaFP Similarity"),
            ContinuousVariable("Shared Fragments"),
            ContinuousVariable("Reference Index"),
        ],
        metas=[
            StringVariable("Name"),
            StringVariable("SMILES"),
        ],
    )
    X = np.array(
        [[float(hit.similarity), float(hit.shared_fragments), float(hit.source_index)] for hit in hits],
        dtype=float,
    )
    metas = np.array([[hit.name, hit.smiles] for hit in hits], dtype=object)
    return Table.from_numpy(domain, X=X, metas=metas)


def selected_compound_table(detail: CompoundDetail) -> Table:
    domain = Domain(
        [
            ContinuousVariable("MolWt"),
            ContinuousVariable("LogP"),
            ContinuousVariable("TPSA"),
            ContinuousVariable("HBA"),
            ContinuousVariable("HBD"),
            ContinuousVariable("RotB"),
            ContinuousVariable("PharmaFP Coverage"),
            ContinuousVariable("Matched Fragment Count"),
            ContinuousVariable("Category Count"),
        ],
        metas=[
            StringVariable("Name"),
            StringVariable("SMILES"),
            StringVariable("Formula"),
            StringVariable("InChIKey"),
        ],
    )
    X = np.array(
        [[
            detail.properties.mol_weight,
            detail.properties.logp,
            detail.properties.tpsa,
            float(detail.properties.hba),
            float(detail.properties.hbd),
            float(detail.properties.rot_bonds),
            float(detail.coverage),
            float(len(detail.fragment_hits)),
            float(detail.category_count),
        ]],
        dtype=float,
    )
    metas = np.array(
        [[
            detail.name or "",
            detail.properties.smiles,
            detail.properties.formula,
            detail.properties.inchikey,
        ]],
        dtype=object,
    )
    return Table.from_numpy(domain, X=X, metas=metas)


def query_molecule_from_detail(detail: CompoundDetail) -> ChemMol:
    mol = _safe_mol_from_smiles(detail.properties.smiles)
    if mol is None:
        raise ValueError("Cannot create query molecule from detail.")
    props = {
        "SMILES": detail.properties.smiles,
        "Formula": detail.properties.formula,
        "InChIKey": detail.properties.inchikey,
    }
    return ChemMol(mol=mol, name=detail.name or None, props=props)


def fragment_query_table(hits: tuple[PharmaFragmentHit, ...]) -> Optional[Table]:
    if not hits:
        return None
    domain = Domain(
        [
            ContinuousVariable("Match Count"),
            ContinuousVariable("Matched Atom Count"),
        ],
        metas=[
            StringVariable("Fragment"),
            StringVariable("Category"),
            StringVariable("SMARTS"),
            StringVariable("SMILES"),
            StringVariable("Matched Atoms"),
            StringVariable("Frequency"),
            StringVariable("Example Drug"),
            StringVariable("Primary Target"),
            StringVariable("Mechanism"),
        ],
    )
    X = np.array([[float(hit.match_count), float(len(hit.matched_atoms))] for hit in hits], dtype=float)
    metas = np.array(
        [
            [
                hit.name,
                hit.category,
                hit.smarts or "",
                hit.smiles,
                ", ".join(str(atom) for atom in hit.matched_atoms),
                hit.frequency or "",
                hit.example_drug or "",
                hit.primary_target or "",
                hit.mechanism or "",
            ]
            for hit in hits
        ],
        dtype=object,
    )
    return Table.from_numpy(domain, X=X, metas=metas)


def scaffold_query_table(detail: CompoundDetail) -> Optional[Table]:
    mol = _safe_mol_from_smiles(detail.properties.smiles)
    if mol is None:
        return None
    murcko = _murcko_scaffold_smiles(mol)
    generic = _generic_scaffold_smiles(mol)
    domain = Domain(
        [],
        metas=[
            StringVariable("Name"),
            StringVariable("SMILES"),
            StringVariable("Murcko Scaffold"),
            StringVariable("Generic Scaffold"),
        ],
    )
    metas = np.array(
        [[detail.name or "", detail.properties.smiles, murcko, generic]],
        dtype=object,
    )
    return Table.from_numpy(domain, X=np.zeros((1, 0), dtype=float), metas=metas)


def search_profile_table(
    detail: CompoundDetail,
    *,
    similarity_threshold: float = 0.35,
    search_mode: str = "hybrid",
    motif_queries: Optional[tuple[MotifHit, ...]] = None,
    motif_logic: str = "or",
) -> Table:
    fragment_names = "|".join(hit.name for hit in detail.fragment_hits)
    fragment_smarts = "|".join((hit.smarts or hit.smiles) for hit in detail.fragment_hits)
    motif_names = "|".join(hit.name for hit in (motif_queries or ()))
    motif_smarts = "|".join(hit.smarts for hit in (motif_queries or ()) if hit.smarts)
    mol = _safe_mol_from_smiles(detail.properties.smiles)
    murcko = _murcko_scaffold_smiles(mol) if mol is not None else ""
    generic = _generic_scaffold_smiles(mol) if mol is not None else ""
    domain = Domain(
        [
            ContinuousVariable("Similarity Threshold"),
            ContinuousVariable("Matched Fragment Count"),
            ContinuousVariable("PharmaFP Coverage"),
        ],
        metas=[
            StringVariable("Query SMILES"),
            StringVariable("Query Scaffold"),
            StringVariable("Query Generic Scaffold"),
            StringVariable("Matched Fragment Names"),
            StringVariable("Matched Fragment SMARTS"),
            StringVariable("Selected Motif Names"),
            StringVariable("Selected Motif SMARTS"),
            StringVariable("Motif Logic"),
            StringVariable("Preferred Search Mode"),
        ],
    )
    X = np.array(
        [[float(similarity_threshold), float(len(detail.fragment_hits)), float(detail.coverage)]],
        dtype=float,
    )
    metas = np.array(
        [[
            detail.properties.smiles,
            murcko,
            generic,
            fragment_names,
            fragment_smarts,
            motif_names,
            motif_smarts,
            motif_logic,
            search_mode,
        ]],
        dtype=object,
    )
    return Table.from_numpy(domain, X=X, metas=metas)


def _table_meta_text(row, name: str) -> str:
    try:
        value = row[name]
    except Exception:
        return ""
    if value is None:
        return ""
    return str(value).strip()


def _table_attr_float(row, name: str, default: float = 0.0) -> float:
    try:
        return float(row[name])
    except Exception:
        return default


def query_from_search_profile(profile: Optional[Table]) -> tuple[str, str, str, float, str]:
    if profile is None or len(profile) == 0:
        return "", "", "", 0.35, "or"
    row = profile[0]
    return (
        _table_meta_text(row, "Query SMILES"),
        _table_meta_text(row, "Query Scaffold"),
        _table_meta_text(row, "Query Generic Scaffold"),
        _table_attr_float(row, "Similarity Threshold", 0.35),
        _table_meta_text(row, "Motif Logic") or "or",
    )


def fragments_from_query_table(data: Optional[Table]) -> list[str]:
    if data is None:
        return []
    queries: list[str] = []
    for row in data:
        value = _table_meta_text(row, "SMARTS") or _table_meta_text(row, "SMILES")
        if value:
            queries.append(value)
    return queries


def motif_hits_table(hits: tuple[MotifHit, ...]) -> Optional[Table]:
    if not hits:
        return None
    domain = Domain(
        [
            ContinuousVariable("Match Count"),
            ContinuousVariable("Matched Atom Count"),
        ],
        metas=[
            StringVariable("Key"),
            StringVariable("Category"),
            StringVariable("Name"),
            StringVariable("SMARTS"),
            StringVariable("Matched Atoms"),
            StringVariable("Family"),
            StringVariable("Superclass"),
            StringVariable("Source"),
        ],
    )
    X = np.array([[float(hit.match_count), float(len(hit.matched_atoms))] for hit in hits], dtype=float)
    metas = np.array(
        [
            [
                hit.key,
                hit.category,
                hit.name,
                hit.smarts,
                ", ".join(str(atom) for atom in hit.matched_atoms),
                hit.family or "",
                hit.superclass or "",
                hit.source or "",
            ]
            for hit in hits
        ],
        dtype=object,
    )
    return Table.from_numpy(domain, X=X, metas=metas)


def selected_motif_query_table(hits: tuple[MotifHit, ...]) -> Optional[Table]:
    if not hits:
        return None
    domain = Domain(
        [],
        metas=[
            StringVariable("Key"),
            StringVariable("Category"),
            StringVariable("Name"),
            StringVariable("SMARTS"),
            StringVariable("Matched Atoms"),
            StringVariable("Family"),
            StringVariable("Superclass"),
        ],
    )
    metas = np.array(
        [
            [
                hit.key,
                hit.category,
                hit.name,
                hit.smarts,
                ", ".join(str(atom) for atom in hit.matched_atoms),
                hit.family or "",
                hit.superclass or "",
            ]
            for hit in hits
        ],
        dtype=object,
    )
    return Table.from_numpy(domain, X=np.zeros((len(hits), 0), dtype=float), metas=metas)


def run_pharmafp_search(
    *,
    query_smiles: str,
    reference: list[CompoundReference],
    fragment_queries: Optional[list[str]] = None,
    motif_queries: Optional[list[str]] = None,
    motif_logic: str = "or",
    query_scaffold: str = "",
    query_generic_scaffold: str = "",
    top_k: int = 25,
    min_similarity: float = 0.0,
    mode: str = "hybrid",
) -> list[PharmaSearchHit]:
    query_mol = _safe_mol_from_smiles(query_smiles)
    if query_mol is None or not reference:
        return []
    query_hits, query_bits = compute_pharmafp_hits(query_mol)
    query_fragment_set = {hit.smarts or hit.smiles for hit in query_hits}
    if fragment_queries:
        query_fragment_set = {value for value in fragment_queries if value}
    motif_query_mols = [qmol for value in (motif_queries or []) if (qmol := _safe_smarts_mol(value)) is not None]
    motif_logic_norm = (motif_logic or "or").strip().lower()

    out: list[PharmaSearchHit] = []
    for record in reference:
        ref_smiles = (record.smiles or "").strip()
        if not ref_smiles or ref_smiles == query_smiles:
            continue
        ref_hits, ref_bits = compute_pharmafp_hits(record.mol)
        pharmafp_similarity = round(tanimoto_bits(query_bits, ref_bits), 4)
        if pharmafp_similarity < float(min_similarity):
            continue

        ref_fragment_set = {hit.smarts or hit.smiles for hit in ref_hits}
        overlap_count = len(query_fragment_set & ref_fragment_set) if query_fragment_set else 0
        fragment_overlap = (overlap_count / len(query_fragment_set)) if query_fragment_set else 0.0

        motif_match_fraction = 0.0
        if motif_query_mols:
            motif_matches: list[bool] = []
            for qmol in motif_query_mols:
                try:
                    motif_matches.append(bool(record.mol.HasSubstructMatch(qmol)))
                except Exception:
                    motif_matches.append(False)
            motif_match_fraction = sum(1 for flag in motif_matches if flag) / len(motif_matches)
            if motif_logic_norm == "and" and not all(motif_matches):
                continue
            if motif_logic_norm != "and" and not any(motif_matches):
                continue

        scaffold_match = 0.0
        ref_murcko = _murcko_scaffold_smiles(record.mol)
        ref_generic = _generic_scaffold_smiles(record.mol)
        if query_scaffold and ref_murcko and query_scaffold == ref_murcko:
            scaffold_match = 1.0
        elif query_generic_scaffold and ref_generic and query_generic_scaffold == ref_generic:
            scaffold_match = 0.7

        mode_l = (mode or "hybrid").strip().lower()
        if mode_l == "fragment":
            hybrid_score = fragment_overlap
        elif mode_l == "similarity":
            hybrid_score = pharmafp_similarity
        elif mode_l == "scaffold":
            hybrid_score = scaffold_match
        else:
            if motif_query_mols:
                hybrid_score = 0.4 * pharmafp_similarity + 0.25 * scaffold_match + 0.35 * motif_match_fraction
            else:
                hybrid_score = 0.5 * pharmafp_similarity + 0.3 * scaffold_match + 0.2 * fragment_overlap

        out.append(
            PharmaSearchHit(
                source_index=record.source_index,
                name=record.name,
                smiles=record.smiles,
                pharmafp_similarity=pharmafp_similarity,
                fragment_overlap=round(fragment_overlap, 4),
                shared_fragments=_shared_bits(query_bits, ref_bits),
                scaffold_match=round(scaffold_match, 4),
                motif_match_fraction=round(motif_match_fraction, 4),
                hybrid_score=round(hybrid_score, 4),
            )
        )
    out.sort(key=lambda item: (-item.hybrid_score, -item.pharmafp_similarity, -item.shared_fragments, item.source_index))
    return out[: max(1, int(top_k))]


def pharmafp_search_hits_table(hits: list[PharmaSearchHit]) -> Table:
    domain = Domain(
        [
            ContinuousVariable("Hybrid Score"),
            ContinuousVariable("PharmaFP Similarity"),
            ContinuousVariable("Motif Match Fraction"),
            ContinuousVariable("Fragment Overlap"),
            ContinuousVariable("Shared Fragments"),
            ContinuousVariable("Scaffold Match"),
            ContinuousVariable("Reference Index"),
        ],
        metas=[
            StringVariable("Name"),
            StringVariable("SMILES"),
        ],
    )
    X = np.array(
        [
            [
                float(hit.hybrid_score),
                float(hit.pharmafp_similarity),
                float(hit.motif_match_fraction),
                float(hit.fragment_overlap),
                float(hit.shared_fragments),
                float(hit.scaffold_match),
                float(hit.source_index),
            ]
            for hit in hits
        ],
        dtype=float,
    ) if hits else np.zeros((0, 7), dtype=float)
    metas = np.array([[hit.name, hit.smiles] for hit in hits], dtype=object) if hits else np.zeros((0, 2), dtype=object)
    return Table.from_numpy(domain, X=X, metas=metas)
