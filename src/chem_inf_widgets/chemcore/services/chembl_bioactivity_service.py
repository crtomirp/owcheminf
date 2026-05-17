from __future__ import annotations

import math
import time
from typing import List, Optional

import requests

from .chembl_models import ChemBLBioactivityRecord
from .rdkit_safe import safe_mol_from_smiles


class ChemBLBioactivityService:
    BASE = "https://www.ebi.ac.uk/chembl/api/data"

    def __init__(self, timeout_s: int = 60, retries: int = 3, backoff_s: float = 1.0) -> None:
        self.timeout_s = int(timeout_s)
        self.retries = int(retries)
        self.backoff_s = float(backoff_s)

    def _get(self, url: str, params: Optional[dict] = None) -> requests.Response:
        last_exc: Optional[Exception] = None
        for attempt in range(self.retries + 1):
            try:
                r = requests.get(url, params=params, timeout=self.timeout_s)
                if r.status_code in (429, 500, 502, 503, 504):
                    r.raise_for_status()
                return r
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, requests.exceptions.HTTPError) as e:
                last_exc = e
                if attempt < self.retries:
                    time.sleep(self.backoff_s * (2**attempt))
        raise RuntimeError(f"ChEMBL request failed after retries: {last_exc}") from last_exc

    def fetch_for_target(
        self,
        target_chembl_id: str,
        standard_type: str = "IC50",
        limit: int = 1000,
    ) -> List[ChemBLBioactivityRecord]:
        tid = (target_chembl_id or "").strip().upper()
        if not tid:
            return []

        st = (standard_type or "").strip()
        url = f"{self.BASE}/activity.json"

        # 1) try with standard_type filter (fast)
        params = {"target_chembl_id": tid, "limit": int(limit)}
        if st:
            params["standard_type"] = st

        try:
            r = self._get(url, params=params)
            payload = r.json() or {}
            acts = payload.get("activities") or []
            return self._parse_activities(acts, tid, prefer_standard_type=st)

        except Exception as e:
            # Fallback: if ChEMBL returns 500 when standard_type is set, retry without it
            if st:
                try:
                    r2 = self._get(url, params={"target_chembl_id": tid, "limit": int(limit)})
                    payload2 = r2.json() or {}
                    acts2 = payload2.get("activities") or []
                    return self._parse_activities(acts2, tid, prefer_standard_type=st)
                except Exception as e2:
                    raise RuntimeError(f"ChEMBL activity fetch failed (with fallback): {e2}") from e2
            raise RuntimeError(f"ChEMBL activity fetch failed: {e}") from e

    def _parse_activities(
        self,
        acts: list,
        target_id: str,
        prefer_standard_type: str,
    ) -> List[ChemBLBioactivityRecord]:
        out: List[ChemBLBioactivityRecord] = []

        for a in acts:
            mol_id = (a.get("molecule_chembl_id") or "").strip()
            if not mol_id:
                continue

            stype = (a.get("standard_type") or "").strip()
            units = (a.get("standard_units") or "").strip()

            # If we ran fallback, optionally filter locally
            if prefer_standard_type and stype and stype.upper() != prefer_standard_type.upper():
                continue

            svalue = _to_float(a.get("standard_value"))
            pchembl = _to_float(a.get("pchembl_value"))

            # canonical smiles (prefer canonical_smiles, else molecule_structures/canonical_smiles)
            smi = (a.get("canonical_smiles") or "").strip()
            if not smi:
                # some payloads include nested structures
                ms = a.get("molecule_structures") or {}
                smi = (ms.get("canonical_smiles") or "").strip()

            smi = canonical_smiles_no_h(smi)

            ic50_nM = None
            if stype.upper() == "IC50":
                ic50_nM = _to_nM(svalue, units)

            out.append(
                ChemBLBioactivityRecord(
                    molecule_chembl_id=mol_id,
                    target_chembl_id=target_id,
                    smiles=smi,
                    standard_type=stype,
                    standard_value=svalue,
                    standard_units=units,
                    pchembl_value=pchembl,
                    ic50_nM=ic50_nM,
                )
            )

        return out


def _to_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None


def _to_nM(value: Optional[float], units: str) -> Optional[float]:
    if value is None:
        return None
    u = (units or "").strip().lower()
    try:
        if u in ("nm", "nmol/l", "nanomolar"):
            return float(value)
        if u in ("um", "µm", "micromolar"):
            return float(value) * 1e3
        if u in ("mm", "millimolar"):
            return float(value) * 1e6
        if u in ("m", "molar"):
            return float(value) * 1e9
    except Exception:
        return None
    return None


def canonical_smiles_no_h(smiles: str) -> str:
    smi = (smiles or "").strip()
    if not smi:
        return ""
    parsed = safe_mol_from_smiles(smi)
    return parsed.canonical_smiles or smi
