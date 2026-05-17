from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

from .chembl_models import ChemBLMoleculeRecord


@dataclass(frozen=True)
class ChemBLMoleculePropsRecord:
    chembl_id: str
    pref_name: str
    canonical_smiles: str
    props: Dict[str, Any]


def _to_number_if_possible(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return x
    try:
        s = str(x).strip()
        if s == "":
            return None
        return float(s)
    except Exception:
        return x


class ChemBLMoleculeService:
    """ChEMBL molecule-related operations (activities→molecules, molecule_properties, SDF export).

    Key reliability features:
    - Never send None/"" query params (avoids `standard_type=None` bug).
    - Paginate activity requests with small page size to avoid ChEMBL 500s on large limits.
    - Retry/backoff on transient HTTP errors (429/5xx/timeouts).
    """

    BASE = "https://www.ebi.ac.uk/chembl/api/data"

    def __init__(self, timeout_s: int = 60, retries: int = 3, backoff_s: float = 1.0) -> None:
        self.timeout_s = int(timeout_s)
        self.retries = int(retries)
        self.backoff_s = float(backoff_s)

    # ---------------- low-level helpers ----------------

    def _get(self, url: str, params: Optional[Dict[str, Any]] = None) -> requests.Response:
        last_exc: Optional[Exception] = None
        for attempt in range(self.retries + 1):
            try:
                r = requests.get(url, params=params, timeout=self.timeout_s)
                # Retry common transient errors / rate limiting
                if r.status_code in (429, 500, 502, 503, 504):
                    r.raise_for_status()
                return r
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, requests.exceptions.HTTPError) as e:
                last_exc = e
                if attempt < self.retries:
                    time.sleep(self.backoff_s * (2**attempt))
        raise RuntimeError(f"ChEMBL request failed after retries: {last_exc}") from last_exc

    @staticmethod
    def _params_clean(d: Dict[str, Any]) -> Dict[str, Any]:
        """Drop None/empty params so requests doesn't serialize them as strings."""
        out: Dict[str, Any] = {}
        for k, v in d.items():
            if v is None:
                continue
            if isinstance(v, str) and v.strip() == "":
                continue
            out[k] = v
        return out

    @staticmethod
    def _clean_ids(ids: Iterable[str]) -> List[str]:
        clean = [str(x).strip().upper() for x in ids if x and str(x).strip()]
        clean = [x for x in clean if x.startswith("CHEMBL")]
        seen = set()
        out: List[str] = []
        for x in clean:
            if x in seen:
                continue
            seen.add(x)
            out.append(x)
        return out

    # ---------------- counts ----------------

    def fetch_activity_total_count_for_target(self, target_chembl_id: str) -> Optional[int]:
        tid = (target_chembl_id or "").strip().upper()
        if not tid:
            return None
        url = f"{self.BASE}/activity.json"
        params = self._params_clean({"target_chembl_id": tid, "limit": 1})
        r = self._get(url, params=params)
        payload = r.json() or {}
        meta = payload.get("page_meta") or {}
        total = meta.get("total_count")
        try:
            return int(total) if total is not None else None
        except Exception:
            return None

    # ---------------- activities → molecules (paginated) ----------------

    def fetch_molecules_for_target(
        self,
        target_chembl_id: str,
        standard_type: Optional[str] = None,
        limit: int = 1000,
        page_size: int = 200,
    ) -> List[ChemBLMoleculeRecord]:
        """Return unique molecules for a target by scanning activity records.

        Parameters
        ----------
        target_chembl_id:
            e.g. CHEMBL202
        standard_type:
            e.g. IC50, Ki... If None/""/"ANY" then no standard_type filter is sent.
        limit:
            Max number of activity records to scan (NOT number of unique molecules).
        page_size:
            Activity page size (keep small to avoid ChEMBL 500s).
        """
        tid = (target_chembl_id or "").strip().upper()
        if not tid:
            return []

        st = (standard_type or "").strip()
        if st.upper() == "ANY":
            st = ""

        # ChEMBL API is prone to 500 when `limit` is large. Paginate with offset.
        remaining = max(1, int(limit))
        page_size = max(1, min(int(page_size), 200))

        url = f"{self.BASE}/activity.json"
        by_id: Dict[str, ChemBLMoleculeRecord] = {}

        offset = 0
        # dynamic page size fallback if we hit 500s
        fallback_sizes = [page_size, 150, 100, 50, 25]

        while remaining > 0:
            this_limit = min(remaining, fallback_sizes[0])

            params = {"target_chembl_id": tid, "limit": this_limit, "offset": offset}
            if st:
                params["standard_type"] = st

            # try a few decreasing page sizes on server errors
            r = None
            last_err: Optional[Exception] = None
            for ps in fallback_sizes:
                try:
                    params["limit"] = min(remaining, ps)
                    r = self._get(url, params=self._params_clean(params))
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    time.sleep(self.backoff_s)
                    continue
            if r is None:
                raise RuntimeError(f"ChEMBL activity paging failed: {last_err}")

            payload = r.json() or {}
            acts = payload.get("activities") or []
            if not acts:
                break

            for a in acts:
                mid = (a.get("molecule_chembl_id") or "").strip().upper()
                if not mid:
                    continue
                smi = (a.get("canonical_smiles") or "").strip()
                if mid not in by_id:
                    by_id[mid] = ChemBLMoleculeRecord(chembl_id=mid, pref_name="", canonical_smiles=smi)
                else:
                    if smi and not by_id[mid].canonical_smiles:
                        by_id[mid] = ChemBLMoleculeRecord(chembl_id=mid, pref_name="", canonical_smiles=smi)

            got = len(acts)
            remaining -= got
            offset += got

            # safety: stop if ChEMBL stops paginating properly
            if got < params["limit"]:
                break

        return list(by_id.values())

    # ---------------- molecule_properties ----------------

    def fetch_molecules_with_properties(
        self,
        ids: Iterable[str],
        prop_keys: Optional[List[str]] = None,
    ) -> List[ChemBLMoleculePropsRecord]:
        mids = self._clean_ids(ids)
        if not mids:
            return []

        # Prefer set endpoint, fall back to per-molecule if needed
        try:
            url = f"{self.BASE}/molecule/set/{','.join(mids)}.json"
            r = self._get(url)
            payload = r.json() or {}
            mols = payload.get("molecules") or []
            out: List[ChemBLMoleculePropsRecord] = []
            for m in mols:
                rec = self._parse_with_props(m, prop_keys)
                if rec is not None:
                    out.append(rec)
            return out
        except Exception:
            out: List[ChemBLMoleculePropsRecord] = []
            for mid in mids:
                url = f"{self.BASE}/molecule/{mid}.json"
                r = self._get(url)
                rec = self._parse_with_props(r.json() or {}, prop_keys)
                if rec is not None:
                    out.append(rec)
            return out

    def _parse_with_props(self, m: dict, prop_keys: Optional[List[str]]) -> Optional[ChemBLMoleculePropsRecord]:
        chembl_id = (m.get("molecule_chembl_id") or "").strip().upper()
        if not chembl_id:
            return None

        pref = (m.get("pref_name") or "").strip()
        ms = m.get("molecule_structures") or {}
        smi = (ms.get("canonical_smiles") or "").strip()

        raw = m.get("molecule_properties") or {}
        props: Dict[str, Any] = dict(raw) if isinstance(raw, dict) else {}

        if prop_keys:
            props = {k: props.get(k) for k in prop_keys}

        props = {k: _to_number_if_possible(v) for k, v in props.items()}

        return ChemBLMoleculePropsRecord(
            chembl_id=chembl_id,
            pref_name=pref,
            canonical_smiles=smi,
            props=props,
        )

    def fetch_available_property_keys(self, sample_ids: Optional[Iterable[str]] = None) -> List[str]:
        ids = self._clean_ids(sample_ids or [])
        keys: set[str] = set()

        if ids:
            recs = self.fetch_molecules_with_properties(ids[:20], prop_keys=None)
            for r in recs:
                keys.update((r.props or {}).keys())
            return sorted(keys)

        url = f"{self.BASE}/molecule.json"
        r = self._get(url, params={"limit": 20})
        payload = r.json() or {}
        mols = payload.get("molecules") or []
        mids = [(m.get("molecule_chembl_id") or "").strip().upper() for m in mols if (m.get("molecule_chembl_id") or "").strip()]
        recs = self.fetch_molecules_with_properties(mids[:20], prop_keys=None)
        for rr in recs:
            keys.update((rr.props or {}).keys())
        return sorted(keys)

    # ---------------- SDF export ----------------

    def fetch_sdf_text(self, ids: Iterable[str]) -> str:
        mids = self._clean_ids(ids)
        if not mids:
            return ""
        try:
            url = f"{self.BASE}/molecule/set/{','.join(mids)}.sdf"
            r = self._get(url)
            return r.text
        except Exception:
            chunks: List[str] = []
            for mid in mids:
                url = f"{self.BASE}/molecule/{mid}.sdf"
                r = self._get(url)
                chunks.append(r.text.rstrip() + "\n")
            return "\n".join(chunks)
