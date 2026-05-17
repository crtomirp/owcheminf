from __future__ import annotations

import time
from typing import List, Optional
from urllib.parse import urljoin

import requests
from requests import Response

from .chembl_models import ChemBLTargetRecord
from .disk_cache import CachePolicy, DiskCache


class ChemBLTargetService:
    """Target search/fetch via ChEMBL REST API (chemcore only) with retry + pagination + disk cache."""

    BASE = "https://www.ebi.ac.uk/chembl/api/data"

    def __init__(
        self,
        timeout_s: int = 60,
        retries: int = 3,
        backoff_s: float = 1.0,
        cache: Optional[DiskCache] = None,
        cache_policy: CachePolicy = CachePolicy(enabled=True, ttl_s=24 * 3600),
    ) -> None:
        self.timeout_s = int(timeout_s)
        self.retries = int(retries)
        self.backoff_s = float(backoff_s)
        self.cache = cache or DiskCache("chembl")
        self.cache_policy = cache_policy

    def _request(self, url: str, params: Optional[dict] = None) -> Response:
        last_exc: Optional[Exception] = None

        for attempt in range(self.retries + 1):
            try:
                r = requests.get(url, params=params, timeout=self.timeout_s)

                # Retry-worthy responses
                if r.status_code in (429, 500, 502, 503, 504):
                    r.raise_for_status()

                return r

            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                last_exc = e

            except requests.exceptions.HTTPError as e:
                status = getattr(e.response, "status_code", None)
                if status in (429, 500, 502, 503, 504):
                    last_exc = e
                else:
                    raise

            if attempt < self.retries:
                time.sleep(self.backoff_s * (2**attempt))

        raise RuntimeError(f"ChEMBL target request failed after retries: {last_exc}") from last_exc

    @staticmethod
    def _serialize(records: List[ChemBLTargetRecord]) -> list[dict]:
        return [
            {
                "chembl_id": r.chembl_id,
                "pref_name": r.pref_name,
                "organism": r.organism,
                "target_type": r.target_type,
            }
            for r in records
        ]

    @staticmethod
    def _deserialize(data: list[dict]) -> List[ChemBLTargetRecord]:
        out: List[ChemBLTargetRecord] = []
        for d in data or []:
            out.append(
                ChemBLTargetRecord(
                    chembl_id=(d.get("chembl_id") or "").strip(),
                    pref_name=(d.get("pref_name") or "").strip(),
                    organism=(d.get("organism") or "").strip(),
                    target_type=(d.get("target_type") or "").strip(),
                )
            )
        return out

    def search(self, query: str, limit: int = 50) -> List[ChemBLTargetRecord]:
        q = (query or "").strip()
        if not q:
            return []

        limit = int(limit)
        cache_key = f"targets.search:q={q}|limit={limit}"

        # disk cache
        if self.cache_policy.enabled:
            cached = self.cache.get(cache_key, ttl_s=self.cache_policy.ttl_s)
            if cached is not None:
                return self._deserialize(cached)

        url = f"{self.BASE}/target/search.json"

        # Pagination: follow page_meta.next if present
        collected: List[ChemBLTargetRecord] = []
        next_url: Optional[str] = url
        params = {"q": q, "limit": min(limit, 1000)}  # ChEMBL limit cap varies; keep safe

        while next_url and len(collected) < limit:
            r = self._request(next_url, params=params)
            payload = r.json() or {}

            raw = payload.get("targets") or []
            for item in raw:
                tid = (item.get("target_chembl_id") or "").strip()
                if not tid:
                    continue
                collected.append(
                    ChemBLTargetRecord(
                        chembl_id=tid,
                        pref_name=(item.get("pref_name") or item.get("target_pref_name") or "").strip(),
                        organism=(item.get("organism") or item.get("target_organism") or "").strip(),
                        target_type=(item.get("target_type") or "").strip(),
                    )
                )
                if len(collected) >= limit:
                    break

            # next link
            page_meta = payload.get("page_meta") or {}
            nxt = page_meta.get("next") or payload.get("next")
            if nxt:
                next_url = urljoin(self.BASE + "/", str(nxt))
                params = None  # next already includes params
            else:
                next_url = None

        # de-dup stable
        seen = set()
        uniq: List[ChemBLTargetRecord] = []
        for t in collected:
            if t.chembl_id in seen:
                continue
            seen.add(t.chembl_id)
            uniq.append(t)

        if self.cache_policy.enabled:
            self.cache.set(cache_key, self._serialize(uniq))
        return uniq

    def get(self, target_chembl_id: str) -> Optional[ChemBLTargetRecord]:
        tid = (target_chembl_id or "").strip().upper()
        if not tid:
            return None

        cache_key = f"targets.get:{tid}"
        if self.cache_policy.enabled:
            cached = self.cache.get(cache_key, ttl_s=self.cache_policy.ttl_s)
            if cached is not None:
                recs = self._deserialize(cached)
                return recs[0] if recs else None

        url = f"{self.BASE}/target/{tid}.json"
        r = self._request(url)
        if r.status_code == 404:
            return None

        item = r.json() or {}
        chembl_id = (item.get("target_chembl_id") or "").strip()
        if not chembl_id:
            return None

        rec = ChemBLTargetRecord(
            chembl_id=chembl_id,
            pref_name=(item.get("pref_name") or "").strip(),
            organism=(item.get("organism") or "").strip(),
            target_type=(item.get("target_type") or "").strip(),
        )

        if self.cache_policy.enabled:
            self.cache.set(cache_key, self._serialize([rec]))
        return rec
