from __future__ import annotations

import time
from typing import List, Optional
from urllib.parse import urljoin

import requests
from requests import Response

from .chembl_models import ChemBLAssayRecord
from .disk_cache import CachePolicy, DiskCache


class ChemBLAssayService:
    """Assay browse for a target via ChEMBL REST API with retry + pagination + disk cache."""

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

        raise RuntimeError(f"ChEMBL assay request failed after retries: {last_exc}") from last_exc

    @staticmethod
    def _serialize(records: List[ChemBLAssayRecord]) -> list[dict]:
        return [
            {
                "assay_chembl_id": r.assay_chembl_id,
                "description": r.description,
                "assay_type": r.assay_type,
                "confidence_score": r.confidence_score,
                "organism": r.organism,
            }
            for r in records
        ]

    @staticmethod
    def _deserialize(data: list[dict]) -> List[ChemBLAssayRecord]:
        out: List[ChemBLAssayRecord] = []
        for d in data or []:
            out.append(
                ChemBLAssayRecord(
                    assay_chembl_id=(d.get("assay_chembl_id") or "").strip(),
                    description=(d.get("description") or "").strip(),
                    assay_type=(d.get("assay_type") or "").strip(),
                    confidence_score=_to_int(d.get("confidence_score")),
                    organism=(d.get("organism") or "").strip(),
                )
            )
        return out

    def fetch_for_target(
        self,
        target_chembl_id: str,
        min_confidence: int = 7,
        assay_type: str = "ANY",
        limit: int = 1000,
        page_size: int = 200,
    ) -> List[ChemBLAssayRecord]:
        tid = (target_chembl_id or "").strip().upper()
        if not tid:
            return []

        at = (assay_type or "ANY").strip().upper()
        if at not in {"ANY", "B", "F", "A"}:
            at = "ANY"

        limit = int(limit)
        page_size = max(1, min(int(page_size), 1000))

        cache_key = f"assays:tid={tid}|minconf={int(min_confidence)}|type={at}|limit={limit}"
        if self.cache_policy.enabled:
            cached = self.cache.get(cache_key, ttl_s=self.cache_policy.ttl_s)
            if cached is not None:
                return self._deserialize(cached)

        url = f"{self.BASE}/assay.json"

        collected: List[ChemBLAssayRecord] = []
        next_url: Optional[str] = url
        params = {"target_chembl_id": tid, "limit": page_size}

        while next_url and len(collected) < limit:
            r = self._request(next_url, params=params)
            payload = r.json() or {}
            raw = payload.get("assays") or []

            for item in raw:
                aid = (item.get("assay_chembl_id") or "").strip()
                if not aid:
                    continue

                conf = _to_int(item.get("confidence_score"))
                a_type = (item.get("assay_type") or "").strip().upper()
                org = (item.get("assay_organism") or item.get("organism") or "").strip()
                desc = (item.get("description") or "").strip()

                # filters
                if conf is not None and conf < int(min_confidence):
                    continue
                if at != "ANY" and a_type != at:
                    continue

                collected.append(
                    ChemBLAssayRecord(
                        assay_chembl_id=aid,
                        description=desc,
                        assay_type=a_type,
                        confidence_score=conf,
                        organism=org,
                    )
                )
                if len(collected) >= limit:
                    break

            page_meta = payload.get("page_meta") or {}
            nxt = page_meta.get("next") or payload.get("next")
            if nxt:
                next_url = urljoin(self.BASE + "/", str(nxt))
                params = None
            else:
                next_url = None

        # stable de-dup
        seen = set()
        uniq: List[ChemBLAssayRecord] = []
        for a in collected:
            if a.assay_chembl_id in seen:
                continue
            seen.add(a.assay_chembl_id)
            uniq.append(a)

        if self.cache_policy.enabled:
            self.cache.set(cache_key, self._serialize(uniq))
        return uniq


def _to_int(x) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None
