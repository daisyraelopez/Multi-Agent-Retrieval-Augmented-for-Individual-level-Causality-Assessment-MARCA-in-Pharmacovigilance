"""TF-IDF index over the expert KB with two-tier (same drug → same class) retrieval."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MIN_SIMILARITY = 0.01


class ExpertVectorDB:
    """Flat TF-IDF index plus per-drug and per-class indices.

    Every retrieval method returns record dicts extended with ``_sim`` (cosine
    similarity) and ``_sim_tier`` (``exact_drug``, ``same_class``, ``global``,
    ``ae_match`` or ``no_date_pool``).
    """

    def __init__(self, records: list):
        self.records = records
        if not records:
            return

        self._by_id = {r["case_id"]: r for r in records}

        # flat indices over all records
        self._ae_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self._ae_mat = self._ae_vec.fit_transform([r["pt"].lower() for r in records])
        self._ctx_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=5000)
        self._ctx_mat = self._ctx_vec.fit_transform([
            f"{r['drug'].lower()} {r['pt'].lower()} {r['narrative'][:400].lower()}"
            for r in records
        ])

        # per-drug (tier 1) and per-class (tier 2) indices, searched by adverse event
        self._drug_recs = defaultdict(list)
        self._class_recs = defaultdict(list)
        for r in records:
            self._drug_recs[r["drug"].upper()].append(r)
            if r["drug_class"] != "UNKNOWN":
                self._class_recs[r["drug_class"]].append(r)

        self._drug_idx = {}
        for key, recs in self._drug_recs.items():
            vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
            self._drug_idx[key] = (vec, vec.fit_transform([r["pt"].lower() for r in recs]))
        self._class_idx = {}
        for key, recs in self._class_recs.items():
            vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
            self._class_idx[key] = (vec, vec.fit_transform([r["pt"].lower() for r in recs]))

        # sub-pool of records without any date (for Q2)
        self._no_date_records = [r for r in records if r["no_dates"]]
        if self._no_date_records:
            self._nd_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
            self._nd_mat = self._nd_vec.fit_transform(
                [f"{r['drug'].lower()} {r['pt'].lower()}" for r in self._no_date_records])
        else:
            self._nd_vec = self._nd_mat = None

        print(f"[VectorDB] Built — {len(self._drug_idx)} drug buckets, "
              f"{len(self._class_idx)} class buckets, "
              f"{len(self._no_date_records)} no-date records.")

    # ── shared scorer ─────────────────────────────────────────────────────────
    @staticmethod
    def _query(mat, vec, pool: list, text: str, top_k: int, exclude_id: str) -> list:
        sims = cosine_similarity(vec.transform([text.lower()]), mat).flatten()
        out = []
        for i in np.argsort(sims)[::-1]:
            r = pool[i]
            if exclude_id and r["case_id"] == exclude_id:
                continue
            if sims[i] < MIN_SIMILARITY:
                break
            out.append({**r, "_sim": float(sims[i])})
            if len(out) >= top_k:
                break
        return out

    # ── public retrieval methods ──────────────────────────────────────────────
    def lookup_exact(self, case_id) -> dict | None:
        """The KB record of exactly this case (used for the head-to-head comparison)."""
        if not self.records:
            return None
        return self._by_id.get(str(case_id).strip())

    def query_two_tier(self, drug: str, pt: str, top_k: int = 3,
                       exclude_id: str = "") -> list:
        """Tier 1: same drug. Tier 2: same drug class. Fallback: global search."""
        drug_key = drug.strip().upper()
        drug_class = "UNKNOWN"
        for r in self.records:
            if r["drug"].upper() == drug_key:
                drug_class = r["drug_class"]
                break

        results, seen = [], set()

        if drug_key in self._drug_idx:
            vec, mat = self._drug_idx[drug_key]
            for r in self._query(mat, vec, self._drug_recs[drug_key], pt, top_k, exclude_id):
                if r["case_id"] not in seen:
                    results.append({**r, "_sim_tier": "exact_drug"})
                    seen.add(r["case_id"])

        if len(results) < top_k and drug_class not in ("UNKNOWN", "", None):
            if drug_class in self._class_idx:
                vec, mat = self._class_idx[drug_class]
                for r in self._query(mat, vec, self._class_recs[drug_class], pt,
                                     top_k * 3, exclude_id):
                    if r["case_id"] not in seen:
                        results.append({**r, "_sim_tier": "same_class"})
                        seen.add(r["case_id"])
                    if len(results) >= top_k:
                        break

        if not results:
            for r in self._query(self._ctx_mat, self._ctx_vec, self.records,
                                 f"{drug} {pt}", top_k, exclude_id):
                if r["case_id"] not in seen:
                    results.append({**r, "_sim_tier": "global"})
                    seen.add(r["case_id"])

        return results[:top_k]

    def query_ae(self, pt: str, top_k: int = 2, exclude_id: str = "") -> list:
        """Search by adverse event only (used for Q10, sign vs symptom)."""
        return [{**r, "_sim_tier": "ae_match"}
                for r in self._query(self._ae_mat, self._ae_vec, self.records,
                                     pt, top_k, exclude_id)]

    def query_missing_dates(self, drug: str, pt: str, top_k: int = 3,
                            exclude_id: str = "") -> list:
        """Search only among KB records that also lack therapy/event dates (Q2)."""
        if self._nd_mat is None or not self._no_date_records:
            return []
        return [{**r, "_sim_tier": "no_date_pool"}
                for r in self._query(self._nd_mat, self._nd_vec, self._no_date_records,
                                     f"{drug} {pt}", top_k, exclude_id)]
