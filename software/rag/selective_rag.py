"""Selective RAG for Naranjo Q2 (temporal plausibility), Q5 (confounding) and Q10
(sign vs symptom).

For every drug–adverse-event pair the LLM answers each question twice: once with
the case data only (``version_no_rag``) and once with expert-assessed reference cases
retrieved from the knowledge base (``version_with_rag``). The case being assessed is
never retrieved as its own reference (leave-one-out).
"""
from __future__ import annotations

from pathlib import Path

from software import config, llm
from software.drug_class_resolution import build_class_map
from software.rag import prompts
from software.rag.knowledge_base import (
    has_any_confounder_info,
    has_any_date,
    kb_drug_names,
    load_expert_kb,
)
from software.rag.vector_db import ExpertVectorDB

QUESTIONS = ("Q2", "Q5", "Q10")


def _call_llm(prompt: str) -> str:
    """LLM call that never raises: a failure is returned as text so that one bad call
    does not abort the whole case."""
    try:
        return llm.chat(prompt, prompts.SYSTEM_MESSAGE, timeout=60)
    except Exception as e:  # noqa: BLE001
        return f"[LLM error: {e}]"


class SelectiveRAG:
    """Expert KB + index + the dual (no-RAG / RAG) answering procedure."""

    def __init__(self, records: list):
        if not records:
            raise ValueError("The expert knowledge base contains no usable records.")
        self.records = records
        self.db = ExpertVectorDB(records)
        self.results: list = []   # one entry per assessed pair (cleared with reset())

    @classmethod
    def from_workbook(cls, path: Path | str | None = None,
                      cache_path: Path | str | None = None) -> "SelectiveRAG":
        """Build the index from ``expert_kb.xlsx``.

        Drug classes are resolved (OpenFDA → RxNorm → LLM) and cached on disk; only
        drugs missing from the cache trigger API calls.
        """
        path = Path(path or config.KB_PATH)
        if not path.exists():
            raise FileNotFoundError(f"Expert knowledge base not found: {path}")
        class_map = build_class_map(kb_drug_names(path), cache_path or config.DRUG_CLASS_CACHE)
        records = load_expert_kb(path, class_map)
        rag = cls(records)
        print(f"[RAG] Ready: {len(records)} records | "
              f"{len({r['drug'] for r in records})} drugs | "
              f"{len({r['drug_class'] for r in records})} classes")
        print("      Retrieval: same drug (T1) → same class (T2) → global (fallback)")
        return rag

    def reset(self) -> None:
        self.results = []

    # ── retrieval + answering for one question ────────────────────────────────
    def _dual_answer(self, qname: str, drug: str, pt: str, context: str, case_id: str = ""):
        db = self.db
        case_block = f"Drug: {drug}\nAdverse Event: {pt}\nContext:\n{context[:1400]}"
        no_dates = not has_any_date(context)
        no_conf_info = not has_any_confounder_info(context)
        expert_exact = db.lookup_exact(case_id)

        if qname == "Q10":
            similar = db.query_ae(pt, top_k=2, exclude_id=case_id)
        elif qname == "Q2" and no_dates:
            similar = db.query_missing_dates(drug, pt, top_k=3, exclude_id=case_id)
        else:
            similar = db.query_two_tier(drug, pt, top_k=3, exclude_id=case_id)
            if qname == "Q5" and no_conf_info:   # prefer refs that also lacked confounder info
                pref = [r for r in similar if r.get("no_confounders")]
                rest = [r for r in similar if not r.get("no_confounders")]
                similar = (pref + rest)[:3]

        base_prompt = (
            f"Answer the following pharmacovigilance question.\n\n"
            f"Question:\n{prompts.Q_DEF[qname]}\n\n"
            f"Case data:\n{case_block}\n\n"
        )

        ans_no_rag = _call_llm(base_prompt + prompts.REPLY_FORMAT)

        rag_parts = []
        if qname == "Q2" and no_dates:
            rag_parts.append(prompts.RULE_Q2_NO_DATES)
        if qname == "Q5" and no_conf_info:
            rag_parts.append(prompts.RULE_Q5_NO_CONFOUNDER_INFO)
        ref_block = prompts.build_ref_block(similar, qname)
        if ref_block:
            rag_parts.append(
                f"--- Expert Reference Cases (annotated KB) ---\n{ref_block}\n"
                "--- End References ---\n"
                "Use these to calibrate your answer where relevant."
            )

        if rag_parts:
            ans_with_rag = _call_llm(
                base_prompt + "\n\n".join(rag_parts) + "\n\n" + prompts.REPLY_FORMAT)
        else:
            ans_with_rag = ans_no_rag + "\n[No expert context available]"

        return ans_no_rag, ans_with_rag, similar, expert_exact

    # ── public API ────────────────────────────────────────────────────────────
    def run_for_pair(self, drug: str, pt: str, context_text: str, case_id: str = "") -> dict:
        """Answer Q2, Q5 and Q10 (no-RAG vs RAG) for one drug–adverse-event pair."""
        entry = {"drug": drug, "pt": pt, "case_id": case_id}
        for qname in QUESTIONS:
            no_rag, with_rag, refs, exact = self._dual_answer(
                qname, drug, pt, context_text, case_id)
            entry[qname] = {
                "version_no_rag": no_rag,
                "version_with_rag": with_rag,
                "retrieved_refs": refs,
                "expert_exact": {
                    "case_id": exact["case_id"],
                    "h_score": exact.get(f"{qname}_h_score"),
                    "h_reason": exact.get(f"{qname}_h_reason", ""),
                } if exact else None,
            }
        self.results.append(entry)
        return entry
