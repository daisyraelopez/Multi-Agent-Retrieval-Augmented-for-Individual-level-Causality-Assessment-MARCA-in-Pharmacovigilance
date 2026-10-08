"""Agent 2 — is the adverse event listed in the drug label? (Naranjo Q1)

If the event is on the label's adverse-event list, the status is ``Listed in label``.
Otherwise the LLM judges whether the event is medically related to the drug and the
status becomes ``Not listed (Yes|No|Unclear)``.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

from software import config, llm

SYSTEM = "You are a medical documentation assistant that evaluates AE-drug relations."

STATUS_LISTED = "Listed in label"
STATUS_NO_LABEL = "No label list available"
STATUS_ERROR = "Error during LLM call"


def _norm(name: str) -> str:
    return re.sub(r"[\s_]+", "", name.lower())


def find_label_file(drug_name: str, labels_dir: Path | str | None = None) -> Path | None:
    """CSV in ``labels_dir`` whose file name contains the drug name (case/space-insensitive)."""
    labels_dir = Path(labels_dir or config.LABELS_DIR)
    if not labels_dir.is_dir() or not drug_name.strip():
        return None
    wanted = _norm(drug_name)
    for f in sorted(labels_dir.glob("*.csv")):
        if wanted in _norm(f.stem):
            return f
    return None


def load_ae_list_for_drug(drug_name: str, labels_dir: Path | str | None = None) -> list:
    """Lower-cased adverse-event names from the drug's label CSV (header row skipped)."""
    path = find_label_file(drug_name, labels_dir)
    if path is None:
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        return [row[0].strip().lower() for row in reader if row and row[0].strip()]


def assess_listedness(drug: str, ae: str, label_aes: list | None) -> dict:
    """Return ``{"drug", "ae", "status", "explanation"}`` for one pair.

    ``label_aes`` is the lower-cased label list, or ``None`` / empty if the drug has no
    label list (status ``No label list available``; the LLM is not called).
    """
    if not label_aes:
        return {"drug": drug, "ae": ae, "status": STATUS_NO_LABEL, "explanation": ""}

    if ae.strip().lower() in label_aes:
        return {"drug": drug, "ae": ae, "status": STATUS_LISTED, "explanation": ""}

    prompt = f"""
Determine if the following adverse event is medically related to the drug based on clinical reasoning.

Drug: {drug}
Adverse Event: {ae}

Output in this format:
Related - start: <Yes/No/Unclear> end.
Explanation - start: <Brief medical explanation> end.
"""
    try:
        response = llm.chat(prompt, SYSTEM)
        rel = re.search(r"Related - start:\s*(.+?)\s*end\.", response, re.DOTALL)
        relation = rel.group(1).strip() if rel else "Unclear"
        exp = re.search(r"Explanation - start:\s*(.+?)\s*end\.", response, re.DOTALL)
        explanation = exp.group(1).strip() if exp else response
        return {"drug": drug, "ae": ae, "status": f"Not listed ({relation})",
                "explanation": explanation}
    except Exception as e:  # noqa: BLE001
        return {"drug": drug, "ae": ae, "status": STATUS_ERROR, "explanation": str(e)}
