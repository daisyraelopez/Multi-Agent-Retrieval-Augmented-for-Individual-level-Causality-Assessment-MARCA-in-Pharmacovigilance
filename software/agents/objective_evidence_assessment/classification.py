"""Agent 3 — is the adverse event a clinical *sign* or a *symptom*? (Naranjo Q10)

Sign = objective, observable/measurable (value 1). Symptom = subjective, patient
reported (value 0). Psychiatric disorders are all classified as symptoms.
"""
from __future__ import annotations

import time

from software import llm

SYSTEM = ("You are a clinical pharmacovigilance expert. "
          "Classify events as 'Sign' (Value 1) or 'Symptom' (Value 0).")

PROMPT = """
Analyze the medical term: "{ae}"

Classify as:
- SIGN: Objective evidence (observable/measurable). Value = 1.
- SYMPTOM: Subjective experience (reported by patient). Value = 0.
- SYMPTOM: Psychiatric disorders are all symptoms. Value = 0.

Return JSON:
{{
  "ae": "{ae}",
  "classification": "Sign" or "Symptom",
  "value": 1 or 0,
  "rationale": "Brief medical explanation"
}}
"""


def classify_ae(ae: str) -> dict:
    """Return ``{"ae", "classification", "value", "rationale"}`` for one adverse event."""
    try:
        return llm.chat_json(PROMPT.format(ae=ae), SYSTEM)
    except Exception as e:  # noqa: BLE001
        return {"ae": ae, "classification": None, "value": None, "rationale": str(e)}


def classify_aes(aes: list, pause: float = 0.2) -> list:
    results = []
    for ae in aes:
        print(f"Analyzing: {ae}")
        results.append(classify_ae(ae))
        time.sleep(pause)
    return results
