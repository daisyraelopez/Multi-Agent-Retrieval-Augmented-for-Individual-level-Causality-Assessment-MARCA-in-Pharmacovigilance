"""Build the case context (patient metadata, drug-specific data, concomitant drugs)
that Agent 4 and the selective RAG receive for one drug–adverse-event pair."""
from __future__ import annotations

import pandas as pd

_MISSING_TOKENS = ("NA", "UNK", "N/A", "")


def value_or_missing(value) -> str:
    if pd.isna(value):
        return "MISSING"
    s = str(value).strip()
    return "MISSING" if s.upper() in _MISSING_TOKENS else s


def extract_case_metadata(df_case: pd.DataFrame) -> dict:
    if df_case.empty:
        return {k: "MISSING" for k in ("age", "sex", "weight", "history_similar_reaction",
                                       "outcome", "reporter_country", "edss")}
    row = df_case.iloc[0]
    return {
        "age": value_or_missing(row.get("Age", "MISSING")),
        "sex": value_or_missing(row.get("Sex", "MISSING")),
        "weight": value_or_missing(row.get("Weight", "MISSING")),
        "history_similar_reaction": value_or_missing(row.get("History of Similar Reaction", "MISSING")),
        "outcome": value_or_missing(row.get("Outcome Code", "MISSING")),
        "reporter_country": value_or_missing(row.get("Reporter Country", "MISSING")),
        "edss": value_or_missing(row.get("EDSS", "MISSING")),
    }


def extract_drug_map(df_drugs: pd.DataFrame) -> dict:
    """``{drug name: {start, end, dechal, rechal, route, dose, indication}}``."""
    out: dict = {}
    if df_drugs.empty:
        return out
    for _, row in df_drugs.iterrows():
        name = value_or_missing(row.get("Drug Name", "MISSING"))
        if name == "MISSING":
            continue
        out[name] = {
            "start": value_or_missing(row.get("Therapy Start Date", "MISSING")),
            "end": value_or_missing(row.get("Therapy End Date", "MISSING")),
            "dechal": value_or_missing(row.get("Dechal", "MISSING")),
            "rechal": value_or_missing(row.get("Rechal", "MISSING")),
            "route": value_or_missing(row.get("Route", "MISSING")),
            "dose": value_or_missing(row.get("Dose VBM", "MISSING")),
            "indication": value_or_missing(row.get("Indication for Use", "MISSING")),
        }
    return out


def unique_drugs_and_events(df_drugs: pd.DataFrame, df_reactions: pd.DataFrame):
    """Distinct drug names and adverse-event terms of the case (missing values removed)."""
    drugs, events = [], []
    if not df_drugs.empty and "Drug Name" in df_drugs.columns:
        drugs = [d for d in df_drugs["Drug Name"].dropna().unique().tolist()
                 if str(d).upper() not in _MISSING_TOKENS]
    if not df_reactions.empty and "Adverse Reaction" in df_reactions.columns:
        events = [e for e in df_reactions["Adverse Reaction"].dropna().unique().tolist()
                  if str(e).upper() not in _MISSING_TOKENS]
    return drugs, events


def build_pair_context(drug: str, df_case: pd.DataFrame, drug_map: dict,
                       metadata: dict) -> str:
    """Text block describing the patient, the suspect drug and the concomitant drugs."""
    event_date = "MISSING"
    if not df_case.empty:
        event_date = value_or_missing(df_case.iloc[0].get("Event Date", "MISSING"))

    factors = drug_map.get(drug, {})
    concurrent = []
    for other, f in drug_map.items():
        if other != drug:
            concurrent.append(
                f"- **Drug Name:** {other}\n"
                f"  - **Indication:** {f.get('indication', 'MISSING')}\n"
                f"  - **Therapy Period:** {f.get('start', 'MISSING')} to {f.get('end', 'MISSING')}\n"
                f"  - **Dechal/Rechal:** Dechal: {f.get('dechal', 'MISSING')}, "
                f"Rechal: {f.get('rechal', 'MISSING')}"
            )
    concurrent_text = "\n".join(concurrent) or "N/A: single drug"

    return f"""
**PATIENT AND CASE METADATA:**
- Age: {metadata['age']}
- Sex: {metadata['sex']}
- Weight: {metadata['weight']}
- Outcome Code: {metadata['outcome']}
- History of Similar Reactions: {metadata['history_similar_reaction']}
- Reporter Country: {metadata['reporter_country']}
- EDSS: {metadata['edss']}

**DRUG-SPECIFIC INFO:**
- Event Date: {event_date}
- Therapy Start: {factors.get('start', 'MISSING')}
- Therapy End: {factors.get('end', 'MISSING')}
- Route: {factors.get('route', 'MISSING')}
- Dose: {factors.get('dose', 'MISSING')}
- Indication: {factors.get('indication', 'MISSING')}
- Dechallenge Result: {factors.get('dechal', 'MISSING')}
- Rechallenge Result: {factors.get('rechal', 'MISSING')}

**Concurrent Medications:**
{concurrent_text}
"""
