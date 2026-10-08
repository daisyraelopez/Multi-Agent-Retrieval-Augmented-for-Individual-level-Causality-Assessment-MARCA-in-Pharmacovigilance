"""Drug → pharmacological class resolution: OpenFDA → RxNorm → LLM, with a disk cache.

Only drugs that are not yet in the cache trigger API calls.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import requests

from software import config, llm

BATCH_SIZE = 30


def normalise_class(raw: str) -> str:
    """``"Kinin B2 Receptor Antagonist [EPC]"`` → ``"KININ_B2_RECEPTOR_ANTAGONIST"``."""
    s = re.sub(r"\s*\[EPC\]", "", raw).strip()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", "_", s)
    return s.upper() or "UNKNOWN"


def openfda_class(drug_name: str):
    """Established pharmacologic class (EPC) from the OpenFDA label endpoint."""
    primary = drug_name.split("\\")[0].strip().lower()
    names = ([drug_name.lower()] if drug_name.lower() != primary else []) + [primary]
    for name in names:
        for field in ("openfda.brand_name", "openfda.substance_name", "openfda.generic_name"):
            try:
                r = requests.get("https://api.fda.gov/drug/label.json",
                                 params={"search": f'{field}:"{name}"', "limit": 1},
                                 timeout=10)
                if r.status_code != 200:
                    continue
                epc = (r.json().get("results", [{}])[0]
                        .get("openfda", {}).get("pharm_class_epc", []))
                if epc:
                    return normalise_class(epc[0])
            except Exception:
                pass
    return None


def rxnorm_class(drug_name: str):
    """Class from RxNorm/RxClass (ATC, then VA, then MeSH)."""
    primary = drug_name.split("\\")[0].strip().lower()
    try:
        r = requests.get("https://rxnav.nlm.nih.gov/REST/rxcui.json",
                         params={"name": primary, "search": 2}, timeout=10)
        if r.status_code != 200:
            return None
        cuis = r.json().get("idGroup", {}).get("rxnormId", [])
        if not cuis:
            return None
        for source in ("ATC", "VA", "MESH"):
            r2 = requests.get("https://rxnav.nlm.nih.gov/REST/rxclass/class/byRxcui.json",
                              params={"rxcui": cuis[0], "relaSource": source}, timeout=10)
            items = (r2.json().get("rxclassDrugInfoList", {}).get("rxclassDrugInfo", [])
                     if r2.status_code == 200 else [])
            if items:
                best = max(items, key=lambda x: len(x["rxclassMinConceptItem"]["classId"]))
                return normalise_class(best["rxclassMinConceptItem"]["className"])
    except Exception:
        pass
    return None


def llm_classes(drugs: list) -> dict:
    """Ask the LLM for the class of each drug (last-resort fallback)."""
    if not drugs:
        return {}
    prompt = (
        "You are a senior pharmacologist. For each drug below return its "
        "pharmacological class as a concise uppercase label with underscores "
        "(e.g. SULFONYLUREA, BRAF_INHIBITOR, ANTI_TNF_BIOLOGIC, INSULIN, SSRI).\n"
        "For combination products use the primary active component class.\n"
        "Return ONLY a JSON object mapping each exact drug name to its class.\n\n"
        "Drugs:\n" + "\n".join(f"- {d}" for d in drugs)
    )
    try:
        parsed = llm.chat_json(prompt, "Return only valid JSON, no other text.", timeout=90)
        return {k: normalise_class(str(v)) for k, v in parsed.items() if k in drugs}
    except Exception as e:
        print(f"    ⚠️  LLM drug-class batch error: {e}")
        return {}


def build_class_map(unique_drugs: list, cache_path: Path | str | None = None) -> dict:
    """Resolve ``drug → class`` for all ``unique_drugs`` and return the full cache.

    Keys of the returned dict are upper-case drug names. Unresolvable drugs are
    stored as ``"UNKNOWN"``.
    """
    cache_path = Path(cache_path or config.DRUG_CLASS_CACHE)
    cache: dict = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        to_resolve = [d for d in unique_drugs if d.upper() not in cache]
        if not to_resolve:
            print(f"[DrugClass] All {len(unique_drugs)} drugs found in cache.")
            return cache
        print(f"[DrugClass] Cache hit for {len(unique_drugs) - len(to_resolve)}/"
              f"{len(unique_drugs)} drugs. Resolving {len(to_resolve)} new ones…")
    else:
        to_resolve = list(unique_drugs)
        print(f"[DrugClass] No cache. Resolving {len(to_resolve)} drugs…")

    unresolved, hits1 = [], 0
    print("  [1/3] OpenFDA…")
    for i, drug in enumerate(to_resolve, 1):
        cls = openfda_class(drug)
        if cls:
            cache[drug.upper()] = cls
            hits1 += 1
        else:
            unresolved.append(drug)
        if i % 20 == 0 or i == len(to_resolve):
            print(f"    {i}/{len(to_resolve)}  ({hits1} resolved so far)")
        time.sleep(config.OPENFDA_PAUSE)
    print(f"  OpenFDA: {hits1} resolved, {len(unresolved)} remain.")

    still, hits2 = [], 0
    print("  [2/3] RxNorm…")
    for drug in unresolved:
        cls = rxnorm_class(drug)
        if cls:
            cache[drug.upper()] = cls
            hits2 += 1
        else:
            still.append(drug)
        time.sleep(config.RXNORM_PAUSE)
    print(f"  RxNorm: {hits2} resolved, {len(still)} remain.")

    if still:
        hits3 = 0
        print(f"  [3/3] LLM batch ({len(still)} drugs)…")
        for start in range(0, len(still), BATCH_SIZE):
            batch = still[start:start + BATCH_SIZE]
            for drug, cls in llm_classes(batch).items():
                cache[drug.upper()] = cls
                hits3 += 1
            for drug in batch:
                cache.setdefault(drug.upper(), "UNKNOWN")
        print(f"  LLM: {hits3} resolved.")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[DrugClass] Cache saved → {cache_path}")
    return cache
