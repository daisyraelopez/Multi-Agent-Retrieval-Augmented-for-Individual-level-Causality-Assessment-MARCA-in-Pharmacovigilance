"""PubMed retrieval (NCBI E-utilities) for one drug–adverse-event pair."""
from __future__ import annotations

import re
import time
from pathlib import Path
from xml.etree import ElementTree as ET

import requests

from software import config

ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def safe_filename(name) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(name).replace("/", "-")).strip("_")[:120]


def search_pubmed(drug_name: str, adr_term: str, max_results: int = 3) -> list:
    """Return up to ``max_results`` articles as ``{"pmid", "title", "abstract"}`` dicts."""
    query = f"{drug_name} AND {adr_term} adverse event"
    print(f"🔍 PubMed: '{query}'")

    try:
        r = requests.get(ESEARCH, params={"db": "pubmed", "term": query,
                                          "retmax": max_results, "retmode": "json"},
                         timeout=30)
        r.raise_for_status()
        ids = r.json().get("esearchresult", {}).get("idlist", [])
    except Exception as e:
        print(f"❌ PubMed search failed: {e}")
        return []
    if not ids:
        return []

    time.sleep(config.PUBMED_PAUSE)
    try:
        r = requests.get(EFETCH, params={"db": "pubmed", "id": ",".join(ids),
                                         "retmode": "xml"}, timeout=30)
        r.raise_for_status()
        root = ET.fromstring(r.text)
    except Exception as e:
        print(f"❌ PubMed fetch failed: {e}")
        return []

    results = []
    for article in root.findall(".//PubmedArticle"):
        parts = [t.text for t in article.findall(".//AbstractText") if t.text]
        results.append({
            "pmid": article.findtext(".//PMID"),
            "title": article.findtext(".//ArticleTitle"),
            "abstract": " ".join(parts) if parts else "No Abstract Available.",
        })
    time.sleep(config.PUBMED_PAUSE)
    return results


def format_literature(drug_name: str, adr_term: str, evidence: list) -> str:
    """Plain-text rendering of the retrieved articles (also fed to the LLM agent)."""
    lines = [
        f"--- Literature Search Results for: {drug_name} AND {adr_term} ---",
        f"Date Generated: {time.ctime()}",
        f"Articles Found: {len(evidence)}",
    ]
    for i, paper in enumerate(evidence, 1):
        lines.append(f"\n{'=' * 50}")
        lines.append(f"ARTICLE {i}")
        lines.append(f"PMID: {paper.get('pmid', 'N/A')}")
        lines.append(f"Title: {paper.get('title', 'N/A')}")
        lines.append(f"Abstract:\n{paper.get('abstract', 'No Abstract Available.')}")
    return "\n".join(lines)


def save_literature(drug_name: str, adr_term: str, evidence: list,
                    out_dir: Path | str | None = None) -> tuple:
    """Write the literature text to ``output/logs/pubmed``. Returns ``(path, text)``."""
    out_dir = Path(out_dir or config.PUBMED_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    text = format_literature(drug_name, adr_term, evidence)
    path = out_dir / f"{safe_filename(drug_name)}__{safe_filename(adr_term)}.txt"
    path.write_text(text, encoding="utf-8")
    return path, text
