"""MARCA end-to-end assessment of one FAERS case, including the four agents.

    from software.pipeline import assess_case
    from software.rag import SelectiveRAG

    rag = SelectiveRAG.from_workbook()
    result = assess_case(case_text, rag)

Agent 1 (PubMed evidence, Gemini 2.5 Pro) and Agent 4 (clinical causality, Gemini 2.5 Pro)
run as a CrewAI pair; Agent 2 (label listedness, Q1) and Agent 3 (sign vs symptom, Q10)
use DeepSeek-V3.2. Selective RAG adds Q2, Q5 and Q10 answers with expert reference cases.
"""
import asyncio
import csv
import os
import re
import tempfile
import time
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd
import requests
from rich.console import Console

from . import config
from .config import call_deepseek, call_deepseek_json
from .case_parsing import deduplicate_drugs, process_case

console = Console()

try:
    from crewai import Agent, Task
except ImportError:      # CrewAI is only needed for Agents 1 and 4
    Agent = Task = None


# ════════════════════════════════════════════════════════════════════════════
# Agent 1 — Scientific Evidence Retrieval (PubMed + summary)
# ════════════════════════════════════════════════════════════════════════════





def safe_filename(name):
    return str(name).replace(" ", "_").replace("/", "-").replace(":", "")

def search_pubmed(drug_name, adr_term, max_results=3):
    query = f"{drug_name} AND {adr_term} adverse event"
    console.print(f"\n🔍 Searching PubMed for: '{query}'", style="bold yellow")

    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    search_params = {"db": "pubmed", "term": query, "retmax": max_results, "retmode": "json"}

    try:
        search_resp = requests.get(search_url, params=search_params, timeout=30)
        search_resp.raise_for_status()
        ids = search_resp.json().get("esearchresult", {}).get("idlist", [])
    except Exception as e:
        console.print(f"❌ PubMed search failed: {e}", style="red")
        return []

    if not ids:
        return []

    fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    fetch_params = {"db": "pubmed", "id": ",".join(ids), "retmode": "xml"}

    try:
        fetch_resp = requests.get(fetch_url, params=fetch_params, timeout=30)
        fetch_resp.raise_for_status()
        root = ET.fromstring(fetch_resp.text)
    except Exception as e:
        console.print(f"❌ PubMed fetch failed: {e}", style="red")
        return []

    results = []
    for article in root.findall(".//PubmedArticle"):
        title = article.findtext(".//ArticleTitle")
        abstract_parts = [t.text for t in article.findall(".//AbstractText") if t.text]
        abstract = " ".join(abstract_parts) if abstract_parts else "No Abstract Available."
        pmid = article.findtext(".//PMID")
        results.append({"pmid": pmid, "title": title, "abstract": abstract})

    return results

def save_literature_to_file(drug_name, adr_term, evidence, assessment_type="pubmed", out_dir=None):
    out_dir = Path(out_dir or config.LOGS_DIR / "literature")
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = str(out_dir / f"{safe_filename(drug_name)}_{safe_filename(adr_term)}_{assessment_type}_results.txt")
    output_lines = [
        f"--- Literature Search Results for: {drug_name} AND {adr_term} ---",
        f"Date Generated: {time.ctime()}",
        f"Articles Found: {len(evidence)}"
    ]

    for i, paper in enumerate(evidence, 1):
        output_lines.append(f"\n{'=' * 50}")
        output_lines.append(f"ARTICLE {i}")
        output_lines.append(f"PMID: {paper.get('pmid', 'N/A')}")
        output_lines.append(f"Title: {paper.get('title', 'N/A')}")
        output_lines.append(f"Abstract:\n{paper.get('abstract', 'No Abstract Available.')}")

    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    return filename, "\n".join(output_lines)




def build_literature_agent(drug: str, event: str) -> Agent:
    return Agent(
        role="Expert Medical Searcher and Literature Summarizer",
        goal=f"Analyze PubMed results for '{drug}' and '{event}' to summarize MOA and clinical evidence.",
        backstory="Detailed researcher.",
        llm=config.CREW_LLM,
        verbose=True,
    )


def build_literature_task(agent: Agent, literature_content: str) -> Task:
    return Task(
        description=f"Summarize PubMed content:\n{literature_content}",
        agent=agent,
        expected_output="MOA & Clinical Evidence Summary",
    )


# ════════════════════════════════════════════════════════════════════════════
# Agent 2 — Drug Label Assessment (Naranjo Q1)
# ════════════════════════════════════════════════════════════════════════════


LABEL_SYSTEM = "You are a medical documentation assistant that evaluates AE-drug relations."


def load_ae_list_for_drug(drug_name, labels_dir=None):
    """Load the AE list CSV whose file name contains the drug name (None if there is no file)."""
    search_dir = str(labels_dir or config.LABELS_DIR)
    # compare letters and digits only, so "A\\B" matches the file "A_B.csv"
    drug_search = re.sub(r"[^a-z0-9]", "", drug_name.lower())
    csv_files = sorted(f for f in os.listdir(search_dir) if f.lower().endswith(".csv"))
    base = lambda f: re.sub(r"[^a-z0-9]", "", os.path.splitext(f)[0].lower())
    # exact file-name match first (e.g. XELJANZ.csv before XELJANZ_XR.csv), then partial match
    ordered = [f for f in csv_files if base(f) == drug_search] + \
              [f for f in csv_files if base(f) != drug_search and drug_search in base(f)]
    for file in ordered:
            full_path = os.path.join(search_dir, file)
            with open(full_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader, None)  # skip header (product name)
                return [row[0].strip().lower() for row in reader if row]
    return None   # no label file for this drug


def clean_name(name):
    """Remove unwanted substrings and extra spaces."""
    if not name:
        return ""
    name = name.strip()
    for remove_str in ["Val", "Drug Record"]:
        name = name.replace(remove_str, "")
    return name.strip()


def assess_pair(drug: str, ae: str, label_aes: list) -> dict:
    """Listedness for one drug-event pair."""
    if ae.lower() in label_aes:
        return {"drug": drug, "ae": ae, "status": "Listed in label", "explanation": ""}

    prompt = f"""
Determine if the following adverse event is medically related to the drug based on clinical reasoning.

Drug: {drug}
Adverse Event: {ae}

Output in this format:
Related - start: <Yes/No/Unclear> end.
Explanation - start: <Brief medical explanation> end.
"""
    try:
        response = call_deepseek(prompt, system=LABEL_SYSTEM)
        rel_match = re.search(r"Related - start:\s*(.+?)\s*end\.", response, re.DOTALL)
        relation = rel_match.group(1).strip() if rel_match else "Unclear"
        exp_match = re.search(r"Explanation - start:\s*(.+?)\s*end\.", response, re.DOTALL)
        explanation = exp_match.group(1).strip() if exp_match else response
        return {"drug": drug, "ae": ae, "status": f"Not listed ({relation})", "explanation": explanation}
    except Exception as e:
        return {"drug": drug, "ae": ae, "status": "Error during DeepSeek call", "explanation": str(e)}


def assess_listedness(drugs: list, adverse_events: list, labels_dir=None) -> list:
    """Q1 results for every drug x AE pair. Drugs without a label file are skipped."""
    drugs = list(dict.fromkeys(clean_name(d) for d in drugs if d and d.strip()))
    adverse_events = list(dict.fromkeys(clean_name(a) for a in adverse_events if a and a.strip()))
    results = []
    for drug in drugs:
        label_aes = load_ae_list_for_drug(drug, labels_dir)
        if label_aes is None:
            print(f"⚠️ No label CSV found for {drug}. Skipping.")
            continue
        label_aes = [x.lower() for x in label_aes]
        for ae in adverse_events:
            results.append(assess_pair(drug, ae, label_aes))
    return results


# ════════════════════════════════════════════════════════════════════════════
# Agent 2 — building the per-drug label files
# ════════════════════════════════════════════════════════════════════════════




# ---------------------------
# PDF download
# ---------------------------
def download_pdf(url, output_path):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/115.0.0.0 Safari/537.36"
        )
    }
    with requests.get(url, headers=headers, timeout=30, stream=True) as r:
        r.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


# ---------------------------
# Text extraction (PyPDF2 primary, pymupdf fallback)
# ---------------------------
def extract_text(pdf_path):
    import PyPDF2
    full_text = ""
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            page_texts = [page.extract_text() for page in reader.pages if page.extract_text()]
            full_text = "\n".join(page_texts)
    except Exception as e:
        print(f"⚠️ PyPDF2 extraction error: {e}")

    if len(full_text.strip()) < 200:
        try:
            import fitz
            doc = fitz.open(pdf_path)
            full_text = ""
            for page in doc:
                full_text += page.get_text("text") or ""
            doc.close()
            print("ℹ️ Used pymupdf fallback for text extraction.")
        except Exception as e:
            print(f"⚠️ pymupdf extraction failed: {e}")

    return full_text


# ---------------------------
# DeepSeek steps
# ---------------------------
def extract_adverse_section(full_text):
    """Return only the label's adverse events / adverse reactions section."""
    prompt = (
        "Extract ONLY the full text section from the following drug label text "
        "that discusses 'Adverse Events', 'Adverse Reactions', or similar safety information. "
        "Do not summarize, rewrite, or add commentary. "
        "Return only the raw text of that section."
    )
    return call_deepseek(f"{prompt}\n\n---\n\n{full_text}",
                         system="You are a precise document parser.", timeout=300)


def extract_adverse_events(section_text):
    """Return the list of adverse events mentioned in the section text."""
    prompt = f"""
    You are given the undesirable effects section of a drug product.
    Extract ONLY the list of adverse events mentioned in the text.
    Each adverse event must be listed separately, without duplicates.
    Do not include any explanatory text, frequencies, or formatting, only the adverse events.
    Output them as a plain list, one per line.

    Text:
    {section_text}
    """
    ae_text = call_deepseek(prompt, system="You are an expert assistant that extracts structured data.",
                            timeout=300)
    return [line.strip() for line in ae_text.split("\n") if line.strip()]


def save_to_csv(product_name, adverse_events, output_file):
    """Save the extracted adverse events (first row = product name)."""
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([product_name])
        for ae in adverse_events:
            writer.writerow([ae])
    print(f"✅ CSV file saved: {output_file}")


def build_label_csv(pdf_url_or_path: str, product_name: str, out_dir=None) -> Path:
    """Full pipeline: label PDF -> adverse-event CSV named after the product."""
    out_dir = Path(out_dir or config.LABELS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        if os.path.exists(pdf_url_or_path):
            pdf_path = pdf_url_or_path
        else:
            pdf_path = os.path.join(tmpdir, "file.pdf")
            print(f"📥 Downloading: {pdf_url_or_path}")
            download_pdf(pdf_url_or_path, pdf_path)
        text = extract_text(pdf_path)
    if not text.strip():
        raise ValueError("No text extracted from PDF.")
    section = extract_adverse_section(text)
    aes = extract_adverse_events(section)
    output_file = out_dir / f"{product_name.upper()}.csv"
    save_to_csv(product_name.upper(), aes, output_file)
    return output_file


def safe_label_name(drug: str) -> str:
    """File-system-safe name for a drug (combination products use '_' instead of '\\')."""
    import re
    return re.sub(r"[^A-Za-z0-9]+", "_", drug.strip().upper()).strip("_")


def build_all_label_csvs(sources_csv=None, out_dir=None, overwrite=False):
    """Rebuild every drug's label CSV from data/knowledge_base/drug_label_sources.csv.

    The sources file lists the FDA label PDF used for each drug in the study (taken from
    the Q1 rationale in expert_kb.xlsx). Drugs with more than one label get the union of
    the adverse events from all their labels. Requires DEEPSEEK_API_KEY and internet access.
    """
    import pandas as pd
    sources_csv = Path(sources_csv or config.KB_DIR / "drug_label_sources.csv")
    out_dir = Path(out_dir or config.LABELS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    src = pd.read_csv(sources_csv)
    done, failed = [], []
    for drug, rows in src.groupby("drug"):
        name = safe_label_name(drug)
        target = out_dir / f"{name}.csv"
        if target.exists() and not overwrite:
            continue
        aes = []
        try:
            for url in rows["fda_label_pdf"]:
                with tempfile.TemporaryDirectory() as tmpdir:
                    pdf_path = os.path.join(tmpdir, "label.pdf")
                    download_pdf(url, pdf_path)
                    text = extract_text(pdf_path)
                if text.strip():
                    aes += extract_adverse_events(extract_adverse_section(text))
            aes = list(dict.fromkeys(a for a in aes if a))
            save_to_csv(name, aes, target)
            done.append(drug)
        except Exception as e:
            print(f"❌ {drug}: {e}")
            failed.append(drug)
    print(f"✅ Built {len(done)} label files; {len(failed)} failed.")
    return done, failed


def labels_from_study_workbook(kb_path=None, out_dir=None):
    """Recreate the drug label files from the study's recorded Agent 2 output.

    `q1_ai_answer` in expert_kb.xlsx stores Agent 2's result for every DEC. Events with
    status "Listed in label" were exact matches in the original label file of that drug,
    so they are written to `<DRUG>.csv`. A file is written for every drug (some with no
    events), so Agent 2 assesses every DEC exactly as in the study: listed events are
    matched directly and all others go to DeepSeek.

    These files contain only the label events that occurred in the study's DECs, not the
    complete label. For complete lists use `build_all_label_csvs()`.
    """
    import re
    import pandas as pd
    kb_path = kb_path or config.KB_PATH
    out_dir = Path(out_dir or config.LABELS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_excel(kb_path, sheet_name=0, usecols=["drug_name", "pt", "q1_ai_answer"])
    df["drug"] = df["drug_name"].astype(str).str.strip()
    df["ae"] = df["pt"].astype(str).str.strip()
    df["status"] = df["q1_ai_answer"].astype(str).str.extract(r"status:\s*([^\n]+)")[0].str.strip()
    n_events = 0
    for drug, rows in df.groupby("drug"):
        listed = rows.loc[rows["status"] == "Listed in label", "ae"]
        listed = list(dict.fromkeys(listed))
        save_to_csv(safe_label_name(drug), listed, out_dir / f"{safe_label_name(drug)}.csv")
        n_events += len(listed)
    print(f"✅ {df['drug'].nunique()} label files, {n_events} listed events, from {kb_path}")
    return out_dir


# ════════════════════════════════════════════════════════════════════════════
# Agent 3 — Objective Evidence Assessment (Naranjo Q10)
# ════════════════════════════════════════════════════════════════════════════


SIGN_SYMPTOM_SYSTEM = (
    "You are a clinical pharmacovigilance expert. "
    "Classify events as 'Sign' (Value 1) or 'Symptom' (Value 0)."
)


def unique_events(adverse_events: list) -> list:
    """Remove duplicate AE terms (case-insensitive), keeping their order."""
    seen, out = set(), []
    for ae in adverse_events:
        key = ae.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(ae.strip())
    return out


def classify_aes(aes: list, pause: float = 0.2) -> list:
    results = []
    for ae in unique_events(aes):
        print(f"Analyzing: {ae}")
        prompt = f"""
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
        try:
            results.append(call_deepseek_json(prompt, system=SIGN_SYMPTOM_SYSTEM))
        except Exception as e:
            print(f"Error analyzing {ae}: {e}")
            results.append({"ae": ae, "classification": None, "value": None, "rationale": str(e)})
        time.sleep(pause)
    return results


# ════════════════════════════════════════════════════════════════════════════
# Agent 4 — Clinical Causality Assessment
# ════════════════════════════════════════════════════════════════════════════





def _get_value_or_missing(value):
    if pd.isna(value):
        return "MISSING"
    str_val = str(value).strip()
    return "MISSING" if str_val.upper() in ("NA", "UNK", "N/A", "") else str_val

def _extract_case_metadata(df_case):
    if df_case.empty:
        return {
            "age": "MISSING",
            "sex": "MISSING",
            "weight": "MISSING",
            "history_similar_reaction": "MISSING",
            "outcome": "MISSING",
            "reporter_country": "MISSING",
            "edss": "MISSING",
        }

    row = df_case.iloc[0]
    return {
        "age": _get_value_or_missing(row.get("Age", "MISSING")),
        "sex": _get_value_or_missing(row.get("Sex", "MISSING")),
        "weight": _get_value_or_missing(row.get("Weight", "MISSING")),
        "history_similar_reaction": _get_value_or_missing(row.get("MISSING", "MISSING")),
        "outcome": _get_value_or_missing(row.get("Outcome Code", "MISSING")),
        "reporter_country": _get_value_or_missing(row.get("Reporter Country", "MISSING")),
        "edss": _get_value_or_missing(row.get("EDSS", "MISSING")),
    }

def _extract_drug_causality_map(df_drugs):
    causality_map = {}
    if df_drugs.empty:
        return causality_map

    for _, row in df_drugs.iterrows():
        drug_name = _get_value_or_missing(row.get("Drug Name", "MISSING"))
        if drug_name == "MISSING":
            continue

        causality_map[drug_name] = {
            "start": _get_value_or_missing(row.get("Therapy Start Date", "MISSING")),
            "end": _get_value_or_missing(row.get("Therapy End Date", "MISSING")),
            "dechal": _get_value_or_missing(row.get("Dechal", "MISSING")),
            "rechal": _get_value_or_missing(row.get("Rechal", "MISSING")),
            "route": _get_value_or_missing(row.get("Route", "MISSING")),
            "dose": _get_value_or_missing(row.get("Dose VBM", "MISSING")),
            "indication": _get_value_or_missing(row.get("Indication for Use", "MISSING")),
        }

    return causality_map

def _parse_drugs_and_events(df_drugs, df_reactions):
    unique_drugs = []
    unique_events = []

    if not df_drugs.empty and "Drug Name" in df_drugs.columns:
        unique_drugs = df_drugs["Drug Name"].dropna().unique().tolist()
        unique_drugs = [d for d in unique_drugs if str(d).upper() not in ("NA", "UNK", "N/A", "")]

    if not df_reactions.empty and "Adverse Reaction" in df_reactions.columns:
        unique_events = df_reactions["Adverse Reaction"].dropna().unique().tolist()
        unique_events = [e for e in unique_events if str(e).upper() not in ("NA", "UNK", "N/A", "")]

    drug_event_pairs = [(d, e) for d in unique_drugs for e in unique_events if d and e]
    console.print(
        f"✅ Found {len(unique_drugs)} drugs and {len(unique_events)} events from dataframes.",
        style="green"
    )
    return drug_event_pairs


def build_case_context(drug, case_metadata, drug_causality_map, case_event_date):
    """Build the temporal/causality context text for one drug (as in the notebook)."""
    drug_factors = drug_causality_map.get(drug, {})
    concurrent_drugs_list = []
    for concurrent_drug, factors in drug_causality_map.items():
        if concurrent_drug != drug:
            concurrent_drugs_list.append(
                f"- **Drug Name:** {concurrent_drug}\n"
                f"  - **Indication:** {factors.get('indication', 'MISSING')}\n"
                f"  - **Therapy Period:** {factors.get('start', 'MISSING')} to {factors.get('end', 'MISSING')}\n"
                f"  - **Dechal/Rechal:** Dechal: {factors.get('dechal', 'MISSING')}, Rechal: {factors.get('rechal', 'MISSING')}"
            )
    concurrent_medication_context = "\n".join(concurrent_drugs_list) or "N/A: single drug"

    return f"""
**PATIENT AND CASE METADATA:**
- Age: {case_metadata['age']}
- Sex: {case_metadata['sex']}
- Weight: {case_metadata['weight']}
- Outcome Code: {case_metadata['outcome']}
- History of Similar Reactions: {case_metadata['history_similar_reaction']}
- Reporter Country: {case_metadata['reporter_country']}
- EDSS: {case_metadata['edss']}

**DRUG-SPECIFIC INFO:**
- Event Date: {case_event_date}
- Therapy Start: {drug_factors.get('start', 'MISSING')}
- Therapy End: {drug_factors.get('end', 'MISSING')}
- Route: {drug_factors.get('route', 'MISSING')}
- Dose: {drug_factors.get('dose', 'MISSING')}
- Indication: {drug_factors.get('indication', 'MISSING')}
- Dechallenge Result: {drug_factors.get('dechal', 'MISSING')}
- Rechallenge Result: {drug_factors.get('rechal', 'MISSING')}

**Concurrent Medications:**
{concurrent_medication_context}
"""


def build_safety_assessor(drug: str, event: str) -> Agent:
    return Agent(
        role="Pharmacovigilance Causality Specialist",
        goal=f"Assess causality for '{event}' caused by '{drug}' using metadata and literature summary.",
        backstory="Experienced safety specialist.",
        llm=config.CREW_LLM,
        verbose=True,
    )


def build_assessment_task(safety_assessor: Agent, drug: str, event: str,
                          temporal_causality_context: str) -> Task:
    return Task(
        description=f"""
Use the following context to provide a comprehensive causality assessment for '{drug}' causing '{event}':

{temporal_causality_context}

Instructions for the agent:
- Follow the exact output structure:

**Comprehensive Causality Assessment for <ADVERSE EVENT> Associated with <DRUG>**

**FINAL PLAUSIBILITY RATING:** HIGH / MEDIUM / LOW

**Detailed Causality Analysis:**
1. Temporal Plausibility
2. Biological Plausibility
3. Dechallenge/Rechallenge
4. Objective Evidence
5. Confounding Factors
6. History of Similar Reactions
7. Synthesis and Final Assessment

- Explicitly include Dechallenge/Rechallenge results, literature evidence (PMIDs), and patient metadata.
- Base all reasoning strictly on the provided information.

""",
        agent=safety_assessor,
        expected_output="Comprehensive causality assessment structured as requested",
    )


# ---------------------------
# Auto-fill missing template variables in CrewAI task descriptions
# ---------------------------
_TEMPLATE_VAR_RE = re.compile(r"{([^{}]+)}")

def _find_template_vars(text: str) -> set[str]:
    return set(_TEMPLATE_VAR_RE.findall(text or ""))

def _ensure_required_inputs(tasks, inputs):
    required = set()
    for t in tasks:
        required |= _find_template_vars(getattr(t, "description", ""))

    for key in required:
        if key not in inputs:
            inputs[key] = "MISSING"

    return inputs


# ════════════════════════════════════════════════════════════════════════════
# Pipeline
# ════════════════════════════════════════════════════════════════════════════





async def run_case_report_plausibility_assessment(df_case, df_reactions, df_drugs, rag=None):
    """Agents 1 + 4 (CrewAI) and selective RAG for every drug-event pair of one case."""
    drug_event_pairs = _parse_drugs_and_events(df_drugs, df_reactions)
    drug_causality_map = _extract_drug_causality_map(df_drugs)
    case_metadata = _extract_case_metadata(df_case)

    case_event_date = "MISSING"
    if not df_case.empty:
        case_event_date = _get_value_or_missing(df_case.iloc[0].get("Event Date", "MISSING"))
    case_id = str(df_case.iloc[0].get("Case ID", "")) if not df_case.empty else ""
    case_text = f"Case ID: {case_id or 'N/A'}"

    all_results, agent_logs = {}, []

    for i, (drug, event) in enumerate(drug_event_pairs, 1):
        console.print(f"\n--- Assessment {i}/{len(drug_event_pairs)}: {drug} vs {event} ---",
                      style="bold yellow on blue")

        temporal_causality_context = build_case_context(drug, case_metadata, drug_causality_map,
                                                        case_event_date)
        literature_evidence = search_pubmed(drug, event, max_results=3)
        _, literature_content = save_literature_to_file(drug, event, literature_evidence, "plausibility")

        lit_output = assess_output = ""
        pair_result = None
        if config.RUN_CREW:
            from crewai import Crew
            literature_analyst = build_literature_agent(drug, event)
            safety_assessor = build_safety_assessor(drug, event)
            task_lit = build_literature_task(literature_analyst, literature_content)
            task_assess = build_assessment_task(safety_assessor, drug, event, temporal_causality_context)
            crew = Crew(agents=[literature_analyst, safety_assessor],
                        tasks=[task_lit, task_assess], verbose=False)
            inputs = {
                "case_report_context": case_text,
                "drug_event_pair": f"{drug} vs {event}",
                "EDSS": case_metadata.get("edss", "MISSING"),
            }
            inputs = _ensure_required_inputs([task_lit, task_assess], inputs)
            pair_result = await crew.kickoff_async(inputs=inputs)
            lit_output = task_lit.output.raw if hasattr(task_lit.output, "raw") else str(task_lit.output)
            assess_output = task_assess.output.raw if hasattr(task_assess.output, "raw") else str(task_assess.output)

        # ── RAG: Q2/Q5/Q10 dual answers (no-RAG vs expert-informed) ──
        rag_entry = {}
        if rag is not None:
            rag_context = temporal_causality_context + "\n\n" + literature_content[:config.LITERATURE_CONTEXT_CHARS]
            try:
                rag_entry = rag.run_rag_for_pair(drug, event, rag_context, case_id)
            except Exception as err:
                console.print(f"⚠️  RAG step failed for {drug}/{event}: {err}", style="yellow")

        log = {"Drug": drug, "Adverse Event": event,
               "Literature Agent Summary": lit_output, "Safety Assessor Assessment": assess_output}
        for q in ("Q2", "Q5", "Q10"):
            log[f"{q}_no_rag"] = rag_entry.get(q, {}).get("version_no_rag", "")
            log[f"{q}_with_rag"] = rag_entry.get(q, {}).get("version_with_rag", "")
        agent_logs.append(log)
        all_results[f"{drug} vs {event}"] = pair_result
        console.print(f"✅ Assessment complete for {drug} vs {event}", style="bold green")

    final_report = "### Comprehensive Causality Assessment Report\n\n"
    final_report += "This report analyzes all drug-event pairs from the submitted case report.\n\n---\n"
    for pair, result in all_results.items():
        final_report += f"## Pair Assessment: {pair}\n**CrewAI Final Output:**\n```\n{result}\n```\n\n---\n"

    return final_report, pd.DataFrame(agent_logs)


async def assess_case_async(case_text: str, rag=None, deduplicate: bool = True) -> dict:
    """Run the full MARCA assessment for one case (use `await` inside Jupyter/Colab)."""
    config.ensure_dirs()
    if rag is not None:
        rag.reset()
    text = deduplicate_drugs(case_text) if deduplicate else case_text
    df_case, df_reactions, df_drugs, df_comprehensive = process_case(text, display=True)

    drugs = df_drugs["Drug Name"].tolist() if "Drug Name" in df_drugs else []
    events = df_reactions["Adverse Reaction"].tolist() if "Adverse Reaction" in df_reactions else []

    q1 = assess_listedness(drugs, events)                 # Agent 2
    q10 = classify_aes(events)                            # Agent 3
    final_report, df_agent_logs = await run_case_report_plausibility_assessment(
        df_case, df_reactions, df_drugs, rag)             # Agents 1 + 4, RAG

    return {
        "case_id": str(df_case.iloc[0].get("Case ID", "")) if not df_case.empty else "",
        "df_case": df_case, "df_reactions": df_reactions, "df_drugs": df_drugs,
        "df_comprehensive": df_comprehensive,
        "q1_listedness": pd.DataFrame(q1), "q10_objective_evidence": pd.DataFrame(q10),
        "agent_logs": df_agent_logs, "final_report": final_report,
        "rag_results": list(rag.results) if rag is not None else [],
    }


def assess_case(case_text: str, rag=None, deduplicate: bool = True) -> dict:
    """Synchronous wrapper for scripts. In Jupyter/Colab use `await assess_case_async(...)`."""
    return asyncio.run(assess_case_async(case_text, rag, deduplicate))


def save_assessment_excel(result: dict, out_dir=None) -> Path:
    """Write output/results/FDA_Case_<id>_Assessment.xlsx (one sheet per output, as in the notebook)."""
    out_dir = Path(out_dir or config.RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"FDA_Case_{result['case_id'] or 'Unknown'}_Assessment.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for key, sheet in (("df_case", "Case_Info"), ("df_drugs", "Drug_Info"),
                           ("df_reactions", "Adverse_Reactions"), ("df_comprehensive", "Comprehensive_Data"),
                           ("q1_listedness", "Q1_Label_Listedness"),
                           ("q10_objective_evidence", "Q10_Sign_vs_Symptom"),
                           ("agent_logs", "Agent_Outputs_Split")):
            df = result.get(key)
            if isinstance(df, pd.DataFrame) and not df.empty:
                df.to_excel(writer, sheet_name=sheet, index=False)
        pd.DataFrame({"Comprehensive Plausibility Report": [result["final_report"]]}).to_excel(
            writer, sheet_name="Plausibility_Report_Text", index=False)
        rag_rows = []
        for entry in result.get("rag_results", []):
            for q in ("Q2", "Q5", "Q10"):
                qd = entry.get(q, {})
                refs = qd.get("retrieved_refs", [])
                rag_rows.append({
                    "Drug": entry["drug"], "Adverse_Event": entry["pt"], "Question": q,
                    "Answer_No_RAG": qd.get("version_no_rag", ""),
                    "Answer_With_RAG": qd.get("version_with_rag", ""),
                    "Expert_Refs": "; ".join(
                        f"{r['drug']}/{r['pt']} (expert {q}={r.get(f'{q}_h_score', '?')}, sim={r.get('_sim', 0):.0%})"
                        for r in refs) or "none",
                })
        pd.DataFrame(rag_rows or [{"Message": "RAG module not run or no results."}]).to_excel(
            writer, sheet_name="RAG_Q2_Q5_Q10", index=False)
    console.print(f"✅ Saved {path}", style="green")
    return path
