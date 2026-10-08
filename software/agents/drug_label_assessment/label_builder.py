"""Offline preparation of the label adverse-event lists used by Agent 2.

``build_label_csv`` turns a drug-label PDF (URL or local path) into
``data/knowledge_base/drug_labels/<product>.csv``: the first row is the product name,
then one adverse event per row.
"""
from __future__ import annotations

import csv
import os
import re
import tempfile
from pathlib import Path

import requests

from software import config, llm

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36")


def download_pdf(url: str, output_path: Path | str) -> None:
    with requests.get(url, headers={"User-Agent": _UA}, timeout=30, stream=True) as r:
        r.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


def extract_text(pdf_path: Path | str) -> str:
    """Text of the PDF (PyPDF2 first, PyMuPDF as fallback for hard-to-parse files)."""
    text = ""
    try:
        import PyPDF2
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = "\n".join(t for t in (p.extract_text() for p in reader.pages) if t)
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ PyPDF2 extraction error: {e}")

    if len(text.strip()) < 200:
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(pdf_path)
            text = "".join(page.get_text("text") or "" for page in doc)
            doc.close()
            print("ℹ️ Used PyMuPDF fallback for text extraction.")
        except Exception as e:  # noqa: BLE001
            print(f"⚠️ PyMuPDF extraction failed: {e}")
    return text


def extract_adverse_section(full_text: str) -> str:
    """Ask the LLM for the verbatim adverse-reactions section of the label."""
    prompt = (
        "Extract ONLY the full text section from the following drug label text "
        "that discusses 'Adverse Events', 'Adverse Reactions', or similar safety information. "
        "Do not summarize, rewrite, or add commentary. "
        "Return only the raw text of that section."
    )
    return llm.chat(f"{prompt}\n\n---\n\n{full_text}", "You are a precise document parser.",
                    timeout=180)


def extract_adverse_events(section_text: str) -> list:
    """Ask the LLM for the plain list of adverse events mentioned in the section."""
    prompt = f"""
    You are given the undesirable effects section of a drug product.
    Extract ONLY the list of adverse events mentioned in the text.
    Each adverse event must be listed separately, without duplicates.
    Do not include any explanatory text, frequencies, or formatting, only the adverse events.
    Output them as a plain list, one per line.

    Text:
    {section_text}
    """
    reply = llm.chat(prompt, "You are an expert assistant that extracts structured data.",
                     timeout=180)
    events, seen = [], set()
    for line in reply.splitlines():
        ae = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
        if ae and ae.lower() not in seen:
            seen.add(ae.lower())
            events.append(ae)
    return events


def save_label_csv(product_name: str, adverse_events: list, output_file: Path | str) -> Path:
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([product_name])
        for ae in adverse_events:
            writer.writerow([ae])
    return output_file


def build_label_csv(pdf_url_or_path: str, product_name: str,
                    output_dir: Path | str | None = None) -> Path:
    """PDF → label adverse-event CSV. Returns the path of the CSV that was written."""
    output_dir = Path(output_dir or config.LABELS_DIR)
    with tempfile.TemporaryDirectory() as tmp:
        if os.path.exists(pdf_url_or_path):
            pdf_path = pdf_url_or_path
        else:
            pdf_path = os.path.join(tmp, "label.pdf")
            print(f"📥 Downloading: {pdf_url_or_path}")
            download_pdf(pdf_url_or_path, pdf_path)
        text = extract_text(pdf_path)

    if not text.strip():
        raise ValueError("No text could be extracted from the PDF.")
    section = extract_adverse_section(text)
    if not section:
        raise ValueError("The model found no adverse-reactions section.")
    events = extract_adverse_events(section)
    if not events:
        raise ValueError("No adverse events could be extracted from the section.")

    out = save_label_csv(product_name, events, output_dir / f"{product_name}.csv")
    print(f"✅ {len(events)} adverse events saved to {out}")
    return out
