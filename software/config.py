"""Central configuration for MARCA.

All file locations, model names and API keys live here. API keys are read from
environment variables (or a local `.env` file) and are never stored in the code.
Set DEEPSEEK_API_KEY and GOOGLE_API_KEY as environment variables before running.
"""
import os
from pathlib import Path

try:  # optional: load variables from a .env file if python-dotenv is installed
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Repository paths ──────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

RAW_DIR = DATA_DIR / "raw"                                   # FAERS case text files
CASE_FILES = sorted(RAW_DIR.glob("case_*.txt")) or [RAW_DIR / "example_case.txt"]

KB_DIR = DATA_DIR / "knowledge_base"
KB_PATH = Path(os.getenv("MARCA_KB_PATH", KB_DIR / "expert_kb.xlsx"))   # expert-assessed DECs
DRUG_CLASS_CACHE_PATH = KB_DIR / "drug_class_map.json"       # cached drug -> class map
LABELS_DIR = KB_DIR / "drug_labels"                          # one AE-list CSV per drug

GOLD_STANDARD_PATH = DATA_DIR / "gold_standard" / "gold_standard.csv"

OUTPUT_DIR = ROOT_DIR / "output"
RESULTS_DIR = OUTPUT_DIR / "results"
FIGURES_DIR = OUTPUT_DIR / "figures"
LOGS_DIR = OUTPUT_DIR / "logs"

# ── Models (all queried at temperature 0) ────────────────────────────────────
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"                               # DeepSeek-V3.2
CREW_LLM = os.getenv("MARCA_CREW_LLM", "gemini/gemini-2.5-pro")  # Agents 1 and 4
TEMPERATURE = 0

# Set MARCA_RUN_CREW=0 to skip the CrewAI agents (Agents 1 and 4), e.g. to test RAG only
RUN_CREW = os.getenv("MARCA_RUN_CREW", "1") == "1"

# ── API keys (from environment only) ─────────────────────────────────────────
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")   # used by CrewAI/LiteLLM for Gemini

# ── Context limits used in the RAG prompts ───────────────────────────────────
CASE_CONTEXT_CHARS = 1400
LITERATURE_CONTEXT_CHARS = 800


def ensure_dirs():
    """Create the output folders if they do not exist."""
    for d in (RESULTS_DIR, FIGURES_DIR, LOGS_DIR, LABELS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def require_key(name: str) -> str:
    """Return an API key from the environment or raise a clear error."""
    value = os.getenv(name, "")
    if not value:
        raise EnvironmentError(
            f"{name} is not set. Copy .env.example to .env and add your key, "
            f"or run: export {name}=..."
        )
    return value


# ════════════════════════════════════════════════════════════════════════════
# Centralized DeepSeek call (Agents 2 and 3, RAG, drug-class fallback)
# ════════════════════════════════════════════════════════════════════════════
import json
import requests

import json
import requests


DEFAULT_SYSTEM = (
    "You are a senior pharmacovigilance expert performing drug causality "
    "assessment. Answer concisely with a numeric score and a brief justification."
)


def call_deepseek(prompt: str, system: str = DEFAULT_SYSTEM, json_mode: bool = False,
                  timeout: int = 60) -> str:
    """Send one chat request to DeepSeek (temperature 0) and return the text reply.

    If `json_mode` is True the response is requested as a JSON object; the raw
    string is still returned, so callers parse it with `json.loads`.
    """
    key = require_key("DEEPSEEK_API_KEY")
    payload = {
        "model": DEEPSEEK_MODEL,
        "temperature": TEMPERATURE,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    r = requests.post(
        DEEPSEEK_API_URL,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def call_deepseek_json(prompt: str, system: str, timeout: int = 60) -> dict:
    """Convenience wrapper: call DeepSeek in JSON mode and parse the reply."""
    return json.loads(call_deepseek(prompt, system=system, json_mode=True, timeout=timeout))
