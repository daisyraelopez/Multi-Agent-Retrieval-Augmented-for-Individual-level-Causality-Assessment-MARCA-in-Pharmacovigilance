# Human expert gold standard

## `gold_standard.csv` (required)
Consensus Naranjo assessments by two independent expert assessors for **215 ICSRs (1,562 DECs)** sampled from FAERS Q1–Q2 2025. Disagreements were resolved by consensus discussion, following a predefined standard operating procedure and a structured extraction form.

Generated from `data/knowledge_base/expert_kb.xlsx` (columns `q1_score_human` … `q10_score_human`) with:

```python
from software.evaluation import split_comparison_workbook
split_comparison_workbook()   # also writes output/results/predictions_rag.csv (MARCA scores)
```

| Column | Description |
|---|---|
| `dec_id` | Row id of the DEC in `expert_kb.xlsx` (unique key) |
| `case_id` | FAERS case ID |
| `drug` | Suspect drug name |
| `ae` | Adverse event (MedDRA Preferred Term) |
| `Q1` … `Q10` | Expert consensus score for each Naranjo question |

Rows are matched to MARCA predictions on `dec_id` (one case has the same drug and event listed twice, so `case_id` + `drug` + `ae` alone is not unique).

## How it is used
`software.evaluation.evaluate()` compares `output/results/predictions_rag.csv` (or `predictions_no_rag.csv`) against this file and reports, for Q2–Q10 and for the final Naranjo category:
- percentage agreement,
- Cohen's κ,
- mean scores and directional differences (MARCA − expert).

Q1 is excluded from the comparative analysis, as described in the manuscript.

## Running MARCA (reproducibility)
Install: `pip install crewai==0.126.0 langchain_google_genai python-dotenv pandas numpy openpyxl scikit-learn pyyaml requests rich PyPDF2 pymupdf plotly umap-learn` (Python ≥ 3.10).

Set the API keys as environment variables: `DEEPSEEK_API_KEY` (DeepSeek-V3.2) and `GOOGLE_API_KEY` (Gemini 2.5 Pro).

```python
# Recompute the published agreement statistics from the data in this repository
from software.evaluation import evaluate, rag_ablation
evaluate()        # question- and classification-level agreement -> output/results/
rag_ablation()    # non-RAG vs RAG on the 60-DEC subset

# Re-run MARCA on a FAERS case
from software.rag import SelectiveRAG
from software.pipeline import assess_case, save_assessment_excel
rag = SelectiveRAG.from_workbook()            # expert KB: data/knowledge_base/expert_kb.xlsx
result = assess_case(open("data/raw/example_case.txt").read(), rag)
save_assessment_excel(result)
```

Data files: `data/knowledge_base/expert_kb.xlsx` holds every DEC with MARCA's and the experts' scores and rationales; this file (`gold_standard.csv`) and `output/results/predictions_rag.csv` are derived from it. The original drug label files used by Agent 2 were not retained; Agent 2's results for every DEC are in the `q1_ai_answer` column of `expert_kb.xlsx` (see `data/knowledge_base/drug_labels/README.md`). `drug_class_map.json` (drug → pharmacological class, used for class-level retrieval) from the study run was not retained; it is recreated automatically on the first run of `SelectiveRAG.from_workbook()` from openFDA, then RxNorm, then DeepSeek. Classes resolved by openFDA and RxNorm are reproducible; classes assigned by the DeepSeek fallback may be worded differently from the study run.
