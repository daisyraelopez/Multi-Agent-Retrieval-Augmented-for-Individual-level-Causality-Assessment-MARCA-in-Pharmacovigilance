# 🤖 Multi-Agent Retrieval-Augmented Causality Assessment (MARCA) for Individual-level Causality Assessment in Pharmacovigilance

## 📌 Background

The growing volume of Individual Case Safety Reports (ICSRs) submitted to spontaneous reporting systems such as the FDA Adverse Event Reporting System (FAERS) has intensified the demand for scalable approaches to individual-level causality assessment in pharmacovigilance (PV). Existing automated approaches predominantly rely on single-agent Large Language Model (LLM) architectures with limited access to external knowledge, which reduces performance on the most clinically demanding parts of causality reasoning — temporal plausibility, alternative causes, and objective evidence.

This project introduces **MARCA (Multi-Agent Retrieval-Augmented Causality Assessment)**, a modular multi-agent framework built on the **Naranjo Adverse Drug Reaction Probability Scale**, which distributes causality reasoning across four specialized agents and applies **selective retrieval-augmented generation (RAG)** to the algorithm's most challenging items.

## 🎯 Objective

To develop MARCA and investigate whether integrating a task-oriented multi-agent architecture with selective RAG improves agreement between automated and human expert causality assessments, compared with prior single-agent LLM approaches.

**Research question:** *Can a multi-agent system with RAG improve inter-rater agreement between automated and human expert causality assessments?*

## 🧩 System Architecture

MARCA was implemented using the **CrewAI orchestration framework (v0.126.0)**, integrating **Google Gemini 2.5 Pro** (primary clinical reasoning) and **DeepSeek-V3.2** (RAG-based question scoring, objective-evidence classification, drug-class resolution), with all models called at temperature 0 for reproducibility.

| Agent | Role | Naranjo Item(s) | Model |
|---|---|---|---|
| **Agent 1 — Scientific Evidence Retrieval** | Queries PubMed (NCBI Entrez API) for mechanism-of-action and supporting/refuting clinical evidence | Feeds Agent 4 | Gemini 2.5 Pro |
| **Agent 2 — Drug Label Assessment** | Classifies AE listedness against FDA drug labels | Q1 | DeepSeek-V3.2 |
| **Agent 3 — Objective Evidence Assessment** | Classifies reported terms as objective sign vs. subjective symptom | Q10 | DeepSeek-V3.2 |
| **Agent 4 — Clinical Causality Assessment** | Produces rationale and plausibility rating from full structured case context + literature summary | Q2–Q9 | Gemini 2.5 Pro |

A continuously updated **Knowledge Base (KB)** of expert-assessed drug–event combinations (DECs) supports **selective RAG restricted to Naranjo Questions 2, 5, and 10** — the items identified in prior work as the most reasoning-intensive. RAG is triggered only when a non-retrieval answer disagrees with the expert assessment for that question, so retrieval is applied where it adds value rather than universally.

## 📊 Key Results & Findings

**Dataset:** Human expert causality assessments for **215 ICSRs** (1,562 drug–event combinations) extracted from FAERS, forming the gold-standard reference alongside a full stratified FAERS dataset (n = 55,177) used for descriptive comparison.

### 🔹 Overall Performance

On the frozen data (1,562 DECs from 215 ICSRs), MARCA's final four-category Naranjo classification (Q1–Q10) agreed with the human expert reference in **79.4% of DECs (Cohen's κ = 0.587)**; exact agreement on the cumulative numerical score was 71.6%. These values are numerically higher than those reported for the comparator configurations of Heckmann et al. (2026a, 2026b), but the comparison is descriptive only: case populations, algorithms, assessment protocols and agreement statistics (κ vs. Gwet's AC1) differ between studies.

| Question | Agreement | Cohen's κ |
|---|---|---|
| Q1 — Previous conclusive reports (descriptive) | 89.3% | 0.787 |
| Q2 — Temporal association | 96.8% | 0.491 |
| Q3 — Improvement after discontinuation | 100% | 1.000 |
| Q4, Q6–Q9 | 100% | not estimable (all scores zero in both MARCA and expert data) |
| Q5 — Alternative causes | 85.4% | 0.269 |
| Q10 — Objective evidence | 96.8% | 0.922 |

Q1 was not part of the primary question-level comparison, and its >95% agreement benchmark was not met. Q4 and Q6–Q9 show no score variation, so their 100% agreement is uninformative about discriminative performance.

### 🔹 Selective RAG: retrospective 60-DEC ablation (descriptive only)

| Question | Non-RAG agreement | RAG-labelled agreement | Δ | κ (non-RAG → RAG-labelled) |
|---|---|---|---|---|
| **Q2** — Temporal association | 98.33% | 100.00% | +1.67 pp | 0.000 → not estimable |
| **Q5** — Alternative causes | 65.00% | 83.33% | **+18.33 pp** | 0.016 → 0.571 |
| **Q10** — Objective evidence | 18.33% | 100.00% | **+81.67 pp** | 0.007 → 1.000 |

The 60 DECs were selected because at least one non-RAG score for Q2, Q5 or Q10 disagreed with the expert reference, so this is a disagreement-enriched retrospective comparison, not a prospective test of RAG triggering. In addition, the RAG-labelled scores were exported from the comparison workbook and their original inference logs were not available (see `output/results/PREDICTION_PROVENANCE.md`). The differences above are therefore descriptive and should not be read as a validated effect of retrieval. The largest difference was on **Question 10** (non-RAG scores were lower than the expert's in 49 of 60 DECs; none in the RAG-labelled file), followed by **Question 5**; **Question 2** changed in a single DEC.

### 🔹 Remaining Disagreement

In the full sample (1,562 DECs), the largest remaining disagreement was on **Question 5 (alternative causes)**: agreement 85.4%, κ = 0.269, with MARCA assigning lower scores than the experts in 205 DECs and higher in 23 (mean scores −0.85 vs. −0.50; mean signed difference −0.35). MARCA therefore tended to credit alternative causes more often than experts, i.e. to assign lower causality. Whether this is "safer" than the behaviour of earlier single-agent approaches has not been tested here; cross-study comparisons are descriptive only.

## 🧪 Conclusions

On the frozen data, MARCA's final four-category Naranjo classification (Q1–Q10) agreed with the expert reference in 79.4% of DECs (κ = 0.587), and exact numerical-score agreement was 71.6%. Question-level agreement was high for Q2, Q3, Q4, Q6–Q10 (≥96.8% raw agreement; κ non-estimable where scores did not vary) and lowest for Q5. The pre-specified percentage-agreement benchmark for Q1 (>95%) was **not** met (89.3%). The architecture is modular and each agent's output and retrieved precedent can be inspected, which supports its use as a **decision-support / triage layer** with human review of ambiguous items such as Q5, rather than as a replacement for expert adjudication.

**Limitations:** sparse FAERS documentation limits score variation on several Naranjo items (no variation for Q4, Q6–Q9); a single model pairing (Gemini 2.5 Pro + DeepSeek-V3.2) and a single database (FAERS) were used; the knowledge base and evaluation set overlap (same-ICSR examples excluded at retrieval, which is not independent validation); no LLM calls were rerun in this release and the provenance of the RAG-labelled scores is unverified; the 60-DEC ablation is retrospective and selected on known disagreements, so its direction and size cannot be generalised; the Q2 rationale for DEC 310 still needs clinical reconciliation with its corrected score (+2).

**Future directions:** a continuously evolving, human-in-the-loop KB with archived inference logs; prospective, independently held-out evaluation of selective RAG; benchmarking against a larger panel of independent expert assessors; evaluation across additional spontaneous reporting systems (EudraVigilance, Yellow Card Scheme, JADER); and testing alternative base LLMs to separate the architecture's contribution from model choice.

## 🗂 Repository Structure

```
├── data/
│   ├── raw/                    # FAERS case extracts and stratified sampling files
│   ├── knowledge_base/         # Expert-assessed DECs (KB) used for selective RAG
│   └── gold_standard/          # Human expert consensus assessments (Naranjo Q1-10)
│
├── software/
│   ├── agents/                 # Agent 1-4 implementations (CrewAI)
│   │   ├── scientific_evidence_retrieval/
│   │   ├── drug_label_assessment/
│   │   ├── objective_evidence_assessment/
│   │   └── clinical_causality_assessment/
│   ├── rag/                    # Selective RAG pipeline (Q2, Q5, Q10) and KB indexing
│   ├── case_parsing/           # FAERS narrative parsing & structured case construction
│   ├── drug_class_resolution/  # OpenFDA / RxNorm / LLM-based drug-class lookup
│   └── evaluation/              # Agreement statistics (% agreement, Cohen's κ), plots
│
├── output/
│   ├── archive_original/                 
│   ├── figures/
│   ├── results/               # Question-level and classification-level agreement tables
├── README.md
```

## ⚙️ Requirements

- Python 3.11
- [CrewAI](https://github.com/crewAIInc/crewAI) v0.126.0
- API access: Google Gemini 2.5 Pro, DeepSeek-V3.2, NCBI Entrez (PubMed), OpenFDA, RxNorm

## 👤 Author

**Daisy Rae Lopez** (Master's Student, Dept. of Drug Design and Pharmacology (ILF), University of Copenhagen, Copenhagen, Denmark)
**Angela Boccia** (PhD Student, Department of Medical and Surgical Sciences (DIMEC), University of Bologna, Bologna, Italy)
**Maurizio Sessa** (Advisor and Coordinator, Dept. of Drug Design and Pharmacology (ILF), University of Copenhagen, Copenhagen, Denmark)


## 📚 Citation

If you use this repository or its materials in academic work, please cite:

### BibTeX
```bibtex
@article{Lopez2026MARCA,
  title   = {Multi-Agent Retrieval-Augmented for Individual-level Causality Assessment (MARCA) in Pharmacovigilance},
  author  = {Lopez, Daisy Rae; Boccia, Angela; Sessa, Maurizio},
  year    = {2026},
  journal = {Manuscript under review}
}
```
