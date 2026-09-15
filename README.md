# 🤖 MARCA: Multi-Agent Retrieval-Augmented for Individual-level Causality Assessment in Pharmacovigilance

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

MARCA achieved an overall Naranjo classification agreement of **82.9% (Cohen's κ = 0.277)** against the human expert gold standard — the highest overall classification agreement of any comparator configuration in the cross-study comparison, and higher agreement than prior single-agent LLM approaches on all three reasoning-intensive questions:

| Question | Agreement | Cohen's κ |
|---|---|---|
| Q2 — Temporal association | 96.8% | 0.491 |
| Q5 — Alternative causes | 84.9% | 0.238 |
| Q10 — Objective evidence | 96.8% | 0.922 |
| Q3, Q4, Q6–Q9 | 100% | — (near-universal default scoring due to sparse FAERS documentation) |

### 🔹 Effect of Selective RAG (60-DEC ablation cohort)

| Question | Non-RAG agreement | With RAG | Δ | κ (non-RAG → RAG) |
|---|---|---|---|---|
| **Q2** — Temporal association | 98.33% | 96.67% | −1.66 pp | 0.433 → 0.491 |
| **Q5** — Alternative causes | 68.33% | 83.33% | **+16.67 pp** | 0.151 → 0.177 |
| **Q10** — Objective evidence | 15.00% | 98.33% | **+83.33 pp** | 0.006 → 0.922 |

Retrieval produced its largest effect on **Question 10**, resolving a systematic inability to distinguish objective clinical signs from subjective symptoms without external reference — an information gap rather than a reasoning failure. A meaningful gain was also seen on **Question 5**, while **Question 2** was largely capped by missing therapy-date documentation in FAERS narratives rather than by reasoning ability.

### 🔹 Remaining Disagreement

The largest remaining source of disagreement was **Question 5 (alternative causes)**, where MARCA identified a plausible alternative cause more often than expert assessors (mean score −0.86 vs. −0.50) — a **directionally conservative, safer failure mode** than the overestimation of causality reported in prior single-agent studies.

## 🧪 Conclusions

Integrating a multi-agent architecture with selective RAG improved automated causality assessment agreement while providing a transparent, modular, and auditable framework for individual-level PV — each agent's contribution and each retrieved precedent case can be inspected individually, unlike an opaque single-model inference. MARCA's question-level agreement met or exceeded every threshold established in consultation with industry collaborators, positioning it as a **decision-support / triage layer** — automating routine, well-documented assessments (Q3, Q4, Q6–Q9, and increasingly Q2/Q10) while escalating genuinely ambiguous cases such as Q5 for mandatory human review — rather than a replacement for expert adjudication.

**Limitations:** sparse FAERS documentation constrains variation on several Naranjo items; the evaluation used a single model pairing (Gemini 2.5 Pro + DeepSeek-V3.2) and a relatively limited, single-database (FAERS) sample; the RAG ablation was evaluated on a 60-DEC cohort against a KB that was still being progressively populated during the study, so reported RAG gains are likely a conservative estimate.

**Future directions:** a continuously evolving, human-in-the-loop KB; benchmarking against a larger panel of independent expert assessors; evaluation across additional international spontaneous reporting systems (EudraVigilance, Yellow Card Scheme, JADER); and testing alternative base LLMs to isolate the architecture's contribution from model choice.

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
│   ├── results/                 # Question-level and classification-level agreement tables
│   ├── figures/                  # Confusion matrices, RAG vs. non-RAG comparisons
│   └── logs/                      # Raw agent outputs and rationales
│
├── README.md
```

## ⚙️ Requirements

- Python 3.11
- [CrewAI](https://github.com/crewAIInc/crewAI) v0.126.0
- API access: Google Gemini 2.5 Pro, DeepSeek-V3.2, NCBI Entrez (PubMed), OpenFDA, RxNorm

## 👤 Author

**Daisy Rae Lopez** — Master's Thesis, MSc in Pharmaceutical Sciences, Department of Drug Design and Pharmacology, University of Copenhagen
Supervisor: **Maurizio Sessa**, Associate Professor, University of Copenhagen

## 📚 Citation

If you use this repository or its materials in academic work, please cite:

```bibtex
@article{Lopez2026MARCA,
  title   = {Multi-Agent Retrieval-Augmented for Individual-level Causality Assessment (MARCA) in Pharmacovigilance},
  author  = {Lopez, Daisy Rae and Sessa, Maurizio},
  year    = {2026},
  journal = {Manuscript under review}
}
```
