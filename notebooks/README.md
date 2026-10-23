# How to use the MARCA one-case notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/daisyraelopez/Multi-Agent-Retrieval-Augmented-for-Individual-level-Causality-Assessment-MARCA-in-Pharmacovigilance/blob/main/notebooks/MARCA_one_case.ipynb)

`MARCA_one_case.ipynb` runs one FAERS case through the MARCA pipeline in Google Colab and returns, for every drug–event combination (DEC) in the case, the agents' outputs and a provisional Naranjo score sheet. It runs the repository code: each code cell names the file and lines of `software/` it was copied from, and the last cell checks the code that ran against GitHub.

| Step | Model | What it returns |
|---|---|---|
| Agent 2 — label listedness (Naranjo Q1) | DeepSeek | `Listed in label`, or `Not listed (Yes / No / Unclear)` |
| Agent 3 — sign vs symptom (Q10) | DeepSeek | Sign = 1, Symptom = 0, with a rationale |
| Agent 1 — PubMed retrieval and summary | Gemini 2.5 Pro (CrewAI) | Mechanism of action and evidence summary (up to 3 PubMed records) |
| Agent 4 — clinical causality report (Q2–Q9) | Gemini 2.5 Pro (CrewAI) | Structured report and a HIGH / MEDIUM / LOW plausibility rating |
| Selective RAG (Q2, Q5, Q10) | DeepSeek + expert knowledge base | A score and reasoning without and with expert reference cases |

## Before you start

- A Google account (for Colab).
- A **DeepSeek** API key (platform.deepseek.com) and a **Gemini** API key from Google AI Studio. Both services bill per call.
- Nothing to install locally. The notebook downloads the expert knowledge base (`data/knowledge_base/expert_kb.xlsx`) from this repository.

## Step by step

1. **Open the notebook in Colab** with the badge above, or upload `MARCA_one_case.ipynb` to colab.research.google.com.
2. **Cell 1 — Install.** Run it. If Colab asks to restart the session, restart and continue from cell 2.
3. **Cell 2 — Settings.**

   | Setting | Default | Meaning |
   |---|---|---|
   | `RUN_CREW` | on | Run Agents 1 and 4 (Gemini). Off: only Agents 2, 3 and RAG run. |
   | `USE_RAG` | on | Run the selective RAG step for Q2, Q5 and Q10. |
   | `RESOLVE_DRUG_CLASSES` | on | Map the 195 knowledge-base drugs to pharmacological classes (openFDA → RxNorm → DeepSeek) for the same-class retrieval tier. Takes 3–5 minutes the first time in a session. Off: that tier is skipped. |
   | `GEMINI_MODEL` | `gemini/gemini-2.5-pro` | Model for Agents 1 and 4. |

4. **Cell 3 — API keys.** Replace the two placeholders with your keys. Alternatively leave them and store the keys in Colab's **Secrets** panel as `DEEPSEEK_API_KEY` and `GOOGLE_API_KEY`. Do not share or upload a copy of the notebook that contains your keys.
5. **Cells 4–14 — MARCA code.** Run them; they only define functions.
6. **Cell 15 — Knowledge base, label files and RAG index.** Downloads the knowledge base, rebuilds Agent 2's label files and builds the retrieval index.
   - Agent 2 compares each event with a list of adverse events from the drug's FDA label. The original label files were not kept, so the notebook rebuilds them from the study's recorded results. These files exist only for the 195 drugs in the knowledge base, and only list events that occurred in the study.
   - **For any other drug**, give the URL of its FDA label PDF (e.g. from Drugs@FDA) in `LABEL_PDF_URL`, and in `LABEL_DRUG_NAME` the drug name exactly as written in the case's `Drug Name` field. Example: `https://www.accessdata.fda.gov/drugsatfda_docs/label/2011/009218s107lbl.pdf` and `COUMADIN`. Without a label file, Agent 2 skips that drug and Q1 stays empty.
7. **Cell 16 — The case.** Replace the text inside `CASE_TEXT = """ … """` with your case (format below). It contains the repository's example case (FAERS 229933175) by default.
8. **Cell 17 — Run.** Runs the pipeline. The output is long because Agents 1 and 4 print their work.
9. **Cells 18–20 — Results.** See *Reading the results*.
10. **Cell 21 — Save.** Downloads two Excel files (see *Output files*).
11. **Cell 22 — Verify (optional).** Downloads the source files from GitHub and compares them line by line with the code that ran. ✅ means the code is the repository's, apart from the 6 lines marked `# [notebook]` (imports and the root folder).

**Run everything at once:** Runtime → Run all, after steps 3, 4, 6 and 7.

## Case format

One field per line, `Field: value`. Leading spaces are ignored. Write `NA` for missing values.

- **Several adverse events:** separate them with `;` in `Adverse Reaction(s)`.
- **Several drugs:** every drug field lists one value per drug, separated by ` ; `, in the same order. `Drug Sequence` must have one entry per drug (e.g. `1 ; 2 ; 3`); it sets how many drugs are read.
- If the same drug name appears more than once, the record with the most filled fields is kept.
- **Every drug × every event becomes a DEC**, including concomitant drugs; `Role Code` is not used to select suspect drugs.

Fields the agents use:

| Field | Used for |
|---|---|
| `Case ID` | Identifies the case. Knowledge-base records with the same case ID are excluded from retrieval. Also used in file names. |
| `Adverse Reaction(s)` | The events assessed (Agents 1–4, RAG). |
| `Drug Name` | The drugs assessed (Agents 1, 2 and 4, RAG); PubMed is searched with "drug AND event adverse event". |
| `Drug Sequence` | Number of drugs read from the drug fields. |
| `Age`, `Sex`, `Weight`, `Reporter Country` | Patient context for Agent 4 and RAG. `Age` is passed without its unit, so give it in years. |
| `Event Date`, `Therapy Start Date`, `Therapy End Date` | Temporal context (Q2). If both `Event Date` and `Therapy Start Date` are missing, RAG retrieves cases with missing dates and applies the rule "Q2 = 0". |
| `Dechal`, `Rechal` | Dechallenge and rechallenge results (Q3, Q4). |
| `Route`, `Dose VBM`, `Indication for Use` | Drug context; for the other drugs of the case, their indication and therapy period are listed as concomitant medication. |

Other fields (e.g. `Product AI`, `NDA Number`, `Lot Number`) are parsed but not used by the agents. `Outcome Code` is not read by the parser, so Agent 4 always sees it as missing.

## Reading the results

- **Cell 18** shows the Agent 2 and Agent 3 tables, the RAG table (score without and with references, plus the references retrieved: drug/event, retrieval tier, the expert's score and similarity), and the Agent 1 and Agent 4 texts for every DEC.
- **Cell 19 — provisional score sheet.** This is a notebook addition; the repository code does not produce Naranjo scores for all questions.

  | Question | Source in the score sheet |
  |---|---|
  | Q1 | Agent 2: `Listed in label` or `Not listed (Yes)` → +1, otherwise 0 (the correspondence found in the study workbook). Empty if the drug has no label file. |
  | Q2, Q5 | RAG answer, with references (`Q2_Q5_SOURCE = "with_rag"`) or without (`"no_rag"`). |
  | Q10 | Agent 3 (Sign = 1). The RAG answer for Q10 is shown next to it. |
  | Q3, Q4, Q6–Q9 | 0 until you score them from Agent 4's report, as in the study (manuscript Section 2.4.5). Enter them in `MANUAL_SCORES`, e.g. `("ICATIBANT", "Epistaxis"): {"Q3": 0, "Q4": 0}`, and re-run cells 19–21. |

  `Total` and `Category` (Doubtful ≤0, Possible 1–4, Probable 5–8, Definite ≥9) are only computed when every question has a value; the `Needs review` column lists what is still a placeholder.
- **Cell 20** compares the run with the study's stored MARCA scores and the expert scores, when the case and its drug–event pairs are in the knowledge base (e.g. the default example case).

## Output files

| File | Content |
|---|---|
| `FDA_Case_<id>_Assessment.xlsx` | The repository's standard output: parsed case, Agent 2 and 3 results, Agent 1 and 4 texts, RAG answers with their references. |
| `FDA_Case_<id>_score_sheet.xlsx` | The provisional score sheet, the comparison with the study (if any) and the run information (repository commit, models, settings). |

## Time and cost

About 2 Gemini calls and 8 DeepSeek calls per DEC (plus the drug-class lookup the first time). The example case (10 DECs) takes about 10–15 minutes. For a quick first try, keep one or two events in `Adverse Reaction(s)`.

## Points to keep in mind

- **Model settings.** The repository code creates the Gemini agents without a temperature, so Gemini runs at Google's default; DeepSeek runs at temperature 0. `deepseek-chat` is an alias that DeepSeek can point to newer models, so note the date of each run.
- **Label matching is exact.** Agent 2 marks an event as listed only if the same term (ignoring case) is in the label file. Otherwise DeepSeek judges whether it is related, which can still give Q1 = +1 through `Not listed (Yes)`.
- **Agent 3 and RAG may disagree on Q10.** Agent 3 classifies the term as sign or symptom; the RAG prompt asks for patient-specific objective confirmation.
- **Retrieval uses the study's expert cases.** If you reuse a case ID that is in the knowledge base, its own records are excluded from retrieval.

## Troubleshooting

| Message or symptom | What to do |
|---|---|
| Colab asks to restart after cell 1 | Restart, then run from cell 2. |
| A key is requested in a prompt | The placeholder in cell 3 was not replaced and no Colab Secret was found: paste the key. |
| `No label CSV found for <drug>. Skipping.` | Build the drug's label file in cell 15 (`LABEL_PDF_URL`, `LABEL_DRUG_NAME`). |
| `PubMed search failed` | Not fatal: Agent 1 summarises without articles. Re-run later if needed. |
| Gemini or DeepSeek error 429 / quota | Wait and re-run cell 17, or reduce the number of events. |
| A score shows `invalid (...)` | The model returned a value that is not a Naranjo score for that question; read the reasoning in cell 18. |
| Cell 22 shows ❌ | A copied code line was changed; the table lists the file and line. |

## Example run

`examples/MARCA_one_case_example_run_coumadin.ipynb` is a run made in Colab on 20 October 2026, kept with its outputs. It uses an earlier version of the notebook and a **constructed test case**, not a real FAERS report (see the note at its top).
