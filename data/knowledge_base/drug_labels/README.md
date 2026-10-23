# Drug label files (Agent 2, Naranjo Q1)

This folder is the location (`LABELS_DIR` in `software/config.py`) where Agent 2 looks for one CSV of listed adverse events per drug. **The original label files used in the study were not retained**, so the folder contains only this README.

## What is available instead

`data/knowledge_base/expert_kb.xlsx` (sheet `Cases`, column `q1_ai_answer`) stores Agent 2's recorded output for all 1,562 DECs, in the form:

```
drug: <DRUG>
  ae: <MedDRA PT>
  status: Listed in label | Not listed (Yes) | Not listed (No) | Not listed (Unclear)
  explanation: <free text>
```

Status counts in the frozen workbook: Listed in label 129; Not listed (Yes) 692; Not listed (No) 684; Not listed (Unclear) 57.

## Recreating the label files

```python
from software.agents.pipeline import labels_from_study_workbook
labels_from_study_workbook()          # writes <DRUG>.csv files into this folder
```

- Events with status "Listed in label" were exact matches in the original label of that drug; only these are written. The files therefore contain **only the label events that occurred in the study DECs, not the complete label**.
- All other events are sent to DeepSeek, as in the study run. Because LLM output is not fully deterministic even at temperature 0, re-running Agent 2 may not reproduce the stored `q1_ai_answer` values exactly.
- To build complete label lists from current FDA labels, use `build_all_label_csvs()` from the same module (requires network access and API keys; results will reflect today's labels, not those used in the study).

