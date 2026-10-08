# Drug label files (Agent 2, Naranjo Q1)

The per-drug label files used by Agent 2 during the study (one CSV per drug listing the adverse events in its FDA label) were not retained.

**Where the Q1 results can be found:** Agent 2's output for every drug–event combination is preserved in `data/knowledge_base/expert_kb.xlsx`, column **`q1_ai_answer`**. For each DEC it records the drug, the adverse event, the listedness status (`Listed in label` or `Not listed (Yes / No / Unclear)`) and the explanation. The resulting Q1 scores are in `q1_score_ai` (MARCA) and `q1_score_human` (expert).

Q1 was excluded from the comparative analysis and from the final Naranjo classification, so none of the reported agreement statistics depend on these files.

**To run Agent 2 again**, recreate the files in this folder from the recorded study output (`q1_ai_answer`). This writes one file per drug with the events marked "Listed in label" during the study, which reproduces the study's exact-match step:

```python
from software.pipeline import labels_from_study_workbook
labels_from_study_workbook()
```
