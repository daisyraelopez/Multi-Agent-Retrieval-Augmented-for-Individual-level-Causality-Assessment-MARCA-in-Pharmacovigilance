# Prediction data provenance

`predictions_rag.csv` is an export of the MARCA AI scores stored in `data/knowledge_base/expert_kb.xlsx` via `split_comparison_workbook()`. **It must not be treated as independently verified live RAG predictions** without original model run logs.

The 60 `predictions_no_rag.csv` rows are the retrospective ablation subset, not an independently held-out sample. This release changes only `dec_id=310, Q2` from +1 to +2 in the AI scores; no AI inference has been rerun. Human question-level annotations are unchanged.
