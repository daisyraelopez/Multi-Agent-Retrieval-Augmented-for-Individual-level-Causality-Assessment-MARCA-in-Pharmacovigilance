# MARCA reproducibility correction release — 19 October 2026

## Authorised source correction

The user confirmed `dec_id=310` (methotrexate/cellulitis), MARCA Q2 **+1 → +2**. This is applied to `expert_kb.xlsx` and `predictions_rag.csv`. The human expert annotation (Q2=0), the remaining question-level scores, and all clinical narratives/rationales were retained. **Q2 rationale for DEC 310 remains to be clinically reconciled**, as it states dates are insufficient for establishing temporality.

## Code corrections

- The final Naranjo cumulative score is now **Q1–Q10**, not Q2–Q10. Primary question analysis remains **Q2–Q10** (Q1 descriptive).
- Enforced valid question-specific Naranjo score domains; invalid scores now raise instead of silently affecting metrics.
- Q2/Q5/Q10 RAG prompts now use valid Naranjo score categories and avoid assigning objective confirmation solely based on sign-vs-symptom term identity.
- Added missing `software/config.py`; fixed case-parser package filename, pipeline imports, and compatibility wrappers for the duplicated RAG entry point.
- Full output tables and verification figures can be generated **offline**, with no LLM calls, by running `python -m software.evaluation.reproduce_release` and `python scripts/plot_verified_results.py`.
- Updated the workbook cached formulas without changing the original formula expressions, preserving its narrative fields.

## Verified results (frozen scores)

- 215 ICSRs; 1,562 DECs; Q5 agreement 85.4%, κ 0.269; Q2 agreement 96.8%, κ 0.491; Q10 96.8%, κ 0.922.
- Final Q1–Q10 classifications: agreement **79.4494%**, κ **0.587002**.
- Exact numerical total-score agreement: **71.5749%**.
- Means: MARCA **0.482074**, expert **0.744558**; signed difference **−0.262484**.
- Expert rows: Doubtful 867, Possible 685, Probable 10, Definite 0.
- Conditional 60-DEC ablation: Q2 98.33→100%; Q5 65.00→83.33%; Q10 18.33→100%.

## Important limitations

**No LLM API calls were rerun.** `predictions_rag.csv` is exported from the expert comparison workbook, and its live RAG provenance could not be independently confirmed. The ablation is retrospective and selected on disagreements with known expert scores; it is not evidence of deployable expert-guided triggering or prospective improvement. The original KB and evaluation population overlap, though same-ICSR examples are excluded during retrieval. No new model weights or published cross-study data were independently regenerated.

The supplied source repository contained no dependency lockfile. Tested offline packages in this environment are documented in `requirements-evaluation.txt`. Runtime CrewAI API evaluation requires credentials and dependencies separately.

## Reproducibility

```sh
cd MARCA
python -m software.evaluation.reproduce_release
python -m unittest discover -s tests -v
python scripts/plot_verified_results.py
```

See `manuscript/METHODS_RESULTS_replacement.md` for Methods and Results replacement language.
