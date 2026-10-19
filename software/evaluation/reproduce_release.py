"""Rebuild the release from frozen CSV scores (no model API calls).

Run from repository root:
    python -m software.evaluation.reproduce_release

The CSV named predictions_rag.csv is workbook-derived; RAG provenance cannot be
established from these scores alone. This module never claims to rerun inference.
"""
import json
from pathlib import Path
import pandas as pd
from . import (evaluate, rag_ablation, merge_with_gold, total_scores,
               percent_agreement, cohen_kappa, validate_naranjo_scores,
               ALL_QUESTIONS, naranjo_category, confusion_matrix)
from ..config import RESULTS_DIR, GOLD_STANDARD_PATH


def main():
    tables = evaluate(excel_name=None)
    ablation = rag_ablation()
    pred = pd.read_csv(RESULTS_DIR / 'predictions_rag.csv', dtype={'case_id': str})
    gold = pd.read_csv(GOLD_STANDARD_PATH, dtype={'case_id': str})
    for data, label in ((pred, 'MARCA'), (gold, 'human')):
        validate_naranjo_scores(data, ALL_QUESTIONS, label)
    merged = merge_with_gold(pred, gold)
    pm, ph = total_scores(merged)
    exact = percent_agreement(pm,ph)
    scores=pd.DataFrame({
        'dec_id':merged['dec_id'], 'case_id':merged['case_id'],
        'total_marca':pm, 'total_human':ph,
        'class_marca':pm.map(naranjo_category),
        'class_human':ph.map(naranjo_category),
        'signed_difference':pm-ph,
    })
    scores.to_csv(RESULTS_DIR/'validated_dec_classifications.csv',index=False)
    overview = {
        'source': 'frozen workbook-exported question-level scores; no LLM inference rerun',
        'n_decs': len(merged),
        'n_icsrs': int(merged['case_id'].nunique()),
        'q1_included_in_total': True,
        'final_agreement_pct':round(percent_agreement(scores['class_marca'],scores['class_human']),4),
        'final_unweighted_kappa':round(cohen_kappa(scores['class_marca'],scores['class_human']),6),
        'exact_total_score_agreement_pct':round(exact,4),
        'mean_total_marca':round(pm.mean(),6),
        'mean_total_human':round(ph.mean(),6),
        'mean_signed_difference':round((pm-ph).mean(),6),
        'final_confusion':confusion_matrix(merged).to_dict(),
        'rag_cohort_size':int(ablation['n'].iloc[0]),
        'rag_provenance_verified':False,
        'warning': 'dec_id=310 Q2 revised to +2 on user instruction; original clinical rationale retained for adjudication',
    }
    (RESULTS_DIR/'release_metrics.json').write_text(json.dumps(overview,indent=2,ensure_ascii=False,default=int)+'\n')
    print(json.dumps({k:v for k,v in overview.items() if k!='final_confusion'}, indent=2))

if __name__=='__main__':main()
