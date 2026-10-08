"""Evaluation: prediction tables and agreement metrics (summary.py: console summary;
plots.py: figures)."""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from .. import config
from .summary import print_rag_summary


# ════════════════════════════════════════════════════════════════════════════
# Predictions
# ════════════════════════════════════════════════════════════════════════════



RAG_QUESTIONS = ["Q2", "Q5", "Q10"]


def parse_score(text):
    """Extract the numeric score from an answer such as 'Score: -1 ...'."""
    if not text:
        return None
    m = re.search(r"Score\s*[:\-]\s*([+-]?\d+(?:\.\d+)?)", str(text), re.IGNORECASE)
    return float(m.group(1)) if m else None


def predictions_from_case_result(rag_results: list):
    """Return (no_rag_df, rag_df) with columns case_id, drug, ae, Q2, Q5, Q10."""
    no_rag_rows, rag_rows = [], []
    for entry in rag_results:
        base = {"case_id": entry.get("case_id", ""), "drug": entry["drug"], "ae": entry["pt"]}
        nr, wr = dict(base), dict(base)
        for q in RAG_QUESTIONS:
            qd = entry.get(q, {}) or {}
            nr[q] = parse_score(qd.get("version_no_rag"))
            wr[q] = parse_score(qd.get("version_with_rag"))
        no_rag_rows.append(nr)
        rag_rows.append(wr)
    cols = ["case_id", "drug", "ae"] + RAG_QUESTIONS
    return pd.DataFrame(no_rag_rows, columns=cols), pd.DataFrame(rag_rows, columns=cols)


def save_predictions(no_rag_df: pd.DataFrame, rag_df: pd.DataFrame, out_dir=None, append=True):
    """Write predictions_no_rag.csv and predictions_rag.csv (appending across cases)."""
    out_dir = Path(out_dir or config.RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    for df, name in ((no_rag_df, "predictions_no_rag.csv"), (rag_df, "predictions_rag.csv")):
        path = out_dir / name
        if append and path.exists():
            df = pd.concat([pd.read_csv(path, dtype={"case_id": str}), df], ignore_index=True)
            df = df.drop_duplicates(subset=["case_id", "drug", "ae"], keep="last")
        df.to_csv(path, index=False)
    return out_dir


def split_comparison_workbook(path=None, gold_path=None, predictions_path=None):
    """Create gold_standard.csv and predictions_rag.csv from the MARCA-vs-expert workbook.

    The workbook (data/knowledge_base/expert_kb.xlsx) has one row per DEC with
    `q1_score_ai` … `q10_score_ai` (MARCA, final scores) and `q1_score_human` …
    `q10_score_human` (expert consensus). `id` is kept as `dec_id`.
    """
    path = path or config.KB_PATH
    df = pd.read_excel(path, sheet_name=0)
    base = pd.DataFrame({
        "dec_id": df["id"],
        "case_id": df["case_id"].astype(str).str.strip(),
        "drug": df["drug_name"].astype(str).str.strip(),
        "ae": df["pt"].astype(str).str.strip(),
    })
    gold, pred = base.copy(), base.copy()
    for i in range(1, 11):
        pred[f"Q{i}"] = df[f"q{i}_score_ai"]
        gold[f"Q{i}"] = df[f"q{i}_score_human"]
    gold_path = Path(gold_path or config.GOLD_STANDARD_PATH)
    predictions_path = Path(predictions_path or config.RESULTS_DIR / "predictions_rag.csv")
    gold_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    gold.to_csv(gold_path, index=False)
    pred.to_csv(predictions_path, index=False)
    print(f"✅ {len(gold)} DECs → {gold_path} and {predictions_path}")
    return gold, pred


# ════════════════════════════════════════════════════════════════════════════
# Agreement metrics
# ════════════════════════════════════════════════════════════════════════════



KEYS = ["case_id", "drug", "ae"]
ALL_QUESTIONS = [f"Q{i}" for i in range(1, 11)]
TOTAL_QUESTIONS = [f"Q{i}" for i in range(2, 11)]      # Q1 excluded from the total score
CATEGORIES = ["Doubtful", "Possible", "Probable", "Definite"]

DESCRIPTIONS = {
    "Q1": "Previous conclusive reports",
    "Q2": "Temporal association",
    "Q3": "Improvement following drug discontinuation",
    "Q4": "Reappearance following re-challenge",
    "Q5": "Alternative cause excluded",
    "Q6": "Reaction following placebo administration",
    "Q7": "Toxic drug level detected",
    "Q8": "Dose-response relationship",
    "Q9": "Similar reaction to previous exposure",
    "Q10": "Objective evidence confirmation",
}


# ── basic measures ────────────────────────────────────────────────────────────
def naranjo_category(total):
    if pd.isna(total):
        return np.nan
    if total >= 9:
        return "Definite"
    if total >= 5:
        return "Probable"
    if total >= 1:
        return "Possible"
    return "Doubtful"


def percent_agreement(a: pd.Series, b: pd.Series) -> float:
    return float((a.astype(str) == b.astype(str)).mean() * 100)


def cohen_kappa(a: pd.Series, b: pd.Series):
    """Unweighted Cohen's kappa; None when undefined (a single value used by both raters)."""
    a, b = a.astype(str), b.astype(str)
    if len(set(a) | set(b)) < 2:
        return None
    return float(cohen_kappa_score(a, b))


def _to_number(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


# ── data preparation ─────────────────────────────────────────────────────────
def _norm_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for k in KEYS:
        df[k] = df[k].astype(str).str.strip().str.upper()
    return df


def merge_with_gold(pred: pd.DataFrame, gold: pd.DataFrame) -> pd.DataFrame:
    """Join MARCA predictions and expert scores; columns become Qn_marca / Qn_expert.

    If both tables have a `dec_id` column (unique row id per DEC) it is used as the key,
    which also handles the rare case of a repeated case_id + drug + ae combination.
    """
    pred, gold = _norm_keys(pred), _norm_keys(gold)
    if "dec_id" in pred and "dec_id" in gold:
        merged = pred.merge(gold.drop(columns=KEYS), on="dec_id", suffixes=("_marca", "_expert"))
    else:
        merged = pred.merge(gold, on=KEYS, suffixes=("_marca", "_expert"))
    unmatched = len(pred) - len(merged)
    if unmatched:
        print(f"⚠️  {unmatched} prediction rows had no matching gold-standard row and were skipped.")
    return merged


def _questions_present(merged, questions):
    return [q for q in questions if f"{q}_marca" in merged and f"{q}_expert" in merged]


def total_scores(merged, questions=TOTAL_QUESTIONS):
    qs = _questions_present(merged, questions)
    marca = merged[[f"{q}_marca" for q in qs]].apply(_to_number).sum(axis=1, min_count=1)
    expert = merged[[f"{q}_expert" for q in qs]].apply(_to_number).sum(axis=1, min_count=1)
    return marca, expert


# ── result tables ────────────────────────────────────────────────────────────
def question_level_table(merged, questions=ALL_QUESTIONS, total_questions=TOTAL_QUESTIONS):
    rows = []
    for q in _questions_present(merged, questions):
        m, e = _to_number(merged[f"{q}_marca"]), _to_number(merged[f"{q}_expert"])
        ok = m.notna() & e.notna()
        m, e = m[ok], e[ok]
        k = cohen_kappa(m, e)
        rows.append({
            "Question": q, "Description": DESCRIPTIONS.get(q, ""), "n": int(ok.sum()),
            "Agreement (%)": round(percent_agreement(m, e), 1),
            "Cohen's k": None if k is None else round(k, 3),
            "Mean MARCA": round(m.mean(), 2), "Mean Human": round(e.mean(), 2),
        })
    tm, te = total_scores(merged, total_questions)
    cm, ce = tm.map(naranjo_category), te.map(naranjo_category)
    ok = cm.notna() & ce.notna()
    k = cohen_kappa(cm[ok], ce[ok])
    rows.append({
        "Question": "Final", "Description": "Overall Naranjo causality classification",
        "n": int(ok.sum()), "Agreement (%)": round(percent_agreement(cm[ok], ce[ok]), 1),
        "Cohen's k": None if k is None else round(k, 3),
        "Mean MARCA": round(tm.mean(), 2), "Mean Human": round(te.mean(), 2),
    })
    return pd.DataFrame(rows)


def directional_table(merged, questions=TOTAL_QUESTIONS):
    rows = []
    for q in _questions_present(merged, questions):
        m, e = _to_number(merged[f"{q}_marca"]), _to_number(merged[f"{q}_expert"])
        ok = m.notna() & e.notna()
        d = (m - e)[ok]
        rows.append({
            "Question": q, "MARCA > Expert (n)": int((d > 0).sum()),
            "MARCA = Expert (n)": int((d == 0).sum()), "MARCA < Expert (n)": int((d < 0).sum()),
            "Mean difference": round(d.mean(), 3), "Total": int(ok.sum()),
        })
    return pd.DataFrame(rows)


def confusion_matrix(merged, total_questions=TOTAL_QUESTIONS):
    """Rows = expert category, columns = MARCA category (categories that occur only)."""
    tm, te = total_scores(merged, total_questions)
    cm, ce = tm.map(naranjo_category), te.map(naranjo_category)
    ok = cm.notna() & ce.notna()
    present = [c for c in CATEGORIES if c in set(cm[ok]) | set(ce[ok])]
    table = pd.crosstab(pd.Categorical(ce[ok], categories=present),
                        pd.Categorical(cm[ok], categories=present), dropna=False)
    table.index.name, table.columns.name = "Expert", "MARCA"
    return table


def per_category_table(conf: pd.DataFrame):
    rows = []
    for cat in conf.index:
        row = conf.loc[cat]
        n, correct = int(row.sum()), int(row.get(cat, 0))
        if n == 0:
            continue
        wrong = row.drop(cat)
        if wrong.sum() > 0:
            other = wrong.idxmax()
            mis = f"{other} (n = {int(wrong.max())})"
        else:
            mis = "N/A"
        rows.append({
            "Category": cat, "Expert-assigned (n)": n, "Correctly classified (n)": correct,
            "Agreement (%)": round(correct / n * 100, 1), "Most common misclassification": mis,
        })
    return pd.DataFrame(rows)


# ── main entry point ─────────────────────────────────────────────────────────
def evaluate(predictions_path=None, gold_path=None, out_dir=None,
             questions=ALL_QUESTIONS, total_questions=TOTAL_QUESTIONS, excel_name="MARCA_analysis.xlsx"):
    """Compute all result tables and save them as CSV files and one Excel workbook.

    Returns a dict with the four tables.
    """
    pred = pd.read_csv(predictions_path or config.RESULTS_DIR / "predictions_rag.csv", dtype={"case_id": str})
    gold = pd.read_csv(gold_path or config.GOLD_STANDARD_PATH, dtype={"case_id": str})
    merged = merge_with_gold(pred, gold)

    conf = confusion_matrix(merged, total_questions)
    tables = {
        "Table6_question_level": question_level_table(merged, questions, total_questions),
        "Table7_directional": directional_table(merged, total_questions),
        "Table8_per_category": per_category_table(conf),
        "Confusion_matrix": conf,
    }

    out_dir = Path(out_dir or config.RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    q = tables["Table6_question_level"]
    q[q["Question"] != "Final"].to_csv(out_dir / "question_level_agreement.csv", index=False)
    pd.concat([q[q["Question"] == "Final"].rename(columns={"Question": "Level"}),
               tables["Table8_per_category"].rename(columns={"Category": "Level"})],
              ignore_index=True).to_csv(out_dir / "classification_level_agreement.csv", index=False)
    tables["Table7_directional"].to_csv(out_dir / "directional_disagreement.csv", index=False)
    conf.to_csv(out_dir / "confusion_matrix.csv")
    with pd.ExcelWriter(out_dir / excel_name, engine="openpyxl") as writer:
        for name, df in tables.items():
            df.to_excel(writer, sheet_name=name, index=(name == "Confusion_matrix"))
    print(f"✅ Results saved to {out_dir}")
    return tables


def rag_ablation(no_rag_path=None, rag_path=None, gold_path=None, out_dir=None,
                 questions=("Q2", "Q5", "Q10")):
    """Non-RAG vs RAG agreement on the DECs in predictions_no_rag.csv (manuscript Table 5).

    For each question and condition: % agreement, Cohen's kappa, mean difference
    (MARCA − expert) and the number of over-, equal and under-estimations.
    """
    res = Path(out_dir or config.RESULTS_DIR)
    no_rag = pd.read_csv(no_rag_path or config.RESULTS_DIR / "predictions_no_rag.csv", dtype={"case_id": str})
    rag = pd.read_csv(rag_path or config.RESULTS_DIR / "predictions_rag.csv", dtype={"case_id": str})
    gold = pd.read_csv(gold_path or config.GOLD_STANDARD_PATH, dtype={"case_id": str})
    rag = rag[rag["dec_id"].isin(no_rag["dec_id"])]
    rows = []
    for cond, pred in (("Non-RAG", no_rag), ("RAG", rag)):
        merged = merge_with_gold(pred[["dec_id"] + KEYS + list(questions)], gold)
        for q in questions:
            m, e = _to_number(merged[f"{q}_marca"]), _to_number(merged[f"{q}_expert"])
            d = m - e
            k = cohen_kappa(m, e)
            rows.append({"Question": q, "Condition": cond, "n": int(len(d)),
                         "Agreement (%)": round(percent_agreement(m, e), 2),
                         "Cohen's k": None if k is None else round(k, 3),
                         "Mean difference": round(d.mean(), 3),
                         "Overestimation (n)": int((d > 0).sum()), "Agreement (n)": int((d == 0).sum()),
                         "Underestimation (n)": int((d < 0).sum())})
    table = pd.DataFrame(rows).sort_values(["Question", "Condition"], key=lambda c: c.map(
        {"Q2": 0, "Q5": 1, "Q10": 2, "Non-RAG": 0, "RAG": 1}), ignore_index=True)
    res.mkdir(parents=True, exist_ok=True)
    table.to_csv(res / "rag_ablation_agreement.csv", index=False)
    return table


# Backward-compatible names
def question_level_agreement(merged, questions=ALL_QUESTIONS):
    return question_level_table(merged, questions)


def classification_level_agreement(merged, questions=TOTAL_QUESTIONS):
    return per_category_table(confusion_matrix(merged, questions))
