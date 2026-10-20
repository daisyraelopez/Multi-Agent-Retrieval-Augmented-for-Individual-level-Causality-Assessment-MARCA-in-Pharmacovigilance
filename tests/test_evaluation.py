"""Tests on the frozen MARCA scores.

They pin every number the manuscript reports, computed from the files in this repository,
so the paper and the data cannot drift apart. If a score is changed in expert_kb.xlsx,
regenerate the CSV files, update the expected values below and the manuscript together.

Run from the repository root:  python -m unittest -v tests.test_evaluation
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from software.evaluation import (
    cohen_kappa, confusion_matrix, directional_table, merge_with_gold, naranjo_category,
    percent_agreement, question_level_table, rag_ablation, total_scores, validate_naranjo_scores,
)

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "output/results/predictions_rag.csv"
NO_RAG = ROOT / "output/results/predictions_no_rag.csv"
GOLD = ROOT / "data/gold_standard/gold_standard.csv"
KB = ROOT / "data/knowledge_base/expert_kb.xlsx"
CORRECTIONS = ROOT / "data/knowledge_base/WORKBOOK_CORRECTIONS.txt"

# Values as they follow from the current data (DEC 310 Q2 = +2).
# If DEC 310 Q2 is set to 0 after adjudication: Q2 agreement 96.86% (rounded 96.9), Q2 kappa 0.496,
# Q2 mean MARCA 0.09, exact total-score agreement 71.64%; Q2 directional row 43 / 1513 / 6, mean +0.048.
TABLE1 = {  # question: (agreement %, kappa or None, mean MARCA, mean expert)
    "Q1": (89.3, 0.787, 0.53, 0.47),    # reported descriptively; >95% threshold not met
    "Q2": (96.8, 0.491, 0.09, 0.04),
    "Q3": (100.0, 1.000, 0.01, 0.01),
    "Q4": (100.0, None, 0.00, 0.00),
    "Q5": (85.4, 0.269, -0.85, -0.50),
    "Q6": (100.0, None, 0.00, 0.00),
    "Q7": (100.0, None, 0.00, 0.00),
    "Q8": (100.0, None, 0.00, 0.00),
    "Q9": (100.0, None, 0.00, 0.00),
    "Q10": (96.8, 0.922, 0.70, 0.72),
}
TABLE2 = {  # question: (MARCA > expert, equal, MARCA < expert, mean difference)
    "Q2": (44, 1512, 6, 0.049),
    "Q3": (0, 1562, 0, 0.0), "Q4": (0, 1562, 0, 0.0), "Q6": (0, 1562, 0, 0.0),
    "Q7": (0, 1562, 0, 0.0), "Q8": (0, 1562, 0, 0.0), "Q9": (0, 1562, 0, 0.0),
    "Q5": (23, 1334, 205, -0.350),
    "Q10": (8, 1512, 42, -0.022),
}
TABLE3 = {  # expert category: MARCA Doubtful, Possible, Probable
    "Doubtful": (765, 100, 2),
    "Possible": (195, 470, 20),
    "Probable": (0, 4, 6),
}
TABLE4 = {  # (question, condition): (agreement %, kappa or None, mean diff, over, equal, under)
    ("Q2", "Non-RAG"): (98.33, 0.000, -0.017, 0, 59, 1),
    ("Q2", "RAG"): (100.00, None, 0.000, 0, 60, 0),
    ("Q5", "Non-RAG"): (65.00, 0.016, -0.967, 1, 39, 20),
    ("Q5", "RAG"): (83.33, 0.571, -0.500, 0, 50, 10),
    ("Q10", "Non-RAG"): (18.33, 0.007, -0.817, 0, 11, 49),
    ("Q10", "RAG"): (100.00, 1.000, 0.000, 0, 60, 0),
}


class FrozenScores(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pred = pd.read_csv(PRED, dtype={"case_id": str})
        cls.gold = pd.read_csv(GOLD, dtype={"case_id": str})
        cls.no_rag = pd.read_csv(NO_RAG, dtype={"case_id": str})
        cls.merged = merge_with_gold(cls.pred, cls.gold)

    # ── Data integrity ────────────────────────────────────────────────────────
    def test_valid_score_domains(self):
        validate_naranjo_scores(self.pred, label="MARCA")
        validate_naranjo_scores(self.gold, label="human")

    def test_validation_rejects_invalid(self):
        bad = self.pred.copy()
        bad.loc[bad["dec_id"] == 310, "Q2"] = 1     # the pre-correction value, not a Naranjo Q2 score
        with self.assertRaises(ValueError):
            validate_naranjo_scores(bad)

    def test_csv_files_match_workbook(self):
        """gold_standard.csv and predictions_rag.csv are exact exports of expert_kb.xlsx."""
        kb = pd.read_excel(KB).set_index("id").sort_index()
        pred = self.pred.set_index("dec_id").sort_index()
        gold = self.gold.set_index("dec_id").sort_index()
        self.assertEqual(list(kb.index), list(pred.index))
        self.assertEqual(list(kb.index), list(gold.index))
        for i in range(1, 11):
            np.testing.assert_array_equal(pred[f"Q{i}"].values, kb[f"q{i}_score_ai"].values)
            np.testing.assert_array_equal(gold[f"Q{i}"].values, kb[f"q{i}_score_human"].values)

    def test_post_hoc_correction_is_logged(self):
        """DEC 310 Q2 was edited by hand on 2026-10-19. The test does not decide the right value
        (the agent's rationale supports 0; the stored value is +2); it requires the edit to be valid
        and logged. Report it in the manuscript."""
        row = self.pred.loc[self.pred["dec_id"] == 310]
        self.assertEqual(len(row), 1)
        self.assertIn(int(row.iloc[0]["Q2"]), {-1, 0, 2})
        self.assertIn("dec_id=310", CORRECTIONS.read_text(encoding="utf-8"))

    # ── Cohort (Results, first paragraph; Supplementary Table 4) ─────────────
    def test_cohort_size(self):
        per_case = self.merged.groupby("case_id").size()
        self.assertEqual(len(self.merged), 1562)
        self.assertEqual(len(per_case), 215)
        self.assertEqual(per_case.median(), 4)
        self.assertAlmostEqual(per_case.mean(), 7.27, places=2)
        self.assertAlmostEqual(per_case.std(), 27.18, places=2)
        self.assertEqual(per_case.max(), 371)
        self.assertAlmostEqual(per_case.max() / len(self.merged) * 100, 23.8, places=1)

    # ── Table 1 and Q1 ───────────────────────────────────────────────────────
    def test_table1_question_level(self):
        t = question_level_table(self.merged).set_index("Question")
        for q, (agree, kappa, m_marca, m_expert) in TABLE1.items():
            with self.subTest(question=q):
                self.assertAlmostEqual(t.loc[q, "Agreement (%)"], agree, places=1)
                if kappa is None:
                    self.assertTrue(pd.isna(t.loc[q, "Cohen's k"]))
                else:
                    self.assertAlmostEqual(t.loc[q, "Cohen's k"], kappa, places=3)
                self.assertAlmostEqual(t.loc[q, "Mean MARCA"], m_marca, places=2)
                self.assertAlmostEqual(t.loc[q, "Mean Human"], m_expert, places=2)

    # ── Table 2 ──────────────────────────────────────────────────────────────
    def test_table2_directional(self):
        t = directional_table(self.merged).set_index("Question")
        for q, (over, equal, under, mean_diff) in TABLE2.items():
            with self.subTest(question=q):
                self.assertEqual(int(t.loc[q, "MARCA > Expert (n)"]), over)
                self.assertEqual(int(t.loc[q, "MARCA = Expert (n)"]), equal)
                self.assertEqual(int(t.loc[q, "MARCA < Expert (n)"]), under)
                self.assertAlmostEqual(t.loc[q, "Mean difference"], mean_diff, places=3)

    # ── Classification level (Results; Abstract) ─────────────────────────────
    def test_final_classification(self):
        m, h = total_scores(self.merged)
        cm, ch = m.map(naranjo_category), h.map(naranjo_category)
        self.assertAlmostEqual(percent_agreement(cm, ch), 79.449, places=3)   # report 79.4%
        self.assertAlmostEqual(cohen_kappa(cm, ch), 0.587, places=3)
        self.assertAlmostEqual(percent_agreement(m, h), 71.575, places=3)    # report 71.6%, not 71.5%
        self.assertAlmostEqual(m.mean(), 0.48, places=2)
        self.assertAlmostEqual(h.mean(), 0.74, places=2)
        self.assertAlmostEqual((m - h).mean(), -0.262, places=3)
        self.assertFalse((cm == "Definite").any() or (ch == "Definite").any())

    # ── Table 3 ──────────────────────────────────────────────────────────────
    def test_table3_confusion_matrix(self):
        conf = confusion_matrix(self.merged)
        for expert_cat, counts in TABLE3.items():
            with self.subTest(expert=expert_cat):
                got = tuple(int(conf.loc[expert_cat, c]) for c in ("Doubtful", "Possible", "Probable"))
                self.assertEqual(got, counts)

    # ── Table 4 (60-DEC retrospective ablation) ──────────────────────────────
    def test_table4_rag_ablation(self):
        with tempfile.TemporaryDirectory() as tmp:          # do not overwrite output/results
            t = rag_ablation(out_dir=tmp).set_index(["Question", "Condition"])
        for key, (agree, kappa, mean_diff, over, equal, under) in TABLE4.items():
            with self.subTest(row=key):
                row = t.loc[key]
                self.assertEqual(int(row["n"]), 60)
                self.assertAlmostEqual(row["Agreement (%)"], agree, places=2)
                if kappa is None:
                    self.assertTrue(pd.isna(row["Cohen's k"]))
                else:
                    self.assertAlmostEqual(row["Cohen's k"], kappa, places=3)
                self.assertAlmostEqual(row["Mean difference"], mean_diff, places=3)
                self.assertEqual((int(row["Overestimation (n)"]), int(row["Agreement (n)"]),
                                  int(row["Underestimation (n)"])), (over, equal, under))

    def test_ablation_cohort_definition(self):
        """Every DEC in the 60-DEC cohort has at least one non-RAG disagreement on Q2, Q5 or Q10."""
        g = self.gold.set_index("dec_id").loc[self.no_rag["dec_id"]]
        n = self.no_rag.set_index("dec_id")
        disagree = (n[["Q2", "Q5", "Q10"]].values != g[["Q2", "Q5", "Q10"]].values).any(axis=1)
        self.assertTrue(disagree.all())
        self.assertEqual(g["case_id"].nunique(), 32)

    @unittest.expectedFailure
    def test_known_issue_cohort_vs_main_scores(self):
        """KNOWN INCONSISTENCY (expected to fail until resolved).

        Manuscript: the 60 DECs are all DECs whose initial non-RAG score disagreed with the expert,
        and RAG is applied only to those. Then every other DEC should keep a score that agrees with
        the expert on Q2, Q5 and Q10. In predictions_rag.csv, 300 DECs outside the 60 disagree
        (310 in total). Remove the decorator once the cohort definition or the data are corrected.
        """
        m = self.merged[~self.merged["dec_id"].isin(self.no_rag["dec_id"])]
        disagree = pd.concat([m[f"{q}_marca"] != m[f"{q}_expert"] for q in ("Q2", "Q5", "Q10")], axis=1).any(axis=1)
        self.assertEqual(int(disagree.sum()), 0)


if __name__ == "__main__":
    unittest.main()
