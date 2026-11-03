import tempfile
import unittest
from pathlib import Path

from part2.evaluate_part2 import evaluate_level, find_runs, full_only, model_name, per_query

QRELS = {"2001": {"A_1": 2, "B_1": 1}, "2002": {"C_1": 2}, "2003": {"D_1": 1}}


def write(path, rows):
    path.write_text("".join(f"{q}\tQ0\t{d}\t{i + 1}\t{1 - i / 10}\tx\n"
                            for q, docs in rows.items() for i, d in enumerate(docs)))


class TestEvaluatePart2(unittest.TestCase):
    def test_model_name(self):
        self.assertEqual(model_name("marcelsSearchParty_paragraph_densePretrained_study.tsv", "paragraph"),
                         "densePretrained")
        self.assertEqual(model_name("TheSearchParty_paper_bm25_study.tsv", "paper"), "bm25")

    def test_full_only_drops_partial_and_empty_queries(self):
        self.assertEqual(full_only(QRELS), {"2001": {"A_1": 2}, "2002": {"C_1": 2}})

    def test_find_and_evaluate(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            write(d / "t_paragraph_good_study.tsv", {"2001": ["A_1", "X"], "2002": ["C_1"], "2003": ["D_1"]})
            write(d / "t_paragraph_bad_study.tsv", {"2001": ["X", "A_1"], "2002": ["Y"], "2003": ["Z"]})
            write(d / "t_paragraph_good_test.tsv", {"2251": ["A_1"]})          # test runs are ignored
            runs = find_runs("paragraph", [d])
            self.assertEqual(sorted(runs), ["bad", "good"])

            res = evaluate_level("paragraph", runs, QRELS)
            self.assertAlmostEqual(res["scores"]["good"]["mrr (any)"], 1.0)
            self.assertAlmostEqual(res["scores"]["bad"]["mrr (any)"], 0.5 / 3)
            # full variant: only 2001 and 2002 count, good finds both at rank 1
            self.assertAlmostEqual(res["scores"]["good"]["mrr (full)"], 1.0)
            pair = next(iter(res["pairs"].values()))
            self.assertIn("p_value", pair["mrr (any)"])

            pq = per_query(runs, QRELS)
            self.assertAlmostEqual(pq["bad"]["2001"]["mrr"], 0.5)
            self.assertAlmostEqual(pq["bad"]["2002"]["mrr"], 0.0)


if __name__ == "__main__":
    unittest.main()
