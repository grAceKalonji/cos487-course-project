import csv
import json
import tempfile
import unittest
from pathlib import Path

from part2.annotate import highlight, load_grades, paragraph_info, save_grades, summary, top_paragraphs
from part2.difficulty_chart import mrr_by_difficulty
from part2.package_results import check_qrel, check_run


class TestAnnotate(unittest.TestCase):
    def test_round_trip_and_summary(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "q.tsv"
            save_grades({"2252": ("P_2", 1), "2251": ("P_1", 2)}, path)
            self.assertEqual(path.read_text().splitlines()[0], "2251\t0\tP_1\t2")
            self.assertEqual(load_grades(path), {"2251": ("P_1", 2), "2252": ("P_2", 1)})
            self.assertIn("Precision@1 (any) = 1.000", summary(load_grades(path), 250))

    def test_top_paragraphs_and_info(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / "run.tsv"
            run.write_text("2251\tQ0\tP_3\t1\t9.0\tx\n2251\tQ0\tP_1\t2\t8.0\tx\n")
            self.assertEqual(top_paragraphs(run), {"2251": "P_3"})
            Path(d, "P.json").write_text(json.dumps({"title": "A title", "paragraphs": {"P_3": "Text [1]."}}))
            self.assertEqual(paragraph_info("P_3", d), ("A title", "Text ."))

    def test_highlight_skips_stopwords(self):
        out = highlight("The dataset uses vowel recordings.", "What is the vowel dataset?")
        self.assertIn("\033[1;33mvowel", out)
        self.assertNotIn("\033[1;33mThe", out)


class TestDifficulty(unittest.TestCase):
    def test_means_per_difficulty(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "pq.csv"
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["query_id", "bm25 ndcg@5", "bm25 mrr", "sparse ndcg@5", "sparse mrr"])
                w.writerows([["1", 0, 1.0, 0, 0.5], ["2", 0, 0.0, 0, 1.0], ["3", 0, 0.5, 0, 0.5]])
            means, counts = mrr_by_difficulty(path, {"1": "Easy", "2": "Easy", "3": "Hard"})
            self.assertEqual(counts, {"Easy": 2, "Medium": 0, "Hard": 1})
            self.assertAlmostEqual(means["bm25"]["Easy"], 0.5)
            self.assertAlmostEqual(means["sparse"]["Easy"], 0.75)
            self.assertEqual(means["sparse"]["Medium"], 0.0)


class TestPackage(unittest.TestCase):
    def test_good_and_bad_runs(self):
        with tempfile.TemporaryDirectory() as d:
            good = Path(d) / "good.tsv"
            good.write_text("1\tQ0\ta\t1\t2.0\tr\n1\tQ0\tb\t2\t1.0\tr\n2\tQ0\tc\t1\t1.0\tr\n")
            problems = check_run(good, ["1", "2"], top_k=2)
            self.assertEqual([p for p in problems if not p.startswith("note")], [])
            spaces = Path(d) / "spaces.tsv"
            spaces.write_text("1 Q0 a 1 2.0 r\n")
            self.assertIn("columns", check_run(spaces, ["1"])[0])
            self.assertTrue(any("missing" in p for p in check_run(good, ["1", "2", "3"], top_k=2)))

    def test_qrel_check(self):
        with tempfile.TemporaryDirectory() as d:
            q = Path(d) / "q.tsv"
            q.write_text("2251\t0\tP_1\t2\n")
            self.assertEqual(check_qrel(q, ["2251"]), [])
            self.assertTrue(check_qrel(q, ["2251", "2252"]))


if __name__ == "__main__":
    unittest.main()
