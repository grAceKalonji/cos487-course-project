import json
import tempfile
import unittest
from pathlib import Path

from part2.dense_finetune import build_pairs, check_train_only, load_paragraph_texts, loss_rows


class TestFinetuneHelpers(unittest.TestCase):
    def test_pairs_one_per_relevant_paragraph(self):
        pairs = build_pairs(["1", "2"], {"1": "q one", "2": "q two"},
                            {"1": {"A_1": 2, "A_2": 1}, "2": {"B_9": 2}},
                            {"A_1": "para a1", "A_2": "para a2"})
        self.assertEqual(pairs, [("q one", "para a1"), ("q one", "para a2")])   # B_9 text missing -> skipped

    def test_refuses_study_or_test_ids(self):
        check_train_only(["1", "2000"])
        with self.assertRaises(ValueError):
            check_train_only(["5", "2001"])

    def test_loss_rows_keep_train_and_val(self):
        log = [{"loss": 2.0, "step": 1, "epoch": 0.5},
               {"eval_loss": 1.5, "eval_val_cosine_ndcg@10": 0.4, "step": 2, "epoch": 1.0},
               {"train_runtime": 3.0}]
        rows = loss_rows(log)
        self.assertEqual([r["train_loss"] for r in rows], [2.0, ""])
        self.assertEqual(rows[1]["val_loss"], 1.5)
        self.assertEqual(rows[1]["val_ndcg@10"], 0.4)

    def test_paragraph_texts_only_needed_plus_distractors(self):
        with tempfile.TemporaryDirectory() as d:
            for pid in ("P1", "P2", "P3"):
                Path(d, f"{pid}.json").write_text(json.dumps(
                    {"paper_id": pid, "paragraphs": {f"{pid}_1": "x [1]", f"{pid}_2": "y"}}))
            texts = load_paragraph_texts({"P1_2"}, extra_papers=1, json_dir=d)
            self.assertIn("P1_2", texts)
            self.assertNotIn("P1_1", texts)           # same paper, not asked for
            self.assertEqual(len(texts), 3)           # P1_2 + both paragraphs of one distractor paper


if __name__ == "__main__":
    unittest.main()
