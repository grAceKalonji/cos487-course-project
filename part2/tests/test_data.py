import json
import tempfile
import unittest
from pathlib import Path

from part2.data import (clean_text, load_paragraphs, load_qrels, load_queries,
                        train_val_split, write_run)


class TestData(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        (root / "JSON Files").mkdir()
        (root / "qrels").mkdir()
        paper = {"paper_id": "2408.00001", "title": "T", "abstract": "A",
                 "paragraphs": {"2408.00001_1": "RAG works [1, 2] well.",
                                "2408.00001_2": "   ",
                                "2408.00001_3": "See https://x.org now."}}
        (root / "JSON Files" / "2408.00001.json").write_text(json.dumps(paper))
        (root / "Study.json").write_text(json.dumps([{"query_id": 2001, "query": "What is RAG?"}]))
        (root / "qrels" / "Study_paragraph_qrel.tsv").write_text(
            "2001\t0\t2408.00001_1\t2\n2001\t0\t2408.00001_3\t0\n")
        self.root = root

    def tearDown(self):
        self.tmp.cleanup()

    def test_clean_keeps_words_drops_citations_and_urls(self):
        self.assertEqual(clean_text("The model [3-5] is at https://a.b/c ."), "The model is at .")

    def test_paragraphs_skip_empty(self):
        ids, texts = load_paragraphs(self.root / "JSON Files")
        self.assertEqual(ids, ["2408.00001_1", "2408.00001_3"])
        self.assertEqual(texts[0], "RAG works well.")

    def test_queries_and_missing_file(self):
        self.assertEqual(load_queries("study", self.root), [("2001", "What is RAG?")])
        with self.assertRaises(FileNotFoundError):
            load_queries("train", self.root)

    def test_qrels_drop_grade_zero(self):
        self.assertEqual(load_qrels("Study", "paragraph", self.root / "qrels"),
                         {"2001": {"2408.00001_1": 2}})

    def test_split_is_disjoint_and_repeatable(self):
        ids = [str(i) for i in range(1, 2001)]
        train, val = train_val_split(ids)
        self.assertEqual((len(train), len(val)), (1800, 200))
        self.assertFalse(set(train) & set(val))
        self.assertEqual(train_val_split(ids), (train, val))

    def test_write_run_is_tab_separated_top_k(self):
        path = self.root / "run.tsv"
        write_run({"2001": [("b", 0.2), ("a", 0.9), ("c", 0.1)]}, path, "test", top_k=2)
        lines = path.read_text().splitlines()
        self.assertEqual(lines[0], "2001\tQ0\ta\t1\t0.900000\ttest")
        self.assertEqual(len(lines), 2)


if __name__ == "__main__":
    unittest.main()
