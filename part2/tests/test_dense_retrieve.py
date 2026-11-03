import unittest

import torch

from part2.dense_retrieve import paper_of, paper_run, paragraph_run, search


def unit(rows):
    t = torch.tensor(rows, dtype=torch.float32)
    return (t / t.norm(dim=1, keepdim=True)).to(torch.float16)


class TestSearch(unittest.TestCase):
    def setUp(self):
        # 5 paragraphs from 3 papers, 2-number "embeddings"
        self.ids = ["A_1", "A_2", "B_1", "C_1", "C_2"]
        self.docs = unit([[1, 0], [0.9, 0.1], [0, 1], [0.7, 0.7], [0.2, 0.9]])
        self.q = unit([[1, 0]])

    def test_matches_brute_force_with_small_chunks(self):
        full = (self.q.float() @ self.docs.float().T)[0]
        expected = torch.argsort(full, descending=True)[:3].tolist()
        s, i = search(self.q, self.docs, k=3, chunk=2)       # forces 3 chunks
        self.assertEqual(i[0].tolist(), expected)
        self.assertTrue(torch.all(s[0][:-1] >= s[0][1:]))

    def test_k_larger_than_corpus(self):
        s, i = search(self.q, self.docs, k=50, chunk=2)
        self.assertEqual(sorted(i[0].tolist()), [0, 1, 2, 3, 4])

    def test_paragraph_run(self):
        s, i = search(self.q, self.docs, k=5)
        run = paragraph_run(["2001"], s, i, self.ids, k=2)
        self.assertEqual([d for d, _ in run["2001"]], ["A_1", "A_2"])

    def test_paper_run_uses_best_paragraph(self):
        s, i = search(self.q, self.docs, k=5)
        run = paper_run(["2001"], s, i, self.ids, k=10)
        self.assertEqual([d for d, _ in run["2001"]], ["A", "C", "B"])
        self.assertAlmostEqual(run["2001"][0][1], 1.0, places=2)

    def test_paper_of(self):
        self.assertEqual(paper_of("2408.00001_12"), "2408.00001")


if __name__ == "__main__":
    unittest.main()
