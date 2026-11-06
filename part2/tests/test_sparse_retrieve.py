import unittest

import numpy as np
import torch
from scipy.sparse import csr_matrix

from part2.dense_retrieve import paper_run
from part2.sparse_retrieve import search, to_csr


class TestSparse(unittest.TestCase):
    def setUp(self):
        # 4 paragraphs x 5 "vocabulary words"; mostly zeros like real SPLADE vectors
        self.docs = csr_matrix(np.array([[2.0, 0, 0, 0, 1.0],
                                         [0, 1.5, 0, 0, 0],
                                         [1.0, 1.0, 0, 0, 0],
                                         [0, 0, 0, 3.0, 0]], dtype=np.float32))
        self.ids = ["A_1", "A_2", "B_1", "C_1"]

    def test_scores_are_dot_products_over_shared_words(self):
        q = csr_matrix(np.array([[1.0, 2.0, 0, 0, 0]], dtype=np.float32))
        s, i = search(q, self.docs, k=4)
        # A_1: 2*1 = 2 | A_2: 1.5*2 = 3 | B_1: 1+2 = 3 | C_1: 0
        self.assertEqual(set(i[0, :2].tolist()), {1, 2})
        self.assertEqual(i[0, 2].item(), 0)
        self.assertAlmostEqual(s[0, 0].item(), 3.0)
        self.assertAlmostEqual(s[0, 3].item(), 0.0)

    def test_matches_brute_force_with_query_chunks(self):
        rng = np.random.default_rng(0)
        docs = csr_matrix((rng.random((50, 30)) > 0.8) * rng.random((50, 30)), dtype=np.float32)
        qs = csr_matrix((rng.random((7, 30)) > 0.7) * rng.random((7, 30)), dtype=np.float32)
        s, i = search(qs, docs, k=5, chunk=3)
        full = (qs @ docs.T).toarray()
        for n in range(7):
            np.testing.assert_allclose(s[n].numpy(), np.sort(full[n])[::-1][:5], rtol=1e-5)

    def test_works_with_paper_run(self):
        q = csr_matrix(np.array([[1.0, 0, 0, 0, 1.0]], dtype=np.float32))
        s, i = search(q, self.docs, k=4)
        self.assertEqual(paper_run(["2001"], s, i, self.ids)["2001"][0][0], "A")

    def test_to_csr_from_torch_sparse(self):
        t = torch.tensor([[0.0, 1.5, 0.0], [2.0, 0.0, 0.0]]).to_sparse()
        m = to_csr(t)
        self.assertEqual(m.shape, (2, 3))
        self.assertEqual(m.nnz, 2)
        self.assertAlmostEqual(m[1, 0], 2.0)


if __name__ == "__main__":
    unittest.main()
