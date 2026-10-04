import unittest

import torch

from sted_gnn.pair_model import PairScoringHead, pair_representation


class PairSymmetryTest(unittest.TestCase):
    def test_representation_and_score_are_exchange_invariant(self):
        torch.manual_seed(7)
        z_a = torch.randn(9, 512)
        z_b = torch.randn(9, 512)
        self.assertTrue(torch.allclose(pair_representation(z_a, z_b), pair_representation(z_b, z_a)))
        head = PairScoringHead(512)
        self.assertTrue(torch.allclose(head(z_a, z_b), head(z_b, z_a), atol=1e-7, rtol=1e-6))


if __name__ == "__main__":
    unittest.main()
