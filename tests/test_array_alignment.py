import unittest

from sted import compute_sted

from _helpers import IdentityTextEncoder


class OrderedArrayAlignmentTest(unittest.TestCase):
    def test_permuted_structural_elements_are_not_treated_as_unordered(self):
        document_a = [{"alpha": 1}, {"beta": 2}]
        document_b = [{"beta": 2}, {"alpha": 1}]
        result = compute_sted(document_a, document_b, encoder=IdentityTextEncoder())
        self.assertGreater(result.distance, 0.0)
        self.assertLess(result.similarity, 1.0)


if __name__ == "__main__":
    unittest.main()
