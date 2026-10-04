import unittest

from sted import compute_sted

from _helpers import IdentityTextEncoder


class UnorderedObjectAlignmentTest(unittest.TestCase):
    def test_object_storage_order_does_not_change_sted(self):
        document_a = {"alpha": {"x": 1}, "beta": {"y": 2}}
        document_b = {"beta": {"y": 99}, "alpha": {"x": -5}}
        result = compute_sted(document_a, document_b, encoder=IdentityTextEncoder())
        self.assertAlmostEqual(result.distance, 0.0)
        self.assertAlmostEqual(result.similarity, 1.0)


if __name__ == "__main__":
    unittest.main()
