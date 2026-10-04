import unittest

from sted import compute_sted
from sted.json_tree import json_to_tree, node_descriptor

from _helpers import IdentityTextEncoder


class KeyPathRepresentationTest(unittest.TestCase):
    def test_scalar_values_are_not_present_in_descriptors(self):
        tree_a = json_to_tree({"alpha": 1, "nested": {"beta": "scalar-a"}})
        tree_b = json_to_tree({"alpha": 999, "nested": {"beta": "scalar-b"}})
        descriptors_a = [node_descriptor(tree_a, i) for i in range(len(tree_a))]
        descriptors_b = [node_descriptor(tree_b, i) for i in range(len(tree_b))]
        self.assertEqual(descriptors_a, descriptors_b)
        self.assertNotIn("scalar-a", "\n".join(descriptors_a))
        result = compute_sted(
            {"alpha": 1, "nested": {"beta": "scalar-a"}},
            {"alpha": 999, "nested": {"beta": "scalar-b"}},
            encoder=IdentityTextEncoder(),
        )
        self.assertAlmostEqual(result.similarity, 1.0)


if __name__ == "__main__":
    unittest.main()
