import unittest

import numpy as np

from sted.alignment import AlignmentEngine
from sted.json_tree import json_to_tree


class CompleteSubtreeCostTest(unittest.TestCase):
    def test_deletion_charges_every_node_in_subtree(self):
        source = json_to_tree({"a": {"b": 1, "c": {"d": 2}}})
        target = json_to_tree({})
        embeddings_a = np.ones((len(source), 2), dtype=np.float32)
        embeddings_b = np.ones((len(target), 2), dtype=np.float32)
        engine = AlignmentEngine(source, target, embeddings_a, embeddings_b)
        node_a = source.nodes[source.root].children[0]
        self.assertEqual(engine.deletion_cost(node_a), 4.0)


if __name__ == "__main__":
    unittest.main()
