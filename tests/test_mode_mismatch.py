import unittest

from sted import compute_sted

from _helpers import IdentityTextEncoder


class ModeMismatchTest(unittest.TestCase):
    def test_incompatible_child_modes_delete_and_insert_children(self):
        result = compute_sted({"alpha": 1}, [2], encoder=IdentityTextEncoder())
        self.assertAlmostEqual(result.distance, 2.0)


if __name__ == "__main__":
    unittest.main()
