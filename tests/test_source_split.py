import unittest

from training.source_split import audit_source_partitions, split_source_documents


class SourceFirstSplitTest(unittest.TestCase):
    def test_source_ids_and_content_hashes_do_not_cross_splits(self):
        records = [
            {"source_document_id": "s1", "json": {"a": 1}},
            {"source_document_id": "s2", "json": {"a": 1}},
            {"source_document_id": "s3", "json": {"b": 2}},
            {"source_document_id": "s4", "json": {"c": 3}},
            {"source_document_id": "s5", "json": {"d": 4}},
            {"source_document_id": "s6", "json": {"e": 5}},
        ]
        partitions = split_source_documents(records, seed=13)
        report = audit_source_partitions(partitions)
        self.assertTrue(all(value == 0 for value in report["overlaps"]["source_ids"].values()))
        self.assertTrue(all(value == 0 for value in report["overlaps"]["content_hashes"].values()))
        locations = {
            row["source_document_id"]: split
            for split, rows in partitions.items()
            for row in rows
        }
        self.assertEqual(locations["s1"], locations["s2"])


if __name__ == "__main__":
    unittest.main()
