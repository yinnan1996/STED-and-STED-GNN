# STED and STED-GNN reference implementation

This repository contains the reference implementations of STED and STED-GNN
described in the paper. It focuses on the method definitions, data
preprocessing, and model execution. The manuscript's full experimental result
artifacts and private evaluation annotations are not part of this repository.

The release includes:

- STED for JSON semantic-structural similarity;
- STED-GNN for learning a reusable document embedding and approximating STED;
- source-document-first split and leakage-audit utilities; and
- small command-line and Python examples.

It does not reproduce every table or metric in the manuscript. No trained
checkpoint is included.

## Method correspondence

STED converts each JSON document into a rooted tree. A node descriptor contains
only its key and root-to-node path. Raw scalar values are deliberately excluded
from semantic encoding. Descriptors are encoded by a frozen
`bert-base-multilingual-cased` encoder using its 768-dimensional CLS output.

The substitution cost is `1 - cosine(h_v, h_u)`. Insertion and deletion each
cost one. Object children are unordered and use augmented Hungarian alignment;
array children retain their order and use sequence dynamic programming.
Unmatched nodes incur the cost of inserting or deleting their complete rooted
subtrees. Child forests with different alignment modes are not directly
matched: source children are deleted and target children are inserted. The
final similarity is

```text
STED(T_a, T_b) = 1 - d_STED(T_a, T_b) / (|V_a| + |V_b|).
```

STED-GNN uses the same frozen node encoder, a shared two-layer GCN, global mean
pooling, and a 512-dimensional document representation. Its pair feature is
exchange invariant:

```text
p(a,b) = [abs(z_a - z_b); z_a * z_b].
```

The MLP maps the resulting 1024-dimensional vector through a 512-dimensional
ReLU layer and a sigmoid output. Training uses mean squared error against STED
scores.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m pip install -r requirements.txt
```

The first BERT-backed run downloads `bert-base-multilingual-cased` from the
Hugging Face Hub unless it is already cached.

## Examples

Compute STED for the included pair:

```bash
python scripts/compute_sted.py examples/example_a.json examples/example_b.json
```

Run the Python example:

```bash
python examples/run_example.py
```

After supplying a trained STED-GNN checkpoint and your own JSONL inputs, encode
documents or score pairs with these command templates:

```bash
python scripts/encode_documents.py documents.jsonl checkpoint.pt embeddings.jsonl
python scripts/score_pairs.py pairs.jsonl checkpoint.pt predictions.jsonl
```

These commands expect JSONL records with a `json` field for documents and
`doc_a`/`doc_b` fields for pairs. A fresh checkout does not contain the three
placeholder input/output files or a checkpoint, so these are not bundled smoke
tests.

## Source-document-first training

`training/source_split.py` partitions source JSON documents before variants or
pairs are generated. Equal normalized-content SHA-256 hashes are kept in the
same partition, and the audit rejects overlap in source IDs or content hashes.
`training/pair_generation.py` then creates pairs independently inside each
partition. The training entry point accepts already separated train,
validation, and test pair files and never performs a random pair-level split.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests cover pair symmetry, object-order invariance, ordered array
alignment, complete subtree costs, alignment-mode mismatch, source-first
splitting, and exclusion of scalar values from node descriptors.

## License

This code is released under the MIT License. See LICENSE for details.
