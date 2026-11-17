# ChunkGen

ChunkGen is a small Python library for generating a seeded, low-rank NumPy matrix and iterating over its rows in chunks. The project metadata declares version 0.1.0. Its chunked API divides an already-generated matrix; it does not generate the matrix as a memory-bounded stream.

## Requirements and installation

- Python 3.9 or newer
- NumPy 1.24 or newer

Install the project from a checkout:

```bash
python -m pip install .
```

For local development, install it in editable mode and install the test dependencies:

```bash
python -m pip install -e .
python -m pip install -r requirements.txt
```

## Usage

```python
from chunkgen import GenRecipe, chunk_boundaries, generate, generate_chunked

recipe = GenRecipe(rows=64, cols=128, rank=8, seed=0)
matrix = generate(recipe)  # shape: (64, 128)

# Each yielded array contains at most 16 rows.
for chunk in generate_chunked(recipe, chunk_rows=16):
    row_means = chunk.mean(axis=1)
    # Process row_means here rather than retaining all chunks.

print(chunk_boundaries(rows=64, chunk_rows=16))
# [(0, 16), (16, 32), (32, 48), (48, 64)]
```

`GenRecipe(rows, cols, rank=8, seed=0)` validates positive dimensions and rank, and caps the effective rank at the smaller matrix dimension. `generate(recipe)` builds the matrix from seeded random factors. `generate_chunked(recipe, chunk_rows)` yields row slices, with a possibly shorter final chunk. `chunk_boundaries(rows, chunk_rows)` returns the corresponding `(start, end)` row ranges. Invalid dimensions, ranks, or chunk sizes raise `ValueError`.

## Memory and scope

`generate_chunked` calls `generate` first and materializes the complete matrix before yielding its first chunk. Its chunks are slices of that matrix. Chunk-wise consumer code can avoid building a second full-size result or retaining every chunk, but this API does **not** provide constant-memory matrix generation or eliminate the full matrix's memory cost.

The repository contains a Python module and tests, but no command-line interface, persistent streaming generator, benchmark suite, or performance measurements. No claims about production readiness or performance are made here.

## Tests and CI

On the checked-out repository revision, `python -m pytest -q` completed with **9 passed**. The tests cover output shape and determinism, invalid arguments, chunk reconstruction, and chunk-boundary behavior. The CI workflow is configured to run tests and smoke checks on Python 3.10, 3.11, and 3.12; no result from a remote CI run is reported here.

## License

The repository's `LICENSE` file contains the MIT License, with copyright notice `Copyright (c) 2026 Kapildhinwa2010-dev`. See [LICENSE](LICENSE) for the full terms.
