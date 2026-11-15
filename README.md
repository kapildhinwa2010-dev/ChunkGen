# ChunkGen

[![CI](https://github.com/kapildhinwa2010-dev/ChunkGen/actions/workflows/ci.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/ChunkGen/actions/workflows/ci.yml)
[![CodeQL](https://github.com/kapildhinwa2010-dev/ChunkGen/actions/workflows/codeql.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/ChunkGen/actions/workflows/codeql.yml)


Chunked matrix generation: row-chunk iteration with constant per-chunk consumer memory.

**Part of the [Kapildhinwa2010-dev Toolkit](https://github.com/kapildhinwa2010-dev/Kapildhinwa2010-dev).**

## What it does

ChunkGen owns the CHUNKED mode of the DNA_DYNIMO generator family:
generate_chunked() iterates the canonical recipe matrix in row
chunks, and chunk_boundaries() plans the (start, end) ranges. The
HONEST LIMITATION (also in the docstring): the function itself still
materializes the full matrix; the constant-memory benefit applies to
the CONSUMER. For true never-materialized streaming see StreamGen.

## Install

```bash
pip install -r requirements.txt
```

## Quick start

```python
from chunkgen import GenRecipe, generate_chunked, chunk_boundaries

recipe = GenRecipe(rows=64, cols=128, rank=8, seed=0)
for chunk in generate_chunked(recipe, chunk_rows=16):
    process(chunk)  # your chunk-wise work
print(chunk_boundaries(64, 16))
```

## API

- GenRecipe(rows, cols, rank, seed)
- generate(recipe) - the canonical matrix
- generate_chunked(recipe, chunk_rows) - row-chunk iterator
- chunk_boundaries(rows, chunk_rows) - (start, end) planning ranges

## Honest results and limitations

- Peak memory inside generate_chunked is still the FULL array; only
  the consumer side is constant-memory.
- vstack of chunks equals the full matrix - no per-chunk randomness.

## License

MIT - see [LICENSE](LICENSE).
