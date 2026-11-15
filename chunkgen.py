# ChunkGen - Chunked matrix generation with constant per-chunk memory
# Part of the Kapildhinwa2010-dev Toolkit.

"""Row-chunked views of a seeded recipe matrix.

Lineage: DNA_DYNIMO generators. This repo owns the CHUNKED mode: iterate
the canonical matrix in row chunks without materializing the whole array
up-front in the consumer, plus chunk_boundaries() for planning.

HONEST LIMITATION: generate_chunked slices a fully materialized matrix,
so peak memory inside this function is still the full array; the
constant-memory benefit applies to the CONSUMER processing chunk by
chunk. For true never-materialized streaming see StreamGen.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["GenRecipe", "generate", "generate_chunked", "chunk_boundaries"]


class GenRecipe:
    """A seeded low-rank recipe: rows x cols output from rank x (rows+cols) latents."""

    def __init__(self, rows: int, cols: int, rank: int = 8, seed: int = 0) -> None:
        if rows < 1 or cols < 1:
            raise ValueError("rows and cols must be >= 1")
        if rank < 1:
            raise ValueError("rank must be >= 1")
        self.rows = rows
        self.cols = cols
        self.rank = min(rank, min(rows, cols))
        self.seed = seed

    def _latent(self, seed_offset: int) -> np.ndarray:
        rng = np.random.default_rng(self.seed + seed_offset)
        left = rng.standard_normal((self.rows, self.rank))
        right = rng.standard_normal((self.rank, self.cols))
        return (left @ right) / math.sqrt(self.rank)


def generate(recipe: GenRecipe) -> np.ndarray:
    """The canonical matrix of the recipe (seed + 0)."""
    return recipe._latent(0)

def generate_chunked(recipe: GenRecipe, chunk_rows: int):
    """Yield ``generate(recipe)`` in row-chunks; vstack of chunks == full."""
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be >= 1")
    full = generate(recipe)
    for start in range(0, recipe.rows, chunk_rows):
        yield full[start:start + chunk_rows]


def chunk_boundaries(rows: int, chunk_rows: int) -> list:
    """(start, end) row ranges produced by chunking ``rows`` rows.

    Useful for planning: the last chunk may be shorter than chunk_rows.
    """
    if rows < 1 or chunk_rows < 1:
        raise ValueError("rows and chunk_rows must be >= 1")
    return [(start, min(start + chunk_rows, rows)) for start in range(0, rows, chunk_rows)]