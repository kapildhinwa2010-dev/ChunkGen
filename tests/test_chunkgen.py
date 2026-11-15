import numpy as np
import pytest

from chunkgen import GenRecipe, chunk_boundaries, generate, generate_chunked


def test_generate_shape():
    recipe = GenRecipe(rows=16, cols=32, rank=4, seed=0)
    assert generate(recipe).shape == (16, 32)


def test_generate_deterministic():
    recipe = GenRecipe(rows=8, cols=8, seed=3)
    assert np.array_equal(generate(recipe), generate(recipe))


def test_recipe_rejects_bad_args():
    with pytest.raises(ValueError):
        GenRecipe(rows=0, cols=8)


def test_chunked_reassembles_to_full():
    recipe = GenRecipe(rows=10, cols=7, rank=3, seed=2)
    chunks = list(generate_chunked(recipe, chunk_rows=4))
    rebuilt = np.vstack(chunks)
    assert np.array_equal(rebuilt, generate(recipe))
    assert all(c.shape[0] <= 4 for c in chunks)


def test_chunked_rejects_bad_chunk():
    with pytest.raises(ValueError):
        list(generate_chunked(GenRecipe(rows=4, cols=4), chunk_rows=0))


# --- chunk planning (ChunkGen-specific) ---


def test_chunk_boundaries_exact_division():
    assert chunk_boundaries(8, 4) == [(0, 4), (4, 8)]


def test_chunk_boundaries_uneven_last_chunk():
    assert chunk_boundaries(10, 4) == [(0, 4), (4, 8), (8, 10)]


def test_chunk_boundaries_match_chunks():
    recipe = GenRecipe(rows=9, cols=3, rank=2, seed=5)
    bounds = chunk_boundaries(9, 4)
    chunks = list(generate_chunked(recipe, chunk_rows=4))
    assert len(bounds) == len(chunks)
    for (start, end), chunk in zip(bounds, chunks):
        assert chunk.shape[0] == end - start


def test_chunk_boundaries_rejects_bad_args():
    with pytest.raises(ValueError):
        chunk_boundaries(0, 4)
    with pytest.raises(ValueError):
        chunk_boundaries(8, 0)