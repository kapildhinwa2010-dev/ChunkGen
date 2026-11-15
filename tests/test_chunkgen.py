import pytest
import torch
from chunkgen import ChunkGen, create_chunkgen

class TestChunkGen:
    def test_init(self):
        m = ChunkGen(dim=256, hidden_dim=512)
        assert m.dim == 256
    
    def test_forward(self):
        m = ChunkGen(dim=256)
        x = torch.randn(32, 256)
        assert m(x).shape == (32, 256)
    
    def test_factory(self):
        m = create_chunkgen()
        assert isinstance(m, ChunkGen)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
