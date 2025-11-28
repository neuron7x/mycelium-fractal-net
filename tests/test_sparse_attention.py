"""Tests for sparse attention module."""
import torch

from mycelium_fractal_net import SPARSE_TOPK
from mycelium_fractal_net.model import SparseAttention


def test_sparse_attention_topk_default() -> None:
    """Verify sparse attention uses default topk=4."""
    attn = SparseAttention(embed_dim=32)
    assert attn.topk == SPARSE_TOPK
    assert attn.topk == 4


def test_sparse_attention_output_shape() -> None:
    """Test sparse attention preserves input shape."""
    batch_size = 2
    seq_len = 8
    embed_dim = 32

    attn = SparseAttention(embed_dim=embed_dim)
    x = torch.randn(batch_size, seq_len, embed_dim)

    out = attn(x)

    assert out.shape == x.shape


def test_sparse_attention_no_nan() -> None:
    """Test sparse attention doesn't produce NaN values."""
    attn = SparseAttention(embed_dim=16)
    x = torch.randn(2, 4, 16)

    out = attn(x)

    assert not torch.isnan(out).any()


def test_sparse_attention_gradient_flow() -> None:
    """Test that gradients flow through sparse attention."""
    attn = SparseAttention(embed_dim=16)
    x = torch.randn(2, 4, 16, requires_grad=True)

    out = attn(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_sparse_attention_handles_short_sequence() -> None:
    """Test sparse attention handles sequences shorter than topk."""
    attn = SparseAttention(embed_dim=16, topk=4)
    x = torch.randn(1, 2, 16)  # seq_len=2 < topk=4

    out = attn(x)

    assert out.shape == x.shape
    assert not torch.isnan(out).any()
