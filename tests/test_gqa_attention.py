import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import pytest

from src.models.block_masked_attention import BlockMaskedAttention
from src.models.gqa.attention import GQAAttention


def _make_sequence_ids(batch_size: int, seq_len: int) -> torch.Tensor:
    """Пример packed batching: две последовательности + PAD в конце."""
    sequence_ids = torch.ones(batch_size, seq_len, dtype=torch.long)
    half = seq_len // 2
    sequence_ids[:, half:] = 2
    # Последние 2 токена — PAD (sequence_id = 0)
    sequence_ids[:, -2:] = 0
    return sequence_ids


def test_gqa_equals_mha_when_kv_heads_equals_heads():
    torch.manual_seed(0)
    d_model, n_heads = 32, 4
    batch_size, seq_len = 2, 10

    mha = BlockMaskedAttention(d_model, n_heads)
    gqa = GQAAttention(d_model, n_heads, n_kv_heads=n_heads)

    # GQAAttention считает Q/K/V одним слитным матмулом (W_qkv), поэтому
    # веса MHA копируем в соответствующие срезы слитной матрицы:
    # [0:q_dim] -> Q, [q_dim:q_dim+kv_dim] -> K, [q_dim+kv_dim:] -> V.
    q_dim, kv_dim = gqa.q_dim, gqa.kv_dim
    with torch.no_grad():
        gqa.W_qkv.weight[:q_dim].copy_(mha.W_q.weight)
        gqa.W_qkv.bias[:q_dim].copy_(mha.W_q.bias)
        gqa.W_qkv.weight[q_dim:q_dim + kv_dim].copy_(mha.W_k.weight)
        gqa.W_qkv.bias[q_dim:q_dim + kv_dim].copy_(mha.W_k.bias)
        gqa.W_qkv.weight[q_dim + kv_dim:].copy_(mha.W_v.weight)
        gqa.W_qkv.bias[q_dim + kv_dim:].copy_(mha.W_v.bias)
        gqa.W_o.weight.copy_(mha.W_o.weight)
        gqa.W_o.bias.copy_(mha.W_o.bias)

    x = torch.randn(batch_size, seq_len, d_model)
    sequence_ids = _make_sequence_ids(batch_size, seq_len)

    out_mha = mha(x, sequence_ids)
    out_gqa = gqa(x, sequence_ids)

    torch.testing.assert_close(out_gqa, out_mha)


def test_gqa_output_shape():
    torch.manual_seed(0)
    d_model, n_heads, n_kv_heads = 32, 8, 2
    batch_size, seq_len = 3, 12

    gqa = GQAAttention(d_model, n_heads, n_kv_heads)
    x = torch.randn(batch_size, seq_len, d_model)
    sequence_ids = _make_sequence_ids(batch_size, seq_len)

    out = gqa(x, sequence_ids)

    assert out.shape == (batch_size, seq_len, d_model)


def test_gqa_invalid_group_count_raises():
    with pytest.raises(AssertionError):
        GQAAttention(d_model=32, n_heads=8, n_kv_heads=3)


def test_gqa_respects_packed_mask():
    """
    Токены разных sequence_ids внутри одного пака не должны влиять друг
    на друга: изменение токенов первой последовательности не меняет
    выход второй последовательности (при фиксированном seed на входе).
    """
    torch.manual_seed(0)
    d_model, n_heads, n_kv_heads = 32, 8, 2
    batch_size, seq_len = 1, 10

    gqa = GQAAttention(d_model, n_heads, n_kv_heads)
    gqa.eval()

    sequence_ids = torch.ones(batch_size, seq_len, dtype=torch.long)
    sequence_ids[:, seq_len // 2:] = 2  # вторая последовательность, без PAD

    x = torch.randn(batch_size, seq_len, d_model)
    x_modified = x.clone()
    x_modified[:, : seq_len // 2, :] = torch.randn(batch_size, seq_len // 2, d_model)

    with torch.no_grad():
        out = gqa(x, sequence_ids)
        out_modified = gqa(x_modified, sequence_ids)

    second_half_out = out[:, seq_len // 2:, :]
    second_half_out_modified = out_modified[:, seq_len // 2:, :]

    torch.testing.assert_close(second_half_out, second_half_out_modified)


def test_gqa_use_cache_raises_not_implemented():
    gqa = GQAAttention(d_model=32, n_heads=8, n_kv_heads=2)
    x = torch.randn(1, 5, 32)
    sequence_ids = torch.ones(1, 5, dtype=torch.long)

    with pytest.raises(NotImplementedError):
        gqa(x, sequence_ids, use_cache=True)


def test_gqa_uses_fused_qkv_projection():
    """
    Q/K/V должны считаться одним nn.Linear (один матмул) вместо трёх
    отдельных W_q/W_k/W_v: это и есть смысл слияния проекций.
    """
    d_model, n_heads, n_kv_heads = 32, 8, 2
    gqa = GQAAttention(d_model, n_heads, n_kv_heads)

    assert hasattr(gqa, "W_qkv")
    assert not hasattr(gqa, "W_q")
    assert not hasattr(gqa, "W_k")
    assert not hasattr(gqa, "W_v")

    expected_out_features = gqa.q_dim + 2 * gqa.kv_dim
    assert gqa.W_qkv.out_features == expected_out_features
    assert gqa.q_dim == n_heads * gqa.d_k
    assert gqa.kv_dim == n_kv_heads * gqa.d_k
