import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import pytest

from src.backend.flash_attention import FlashAttentionFunction, flash_attn_forward, reference_attention


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("seq_len", [128, 300, 512])
def test_flash_attention_forward(causal, seq_len):
    torch.manual_seed(0)
    batch, n_heads, head_dim = 2, 4, 64

    q = torch.randn(batch, n_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(batch, n_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
    v = torch.randn(batch, n_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)

    out_triton = flash_attn_forward(q, k, v, causal=causal)
    out_ref = reference_attention(q, k, v, causal=causal)

    torch.testing.assert_close(out_triton, out_ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("causal", [True, False])
def test_flash_attention_backward(causal):
    torch.manual_seed(0)
    batch, n_heads, seq_len, head_dim = 2, 4, 256, 64
    sm_scale = 1.0 / (head_dim ** 0.5)

    q = torch.randn(batch, n_heads, seq_len, head_dim, device="cuda", dtype=torch.float16, requires_grad=True)
    k = torch.randn(batch, n_heads, seq_len, head_dim, device="cuda", dtype=torch.float16, requires_grad=True)
    v = torch.randn(batch, n_heads, seq_len, head_dim, device="cuda", dtype=torch.float16, requires_grad=True)

    q_ref = q.detach().clone().requires_grad_()
    k_ref = k.detach().clone().requires_grad_()
    v_ref = v.detach().clone().requires_grad_()

    out = FlashAttentionFunction.apply(q, k, v, causal, sm_scale)
    out_ref = reference_attention(q_ref, k_ref, v_ref, causal=causal, sm_scale=sm_scale)

    dout = torch.randn_like(out)
    out.backward(dout)
    out_ref.backward(dout)

    torch.testing.assert_close(q.grad, q_ref.grad, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(k.grad, k_ref.grad, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(v.grad, v_ref.grad, atol=2e-2, rtol=2e-2)