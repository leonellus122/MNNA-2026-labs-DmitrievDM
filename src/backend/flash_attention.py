import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


# ============================================================
# Эталонная (референсная) реализация на чистом PyTorch.
# Используется для тестов и бенчмарков.
# ============================================================
def reference_attention(q, k, v, causal=True, sm_scale=None):
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])

    scores = torch.matmul(q, k.transpose(-2, -1)) * sm_scale

    if causal:
        seq_len = q.shape[-2]
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool)
        )
        scores = scores.masked_fill(~causal_mask, float("-inf"))

    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, v)


# ============================================================
# Forward Triton-кернел
# ============================================================
@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale, L, O,
    stride_qb, stride_qm, stride_qd,
    stride_kb, stride_kn, stride_kd,
    stride_vb, stride_vn, stride_vd,
    stride_ob, stride_om, stride_od,
    N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    start_m = tl.program_id(0)
    bh = tl.program_id(1)

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    q_ptrs = Q + bh * stride_qb + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    hi = (start_m + 1) * BLOCK_M if IS_CAUSAL else N_CTX

    for start_n in range(0, hi, BLOCK_N):
        offs_n_cur = start_n + offs_n

        k_ptrs = K + bh * stride_kb + offs_n_cur[:, None] * stride_kn + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=offs_n_cur[:, None] < N_CTX, other=0.0)

        qk = tl.dot(q, tl.trans(k)) * sm_scale

        valid = offs_n_cur[None, :] < N_CTX
        if IS_CAUSAL:
            valid = valid & (offs_m[:, None] >= offs_n_cur[None, :])
        qk = tl.where(valid, qk, float("-inf"))

        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        p = tl.exp(qk - m_new[:, None])
        alpha = tl.exp(m_i - m_new)

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        v_ptrs = V + bh * stride_vb + offs_n_cur[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=offs_n_cur[:, None] < N_CTX, other=0.0)

        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_new

    acc = acc / l_i[:, None]

    tl.store(L + bh * N_CTX + offs_m, m_i + tl.log(l_i), mask=offs_m < N_CTX)

    o_ptrs = O + bh * stride_ob + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc, mask=offs_m[:, None] < N_CTX)


def _flash_attn_forward_impl(q, k, v, causal, sm_scale):
    batch, n_heads, seq_len, head_dim = q.shape
    bh = batch * n_heads

    q_ = q.reshape(bh, seq_len, head_dim).contiguous()
    k_ = k.reshape(bh, seq_len, head_dim).contiguous()
    v_ = v.reshape(bh, seq_len, head_dim).contiguous()

    o = torch.empty_like(q_)
    L = torch.empty((bh, seq_len), device=q.device, dtype=torch.float32)

    BLOCK_M, BLOCK_N = 64, 64
    grid = (triton.cdiv(seq_len, BLOCK_M), bh)

    _fwd_kernel[grid](
        q_, k_, v_, sm_scale, L, o,
        q_.stride(0), q_.stride(1), q_.stride(2),
        k_.stride(0), k_.stride(1), k_.stride(2),
        v_.stride(0), v_.stride(1), v_.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        seq_len,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=head_dim,
        IS_CAUSAL=causal,
    )

    return o.reshape(batch, n_heads, seq_len, head_dim), L, (q_, k_, v_)


def flash_attn_forward(q, k, v, causal=True, sm_scale=None):
    """Публичный враппер для задания 3.1 — только forward, без градиентов."""
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])
    o, _, _ = _flash_attn_forward_impl(q, k, v, causal, sm_scale)
    return o


# ============================================================
# Backward: предподсчёт Delta_i = rowsum(dO_i * O_i)
# ============================================================
@triton.jit
def _bwd_preprocess(
    O, DO, Delta,
    stride_ob, stride_om, stride_od,
    N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
):
    start_m = tl.program_id(0)
    bh = tl.program_id(1)

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    o_ptrs = O + bh * stride_ob + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    do_ptrs = DO + bh * stride_ob + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od

    o = tl.load(o_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)
    do = tl.load(do_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

    delta = tl.sum(o * do, axis=1)
    tl.store(Delta + bh * N_CTX + offs_m, delta, mask=offs_m < N_CTX)


# ============================================================
# Backward: dK, dV (один program instance = один блок K/V)
# ============================================================
@triton.jit
def _bwd_kv_kernel(
    Q, K, V, sm_scale,
    DO, L, Delta,
    DK, DV,
    stride_qb, stride_qm, stride_qd,
    stride_kb, stride_kn, stride_kd,
    stride_vb, stride_vn, stride_vd,
    N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    start_n = tl.program_id(0)
    bh = tl.program_id(1)

    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    k_ptrs = K + bh * stride_kb + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
    v_ptrs = V + bh * stride_vb + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
    k = tl.load(k_ptrs, mask=offs_n[:, None] < N_CTX, other=0.0)
    v = tl.load(v_ptrs, mask=offs_n[:, None] < N_CTX, other=0.0)

    dk = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)

    lo = start_n * BLOCK_N if IS_CAUSAL else 0
    offs_m_range = tl.arange(0, BLOCK_M)

    for start_m in range(lo, N_CTX, BLOCK_M):
        offs_m = start_m + offs_m_range

        q_ptrs = Q + bh * stride_qb + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
        do_ptrs = DO + bh * stride_qb + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
        q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)
        do = tl.load(do_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

        qk = tl.dot(q, tl.trans(k)) * sm_scale
        valid = (offs_m[:, None] < N_CTX) & (offs_n[None, :] < N_CTX)
        if IS_CAUSAL:
            valid = valid & (offs_m[:, None] >= offs_n[None, :])
        qk = tl.where(valid, qk, float("-inf"))

        l_i = tl.load(L + bh * N_CTX + offs_m, mask=offs_m < N_CTX, other=0.0)
        p = tl.exp(qk - l_i[:, None])
        p = tl.where(valid, p, 0.0)

        dv += tl.dot(tl.trans(p.to(do.dtype)), do)

        delta_i = tl.load(Delta + bh * N_CTX + offs_m, mask=offs_m < N_CTX, other=0.0)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta_i[:, None]) * sm_scale
        ds = tl.where(valid, ds, 0.0)

        dk += tl.dot(tl.trans(ds.to(q.dtype)), q)

    dk_ptrs = DK + bh * stride_kb + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
    dv_ptrs = DV + bh * stride_vb + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
    tl.store(dk_ptrs, dk, mask=offs_n[:, None] < N_CTX)
    tl.store(dv_ptrs, dv, mask=offs_n[:, None] < N_CTX)


# ============================================================
# Backward: dQ (один program instance = один блок Q)
# ============================================================
@triton.jit
def _bwd_q_kernel(
    Q, K, V, sm_scale,
    DO, L, Delta,
    DQ,
    stride_qb, stride_qm, stride_qd,
    stride_kb, stride_kn, stride_kd,
    stride_vb, stride_vn, stride_vd,
    N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    start_m = tl.program_id(0)
    bh = tl.program_id(1)

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    q_ptrs = Q + bh * stride_qb + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    do_ptrs = DO + bh * stride_qb + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)
    do = tl.load(do_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

    l_i = tl.load(L + bh * N_CTX + offs_m, mask=offs_m < N_CTX, other=0.0)
    delta_i = tl.load(Delta + bh * N_CTX + offs_m, mask=offs_m < N_CTX, other=0.0)

    dq = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    offs_n_range = tl.arange(0, BLOCK_N)

    hi = (start_m + 1) * BLOCK_M if IS_CAUSAL else N_CTX

    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + offs_n_range

        k_ptrs = K + bh * stride_kb + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        v_ptrs = V + bh * stride_vb + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        k = tl.load(k_ptrs, mask=offs_n[:, None] < N_CTX, other=0.0)
        v = tl.load(v_ptrs, mask=offs_n[:, None] < N_CTX, other=0.0)

        qk = tl.dot(q, tl.trans(k)) * sm_scale
        valid = (offs_m[:, None] < N_CTX) & (offs_n[None, :] < N_CTX)
        if IS_CAUSAL:
            valid = valid & (offs_m[:, None] >= offs_n[None, :])
        qk = tl.where(valid, qk, float("-inf"))

        p = tl.exp(qk - l_i[:, None])
        p = tl.where(valid, p, 0.0)

        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta_i[:, None]) * sm_scale
        ds = tl.where(valid, ds, 0.0)

        dq += tl.dot(ds.to(k.dtype), k)

    dq_ptrs = DQ + bh * stride_qb + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    tl.store(dq_ptrs, dq, mask=offs_m[:, None] < N_CTX)


# ============================================================
# torch.autograd.Function
# ============================================================
class FlashAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale):
        batch, n_heads, seq_len, head_dim = q.shape

        o, L, (q_, k_, v_) = _flash_attn_forward_impl(q, k, v, causal, sm_scale)
        o_ = o.reshape(batch * n_heads, seq_len, head_dim).contiguous()

        ctx.save_for_backward(q_, k_, v_, o_, L)
        ctx.sm_scale = sm_scale
        ctx.causal = causal
        ctx.shape = (batch, n_heads, seq_len, head_dim)

        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, L = ctx.saved_tensors
        sm_scale = ctx.sm_scale
        causal = ctx.causal
        batch, n_heads, seq_len, head_dim = ctx.shape
        bh = batch * n_heads

        do_ = do.reshape(bh, seq_len, head_dim).contiguous()

        delta = torch.empty((bh, seq_len), device=q.device, dtype=torch.float32)
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)

        BLOCK_M, BLOCK_N = 64, 64

        grid_pre = (triton.cdiv(seq_len, BLOCK_M), bh)
        _bwd_preprocess[grid_pre](
            o, do_, delta,
            o.stride(0), o.stride(1), o.stride(2),
            seq_len,
            BLOCK_M=BLOCK_M, BLOCK_DMODEL=head_dim,
        )

        grid_kv = (triton.cdiv(seq_len, BLOCK_N), bh)
        _bwd_kv_kernel[grid_kv](
            q, k, v, sm_scale,
            do_, L, delta,
            dk, dv,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            seq_len,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=head_dim,
            IS_CAUSAL=causal,
        )

        grid_q = (triton.cdiv(seq_len, BLOCK_M), bh)
        _bwd_q_kernel[grid_q](
            q, k, v, sm_scale,
            do_, L, delta,
            dq,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            seq_len,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=head_dim,
            IS_CAUSAL=causal,
        )

        dq = dq.reshape(batch, n_heads, seq_len, head_dim)
        dk = dk.reshape(batch, n_heads, seq_len, head_dim)
        dv = dv.reshape(batch, n_heads, seq_len, head_dim)

        return dq, dk, dv, None, None


# ============================================================
# nn.Module-обёртка
# ============================================================
class FlashAttention(nn.Module):
    def __init__(self, causal: bool = True, sm_scale: float = None):
        super().__init__()
        self.causal = causal
        self.sm_scale = sm_scale

    def forward(self, q, k, v):
        sm_scale = self.sm_scale if self.sm_scale is not None else 1.0 / math.sqrt(q.shape[-1])
        return FlashAttentionFunction.apply(q, k, v, self.causal, sm_scale)