import math
import torch
import torch.nn as nn
import torch.nn.functional as F
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
# Маска внимания: считается один раз на host-стороне и передаётся
# в кернелы как аргумент (а не пересчитывается внутри них).
#
# elem_mask  - поэлементная маска [N_CTX, N_CTX] (1 = разрешено, 0 = замаскировано)
# block_mask - таблица [num_m_blocks, num_n_blocks]: есть ли в блоке
#              хоть один разрешённый элемент. Используется, чтобы
#              полностью пропускать замаскированные блоки в кернелах.
# ============================================================
def _build_attn_masks(seq_len, causal, block_m, block_n, device):
    idx_m = torch.arange(seq_len, device=device)
    idx_n = torch.arange(seq_len, device=device)
    if causal:
        elem_mask = idx_m[:, None] >= idx_n[None, :]
    else:
        elem_mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)

    num_m = triton.cdiv(seq_len, block_m)
    num_n = triton.cdiv(seq_len, block_n)
    padded = F.pad(elem_mask, (0, num_n * block_n - seq_len, 0, num_m * block_m - seq_len))
    block_mask = padded.view(num_m, block_m, num_n, block_n).any(dim=(1, 3))

    return elem_mask.to(torch.int8).contiguous(), block_mask.to(torch.int8).contiguous()


# ============================================================
# Forward Triton-кернел
# ============================================================
@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale, L, O,
    Mask, BlockMask,
    stride_qb, stride_qm, stride_qd,
    stride_kb, stride_kn, stride_kd,
    stride_vb, stride_vn, stride_vd,
    stride_ob, stride_om, stride_od,
    stride_maskm, stride_maskn,
    stride_bmm, stride_bmn,
    N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
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

    num_n_blocks = tl.cdiv(N_CTX, BLOCK_N)

    for nb in range(0, num_n_blocks):
        start_n = nb * BLOCK_N
        block_valid = tl.load(BlockMask + start_m * stride_bmm + nb * stride_bmn)

        if block_valid != 0:
            offs_n_cur = start_n + offs_n
            bounds = (offs_m[:, None] < N_CTX) & (offs_n_cur[None, :] < N_CTX)

            k_ptrs = K + bh * stride_kb + offs_n_cur[:, None] * stride_kn + offs_d[None, :] * stride_kd
            k = tl.load(k_ptrs, mask=offs_n_cur[:, None] < N_CTX, other=0.0)

            qk = tl.dot(q, tl.trans(k)) * sm_scale

            mask_ptrs = Mask + offs_m[:, None] * stride_maskm + offs_n_cur[None, :] * stride_maskn
            elem_mask = tl.load(mask_ptrs, mask=bounds, other=0)
            qk = tl.where(elem_mask != 0, qk, float("-inf"))

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
    elem_mask, block_mask = _build_attn_masks(seq_len, causal, BLOCK_M, BLOCK_N, q.device)
    grid = (triton.cdiv(seq_len, BLOCK_M), bh)

    _fwd_kernel[grid](
        q_, k_, v_, sm_scale, L, o,
        elem_mask, block_mask,
        q_.stride(0), q_.stride(1), q_.stride(2),
        k_.stride(0), k_.stride(1), k_.stride(2),
        v_.stride(0), v_.stride(1), v_.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        elem_mask.stride(0), elem_mask.stride(1),
        block_mask.stride(0), block_mask.stride(1),
        seq_len,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=head_dim,
    )

    return o.reshape(batch, n_heads, seq_len, head_dim), L, (q_, k_, v_), (elem_mask, block_mask)


def flash_attn_forward(q, k, v, causal=True, sm_scale=None):
    """Публичный враппер для задания 3.1 — только forward, без градиентов."""
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])
    o, _, _, _ = _flash_attn_forward_impl(q, k, v, causal, sm_scale)
    return o


# ============================================================
# Backward: единый fused-кернел.
#
# Один program instance = один блок Q (grid по строкам, как раньше
# у _bwd_q_kernel). Delta_i = rowsum(O_i * DO_i) считается на месте
# (бывший _bwd_preprocess) — O/DO для своего блока и так грузятся
# ровно один раз, отдельный кернел и HBM-буфер Delta не нужны.
#
# Внутри одного и того же внутреннего цикла по K/V-блокам qk/p/dp/ds
# считаются один раз и сразу используются и для dQ (эксклюзивно наш
# блок — копим в регистрах и делаем один tl.store), и для dK/dV
# (общие для всех program instances, которые видят этот K/V-блок —
# накапливаем через tl.atomic_add). Раньше _bwd_kv_kernel и
# _bwd_q_kernel пересчитывали qk/p/dp/ds дважды для одних и тех же
# пар блоков — теперь это одно вычисление.
# ============================================================
@triton.jit
def _bwd_kernel_fused(
    Q, K, V, sm_scale,
    O, DO, L,
    Mask, BlockMask,
    DQ, DK, DV,
    stride_qb, stride_qm, stride_qd,
    stride_kb, stride_kn, stride_kd,
    stride_vb, stride_vn, stride_vd,
    stride_maskm, stride_maskn,
    stride_bmm, stride_bmn,
    N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
):
    start_m = tl.program_id(0)
    bh = tl.program_id(1)

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_n_range = tl.arange(0, BLOCK_N)

    q_ptrs = Q + bh * stride_qb + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    do_ptrs = DO + bh * stride_qb + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    o_ptrs = O + bh * stride_qb + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd

    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)
    do = tl.load(do_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)
    o = tl.load(o_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

    delta_i = tl.sum(o * do, axis=1)
    l_i = tl.load(L + bh * N_CTX + offs_m, mask=offs_m < N_CTX, other=0.0)

    dq = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    num_n_blocks = tl.cdiv(N_CTX, BLOCK_N)

    for nb in range(0, num_n_blocks):
        start_n = nb * BLOCK_N
        block_valid = tl.load(BlockMask + start_m * stride_bmm + nb * stride_bmn)

        if block_valid != 0:
            offs_n = start_n + offs_n_range
            bounds = (offs_m[:, None] < N_CTX) & (offs_n[None, :] < N_CTX)

            k_ptrs = K + bh * stride_kb + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
            v_ptrs = V + bh * stride_vb + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
            k = tl.load(k_ptrs, mask=offs_n[:, None] < N_CTX, other=0.0)
            v = tl.load(v_ptrs, mask=offs_n[:, None] < N_CTX, other=0.0)

            qk = tl.dot(q, tl.trans(k)) * sm_scale

            mask_ptrs = Mask + offs_m[:, None] * stride_maskm + offs_n[None, :] * stride_maskn
            elem_mask = tl.load(mask_ptrs, mask=bounds, other=0)
            qk = tl.where(elem_mask != 0, qk, float("-inf"))

            p = tl.exp(qk - l_i[:, None])
            p = tl.where(elem_mask != 0, p, 0.0)

            dp = tl.dot(do, tl.trans(v))
            ds = p * (dp - delta_i[:, None]) * sm_scale
            ds = tl.where(elem_mask != 0, ds, 0.0)

            dq += tl.dot(ds.to(k.dtype), k)

            dk_contrib = tl.dot(tl.trans(ds.to(q.dtype)), q)
            dv_contrib = tl.dot(tl.trans(p.to(do.dtype)), do)

            dk_ptrs = DK + bh * stride_kb + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
            dv_ptrs = DV + bh * stride_vb + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
            tl.atomic_add(dk_ptrs, dk_contrib, mask=offs_n[:, None] < N_CTX)
            tl.atomic_add(dv_ptrs, dv_contrib, mask=offs_n[:, None] < N_CTX)

    dq_ptrs = DQ + bh * stride_qb + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    tl.store(dq_ptrs, dq, mask=offs_m[:, None] < N_CTX)


# ============================================================
# torch.autograd.Function
# ============================================================
class FlashAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale):
        batch, n_heads, seq_len, head_dim = q.shape

        o, L, (q_, k_, v_), (elem_mask, block_mask) = _flash_attn_forward_impl(q, k, v, causal, sm_scale)
        o_ = o.reshape(batch * n_heads, seq_len, head_dim).contiguous()

        ctx.save_for_backward(q_, k_, v_, o_, L)
        ctx.sm_scale = sm_scale
        ctx.causal = causal
        ctx.shape = (batch, n_heads, seq_len, head_dim)
        # Маска не требует градиента и не участвует в autograd-графе,
        # поэтому храним отдельно от save_for_backward и переиспользуем
        # в backward вместо пересчёта.
        ctx.elem_mask = elem_mask
        ctx.block_mask = block_mask

        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, L = ctx.saved_tensors
        sm_scale = ctx.sm_scale
        batch, n_heads, seq_len, head_dim = ctx.shape
        bh = batch * n_heads

        do_ = do.reshape(bh, seq_len, head_dim).contiguous()

        dq = torch.zeros_like(q)
        # dk/dv аккумулируются через tl.atomic_add -> считаем в float32,
        # чтобы не терять точность и не упираться в ограниченную поддержку
        # атомарного сложения для fp16 на части GPU; к исходному dtype
        # приводим уже после кернела.
        dk = torch.zeros((bh, seq_len, head_dim), device=q.device, dtype=torch.float32)
        dv = torch.zeros((bh, seq_len, head_dim), device=q.device, dtype=torch.float32)

        BLOCK_M, BLOCK_N = 64, 64
        elem_mask, block_mask = ctx.elem_mask, ctx.block_mask

        grid = (triton.cdiv(seq_len, BLOCK_M), bh)
        _bwd_kernel_fused[grid](
            q, k, v, sm_scale,
            o, do_, L,
            elem_mask, block_mask,
            dq, dk, dv,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            elem_mask.stride(0), elem_mask.stride(1),
            block_mask.stride(0), block_mask.stride(1),
            seq_len,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=head_dim,
        )

        dq = dq.reshape(batch, n_heads, seq_len, head_dim)
        dk = dk.reshape(batch, n_heads, seq_len, head_dim).to(k.dtype)
        dv = dv.reshape(batch, n_heads, seq_len, head_dim).to(v.dtype)

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