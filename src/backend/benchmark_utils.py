import math
import torch

from src.backend.flash_attention import FlashAttentionFunction, reference_attention


def bench(fn, warmup=10, iters=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    torch.cuda.reset_peak_memory_stats()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end) / iters
    peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    return ms, peak_mem_mb


def run_case(batch, n_heads, seq_len, head_dim, causal=True, device="cuda", dtype=torch.float16):
    sm_scale = 1.0 / math.sqrt(head_dim)

    q = torch.randn(batch, n_heads, seq_len, head_dim, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(batch, n_heads, seq_len, head_dim, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch, n_heads, seq_len, head_dim, device=device, dtype=dtype, requires_grad=True)

    def flash_fwd():
        with torch.no_grad():
            FlashAttentionFunction.apply(q, k, v, causal, sm_scale)

    def ref_fwd():
        with torch.no_grad():
            reference_attention(q, k, v, causal=causal, sm_scale=sm_scale)

    def flash_fwd_bwd():
        q.grad = k.grad = v.grad = None
        out = FlashAttentionFunction.apply(q, k, v, causal, sm_scale)
        out.backward(torch.ones_like(out))

    def ref_fwd_bwd():
        q.grad = k.grad = v.grad = None
        out = reference_attention(q, k, v, causal=causal, sm_scale=sm_scale)
        out.backward(torch.ones_like(out))

    result = {"seq_len": seq_len, "causal": causal}

    for name, fn in [
        ("flash_fwd", flash_fwd), ("ref_fwd", ref_fwd),
        ("flash_bwd", flash_fwd_bwd), ("ref_bwd", ref_fwd_bwd),
    ]:
        try:
            ms, mem = bench(fn)
        except torch.cuda.OutOfMemoryError:
            ms, mem = None, None
            torch.cuda.empty_cache()
        result[f"{name}_ms"] = ms
        result[f"{name}_mem_mb"] = mem

    return result


def run_benchmark_suite(seq_lens, batch=4, n_heads=8, head_dim=64, causal=True):
    return [run_case(batch, n_heads, sl, head_dim, causal=causal) for sl in seq_lens]