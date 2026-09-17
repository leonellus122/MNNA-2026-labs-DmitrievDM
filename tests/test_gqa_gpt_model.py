import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from omegaconf import OmegaConf
from dotenv import load_dotenv

from src.models.gpt_model import GPTModel
from src.models.gqa.gpt_model import GQAGPTModel


def _make_sequence_ids(batch_size: int, seq_len: int) -> torch.Tensor:
    """Пример packed batching: две последовательности + PAD в конце."""
    sequence_ids = torch.ones(batch_size, seq_len, dtype=torch.long)
    half = seq_len // 2
    sequence_ids[:, half:] = 2
    sequence_ids[:, -2:] = 0
    return sequence_ids


def _make_model(vocab_size=50, d_model=32, n_heads=8, n_kv_heads=2, n_layers=2, d_ff=64, max_len=64):
    torch.manual_seed(0)
    return GQAGPTModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_len=max_len,
    )


def test_gqa_gpt_model_forward_shape():
    vocab_size = 50
    batch_size, seq_len = 3, 16

    model = _make_model(vocab_size=vocab_size, max_len=seq_len)
    input_ids = torch.randint(1, vocab_size, (batch_size, seq_len))
    sequence_ids = _make_sequence_ids(batch_size, seq_len)

    logits = model(input_ids, sequence_ids)

    assert logits.shape == (batch_size, seq_len, vocab_size)


def test_gqa_gpt_model_compute_loss():
    vocab_size = 50
    batch_size, seq_len = 3, 16

    model = _make_model(vocab_size=vocab_size, max_len=seq_len)
    input_ids = torch.randint(1, vocab_size, (batch_size, seq_len))
    sequence_ids = _make_sequence_ids(batch_size, seq_len)

    loss = model.compute_loss(input_ids, sequence_ids)

    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_gqa_gpt_model_backward():
    vocab_size = 50
    batch_size, seq_len = 2, 16

    model = _make_model(vocab_size=vocab_size, max_len=seq_len)
    input_ids = torch.randint(1, vocab_size, (batch_size, seq_len))
    sequence_ids = _make_sequence_ids(batch_size, seq_len)

    loss = model.compute_loss(input_ids, sequence_ids)
    loss.backward()

    for name, p in model.named_parameters():
        assert p.grad is not None, f"Параметр {name} не получил градиент"


def test_gqa_gpt_model_fewer_params_than_mha():
    d_model, n_heads, n_layers, d_ff, vocab_size = 64, 8, 4, 128, 100

    gqa_model = GQAGPTModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=2,
        n_layers=n_layers,
        d_ff=d_ff,
    )
    mha_model = GPTModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
    )

    gqa_params = sum(p.numel() for p in gqa_model.parameters())
    mha_params = sum(p.numel() for p in mha_model.parameters())

    assert gqa_params < mha_params


def test_gqa_gpt_model_builds_from_config():
    load_dotenv()
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "gqa_model_config.yaml")
    config = OmegaConf.load(config_path)
    config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))

    assert config.model.n_heads % config.model.n_kv_heads == 0

    model = GQAGPTModel(
        vocab_size=config.model.vocab_size,
        d_model=config.model.d_model,
        n_heads=config.model.n_heads,
        n_kv_heads=config.model.n_kv_heads,
        n_layers=config.model.n_layers,
        d_ff=config.model.d_ff,
        max_len=config.model.max_len,
    )

    batch_size, seq_len = 2, 8
    input_ids = torch.randint(1, config.model.vocab_size, (batch_size, seq_len))
    sequence_ids = torch.ones(batch_size, seq_len, dtype=torch.long)

    logits = model(input_ids, sequence_ids)

    assert logits.shape == (batch_size, seq_len, config.model.vocab_size)


def test_gqa_gpt_model_cache_matches_full_forward():
    """
    Главный тест на корректность KV-кэша (п. 1.2 ЛР4): для одной
    последовательности без паддинга сравниваем полный forward (use_cache=False)
    с prefill + пошаговым decode (use_cache=True) — логиты должны совпасть.
    """
    vocab_size = 50
    batch_size, seq_len = 2, 12
    prefill_len = 5

    model = _make_model(vocab_size=vocab_size, max_len=seq_len)
    model.eval()

    input_ids = torch.randint(1, vocab_size, (batch_size, seq_len))
    sequence_ids = torch.ones(batch_size, seq_len, dtype=torch.long)  # без паддинга и стыков

    with torch.no_grad():
        full_logits = model(input_ids, sequence_ids, use_cache=False)

        # Prefill
        logits_prefill, past_key_values = model(
            input_ids[:, :prefill_len], sequence_ids[:, :prefill_len], use_cache=True
        )
        torch.testing.assert_close(logits_prefill, full_logits[:, :prefill_len, :], atol=1e-4, rtol=1e-4)

        # Decode по одному токену с накоплением кэша
        decode_logits = []
        for t in range(prefill_len, seq_len):
            step_logits, past_key_values = model(
                input_ids[:, t: t + 1],
                sequence_ids[:, t: t + 1],
                past_key_values=past_key_values,
                use_cache=True,
            )
            decode_logits.append(step_logits)

        decode_logits = torch.cat(decode_logits, dim=1)

    torch.testing.assert_close(decode_logits, full_logits[:, prefill_len:, :], atol=1e-4, rtol=1e-4)


def test_gqa_gpt_model_cache_prefill_then_decode_shapes():
    """После prefill длины P и k шагов decode кэш имеет длину P + k на каждом слое."""
    vocab_size = 50
    batch_size = 2
    prefill_len, n_decode_steps, n_layers = 4, 3, 2

    model = _make_model(vocab_size=vocab_size, n_layers=n_layers, max_len=prefill_len + n_decode_steps)
    model.eval()

    input_ids = torch.randint(1, vocab_size, (batch_size, prefill_len))
    sequence_ids = torch.ones(batch_size, prefill_len, dtype=torch.long)

    with torch.no_grad():
        _, past_key_values = model(input_ids, sequence_ids, use_cache=True)

        for _ in range(n_decode_steps):
            next_token = torch.randint(1, vocab_size, (batch_size, 1))
            next_seq_ids = torch.ones(batch_size, 1, dtype=torch.long)
            _, past_key_values = model(
                next_token, next_seq_ids, past_key_values=past_key_values, use_cache=True
            )

    expected_len = prefill_len + n_decode_steps
    assert len(past_key_values) == n_layers
    for K, V in past_key_values:
        assert K.shape[2] == expected_len
        assert V.shape[2] == expected_len
