import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf

from src.training.gqa.lightning_module import GQALightningModule
from src.inference.gqa_cached_generator import GQACachedGenerator


class _FakeEncoding:
    def __init__(self, ids):
        self.ids = ids


class _FakeTokenizer:
    """
    Минимальный токенизатор для тестов генерации — без реального BPE и без
    обученного чекпоинта (как и остальные GQA-тесты, работает на случайно
    инициализированной модели малой размерности).
    """

    def __init__(self, vocab_size: int, has_eos: bool = True):
        self.vocab_size = vocab_size
        self.has_eos = has_eos
        self.eos_id = vocab_size - 1

    def encode(self, text: str) -> _FakeEncoding:
        ids = [1 + (ord(c) % (self.vocab_size - 2)) for c in text] or [1]
        return _FakeEncoding(ids)

    def decode(self, ids) -> str:
        return " ".join(str(i) for i in ids)

    def token_to_id(self, token: str):
        if token == "<eos>" and self.has_eos:
            return self.eos_id
        return None


def _make_config(vocab_size=50, d_model=16, n_heads=4, n_kv_heads=2, n_layers=2, d_ff=32, max_len=64):
    return OmegaConf.create({
        "model": {
            "vocab_size": vocab_size, "d_model": d_model, "n_heads": n_heads,
            "n_kv_heads": n_kv_heads, "n_layers": n_layers, "d_ff": d_ff, "max_len": max_len,
        },
        "training": {
            "learning_rate": 3.0e-4, "weight_decay": 0.1, "warmup_steps": 10, "max_epochs": 1,
            "optimizer": {"betas": [0.9, 0.999], "eps": 1.0e-8},
            "scheduler": {"T_max": 100, "eta_min": 1.0e-6},
        },
        "paths": {"tokenizer_path": "dummy"},
    })


def _make_lightning_module(**kwargs) -> GQALightningModule:
    torch.manual_seed(0)
    return GQALightningModule(_make_config(**kwargs))


def test_generate_matches_naive_greedy():
    """
    Ключевая проверка: KV-кэш — только оптимизация, а не другой алгоритм.
    При жадном декодинге (top_k=1) GQACachedGenerator.generate и
    GQALightningModule.generate (без кэша) на одной и той же модели/промпте
    должны выдать идентичную последовательность токенов.
    """
    lm = _make_lightning_module()
    lm.eval()
    generator = GQACachedGenerator(lm)
    tokenizer = _FakeTokenizer(vocab_size=lm.model.vocab_size)

    cached_text = generator.generate("hello", tokenizer, max_length=10, temperature=1.0, top_k=1, top_p=1.0)
    naive_text = lm.generate("hello", tokenizer, max_length=10, temperature=1.0, top_k=1, top_p=1.0)

    assert cached_text == naive_text


def test_generate_respects_max_length():
    """Длина сгенерированных токенов не превышает len(prompt) + max_length."""
    lm = _make_lightning_module()
    lm.eval()
    generator = GQACachedGenerator(lm)
    # has_eos=False — модель не должна иметь шанс остановиться раньше срока,
    # тест проверяет именно верхнюю границу длины, а не остановку по EOS.
    tokenizer = _FakeTokenizer(vocab_size=lm.model.vocab_size, has_eos=False)

    prompt = "hello"
    max_length = 15
    text = generator.generate(prompt, tokenizer, max_length=max_length, top_k=0, top_p=1.0)

    prompt_len = len(tokenizer.encode(prompt).ids)
    generated_len = len(text.split())

    assert generated_len <= prompt_len + max_length


def test_generate_stops_on_eos():
    """Если модель семплирует <eos>, генерация останавливается до max_length."""
    lm = _make_lightning_module()
    lm.eval()
    generator = GQACachedGenerator(lm)
    tokenizer = _FakeTokenizer(vocab_size=lm.model.vocab_size)
    eos_id = tokenizer.token_to_id("<eos>")

    # Форсируем выбор <eos> на первом же сэмплированном токене — проверяем
    # именно логику остановки generate(), а не вероятностное поведение
    # необученной модели (ей реальный <eos> может не встретиться вовсе).
    generator._sample = lambda logits, temperature, top_k, top_p: torch.tensor([[eos_id]])

    prompt = "hi"
    max_length = 20
    text = generator.generate(prompt, tokenizer, max_length=max_length)

    prompt_len = len(tokenizer.encode(prompt).ids)
    generated_len = len(text.split())

    assert generated_len == prompt_len + 1
    assert generated_len < prompt_len + max_length


def test_from_checkpoint_roundtrip(tmp_path):
    """
    GQACachedGenerator.from_checkpoint(path) восстанавливает модель без
    ошибок, с архитектурой (d_model, n_heads, n_kv_heads, n_layers,
    vocab_size), совпадающей с тем, что было сохранено.
    """
    config = _make_config(vocab_size=64, d_model=16, n_heads=4, n_kv_heads=2, n_layers=2, d_ff=32)
    lm = GQALightningModule(config)

    ckpt_path = tmp_path / "test.ckpt"
    trainer = pl.Trainer(logger=False, enable_checkpointing=False, accelerator="cpu")
    trainer.strategy.connect(lm)
    trainer.save_checkpoint(str(ckpt_path))

    generator = GQACachedGenerator.from_checkpoint(str(ckpt_path), map_location="cpu")

    assert generator.model.d_model == 16
    assert generator.model.vocab_size == 64
    assert len(generator.model.transformer_layers) == 2
    attn = generator.model.transformer_layers[0].attention
    assert attn.n_heads == 4
    assert attn.n_kv_heads == 2
