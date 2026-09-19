import torch
import torch.nn.functional as F

from src.training.gqa.lightning_module import GQALightningModule


class GQACachedGenerator:
    """
    Генерация текста GQAGPTModel через KV-кэш (use_cache=True, п.1.2).
    Отдельный класс поверх GQALightningModule — не редактирует
    src/training/gqa/lightning_module.py (см. докстринг generate() там).
    """

    def __init__(self, lightning_module: GQALightningModule):
        self.lightning_module = lightning_module
        self.model = lightning_module.model  # GQAGPTModel
        self.model.eval()

    @classmethod
    def from_checkpoint(cls, checkpoint_path, map_location=None) -> "GQACachedGenerator":
        if map_location is None:
            map_location = "cuda" if torch.cuda.is_available() else "cpu"
        lightning_module = GQALightningModule.load_from_checkpoint(
            checkpoint_path, map_location=map_location, weights_only=False,
        )
        return cls(lightning_module)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        tokenizer,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> str:
        device = next(self.model.parameters()).device
        input_ids = torch.tensor([tokenizer.encode(prompt).ids], dtype=torch.long, device=device)

        # Prefill: весь промпт за один шаг, sequence_ids не нужен при use_cache=True
        logits, past_key_values = self.model(input_ids, None, use_cache=True)
        next_token = self._sample(logits[:, -1, :], temperature, top_k, top_p)
        generated = torch.cat([input_ids, next_token], dim=1)

        for _ in range(max_length - 1):
            if next_token.item() == tokenizer.token_to_id("<eos>"):
                break
            logits, past_key_values = self.model(
                next_token, None, past_key_values=past_key_values, use_cache=True
            )
            next_token = self._sample(logits[:, -1, :], temperature, top_k, top_p)
            generated = torch.cat([generated, next_token], dim=1)

        return tokenizer.decode(generated[0].tolist())

    def _sample(self, logits, temperature, top_k, top_p):
        # та же формула top-k/top-p, что и в GQALightningModule.generate() —
        # отдельная копия внутри этого модуля, а не импорт приватной части того файла
        logits = logits / temperature
        if top_k > 0:
            kth = torch.topk(logits, top_k)[0][..., -1, None]
            logits[logits < kth] = float("-inf")
        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cum_probs > top_p
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = 0
            logits[:, sorted_idx[remove]] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)
