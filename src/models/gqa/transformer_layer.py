import torch
import torch.nn as nn
from .attention import GQAAttention
from ..ffn import FFN


class GQATransformerLayer(nn.Module):
    """
    Один слой трансформера с post-norm нормализацией на базе GQAAttention.

    Формула:
    z1 = LayerNorm(x + Attention(x))
    z2 = LayerNorm(z1 + FFN(z1))
    """

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, d_ff: int):
        """
        Args:
            d_model: размерность модели
            n_heads: количество query-голов внимания
            n_kv_heads: количество K/V-групп для GQA
            d_ff: размерность скрытого слоя FFN
        """
        super().__init__()
        self.attention = GQAAttention(d_model, n_heads, n_kv_heads)
        self.ffn = FFN(d_model, d_ff)

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        sequence_ids: torch.Tensor,
        past_key_value=None,
        use_cache: bool = False,
    ):
        """
        Args:
            x: (batch_size, seq_len, d_model); при use_cache=True — только новые токены
            sequence_ids: (batch_size, seq_len); игнорируется при use_cache=True
            past_key_value: см. GQAAttention.forward (п. 1.2 ЛР4)
            use_cache: False — обучение/packed batching (как в п. 1.1);
                       True — инкрементальный шаг инференса с KV-кэшем (п. 1.2)

        Returns:
            use_cache=False: (batch_size, seq_len, d_model)
            use_cache=True: (z2, present_key_value)
        """
        if use_cache:
            attn_out, present = self.attention(x, sequence_ids, past_key_value=past_key_value, use_cache=True)
            z1 = self.ln1(x + attn_out)
            ffn_out = self.ffn(z1)
            z2 = self.ln2(z1 + ffn_out)
            return z2, present

        # Post-norm: z1 = LayerNorm(x + Attention(x))
        attn_output = self.attention(x, sequence_ids)
        z1 = self.ln1(x + attn_output)

        # Post-norm: z2 = LayerNorm(z1 + FFN(z1))
        ffn_output = self.ffn(z1)
        z2 = self.ln2(z1 + ffn_output)

        return z2
