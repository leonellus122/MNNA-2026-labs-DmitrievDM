import torch
import torch.nn as nn
from .block_masked_attention import BlockMaskedAttention
from .ffn import FFN


class TransformerLayer(nn.Module):
    """
    Один слой трансформера с post-norm нормализацией.
    
    Формула:
    z1 = LayerNorm(x + Attention(x))
    z2 = LayerNorm(z1 + FFN(z1))
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        """
        Args:
            d_model: размерность модели
            n_heads: количество голов внимания
            d_ff: размерность скрытого слоя FFN
        """
        super().__init__()
        self.attention = BlockMaskedAttention(d_model, n_heads)
        self.ffn = FFN(d_model, d_ff)
        
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, sequence_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
            sequence_ids: (batch_size, seq_len)
        
        Returns:
            (batch_size, seq_len, d_model)
        """
        # Post-norm: z1 = LayerNorm(x + Attention(x))
        attn_output = self.attention(x, sequence_ids)
        z1 = self.ln1(x + attn_output)
        
        # Post-norm: z2 = LayerNorm(z1 + FFN(z1))
        ffn_output = self.ffn(z1)
        z2 = self.ln2(z1 + ffn_output)
        
        return z2