import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .sinusoidal_positional_encoding import SinusoidalPositionalEncoding
from .lm_head import LMHead
from .transformer_layer import TransformerLayer



class GPTModel(nn.Module):
    """
    Полная GPT-like модель для языкового моделирования.
    
    Включает:
    - Token embeddings
    - Sinusoidal positional encoding (с учетом packed batching)
    - N трансформерных слоев
    - LM-head (без Softmax, так как CrossEntropyLoss делает это сам)
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        max_len: int = 5000
    ):
        """
        Args:
            vocab_size: размер словаря
            d_model: размерность модели
            n_heads: количество голов внимания
            n_layers: количество трансформерных слоев
            d_ff: размерность скрытого слоя FFN
            max_len: максимальная длина последовательности
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.positional_encoding = SinusoidalPositionalEncoding(d_model, max_len)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])
        
        # LM-head
        self.lm_head = LMHead(d_model, vocab_size)
    
    def forward(self, input_ids: torch.Tensor, sequence_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (batch_size, seq_len) - токены
            sequence_ids: (batch_size, seq_len) - ID последовательностей для packed batching
        
        Returns:
            logits: (batch_size, seq_len, vocab_size) - логиты для каждого токена
        """
        # Token embeddings
        x = self.token_embedding(input_ids)  # (batch_size, seq_len, d_model)
        
        # Positional encoding (с учетом sequence_ids)
        x = self.positional_encoding(x, sequence_ids)
        
        # Прогоняем через все трансформерные слои
        for layer in self.transformer_layers:
            x = layer(x, sequence_ids)
        
        # LM-head (без Softmax!)
        logits = self.lm_head(x)  # (batch_size, seq_len, vocab_size)
        
        return logits
    
    def compute_loss(
        self,
        input_ids: torch.Tensor,
        sequence_ids: torch.Tensor,
        pad_token_id: int = 0
    ) -> torch.Tensor:
        """
        Вычисляет loss с учетом маски для packed batching.
        
        Args:
            input_ids: (batch_size, seq_len)
            sequence_ids: (batch_size, seq_len)
            pad_token_id: ID токена PAD
        
        Returns:
            loss: скаляр
        """
        # Сдвигаем input_ids для предсказания следующего токена
        # logits на позиции i предсказывает токен на позиции i+1
        logits = self.forward(input_ids[:, :-1], sequence_ids[:, :-1])
        targets = input_ids[:, 1:]
        
        # Создаем маску для loss
        # M_loss[i] = (s_i == s_{i+1}) AND (s_i != 0)
        # Это означает, что мы не считаем loss на стыках между последовательностями
        seq_ids_shifted = sequence_ids[:, :-1]
        seq_ids_next = sequence_ids[:, 1:]
        
        loss_mask = (seq_ids_shifted == seq_ids_next) & (seq_ids_shifted != 0)
        
        # Применяем маску
        # logits: (batch_size, seq_len-1, vocab_size)
        # targets: (batch_size, seq_len-1)
        # loss_mask: (batch_size, seq_len-1)
        
        # Вычисляем cross-entropy только для замаскированных позиций
        # Flatten для удобства
        logits_flat = logits.reshape(-1, self.vocab_size)
        targets_flat = targets.reshape(-1)
        mask_flat = loss_mask.reshape(-1) # заменял в этих трех строчках view на reshape
        
        # Применяем маску
        logits_masked = logits_flat[mask_flat]
        targets_masked = targets_flat[mask_flat]
        
        # Вычисляем loss (CrossEntropyLoss уже делает Softmax внутри!)
        loss = F.cross_entropy(logits_masked, targets_masked)
        
        return loss