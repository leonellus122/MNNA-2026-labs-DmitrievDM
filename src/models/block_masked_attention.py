import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class BlockMaskedAttention(nn.Module):
    """
    Многоголовое маскированное внимание с поддержкой packed batching.
    
    Использует block-masked attention: токены из одной последовательности
    не могут обращаться к токенам из другой последовательности.
    
    Маска: M[i,j] = (s_i == s_j) AND (j <= i) AND (s_i != 0)
    """
    
    def __init__(self, d_model: int, n_heads: int):
        """
        Args:
            d_model: размерность модели (размерность вектора-эмбеддинга)
            n_heads: количество голов внимания
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model должен быть кратен n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # размерность каждой головы
        
        # Линейные проекции для Q, K, V и выхода
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, x: torch.Tensor, sequence_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: тензор формы (batch_size, seq_len, d_model)
            sequence_ids: тензор формы (batch_size, seq_len) с ID последовательностей
        
        Returns:
            результат внимания формы (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # Проецируем Q, K, V
        Q = self.W_q(x)  # (batch_size, seq_len, d_model)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # Разбиваем на головы
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Вычисляем attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Создаем block mask
        mask = self._create_block_mask(sequence_ids)  # (batch_size, seq_len, seq_len)
        
        # Расширяем маску для всех голов
        mask = mask.unsqueeze(1)
        
        # Применяем маску: заполняем запрещенные позиции очень маленьким значением
        # scores = scores.masked_fill(mask == 0, float('-inf'))
        scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)

        # Softmax по последней оси
        attn_weights = F.softmax(scores, dim=-1)
        
        # Применяем внимания к V
        attn_output = torch.matmul(attn_weights, V)
        
        # Объединяем головы обратно
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Финальная проекция
        output = self.W_o(attn_output)
        
        return output
    
    def _create_block_mask(self, sequence_ids: torch.Tensor) -> torch.Tensor:
        """
        Создает block mask для packed batching.
        
        Args:
            sequence_ids: (batch_size, seq_len) с ID последовательностей
        
        Returns:
            mask: (batch_size, seq_len, seq_len) булев тензор
                  True если позиция (i, j) разрешена
        """
        batch_size, seq_len = sequence_ids.shape
        
        # Создаем causal mask (j <= i)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=sequence_ids.device, dtype=torch.bool)) # Добавлял dtype=torch.bool для исправления
        # Создаем маску одинаковых последовательностей (s_i == s_j)
        seq_mask = (sequence_ids.unsqueeze(2) == sequence_ids.unsqueeze(1))
        
        # Создаем маску не-PAD токенов (s_i != 0)
        not_pad_mask = (sequence_ids.unsqueeze(2) != 0).expand(-1, -1, seq_len)
        
        # Объединяем все условия: (s_i == s_j) AND (j <= i) AND (s_i != 0)
        # causal_mask имеет форму (seq_len, seq_len), нужно расширить до (batch_size, seq_len, seq_len)
        causal_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Финальная маска
        mask = seq_mask & causal_mask & not_pad_mask
        
        return mask