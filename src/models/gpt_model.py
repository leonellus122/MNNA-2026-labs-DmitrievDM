import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SinusoidalPositionalEncoding(nn.Module):
    """
    Синусоидальное позиционное кодирование с поддержкой packed batching.
    
    Для packed batching позиции сбрасываются в 0 для начала каждой новой 
    последовательности (определяется по sequence_ids/mask).
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Args:
            d_model: размерность эмбеддингов модели
            max_len: максимальная длина последовательности
        """
        super().__init__()
        self.d_model = d_model
        
        # Создаем матрицу позиционных кодирований
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Регистрируем как буфер (не обучается, но сохраняется в state_dict)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor, sequence_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: тензор формы (batch_size, seq_len, d_model)
            sequence_ids: тензор формы (batch_size, seq_len) с ID последовательностей
                         (0 для PAD, 1, 2, 3... для разных последовательностей)
        
        Returns:
            x + positional_encoding той же формы, что и x
        """
        batch_size, seq_len, d_model = x.shape
        
        # Создаем позиции с учетом sequence_ids
        # Для каждой последовательности позиции начинаются с 0
        positions = torch.zeros_like(sequence_ids, dtype=torch.long)
        
        for b in range(batch_size):
            current_pos = 0
            prev_seq_id = -1
            
            for i in range(seq_len):
                seq_id = sequence_ids[b, i].item()
                
                # Если это PAD (seq_id == 0), позиция не важна
                if seq_id == 0:
                    positions[b, i] = 0
                    continue
                
                # Если началась новая последовательность, сбрасываем позицию
                if seq_id != prev_seq_id:
                    current_pos = 0
                    prev_seq_id = seq_id
                
                positions[b, i] = current_pos
                current_pos += 1
        
        # Добавляем позиционные кодирования
        # self.pe имеет форму (max_len, d_model), нужно взять нужные позиции
        pos_encoding = self.pe[positions]  # (batch_size, seq_len, d_model)
        
        return x + pos_encoding


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
            d_model: размерность модели
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
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, n_heads, d_k)
        # -> (batch_size, n_heads, seq_len, d_k)
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Вычисляем attention scores
        # (batch_size, n_heads, seq_len, d_k) @ (batch_size, n_heads, d_k, seq_len)
        # -> (batch_size, n_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Создаем block mask
        # M[i,j] = (s_i == s_j) AND (j <= i) AND (s_i != 0)
        mask = self._create_block_mask(sequence_ids)  # (batch_size, seq_len, seq_len)
        
        # Расширяем маску для всех голов
        # (batch_size, seq_len, seq_len) -> (batch_size, 1, seq_len, seq_len)
        mask = mask.unsqueeze(1)
        
        # Применяем маску: заполняем запрещенные позиции очень маленьким значением
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax по последней оси
        attn_weights = F.softmax(scores, dim=-1)
        
        # Применяем внимания к V
        # (batch_size, n_heads, seq_len, seq_len) @ (batch_size, n_heads, seq_len, d_k)
        # -> (batch_size, n_heads, seq_len, d_k)
        attn_output = torch.matmul(attn_weights, V)
        
        # Объединяем головы обратно
        # (batch_size, n_heads, seq_len, d_k) -> (batch_size, seq_len, n_heads, d_k)
        # -> (batch_size, seq_len, d_model)
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
        # (seq_len, 1) - (1, seq_len) -> (seq_len, seq_len)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=sequence_ids.device))
        
        # Создаем маску одинаковых последовательностей (s_i == s_j)
        # (batch_size, seq_len, 1) == (batch_size, 1, seq_len) -> (batch_size, seq_len, seq_len)
        seq_mask = (sequence_ids.unsqueeze(2) == sequence_ids.unsqueeze(1))
        
        # Создаем маску не-PAD токенов (s_i != 0)
        # (batch_size, seq_len, 1) != 0 -> (batch_size, seq_len, 1)
        # -> (batch_size, seq_len, seq_len) через расширение
        not_pad_mask = (sequence_ids.unsqueeze(2) != 0).expand(-1, -1, seq_len)
        
        # Объединяем все условия: (s_i == s_j) AND (j <= i) AND (s_i != 0)
        # causal_mask имеет форму (seq_len, seq_len), нужно расширить до (batch_size, seq_len, seq_len)
        causal_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Финальная маска
        mask = seq_mask & causal_mask & not_pad_mask
        
        return mask


class FFN(nn.Module):
    """
    Feed-Forward Network модуль.
    
    Двухслойный перцептрон с активацией GELU.
    """
    
    def __init__(self, d_model: int, d_ff: int):
        """
        Args:
            d_model: размерность модели
            d_ff: размерность скрытого слоя FFN
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
        
        Returns:
            (batch_size, seq_len, d_model)
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x


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
        
        # LM-head (без Softmax!)
        self.lm_head = nn.Linear(d_model, vocab_size)
    
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
        logits_flat = logits.view(-1, self.vocab_size)
        targets_flat = targets.view(-1)
        mask_flat = loss_mask.view(-1)
        
        # Применяем маску
        logits_masked = logits_flat[mask_flat]
        targets_masked = targets_flat[mask_flat]
        
        # Вычисляем loss (CrossEntropyLoss уже делает Softmax внутри!)
        loss = F.cross_entropy(logits_masked, targets_masked)
        
        return loss