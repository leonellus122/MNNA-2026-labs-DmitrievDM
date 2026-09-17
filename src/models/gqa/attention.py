import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GQAAttention(nn.Module):
    """
    Grouped Query Attention (GQA) с поддержкой packed batching.

    Сокращает число K/V-голов до n_kv_heads < n_heads: query-головы
    группируются вокруг общих K/V-голов, что уменьшает размер будущего
    KV-кэша и объём вычислений на инференсе. При n_kv_heads == n_heads
    GQA эквивалентен обычному MHA (BlockMaskedAttention).

    Маска: M[i,j] = (s_i == s_j) AND (j <= i) AND (s_i != 0)

    forward спроектирован cache-ready: принимает необязательные
    use_cache/past_key_value, но в рамках этого класса кэш не
    реализуется (см. п. 1.2 ЛР4 — отдельный класс поверх этого).

    Оптимизация: Q/K/V считаются одним слитным матричным умножением
    (self.W_qkv) вместо трёх отдельных — вход x общий для всех трёх
    проекций, поэтому один большой GEMM с последующим split эквивалентен
    трём маленьким, но дешевле на GPU за счёт меньшего числа запусков ядер.
    """

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int):
        """
        Args:
            d_model: размерность модели (размерность вектора-эмбеддинга)
            n_heads: количество query-голов внимания
            n_kv_heads: количество K/V-групп (n_heads % n_kv_heads == 0)
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model должен быть кратен n_heads"
        assert n_heads % n_kv_heads == 0, "n_heads должен быть кратен n_kv_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_k = d_model // n_heads
        self.n_rep = n_heads // n_kv_heads

        self.q_dim = n_heads * self.d_k
        self.kv_dim = n_kv_heads * self.d_k

        # Слитная Q/K/V-проекция: один Linear и один матмул вместо трёх.
        # Выход разбивается на [Q | K | V] по последней оси.
        self.W_qkv = nn.Linear(d_model, self.q_dim + 2 * self.kv_dim)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(
        self,
        x: torch.Tensor,
        sequence_ids: torch.Tensor,
        past_key_value=None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: тензор формы (batch_size, seq_len, d_model)
            sequence_ids: тензор формы (batch_size, seq_len) с ID последовательностей
            past_key_value: зарезервировано для KV-кэша (п. 1.2 ЛР4), пока не используется
            use_cache: зарезервировано для KV-кэша (п. 1.2 ЛР4)

        Returns:
            результат внимания формы (batch_size, seq_len, d_model)
        """
        if use_cache:
            raise NotImplementedError(
                "KV-кэш для GQAAttention пока не реализован (use_cache=True). "
                "Он появится в отдельном классе следующего пункта ЛР4 (п. 1.2), "
                "построенном поверх GQAAttention с использованием уже заложенных "
                "аргументов use_cache/past_key_value."
            )

        batch_size, seq_len, _ = x.shape

        # Один матмул на Q, K и V вместо трёх, затем split по последней оси
        qkv = self.W_qkv(x)  # (batch_size, seq_len, q_dim + 2*kv_dim)
        Q, K, V = torch.split(qkv, [self.q_dim, self.kv_dim, self.kv_dim], dim=-1)

        # Разбиваем на головы
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_kv_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_kv_heads, self.d_k).transpose(1, 2)

        # Размножаем K/V до n_heads: query-голова i использует kv-группу i // n_rep
        K_rep = K.repeat_interleave(self.n_rep, dim=1)  # (batch_size, n_heads, seq_len, d_k)
        V_rep = V.repeat_interleave(self.n_rep, dim=1)  # (batch_size, n_heads, seq_len, d_k)

        # Вычисляем attention scores
        scores = torch.matmul(Q, K_rep.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Создаем block mask
        mask = self._create_block_mask(sequence_ids)  # (batch_size, seq_len, seq_len)

        # Расширяем маску для всех голов
        mask = mask.unsqueeze(1)

        # Применяем маску: заполняем запрещенные позиции очень маленьким значением
        scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)

        # Softmax по последней оси
        attn_weights = F.softmax(scores, dim=-1)

        # Применяем внимание к V
        attn_output = torch.matmul(attn_weights, V_rep)

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
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=sequence_ids.device, dtype=torch.bool))
        # Создаем маску одинаковых последовательностей (s_i == s_j)
        seq_mask = (sequence_ids.unsqueeze(2) == sequence_ids.unsqueeze(1))

        # Создаем маску не-PAD токенов (s_i != 0)
        not_pad_mask = (sequence_ids.unsqueeze(2) != 0).expand(-1, -1, seq_len)

        # Объединяем все условия: (s_i == s_j) AND (j <= i) AND (s_i != 0)
        causal_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)

        # Финальная маска
        mask = seq_mask & causal_mask & not_pad_mask

        return mask
