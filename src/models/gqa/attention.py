import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GQAAttention(nn.Module):    

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
            x: (batch_size, seq_len, d_model) при use_cache=False;
               (batch_size, new_len, d_model) — только новые токены — при use_cache=True
            sequence_ids: тензор формы (batch_size, seq_len) с ID последовательностей;
                          игнорируется при use_cache=True (см. класс докстринг)
            past_key_value: при use_cache=True — None (первый шаг/prefill) либо
                            кортеж (K_cache, V_cache) формы (batch_size, n_kv_heads, past_len, d_k)
            use_cache: False — обычный packed-batching forward (как в п. 1.1);
                       True — инкрементальный шаг инференса с KV-кэшем (п. 1.2)

        Returns:
            use_cache=False: результат внимания формы (batch_size, seq_len, d_model)
            use_cache=True: кортеж (output, present_key_value), где output —
                            (batch_size, new_len, d_model), а present_key_value —
                            (K, V) формы (batch_size, n_kv_heads, past_len+new_len, d_k)
        """
        if use_cache:
            return self._forward_with_cache(x, past_key_value)

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

    def _forward_with_cache(self, x: torch.Tensor, past_key_value):
        """
        Инкрементальный шаг инференса с KV-кэшем (п. 1.2 ЛР4).

        Args:
            x: (batch_size, new_len, d_model) — только новые токены текущего шага
               (new_len == длина промпта при prefill, new_len == 1 при decode)
            past_key_value: None либо (K_cache, V_cache) формы (batch_size, n_kv_heads, past_len, d_k)

        Returns:
            (output, present_key_value): output формы (batch_size, new_len, d_model),
            present_key_value — (K, V) формы (batch_size, n_kv_heads, past_len+new_len, d_k)
        """
        batch_size, new_len, _ = x.shape

        # Один матмул на Q, K и V для новых токенов, затем split по последней оси
        qkv = self.W_qkv(x)  # (batch_size, new_len, q_dim + 2*kv_dim)
        Q_new, K_new, V_new = torch.split(qkv, [self.q_dim, self.kv_dim, self.kv_dim], dim=-1)

        Q_new = Q_new.view(batch_size, new_len, self.n_heads, self.d_k).transpose(1, 2)
        K_new = K_new.view(batch_size, new_len, self.n_kv_heads, self.d_k).transpose(1, 2)
        V_new = V_new.view(batch_size, new_len, self.n_kv_heads, self.d_k).transpose(1, 2)

        # Дописываем новые K/V к кэшу (компактный размер: n_kv_heads, а не n_heads)
        if past_key_value is not None:
            K_cache, V_cache = past_key_value
            K = torch.cat([K_cache, K_new], dim=2)
            V = torch.cat([V_cache, V_new], dim=2)
        else:
            K, V = K_new, V_new

        present_key_value = (K, V)

        # Размножение K/V до n_heads выполняется на лету, в кэше не хранится
        K_rep = K.repeat_interleave(self.n_rep, dim=1)  # (batch_size, n_heads, total_len, d_k)
        V_rep = V.repeat_interleave(self.n_rep, dim=1)  # (batch_size, n_heads, total_len, d_k)

        # scores: (batch_size, n_heads, new_len, total_len)
        scores = torch.matmul(Q_new, K_rep.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Маска нужна только при new_len > 1 (prefill): новые токены не должны
        # видеть друг друга "из будущего"; кэшированная часть разрешена целиком.
        # При new_len == 1 маска не нужна вовсе — единственный новый токен
        # закономерно видит весь кэш и себя самого.
        if new_len > 1:
            total_len = K.shape[2]
            past_len = total_len - new_len
            # allowed[i, j] = (j <= past_len + i)
            causal_mask = torch.tril(
                torch.ones(new_len, total_len, device=x.device, dtype=torch.bool),
                diagonal=past_len,
            )
            scores = scores.masked_fill(~causal_mask, torch.finfo(scores.dtype).min)

        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V_rep)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, new_len, self.d_model)
        output = self.W_o(attn_output)

        return output, present_key_value

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
