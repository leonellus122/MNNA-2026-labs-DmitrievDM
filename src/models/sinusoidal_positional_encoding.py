import math
import torch
import torch.nn as nn


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
        # batch_size, seq_len, d_model = x.shape
        
        # # Создаем позиции с учетом sequence_ids
        # # Для каждой последовательности позиции начинаются с 0
        # positions = torch.zeros_like(sequence_ids, dtype=torch.long)
        
        # for b in range(batch_size):
        #     current_pos = 0
        #     prev_seq_id = -1
            
        #     for i in range(seq_len):
        #         seq_id = sequence_ids[b, i].item()
                
        #         # Если это PAD (seq_id == 0), позиция не важна
        #         if seq_id == 0:
        #             positions[b, i] = 0
        #             continue
                
        #         # Если началась новая последовательность, сбрасываем позицию
        #         if seq_id != prev_seq_id:
        #             current_pos = 0
        #             prev_seq_id = seq_id
                
        #         positions[b, i] = current_pos
        #         current_pos += 1
        
        # # Добавляем позиционные кодирования
        # # self.pe имеет форму (max_len, d_model), нужно взять нужные позиции
        # pos_encoding = self.pe[positions]  # (batch_size, seq_len, d_model)
        
        # return x + pos_encoding
        batch_size, seq_len, d_model = x.shape

        is_start = torch.cat([
            torch.ones_like(sequence_ids[:, :1]),
            (sequence_ids[:, 1:] != sequence_ids[:, :-1]).long()
        ], dim=1)
        idx = torch.arange(seq_len, device=sequence_ids.device).unsqueeze(0).expand_as(sequence_ids)
        seg_start = torch.where(is_start.bool(), idx, torch.zeros_like(idx))
        seg_start = torch.cummax(seg_start, dim=1).values
        positions = (idx - seg_start).clamp(min=0)
        positions = positions.masked_fill(sequence_ids == 0, 0)

        pos_encoding = self.pe[positions]
        return x + pos_encoding

