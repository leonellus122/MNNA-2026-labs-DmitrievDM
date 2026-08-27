import torch
import torch.nn as nn

class FFN(nn.Module):
    """
    Feed-Forward Network модуль.
    
    Двухслойный перцептрон с активацией GELU.
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Args:
            d_model: размерность модели
            d_ff: размерность скрытого слоя FFN
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout) # Добавил
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
        
        Returns:
            (batch_size, seq_len, d_model)
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x) # Добавил
        x = self.linear2(x)
        return x