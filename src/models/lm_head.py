import torch.nn as nn 

class LMHead(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        return self.linear(x)