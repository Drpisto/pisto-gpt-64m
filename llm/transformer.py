import torch.nn as nn
from decoder import Block


class transformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.block = Block(d_model, nhead, dim_feedforward, dropout)
                        
    def forward(self, x):
        return self.block(x)
