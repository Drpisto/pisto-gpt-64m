import torch as th
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        seq_len = x.size(1)
        attn_mask = th.triu(th.ones(seq_len, seq_len, device=x.device, dtype=th.bool), diagonal=1)
        x1 = self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]
        x = self.ln1(x + self.drop1(x1))
        x = self.ln2(x + self.drop2(self.mlp(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.block = Block(d_model, nhead, dim_feedforward, dropout)

    def forward(self, x):
        return self.block(x)
