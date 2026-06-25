import torch
import torch.nn as nn


class ExpertFFN(nn.Module):
    def __init__(self, d_model: int, expansion: int = 4):
        super().__init__()
        hidden = d_model * expansion
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d_model, bias=False)
        self.act = nn.GELU()

    def forward(self, x):
        return self.w2(self.act(self.w1(x)))