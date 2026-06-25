import torch.nn as nn


class TextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size, config.d_model)

    def forward(self, input_ids):
        return self.embed(input_ids)