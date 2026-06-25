import torch
import torch.nn as nn
import torch.nn.functional as F

from .moe import MoEFFN


class MoEDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_model = config.d_model

        self.self_attn = nn.MultiheadAttention(
            d_model, config.decoder_heads, batch_first=True,
        )
        self.self_attn_norm = nn.LayerNorm(d_model)

        self.cross_attn = nn.MultiheadAttention(
            d_model, config.decoder_heads, batch_first=True,
        )
        self.cross_attn_norm = nn.LayerNorm(d_model)

        self.moe_ffn = MoEFFN(
            d_model, config.num_experts, config.experts_per_token,
            config.expert_ffn_expansion,
        )
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(self, x, slot_k, slot_v, slot_pool_summary, causal_mask=None):
        residual = x
        x_norm = self.self_attn_norm(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, attn_mask=causal_mask)
        x = residual + attn_out

        residual = x
        x_norm = self.cross_attn_norm(x)
        B, N, D = x_norm.shape
        cross_out, _ = self.cross_attn(
            x_norm,
            slot_v.view(B, -1, D),
            slot_v.view(B, -1, D),
        )
        x = residual + cross_out

        residual = x
        x_norm = self.ffn_norm(x)
        ffn_out, balance_loss, z_loss = self.moe_ffn(x_norm, slot_pool_summary)
        x = residual + ffn_out

        return x, balance_loss, z_loss