import torch
import torch.nn as nn
import torch.nn.functional as F

from .router import MoERouter
from .experts import ExpertFFN


class MoEFFN(nn.Module):
    def __init__(self, d_model: int, num_experts: int, top_k: int, expansion: int = 4):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k

        self.experts = nn.ModuleList([
            ExpertFFN(d_model, expansion) for _ in range(num_experts)
        ])
        self.router = MoERouter(d_model, num_experts, top_k)

    def forward(self, x, slot_pool_summary):
        B, N, D = x.shape
        topk_gates, topk_indices, balance_loss, z_loss = self.router(x, slot_pool_summary)

        x_flat = x.view(-1, D)
        indices_flat = topk_indices.view(-1, self.top_k)
        gates_flat = topk_gates.view(-1, self.top_k)

        output = torch.zeros_like(x_flat)
        for e in range(self.num_experts):
            expert_mask = (indices_flat == e).any(dim=-1)
            if not expert_mask.any():
                continue
            expert_input = x_flat[expert_mask]
            expert_output = self.experts[e](expert_input)
            for k in range(self.top_k):
                k_mask = indices_flat[:, k] == e
                k_mask &= expert_mask
                if k_mask.any():
                    output[k_mask] += gates_flat[k_mask, k:k + 1] * expert_output[
                        k_mask[expert_mask].nonzero(as_tuple=True)[0]
                    ]

        return output.view(B, N, D), balance_loss, z_loss