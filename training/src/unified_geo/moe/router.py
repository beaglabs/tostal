import torch
import torch.nn as nn
import torch.nn.functional as F


class MoERouter(nn.Module):
    def __init__(self, d_model: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(d_model * 2, num_experts, bias=False)

    def forward(self, hidden, slot_pool_summary):
        combined = torch.cat([
            hidden,
            slot_pool_summary.unsqueeze(1).expand(-1, hidden.shape[1], -1),
        ], dim=-1)
        logits = self.gate(combined)
        gates = F.softmax(logits, dim=-1)
        topk_gates, topk_indices = torch.topk(gates, k=self.top_k, dim=-1)
        topk_gates = topk_gates / (topk_gates.sum(dim=-1, keepdim=True) + 1e-8)

        balance_loss = self._load_balance_loss(gates)
        z_loss = self._router_z_loss(logits)

        return topk_gates, topk_indices, balance_loss, z_loss

    def _load_balance_loss(self, gates):
        f = gates.float().mean(dim=0)
        return self.num_experts * (f * f).sum()

    def _router_z_loss(self, logits):
        return torch.logsumexp(logits, dim=-1).pow(2).mean()