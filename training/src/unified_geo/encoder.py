import torch
import torch.nn as nn
from typing import Optional, Tuple

from .config import ModelConfig
from .encoders import WellLogEncoder, ImageEncoder, SpatialEncoder, TextEncoder
from .reference import slot_murmurate_reference


class MurmurativeEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        H = config.num_heads
        M = config.num_slots

        self.well_log_encoder = WellLogEncoder(config)
        self.image_encoder = ImageEncoder(config)
        self.spatial_encoder = SpatialEncoder(config)
        self.text_encoder = TextEncoder(config)

        self.q_text_spatial = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_text_spatial = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_text_spatial = nn.Linear(config.d_model, config.d_model, bias=False)

        self.q_image = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_image = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_image = nn.Linear(config.d_model, config.d_model, bias=False)

        self.slot_k_emb = nn.Parameter(torch.randn(H, M, config.dh) * 0.02)
        self.slot_v_emb = nn.Parameter(torch.randn(H, M, config.dh) * 0.02)

        self.output_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.ln_post = nn.LayerNorm(config.d_model)

    def forward(
        self,
        well_log: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
        spatial: Optional[torch.Tensor] = None,
        text_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens_list = []

        if well_log is not None:
            tokens_list.append(self.well_log_encoder(well_log))
        if image is not None:
            tokens_list.append(self.image_encoder(image))
        if spatial is not None:
            tokens_list.append(self.spatial_encoder(spatial))
        if text_ids is not None:
            tokens_list.append(self.text_encoder(text_ids))

        if not tokens_list:
            raise ValueError("At least one modality must be provided")

        x = torch.cat(tokens_list, dim=1)
        B, N, D = x.shape

        text_spatial_tokens = x.clone()
        image_tokens = x.clone()

        q_text_sp = self.q_text_spatial(text_spatial_tokens)
        k_text_sp = self.k_text_spatial(text_spatial_tokens)
        v_text_sp = self.v_text_spatial(text_spatial_tokens)
        q_img = self.q_image(image_tokens)
        k_img = self.k_image(image_tokens)
        v_img = self.v_image(image_tokens)

        q = q_text_sp + q_img
        k = k_text_sp + k_img
        v = v_text_sp + v_img

        attended = slot_murmurate_reference(
            x, q, k, v,
            self.slot_k_emb, self.slot_v_emb,
            num_heads=self.config.num_heads,
            rounds=self.config.num_rounds,
            alpha=self.config.alpha,
            gamma=self.config.gamma,
        )

        out = self.output_proj(attended)
        out = self.ln_post(out)

        H = self.config.num_heads
        M = self.config.num_slots
        slot_k_out = self.slot_k_emb.unsqueeze(0).expand(B, H, M, self.config.dh).clone()
        slot_v_out = self.slot_v_emb.unsqueeze(0).expand(B, H, M, self.config.dh).clone()

        return out, slot_k_out, slot_v_out