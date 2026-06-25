import torch
import torch.nn as nn
from typing import Optional, Tuple

from .config import ModelConfig
from .encoder import MurmurativeEncoder
from .decoder import MoEDecoderLayer
from .heads import FaciesHead, LithologyHead, KrigingHead


class MoELanguageDecoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            MoEDecoderLayer(config) for _ in range(config.decoder_layers)
        ])
        self.ln_final = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(
        self,
        decoder_input_ids: torch.Tensor,
        slot_k: torch.Tensor,
        slot_v: torch.Tensor,
        text_embed: nn.Embedding,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = text_embed(decoder_input_ids)
        N = x.shape[1]

        causal_mask = torch.triu(
            torch.full((N, N), float("-inf"), device=x.device), diagonal=1
        )

        slot_pool_summary = slot_v.mean(dim=2).reshape(slot_v.shape[0], -1)

        total_balance_loss = 0.0
        total_z_loss = 0.0

        for layer in self.layers:
            x, bal, z = layer(x, slot_k, slot_v, slot_pool_summary, causal_mask)
            total_balance_loss = total_balance_loss + bal
            total_z_loss = total_z_loss + z

        x = self.ln_final(x)
        logits = self.lm_head(x)

        return logits, total_balance_loss, total_z_loss


class UnifiedGeoscienceModel(nn.Module):
    def __init__(self, config: ModelConfig = None):
        super().__init__()
        if config is None:
            config = ModelConfig()
        self.config = config

        self.encoder = MurmurativeEncoder(config)
        self.decoder = MoELanguageDecoder(config)
        self.facies_head = FaciesHead(config)
        self.lithology_head = LithologyHead(config)
        self.kriging_head = KrigingHead(config)

        if config.tie_embeddings:
            self.decoder.lm_head.weight = self.encoder.text_encoder.embed.weight

    def forward(
        self,
        well_log: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
        spatial: Optional[torch.Tensor] = None,
        text_ids: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        kriging_query_points: Optional[torch.Tensor] = None,
        mode: str = "encode",
    ) -> dict:
        results = {}

        encoder_tokens, slot_k, slot_v = self.encoder(
            well_log=well_log, image=image, spatial=spatial, text_ids=text_ids,
        )
        results["encoder_tokens"] = encoder_tokens
        results["slot_k"] = slot_k
        results["slot_v"] = slot_v

        if mode in ("encode", "full"):
            results["facies_logits"] = self.facies_head(slot_v)
            results["lithology_logits"] = self.lithology_head(slot_v)
            if kriging_query_points is not None:
                results["krige_values"] = self.kriging_head(slot_v, kriging_query_points)

        if mode in ("decode", "full"):
            if decoder_input_ids is None:
                raise ValueError("decoder_input_ids required for decode mode")
            lm_logits, balance_loss, z_loss = self.decoder(
                decoder_input_ids, slot_k, slot_v,
                self.encoder.text_encoder.embed,
            )
            results["lm_logits"] = lm_logits
            results["balance_loss"] = balance_loss
            results["z_loss"] = z_loss

        return results

    def count_params(self) -> dict:
        counts = {}
        for name, module in [
            ("encoder", self.encoder),
            ("decoder", self.decoder),
            ("facies_head", self.facies_head),
            ("lithology_head", self.lithology_head),
            ("kriging_head", self.kriging_head),
        ]:
            counts[name] = sum(p.numel() for p in module.parameters())
        counts["total"] = sum(counts.values())
        return counts

    def save(self, path: str) -> None:
        import json
        import os
        data = {
            "state_dict": self.state_dict(),
            "config": {
                "d_model": self.config.d_model,
                "num_heads": self.config.num_heads,
                "num_slots": self.config.num_slots,
                "num_rounds": self.config.num_rounds,
                "num_experts": self.config.num_experts,
                "decoder_layers": self.config.decoder_layers,
                "vocab_size": self.config.vocab_size,
            },
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(data, path)

    @staticmethod
    def load(path: str, map_location: str = "cpu") -> "UnifiedGeoscienceModel":
        data = torch.load(path, map_location=map_location, weights_only=False)
        from .config import ModelConfig
        config = ModelConfig(**{k: v for k, v in data["config"].items() if hasattr(ModelConfig, k)})
        model = UnifiedGeoscienceModel(config)
        model.load_state_dict(data["state_dict"], strict=False)
        return model