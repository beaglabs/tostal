from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    d_model: int = 512
    num_heads: int = 8
    num_slots: int = 512
    num_rounds: int = 3
    alpha: float = 0.9
    gamma: float = 0.15
    k_neighbors: int = 7

    vocab_size: int = 32000

    well_log_in_channels: int = 6
    well_log_conv_channels: list = field(default_factory=lambda: [32, 64, 128, 256])
    well_log_stride: int = 2
    well_log_output_tokens: int = 64

    image_patch_size: int = 16
    image_vit_layers: int = 2
    image_vit_heads: int = 4
    image_channels: int = 3

    spatial_mlp_hidden: int = 256

    num_experts: int = 6
    experts_per_token: int = 2
    expert_ffn_expansion: int = 4
    decoder_layers: int = 6
    decoder_heads: int = 8

    facies_num_classes: int = 12
    lithology_num_classes: int = 12

    tie_embeddings: bool = True

    def __post_init__(self):
        self.dh = self.d_model // self.num_heads
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")


@dataclass
class TrainingConfig:
    lr: float = 5e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_steps: int = 10000
    batch_size: int = 8
    seq_len: int = 128
    grad_clip: float = 1.0

    well_log_depth: int = 512
    image_size: int = 64
    spatial_n_points: int = 64
    text_seq_len: int = 96

    well_mask_prob: float = 0.2
    image_mask_prob: float = 0.3
    spatial_mask_prob: float = 0.25

    load_balance_alpha: float = 0.01
    router_z_loss_beta: float = 0.001

    phase2_freeze_steps: int = 2500
    phase4_lr_multiplier: float = 0.1


DEFAULT_MODEL = ModelConfig()
DEFAULT_TRAINING = TrainingConfig()