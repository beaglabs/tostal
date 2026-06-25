import torch
import torch.nn.functional as F

from .train_utils import (
    create_optimizer, cosine_lr_schedule, TrainingLogger,
    save_checkpoint, freeze_module,
)
from ..model import UnifiedGeoscienceModel
from ..config import ModelConfig, TrainingConfig
from ..data import mixed_batch_iterator


def run_phase1(
    model: UnifiedGeoscienceModel = None,
    model_cfg: ModelConfig = None,
    train_cfg: TrainingConfig = None,
    device: str = "cpu",
):
    if model_cfg is None:
        model_cfg = ModelConfig()
    if train_cfg is None:
        train_cfg = TrainingConfig()
    if model is None:
        model = UnifiedGeoscienceModel(model_cfg)
    model = model.to(device)
    model.train()

    optimizer = create_optimizer(model, train_cfg.lr, train_cfg.weight_decay)
    logger = TrainingLogger(log_interval=100)

    data_iter = mixed_batch_iterator(
        train_cfg.batch_size, model_cfg, train_cfg, device,
        num_batches=train_cfg.max_steps,
    )

    for step in range(train_cfg.max_steps):
        batch = next(data_iter)
        lr = cosine_lr_schedule(
            optimizer, step, train_cfg.max_steps,
            train_cfg.warmup_steps, train_cfg.lr,
        )

        results = model(
            well_log=batch["well_log"],
            image=batch["image"],
            spatial=batch["spatial"],
            mode="encode",
        )

        well_pred = model.encoder.well_log_encoder(batch["well_log"])
        well_loss = F.mse_loss(well_pred, model.encoder.well_log_encoder(batch["well_target"]))

        image_pred = model.encoder.image_encoder(batch["image"])
        image_loss = F.mse_loss(image_pred, model.encoder.image_encoder(batch["image_target"]))

        spatial_values = batch["spatial_target"][..., 3:]
        spatial_loss = F.mse_loss(
            results["encoder_tokens"][:, :batch["spatial_target"].shape[1], :3].mean(-1),
            spatial_values.squeeze(-1),
        )

        loss = well_loss + image_loss + spatial_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()

        logger.log(step, train_cfg.max_steps, loss.item(), lr)

    summary = logger.summary(train_cfg.max_steps)
    save_checkpoint(
        model, optimizer, train_cfg.max_steps, logger.losses,
        "checkpoints/phase1/model.pt",
        {"phase": 1, **summary},
    )
    print(f"\nPhase 1 complete: ppl={summary['ppl']:.2f} in {summary['time_s']:.0f}s")
    return model, summary