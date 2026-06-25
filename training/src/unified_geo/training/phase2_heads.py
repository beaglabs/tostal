import torch
import torch.nn.functional as F

from .train_utils import (
    create_optimizer, cosine_lr_schedule, TrainingLogger,
    save_checkpoint, freeze_module, unfreeze_module, load_checkpoint,
)
from ..model import UnifiedGeoscienceModel
from ..config import ModelConfig, TrainingConfig
from ..data import mixed_batch_iterator


def run_phase2(
    model: UnifiedGeoscienceModel = None,
    model_cfg: ModelConfig = None,
    train_cfg: TrainingConfig = None,
    device: str = "cpu",
    checkpoint_path: str = "checkpoints/phase1/model.pt",
):
    if model_cfg is None:
        model_cfg = ModelConfig()
    if train_cfg is None:
        train_cfg = TrainingConfig()
    if model is None:
        model = UnifiedGeoscienceModel(model_cfg)
        if checkpoint_path:
            try:
                model = UnifiedGeoscienceModel.load(checkpoint_path, map_location=device)
                print(f"Loaded checkpoint from {checkpoint_path}")
            except FileNotFoundError:
                print("No phase1 checkpoint found, starting from scratch")
    model = model.to(device)
    model.train()

    freeze_module(model.encoder)
    optimizer = create_optimizer(
        model, train_cfg.lr, train_cfg.weight_decay
    )
    logger = TrainingLogger(log_interval=100)

    data_iter = mixed_batch_iterator(
        train_cfg.batch_size, model_cfg, train_cfg, device,
        num_batches=train_cfg.max_steps,
    )

    for step in range(train_cfg.max_steps):
        if step == train_cfg.phase2_freeze_steps:
            unfreeze_module(model.encoder)
            print("  Unfroze encoder slot pool for joint fine-tuning")

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

        facies_target = torch.randint(
            0, model_cfg.facies_num_classes, (train_cfg.batch_size,), device=device
        )
        facies_loss = F.cross_entropy(results["facies_logits"], facies_target)

        litho_target = torch.randint(
            0, model_cfg.lithology_num_classes,
            (train_cfg.batch_size, 64, 64), device=device,
        )
        litho_loss = F.cross_entropy(results["lithology_logits"], litho_target)

        krige_q = torch.rand(train_cfg.batch_size, 32, 3, device=device)
        krige_target = torch.rand(train_cfg.batch_size, 32, device=device)
        krige_pred = model.kriging_head(results["slot_v"], krige_q)
        krige_loss = F.mse_loss(krige_pred, krige_target)

        loss = facies_loss + litho_loss + krige_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()

        logger.log(step, train_cfg.max_steps, loss.item(), lr)

    summary = logger.summary(train_cfg.max_steps)
    save_checkpoint(
        model, optimizer, train_cfg.max_steps, logger.losses,
        "checkpoints/phase2/model.pt",
        {"phase": 2, **summary},
    )
    print(f"\nPhase 2 complete: ppl={summary['ppl']:.2f} in {summary['time_s']:.0f}s")
    return model, summary