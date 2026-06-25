import torch
import torch.nn.functional as F

from .train_utils import (
    create_optimizer, cosine_lr_schedule, TrainingLogger,
    save_checkpoint, unfreeze_module, load_checkpoint,
)
from ..model import UnifiedGeoscienceModel
from ..config import ModelConfig, TrainingConfig
from ..data import mixed_batch_iterator, geology_text_generator


def run_phase4(
    model: UnifiedGeoscienceModel = None,
    model_cfg: ModelConfig = None,
    train_cfg: TrainingConfig = None,
    device: str = "cpu",
    checkpoint_path: str = "checkpoints/phase3/model.pt",
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
                print("No phase3 checkpoint found, starting from scratch")
    model = model.to(device)
    model.train()

    unfreeze_module(model.encoder)
    unfreeze_module(model.facies_head)
    unfreeze_module(model.lithology_head)
    unfreeze_module(model.kriging_head)
    unfreeze_module(model.decoder)

    base_lr = train_cfg.lr * train_cfg.phase4_lr_multiplier
    optimizer = create_optimizer(model, base_lr, train_cfg.weight_decay)
    logger = TrainingLogger(log_interval=100)

    data_iter = mixed_batch_iterator(
        train_cfg.batch_size, model_cfg, train_cfg, device,
        num_batches=train_cfg.max_steps,
    )
    text_iter = geology_text_generator(
        model_cfg.vocab_size, train_cfg.text_seq_len,
        train_cfg.max_steps, seed=99,
    )

    for step in range(train_cfg.max_steps):
        batch = next(data_iter)
        text_batch = next(text_iter).expand(train_cfg.batch_size, -1).to(device)

        lr = cosine_lr_schedule(
            optimizer, step, train_cfg.max_steps,
            train_cfg.warmup_steps, base_lr,
        )

        results = model(
            well_log=batch["well_log"],
            image=batch["image"],
            spatial=batch["spatial"],
            decoder_input_ids=text_batch[:, :-1],
            kriging_query_points=torch.rand(train_cfg.batch_size, 32, 3, device=device),
            mode="full",
        )

        well_pred = model.encoder.well_log_encoder(batch["well_log"])
        well_loss = F.mse_loss(well_pred, model.encoder.well_log_encoder(batch["well_target"]))

        image_pred = model.encoder.image_encoder(batch["image"])
        image_loss = F.mse_loss(image_pred, model.encoder.image_encoder(batch["image_target"]))

        facies_target = torch.randint(
            0, model_cfg.facies_num_classes, (train_cfg.batch_size,), device=device
        )
        facies_loss = F.cross_entropy(results["facies_logits"], facies_target)

        litho_target = torch.randint(
            0, model_cfg.lithology_num_classes,
            (train_cfg.batch_size, 64, 64), device=device,
        )
        litho_loss = F.cross_entropy(results["lithology_logits"], litho_target)

        lm_loss = F.cross_entropy(
            results["lm_logits"].flatten(0, 1),
            text_batch[:, 1:].flatten(0, 1),
        )

        loss = (
            well_loss + image_loss +
            facies_loss + litho_loss +
            lm_loss +
            train_cfg.load_balance_alpha * results["balance_loss"] +
            train_cfg.router_z_loss_beta * results["z_loss"]
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()

        logger.log(step, train_cfg.max_steps, loss.item(), lr)

    summary = logger.summary(train_cfg.max_steps)
    save_checkpoint(
        model, optimizer, train_cfg.max_steps, logger.losses,
        "checkpoints/phase4/model.pt",
        {"phase": 4, **summary},
    )
    print(f"\nPhase 4 complete: ppl={summary['ppl']:.2f} in {summary['time_s']:.0f}s")
    return model, summary