import torch
import torch.nn.functional as F

from .train_utils import (
    create_optimizer, cosine_lr_schedule, TrainingLogger,
    save_checkpoint, freeze_module, load_checkpoint,
)
from ..model import UnifiedGeoscienceModel
from ..config import ModelConfig, TrainingConfig
from ..data import mixed_batch_iterator, geology_text_generator


def run_phase3(
    model: UnifiedGeoscienceModel = None,
    model_cfg: ModelConfig = None,
    train_cfg: TrainingConfig = None,
    device: str = "cpu",
    checkpoint_path: str = "checkpoints/phase2/model.pt",
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
                print("No phase2 checkpoint found, starting from scratch")
    model = model.to(device)
    model.train()

    freeze_module(model.encoder)
    freeze_module(model.facies_head)
    freeze_module(model.lithology_head)
    freeze_module(model.kriging_head)

    optimizer = create_optimizer(model.decoder, train_cfg.lr, train_cfg.weight_decay)
    logger = TrainingLogger(log_interval=100)

    data_iter = mixed_batch_iterator(
        train_cfg.batch_size, model_cfg, train_cfg, device,
        num_batches=train_cfg.max_steps,
    )
    text_iter = geology_text_generator(
        model_cfg.vocab_size, train_cfg.text_seq_len,
        train_cfg.max_steps, seed=42,
    )

    for step in range(train_cfg.max_steps):
        batch = next(data_iter)
        text_batch = next(text_iter).expand(train_cfg.batch_size, -1).to(device)

        lr = cosine_lr_schedule(
            optimizer, step, train_cfg.max_steps,
            train_cfg.warmup_steps, train_cfg.lr,
        )

        results = model(
            well_log=batch["well_log"],
            image=batch["image"],
            spatial=batch["spatial"],
            decoder_input_ids=text_batch[:, :-1],
            mode="decode",
        )

        lm_loss = F.cross_entropy(
            results["lm_logits"].flatten(0, 1),
            text_batch[:, 1:].flatten(0, 1),
        )

        balance_loss = results["balance_loss"]
        z_loss = results["z_loss"]
        loss = lm_loss + train_cfg.load_balance_alpha * balance_loss + train_cfg.router_z_loss_beta * z_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()

        logger.log(step, train_cfg.max_steps, lm_loss.item(), lr)

    summary = logger.summary(train_cfg.max_steps)
    save_checkpoint(
        model, optimizer, train_cfg.max_steps, logger.losses,
        "checkpoints/phase3/model.pt",
        {"phase": 3, **summary},
    )
    print(f"\nPhase 3 complete: ppl={summary['ppl']:.2f} in {summary['time_s']:.0f}s")
    return model, summary