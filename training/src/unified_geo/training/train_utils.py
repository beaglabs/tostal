import math
import time
import torch
from typing import Optional


def create_optimizer(model, lr, weight_decay=0.01):
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, eps=1e-4)


def cosine_lr_schedule(optimizer, step, total_steps, warmup_steps, base_lr):
    if step < warmup_steps:
        lr = base_lr * step / max(1, warmup_steps)
    else:
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        lr = base_lr * 0.5 * (1 + math.cos(math.pi * progress))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def compute_perplexity(loss):
    return math.exp(min(loss, 20))


def freeze_module(module):
    for p in module.parameters():
        p.requires_grad = False


def unfreeze_module(module):
    for p in module.parameters():
        p.requires_grad = True


class TrainingLogger:
    def __init__(self, log_interval=100):
        self.log_interval = log_interval
        self.losses = []
        self.start_time = time.perf_counter()
        self.step_times = []

    def log(self, step, total_steps, loss, lr=None):
        self.losses.append(loss)
        self.step_times.append(time.perf_counter())
        if (step + 1) % self.log_interval == 0:
            elapsed = self.step_times[-1] - self.start_time
            steps_per_sec = (step + 1) / elapsed if elapsed > 0 else 0
            recent = self.losses[-self.log_interval:]
            avg_loss = sum(recent) / len(recent)
            extra = f" lr={lr:.2e}" if lr is not None else ""
            print(
                f"  step {step + 1:>5d}/{total_steps} | "
                f"loss={avg_loss:.4f} | ppl={compute_perplexity(avg_loss):.2f} | "
                f"{elapsed:.0f}s ({steps_per_sec:.1f} step/s){extra}"
            )
        return self.losses

    def summary(self, steps_done):
        elapsed = time.perf_counter() - self.start_time
        recent = self.losses[-min(100, len(self.losses)):]
        avg_loss = sum(recent) / len(recent) if recent else 0
        return {
            "steps": steps_done,
            "avg_loss": avg_loss,
            "ppl": compute_perplexity(avg_loss),
            "time_s": elapsed,
            "steps_per_sec": steps_done / elapsed if elapsed > 0 else 0,
        }


def save_checkpoint(model, optimizer, step, loss_history, path, metadata=None):
    import os
    data = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "step": step,
        "loss_history": loss_history,
        "metadata": metadata or {},
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(data, path)


def load_checkpoint(model, optimizer, path, map_location="cpu"):
    data = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(data["model_state"], strict=False)
    if optimizer is not None:
        optimizer.load_state_dict(data["optimizer_state"])
    return data["step"], data.get("loss_history", []), data.get("metadata", {})