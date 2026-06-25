#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from unified_geo import UnifiedGeoscienceModel, ModelConfig, TrainingConfig, run_phase3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    cfg = ModelConfig()
    train_cfg = TrainingConfig(
        max_steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    model = UnifiedGeoscienceModel(cfg)
    print(f"Model params: {model.count_params()['total']:,} total")

    model, summary = run_phase3(
        model, cfg, train_cfg, device=args.device,
        checkpoint_path=args.checkpoint,
    )
    print(f"Phase 3: {summary}")


if __name__ == "__main__":
    main()