"""Baseline FP32 MobileNet-v2 training script for CIFAR-10 (Question 1).

Usage:
    python -m src.train --epochs 200 --batch_size 128 --lr 0.1 --seed 42
"""

import argparse
import csv
import math
import os
import time
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm

from src.data import get_cifar10_loaders
from src.evaluate import analyze_failure_modes, evaluate_model
from src.model import build_mobilenetv2_cifar10
from src.utils import AverageMeter, ensure_dir, get_device, get_environment_info, save_json, set_seed


def build_optimizer_param_groups(model: nn.Module, weight_decay: float) -> List[Dict]:
    """Exclude BatchNorm and bias parameters from L2 weight decay (MobileNet-v2 best practice)."""
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1 or name.endswith(".bias") or "bn" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    total_epochs: int,
    warmup_epochs: int = 5,
    min_lr_ratio: float = 1e-4,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup followed by cosine decay down to `min_lr_ratio * base_lr`."""

    def lr_lambda(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def plot_training_curves(history: List[Dict[str, float]], out_path: str) -> None:
    """Plot training & test loss and Top-1 accuracy curves for Question 1(c)."""
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    test_loss = [h["test_loss"] for h in history]
    train_acc = [h["train_acc"] for h in history]
    test_acc = [h["test_acc"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)

    axes[0].plot(epochs, train_loss, label="Train Loss", linewidth=2.0)
    axes[0].plot(epochs, test_loss, label="Test Loss", linewidth=2.0)
    axes[0].set_title("MobileNet-v2 CIFAR-10 Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, train_acc, label="Train Top-1 Acc", linewidth=2.0)
    axes[1].plot(epochs, test_acc, label="Test Top-1 Acc", linewidth=2.0)
    best_acc = max(test_acc)
    best_ep = epochs[test_acc.index(best_acc)]
    axes[1].scatter([best_ep], [best_acc], color="red", zorder=5, label=f"Best Test: {best_acc:.2f}% (ep {best_ep})")
    axes[1].set_title("MobileNet-v2 CIFAR-10 Accuracy Curves")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Top-1 Accuracy (%)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    ensure_dir(os.path.dirname(out_path) or ".")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def train_baseline(args: argparse.Namespace) -> Dict[str, object]:
    """Train the uncompressed FP32 MobileNet-v2 baseline on CIFAR-10."""
    set_seed(args.seed, deterministic=not args.fast_cudnn)
    device = get_device(args.device)
    ensure_dir(args.save_dir)
    ensure_dir(args.results_dir)

    print(f"[Baseline] Device: {device} | Seed: {args.seed} | Epochs: {args.epochs}")

    train_loader, test_loader, _ = get_cifar10_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        use_random_erasing=not args.no_random_erasing,
        seed=args.seed,
    )

    model = build_mobilenetv2_cifar10(
        width_mult=args.width_mult,
        dropout=args.dropout,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        num_classes=10,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Baseline] MobileNetV2CIFAR (width_mult={args.width_mult}) | Params: {total_params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    param_groups = build_optimizer_param_groups(model, weight_decay=args.weight_decay)
    optimizer = torch.optim.SGD(
        param_groups, lr=args.lr, momentum=args.momentum, nesterov=True
    )
    scheduler = build_warmup_cosine_scheduler(
        optimizer, total_epochs=args.epochs, warmup_epochs=args.warmup_epochs
    )

    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    wandb_run = None
    if args.wandb:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=f"baseline_fp32_w{args.width_mult}_seed{args.seed}",
            config=vars(args),
        )

    history: List[Dict[str, float]] = []
    best_acc = 0.0
    best_epoch = 0
    best_ckpt_path = os.path.join(args.save_dir, "baseline_best.pth")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_meter = AverageMeter("train_loss")
        acc_meter = AverageMeter("train_acc")

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch:03d}/{args.epochs:03d}",
            leave=False,
            disable=args.no_tqdm,
        )
        for images, targets in pbar:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda" if use_amp else "cpu", enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            preds = logits.argmax(dim=1)
            acc = 100.0 * (preds == targets).float().mean().item()
            loss_meter.update(loss.item(), images.size(0))
            acc_meter.update(acc, images.size(0))
            pbar.set_postfix(loss=f"{loss_meter.avg:.3f}", acc=f"{acc_meter.avg:.2f}%")

        scheduler.step()
        eval_res = evaluate_model(model, test_loader, device)
        cur_lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "lr": round(cur_lr, 6),
            "train_loss": round(loss_meter.avg, 5),
            "train_acc": round(acc_meter.avg, 3),
            "test_loss": round(eval_res["loss"], 5),
            "test_acc": round(eval_res["top1_acc"], 3),
            "test_top5_acc": round(eval_res["top5_acc"], 3),
        }
        history.append(row)

        if eval_res["top1_acc"] > best_acc:
            best_acc = eval_res["top1_acc"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_acc": best_acc,
                    "epoch": epoch,
                    "seed": args.seed,
                    "model_config": {
                        "width_mult": args.width_mult,
                        "dropout": args.dropout,
                        "bn_momentum": args.bn_momentum,
                        "bn_eps": args.bn_eps,
                        "num_classes": 10,
                    },
                    "train_args": vars(args),
                },
                best_ckpt_path,
            )

        if wandb_run is not None:
            wandb_run.log(row)

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"Epoch {epoch:03d}/{args.epochs:03d} | LR {cur_lr:.5f} | "
                f"Train Loss {loss_meter.avg:.4f} Acc {acc_meter.avg:.2f}% | "
                f"Test Loss {eval_res['loss']:.4f} Acc {eval_res['top1_acc']:.2f}% "
                f"(Best {best_acc:.2f}% @ ep {best_epoch})"
            )

    elapsed = time.time() - start_time
    print(f"[Baseline] Training complete in {elapsed/60.0:.2f} min | Best Test Top-1: {best_acc:.2f}%")

    # Save CSV training history and curves
    csv_path = os.path.join(args.results_dir, "baseline_history.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    curves_path = os.path.join(args.results_dir, "baseline_curves.png")
    plot_training_curves(history, curves_path)

    # Load best checkpoint and run failure mode analysis for Q1(c)
    ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    failure_summary = analyze_failure_modes(
        model, test_loader, device, output_dir=args.results_dir
    )

    summary = {
        "best_test_top1_acc": best_acc,
        "best_epoch": best_epoch,
        "total_params": total_params,
        "training_minutes": round(elapsed / 60.0, 2),
        "checkpoint_path": best_ckpt_path,
        "curves_path": curves_path,
        "failure_summary": failure_summary,
        "environment": get_environment_info(),
        "train_args": vars(args),
    }
    save_json(summary, os.path.join(args.results_dir, "baseline_summary.json"))

    if wandb_run is not None:
        wandb_run.summary["best_test_top1_acc"] = best_acc
        wandb_run.finish()

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FP32 MobileNet-v2 baseline on CIFAR-10")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--width_mult", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--bn_momentum", type=float, default=0.1)
    parser.add_argument("--bn_eps", type=float, default=1e-5)
    parser.add_argument("--no_random_erasing", action="store_true")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--fast_cudnn", action="store_true")
    parser.add_argument("--no_tqdm", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="mobilenetv2-cifar10-compression")
    return parser.parse_args()


if __name__ == "__main__":
    train_baseline(parse_args())
