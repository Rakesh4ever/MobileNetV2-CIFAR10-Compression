"""Evaluation, confusion matrix, and failure mode analysis utilities (Question 1c)."""

import argparse
import os
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from src.data import CIFAR10_CLASSES, CIFAR10_MEAN, CIFAR10_STD, get_cifar10_loaders
from src.model import build_mobilenetv2_cifar10
from src.utils import ensure_dir, get_device, save_json


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    return_predictions: bool = False,
) -> Dict[str, object]:
    """Compute CrossEntropy loss, Top-1 accuracy, and Top-5 accuracy."""
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_samples = 0
    correct_top1 = 0
    correct_top5 = 0

    all_preds: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    all_confs: List[np.ndarray] = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        probs = F.softmax(logits, dim=1)
        top5_idx = logits.topk(5, dim=1).indices
        preds = top5_idx[:, 0]

        correct_top1 += (preds == targets).sum().item()
        correct_top5 += (top5_idx == targets.unsqueeze(1)).any(dim=1).sum().item()

        if return_predictions:
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_confs.append(probs.max(dim=1).values.cpu().numpy())

    result: Dict[str, object] = {
        "loss": total_loss / max(total_samples, 1),
        "top1_acc": 100.0 * correct_top1 / max(total_samples, 1),
        "top5_acc": 100.0 * correct_top5 / max(total_samples, 1),
        "num_samples": total_samples,
    }
    if return_predictions:
        result["preds"] = np.concatenate(all_preds)
        result["targets"] = np.concatenate(all_targets)
        result["confs"] = np.concatenate(all_confs)
    return result


def _unnormalize_cifar(img_tensor: torch.Tensor) -> np.ndarray:
    """Invert CIFAR-10 normalization for plotting."""
    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(3, 1, 1)
    img = (img_tensor.cpu() * std + mean).clamp(0.0, 1.0)
    return img.permute(1, 2, 0).numpy()


@torch.no_grad()
def analyze_failure_modes(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    output_dir: str = "./results",
    num_examples: int = 16,
) -> Dict[str, object]:
    """Generate confusion matrix, per-class accuracy, top confused pairs, and sample errors."""
    ensure_dir(output_dir)
    eval_res = evaluate_model(model, test_loader, device, return_predictions=True)
    preds = eval_res["preds"]
    targets = eval_res["targets"]
    confs = eval_res["confs"]

    cm = confusion_matrix(targets, preds, labels=list(range(10)))

    # 1. Plot confusion matrix heatmap
    fig, ax = plt.subplots(figsize=(8.5, 7.2), dpi=150)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(10),
        yticks=np.arange(10),
        xticklabels=CIFAR10_CLASSES,
        yticklabels=CIFAR10_CLASSES,
        xlabel="Predicted Label",
        ylabel="True Label",
        title=f"MobileNet-v2 CIFAR-10 Confusion Matrix (Top-1: {eval_res['top1_acc']:.2f}%)",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(10):
        for j in range(10):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8,
            )

    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close(fig)

    # 2. Per-class accuracy and top confused pairs
    per_class_acc = {
        CIFAR10_CLASSES[i]: round(100.0 * cm[i, i] / max(cm[i].sum(), 1), 2)
        for i in range(10)
    }

    confused_pairs: List[Tuple[int, str, str]] = []
    for i in range(10):
        for j in range(10):
            if i != j and cm[i, j] > 0:
                confused_pairs.append((int(cm[i, j]), CIFAR10_CLASSES[i], CIFAR10_CLASSES[j]))
    confused_pairs.sort(reverse=True)
    top_confused = [
        {"true": t, "pred": p, "count": c} for c, t, p in confused_pairs[:10]
    ]

    # 3. Plot grid of high-confidence misclassified images
    wrong_indices = np.where(preds != targets)[0]
    if len(wrong_indices) > 0:
        wrong_order = wrong_indices[np.argsort(-confs[wrong_indices])][:num_examples]
        dataset = test_loader.dataset
        cols = 4
        rows = int(np.ceil(len(wrong_order) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.7), dpi=140)
        axes = np.array(axes).reshape(-1)
        for ax_idx, ax in enumerate(axes):
            if ax_idx < len(wrong_order):
                sample_idx = int(wrong_order[ax_idx])
                img_t, true_lbl = dataset[sample_idx]
                pred_lbl = int(preds[sample_idx])
                conf = float(confs[sample_idx]) * 100.0
                ax.imshow(_unnormalize_cifar(img_t))
                ax.set_title(
                    f"T: {CIFAR10_CLASSES[true_lbl]}\nP: {CIFAR10_CLASSES[pred_lbl]} ({conf:.1f}%)",
                    fontsize=8,
                    color="crimson",
                )
            ax.axis("off")
        fig.suptitle("Highest-Confidence Misclassified CIFAR-10 Test Samples", fontsize=11)
        plt.tight_layout()
        fail_path = os.path.join(output_dir, "failure_modes.png")
        plt.savefig(fail_path, bbox_inches="tight")
        plt.close(fig)
    else:
        fail_path = ""

    summary = {
        "top1_acc": round(float(eval_res["top1_acc"]), 3),
        "top5_acc": round(float(eval_res["top5_acc"]), 3),
        "test_loss": round(float(eval_res["loss"]), 5),
        "total_errors": int(len(wrong_indices)),
        "per_class_acc": per_class_acc,
        "top_confused_pairs": top_confused,
        "confusion_matrix_path": cm_path,
        "failure_modes_path": fail_path,
    }
    save_json(summary, os.path.join(output_dir, "baseline_eval.json"))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate baseline checkpoint and analyze failures")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/baseline_best.pth")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = get_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("model_config", {})
    model = build_mobilenetv2_cifar10(**cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    _, test_loader, _ = get_cifar10_loaders(
        data_dir=args.data_dir, eval_batch_size=args.batch_size
    )
    summary = analyze_failure_modes(model, test_loader, device, output_dir=args.output_dir)
    print(f"Top-1 Acc: {summary['top1_acc']:.2f}% | Top-5 Acc: {summary['top5_acc']:.2f}%")
    print("Top confused pairs:", summary["top_confused_pairs"][:5])


if __name__ == "__main__":
    main()
