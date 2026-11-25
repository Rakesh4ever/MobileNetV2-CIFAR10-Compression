"""Compression sweep runner with Weights & Biases (WandB) Parallel Coordinates logging.

Runs >= 8 (default 16) compression configurations across varying weight bit-widths,
activation bit-widths, and structured channel pruning ratios. Supports `--resume`
so Colab runs can safely continue if disconnected.

Usage:
    python sweep.py --checkpoint checkpoints/baseline_best.pth --wandb --wandb_project mobilenetv2-cifar10-compression
"""

import argparse
import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.compress_pipeline import run_compression_pipeline
from src.data import get_cifar10_loaders
from src.evaluate import evaluate_model
from src.model import build_mobilenetv2_cifar10
from src.utils import ensure_dir, get_device, save_json, set_seed


# Curated 16-configuration grid spanning pure quantization + structured pruning + QAT
# Designed to explore the accuracy-vs-compression Pareto frontier for relative grading.
FULL_SWEEP_CONFIGS: List[Dict[str, object]] = [
    # --- Pure Quantization (prune_ratio = 0.0) ---
    {"name": "W8_A8_P00", "w_bits": 8, "a_bits": 8, "prune": 0.00, "qat_ep": 10, "p_ep": 0},
    {"name": "W6_A6_P00", "w_bits": 6, "a_bits": 6, "prune": 0.00, "qat_ep": 15, "p_ep": 0},
    {"name": "W4_A8_P00", "w_bits": 4, "a_bits": 8, "prune": 0.00, "qat_ep": 20, "p_ep": 0},
    {"name": "W4_A6_P00", "w_bits": 4, "a_bits": 6, "prune": 0.00, "qat_ep": 20, "p_ep": 0},
    {"name": "W4_A4_P00", "w_bits": 4, "a_bits": 4, "prune": 0.00, "qat_ep": 25, "p_ep": 0},
    {"name": "W3_A4_P00", "w_bits": 3, "a_bits": 4, "prune": 0.00, "qat_ep": 25, "p_ep": 0},
    {"name": "W3_A3_P00", "w_bits": 3, "a_bits": 3, "prune": 0.00, "qat_ep": 30, "p_ep": 0},
    {"name": "W2_A4_P00", "w_bits": 2, "a_bits": 4, "prune": 0.00, "qat_ep": 30, "p_ep": 0},
    # --- Structured Pruning + Quantization QAT (high in-memory CR for BOTH weights & activations) ---
    {"name": "W8_A8_P30", "w_bits": 8, "a_bits": 8, "prune": 0.30, "qat_ep": 10, "p_ep": 12},
    {"name": "W4_A4_P25", "w_bits": 4, "a_bits": 4, "prune": 0.25, "qat_ep": 25, "p_ep": 12},
    {"name": "W4_A4_P40", "w_bits": 4, "a_bits": 4, "prune": 0.40, "qat_ep": 25, "p_ep": 15},
    {"name": "W4_A4_P55", "w_bits": 4, "a_bits": 4, "prune": 0.55, "qat_ep": 25, "p_ep": 15},
    {"name": "W3_A4_P30", "w_bits": 3, "a_bits": 4, "prune": 0.30, "qat_ep": 25, "p_ep": 15},
    {"name": "W3_A4_P45", "w_bits": 3, "a_bits": 4, "prune": 0.45, "qat_ep": 30, "p_ep": 15},
    {"name": "W3_A3_P40", "w_bits": 3, "a_bits": 3, "prune": 0.40, "qat_ep": 30, "p_ep": 15},
    {"name": "W2_A4_P35", "w_bits": 2, "a_bits": 4, "prune": 0.35, "qat_ep": 30, "p_ep": 15},
]

# Faster 8-run subset (meets minimum 8-run requirement in half the GPU time)
QUICK_SWEEP_CONFIGS: List[Dict[str, object]] = [
    FULL_SWEEP_CONFIGS[0],   # W8_A8_P00
    FULL_SWEEP_CONFIGS[1],   # W6_A6_P00
    FULL_SWEEP_CONFIGS[4],   # W4_A4_P00
    FULL_SWEEP_CONFIGS[5],   # W3_A4_P00
    FULL_SWEEP_CONFIGS[7],   # W2_A4_P00
    FULL_SWEEP_CONFIGS[10],  # W4_A4_P40
    FULL_SWEEP_CONFIGS[13],  # W3_A4_P45
    FULL_SWEEP_CONFIGS[15],  # W2_A4_P35
]


def plot_pareto_frontier(df: pd.DataFrame, out_path: str) -> None:
    """Plot Top-1 Accuracy vs. Model Compression Ratio and vs. Model Size (MB)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

    sc0 = axes[0].scatter(
        df["compression_ratio"],
        df["quantized_acc"],
        c=df["prune_ratio"],
        cmap="viridis",
        s=90,
        edgecolors="black",
        linewidth=0.7,
    )
    for _, r in df.iterrows():
        axes[0].annotate(
            r["run_name"],
            (r["compression_ratio"], r["quantized_acc"]),
            fontsize=7,
            xytext=(4, 3),
            textcoords="offset points",
        )
    axes[0].set_title("Accuracy vs. In-Memory Model Compression Ratio")
    axes[0].set_xlabel("Model Compression Ratio (x)")
    axes[0].set_ylabel("Quantized Top-1 Accuracy (%)")
    axes[0].grid(True, alpha=0.3)
    fig.colorbar(sc0, ax=axes[0], label="Prune Ratio")

    sc1 = axes[1].scatter(
        df["model_size_mb"],
        df["quantized_acc"],
        c=df["activation_compression_ratio"],
        cmap="plasma",
        s=90,
        edgecolors="black",
        linewidth=0.7,
    )
    for _, r in df.iterrows():
        axes[1].annotate(
            r["run_name"],
            (r["model_size_mb"], r["quantized_acc"]),
            fontsize=7,
            xytext=(4, 3),
            textcoords="offset points",
        )
    axes[1].set_title("Accuracy vs. Final Approximated Model Size (MB)")
    axes[1].set_xlabel("Compressed In-Memory Model Size (MB)")
    axes[1].set_ylabel("Quantized Top-1 Accuracy (%)")
    axes[1].grid(True, alpha=0.3)
    fig.colorbar(sc1, ax=axes[1], label="Activation CR (x)")

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_local_parallel_coordinates(df: pd.DataFrame, out_path: str) -> None:
    """Render a local parallel coordinates chart mirroring WandB Figure 1."""
    cols = [
        "activation_quant_bits",
        "weight_quant_bits",
        "prune_ratio",
        "compression_ratio",
        "model_size_mb",
        "quantized_acc",
    ]
    labels = [
        "activation_quant_bits",
        "weight_quant_bits",
        "prune_ratio",
        "compression_ratio",
        "model_size_mb",
        "quantized_acc",
    ]

    data = df[cols].to_numpy(dtype=float)
    mins = data.min(axis=0)
    maxs = data.max(axis=0)
    spans = np.where((maxs - mins) < 1e-8, 1.0, maxs - mins)
    normed = (data - mins) / spans

    fig, ax = plt.subplots(figsize=(12, 5.2), dpi=150)
    x = np.arange(len(cols))
    cmap = plt.cm.turbo

    acc_vals = df["quantized_acc"].to_numpy(dtype=float)
    acc_norm = (acc_vals - acc_vals.min()) / max(acc_vals.max() - acc_vals.min(), 1e-6)

    for i in range(len(df)):
        ax.plot(x, normed[i], color=cmap(acc_norm[i]), alpha=0.82, linewidth=2.0)

    for j in range(len(cols)):
        ax.axvline(j, color="gray", linewidth=0.9, alpha=0.6)
        ax.text(j, 1.05, f"{maxs[j]:.2f}", ha="center", va="bottom", fontsize=8)
        ax.text(j, -0.06, f"{mins[j]:.2f}", ha="center", va="top", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticks([])
    ax.set_ylim(-0.12, 1.14)
    ax.set_title("MobileNet-v2 CIFAR-10 Compression Sweep (Parallel Coordinates)")
    for spine in ("top", "right", "left", "bottom"):
        ax.spines[spine].set_visible(False)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(acc_vals.min(), acc_vals.max()))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="quantized_acc (%)", pad=0.02)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def write_markdown_summary(df: pd.DataFrame, out_path: str) -> None:
    """Save a publication-ready Markdown summary table for the PDF report."""
    cols = [
        "run_name",
        "weight_quant_bits",
        "activation_quant_bits",
        "prune_ratio",
        "baseline_acc",
        "ptq_acc",
        "quantized_acc",
        "weight_compression_ratio",
        "activation_compression_ratio",
        "compression_ratio",
        "model_size_mb",
        "storage_overhead_kb",
    ]
    sub = df[cols].copy()
    sub = sub.sort_values(by=["quantized_acc", "compression_ratio"], ascending=[False, False])
    lines = [
        "# MobileNet-v2 CIFAR-10 Compression Sweep Results\n",
        "| Run | W-bits | A-bits | Prune | Baseline Acc (%) | PTQ Acc (%) | Final Acc (%) | Weight CR (x) | Act CR (x) | Model CR (x) | Size (MB) | Overhead (KB) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for _, r in sub.iterrows():
        lines.append(
            f"| {r['run_name']} | {int(r['weight_quant_bits'])} | {int(r['activation_quant_bits'])} | "
            f"{r['prune_ratio']:.2f} | {r['baseline_acc']:.2f} | {r['ptq_acc']:.2f} | "
            f"**{r['quantized_acc']:.2f}** | {r['weight_compression_ratio']:.2f}x | "
            f"{r['activation_compression_ratio']:.2f}x | **{r['compression_ratio']:.2f}x** | "
            f"{r['model_size_mb']:.3f} | {r['storage_overhead_kb']:.1f} |"
        )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MobileNet-v2 compression sweep with WandB logging")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/baseline_best.pth")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--quick", action="store_true", help="Run 8-config subset instead of 16")
    parser.add_argument("--resume", action="store_true", help="Skip runs already in sweep_results.csv")
    parser.add_argument("--epoch_scale", type=float, default=1.0, help="Multiply QAT/FT epochs (e.g., 0.5 for faster runs)")
    parser.add_argument("--no_protect_first_last", action="store_true")
    parser.add_argument("--kd_alpha", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--wandb", action="store_true", help="Log every sweep run to Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="mobilenetv2-cifar10-compression")
    parser.add_argument("--wandb_group", type=str, default="compression-sweep")
    parser.add_argument("--no_tqdm", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    ensure_dir(args.results_dir)
    ensure_dir(os.path.join(args.results_dir, "runs"))

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(
            f"Baseline checkpoint not found at '{args.checkpoint}'. Train baseline first."
        )

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_cfg = ckpt.get("model_config", {})
    baseline_model = build_mobilenetv2_cifar10(**model_cfg).to(device)
    baseline_model.load_state_dict(ckpt["model_state_dict"])

    train_loader, test_loader, calib_loader = get_cifar10_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    baseline_acc = float(evaluate_model(baseline_model, test_loader, device)["top1_acc"])
    print(f"[Sweep] Loaded baseline ({args.checkpoint}) | FP32 Top-1 Acc: {baseline_acc:.2f}%")

    csv_path = os.path.join(args.results_dir, "sweep_results.csv")
    rows: List[Dict[str, object]] = []
    completed_names = set()

    if args.resume and os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        rows = existing_df.to_dict(orient="records")
        completed_names = set(existing_df["run_name"].tolist())
        print(f"[Sweep] Resuming: found {len(completed_names)} completed runs in {csv_path}")

    configs = QUICK_SWEEP_CONFIGS if args.quick else FULL_SWEEP_CONFIGS

    for idx, cfg in enumerate(configs, start=1):
        run_name = cfg["name"]
        if run_name in completed_names:
            print(f"[Sweep {idx:02d}/{len(configs):02d}] Skipping completed run: {run_name}")
            continue

        print(f"\n[Sweep {idx:02d}/{len(configs):02d}] Starting {run_name} ...")
        set_seed(args.seed)

        qat_ep = max(1, int(round(cfg["qat_ep"] * args.epoch_scale)))
        p_ep = int(round(cfg["p_ep"] * args.epoch_scale)) if cfg["prune"] > 0 else 0

        _, report = run_compression_pipeline(
            baseline_model=baseline_model,
            train_loader=train_loader,
            test_loader=test_loader,
            calib_loader=calib_loader,
            device=device,
            weight_quant_bits=int(cfg["w_bits"]),
            activation_quant_bits=int(cfg["a_bits"]),
            prune_ratio=float(cfg["prune"]),
            prune_head=True,
            protect_first_last=not args.no_protect_first_last,
            prune_ft_epochs=p_ep,
            qat_epochs=qat_ep,
            kd_alpha=args.kd_alpha,
            baseline_acc_cached=baseline_acc,
            verify_packing=True,
            disable_tqdm=args.no_tqdm,
        )

        save_json(report, os.path.join(args.results_dir, "runs", f"{run_name}.json"))

        comp = report["compression"]
        acc = report["accuracy"]
        ov = comp["overhead_breakdown"]

        row = {
            "run_name": run_name,
            "weight_quant_bits": int(cfg["w_bits"]),
            "activation_quant_bits": int(cfg["a_bits"]),
            "prune_ratio": float(cfg["prune"]),
            "protect_first_last": not args.no_protect_first_last,
            "baseline_acc": acc["baseline_fp32_acc"],
            "ptq_acc": acc["ptq_acc"],
            "quantized_acc": acc["quantized_acc"],
            "acc_drop": acc["acc_drop_vs_baseline"],
            "compression_ratio": round(float(comp["model_compression_ratio"]), 4),
            "weight_compression_ratio": round(float(comp["weight_compression_ratio"]), 4),
            "activation_compression_ratio": round(float(comp["activation_compression_ratio"]), 4),
            "activation_peak_compression_ratio": round(float(comp["activation_peak_compression_ratio"]), 4),
            "runtime_combined_cr": round(float(comp["runtime_combined_compression_ratio"]), 4),
            "baseline_size_mb": round(float(comp["baseline_model_mb"]), 4),
            "model_size_mb": round(float(comp["compressed_model_mb"]), 4),
            "storage_overhead_kb": round(float(comp["storage_overhead_kb"]), 3),
            "scale_overhead_kb": round(ov["scale_overhead_bytes"] / 1024.0, 3),
            "bias_overhead_kb": round(ov["bias_overhead_bytes"] / 1024.0, 3),
            "act_meta_kb": round(ov["act_meta_bytes"] / 1024.0, 3),
            "header_meta_kb": round(ov["header_meta_bytes"] / 1024.0, 3),
            "packing_exact_match": bool(comp["packing_verification"]["matches_analytical_bytes"]),
            "elapsed_sec": report["elapsed_seconds"],
        }
        rows.append(row)

        # Persist incrementally after each run
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)

        # Log to Weights & Biases (each config is a separate WandB run for Parallel Coordinates)
        if args.wandb:
            import wandb

            wb_run = wandb.init(
                project=args.wandb_project,
                group=args.wandb_group,
                name=run_name,
                reinit=True,
                config={
                    "weight_quant_bits": row["weight_quant_bits"],
                    "activation_quant_bits": row["activation_quant_bits"],
                    "prune_ratio": row["prune_ratio"],
                    "protect_first_last": row["protect_first_last"],
                    "qat_epochs": qat_ep,
                    "prune_ft_epochs": p_ep,
                    "seed": args.seed,
                },
            )
            wb_run.log(
                {
                    "activation_quant_bits": row["activation_quant_bits"],
                    "weight_quant_bits": row["weight_quant_bits"],
                    "prune_ratio": row["prune_ratio"],
                    "compression_ratio": row["compression_ratio"],
                    "weight_compression_ratio": row["weight_compression_ratio"],
                    "activation_compression_ratio": row["activation_compression_ratio"],
                    "model_size_mb": row["model_size_mb"],
                    "storage_overhead_kb": row["storage_overhead_kb"],
                    "quantized_acc": row["quantized_acc"],
                    "ptq_acc": row["ptq_acc"],
                    "baseline_acc": row["baseline_acc"],
                }
            )
            wb_run.finish()

        print(
            f"  --> {run_name} DONE | Acc: {row['quantized_acc']:.2f}% | "
            f"Model CR: {row['compression_ratio']:.2f}x | Weight CR: {row['weight_compression_ratio']:.2f}x | "
            f"Act CR: {row['activation_compression_ratio']:.2f}x | Size: {row['model_size_mb']:.3f} MB"
        )

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    write_markdown_summary(df, os.path.join(args.results_dir, "sweep_summary.md"))
    plot_pareto_frontier(df, os.path.join(args.results_dir, "pareto_frontier.png"))
    plot_local_parallel_coordinates(df, os.path.join(args.results_dir, "parallel_coordinates_local.png"))

    print(f"\n[Sweep Complete] Results saved to {csv_path}")
    print(f"  - Markdown summary : {os.path.join(args.results_dir, 'sweep_summary.md')}")
    print(f"  - Pareto plot      : {os.path.join(args.results_dir, 'pareto_frontier.png')}")
    print(f"  - Local PC plot    : {os.path.join(args.results_dir, 'parallel_coordinates_local.png')}")


if __name__ == "__main__":
    main()
