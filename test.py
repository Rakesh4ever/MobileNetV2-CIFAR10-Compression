"""CLI entrypoint for compressing and evaluating MobileNet-v2 on CIFAR-10.

Examples:
    python test.py --weight_quant_bits 8 --activation_quant_bits 8
    python test.py --weight_quant_bits 4 --activation_quant_bits 4 --prune_ratio 0.40 --qat_epochs 25
"""

import argparse
import os

import torch

from src.compress_pipeline import run_compression_pipeline
from src.data import get_cifar10_loaders
from src.model import build_mobilenetv2_cifar10
from src.utils import ensure_dir, get_device, save_json, set_seed


def print_report_summary(report: dict) -> None:
    """Print a structured summary of accuracy, memory footprint, and compression ratios."""
    cfg = report["config"]
    acc = report["accuracy"]
    comp = report["compression"]
    ov = comp["overhead_breakdown"]
    pack = comp.get("packing_verification")

    print("\n" + "=" * 74)
    print(" MobileNetV2-CIFAR10-Compression — Custom In-Memory Compression Report")
    print("=" * 74)
    print(
        f" Config: W={cfg['weight_quant_bits']}b | A={cfg['activation_quant_bits']}b | "
        f"Prune={cfg['prune_ratio']:.2f} | ProtectFirstLast={cfg['protect_first_last']} | "
        f"QAT={cfg['qat_epochs']}ep"
    )
    print("-" * 74)
    print(f" Baseline FP32 Test Top-1 Accuracy      : {acc['baseline_fp32_acc']:.2f}%")
    if cfg["prune_ratio"] > 0:
        print(f" Post-Pruning FP32 Accuracy             : {acc['pruned_fp32_acc']:.2f}%")
    print(f" Post-Calibration PTQ Accuracy          : {acc['ptq_acc']:.2f}%")
    print(f" Final Compressed Test Top-1 Accuracy   : {acc['quantized_acc']:.2f}% (drop: {acc['acc_drop_vs_baseline']:+.2f}%)")
    print("-" * 74)
    print(f" Baseline In-Memory Model Size          : {comp['baseline_model_mb']:.4f} MB ({comp['baseline_model_bytes']:,} bytes)")
    print(f" Final Approximated Model Size (RAM)    : {comp['compressed_model_mb']:.4f} MB ({comp['compressed_model_bytes']:,} bytes)")
    print(f"   - Packed Weight Payload              : {comp['weight_payload_bytes']/1024.0:.2f} KB")
    print(f"   - Storage Overheads Total            : {comp['storage_overhead_kb']:.2f} KB")
    print(f"       * Per-channel Weight Scales      : {ov['scale_overhead_bytes']/1024.0:.2f} KB")
    print(f"       * Biases (FP16)                  : {ov['bias_overhead_bytes']/1024.0:.2f} KB")
    print(f"       * Activation Scales & ZPs        : {ov['act_meta_bytes']/1024.0:.2f} KB")
    print(f"       * Layer Metadata Headers         : {ov['header_meta_bytes']/1024.0:.2f} KB")
    print("-" * 74)
    print(f" Model Compression Ratio (In-Memory)    : {comp['model_compression_ratio']:.2f}x")
    print(f" Weight Compression Ratio (In-Memory)   : {comp['weight_compression_ratio']:.2f}x")
    print(f" Activation Compression Ratio (Total)   : {comp['activation_compression_ratio']:.2f}x "
          f"({comp['baseline_activation_total_mb']:.3f} MB -> {comp['compressed_activation_total_mb']:.3f} MB)")
    print(f" Activation Compression Ratio (Peak)    : {comp['activation_peak_compression_ratio']:.2f}x "
          f"({comp['baseline_activation_peak_mb']:.3f} MB -> {comp['compressed_activation_peak_mb']:.3f} MB)")
    print(f" Combined Runtime RAM CR (Weights+Acts) : {comp['runtime_combined_compression_ratio']:.2f}x")
    if pack is not None:
        print("-" * 74)
        print(
            f" Live Bit-Packed Tensor Verification    : {pack['live_packed_total_mb']:.4f} MB | "
            f"Exact Byte Match={pack['matches_analytical_bytes']} | "
            f"Max Roundtrip Err={pack['max_roundtrip_dequant_error']:.2e}"
        )
    print("=" * 74 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compress and evaluate MobileNet-v2 on CIFAR-10 (custom in-memory pipeline)"
    )
    parser.add_argument("--weight_quant_bits", type=int, default=8, help="Weight bit-width (2..8 or 32)")
    parser.add_argument("--activation_quant_bits", type=int, default=8, help="Activation bit-width (2..8 or 32)")
    parser.add_argument("--prune_ratio", type=float, default=0.0, help="Structured expansion pruning ratio [0.0, 0.85)")
    parser.add_argument("--no_prune_head", action="store_true", help="Do not prune 1280-ch head_conv")
    parser.add_argument("--no_protect_first_last", action="store_true", help="Quantize stem & classifier uniformly")
    parser.add_argument("--prune_ft_epochs", type=int, default=12, help="Recovery epochs after structured pruning")
    parser.add_argument("--qat_epochs", type=int, default=0, help="QAT fine-tuning epochs (0 for fast calibrated PTQ; e.g. 20-30 for low-bit QAT)")
    parser.add_argument("--prune_lr", type=float, default=0.015)
    parser.add_argument("--qat_lr", type=float, default=1e-3)
    parser.add_argument("--kd_alpha", type=float, default=0.5, help="Teacher distillation weight (0 disables KD)")
    parser.add_argument("--kd_temp", type=float, default=4.0)
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/baseline_best.pth")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save_json", type=str, default=None, help="Optional JSON report output path")
    parser.add_argument("--save_model", type=str, default=None, help="Optional compressed checkpoint save path")
    parser.add_argument("--no_tqdm", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(
            f"Baseline checkpoint not found at '{args.checkpoint}'. "
            f"Train the baseline first with: python -m src.train --epochs 200"
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

    quant_model, report = run_compression_pipeline(
        baseline_model=baseline_model,
        train_loader=train_loader,
        test_loader=test_loader,
        calib_loader=calib_loader,
        device=device,
        weight_quant_bits=args.weight_quant_bits,
        activation_quant_bits=args.activation_quant_bits,
        prune_ratio=args.prune_ratio,
        prune_head=not args.no_prune_head,
        protect_first_last=not args.no_protect_first_last,
        prune_ft_epochs=args.prune_ft_epochs,
        qat_epochs=args.qat_epochs,
        prune_lr=args.prune_lr,
        qat_lr=args.qat_lr,
        kd_alpha=args.kd_alpha,
        kd_temp=args.kd_temp,
        verify_packing=True,
        disable_tqdm=args.no_tqdm,
    )

    print_report_summary(report)

    if args.save_json:
        save_json(report, args.save_json)
        print(f"Saved detailed JSON report to {args.save_json}")

    if args.save_model:
        ensure_dir(os.path.dirname(args.save_model) or ".")
        torch.save(
            {"state_dict": quant_model.state_dict(), "report": report},
            args.save_model,
        )
        print(f"Saved compressed model checkpoint to {args.save_model}")


if __name__ == "__main__":
    main()
