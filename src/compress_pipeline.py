"""End-to-end custom compression pipeline: Structured Pruning + BN Folding + QAT/PTQ.

Supports optional self-distillation (KL-divergence from the frozen FP32 baseline
teacher) during pruning recovery and Quantization-Aware Training (QAT), written
100% in pure PyTorch with zero external compression APIs.
"""

import copy
import time
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.compression.metrics import compute_full_compression_report
from src.compression.prune import prune_mobilenetv2_structured
from src.compression.quantize import (
    CustomActivationQuantizer,
    CustomWeightQuantizer,
    calibrate_model,
    quantize_mobilenetv2,
)
from src.evaluate import evaluate_model
from src.model import MobileNetV2CIFAR


def _distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.5,
    temperature: float = 4.0,
) -> torch.Tensor:
    """Combine hard-label CrossEntropy with soft-target KL divergence from FP32 teacher."""
    ce_loss = F.cross_entropy(student_logits, targets)
    if alpha <= 0.0:
        return ce_loss
    kd_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction="batchmean",
    ) * (temperature * temperature)
    return (1.0 - alpha) * ce_loss + alpha * kd_loss


def fine_tune_model(
    student: nn.Module,
    teacher: Optional[nn.Module],
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    optimizer_type: str = "adamw",
    kd_alpha: float = 0.5,
    kd_temp: float = 4.0,
    stage_name: str = "QAT",
    disable_tqdm: bool = False,
) -> Tuple[nn.Module, float]:
    """Fine-tune a pruned or quantized student model with optional teacher KD."""
    if epochs <= 0:
        acc = float(evaluate_model(student, test_loader, device)["top1_acc"])
        return student, acc

    if teacher is not None:
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

    # Separate quantizer scale/range parameters from conv/linear weights
    quant_params = []
    weight_params = []
    no_decay_params = []

    quant_param_ids = set()
    for m in student.modules():
        if isinstance(m, (CustomWeightQuantizer, CustomActivationQuantizer)):
            for p in m.parameters():
                if p.requires_grad:
                    quant_params.append(p)
                    quant_param_ids.add(id(p))

    for name, p in student.named_parameters():
        if not p.requires_grad or id(p) in quant_param_ids:
            continue
        if p.ndim == 1 or name.endswith(".bias") or "bn" in name.lower():
            no_decay_params.append(p)
        else:
            weight_params.append(p)

    param_groups = [
        {"params": weight_params, "lr": lr, "weight_decay": weight_decay},
        {"params": no_decay_params, "lr": lr, "weight_decay": 0.0},
    ]
    if quant_params:
        param_groups.append({"params": quant_params, "lr": lr * 0.5, "weight_decay": 0.0})

    if optimizer_type.lower() == "sgd":
        optimizer = torch.optim.SGD(param_groups, momentum=0.9, nesterov=True)
    else:
        optimizer = torch.optim.AdamW(param_groups)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=lr * 0.01
    )

    best_acc = -1.0
    best_state = None

    for ep in range(1, epochs + 1):
        student.train()
        pbar = tqdm(
            train_loader,
            desc=f"[{stage_name}] Ep {ep:02d}/{epochs:02d}",
            leave=False,
            disable=disable_tqdm,
        )
        for images, targets in pbar:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            student_logits = student(images)

            if teacher is not None and kd_alpha > 0.0:
                with torch.no_grad():
                    teacher_logits = teacher(images)
                loss = _distillation_loss(
                    student_logits, teacher_logits, targets, alpha=kd_alpha, temperature=kd_temp
                )
            else:
                loss = F.cross_entropy(student_logits, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=5.0)
            optimizer.step()

        scheduler.step()
        eval_res = evaluate_model(student, test_loader, device)
        acc = float(eval_res["top1_acc"])
        if acc > best_acc:
            best_acc = acc
            best_state = copy.deepcopy(student.state_dict())

        if ep == 1 or ep % max(1, epochs // 4) == 0 or ep == epochs:
            print(f"  [{stage_name}] Ep {ep:02d}/{epochs:02d} | Test Top-1: {acc:.2f}% (Best: {best_acc:.2f}%)")

    if best_state is not None:
        student.load_state_dict(best_state)
    return student, best_acc


def run_compression_pipeline(
    baseline_model: MobileNetV2CIFAR,
    train_loader: DataLoader,
    test_loader: DataLoader,
    calib_loader: DataLoader,
    device: torch.device,
    weight_quant_bits: int = 8,
    activation_quant_bits: int = 8,
    prune_ratio: float = 0.0,
    prune_head: bool = True,
    protect_first_last: bool = True,
    prune_ft_epochs: int = 12,
    qat_epochs: int = 20,
    prune_lr: float = 0.015,
    qat_lr: float = 1e-3,
    kd_alpha: float = 0.5,
    kd_temp: float = 4.0,
    baseline_acc_cached: Optional[float] = None,
    verify_packing: bool = True,
    disable_tqdm: bool = False,
) -> Tuple[nn.Module, Dict[str, object]]:
    """Run full custom compression pipeline and return `(compressed_model, report)`."""
    t0 = time.time()
    baseline_model = baseline_model.to(device).eval()

    if baseline_acc_cached is None:
        baseline_acc = float(evaluate_model(baseline_model, test_loader, device)["top1_acc"])
    else:
        baseline_acc = float(baseline_acc_cached)

    # Stage 1: Optional structured channel pruning + recovery fine-tuning
    working_model, prune_summary = prune_mobilenetv2_structured(
        baseline_model, prune_ratio=prune_ratio, prune_head=prune_head
    )
    working_model = working_model.to(device)

    pruned_fp32_acc = baseline_acc
    if prune_ratio > 0.0:
        pre_ft_acc = float(evaluate_model(working_model, test_loader, device)["top1_acc"])
        print(
            f"[Stage 1: Structured Pruning r={prune_ratio:.2f}] "
            f"Immediate Acc: {pre_ft_acc:.2f}% -> Fine-tuning {prune_ft_epochs} epochs..."
        )
        working_model, pruned_fp32_acc = fine_tune_model(
            student=working_model,
            teacher=baseline_model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            epochs=prune_ft_epochs,
            lr=prune_lr,
            weight_decay=2e-4,
            optimizer_type="sgd",
            kd_alpha=kd_alpha,
            kd_temp=kd_temp,
            stage_name="Prune-FT",
            disable_tqdm=disable_tqdm,
        )

    # Stage 2: BatchNorm folding + custom weight/activation quantization + calibration
    quant_model, policy_log = quantize_mobilenetv2(
        working_model,
        weight_bits=weight_quant_bits,
        act_bits=activation_quant_bits,
        protect_first_last=protect_first_last,
        fold_bn=True,
    )
    quant_model = quant_model.to(device)
    calibrate_model(quant_model, calib_loader, device)
    ptq_acc = float(evaluate_model(quant_model, test_loader, device)["top1_acc"])
    print(
        f"[Stage 2: Custom Quant W{weight_quant_bits}A{activation_quant_bits}] "
        f"Calibrated PTQ Acc: {ptq_acc:.2f}%"
    )

    # Stage 3: Quantization-Aware Training (QAT) fine-tuning via STE
    final_acc = ptq_acc
    if qat_epochs > 0 and (weight_quant_bits < 32 or activation_quant_bits < 32):
        # Slightly higher LR helps very low bit-widths (2-3 bits) adapt quickly
        eff_qat_lr = qat_lr * (1.5 if min(weight_quant_bits, activation_quant_bits) <= 3 else 1.0)
        print(f"[Stage 3: QAT Fine-Tuning ({qat_epochs} epochs, lr={eff_qat_lr:.1e})]...")
        quant_model, final_acc = fine_tune_model(
            student=quant_model,
            teacher=baseline_model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            epochs=qat_epochs,
            lr=eff_qat_lr,
            weight_decay=1e-4,
            optimizer_type="adamw",
            kd_alpha=kd_alpha,
            kd_temp=kd_temp,
            stage_name=f"QAT-W{weight_quant_bits}A{activation_quant_bits}",
            disable_tqdm=disable_tqdm,
        )

    # Stage 4: Exact in-memory RAM/VRAM metrics & bit-packing round-trip check
    metrics = compute_full_compression_report(
        baseline_model=baseline_model,
        compressed_model=quant_model,
        device=device,
        verify_packing=verify_packing,
    )

    elapsed = time.time() - t0
    report = {
        "config": {
            "weight_quant_bits": weight_quant_bits,
            "activation_quant_bits": activation_quant_bits,
            "prune_ratio": prune_ratio,
            "prune_head": prune_head,
            "protect_first_last": protect_first_last,
            "prune_ft_epochs": prune_ft_epochs if prune_ratio > 0 else 0,
            "qat_epochs": qat_epochs,
            "kd_alpha": kd_alpha,
        },
        "accuracy": {
            "baseline_fp32_acc": round(baseline_acc, 3),
            "pruned_fp32_acc": round(pruned_fp32_acc, 3),
            "ptq_acc": round(ptq_acc, 3),
            "quantized_acc": round(final_acc, 3),
            "acc_drop_vs_baseline": round(baseline_acc - final_acc, 3),
        },
        "compression": metrics,
        "pruning_summary": prune_summary,
        "layer_policy": policy_log,
        "elapsed_seconds": round(elapsed, 2),
    }
    return quant_model, report
