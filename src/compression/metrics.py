"""Exact in-memory RAM/VRAM size, storage overhead, and activation compression metrics.

Strictly evaluates IN-MEMORY model footprint (RAM/VRAM) — NOT disk file size:
  - Weight payload bytes: exact bit-packed size `ceil(numel * weight_bits / 8)` per layer.
  - Storage overheads (Q2c):
      * Per-channel weight scales (FP16 = 2 bytes per output channel)
      * Layer biases (FP16 = 2 bytes per bias element)
      * Activation scales & zero-points (4 bytes per quantized activation tensor)
      * Layer metadata headers (16 bytes per quantized layer: bit-width, shape, qmin)
  - Activation memory (Q4b):
      * Measured via PyTorch forward hooks across every Conv/Linear layer for a
        single CIFAR-10 sample `(1, 3, 32, 32)`.
      * Reports BOTH total intermediate activation memory across all layers AND
        peak concurrent working-set activation memory during forward execution.
"""

import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from src.compression.packing import PackedCompressedModel
from src.compression.quantize import QuantConv2d, QuantLinear

BYTES_PER_MB = 1024.0 * 1024.0


def compute_baseline_model_memory(model: nn.Module) -> Dict[str, float]:
    """Compute exact FP32 in-memory size of the uncompressed baseline model."""
    weight_params = 0
    bias_params = 0
    bn_params = 0
    bn_buffer_elems = 0

    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            weight_params += m.weight.numel()
            if m.bias is not None:
                bias_params += m.bias.numel()
        elif isinstance(m, nn.BatchNorm2d):
            if m.weight is not None:
                bn_params += m.weight.numel()
            if m.bias is not None:
                bn_params += m.bias.numel()
            if m.running_mean is not None:
                bn_buffer_elems += m.running_mean.numel()
            if m.running_var is not None:
                bn_buffer_elems += m.running_var.numel()

    weight_bytes = weight_params * 4
    bias_bytes = bias_params * 4
    bn_param_bytes = bn_params * 4
    bn_buffer_bytes = bn_buffer_elems * 4
    total_bytes = weight_bytes + bias_bytes + bn_param_bytes + bn_buffer_bytes

    return {
        "weight_params": weight_params,
        "bias_params": bias_params,
        "bn_params": bn_params,
        "bn_buffer_elems": bn_buffer_elems,
        "total_trainable_params": weight_params + bias_params + bn_params,
        "weight_bytes": weight_bytes,
        "bias_bytes": bias_bytes,
        "bn_bytes": bn_param_bytes + bn_buffer_bytes,
        "total_bytes": total_bytes,
        "total_mb": total_bytes / BYTES_PER_MB,
    }


def compute_compressed_model_memory(
    model: nn.Module,
    scale_dtype_bytes: int = 2,
    bias_dtype_bytes: int = 2,
    header_bytes_per_layer: int = 16,
) -> Dict[str, object]:
    """Compute exact in-memory size and overhead breakdown of a compressed model."""
    weight_payload_bytes = 0
    scale_overhead_bytes = 0
    bias_overhead_bytes = 0
    act_meta_bytes = 0
    header_meta_bytes = 0
    total_weight_params = 0
    layer_breakdown: List[Dict[str, object]] = []

    for name, m in model.named_modules():
        if isinstance(m, (QuantConv2d, QuantLinear)):
            w_bits = m.weight_quant.num_bits
            a_bits = m.act_quant.num_bits
            n_w = m.weight.numel()
            out_ch = m.weight.shape[0]
            total_weight_params += n_w

            # Exact bit-packed payload bytes in uint8 memory
            w_bytes = math.ceil(n_w * w_bits / 8)
            s_bytes = (out_ch * scale_dtype_bytes) if w_bits < 32 else 0
            b_bytes = (
                m.bias.numel() * (bias_dtype_bytes if w_bits < 32 else 4)
                if m.bias is not None
                else 0
            )
            a_bytes = (scale_dtype_bytes + 2) if a_bits < 32 else 0
            h_bytes = header_bytes_per_layer

            weight_payload_bytes += w_bytes
            scale_overhead_bytes += s_bytes
            bias_overhead_bytes += b_bytes
            act_meta_bytes += a_bytes
            header_meta_bytes += h_bytes

            layer_breakdown.append(
                {
                    "layer": name,
                    "params": n_w,
                    "w_bits": w_bits,
                    "a_bits": a_bits,
                    "weight_payload_bytes": w_bytes,
                    "scale_bytes": s_bytes,
                    "bias_bytes": b_bytes,
                    "act_meta_bytes": a_bytes,
                    "header_bytes": h_bytes,
                }
            )
        elif isinstance(m, (nn.Conv2d, nn.Linear)):
            # Unwrapped FP32 layer fallback
            n_w = m.weight.numel()
            total_weight_params += n_w
            w_bytes = n_w * 4
            b_bytes = m.bias.numel() * 4 if m.bias is not None else 0
            weight_payload_bytes += w_bytes
            bias_overhead_bytes += b_bytes
        elif isinstance(m, nn.BatchNorm2d):
            # Only present if BN was not folded
            bn_b = 0
            for t in (m.weight, m.bias, m.running_mean, m.running_var):
                if t is not None:
                    bn_b += t.numel() * 4
            bias_overhead_bytes += bn_b

    total_overhead_bytes = (
        scale_overhead_bytes + bias_overhead_bytes + act_meta_bytes + header_meta_bytes
    )
    total_bytes = weight_payload_bytes + total_overhead_bytes

    return {
        "total_weight_params": total_weight_params,
        "weight_payload_bytes": weight_payload_bytes,
        "scale_overhead_bytes": scale_overhead_bytes,
        "bias_overhead_bytes": bias_overhead_bytes,
        "act_meta_bytes": act_meta_bytes,
        "header_meta_bytes": header_meta_bytes,
        "total_overhead_bytes": total_overhead_bytes,
        "total_overhead_kb": total_overhead_bytes / 1024.0,
        "overhead_pct_of_compressed": 100.0 * total_overhead_bytes / max(total_bytes, 1),
        "total_bytes": total_bytes,
        "total_mb": total_bytes / BYTES_PER_MB,
        "layer_breakdown": layer_breakdown,
    }


@torch.no_grad()
def measure_activation_memory(
    model: nn.Module,
    input_shape: Tuple[int, int, int, int] = (1, 3, 32, 32),
    device: torch.device = torch.device("cpu"),
) -> Dict[str, object]:
    """Measure in-memory activation footprint via forward hooks on a single CIFAR-10 input.

    Methodology (for Q4b):
      - Registers forward hooks on every Conv2d / QuantConv2d / Linear / QuantLinear layer.
      - Runs one forward pass with `torch.randn(input_shape)` (batch size = 1, 3x32x32).
      - For each layer, records the exact input activation tensor element count `numel`
        and its in-memory bit-width (`act_quant.num_bits` for quantized layers, 32 for FP32).
      - Computes:
          1. `total_activation_bytes`: sum of bit-packed input activation bytes across all
             layers + output logits + activation scale/zero-point metadata.
          2. `peak_activation_bytes`: maximum concurrent (input + output) activation bytes
             live in memory during any single layer execution.
    """
    model.eval()
    records: List[Dict[str, object]] = []
    hooks = []

    def _make_hook(layer_name: str, layer_mod: nn.Module):
        def _hook(_mod, inputs, output):
            inp = inputs[0]
            in_elems = int(inp.numel())
            out_elems = int(output.numel())
            if isinstance(layer_mod, (QuantConv2d, QuantLinear)):
                a_bits = int(layer_mod.act_quant.num_bits)
                meta_b = 4 if a_bits < 32 else 0  # FP16 scale + int16 zero-point
            else:
                a_bits = 32
                meta_b = 0

            in_bytes = math.ceil(in_elems * a_bits / 8)
            out_bytes = math.ceil(out_elems * a_bits / 8)
            records.append(
                {
                    "layer": layer_name,
                    "input_shape": list(inp.shape),
                    "output_shape": list(output.shape),
                    "in_elems": in_elems,
                    "out_elems": out_elems,
                    "act_bits": a_bits,
                    "in_bytes": in_bytes,
                    "out_bytes": out_bytes,
                    "working_set_bytes": in_bytes + out_bytes + meta_b,
                    "meta_bytes": meta_b,
                }
            )

        return _hook

    for name, m in model.named_modules():
        if isinstance(m, (QuantConv2d, QuantLinear, nn.Conv2d, nn.Linear)):
            hooks.append(m.register_forward_hook(_make_hook(name, m)))

    try:
        dummy = torch.randn(input_shape, device=device)
        _ = model(dummy)
    finally:
        for h in hooks:
            h.remove()

    total_act_payload_bytes = sum(r["in_bytes"] for r in records)
    # Include final classifier output logits
    if records:
        total_act_payload_bytes += records[-1]["out_bytes"]
    total_act_meta_bytes = sum(r["meta_bytes"] for r in records)
    total_act_bytes = total_act_payload_bytes + total_act_meta_bytes
    peak_act_bytes = max((r["working_set_bytes"] for r in records), default=0)
    total_act_elems = sum(r["in_elems"] for r in records) + (
        records[-1]["out_elems"] if records else 0
    )

    return {
        "input_shape": list(input_shape),
        "num_measured_layers": len(records),
        "total_activation_elements": total_act_elems,
        "total_activation_payload_bytes": total_act_payload_bytes,
        "total_activation_meta_bytes": total_act_meta_bytes,
        "total_activation_bytes": total_act_bytes,
        "total_activation_mb": total_act_bytes / BYTES_PER_MB,
        "peak_activation_bytes": peak_act_bytes,
        "peak_activation_mb": peak_act_bytes / BYTES_PER_MB,
        "layer_records": records,
    }


def compute_full_compression_report(
    baseline_model: nn.Module,
    compressed_model: nn.Module,
    device: torch.device = torch.device("cpu"),
    verify_packing: bool = True,
) -> Dict[str, object]:
    """Compute complete baseline vs. compressed in-memory metrics for Q2c & Q4."""
    base_mem = compute_baseline_model_memory(baseline_model)
    comp_mem = compute_compressed_model_memory(compressed_model)

    base_act = measure_activation_memory(baseline_model, device=device)
    comp_act = measure_activation_memory(compressed_model, device=device)

    # Weight compression ratio: baseline weights vs. compressed weights + weight overheads
    comp_weight_with_overhead = (
        comp_mem["weight_payload_bytes"]
        + comp_mem["scale_overhead_bytes"]
        + comp_mem["bias_overhead_bytes"]
        + comp_mem["header_meta_bytes"]
    )
    weight_cr = (base_mem["weight_bytes"] + base_mem["bias_bytes"] + base_mem["bn_bytes"]) / max(
        comp_weight_with_overhead, 1
    )

    # Activation compression ratios (total across all layers & peak working-set)
    act_total_cr = base_act["total_activation_bytes"] / max(comp_act["total_activation_bytes"], 1)
    act_peak_cr = base_act["peak_activation_bytes"] / max(comp_act["peak_activation_bytes"], 1)

    # Overall model compression ratio (entire model loaded in memory: weights + all overheads)
    model_cr = base_mem["total_bytes"] / max(comp_mem["total_bytes"], 1)

    # Combined runtime RAM footprint (model weights/overheads + total activations for batch=1)
    base_runtime_bytes = base_mem["total_bytes"] + base_act["total_activation_bytes"]
    comp_runtime_bytes = comp_mem["total_bytes"] + comp_act["total_activation_bytes"]
    runtime_cr = base_runtime_bytes / max(comp_runtime_bytes, 1)

    packing_verification = None
    if verify_packing:
        packed = PackedCompressedModel.from_quantized_model(compressed_model)
        live_bytes = packed.actual_tensor_bytes()
        roundtrip_err = packed.verify_roundtrip(compressed_model)
        packing_verification = {
            "live_packed_total_bytes": live_bytes["total_in_memory_bytes"],
            "live_packed_total_mb": live_bytes["total_in_memory_bytes"] / BYTES_PER_MB,
            "matches_analytical_bytes": live_bytes["total_in_memory_bytes"] == comp_mem["total_bytes"],
            "max_roundtrip_dequant_error": roundtrip_err,
        }

    return {
        "baseline_model_mb": base_mem["total_mb"],
        "baseline_model_bytes": base_mem["total_bytes"],
        "baseline_params": base_mem["total_trainable_params"],
        "compressed_model_mb": comp_mem["total_mb"],
        "compressed_model_bytes": comp_mem["total_bytes"],
        "compressed_weight_params": comp_mem["total_weight_params"],
        "weight_payload_bytes": comp_mem["weight_payload_bytes"],
        "storage_overhead_bytes": comp_mem["total_overhead_bytes"],
        "storage_overhead_kb": comp_mem["total_overhead_kb"],
        "overhead_breakdown": {
            "scale_overhead_bytes": comp_mem["scale_overhead_bytes"],
            "bias_overhead_bytes": comp_mem["bias_overhead_bytes"],
            "act_meta_bytes": comp_mem["act_meta_bytes"],
            "header_meta_bytes": comp_mem["header_meta_bytes"],
        },
        "weight_compression_ratio": weight_cr,
        "activation_compression_ratio": act_total_cr,
        "activation_peak_compression_ratio": act_peak_cr,
        "model_compression_ratio": model_cr,
        "runtime_combined_compression_ratio": runtime_cr,
        "baseline_activation_total_mb": base_act["total_activation_mb"],
        "compressed_activation_total_mb": comp_act["total_activation_mb"],
        "baseline_activation_peak_mb": base_act["peak_activation_mb"],
        "compressed_activation_peak_mb": comp_act["peak_activation_mb"],
        "packing_verification": packing_verification,
    }
