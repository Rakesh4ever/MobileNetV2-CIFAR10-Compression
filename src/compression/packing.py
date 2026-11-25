"""Custom pure-PyTorch sub-byte bit-packing & in-memory packed model verification.

Packs arbitrary b-bit integer tensors (b in {1..8}) tightly into 1D `torch.uint8`
buffers so the actual in-memory RAM/VRAM footprint of quantized weights matches
`ceil(N * b / 8)` bytes with zero external compression libraries.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn


def pack_int_tensor(q: torch.Tensor, num_bits: int, qmin: int) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    """Pack an integer tensor with values in [qmin, qmin + 2^num_bits - 1] into uint8 bytes.

    Args:
        q: Integer tensor (any shape).
        num_bits: Bit-width per element (1..8).
        qmin: Minimum representable signed integer value (offset shifted to 0).

    Returns:
        (packed_uint8_1d, original_shape)
    """
    if not (1 <= num_bits <= 8):
        raise ValueError(f"pack_int_tensor supports 1..8 bits, got {num_bits}")

    orig_shape = tuple(q.shape)
    u = (q.reshape(-1).to(torch.int64) - int(qmin))
    max_val = (1 << num_bits) - 1
    u = torch.clamp(u, 0, max_val)

    if num_bits == 8:
        return u.to(torch.uint8), orig_shape

    if num_bits == 4:
        n = u.numel()
        if n % 2 != 0:
            u = torch.cat([u, torch.zeros(1, dtype=torch.int64, device=u.device)])
        u = u.view(-1, 2)
        packed = (u[:, 0] | (u[:, 1] << 4)).to(torch.uint8)
        return packed, orig_shape

    if num_bits == 2:
        n = u.numel()
        rem = (-n) % 4
        if rem:
            u = torch.cat([u, torch.zeros(rem, dtype=torch.int64, device=u.device)])
        u = u.view(-1, 4)
        packed = (u[:, 0] | (u[:, 1] << 2) | (u[:, 2] << 4) | (u[:, 3] << 6)).to(torch.uint8)
        return packed, orig_shape

    # Vectorized general bit-stream packer for arbitrary num_bits (e.g., 1, 3, 5, 6, 7)
    bit_positions = torch.arange(num_bits, dtype=torch.int64, device=u.device)
    bits = ((u.unsqueeze(1) >> bit_positions) & 1).to(torch.uint8).reshape(-1)
    pad = (-bits.numel()) % 8
    if pad:
        bits = torch.cat([bits, torch.zeros(pad, dtype=torch.uint8, device=u.device)])
    bits = bits.view(-1, 8)
    byte_weights = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8, device=u.device)
    packed = (bits * byte_weights).sum(dim=1).to(torch.uint8)
    return packed, orig_shape


def unpack_int_tensor(
    packed: torch.Tensor,
    orig_shape: Tuple[int, ...],
    num_bits: int,
    qmin: int,
) -> torch.Tensor:
    """Losslessly unpack a `uint8` bit-packed tensor back to signed int32 values."""
    if not (1 <= num_bits <= 8):
        raise ValueError(f"unpack_int_tensor supports 1..8 bits, got {num_bits}")

    numel = 1
    for d in orig_shape:
        numel *= d

    if num_bits == 8:
        u = packed.to(torch.int32)[:numel]
        return (u + int(qmin)).view(orig_shape)

    if num_bits == 4:
        p = packed.to(torch.int32)
        lo = p & 0x0F
        hi = (p >> 4) & 0x0F
        u = torch.stack([lo, hi], dim=1).reshape(-1)[:numel]
        return (u + int(qmin)).view(orig_shape)

    if num_bits == 2:
        p = packed.to(torch.int32)
        b0 = p & 0x03
        b1 = (p >> 2) & 0x03
        b2 = (p >> 4) & 0x03
        b3 = (p >> 6) & 0x03
        u = torch.stack([b0, b1, b2, b3], dim=1).reshape(-1)[:numel]
        return (u + int(qmin)).view(orig_shape)

    # Vectorized general bit-stream unpacker
    bit_shifts = torch.arange(8, dtype=torch.int64, device=packed.device)
    bits = ((packed.to(torch.int64).unsqueeze(1) >> bit_shifts) & 1).reshape(-1)
    bits = bits[: numel * num_bits].view(numel, num_bits)
    powers = (1 << torch.arange(num_bits, dtype=torch.int64, device=packed.device))
    u = (bits * powers).sum(dim=1).to(torch.int32)
    return (u + int(qmin)).view(orig_shape)


class PackedCompressedModel:
    """True in-memory bit-packed weight representation for RAM/VRAM verification.

    Stores every quantized layer's weights strictly as bit-packed `uint8` tensors
    plus compact FP16 scales/biases and layer metadata. Allows measuring exact live
    PyTorch tensor bytes (`numel() * element_size()`) and verifying lossless round-trip
    dequantization against the QAT/PTQ model.
    """

    def __init__(self) -> None:
        self.layers: Dict[str, Dict[str, object]] = {}

    @classmethod
    @torch.no_grad()
    def from_quantized_model(cls, model: nn.Module) -> "PackedCompressedModel":
        from .quantize import QuantConv2d, QuantLinear

        packed_model = cls()
        for name, module in model.named_modules():
            if isinstance(module, (QuantConv2d, QuantLinear)):
                w_bits = module.weight_quant.num_bits
                if w_bits < 32:
                    q_int, scale = module.weight_quant.export_quantized(module.weight)
                    qmin = module.weight_quant.qmin
                    packed_w, orig_shape = pack_int_tensor(q_int, w_bits, qmin)
                    entry = {
                        "packed_weight": packed_w.contiguous(),
                        "orig_shape": orig_shape,
                        "w_bits": w_bits,
                        "qmin": qmin,
                        "scale": scale.to(torch.float16).contiguous(),
                        "bias": module.bias.detach().to(torch.float16).contiguous()
                        if module.bias is not None
                        else None,
                    }
                else:
                    entry = {
                        "packed_weight": module.weight.detach().to(torch.float32).contiguous(),
                        "orig_shape": tuple(module.weight.shape),
                        "w_bits": 32,
                        "qmin": 0,
                        "scale": None,
                        "bias": module.bias.detach().to(torch.float32).contiguous()
                        if module.bias is not None
                        else None,
                    }
                if module.act_quant.num_bits < 32:
                    a_scale, a_zp = module.act_quant.export_params()
                    entry["act_scale"] = a_scale.to(torch.float16)
                    entry["act_zp"] = a_zp.to(torch.int16)
                    entry["a_bits"] = module.act_quant.num_bits
                else:
                    entry["a_bits"] = 32
                packed_model.layers[name] = entry
        return packed_model

    def actual_tensor_bytes(self) -> Dict[str, int]:
        """Measure actual live PyTorch tensor bytes held in RAM/VRAM."""
        weight_payload_bytes = 0
        scale_bytes = 0
        bias_bytes = 0
        act_meta_bytes = 0
        header_meta_bytes = 0

        for entry in self.layers.values():
            w_tensor = entry["packed_weight"]
            weight_payload_bytes += w_tensor.numel() * w_tensor.element_size()
            if entry["scale"] is not None:
                s = entry["scale"]
                scale_bytes += s.numel() * s.element_size()
            if entry["bias"] is not None:
                b = entry["bias"]
                bias_bytes += b.numel() * b.element_size()
            if "act_scale" in entry:
                act_meta_bytes += (
                    entry["act_scale"].numel() * entry["act_scale"].element_size()
                    + entry["act_zp"].numel() * entry["act_zp"].element_size()
                )
            # Fixed 16-byte header per layer (shape dims, bit-width, qmin)
            header_meta_bytes += 16

        total = weight_payload_bytes + scale_bytes + bias_bytes + act_meta_bytes + header_meta_bytes
        return {
            "weight_payload_bytes": weight_payload_bytes,
            "scale_overhead_bytes": scale_bytes,
            "bias_overhead_bytes": bias_bytes,
            "act_meta_bytes": act_meta_bytes,
            "header_meta_bytes": header_meta_bytes,
            "total_in_memory_bytes": total,
        }

    @torch.no_grad()
    def verify_roundtrip(self, model: nn.Module) -> float:
        """Unpack every layer and return max absolute dequantization discrepancy."""
        from .quantize import QuantConv2d, QuantLinear

        max_err = 0.0
        for name, module in model.named_modules():
            if name not in self.layers or not isinstance(module, (QuantConv2d, QuantLinear)):
                continue
            entry = self.layers[name]
            if entry["w_bits"] >= 32:
                continue
            unpacked_q = unpack_int_tensor(
                entry["packed_weight"], entry["orig_shape"], entry["w_bits"], entry["qmin"]
            ).to(dtype=module.weight.dtype, device=module.weight.device)
            scale_fp16 = entry["scale"].to(dtype=module.weight.dtype, device=module.weight.device)
            view_shape = [-1] + [1] * (unpacked_q.ndim - 1)
            reconstructed = unpacked_q * scale_fp16.view(view_shape)

            # Compare against quantized weight using FP16-rounded scale
            q_ref, s_ref = module.weight_quant.export_quantized(module.weight)
            ref = q_ref.to(module.weight.dtype) * s_ref.to(torch.float16).to(module.weight.dtype).view(view_shape)
            err = (reconstructed - ref).abs().max().item()
            max_err = max(max_err, err)
        return max_err
