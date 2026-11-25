"""Custom structured channel pruning for MobileNet-v2 (pure PyTorch, no pruning APIs).

Design choices:
  1. MobileNet-v2 inverted residual blocks expand channels by 6x internally
     (pw_expand -> dw_conv -> pw_project). Those expanded hidden channels dominate
     both parameter count and intermediate activation memory.
  2. By pruning hidden expansion channels *structurally* (physically constructing
     smaller dense Conv2d and BatchNorm2d layers), we achieve:
       - Immediate dense in-memory RAM/VRAM reduction for BOTH weights and activations.
       - Zero sparse-mask / index metadata overhead.
       - Full preservation of residual skip-connection input/output dimensions.
  3. Optionally also prunes the 320 -> 1280 head_conv layer together with matching
     input columns of the final Linear classifier (which otherwise holds ~18.4% of
     MobileNet-v2 parameters).
"""

import copy
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from src.model import InvertedResidual, MobileNetV2CIFAR


def _round_channels(count: int, min_channels: int = 8, divisor: int = 4) -> int:
    """Round kept channel count to a hardware-friendly multiple with a safe floor."""
    count = max(min_channels, int(round(count / divisor) * divisor))
    return max(min_channels, count)


def _extract_conv_bn(seq: nn.Sequential) -> Tuple[nn.Conv2d, nn.BatchNorm2d]:
    """Locate the Conv2d and optional BatchNorm2d inside a sequential block."""
    conv = None
    bn = None
    for layer in seq.children():
        if isinstance(layer, nn.Conv2d):
            conv = layer
        elif isinstance(layer, nn.BatchNorm2d):
            bn = layer
    if conv is None:
        raise RuntimeError("Expected a Conv2d inside block sequential.")
    return conv, bn


def _slice_conv2d(
    conv: nn.Conv2d,
    in_indices: torch.Tensor = None,
    out_indices: torch.Tensor = None,
    depthwise: bool = False,
) -> nn.Conv2d:
    """Physically construct a smaller Conv2d keeping only selected channels."""
    w = conv.weight.detach()
    b = conv.bias.detach() if conv.bias is not None else None

    if depthwise:
        assert out_indices is not None
        new_c = len(out_indices)
        new_conv = nn.Conv2d(
            in_channels=new_c,
            out_channels=new_c,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=new_c,
            bias=b is not None,
            padding_mode=conv.padding_mode,
        ).to(device=w.device, dtype=w.dtype)
        new_conv.weight.data.copy_(w[out_indices])
        if b is not None:
            new_conv.bias.data.copy_(b[out_indices])
        return new_conv

    new_w = w
    if out_indices is not None:
        new_w = new_w[out_indices]
    if in_indices is not None:
        new_w = new_w[:, in_indices]

    new_conv = nn.Conv2d(
        in_channels=new_w.shape[1],
        out_channels=new_w.shape[0],
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=b is not None,
        padding_mode=conv.padding_mode,
    ).to(device=w.device, dtype=w.dtype)
    new_conv.weight.data.copy_(new_w)
    if b is not None:
        new_b = b[out_indices] if out_indices is not None else b
        new_conv.bias.data.copy_(new_b)
    return new_conv


def _slice_bn2d(bn: nn.BatchNorm2d, indices: torch.Tensor) -> nn.BatchNorm2d:
    """Physically construct a smaller BatchNorm2d keeping only selected channels."""
    new_bn = nn.BatchNorm2d(
        num_features=len(indices),
        eps=bn.eps,
        momentum=bn.momentum,
        affine=bn.affine,
        track_running_stats=bn.track_running_stats,
    ).to(device=bn.weight.device, dtype=bn.weight.dtype)
    if bn.affine:
        new_bn.weight.data.copy_(bn.weight.detach()[indices])
        new_bn.bias.data.copy_(bn.bias.detach()[indices])
    if bn.track_running_stats:
        new_bn.running_mean.data.copy_(bn.running_mean.detach()[indices])
        new_bn.running_var.data.copy_(bn.running_var.detach()[indices])
        new_bn.num_batches_tracked.copy_(bn.num_batches_tracked)
    return new_bn


def _rebuild_sequential(
    seq: nn.Sequential,
    new_conv: nn.Conv2d,
    new_bn: nn.BatchNorm2d = None,
) -> nn.Sequential:
    """Rebuild a Conv(+BN)(+ReLU6) sequential with newly sliced layers."""
    new_layers = []
    for layer in seq.children():
        if isinstance(layer, nn.Conv2d):
            new_layers.append(new_conv)
        elif isinstance(layer, nn.BatchNorm2d):
            if new_bn is not None:
                new_layers.append(new_bn)
        else:
            new_layers.append(layer)
    return nn.Sequential(*new_layers)


@torch.no_grad()
def _score_expansion_channels(block: InvertedResidual) -> torch.Tensor:
    """Compute coupled channel saliency across (pw_expand -> dw_conv -> pw_project).

    Importance combines:
      - Incoming expansion filter L2 norm ||W_expand[j]||_2
      - Depthwise BN scale |gamma_dw[j]| (or depthwise kernel L2 norm if BN is folded)
      - Outgoing projection filter L2 norm ||W_project[:, j]||_2
    """
    exp_conv, exp_bn = _extract_conv_bn(block.pw_expand)
    dw_conv, dw_bn = _extract_conv_bn(block.dw_conv)
    proj_conv, _ = _extract_conv_bn(block.pw_project)

    # Outgoing projection contribution per hidden channel j
    proj_norm = proj_conv.weight.detach().pow(2).sum(dim=(0, 2, 3)).sqrt()
    # Incoming expansion filter strength
    exp_norm = exp_conv.weight.detach().pow(2).sum(dim=(1, 2, 3)).sqrt()
    dw_norm = dw_conv.weight.detach().pow(2).sum(dim=(1, 2, 3)).sqrt()

    if dw_bn is not None and dw_bn.affine:
        bn_scale = dw_bn.weight.detach().abs()
        # Normalize each factor to unit mean so no single scale dominates
        score = (
            (bn_scale / (bn_scale.mean() + 1e-8))
            * (proj_norm / (proj_norm.mean() + 1e-8))
            * (exp_norm / (exp_norm.mean() + 1e-8)).sqrt()
        )
    else:
        score = exp_norm * dw_norm * proj_norm

    return score


@torch.no_grad()
def prune_mobilenetv2_structured(
    model: MobileNetV2CIFAR,
    prune_ratio: float = 0.0,
    prune_head: bool = True,
    min_channels: int = 8,
    channel_divisor: int = 4,
) -> Tuple[MobileNetV2CIFAR, Dict[str, object]]:
    """Structurally prune inverted-residual expansion channels (and optionally head_conv).

    Args:
        model: Trained FP32 MobileNetV2CIFAR model.
        prune_ratio: Fraction of eligible channels to remove in [0.0, 0.85].
        prune_head: If True, also prune the 1280-channel head_conv + classifier input.
        min_channels: Minimum channels retained in any pruned layer.
        channel_divisor: Round kept channel count to a multiple of this divisor.

    Returns:
        (pruned_model, pruning_summary)
    """
    if not (0.0 <= prune_ratio < 0.9):
        raise ValueError(f"prune_ratio must be in [0.0, 0.9), got {prune_ratio}")

    pruned = copy.deepcopy(model)
    layer_records: List[Dict[str, object]] = []

    if prune_ratio == 0.0:
        return pruned, {
            "prune_ratio_target": 0.0,
            "layers_pruned": 0,
            "details": layer_records,
        }

    # 1. Prune hidden expansion channels inside all expanded InvertedResidual blocks
    for idx, block in enumerate(pruned.blocks):
        if not isinstance(block, InvertedResidual) or block.pw_expand is None:
            continue

        orig_hidden = block.hidden_dim
        target_keep = int(round(orig_hidden * (1.0 - prune_ratio)))
        keep_c = min(orig_hidden, _round_channels(target_keep, min_channels, channel_divisor))
        if keep_c >= orig_hidden:
            continue

        scores = _score_expansion_channels(block)
        # Keep top-k indices sorted in original channel order
        keep_idx = torch.topk(scores, k=keep_c, largest=True).indices.sort().values

        exp_conv, exp_bn = _extract_conv_bn(block.pw_expand)
        dw_conv, dw_bn = _extract_conv_bn(block.dw_conv)
        proj_conv, proj_bn = _extract_conv_bn(block.pw_project)

        new_exp_conv = _slice_conv2d(exp_conv, out_indices=keep_idx)
        new_exp_bn = _slice_bn2d(exp_bn, keep_idx) if exp_bn is not None else None
        block.pw_expand = _rebuild_sequential(block.pw_expand, new_exp_conv, new_exp_bn)

        new_dw_conv = _slice_conv2d(dw_conv, out_indices=keep_idx, depthwise=True)
        new_dw_bn = _slice_bn2d(dw_bn, keep_idx) if dw_bn is not None else None
        block.dw_conv = _rebuild_sequential(block.dw_conv, new_dw_conv, new_dw_bn)

        new_proj_conv = _slice_conv2d(proj_conv, in_indices=keep_idx)
        block.pw_project = _rebuild_sequential(block.pw_project, new_proj_conv, proj_bn)
        block.hidden_dim = keep_c

        layer_records.append(
            {
                "layer": f"blocks.{idx} (hidden_expand)",
                "orig_channels": orig_hidden,
                "kept_channels": keep_c,
                "actual_prune_ratio": round(1.0 - keep_c / orig_hidden, 4),
            }
        )

    # 2. Optionally prune the 1280-channel head_conv + classifier Linear input dim
    if prune_head:
        head_conv, head_bn = _extract_conv_bn(pruned.head_conv)
        linear_idx = None
        linear_layer = None
        for i, layer in enumerate(pruned.classifier):
            if isinstance(layer, nn.Linear):
                linear_idx = i
                linear_layer = layer
                break

        if linear_layer is not None:
            orig_head = head_conv.out_channels
            target_keep = int(round(orig_head * (1.0 - prune_ratio)))
            keep_c = min(orig_head, _round_channels(target_keep, max(min_channels, 64), 8))
            if keep_c < orig_head:
                cls_norm = linear_layer.weight.detach().pow(2).sum(dim=0).sqrt()
                head_norm = head_conv.weight.detach().pow(2).sum(dim=(1, 2, 3)).sqrt()
                if head_bn is not None and head_bn.affine:
                    bn_scale = head_bn.weight.detach().abs()
                    scores = (
                        (bn_scale / (bn_scale.mean() + 1e-8))
                        * (cls_norm / (cls_norm.mean() + 1e-8))
                    )
                else:
                    scores = head_norm * cls_norm

                keep_idx = torch.topk(scores, k=keep_c, largest=True).indices.sort().values

                new_head_conv = _slice_conv2d(head_conv, out_indices=keep_idx)
                new_head_bn = _slice_bn2d(head_bn, keep_idx) if head_bn is not None else None
                pruned.head_conv = _rebuild_sequential(
                    pruned.head_conv, new_head_conv, new_head_bn
                )

                new_linear = nn.Linear(
                    keep_c, linear_layer.out_features, bias=linear_layer.bias is not None
                ).to(device=linear_layer.weight.device, dtype=linear_layer.weight.dtype)
                new_linear.weight.data.copy_(linear_layer.weight.detach()[:, keep_idx])
                if linear_layer.bias is not None:
                    new_linear.bias.data.copy_(linear_layer.bias.detach())
                pruned.classifier[linear_idx] = new_linear
                pruned.last_channel = keep_c

                layer_records.append(
                    {
                        "layer": "head_conv + classifier",
                        "orig_channels": orig_head,
                        "kept_channels": keep_c,
                        "actual_prune_ratio": round(1.0 - keep_c / orig_head, 4),
                    }
                )

    summary = {
        "prune_ratio_target": prune_ratio,
        "prune_head": prune_head,
        "layers_pruned": len(layer_records),
        "details": layer_records,
    }
    return pruned, summary
