"""Custom BatchNorm2d-into-Conv2d folding implemented from scratch in PyTorch.

Folding BatchNorm into its preceding Conv2d mathematically absorbs the BN scale,
shift, running mean, and running variance into the Conv2d weight and bias:
    w_fold = w * (gamma / sqrt(var + eps))
    b_fold = beta + (b - mean) * (gamma / sqrt(var + eps))

Benefits for in-memory compression:
  1. Eliminates all BatchNorm2d weights, biases, and running_mean/running_var buffers
     from RAM/VRAM.
  2. Eliminates intermediate pre-BN activation buffers during inference.
  3. Enables direct Quantization-Aware Training (QAT) on the exact inference weights.
"""

import copy
import torch
import torch.nn as nn

from src.model import ConvBNReLU, InvertedResidual, MobileNetV2CIFAR


@torch.no_grad()
def fold_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """Fold a Conv2d + BatchNorm2d pair into a single Conv2d with bias."""
    if conv.out_channels != bn.num_features:
        raise ValueError(
            f"Channel mismatch: conv.out_channels={conv.out_channels}, "
            f"bn.num_features={bn.num_features}"
        )

    folded = nn.Conv2d(
        in_channels=conv.in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=True,
        padding_mode=conv.padding_mode,
    ).to(device=conv.weight.device, dtype=conv.weight.dtype)

    gamma = bn.weight.detach()
    beta = bn.bias.detach()
    mean = bn.running_mean.detach()
    var = bn.running_var.detach()
    eps = bn.eps

    inv_std = torch.rsqrt(var + eps)
    scale = gamma * inv_std

    # Reshape scale for broadcasting across (C_out, C_in/groups, kH, kW)
    w_folded = conv.weight.detach() * scale.view(-1, 1, 1, 1)
    if conv.bias is not None:
        b_orig = conv.bias.detach()
    else:
        b_orig = torch.zeros_like(mean)
    b_folded = beta + (b_orig - mean) * scale

    folded.weight.copy_(w_folded)
    folded.bias.copy_(b_folded)
    return folded


def _fold_sequential_conv_bn(seq: nn.Sequential) -> nn.Sequential:
    """Fold consecutive (Conv2d, BatchNorm2d) pairs inside an nn.Sequential."""
    layers = list(seq.children())
    new_layers = []
    i = 0
    while i < len(layers):
        if (
            i + 1 < len(layers)
            and isinstance(layers[i], nn.Conv2d)
            and isinstance(layers[i + 1], nn.BatchNorm2d)
        ):
            new_layers.append(fold_conv_bn(layers[i], layers[i + 1]))
            i += 2
        else:
            new_layers.append(layers[i])
            i += 1
    return nn.Sequential(*new_layers)


def fold_model_batchnorm(model: MobileNetV2CIFAR, inplace: bool = False) -> MobileNetV2CIFAR:
    """Fold every Conv2d + BatchNorm2d pair across MobileNetV2CIFAR.

    The model is placed in eval() mode prior to folding so running statistics are used.
    """
    m = model if inplace else copy.deepcopy(model)
    m.eval()

    if isinstance(m.stem, (ConvBNReLU, nn.Sequential)):
        m.stem = _fold_sequential_conv_bn(m.stem)

    if isinstance(m.head_conv, (ConvBNReLU, nn.Sequential)):
        m.head_conv = _fold_sequential_conv_bn(m.head_conv)

    for block in m.blocks:
        if isinstance(block, InvertedResidual):
            if block.pw_expand is not None:
                block.pw_expand = _fold_sequential_conv_bn(block.pw_expand)
            block.dw_conv = _fold_sequential_conv_bn(block.dw_conv)
            block.pw_project = _fold_sequential_conv_bn(block.pw_project)

    return m
