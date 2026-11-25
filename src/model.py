"""CIFAR-10 adapted MobileNet-v2 architecture implemented from scratch in PyTorch.

Reference architecture:
    Sandler et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks", CVPR 2018.
    Torchvision reference: https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py

CIFAR-10 adaptations:
    Standard ImageNet MobileNet-v2 applies 5 stride-2 downsampling stages (224 -> 7),
    which collapses 32x32 CIFAR-10 inputs down to 1x1 prematurely. Here we set:
      - Stem conv stride = 1 (preserves 32x32)
      - Stage 2 (c=24) stride = 1 (preserves 32x32)
    Leaving 3 downsampling stages (32 -> 16 -> 8 -> 4), matching standard CIFAR-10 design
    while keeping identical parameter count (~2.24M at width_mult=1.0) and block structure.
"""

from typing import List, Optional

import torch
import torch.nn as nn


def _make_divisible(v: float, divisor: int = 8, min_value: Optional[int] = None) -> int:
    """Ensure every layer channel count is divisible by `divisor`."""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    """Standard Conv2d + BatchNorm2d + ReLU6 block."""

    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        bn_momentum: float = 0.1,
        bn_eps: float = 1e-5,
    ) -> None:
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_planes, momentum=bn_momentum, eps=bn_eps),
            nn.ReLU6(inplace=False),
        )


class InvertedResidual(nn.Module):
    """MobileNet-v2 Inverted Residual block with explicit named sub-layers.

    Naming sub-layers explicitly (pw_expand, dw_conv, pw_project) makes structured
    channel pruning of the expansion bottleneck and layer-wise quantization transparent.
    """

    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int,
        expand_ratio: int,
        bn_momentum: float = 0.1,
        bn_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if stride not in (1, 2):
            raise ValueError(f"stride must be 1 or 2, got {stride}")

        self.inp = inp
        self.oup = oup
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.use_res_connect = self.stride == 1 and inp == oup

        hidden_dim = int(round(inp * expand_ratio))
        self.hidden_dim = hidden_dim
        self.has_expand = expand_ratio != 1

        if self.has_expand:
            self.pw_expand = ConvBNReLU(
                inp,
                hidden_dim,
                kernel_size=1,
                stride=1,
                bn_momentum=bn_momentum,
                bn_eps=bn_eps,
            )
        else:
            self.pw_expand = None

        self.dw_conv = ConvBNReLU(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            stride=stride,
            groups=hidden_dim,
            bn_momentum=bn_momentum,
            bn_eps=bn_eps,
        )

        # Linear bottleneck projection (no ReLU6 after projection)
        self.pw_project = nn.Sequential(
            nn.Conv2d(hidden_dim, oup, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(oup, momentum=bn_momentum, eps=bn_eps),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        if self.pw_expand is not None:
            out = self.pw_expand(out)
        out = self.dw_conv(out)
        out = self.pw_project(out)
        if self.use_res_connect:
            return x + out
        return out


class MobileNetV2CIFAR(nn.Module):
    """MobileNet-v2 adapted for 32x32 CIFAR-10 classification."""

    DEFAULT_CFG = [
        # t (expand), c (out_channels), n (num_blocks), s (stride)
        [1, 16, 1, 1],
        [6, 24, 2, 1],   # Stride 1 for CIFAR-10 (ImageNet uses 2)
        [6, 32, 3, 2],   # 32x32 -> 16x16
        [6, 64, 4, 2],   # 16x16 -> 8x8
        [6, 96, 3, 1],   # 8x8
        [6, 160, 3, 2],  # 8x8 -> 4x4
        [6, 320, 1, 1],  # 4x4
    ]

    def __init__(
        self,
        num_classes: int = 10,
        width_mult: float = 1.0,
        dropout: float = 0.2,
        bn_momentum: float = 0.1,
        bn_eps: float = 1e-5,
        inverted_residual_setting: Optional[List[List[int]]] = None,
        round_nearest: int = 8,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.width_mult = width_mult
        self.dropout_p = dropout
        self.bn_momentum = bn_momentum
        self.bn_eps = bn_eps

        if inverted_residual_setting is None:
            inverted_residual_setting = self.DEFAULT_CFG

        input_channel = _make_divisible(32 * width_mult, round_nearest)
        self.last_channel = _make_divisible(
            1280 * max(1.0, width_mult), round_nearest
        )

        # Stem: stride=1 for CIFAR-10 (32x32)
        self.stem = ConvBNReLU(
            3,
            input_channel,
            kernel_size=3,
            stride=1,
            bn_momentum=bn_momentum,
            bn_eps=bn_eps,
        )

        # Inverted residual blocks
        blocks: List[nn.Module] = []
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                blocks.append(
                    InvertedResidual(
                        input_channel,
                        output_channel,
                        stride=stride,
                        expand_ratio=t,
                        bn_momentum=bn_momentum,
                        bn_eps=bn_eps,
                    )
                )
                input_channel = output_channel
        self.blocks = nn.Sequential(*blocks)

        # Final 1x1 expansion before global average pooling
        self.head_conv = ConvBNReLU(
            input_channel,
            self.last_channel,
            kernel_size=1,
            stride=1,
            bn_momentum=bn_momentum,
            bn_eps=bn_eps,
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.last_channel, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head_conv(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def build_mobilenetv2_cifar10(
    width_mult: float = 1.0,
    dropout: float = 0.2,
    bn_momentum: float = 0.1,
    bn_eps: float = 1e-5,
    num_classes: int = 10,
) -> MobileNetV2CIFAR:
    """Factory helper for CIFAR-10 MobileNet-v2."""
    return MobileNetV2CIFAR(
        num_classes=num_classes,
        width_mult=width_mult,
        dropout=dropout,
        bn_momentum=bn_momentum,
        bn_eps=bn_eps,
    )
