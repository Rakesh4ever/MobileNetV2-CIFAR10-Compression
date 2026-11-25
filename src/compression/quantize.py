"""Custom configurable Weight & Activation Quantization (pure PyTorch, zero APIs).

Implements from scratch:
  1. Per-channel symmetric uniform weight quantization with optional MSE-optimal
     clipping initialization and LSQ-style learnable per-channel scales (via STE).
  2. Per-tensor asymmetric uniform activation quantization with EMA calibration,
     exact zero-point representation (critical for ReLU6 zero-padding), and
     learnable clipping bounds during QAT (via STE).
  3. Custom `QuantConv2d` and `QuantLinear` drop-in wrappers around `F.conv2d` and
     `F.linear` with configurable per-layer bit-widths (including optional mixed-precision
     protection for the first stem Conv2d and final Linear classifier).
"""

import copy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.compression.bn_fold import fold_model_batchnorm
from src.model import InvertedResidual, MobileNetV2CIFAR


def _ste_round(x: torch.Tensor) -> torch.Tensor:
    """Round in the forward pass, pass gradient unchanged in the backward pass (STE)."""
    return (x.round() - x).detach() + x


def _grad_scale(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Scale gradients in the backward pass while keeping forward value unchanged (LSQ)."""
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad


class CustomWeightQuantizer(nn.Module):
    """Per-channel symmetric uniform quantizer for Conv2d / Linear weights.

    Quantization formula for output channel c:
        q_c = clamp(round(w_c / s_c), -qmax, +qmax)
        w_hat_c = q_c * s_c
    where qmax = 2^(b-1) - 1 and s_c > 0 is a learnable per-channel scale.
    """

    def __init__(
        self,
        out_channels: int,
        num_bits: int = 8,
        mse_init: bool = True,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.num_bits = int(num_bits)
        self.mse_init = mse_init

        if self.num_bits < 32:
            if not (2 <= self.num_bits <= 8):
                raise ValueError(f"Weight bit-width must be in 2..8 or 32, got {num_bits}")
            self.qmax = (1 << (self.num_bits - 1)) - 1
            self.qmin = -self.qmax
        else:
            self.qmax = 0
            self.qmin = 0

        # Learnable per-channel scale (initialized lazily from actual weights)
        self.scale = nn.Parameter(torch.ones(out_channels, dtype=torch.float32))
        self.register_buffer("initialized", torch.tensor(False, dtype=torch.bool))

    @torch.no_grad()
    def initialize_from_weight(self, weight: torch.Tensor) -> None:
        """Initialize per-channel scale via max-abs or MSE-optimal clipping search."""
        if self.num_bits >= 32:
            self.initialized.fill_(True)
            return

        w_flat = weight.detach().reshape(self.out_channels, -1)
        max_abs = w_flat.abs().amax(dim=1).clamp(min=1e-8)
        best_scale = max_abs / float(self.qmax)

        # For low bit-widths (<=4 bits), MSE-optimal clipping prevents outlier distortion
        if self.mse_init and self.num_bits <= 4:
            best_mse = torch.full((self.out_channels,), float("inf"), device=weight.device)
            for alpha in torch.linspace(0.70, 1.00, steps=16, device=weight.device):
                cand_scale = (max_abs * alpha / float(self.qmax)).clamp(min=1e-8)
                q = torch.clamp((w_flat / cand_scale.unsqueeze(1)).round(), self.qmin, self.qmax)
                recon = q * cand_scale.unsqueeze(1)
                mse = (w_flat - recon).pow(2).mean(dim=1)
                improved = mse < best_mse
                best_mse = torch.where(improved, mse, best_mse)
                best_scale = torch.where(improved, cand_scale, best_scale)

        self.scale.data.copy_(best_scale.to(self.scale.dtype))
        self.initialized.fill_(True)

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        if self.num_bits >= 32:
            return weight

        if not bool(self.initialized.item()):
            self.initialize_from_weight(weight)

        # Apply LSQ gradient scaling for stable low-bit QAT
        n_per_ch = max(weight.numel() // self.out_channels, 1)
        g = 1.0 / ((n_per_ch * self.qmax) ** 0.5)
        s = _grad_scale(self.scale, g).abs().clamp(min=1e-8)

        view_shape = [self.out_channels] + [1] * (weight.ndim - 1)
        s_view = s.view(view_shape)

        w_scaled = weight / s_view
        w_clamped = torch.clamp(w_scaled, float(self.qmin), float(self.qmax))
        w_q = _ste_round(w_clamped)
        return w_q * s_view

    @torch.no_grad()
    def export_quantized(self, weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return exact integer quantized weights and per-channel scales for bit-packing."""
        if not bool(self.initialized.item()):
            self.initialize_from_weight(weight)
        s = self.scale.detach().abs().clamp(min=1e-8)
        view_shape = [self.out_channels] + [1] * (weight.ndim - 1)
        q_int = torch.clamp((weight.detach() / s.view(view_shape)).round(), self.qmin, self.qmax).to(
            torch.int32
        )
        return q_int, s


class CustomActivationQuantizer(nn.Module):
    """Per-tensor asymmetric uniform quantizer for activations.

    Supports:
      - Calibration mode: collects exponential moving average (EMA) min/max bounds.
      - QAT mode: learns clipping bounds (min_val, max_val) via STE with exact
        zero-point nudging so 0.0 maps cleanly to an integer level.
    """

    def __init__(
        self,
        num_bits: int = 8,
        ema_decay: float = 0.9,
        non_negative: bool = False,
    ) -> None:
        super().__init__()
        self.num_bits = int(num_bits)
        self.ema_decay = ema_decay
        self.non_negative = non_negative
        self.calibrating = False

        if self.num_bits < 32:
            if not (2 <= self.num_bits <= 8):
                raise ValueError(f"Activation bit-width must be in 2..8 or 32, got {num_bits}")
            self.qmin = 0
            self.qmax = (1 << self.num_bits) - 1
        else:
            self.qmin = 0
            self.qmax = 0

        self.min_val = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.max_val = nn.Parameter(torch.tensor(6.0, dtype=torch.float32))
        self.register_buffer("initialized", torch.tensor(False, dtype=torch.bool))

    def _compute_scale_zp(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Derive positive scale and integer zero-point ensuring 0.0 is representable."""
        min_v = torch.zeros_like(self.min_val) if self.non_negative else torch.minimum(
            self.min_val, torch.zeros_like(self.min_val)
        )
        max_v = torch.maximum(self.max_val, min_v + 1e-5)
        scale = (max_v - min_v) / float(self.qmax - self.qmin)
        scale = scale.clamp(min=1e-8)

        # Nudge zero-point to an exact integer in [qmin, qmax]
        zp_float = self.qmin - min_v / scale
        zp = _ste_round(torch.clamp(zp_float, float(self.qmin), float(self.qmax)))
        return scale, zp

    @torch.no_grad()
    def observe(self, x: torch.Tensor) -> None:
        """Update running min/max statistics from a calibration batch."""
        x_det = x.detach()
        flat = x_det.float().reshape(-1)
        # Subsample large activation batches to stay well below PyTorch's 2^24 quantile limit
        if flat.numel() > 500_000:
            stride = (flat.numel() // 500_000) + 1
            flat_sample = flat[::stride]
        else:
            flat_sample = flat

        if self.non_negative:
            cur_min = torch.tensor(0.0, device=x.device, dtype=torch.float32)
            # Use 99.95% quantile for low-bit activations to ignore extreme single-pixel outliers
            if self.num_bits <= 4 and flat_sample.numel() > 1024:
                cur_max = torch.quantile(flat_sample, 0.9995)
            else:
                cur_max = flat.max()
        else:
            if self.num_bits <= 4 and flat_sample.numel() > 1024:
                cur_min = torch.quantile(flat_sample, 0.0005)
                cur_max = torch.quantile(flat_sample, 0.9995)
            else:
                cur_min = flat.min()
                cur_max = flat.max()

        cur_min = torch.minimum(cur_min, torch.tensor(0.0, device=x.device))
        cur_max = torch.maximum(cur_max, cur_min + 1e-5)

        if not bool(self.initialized.item()):
            self.min_val.data.copy_(cur_min)
            self.max_val.data.copy_(cur_max)
            self.initialized.fill_(True)
        else:
            self.min_val.data.mul_(self.ema_decay).add_(cur_min * (1.0 - self.ema_decay))
            self.max_val.data.mul_(self.ema_decay).add_(cur_max * (1.0 - self.ema_decay))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_bits >= 32:
            return x

        if self.calibrating or not bool(self.initialized.item()):
            self.observe(x)
            if self.calibrating:
                return x

        scale, zp = self._compute_scale_zp()
        x_scaled = x / scale + zp
        x_clamped = torch.clamp(x_scaled, float(self.qmin), float(self.qmax))
        x_q = _ste_round(x_clamped)
        return (x_q - zp) * scale

    @torch.no_grad()
    def export_params(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return scalar (scale, zero_point) for overhead accounting and packing."""
        scale, zp = self._compute_scale_zp()
        return scale.detach(), zp.detach().round().to(torch.int32)


class QuantConv2d(nn.Module):
    """Custom quantized Conv2d wrapper using pure `F.conv2d`."""

    def __init__(
        self,
        conv: nn.Conv2d,
        weight_bits: int = 8,
        act_bits: int = 8,
        input_non_negative: bool = False,
        layer_name: str = "",
        is_protected: bool = False,
    ) -> None:
        super().__init__()
        self.layer_name = layer_name
        self.is_protected = is_protected
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups

        self.weight = nn.Parameter(conv.weight.detach().clone())
        if conv.bias is not None:
            self.bias = nn.Parameter(conv.bias.detach().clone())
        else:
            self.register_parameter("bias", None)

        self.weight_quant = CustomWeightQuantizer(
            out_channels=self.out_channels,
            num_bits=weight_bits,
            mse_init=True,
        ).to(conv.weight.device)
        self.act_quant = CustomActivationQuantizer(
            num_bits=act_bits,
            non_negative=input_non_negative,
        ).to(conv.weight.device)
        # Initialize weight scale immediately from loaded FP32 weights
        self.weight_quant.initialize_from_weight(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_q = self.act_quant(x)
        w_q = self.weight_quant(self.weight)
        return F.conv2d(
            x_q,
            w_q,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class QuantLinear(nn.Module):
    """Custom quantized Linear wrapper using pure `F.linear`."""

    def __init__(
        self,
        linear: nn.Linear,
        weight_bits: int = 8,
        act_bits: int = 8,
        input_non_negative: bool = True,
        layer_name: str = "",
        is_protected: bool = False,
    ) -> None:
        super().__init__()
        self.layer_name = layer_name
        self.is_protected = is_protected
        self.in_features = linear.in_features
        self.out_features = linear.out_features

        self.weight = nn.Parameter(linear.weight.detach().clone())
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.detach().clone())
        else:
            self.register_parameter("bias", None)

        self.weight_quant = CustomWeightQuantizer(
            out_channels=self.out_features,
            num_bits=weight_bits,
            mse_init=True,
        ).to(linear.weight.device)
        self.act_quant = CustomActivationQuantizer(
            num_bits=act_bits,
            non_negative=input_non_negative,
        ).to(linear.weight.device)
        self.weight_quant.initialize_from_weight(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_q = self.act_quant(x)
        w_q = self.weight_quant(self.weight)
        return F.linear(x_q, w_q, self.bias)


def _replace_conv_in_sequential(
    seq: nn.Sequential,
    w_bits: int,
    a_bits: int,
    input_non_negative: bool,
    name_prefix: str,
    is_protected: bool,
    policy_log: List[Dict[str, object]],
) -> nn.Sequential:
    new_layers = []
    for i, layer in enumerate(seq.children()):
        if isinstance(layer, nn.Conv2d):
            qconv = QuantConv2d(
                layer,
                weight_bits=w_bits,
                act_bits=a_bits,
                input_non_negative=input_non_negative,
                layer_name=f"{name_prefix}.{i}",
                is_protected=is_protected,
            )
            policy_log.append(
                {
                    "layer": f"{name_prefix}.{i}",
                    "type": "DepthwiseConv2d" if layer.groups == layer.in_channels and layer.groups > 1 else "Conv2d",
                    "weight_shape": list(layer.weight.shape),
                    "weight_bits": w_bits,
                    "act_bits": a_bits,
                    "protected_exception": is_protected,
                }
            )
            new_layers.append(qconv)
        else:
            new_layers.append(layer)
    return nn.Sequential(*new_layers)


def quantize_mobilenetv2(
    model: MobileNetV2CIFAR,
    weight_bits: int = 8,
    act_bits: int = 8,
    protect_first_last: bool = True,
    fold_bn: bool = True,
) -> Tuple[MobileNetV2CIFAR, List[Dict[str, object]]]:
    """Convert MobileNetV2CIFAR to a custom quantized model with layer-wise policy log.

    Layer policy (for Q2b documentation):
      - All hidden InvertedResidual blocks (pw_expand, dw_conv, pw_project) and head_conv
        are quantized to `(weight_bits, act_bits)`.
      - When `protect_first_last=True` (default), the first 3-channel `stem` Conv2d and
        final 10-class `classifier` Linear layer are kept at `max(8, weight_bits)` /
        `max(8, act_bits)` (or 32 if baseline). These two boundary layers represent <1%
        of total parameters while preventing severe accuracy collapse at 2-4 bits.
      - When `protect_first_last=False`, every single Conv2d and Linear layer uses uniform
        `(weight_bits, act_bits)`.
    """
    qmodel = fold_model_batchnorm(model, inplace=False) if fold_bn else copy.deepcopy(model)
    policy_log: List[Dict[str, object]] = []

    edge_w_bits = weight_bits if (not protect_first_last or weight_bits >= 8) else 8
    edge_a_bits = act_bits if (not protect_first_last or act_bits >= 8) else 8
    edge_protected = protect_first_last and (weight_bits < 8 or act_bits < 8)

    # 1. Stem (receives normalized RGB image, which has negative values -> non_negative=False)
    qmodel.stem = _replace_conv_in_sequential(
        qmodel.stem,
        w_bits=edge_w_bits,
        a_bits=edge_a_bits,
        input_non_negative=False,
        name_prefix="stem",
        is_protected=edge_protected,
        policy_log=policy_log,
    )

    # 2. Inverted residual blocks
    for idx, block in enumerate(qmodel.blocks):
        if not isinstance(block, InvertedResidual):
            continue
        # Block 0 receives ReLU6 output from stem (non-negative); subsequent blocks receive
        # linear bottleneck outputs (signed -> non_negative=False).
        block_inp_nonneg = idx == 0

        if block.pw_expand is not None:
            block.pw_expand = _replace_conv_in_sequential(
                block.pw_expand,
                w_bits=weight_bits,
                a_bits=act_bits,
                input_non_negative=block_inp_nonneg,
                name_prefix=f"blocks.{idx}.pw_expand",
                is_protected=False,
                policy_log=policy_log,
            )
            dw_inp_nonneg = True  # Output of pw_expand ReLU6
        else:
            dw_inp_nonneg = block_inp_nonneg

        block.dw_conv = _replace_conv_in_sequential(
            block.dw_conv,
            w_bits=weight_bits,
            a_bits=act_bits,
            input_non_negative=dw_inp_nonneg,
            name_prefix=f"blocks.{idx}.dw_conv",
            is_protected=False,
            policy_log=policy_log,
        )

        block.pw_project = _replace_conv_in_sequential(
            block.pw_project,
            w_bits=weight_bits,
            a_bits=act_bits,
            input_non_negative=True,  # Output of dw_conv ReLU6
            name_prefix=f"blocks.{idx}.pw_project",
            is_protected=False,
            policy_log=policy_log,
        )

    # 3. Head 1x1 conv (receives signed output of last linear bottleneck)
    qmodel.head_conv = _replace_conv_in_sequential(
        qmodel.head_conv,
        w_bits=weight_bits,
        a_bits=act_bits,
        input_non_negative=False,
        name_prefix="head_conv",
        is_protected=False,
        policy_log=policy_log,
    )

    # 4. Final classifier Linear (receives ReLU6 + AvgPool output -> non-negative)
    new_cls = []
    for i, layer in enumerate(qmodel.classifier.children()):
        if isinstance(layer, nn.Linear):
            qlinear = QuantLinear(
                layer,
                weight_bits=edge_w_bits,
                act_bits=edge_a_bits,
                input_non_negative=True,
                layer_name=f"classifier.{i}",
                is_protected=edge_protected,
            )
            policy_log.append(
                {
                    "layer": f"classifier.{i}",
                    "type": "Linear",
                    "weight_shape": list(layer.weight.shape),
                    "weight_bits": edge_w_bits,
                    "act_bits": edge_a_bits,
                    "protected_exception": edge_protected,
                }
            )
            new_cls.append(qlinear)
        else:
            new_cls.append(layer)
    qmodel.classifier = nn.Sequential(*new_cls)

    return qmodel, policy_log


@torch.no_grad()
def calibrate_model(
    model: nn.Module,
    calib_loader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> None:
    """Calibrate all CustomActivationQuantizer ranges on calibration data."""
    model.eval()
    act_quantizers = [m for m in model.modules() if isinstance(m, CustomActivationQuantizer)]
    for q in act_quantizers:
        q.calibrating = True
        q.initialized.fill_(False)

    for batch_idx, (images, _) in enumerate(calib_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        _ = model(images)

    for q in act_quantizers:
        q.calibrating = False
