"""Custom in-memory model compression package (zero external compression APIs)."""

from .bn_fold import fold_conv_bn, fold_model_batchnorm
from .metrics import (
    compute_baseline_model_memory,
    compute_compressed_model_memory,
    compute_full_compression_report,
    measure_activation_memory,
)
from .packing import PackedCompressedModel, pack_int_tensor, unpack_int_tensor
from .prune import prune_mobilenetv2_structured
from .quantize import (
    CustomActivationQuantizer,
    CustomWeightQuantizer,
    QuantConv2d,
    QuantLinear,
    calibrate_model,
    quantize_mobilenetv2,
)

__all__ = [
    "fold_conv_bn",
    "fold_model_batchnorm",
    "prune_mobilenetv2_structured",
    "pack_int_tensor",
    "unpack_int_tensor",
    "PackedCompressedModel",
    "CustomWeightQuantizer",
    "CustomActivationQuantizer",
    "QuantConv2d",
    "QuantLinear",
    "quantize_mobilenetv2",
    "calibrate_model",
    "compute_baseline_model_memory",
    "compute_compressed_model_memory",
    "measure_activation_memory",
    "compute_full_compression_report",
]
