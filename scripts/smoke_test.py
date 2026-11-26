"""Fast synthetic-data smoke test verifying every custom module (runs in <10s on CPU/GPU)."""

import os
import sys

# Ensure project root is on sys.path when invoked as `python scripts/smoke_test.py`
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from src.compression import (
    compute_full_compression_report,
    fold_model_batchnorm,
    pack_int_tensor,
    prune_mobilenetv2_structured,
    quantize_mobilenetv2,
    unpack_int_tensor,
)
from src.model import build_mobilenetv2_cifar10
from src.utils import get_device, set_seed


def test_bit_packing_all_bitwidths() -> None:
    for bits in (2, 3, 4, 5, 6, 7, 8):
        qmax = (1 << (bits - 1)) - 1
        qmin = -qmax
        q = torch.randint(qmin, qmax + 1, (17, 13, 3, 3), dtype=torch.int32)
        packed, shape = pack_int_tensor(q, bits, qmin)
        unpacked = unpack_int_tensor(packed, shape, bits, qmin)
        assert torch.equal(q, unpacked), f"Bit-packing mismatch at {bits} bits!"
    print("[PASS] Pure-PyTorch bit-packing verified lossless for 2..8 bits.")


def test_bn_folding_equivalence(device: torch.device) -> None:
    model = build_mobilenetv2_cifar10().to(device)
    # Populate realistic running stats
    model.train()
    for _ in range(3):
        _ = model(torch.randn(8, 3, 32, 32, device=device))
    model.eval()

    x = torch.randn(4, 3, 32, 32, device=device)
    with torch.no_grad():
        y_ref = model(x)
        folded = fold_model_batchnorm(model)
        y_fold = folded(x)
    max_diff = (y_ref - y_fold).abs().max().item()
    assert max_diff < 1e-4, f"BN folding discrepancy too high: {max_diff}"
    print(f"[PASS] Custom BN folding verified (max output diff = {max_diff:.2e}).")


def test_pruning_quantization_qat_and_metrics(device: torch.device) -> None:
    baseline = build_mobilenetv2_cifar10().to(device).eval()
    x = torch.randn(4, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (4,), device=device)

    # Structured pruning
    pruned, prune_summary = prune_mobilenetv2_structured(baseline, prune_ratio=0.40, prune_head=True)
    assert prune_summary["layers_pruned"] > 0
    out_p = pruned(x)
    assert out_p.shape == (4, 10)

    # Custom W4A4 quantization + calibration + STE backward pass
    qmodel, policy = quantize_mobilenetv2(
        pruned, weight_bits=4, act_bits=4, protect_first_last=True, fold_bn=True
    )
    qmodel = qmodel.to(device)
    # One forward pass initializes activation quantizers
    out_q = qmodel(x)
    loss = torch.nn.functional.cross_entropy(out_q, y)
    loss.backward()

    # Check gradients flow through STE to weights and quantizer scales
    has_weight_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in qmodel.parameters())
    assert has_weight_grad, "No gradients flowed through custom STE quantizers!"

    # Full in-memory metrics + live packed verification
    report = compute_full_compression_report(baseline, qmodel, device=device, verify_packing=True)
    pv = report["packing_verification"]
    assert pv["matches_analytical_bytes"], "Live packed bytes differ from analytical bytes!"
    assert pv["max_roundtrip_dequant_error"] < 1e-4, f"Packing error: {pv['max_roundtrip_dequant_error']}"

    print(
        f"[PASS] Prune(0.40)+W4A4+STE+Metrics verified | "
        f"Base: {report['baseline_model_mb']:.3f} MB -> Comp: {report['compressed_model_mb']:.3f} MB | "
        f"Model CR: {report['model_compression_ratio']:.2f}x | "
        f"Weight CR: {report['weight_compression_ratio']:.2f}x | "
        f"Act CR: {report['activation_compression_ratio']:.2f}x"
    )


def main() -> None:
    set_seed(42)
    device = get_device()
    print(f"Running MobileNetV2-CIFAR10-Compression smoke tests on {device}...")
    test_bit_packing_all_bitwidths()
    test_bn_folding_equivalence(device)
    test_pruning_quantization_qat_and_metrics(device)
    print("\nALL SMOKE TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
