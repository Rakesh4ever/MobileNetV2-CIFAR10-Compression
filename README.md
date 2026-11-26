# MobileNetV2-CIFAR10-Compression
## Custom In-Memory Model Compression for MobileNet-v2 on CIFAR-10

*CS6886W — Systems Engineering for Deep Learning (IIT Madras)*

This repository implements **MobileNet-v2** trained on **CIFAR-10** and a **from-scratch, pure-PyTorch in-memory model compression pipeline** (zero external compression APIs such as `torch.ao.quantization`, `torch.nn.utils.prune`, or `bitsandbytes`).

Compression is strictly evaluated on **in-memory RAM/VRAM footprint** (weights, activations, and explicit metadata/scaling overheads), verified with actual pure-PyTorch sub-byte bit-packing (`uint8` tensors).

---

## Repository Structure

```text
mobilenetv2-cifar10-compression/
├── README.md                        # Setup, seed config, results, and reproduction commands
├── requirements.txt                 # Python dependencies
├── colab_runner.ipynb               # End-to-end Google Colab GPU notebook
├── report_final.md                  # Full experimental report and compression analysis
├── test.py                          # CLI entrypoint (`--weight_quant_bits`, `--activation_quant_bits`, ...)
├── sweep.py                         # 8- or 16-run compression sweep with WandB logging
├── checkpoints/
│   └── baseline_best.pth            # Pretrained FP32 MobileNet-v2 baseline (95.01% Top-1)
├── results/                         # Training curves, confusion matrix, sweep tables & plots
├── scripts/
│   └── smoke_test.py                # Unit & integration verification of custom compression modules
└── src/
    ├── utils.py                     # Reproducibility seeding, device helpers, environment info
    ├── data.py                      # CIFAR-10 normalization, augmentation, and calibration loader
    ├── model.py                     # CIFAR-10 adapted MobileNet-v2 architecture
    ├── train.py                     # FP32 baseline training + loss/accuracy curve plotting
    ├── evaluate.py                  # Top-1/Top-5 evaluation, confusion matrix, failure mode analysis
    ├── compress_pipeline.py         # End-to-end Structured Pruning + BN Folding + QAT + KD pipeline
    └── compression/
        ├── bn_fold.py               # Custom Conv2d + BatchNorm2d folding
        ├── prune.py                 # Custom structured expansion-channel & head pruning
        ├── quantize.py              # Custom per-channel weight & asymmetric activation quantizers (STE/LSQ)
        ├── packing.py               # Pure-PyTorch arbitrary bit-width (2..8-bit) uint8 bit-packer
        └── metrics.py               # Exact RAM/VRAM size, storage overhead, and hook-based activation CR
```

---

## Compression Methodology Overview

1. **Custom Structured Expansion-Channel Pruning (`src/compression/prune.py`):**
   - Scores hidden channels in MobileNet-v2 inverted residual blocks (`expand_ratio=6`) and the `1280`-channel `head_conv` using coupled BN-$\gamma$ and filter L2 saliency.
   - Physically reconstructs smaller dense `Conv2d`/`BatchNorm2d` layers while preserving all residual skip-connection widths—directly shrinking **both weights and intermediate activations** in RAM/VRAM with zero sparse-index overhead.
2. **Custom BatchNorm Folding (`src/compression/bn_fold.py`):**
   - Mathematically folds every `BatchNorm2d` into its preceding `Conv2d` weights and biases, removing all BN parameters and buffers from memory.
3. **Custom Weight & Activation Quantization + QAT (`src/compression/quantize.py`):**
   - **Weights:** Per-channel symmetric uniform quantization ($b_w \in \{2..8\}$) with MSE-optimal clipping initialization and LSQ-scaled learnable per-channel scales via Straight-Through Estimator (STE).
   - **Activations:** Per-tensor asymmetric uniform quantization ($b_a \in \{2..8\}$) with EMA calibration, exact zero-point representation (for ReLU6), and learnable clipping bounds during QAT.
   - **Mixed-Precision Boundary Protection (configurable):** By default keeps the first 3-channel `stem` Conv2d and final 10-class `classifier` Linear at 8-bit (`--no_protect_first_last` disables this for uniform quantization).
   - **Self-Distillation QAT:** Optional KL-divergence distillation from the frozen FP32 baseline teacher preserves accuracy at aggressive 2–4 bit + pruned regimes.
4. **Live In-Memory Bit-Packing & Overhead Accounting (`src/compression/packing.py`, `src/compression/metrics.py`):**
   - Packs quantized weights into contiguous `torch.uint8` tensors and verifies exact byte equality against analytical estimates and lossless round-trip dequantization.
   - Explicitly counts per-channel FP16 weight scales, FP16 biases, activation scales/zero-points, and per-layer metadata headers in all compressed size and compression ratio calculations.

---

## Environment & Reproducibility

- **Verified Experimental Environment (Google Colab):** Python `3.13.16`, PyTorch `2.14.1+cu130`, torchvision `0.29.1+cu130`, CUDA `13.0`, NVIDIA `Tesla T4` GPU.
- **Random Seed Configuration:** Default `--seed 42` fixes `random`, `numpy`, `torch.manual_seed`, `torch.cuda.manual_seed_all`, `cudnn.deterministic = True`, `cudnn.benchmark = False`, and PyTorch `DataLoader` generators.

---

## Experimental Results Summary

- **Uncompressed FP32 MobileNet-v2 Baseline:** **95.01% Top-1** (`99.71%` Top-5) | `2,236,682` trainable params | **8.6624 MB** in-memory (`6.1944 MB` total / `1.1250 MB` peak activations for `(1, 3, 32, 32)`).
- **Live Bit-Packing Verification:** All 16 sweep configurations verified exact byte match (`PackedCompressedModel` live `uint8`+`float16` RAM bytes == analytical bytes) with `0.0` max dequantization roundtrip error.

| Pareto Configuration | W-bits | A-bits | Prune | PTQ Acc (%) | Final QAT Acc (%) | Weight CR | Act CR (Total / Peak) | Model CR | Final Size (MB) | Overhead (KB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `W4_A6_P00` | 4 | 6 | 0.00 | 94.23 | **94.90** | 7.72x | 5.33x / 5.33x | **7.71x** | 1.1229 | 67.70 |
| `W4_A4_P40` | 4 | 4 | 0.40 | 91.04 | **93.93** | 12.80x | 12.20x / 13.09x | **12.79x** | 0.6772 | 43.54 |
| `W4_A4_P55` | 4 | 4 | 0.55 | 90.11 | **93.05** | 16.99x | 15.30x / **18.00x** | **16.98x** | 0.5101 | 34.45 |
| `W3_A4_P45` | 3 | 4 | 0.45 | 81.70 | **92.89** | 18.15x | 13.13x / 14.40x | **18.14x** | 0.4776 | 40.45 |
| `W3_A3_P40` | 3 | 3 | 0.40 | 41.95 | **91.80** | 16.65x | **16.23x** / 17.45x | **16.64x** | 0.5205 | 43.54 |
| `W2_A4_P35` | 2 | 4 | 0.35 | 10.71 | **91.48** | **22.02x** | 11.38x / 12.00x | **22.01x** | **0.3936** | 46.54 |

Full 16-run results and figures are available in [`results/sweep_summary.md`](results/sweep_summary.md) and [`report_final.md`](report_final.md).

### Installation

```bash
pip install -r requirements.txt
```

Verify all custom compression modules pass locally or on Colab:

```bash
python scripts/smoke_test.py
```

---

## Exact Reproduction Commands

### 1. Train FP32 MobileNet-v2 Baseline on CIFAR-10 (Question 1)

```bash
python -m src.train \
    --epochs 200 \
    --batch_size 128 \
    --lr 0.1 \
    --weight_decay 5e-4 \
    --warmup_epochs 5 \
    --label_smoothing 0.05 \
    --seed 42 \
    --save_dir ./checkpoints \
    --results_dir ./results
```

Outputs generated:
- `checkpoints/baseline_best.pth`
- `results/baseline_history.csv` & `results/baseline_curves.png`
- `results/confusion_matrix.png`, `results/failure_modes.png`, `results/baseline_eval.json`

### 2. Evaluate Baseline Independently (Optional)

```bash
python -m src.evaluate --checkpoint ./checkpoints/baseline_best.pth --output_dir ./results
```

### 3. Run Single Compression Configuration (`test.py`)

The trained FP32 baseline (`95.01%` Top-1) is included at `checkpoints/baseline_best.pth`, so you can run compression evaluation immediately after cloning.

8-bit weights and 8-bit activations (fast calibrated PTQ, ~95.04% Top-1):

```bash
python test.py --weight_quant_bits 8 --activation_quant_bits 8
```

Extended custom compression with structured pruning + QAT fine-tuning (e.g., `W3_A4_P45` — `92.89%` Top-1, `18.14x` model CR, `0.4776 MB`):

```bash
python test.py \
    --weight_quant_bits 3 \
    --activation_quant_bits 4 \
    --prune_ratio 0.45 \
    --prune_ft_epochs 15 \
    --qat_epochs 30 \
    --seed 42
```

### 4. Run Compression Sweep & Log to WandB (Questions 3 & 4)

Log in to WandB once (`wandb login`), then run the resume-safe 16-configuration sweep (add `--quick` for the 8-configuration subset):

```bash
python sweep.py \
    --checkpoint ./checkpoints/baseline_best.pth \
    --results_dir ./results \
    --resume \
    --wandb \
    --wandb_project mobilenetv2-cifar10-compression \
    --seed 42
```

Outputs generated:
- WandB runs logged with `activation_quant_bits`, `weight_quant_bits`, `prune_ratio`, `compression_ratio`, `weight_compression_ratio`, `activation_compression_ratio`, `model_size_mb`, and `quantized_acc` (ready to generate the **WandB Parallel Coordinates** plot).
- `results/sweep_results.csv` & `results/sweep_summary.md`
- `results/pareto_frontier.png` & `results/parallel_coordinates_local.png`
