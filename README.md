# MobileNetV2-CIFAR10-Compression
## Custom In-Memory Model Compression for MobileNet-v2 on CIFAR-10

*CS6886W — Systems Engineering for Deep Learning (IIT Madras)*

This repository implements **MobileNet-v2** trained on **CIFAR-10** and a **from-scratch, pure-PyTorch in-memory model compression pipeline** (zero external compression APIs such as `torch.ao.quantization`, `torch.nn.utils.prune`, or `bitsandbytes`).

Compression is strictly evaluated on **in-memory RAM/VRAM footprint** (weights, activations, and explicit metadata/scaling overheads), verified with actual pure-PyTorch sub-byte bit-packing (`uint8` tensors).

---

## Repository Structure

```text
mobilenetv2-cifar10-compression/
├── README.md                        # Setup, seed config, and exact reproduction commands
├── requirements.txt                 # Python dependencies
├── colab_runner.ipynb               # End-to-end Google Colab GPU notebook
├── report_template.md               # Structured report template for PDF submission
├── test.py                          # CLI entrypoint (`--weight_quant_bits`, `--activation_quant_bits`, ...)
├── sweep.py                         # 8- or 16-run compression sweep with WandB Parallel Coordinates logging
├── scripts/
│   └── smoke_test.py                # Fast synthetic-tensor verification of all custom modules
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

- **Recommended Runtime:** Google Colab GPU (NVIDIA T4 / L4 / A100), Python 3.10+, PyTorch 2.1+
- **Random Seed Configuration:** Default `--seed 42` fixes `random`, `numpy`, `torch.manual_seed`, `torch.cuda.manual_seed_all`, `cudnn.deterministic = True`, `cudnn.benchmark = False`, and PyTorch `DataLoader` generators.

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

### 3. Run Single Compression Configuration (`test.py` — Question 2)

Standard assignment CLI pattern (8-bit weights, 8-bit activations):

```bash
python test.py --weight_quant_bits 8 --activation_quant_bits 8 --checkpoint ./checkpoints/baseline_best.pth
```

Extended custom compression (40% structured pruning + 4-bit weights + 4-bit activations + QAT):

```bash
python test.py \
    --weight_quant_bits 4 \
    --activation_quant_bits 4 \
    --prune_ratio 0.40 \
    --qat_epochs 25 \
    --prune_ft_epochs 15 \
    --checkpoint ./checkpoints/baseline_best.pth \
    --seed 42 \
    --save_json ./results/w4a4_p40_report.json
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
