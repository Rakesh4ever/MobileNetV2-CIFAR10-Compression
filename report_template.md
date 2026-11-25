# MobileNetV2-CIFAR10-Compression
## Custom In-Memory Model Compression Report (CS6886W — Systems Engineering for Deep Learning)

**Name:** `<Your Name>`  
**Roll Number:** `<Your Roll Number>`  
**GitHub Repository:** `<https://github.com/your-username/mobilenetv2-cifar10-compression>`

---

## Executive Summary (Mandatory Checklist)

| Metric | Reported Value |
| :--- | :--- |
| **Baseline FP32 Test Top-1 Accuracy (Uncompressed)** | `<e.g., 94.12%>` |
| **Baseline In-Memory Model Size (FP32)** | `<e.g., 8.64 MB>` |
| **Best Compressed Test Top-1 Accuracy (Selected Pareto Config)** | `<e.g., 92.45% (W4A4 + P40)>` |
| **Storage Overheads (Metadata, Scales, Biases, Act ZP/Scale)** | `<e.g., 41.2 KB (scales: X KB, biases: Y KB, act meta: Z KB, headers: W KB)>` |
| **Best Compression Ratio of the Model (In-Memory)** | `<e.g., 14.8x>` |
| **Best Compression Ratio of the Weights (In-Memory)** | `<e.g., 15.1x>` |
| **Best Compression Ratio of the Activations (In-Memory)** | `<e.g., 12.6x total / 11.8x peak (measured via forward hooks on 1×3×32×32 input)>` |
| **Final Approximated Model Size (MB) After Compression** | `<e.g., 0.58 MB>` |

---

## Question 1. Training Baseline (20 points)

### 1(a) CIFAR-10 Dataset Preparation, Normalization, and Augmentation
- **Dataset:** CIFAR-10 (50,000 training images, 10,000 test images, 10 classes, $3 \times 32 \times 32$ RGB).
- **Normalization:** Per-channel zero-mean/unit-variance using CIFAR-10 training statistics:
  - Mean: `(0.4914, 0.4822, 0.4465)`
  - Std: `(0.2470, 0.2435, 0.2616)`
- **Training Transforms:**
  1. `RandomCrop(32, padding=4, padding_mode='reflect')`
  2. `RandomHorizontalFlip(p=0.5)`
  3. `ToTensor()`
  4. `Normalize(mean, std)`
  5. `RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3))`
- **Test & Calibration Transforms:** `ToTensor()` followed by `Normalize(mean, std)`. A deterministic 1,024-sample subset of unaugmented training images (`seed=42`) is used for activation quantization calibration.

### 1(b) MobileNet-v2 Configuration & Training Strategy
- **Architecture Adaptations for $32 \times 32$ CIFAR-10:**
  - Width multiplier $\alpha = 1.0$, dropout $p = 0.2$, BatchNorm momentum $= 0.1$, $\epsilon = 10^{-5}$.
  - Stem `Conv2d` stride changed from `2` to `1` and stage-2 (`c=24`) stride changed from `2` to `1`, yielding 3 downsampling stages ($32 \to 16 \to 8 \to 4$) while preserving MobileNet-v2's exact 17 inverted residual blocks and ~2.24M parameter count.
- **Training Strategy:**
  - **Optimizer:** Nesterov SGD (`lr=0.1`, `momentum=0.9`, `weight_decay=5e-4`, with weight decay excluded on BatchNorm and bias parameters).
  - **LR Schedule:** 5-epoch linear warmup followed by Cosine Annealing down to $10^{-5}$ over `200` epochs.
  - **Regularization:** Label smoothing (`0.05`), dropout (`0.2`), reflection-padded random cropping, horizontal flipping, and random erasing.
  - **Batch Size & Seed:** `batch_size=128`, `seed=42`.

### 1(c) Final Test Accuracy, Training Curves, and Failure Modes
- **Final Test Top-1 Accuracy:** `<XX.XX%>` (Top-5: `<XX.XX%>`).
- *Insert `results/baseline_curves.png` here.*
- *Insert `results/confusion_matrix.png` and `results/failure_modes.png` here.*
- **Failure Mode Discussion:** `<Fill from results/baseline_eval.json — e.g., highest confusion occurs between visually similar fine-grained categories such as cat vs. dog, deer vs. horse, and automobile vs. truck due to low 32×32 spatial resolution and background/pose ambiguity.>`

---

## Question 2. Model Compression Implementation (30 points)

### 2(a) Configurable Weight & Activation Compression Design (Zero External APIs)
Our custom pure-PyTorch compression pipeline combines three complementary in-memory techniques:
1. **Custom Structured Expansion-Channel Pruning (`src/compression/prune.py`):**
   - MobileNet-v2 expands channels $6\times$ inside inverted residual blocks (`pw_expand -> dw_conv -> pw_project`), making expanded hidden feature maps the dominant source of both parameters and intermediate activation memory.
   - Channels are scored via coupled BN-$\gamma$ and incoming/outgoing filter L2 saliency and **physically sliced into smaller dense layers** (also pruning `head_conv` and matching `classifier` columns). Residual skip-connection dimensions remain untouched, requiring zero sparse-index metadata while shrinking both weights and activations in RAM/VRAM.
2. **Custom BatchNorm Folding (`src/compression/bn_fold.py`):**
   - Absorbs every `BatchNorm2d` ($\gamma, \beta, \mu, \sigma^2$) directly into preceding `Conv2d` weights and biases, removing all BN parameters/buffers from memory.
3. **Custom Weight & Activation Quantization with QAT + Self-Distillation (`src/compression/quantize.py`, `src/compress_pipeline.py`):**
   - **Weights:** Per-channel symmetric uniform quantization for arbitrary $b_w \in \{2..8\}$ bits with MSE-optimal clipping initialization and LSQ gradient-scaled learnable per-channel scales via Straight-Through Estimator (STE).
   - **Activations:** Per-tensor asymmetric uniform quantization for arbitrary $b_a \in \{2..8\}$ bits with EMA quantile-clipped calibration, exact integer zero-point nudging (preserving `0.0` for ReLU6), and learnable clipping bounds during QAT.
   - **In-Memory Verification (`src/compression/packing.py`):** Pure-PyTorch sub-byte bit-packer packs $b_w$-bit integers into contiguous `torch.uint8` tensors and verifies exact byte equality and lossless round-trip dequantization.

### 2(b) Application to MobileNet-v2 & Layer Exceptions
- **Compressed Layers:** All 17 `InvertedResidual` blocks (`pw_expand`, `dw_conv`, `pw_project`) and `head_conv` are quantized to `(weight_quant_bits, activation_quant_bits)` and structurally pruned (all 16 expanded blocks + `head_conv`).
- **Exceptions (`protect_first_last=True`):** The first 3-channel RGB `stem` `Conv2d` ($3 \times 32 \times 3 \times 3 = 864$ weights) and final 10-class `classifier` `Linear` layer are kept at 8-bit when target bit-width $< 8$. Together they represent $< 1\%$ of model weights while preventing severe accuracy loss on RGB inputs and final logits. (Configurable via `--no_protect_first_last`.)

### 2(c) Storage Overheads & Size Estimation
All compressed sizes strictly account for every byte held in RAM/VRAM:
- **Packed Weight Payload:** $\sum_{\ell} \lceil N_\ell \cdot b_{w,\ell} / 8 \rceil$ bytes (`uint8` packed buffers).
- **Per-Channel Weight Scales:** $2$ bytes (FP16) per output channel of every quantized Conv/Linear layer.
- **Layer Biases:** $2$ bytes (FP16) per output channel (resulting from BN folding).
- **Activation Quantization Metadata:** $4$ bytes per quantized activation tensor ($2$-byte FP16 scale + $2$-byte int16 zero-point).
- **Layer Metadata Headers:** $16$ bytes per quantized layer (shape, bit-width, offset metadata).
- *Insert exact overhead table from `results/sweep_results.csv` / `test.py` output.*

---

## Question 3. Compression Results

### 3(a) Compression Sweep Across Bit-Widths and Pruning Ratios
*Paste the Markdown table from `results/sweep_summary.md` covering all 16 configurations (minimum 8 required).*

### 3(b) WandB Parallel Coordinates Chart & Accuracy Comparison
- *Insert the exported **WandB Parallel Coordinates Chart** here (axes: `activation_quant_bits`, `weight_quant_bits`, `prune_ratio`, `compression_ratio`, `model_size_mb`, `quantized_acc`).*
- *Optionally insert `results/pareto_frontier.png` for additional clarity.*
- **Key Observations:** `<Discuss PTQ vs. QAT recovery on MobileNet-v2 depthwise convolutions, the sweet spot around W4A4/W3A4 + 30–45% structured pruning, and where accuracy starts degrading sharply (e.g., 2-bit weights / 3-bit activations).>`

---

## Question 4. Compression Analysis

### 4(a) Compression Ratio of the Weights
$$\text{CR}_{\text{weights}} = \frac{\text{Baseline FP32 (Weights + Biases + BN) Bytes}}{\text{Bit-Packed Weight Payload Bytes} + \text{Weight Scale, Bias \& Header Overhead Bytes}}$$
- Baseline FP32 weight/BN footprint: `<X.XXXX MB>`
- Best compressed weight + overhead footprint: `<X.XXXX MB>`
- **Weight Compression Ratio:** `<XX.XXx>`

### 4(b) Compression Ratio of the Activations & Measurement Methodology
- **Measurement Methodology:** PyTorch forward hooks are registered on every `Conv2d`/`QuantConv2d` and `Linear`/`QuantLinear` layer (`src/compression/metrics.py::measure_activation_memory`) during a forward pass with a single CIFAR-10 input `(1, 3, 32, 32)`. At each layer we record exact input/output element counts (`numel()`) and active bit-width (`32` for FP32 baseline; `act_bits` for quantized layers), summing bit-packed activation bytes plus `4` bytes per quantized activation tensor for `(scale, zero_point)`.
- **Total Activation Memory (across all layers, batch=1):** `<Baseline MB> -> <Compressed MB>` (**`<XX.XXx>`** compression).
- **Peak Working-Set Activation Memory (max concurrent input+output):** `<Baseline MB> -> <Compressed MB>` (**`<XX.XXx>`** compression).

### 4(v) Final Approximated Model Size (MB) After Compression
- **Baseline FP32 Model Size:** `<X.XXXX MB>`
- **Final Compressed Model Size (In-Memory RAM/VRAM, including all overheads):** **`<X.XXXX MB>`**
- **Overall Model Compression Ratio:** **`<XX.XXx>`**
- **Live Bit-Packing Verification:** Confirmed `PackedCompressedModel` live PyTorch tensor bytes match analytical size to the exact byte (`matches_analytical_bytes = True`, max dequantization round-trip error $< 10^{-5}$).

---

## Question 5. Reproducibility & Repository (10 points)

- **5(a) Modular Codebase:** Clean separation of data (`src/data.py`), model (`src/model.py`), training (`src/train.py`), evaluation (`src/evaluate.py`), custom compression (`src/compression/*`), CLI (`test.py`), and WandB sweep (`sweep.py`).
- **5(b) README & Seeds:** Full setup instructions, pinned dependencies (`requirements.txt`), fixed seed (`seed=42` across Python/NumPy/PyTorch/DataLoader), and copy-pasteable commands documented in `README.md`.
- **5(c) GitHub Repository Link:** `<https://github.com/your-username/mobilenetv2-cifar10-compression>`
