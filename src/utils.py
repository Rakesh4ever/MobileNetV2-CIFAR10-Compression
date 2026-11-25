"""Utility functions for reproducibility, device selection, and checkpointing."""

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Fix all random seeds across Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def get_device(preferred: Optional[str] = None) -> torch.device:
    """Return the best available compute device (CUDA > MPS > CPU)."""
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dir(path: str) -> Path:
    """Create a directory (and parents) if it does not exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: Dict[str, Any], path: str) -> None:
    """Save a dictionary as pretty-printed JSON."""
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path: str) -> Dict[str, Any]:
    """Load a JSON file into a dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class AverageMeter:
    """Tracks running average and current value of scalar metrics."""

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


def get_environment_info() -> Dict[str, str]:
    """Collect Python, PyTorch, and GPU metadata for reproducibility reports."""
    import platform
    import sys

    info = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": str(torch.cuda.is_available()),
        "cuda_version": str(torch.version.cuda),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU/MPS",
    }
    try:
        import torchvision

        info["torchvision_version"] = torchvision.__version__
    except ImportError:
        info["torchvision_version"] = "not_installed"
    return info
