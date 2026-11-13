"""LightGBM behind an import guard.

On macOS, PyTorch and LightGBM each load their own OpenMP runtime. Importing
LightGBM first hangs, and training LightGBM multi-threaded after PyTorch has
loaded crashes. The combination that works is: import torch first, then cap
LightGBM to one thread. Every use of LightGBM in this project goes through this
module so the order is guaranteed.
"""

from __future__ import annotations

import ctypes
import sys

import torch  # noqa: F401  (must be imported before lightgbm, see module docstring)
import lightgbm

if sys.platform == "darwin":
    lightgbm.basic._LIB.LGBM_SetMaxThreads(ctypes.c_int(1))

LGBMClassifier = lightgbm.LGBMClassifier

__all__ = ["LGBMClassifier", "lightgbm"]
