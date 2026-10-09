#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Deterministic seeding helpers for reproducible model inference."""

from __future__ import annotations

import os
import random


def set_seed(seed: int = 42) -> None:
    """Set global RNG seeds for Python, NumPy, PyTorch, and sklearn.

    Args:
        seed: Integer seed. Defaults to 42 to match RunContext.
    """
    random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))

    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        pass
    else:
        np.random.seed(seed)

    try:
        import torch
    except ImportError:  # pragma: no cover
        pass
    else:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    try:
        import tensorflow as tf
    except ImportError:  # pragma: no cover
        pass
    else:
        tf.random.set_seed(seed)
