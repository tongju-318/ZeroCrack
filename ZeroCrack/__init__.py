# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import importlib.util
import sys
from pathlib import Path

sys.modules.setdefault("zerocrack", sys.modules[__name__])


def build_sam3_image_model(*args, **kwargs):
    from .model_builder import build_sam3_image_model as _build_sam3_image_model

    return _build_sam3_image_model(*args, **kwargs)

_MODEL_FILE = Path(__file__).resolve().parent.parent / "ZeroCrack.py"
_MODEL_SPEC = importlib.util.spec_from_file_location("_zerocrack_model", _MODEL_FILE)
_MODEL_MODULE = importlib.util.module_from_spec(_MODEL_SPEC)
_MODEL_SPEC.loader.exec_module(_MODEL_MODULE)
ZeroCrack = _MODEL_MODULE.ZeroCrack

__version__ = "0.1.0"

__all__ = ["build_sam3_image_model", "ZeroCrack"]
