# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

"""Test configuration: pre-mock unavailable GPU/scientific dependencies.

The renderer package eagerly imports cv2, onnxruntime, tensorrt, kornia,
skimage, roma, etc. at import time. We pre-populate sys.modules with
MagicMock instances so these imports succeed on a CPU-only test machine.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock


def _mock_module(name: str) -> MagicMock:
    mod = MagicMock()
    mod.__name__ = name
    mod.__path__ = []
    mod.__file__ = f"<mock:{name}>"
    mod.__spec__ = None
    sys.modules[name] = mod
    return mod


_HEAVY_PACKAGES = [
    "cv2",
    "onnxruntime",
    "tensorrt",
    "kornia", "kornia.color",
    "skimage", "skimage.transform", "skimage.feature",
    "roma",
    "scipy", "scipy.linalg", "scipy.ndimage",
    "safetensors",
    "onnx",
    "PIL", "PIL.Image",
    "av",
    "decord",
    "anyio", "anyio.streams.memory",
    "fastapi", "fastapi.responses",
    "httpx",
    "tenacity",
    "uvicorn",
]

for _pkg in _HEAVY_PACKAGES:
    if _pkg not in sys.modules:
        _mock_module(_pkg)
