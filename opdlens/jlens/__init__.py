# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Jacobian lens: fit and apply the average input-output Jacobian as a readout
of decoder-transformer residuals."""

from opdlens.jlens._logging import configure_logging
from opdlens.jlens.fitting import fit, jacobian_for_prompt
from opdlens.jlens.hf import HFLensModel, Layout, from_hf
from opdlens.jlens.hooks import ActivationRecorder
from opdlens.jlens.lens import JacobianLens
from opdlens.jlens.protocol import LensModel

__all__ = [
    "ActivationRecorder",
    "HFLensModel",
    "JacobianLens",
    "Layout",
    "LensModel",
    "configure_logging",
    "fit",
    "from_hf",
    "jacobian_for_prompt",
]
