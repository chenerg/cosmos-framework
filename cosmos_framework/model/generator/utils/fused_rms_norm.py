# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Device fused kernels for RMSNorm.

Ascend ``npu_rms_norm`` replaces the eager Cast/Pow/Mean/Rsqrt/Mul chain with
one kernel. CUDA / CPU callers get ``None`` and keep their original eager
formula.
"""

from __future__ import annotations

import torch


def npu_fused_rms_norm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor | None:
    """Run fused RMSNorm on NPU tensors, otherwise return ``None``.

    ``gamma`` must match ``hidden_states`` dtype (CANN constraint). FSDP may
    keep an fp32 master weight, so the scale is cast when needed.
    """
    if hidden_states.device.type != "npu":
        return None
    import torch_npu

    gamma = weight if weight.dtype == hidden_states.dtype else weight.to(dtype=hidden_states.dtype)
    return torch_npu.npu_rms_norm(hidden_states, gamma, epsilon=float(eps))[0]
