# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU + optional NPU tests for fused RMSNorm."""

import pytest
import torch

from cosmos_framework.model.generator.reasoner.nemotron_3_dense_vl.nemotron_3_dense_vl import (
    Nemotron3DenseVLRMSNorm,
)
from cosmos_framework.model.generator.utils.fused_rms_norm import npu_fused_rms_norm


def _eager_nemotron_rms_norm(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    hidden = hidden_states.float()
    variance = hidden.pow(2).mean(-1, keepdim=True)
    hidden = hidden * torch.rsqrt(variance + eps)
    return (weight.float() * hidden).to(hidden_states.dtype)


def _npu_available() -> bool:
    try:
        import torch_npu  # noqa: F401

        return bool(torch.npu.is_available())
    except Exception:
        return False


def test_npu_fused_rms_norm_returns_none_on_cpu() -> None:
    hidden = torch.randn(4, 8)
    weight = torch.ones(8)
    assert npu_fused_rms_norm(hidden, weight, 1e-5) is None


def test_nemotron_rms_norm_cpu_matches_reference() -> None:
    torch.manual_seed(0)
    eps = 1e-5
    mod = Nemotron3DenseVLRMSNorm(16, eps=eps)
    hidden = torch.randn(7, 16)
    out = mod(hidden)
    ref = _eager_nemotron_rms_norm(hidden, mod.weight, eps)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


@pytest.mark.skipif(not _npu_available(), reason="NPU required")
def test_nemotron_rms_norm_npu_close_to_eager() -> None:
    torch.manual_seed(0)
    device = torch.device("npu:0")
    eps = 1e-5
    mod = Nemotron3DenseVLRMSNorm(64, eps=eps).to(device=device, dtype=torch.bfloat16)
    hidden = torch.randn(32, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
    fused = mod(hidden)
    eager = _eager_nemotron_rms_norm(hidden.detach(), mod.weight.detach(), eps)
    diff = (fused.float() - eager.float()).abs()
    assert float(diff.mean()) < 1e-3
    assert float(diff.max()) < 5e-2
    fused.square().mean().backward()
    assert hidden.grad is not None
    assert mod.weight.grad is not None
