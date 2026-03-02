"""
Qwen3 RMSNorm CUDA Kernels — RTX 5090 Blackwell (sm_100)

Vectorized RMSNorm kernel optimized for Qwen3-8B:
  - hidden_size=4096 (layer norms)
  - head_dim=128 (QK norms)
  - eps=1e-6

Usage:
    from qwen3_kernels import rmsnorm

    output = rmsnorm(hidden_states, weight, eps=1e-6)
"""

from typing import Optional

import torch

from qwen3_kernels._C import rmsnorm_forward


def rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Vectorized RMSNorm: y = (x / sqrt(mean(x^2) + eps)) * weight

    Args:
        input:  [..., hidden_size] tensor (bf16, fp16, or fp32)
        weight: [hidden_size] tensor (same dtype as input)
        eps:    epsilon for numerical stability (default: 1e-6)
        out:    optional pre-allocated output tensor

    Returns:
        Normalized tensor with same shape and dtype as input.
    """
    if out is None:
        out = torch.empty_like(input)

    rmsnorm_forward(out, input.contiguous(), weight.contiguous(), eps)
    return out
