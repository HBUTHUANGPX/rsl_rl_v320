from __future__ import annotations

import torch
import torch.nn as nn
from typing import Tuple, Sequence
"""
仍可能存在的差异/局限（不是 bug）

不支持 dim != codebook_dim 的 projection。
不支持多 codebook / channel‑first / pack‑unpack。
noise_dropout 只对 quantize() 生效，没有强制 float32 或 dtype 限制。
没有 allowed_dtypes / force_quantization_f32。
codes_to_indices() 使用 levels 作为 Tensor dtype，若输入为 half 精度可能会发生精度损失；通常不会出现问题。
"""
def _round_ste(x: torch.Tensor) -> torch.Tensor:
    """Straight-through estimator for rounding."""
    return (x.round() - x).detach() + x


def _floor_ste(x: torch.Tensor) -> torch.Tensor:
    """Straight-through estimator for flooring."""
    return (x.floor() - x).detach() + x


class FSQQuantizer(nn.Module):
    """
    Minimal Finite Scalar Quantization (FSQ) module.

    This implementation follows the core FSQ idea:
    - bound to a finite range with tanh
    - quantize with round + STE
    - normalize to [-1, 1]
    """

    def __init__(
        self,
        levels: Sequence[int],
        eps: float = 1e-3,
        noise_dropout: float = 0.0,
        preserve_symmetry: bool = False,
    ) -> None:
        super().__init__()
        if not levels:
            raise ValueError("levels must be non-empty.")
        if any(l <= 1 for l in levels):
            raise ValueError("all levels must be > 1.")

        self.levels = torch.tensor(levels, dtype=torch.int64)
        self.dim = int(self.levels.numel())
        self.eps = eps
        self.noise_dropout = noise_dropout
        self.preserve_symmetry = preserve_symmetry

        # Precompute per-dimension constants (register as buffers for device moves)
        half_l = (self.levels - 1) * (1.0 + eps) * 0.5
        # Even levels need half-step offset for symmetry
        offset = torch.where(
            self.levels % 2 == 0, torch.full_like(half_l, 0.5), torch.zeros_like(half_l)
        )
        half_width = self.levels // 2
        self.register_buffer("half_l", half_l.float())
        self.register_buffer("offset", offset.float())
        self.register_buffer("half_width", half_width.float())

    def bound(self, z: torch.Tensor) -> torch.Tensor:
        """
        Bound z to the valid quantization range per-dimension and normalize.
        Output is quantized and normalized to approximately [-1, 1].
        """
        if self.preserve_symmetry:
            # symmetry-preserving quantization (reference behavior)
            levels_minus_1 = self.levels.to(z.device).to(z.dtype) - 1
            scale = 2.0 / levels_minus_1
            bracket = (levels_minus_1 * (torch.tanh(z) + 1.0) / 2.0) + 0.5
            bracket = _floor_ste(bracket)
            return scale * bracket - 1.0
        # default FSQ bound (reference behavior)
        shift = torch.atanh(self.offset / self.half_l)
        bounded = torch.tanh(z + shift) * self.half_l - self.offset
        return _round_ste(bounded) / self.half_width

    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        """
        Quantize bounded z with STE rounding and normalize to [-1, 1].
        """
        z_q = self.bound(z)
        if not self.training or self.noise_dropout == 0.0:
            return z_q
        # optional noise dropout (reference behavior)
        mask = torch.bernoulli(torch.full_like(z_q, self.noise_dropout)).bool()
        offset = torch.rand_like(z_q) - 0.5
        return torch.where(mask, z_q + offset, z_q)

    def _scale_and_shift(self, codes: torch.Tensor) -> torch.Tensor:
        if self.preserve_symmetry:
            levels_minus_1 = self.levels.to(codes.device).to(codes.dtype) - 1
            return (codes + 1.0) / (2.0 / levels_minus_1)
        return (codes * self.half_width) + self.half_width

    def _scale_and_shift_inverse(self, zhat: torch.Tensor) -> torch.Tensor:
        if self.preserve_symmetry:
            levels_minus_1 = self.levels.to(zhat.device).to(zhat.dtype) - 1
            return zhat * (2.0 / levels_minus_1) - 1.0
        return (zhat - self.half_width) / self.half_width

    def codes_to_indices(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Map per-dimension codes in [-1, 1] to a single integer index.
        codes: (..., dim) in [-1, 1]
        """
        # convert back to integer bins [0, levels-1]
        z = self._scale_and_shift(codes)
        z = z.round()
        z = torch.maximum(z, torch.zeros_like(z))
        z = torch.minimum(z, (self.levels - 1).to(device=z.device, dtype=z.dtype))
        # compute mixed radix index
        basis = torch.cumprod(
            torch.cat(
                [
                    torch.ones(1, device=z.device, dtype=z.dtype),
                    self.levels[:-1].to(z.dtype),
                ]
            ),
            dim=0,
        )
        return (z * basis).sum(dim=-1).to(torch.int64)

    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Map integer indices back to per-dimension codes in [-1, 1].
        indices: (...,)
        """
        levels = self.levels.to(indices.device)
        basis = torch.cumprod(
            torch.cat([torch.ones(1, device=indices.device), levels[:-1]]), dim=0
        )
        rem = indices.to(torch.int64)
        basis = basis.to(torch.int64)
        levels = levels.to(torch.int64)
        z = (rem.unsqueeze(-1) // basis) % levels
        z = z.to(torch.float32)
        z = self._scale_and_shift_inverse(z)
        return z

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return quantized codes and indices.
        """
        z_q = self.quantize(z)
        indices = self.codes_to_indices(z_q)
        return z_q, indices
