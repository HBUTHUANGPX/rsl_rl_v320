"""Quantization modules for VQ-VAE, FSQ, and iFSQ variants.

This module provides two quantizers:

1. ``VectorQuantizer``: classic VQ codebook quantization with commitment loss.
2. ``FSQQuantizer``: finite scalar quantization with straight-through gradients.
3. ``IFSQuantizer``: improved FSQ with configurable boundary mapping.
"""

from __future__ import annotations

from typing import Dict, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """Classic vector quantizer with a trainable codebook.

    The quantizer maps each encoder vector to its nearest embedding in a
    learnable codebook and returns straight-through quantized vectors.
    """

    def __init__(
        self, num_embeddings: int, embedding_dim: int, beta: float = 0.25
    ) -> None:
        """Initializes the vector quantizer.

        Args:
            num_embeddings: Number of codebook entries.
            embedding_dim: Dimensionality of each code vector.
            beta: Commitment loss weight.
        """
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.beta = float(beta)
        self.codebook = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.codebook.weight.data.uniform_(
            -1.0 / self.num_embeddings, 1.0 / self.num_embeddings
        )

    def forward(self, z_e: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Quantizes latent vectors with nearest-neighbor codebook lookup.

        Args:
            z_e: Encoder output with shape ``[B, D]``.

        Returns:
            A dictionary with:
                - ``z_q``: Straight-through quantized vectors ``[B, D]``.
                - ``indices``: Nearest codebook indices ``[B]``.
                - ``quant_loss``: VQ quantization loss scalar.
                - ``perplexity``: Codebook usage perplexity scalar.
        """
        distances = (
            torch.sum(z_e**2, dim=1, keepdim=True)
            + torch.sum(self.codebook.weight**2, dim=1)
            - 2 * z_e @ self.codebook.weight.t()
        )
        indices = torch.argmin(distances, dim=1)
        z_q = self.codebook(indices)

        codebook_loss = F.mse_loss(z_q, z_e.detach())
        commitment_loss = F.mse_loss(z_e, z_q.detach())
        quant_loss = codebook_loss + self.beta * commitment_loss
        z_q_st = z_e + (z_q - z_e).detach()

        one_hot = F.one_hot(indices, num_classes=self.num_embeddings).float()
        avg_probs = one_hot.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return {
            "z_q": z_q_st,
            "indices": indices,
            "quant_loss": quant_loss,
            "perplexity": perplexity,
        }


class FSQQuantizer(nn.Module):
    """Finite Scalar Quantizer (FSQ) with standard scalar rounding.

    FSQ discretizes each latent scalar independently into ``levels`` bins in
    ``[-1, 1]``. The module does not add VQ-style codebook/commitment losses.
    Training relies on reconstruction loss through straight-through estimation.
    """

    def __init__(self, levels: int | Iterable[int] = 8) -> None:
        """Initializes FSQ quantizer.

        Args:
            levels: Number of quantization bins per latent dimension. If an
                iterable is provided, only its first element is used.

        Raises:
            ValueError: If levels is smaller than 2.
        """
        super().__init__()
        if isinstance(levels, int):
            self.levels = int(levels)
        else:
            level_list = list(levels)
            if not level_list:
                raise ValueError("FSQ levels iterable must not be empty.")
            self.levels = int(level_list[0])
        if self.levels < 2:
            raise ValueError("FSQ levels must be >= 2.")

    def forward(self, z_e: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Quantizes latent vectors with per-dimension scalar bins.

        The implementation follows the standard FSQ idea:
        1. Bound continuous latents into ``[-1, 1]`` with ``tanh``.
        2. Convert values to level indices by scalar rounding.
        3. Map indices back to quantized values in ``[-1, 1]``.
        4. Apply straight-through estimation to preserve gradients.

        Args:
            z_e: Encoder output with shape ``[B, D]``.

        Returns:
            A dictionary with:
                - ``z_q``: Straight-through quantized vectors ``[B, D]``.
                - ``indices``: Per-dimension level indices ``[B, D]``.
                - ``level_histogram``: Global level usage histogram ``[levels]``.
                - ``per_dim_usage``: Per-dimension level usage ``[D, levels]``.
                - ``avg_utilization``: Percentage of used bins across all dims.
                - ``effective_bits``: Mean per-dim unique-level bits.
                - ``effective_bits_entropy``: Mean per-dim entropy bits.
        """
        if z_e.ndim != 2:
            raise ValueError(f"FSQ expects [B, D] input, got shape {tuple(z_e.shape)}.")

        # Bound latent activations before scalar quantization.
        z_bound = torch.tanh(z_e)
        level_scale = float(self.levels - 1)
        scaled = (z_bound + 1.0) * 0.5 * level_scale
        indices = torch.round(scaled).clamp(0, self.levels - 1).to(dtype=torch.long)
        z_q = (indices.to(z_e.dtype) / level_scale) * 2.0 - 1.0
        z_q_st = z_bound + (z_q - z_bound).detach()

        one_hot = F.one_hot(indices, num_classes=self.levels).to(
            dtype=z_e.dtype
        )  # [B, D, L]
        per_dim_usage = one_hot.mean(dim=0)  # [D, L]
        level_histogram = per_dim_usage.mean(dim=0)  # [L], averaged over latent dims
        used_mask = per_dim_usage > 1e-6
        avg_utilization = used_mask.to(dtype=z_e.dtype).mean() * 100.0
        unique_per_dim = used_mask.sum(dim=1).clamp(min=1).to(dtype=z_e.dtype)
        effective_bits = torch.log2(unique_per_dim).mean()
        per_dim_usage_norm = per_dim_usage / per_dim_usage.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-10)
        entropy = -torch.sum(
            per_dim_usage_norm * torch.log2(per_dim_usage_norm.clamp_min(1e-10)),
            dim=1,
        )
        effective_bits_entropy = entropy.mean()

        return {
            "z_q": z_q_st,
            "indices": indices,
            "level_histogram": level_histogram,
            "per_dim_usage": per_dim_usage,
            "avg_utilization": avg_utilization,
            "effective_bits": effective_bits,
            "effective_bits_entropy": effective_bits_entropy,
        }


class IFSQuantizer(FSQQuantizer):
    """Improved FSQ quantizer with configurable latent boundary function.

    iFSQ-style quantization keeps the same scalar rounding/discretization as FSQ
    but replaces the latent bounding transform to improve level utilization.
    """

    def __init__(
        self,
        levels: int | Iterable[int] = 8,
        boundary_fn: str = "sigmoid",
        boundary_scale: float = 1.6,
    ) -> None:
        """Initializes iFSQ quantizer.

        Args:
            levels: Number of quantization bins per latent dimension.
            boundary_fn: Boundary transform name in ``{"sigmoid", "tanh"}``.
            boundary_scale: Scaling factor applied before boundary transform.

        Raises:
            ValueError: If boundary function is unsupported.
        """
        super().__init__(levels=levels)
        norm_boundary = boundary_fn.lower().strip()
        if norm_boundary not in {"sigmoid", "tanh"}:
            raise ValueError(
                f"Unsupported boundary_fn: {boundary_fn}. Use 'sigmoid' or 'tanh'."
            )
        self.boundary_fn = norm_boundary
        self.boundary_scale = float(boundary_scale)

    def _bound_latent(self, z_e: torch.Tensor) -> torch.Tensor:
        """Maps latent values to ``[-1, 1]`` using configured boundary function.

        Args:
            z_e: Continuous encoder latent tensor ``[B, D]``.

        Returns:
            Bounded latent tensor in ``[-1, 1]``.
        """
        scaled = z_e * self.boundary_scale
        if self.boundary_fn == "sigmoid":
            return torch.sigmoid(scaled) * 2.0 - 1.0
        return torch.tanh(scaled)

    def forward(self, z_e: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Quantizes latent vectors with iFSQ boundary transform.

        Args:
            z_e: Encoder output with shape ``[B, D]``.

        Returns:
            Same output dictionary as :class:`FSQQuantizer`.
        """
        if z_e.ndim != 2:
            raise ValueError(
                f"iFSQ expects [B, D] input, got shape {tuple(z_e.shape)}."
            )

        z_bound = self._bound_latent(z_e)
        level_scale = float(self.levels - 1)
        scaled = (z_bound + 1.0) * 0.5 * level_scale
        indices = torch.round(scaled).clamp(0, self.levels - 1).to(dtype=torch.long)
        z_q = (indices.to(z_e.dtype) / level_scale) * 2.0 - 1.0
        z_q_st = z_bound + (z_q - z_bound).detach()

        one_hot = F.one_hot(indices, num_classes=self.levels).to(dtype=z_e.dtype)
        per_dim_usage = one_hot.mean(dim=0)
        level_histogram = per_dim_usage.mean(dim=0)
        used_mask = per_dim_usage > 1e-6
        avg_utilization = used_mask.to(dtype=z_e.dtype).mean() * 100.0
        unique_per_dim = used_mask.sum(dim=1).clamp(min=1).to(dtype=z_e.dtype)
        effective_bits = torch.log2(unique_per_dim).mean()
        per_dim_usage_norm = per_dim_usage / per_dim_usage.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-10)
        entropy = -torch.sum(
            per_dim_usage_norm * torch.log2(per_dim_usage_norm.clamp_min(1e-10)),
            dim=1,
        )
        effective_bits_entropy = entropy.mean()

        return {
            "z_q": z_q_st,
            "indices": indices,
            "level_histogram": level_histogram,
            "per_dim_usage": per_dim_usage,
            "avg_utilization": avg_utilization,
            "effective_bits": effective_bits,
            "effective_bits_entropy": effective_bits_entropy,
        }
