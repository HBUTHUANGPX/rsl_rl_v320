"""Frame-level VQ-VAE and FSQ-VAE models for motion reconstruction.

The module provides two quantized autoencoders:

1. ``FrameVQVAE``: classic vector-quantized model with codebook losses.
2. ``FrameFSQVAE``: finite scalar quantized model with reconstruction-only loss.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quantizers import FSQQuantizer, IFSQuantizer, VectorQuantizer


class _FrameQuantizedAutoencoderBase(nn.Module):
    """Shared building blocks for frame-level quantized autoencoders.

    This base class encapsulates encoder/decoder construction and reconstruction
    loss logic so VQ and FSQ variants can plug in different quantizers cleanly.
    """

    def __init__(
        self,
        encoder_input_dim: int,
        decoder_condition_dim: int,
        target_dim: int,
        embedding_dim: int = 32,
        hidden_dim: int = 256,
        recon_loss_mode: str = "mse",
    ) -> None:
        """Initializes common encoder/decoder architecture.

        Args:
            encoder_input_dim: Input feature dimension for encoder.
            decoder_condition_dim: Optional decoder condition feature dimension.
            target_dim: Output reconstruction feature dimension.
            embedding_dim: Latent embedding dimension before quantization.
            hidden_dim: Hidden width for encoder/decoder MLP.
            recon_loss_mode: Reconstruction loss mode, one of ``{"mse", "bce"}``.
        """
        super().__init__()
        self.encoder_input_dim = int(encoder_input_dim)
        self.decoder_condition_dim = int(decoder_condition_dim)
        self.target_dim = int(target_dim)
        self.embedding_dim = int(embedding_dim)
        self.recon_loss_mode = self._resolve_recon_loss_mode(recon_loss_mode)

        self.encoder = nn.Sequential(
            nn.Linear(self.encoder_input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self.embedding_dim),
        )

        decoder_in_dim = self.embedding_dim + self.decoder_condition_dim
        self.decoder = nn.Sequential(
            nn.Linear(decoder_in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self.target_dim),
        )

    @staticmethod
    def _resolve_recon_loss_mode(mode: str) -> str:
        """Validates and normalizes reconstruction loss mode.

        Args:
            mode: Input reconstruction mode text.

        Returns:
            Normalized mode in lowercase.

        Raises:
            ValueError: If mode is not supported.
        """
        norm = mode.lower().strip()
        if norm not in {"mse", "bce"}:
            raise ValueError(
                f"Unsupported reconstruction loss mode: {mode}. " "Use 'mse' or 'bce'."
            )
        return norm

    @staticmethod
    def _reconstruction_loss(
        x_hat: torch.Tensor,
        target: torch.Tensor,
        mode: str,
    ) -> torch.Tensor:
        """Computes reconstruction loss.

        Args:
            x_hat: Decoder reconstruction tensor.
            target: Ground-truth target tensor.
            mode: Loss mode, either ``"mse"`` or ``"bce"``.

        Returns:
            Batch-averaged reconstruction loss scalar.
        """
        if mode == "bce":
            return (
                F.binary_cross_entropy(x_hat, target, reduction="sum") / target.shape[0]
            )
        return F.mse_loss(x_hat, target, reduction="sum") / target.shape[0]

    def _forward_with_quantizer(
        self,
        encoder_input: torch.Tensor,
        decoder_condition: torch.Tensor,
        quantizer: nn.Module,
    ) -> Dict[str, torch.Tensor]:
        """Runs shared forward pipeline with a pluggable quantizer.

        Args:
            encoder_input: Encoder input tensor ``[B, D_enc]``.
            decoder_condition: Decoder condition tensor ``[B, D_cond]``.
            quantizer: Quantizer module implementing ``forward(z_e)``.

        Returns:
            Forward output dictionary with reconstruction and quantization terms.
        """
        z_e = self.encoder(encoder_input)
        q = quantizer(z_e)
        decoder_input = torch.cat([q["z_q"], decoder_condition], dim=1)
        x_hat = self.decoder(decoder_input)
        output = {
            "x_hat": x_hat,
            "z_e": z_e,
        }
        # Merge quantizer-specific outputs so VQ and FSQ can expose different metrics.
        output.update(q)
        return output


class FrameVQVAE(_FrameQuantizedAutoencoderBase):
    """Vector-quantized autoencoder with conditional decoder."""

    def __init__(
        self,
        encoder_input_dim: int,
        decoder_condition_dim: int,
        target_dim: int,
        embedding_dim: int = 32,
        hidden_dim: int = 256,
        num_embeddings: int = 512,
        beta: float = 0.25,
        recon_loss_mode: str = "mse",
    ) -> None:
        """Initializes conditional frame-level VQ-VAE.

        Args:
            encoder_input_dim: Encoder input feature dimension.
            decoder_condition_dim: Decoder condition feature dimension.
            target_dim: Reconstruction target dimension.
            embedding_dim: Latent embedding dimension.
            hidden_dim: Hidden MLP width.
            num_embeddings: Number of VQ codebook vectors.
            beta: VQ commitment loss weight.
            recon_loss_mode: Reconstruction mode in ``{"mse", "bce"}``.
        """
        super().__init__(
            encoder_input_dim=encoder_input_dim,
            decoder_condition_dim=decoder_condition_dim,
            target_dim=target_dim,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            recon_loss_mode=recon_loss_mode,
        )
        self.quantizer = VectorQuantizer(
            num_embeddings=int(num_embeddings),
            embedding_dim=self.embedding_dim,
            beta=float(beta),
        )

    def forward(
        self,
        encoder_input: torch.Tensor,
        decoder_condition: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Runs VQ-VAE forward pass.

        Args:
            encoder_input: Encoder input tensor with shape ``[B, D_enc]``.
            decoder_condition: Decoder condition tensor with shape ``[B, D_cond]``.

        Returns:
            Dictionary with reconstruction, indices, quantization loss, and perplexity.
        """
        return self._forward_with_quantizer(
            encoder_input=encoder_input,
            decoder_condition=decoder_condition,
            quantizer=self.quantizer,
        )

    def loss_function(
        self,
        target: torch.Tensor,
        outputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Computes VQ-VAE losses.

        Args:
            target: Reconstruction target tensor.
            outputs: Forward output dictionary.

        Returns:
            Dictionary containing total, reconstruction, quantization losses,
            and perplexity metric.
        """
        recon = self._reconstruction_loss(
            outputs["x_hat"], target, self.recon_loss_mode
        )
        quant = outputs["quant_loss"]
        total = recon + quant
        return {
            "loss": total,
            "recon_loss": recon,
            "quant_loss": quant,
            "perplexity": outputs["perplexity"],
        }


class FrameFSQVAE(_FrameQuantizedAutoencoderBase):
    """Finite-scalar-quantized autoencoder with conditional decoder.

    FSQ variant uses reconstruction loss as the only training objective and does
    not add VQ-style commitment/codebook auxiliary losses.
    """

    def __init__(
        self,
        encoder_input_dim: int,
        decoder_condition_dim: int,
        target_dim: int,
        embedding_dim: int = 32,
        hidden_dim: int = 256,
        fsq_levels: int = 8,
        recon_loss_mode: str = "mse",
    ) -> None:
        """Initializes conditional frame-level FSQ-VAE.

        Args:
            encoder_input_dim: Encoder input feature dimension.
            decoder_condition_dim: Decoder condition feature dimension.
            target_dim: Reconstruction target dimension.
            embedding_dim: Latent embedding dimension.
            hidden_dim: Hidden MLP width.
            fsq_levels: Number of scalar quantization bins.
            recon_loss_mode: Reconstruction mode in ``{"mse", "bce"}``.
        """
        super().__init__(
            encoder_input_dim=encoder_input_dim,
            decoder_condition_dim=decoder_condition_dim,
            target_dim=target_dim,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            recon_loss_mode=recon_loss_mode,
        )
        self.quantizer = FSQQuantizer(levels=int(fsq_levels))

    def forward(
        self,
        encoder_input: torch.Tensor,
        decoder_condition: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Runs FSQ-VAE forward pass.

        Args:
            encoder_input: Encoder input tensor with shape ``[B, D_enc]``.
            decoder_condition: Decoder condition tensor with shape ``[B, D_cond]``.

        Returns:
            Dictionary with reconstruction, scalar-level indices, and metrics.
        """
        return self._forward_with_quantizer(
            encoder_input=encoder_input,
            decoder_condition=decoder_condition,
            quantizer=self.quantizer,
        )

    def loss_function(
        self,
        target: torch.Tensor,
        outputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Computes FSQ-VAE losses.

        Args:
            target: Reconstruction target tensor.
            outputs: Forward output dictionary.

        Returns:
            Dictionary containing total and reconstruction losses. ``quant_loss``
            is intentionally absent for FSQ.
        """
        recon = self._reconstruction_loss(
            outputs["x_hat"], target, self.recon_loss_mode
        )
        return {
            "loss": recon,
            "recon_loss": recon,
            "effective_bits": outputs["effective_bits"],
            "effective_bits_entropy": outputs["effective_bits_entropy"],
            "avg_utilization": outputs["avg_utilization"],
            "level_histogram": outputs["level_histogram"],
            "per_dim_usage": outputs["per_dim_usage"],
        }


class FrameIFSQVAE(_FrameQuantizedAutoencoderBase):
    """Improved finite-scalar-quantized autoencoder with configurable boundary."""

    def __init__(
        self,
        encoder_input_dim: int,
        decoder_condition_dim: int,
        target_dim: int,
        embedding_dim: int = 32,
        hidden_dim: int = 256,
        fsq_levels: int = 8,
        boundary_fn: str = "sigmoid",
        boundary_scale: float = 1.6,
        recon_loss_mode: str = "mse",
    ) -> None:
        """Initializes frame-level iFSQ-VAE.

        Args:
            encoder_input_dim: Encoder input feature dimension.
            decoder_condition_dim: Decoder condition feature dimension.
            target_dim: Reconstruction target dimension.
            embedding_dim: Latent embedding dimension.
            hidden_dim: Hidden MLP width.
            fsq_levels: Number of scalar quantization bins.
            boundary_fn: iFSQ boundary function in ``{"sigmoid", "tanh"}``.
            boundary_scale: Scaling factor before boundary mapping.
            recon_loss_mode: Reconstruction mode in ``{"mse", "bce"}``.
        """
        super().__init__(
            encoder_input_dim=encoder_input_dim,
            decoder_condition_dim=decoder_condition_dim,
            target_dim=target_dim,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            recon_loss_mode=recon_loss_mode,
        )
        self.quantizer = IFSQuantizer(
            levels=int(fsq_levels),
            boundary_fn=boundary_fn,
            boundary_scale=float(boundary_scale),
        )

    def forward(
        self,
        encoder_input: torch.Tensor,
        decoder_condition: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Runs iFSQ-VAE forward pass."""
        return self._forward_with_quantizer(
            encoder_input=encoder_input,
            decoder_condition=decoder_condition,
            quantizer=self.quantizer,
        )

    def loss_function(
        self,
        target: torch.Tensor,
        outputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Computes iFSQ-VAE losses (reconstruction-only objective)."""
        recon = self._reconstruction_loss(
            outputs["x_hat"], target, self.recon_loss_mode
        )
        return {
            "loss": recon,
            "recon_loss": recon,
            "effective_bits": outputs["effective_bits"],
            "effective_bits_entropy": outputs["effective_bits_entropy"],
            "avg_utilization": outputs["avg_utilization"],
            "level_histogram": outputs["level_histogram"],
            "per_dim_usage": outputs["per_dim_usage"],
        }
