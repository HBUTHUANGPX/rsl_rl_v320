from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quantizers import FSQQuantizer


class FSQMLPEncoder(nn.Module):
    """Frame-level encoder for FSQ branches."""

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 32,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.embedding_dim),
        )

    def forward(self, encoder_input: torch.Tensor) -> torch.Tensor:
        return self.network(encoder_input)


class FSQMLPDecoder(nn.Module):
    """Frame-level decoder for FSQ reconstruction heads."""

    def __init__(
        self,
        latent_dim: int,
        target_dim: int,
        condition_dim: int = 0,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.target_dim = int(target_dim)
        self.condition_dim = int(condition_dim)
        self.hidden_dim = int(hidden_dim)
        self.network = nn.Sequential(
            nn.Linear(self.latent_dim + self.condition_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.target_dim),
        )

    def forward(
        self,
        latent: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if condition is not None:
            latent = torch.cat((latent, condition), dim=-1)
        return self.network(latent)


class FSQBranch(nn.Module):
    """Reusable FSQ branch composed of an encoder and a quantizer."""

    def __init__(
        self,
        encoder: FSQMLPEncoder,
        quantizer: nn.Module,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.quantizer = quantizer
        self.embedding_dim = encoder.embedding_dim
        self.encoder_input_dim = encoder.input_dim

    def encode(
        self,
        encoder_input: torch.Tensor,
        detach_quantized: bool = False,
    ) -> dict[str, torch.Tensor]:
        z_e = self.encoder(encoder_input)
        q_out = self.quantizer(z_e)
        if detach_quantized:
            q_out = dict(q_out)
            q_out["z_q"] = q_out["z_q"].detach()
        outputs = {"z_e": z_e}
        outputs.update(q_out)
        return outputs

    def latent_for_policy(
        self,
        encoder_input: torch.Tensor,
        detach: bool = True,
    ) -> torch.Tensor:
        return self.encode(
            encoder_input=encoder_input,
            detach_quantized=detach,
        )["z_q"]


class FSQReconstructionHead(nn.Module):
    """Decoder head that reconstructs a target from one or more FSQ latents."""

    def __init__(
        self,
        decoder: FSQMLPDecoder,
    ) -> None:
        super().__init__()
        self.decoder = decoder

    def decode(
        self,
        latents: list[torch.Tensor] | tuple[torch.Tensor, ...] | torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if isinstance(latents, torch.Tensor):
            latent = latents
        else:
            latent = torch.cat(tuple(latents), dim=-1)
        return self.decoder(latent, condition)

    def reconstruction_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return F.mse_loss(prediction, target, reduction="sum") / target.shape[0]


class FSQAutoEncoder(nn.Module):
    """Convenience wrapper for the common single-branch single-decoder case."""

    def __init__(
        self,
        encoder_input_dim: int,
        target_dim: int,
        decoder_condition_dim: int = 0,
        embedding_dim: int = 32,
        hidden_dim: int = 256,
        fsq_levels: int = 8,
        quantizer: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.encoder_input_dim = int(encoder_input_dim)
        self.target_dim = int(target_dim)
        self.decoder_condition_dim = int(decoder_condition_dim)
        self.embedding_dim = int(embedding_dim)
        self.branch = FSQBranch(
            encoder=FSQMLPEncoder(
                input_dim=self.encoder_input_dim,
                embedding_dim=self.embedding_dim,
                hidden_dim=hidden_dim,
            ),
            quantizer=quantizer if quantizer is not None else FSQQuantizer(levels=int(fsq_levels)),
        )
        self.decoder_head = FSQReconstructionHead(
            decoder=FSQMLPDecoder(
                latent_dim=self.embedding_dim,
                target_dim=self.target_dim,
                condition_dim=self.decoder_condition_dim,
                hidden_dim=hidden_dim,
            ),
        )

    @property
    def encoder(self) -> FSQMLPEncoder:
        return self.branch.encoder

    @property
    def quantizer(self) -> nn.Module:
        return self.branch.quantizer

    @property
    def decoder(self) -> FSQMLPDecoder:
        return self.decoder_head.decoder

    def forward(
        self,
        encoder_input: torch.Tensor,
        decoder_condition: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        outputs = self.branch.encode(encoder_input)
        outputs["x_hat"] = self.decoder_head.decode(outputs["z_q"], decoder_condition)
        return outputs

    def encoder_forward(
        self,
        encoder_input: torch.Tensor,
        detach_quantized: bool = False,
    ) -> dict[str, torch.Tensor]:
        return self.branch.encode(
            encoder_input=encoder_input,
            detach_quantized=detach_quantized,
        )

    def latent_for_policy(
        self,
        encoder_input: torch.Tensor,
        detach: bool = True,
    ) -> torch.Tensor:
        return self.branch.latent_for_policy(encoder_input, detach=detach)

    def loss_function(
        self,
        target: torch.Tensor,
        outputs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        recon = self.decoder_head.reconstruction_loss(outputs["x_hat"], target)
        loss_dict = {
            "loss": recon,
            "recon_loss": recon,
        }
        for key in (
            "effective_bits",
            "effective_bits_entropy",
            "avg_utilization",
            "level_histogram",
            "per_dim_usage",
            "indices",
            "z_q",
            "z_e",
            "x_hat",
        ):
            if key in outputs:
                loss_dict[key] = outputs[key]
        return loss_dict

    def compute_loss(
        self,
        target: torch.Tensor,
        decoder_condition: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        outputs = self.forward(target, decoder_condition=decoder_condition)
        return self.loss_function(target, outputs)

    @classmethod
    def from_config(cls, **kwargs: Any) -> "FSQAutoEncoder":
        """Small helper to mirror config-driven construction in policies."""
        return cls(**kwargs)
