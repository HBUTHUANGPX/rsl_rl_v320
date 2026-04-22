from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from rsl_rl.networks import MLP

from .quantizers import FSQQuantizer, IFSQuantizer


@dataclass
class DualFSQOutput:
    z_robot: torch.Tensor
    z_human: torch.Tensor
    q_robot: torch.Tensor
    q_human: torch.Tensor
    q_cycle: torch.Tensor
    robot_recon_from_robot: torch.Tensor
    robot_recon_from_human: torch.Tensor
    robot_aux: dict[str, torch.Tensor]
    human_aux: dict[str, torch.Tensor]
    cycle_aux: dict[str, torch.Tensor]


def build_dual_fsq_quantizer(
    quantizer_type: str = "ifsq",
    fsq_levels: int = 16,
    ifsq_boundary_fn: str = "sigmoid",
    ifsq_boundary_scale: float = 1.6,
) -> nn.Module:
    quantizer_type = quantizer_type.lower().strip()
    if quantizer_type == "fsq":
        return FSQQuantizer(levels=fsq_levels)
    if quantizer_type == "ifsq":
        return IFSQuantizer(
            levels=fsq_levels,
            boundary_fn=ifsq_boundary_fn,
            boundary_scale=ifsq_boundary_scale,
        )
    raise ValueError(f"不支持的量化器类型: {quantizer_type}")


class DualFSQAutoEncoder(nn.Module):
    """双编码器、共享量化器、单解码器的 FSQ 重构模块。"""

    def __init__(
        self,
        robot_input_dim: int,
        human_input_dim: int,
        latent_dim: int = 64,
        robot_encoder_hidden_dims: Sequence[int] = (512, 256),
        human_encoder_hidden_dims: Sequence[int] = (512, 256),
        decoder_hidden_dims: Sequence[int] = (256, 512),
        fsq_levels: int = 16,
        activation: str = "elu",
        quantizer_type: str = "ifsq",
        ifsq_boundary_fn: str = "sigmoid",
        ifsq_boundary_scale: float = 1.6,
        quantizer: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.robot_input_dim = int(robot_input_dim)
        self.human_input_dim = int(human_input_dim)
        self.latent_dim = int(latent_dim)
        self.embedding_dim = self.latent_dim

        self.robot_encoder = MLP(
            self.robot_input_dim,
            self.latent_dim,
            list(robot_encoder_hidden_dims),
            activation,
        )
        self.human_encoder = MLP(
            self.human_input_dim,
            self.latent_dim,
            list(human_encoder_hidden_dims),
            activation,
        )
        self.quantizer = quantizer or build_dual_fsq_quantizer(
            quantizer_type=quantizer_type,
            fsq_levels=fsq_levels,
            ifsq_boundary_fn=ifsq_boundary_fn,
            ifsq_boundary_scale=ifsq_boundary_scale,
        )
        self.decoder = MLP(
            self.latent_dim,
            self.robot_input_dim,
            list(decoder_hidden_dims),
            activation,
        )

    def encode_robot(self, robot_window: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        z_robot = self.robot_encoder(robot_window)
        aux = self.quantizer(z_robot)
        return z_robot, aux["z_q"], aux

    def encode_human(self, human_window: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        z_human = self.human_encoder(human_window)
        aux = self.quantizer(z_human)
        return z_human, aux["z_q"], aux

    def latent_from_robot(self, robot_window: torch.Tensor, detach: bool = False) -> torch.Tensor:
        _, q_robot, _ = self.encode_robot(robot_window)
        return q_robot.detach() if detach else q_robot

    def latent_from_human(self, human_window: torch.Tensor, detach: bool = False) -> torch.Tensor:
        _, q_human, _ = self.encode_human(human_window)
        return q_human.detach() if detach else q_human

    def decode_robot(self, quantized_latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(quantized_latent)

    def forward(self, robot_window: torch.Tensor, human_window: torch.Tensor) -> DualFSQOutput:
        z_robot, q_robot, robot_aux = self.encode_robot(robot_window)
        z_human, q_human, human_aux = self.encode_human(human_window)

        robot_recon_from_robot = self.decode_robot(q_robot)
        robot_recon_from_human = self.decode_robot(q_human)

        z_cycle = self.robot_encoder(robot_recon_from_human)
        cycle_aux = self.quantizer(z_cycle)

        return DualFSQOutput(
            z_robot=z_robot,
            z_human=z_human,
            q_robot=q_robot,
            q_human=q_human,
            q_cycle=cycle_aux["z_q"],
            robot_recon_from_robot=robot_recon_from_robot,
            robot_recon_from_human=robot_recon_from_human,
            robot_aux=robot_aux,
            human_aux=human_aux,
            cycle_aux=cycle_aux,
        )

    def compute_loss(
        self,
        robot_window: torch.Tensor,
        human_window: torch.Tensor,
        weights: dict[str, float] | None = None,
    ) -> tuple[torch.Tensor, DualFSQOutput, dict[str, torch.Tensor]]:
        weights = weights or {}
        output = self.forward(robot_window, human_window)
        terms = {
            "robot_recon": F.mse_loss(output.robot_recon_from_robot, robot_window),
            "human_recon": F.mse_loss(output.robot_recon_from_human, robot_window),
            "latent_align": F.mse_loss(output.q_human, output.q_robot),
            "cycle_latent": F.mse_loss(output.q_cycle, output.q_human),
        }
        loss = sum(
            float(weights.get(name, 1.0)) * value
            for name, value in terms.items()
        )
        return loss, output, terms
