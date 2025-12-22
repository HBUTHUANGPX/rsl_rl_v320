# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal
from typing import Any, NoReturn, Tuple

from rsl_rl.networks import MLP, EmpiricalNormalization, HiddenState


class StudentTeacher_CVAE(nn.Module):
    """
    CVAE-based Student-Teacher 模块，基于论文第二阶段设计。
    - Teacher: 标准 MLP，使用特权观测生成标签动作。
    - Student (CVAE): 包括 encoder (后验，使用 teacher obs)、prior (先验，使用 policy obs)、decoder (使用 policy obs + latent 生成动作)。
    - 在蒸馏训练中，使用重参数化采样后验 latent，计算重构损失 (MSE) 和 KL 损失。
    - 在推理中，使用 prior 采样 latent。
    """

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        student_obs_normalization: bool = False,
        teacher_obs_normalization: bool = False,
        encoder_hidden_dims: tuple[int] | list[int] = [256, 256, 256],  # Encoder 隐藏层
        prior_hidden_dims: tuple[int] | list[int] = [256, 256, 256],  # Prior 隐藏层
        decoder_hidden_dims: tuple[int] | list[int] = [256, 256, 256],  # Decoder 隐藏层
        teacher_hidden_dims: tuple[int] | list[int] = [
            256,
            256,
            256,
        ],  # Teacher MLP 隐藏层
        latent_dim: int = 8,  # 潜在维度
        kl_weight: float = 1.0,  # KL 权重（在自定义损失中使用）
        activation: str = "elu",
        init_noise_std: float = 0.1,
        noise_std_type: str = "scalar",
        **kwargs: dict[str, Any],
    ) -> None:
        """
        初始化 CVAE 组件。
        - Encoder: 输入 num_teacher_obs，输出 2 * latent_dim (mu + logvar)。
        - Prior: 输入 num_student_obs，输出 2 * latent_dim (mu + logvar)。
        - Decoder: 输入 num_student_obs + latent_dim，输出 num_actions。
        - Teacher: 与原类相同。
        """
        if kwargs:
            print(
                "StudentTeacher_CVAE.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs])
            )
        super().__init__()

        self.loaded_teacher = False  # 表示教师是否已加载
        self.kl_weight = kl_weight  # KL 损失权重

        # 获取观测维度
        self.obs_groups = obs_groups
        num_student_obs = 0
        for obs_group in obs_groups["policy"]:
            assert len(obs[obs_group].shape) == 2, "仅支持 1D 观测。"
            num_student_obs += obs[obs_group].shape[-1]
        num_teacher_obs = 0
        for obs_group in obs_groups["teacher"]:
            assert len(obs[obs_group].shape) == 2, "仅支持 1D 观测。"
            num_teacher_obs += obs[obs_group].shape[-1]

        # CVAE 组件
        self.encoder = MLP(
            num_teacher_obs, 2 * latent_dim, encoder_hidden_dims, activation
        )  # 输出 mu + logvar
        print(f"CVAE Encoder: {self.encoder}")
        self.prior = MLP(
            num_student_obs, 2 * latent_dim, prior_hidden_dims, activation
        )  # 输出 mu + logvar
        print(f"CVAE Prior: {self.prior}")
        self.decoder = MLP(
            num_student_obs + latent_dim, num_actions, decoder_hidden_dims, activation
        )
        print(f"CVAE Decoder: {self.decoder}")

        # 观测归一化（继承原类）
        self.student_obs_normalization = student_obs_normalization
        if student_obs_normalization:
            self.student_obs_normalizer = EmpiricalNormalization(num_student_obs)
        else:
            self.student_obs_normalizer = torch.nn.Identity()

        # Teacher MLP（继承原类）
        self.teacher = MLP(
            num_teacher_obs, num_actions, teacher_hidden_dims, activation
        )
        print(f"Teacher MLP: {self.teacher}")

        self.teacher_obs_normalization = teacher_obs_normalization
        if teacher_obs_normalization:
            self.teacher_obs_normalizer = EmpiricalNormalization(num_teacher_obs)
        else:
            self.teacher_obs_normalizer = torch.nn.Identity()

        # 动作噪声（继承原类）
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(
                torch.log(init_noise_std * torch.ones(num_actions))
            )
        else:
            raise ValueError(
                f"未知标准差类型: {self.noise_std_type}。应为 'scalar' 或 'log'"
            )

        # 动作分布（在 update_distribution 中填充）
        self.distribution = None

        # 禁用 Normal 验证以加速
        Normal.set_default_validate_args(False)

        self.latent_dim = latent_dim  # 潜在维度

    def reset(
        self,
        dones: torch.Tensor | None = None,
        hidden_states: tuple[HiddenState, HiddenState] = (None, None),
    ) -> None:
        """重置模块（继承原类，无需修改）。"""
        pass

    def forward(self) -> NoReturn:
        raise NotImplementedError

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def _sample_latent(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        重参数化采样 latent z = mu + std * epsilon, 其中 epsilon ~ N(0,1)。
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _get_posterior(
        self, teacher_obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算后验分布：encoder 输出 mu, logvar；采样 z。
        """
        enc_out = self.encoder(teacher_obs)
        mu = enc_out[..., : self.latent_dim]
        logvar = enc_out[..., self.latent_dim :]
        z = self._sample_latent(mu, logvar)
        return z, mu, logvar

    def _get_prior(
        self, student_obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算先验分布：prior 输出 mu, logvar；采样 z。
        """
        prior_out = self.prior(student_obs)
        mu = prior_out[..., : self.latent_dim]
        logvar = prior_out[..., self.latent_dim :]
        z = self._sample_latent(mu, logvar)
        return z, mu, logvar

    def _update_distribution(self, obs: TensorDict, use_prior: bool = True) -> None:
        """
        更新动作分布。
        - 使用 policy obs + latent 生成 mean。
        - 若 use_prior=True（推理），使用 prior；否则使用 posterior（训练）。
        """
        student_obs = self.get_student_obs(obs)
        student_obs = self.student_obs_normalizer(student_obs)

        if use_prior:
            z, _, _ = self._get_prior(student_obs)  # 先验采样
        else:
            teacher_obs = self.get_teacher_obs(obs)
            teacher_obs = self.teacher_obs_normalizer(teacher_obs)
            z, _, _ = self._get_posterior(teacher_obs)  # 后验采样（训练时）

        # Decoder 输入：policy obs + z
        decoder_input = torch.cat([student_obs, z], dim=-1)
        mean = self.decoder(decoder_input)

        # 计算 std（继承原类）
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"未知标准差类型: {self.noise_std_type}。")

        self.distribution = Normal(mean, std)

    def act(self, obs: TensorDict) -> torch.Tensor:
        """
        生成动作：使用 prior 采样 latent，然后更新分布并采样。
        """
        self._update_distribution(obs, use_prior=True)  # 使用 prior（部署模式）
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        """
        确定性推理：使用 prior 的均值 latent，然后 decoder 输出 mean。
        """
        student_obs = self.get_student_obs(obs)
        student_obs = self.student_obs_normalizer(student_obs)
        prior_out = self.prior(student_obs)
        mu_prior = prior_out[
            ..., : self.latent_dim
        ]  # 使用 prior mu 作为 deterministic z
        decoder_input = torch.cat([student_obs, mu_prior], dim=-1)
        return self.decoder(decoder_input)

    def evaluate(self, obs: TensorDict) -> torch.Tensor:
        """
        教师评估：使用 teacher obs 生成标签动作（继承原类）。
        """
        obs = self.get_teacher_obs(obs)
        obs = self.teacher_obs_normalizer(obs)
        with torch.no_grad():
            return self.teacher(obs)

    def get_student_obs(self, obs: TensorDict) -> torch.Tensor:
        """获取 policy obs（继承原类）。"""
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["policy"]]
        return torch.cat(obs_list, dim=-1)

    def get_teacher_obs(self, obs: TensorDict) -> torch.Tensor:
        """获取 teacher obs（继承原类）。"""
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["teacher"]]
        return torch.cat(obs_list, dim=-1)

    def get_hidden_states(self) -> tuple[HiddenState, HiddenState]:
        """获取隐藏状态（继承原类）。"""
        return None, None

    def detach_hidden_states(self, dones: torch.Tensor | None = None) -> None:
        """分离隐藏状态（继承原类）。"""
        pass

    def train(self, mode: bool = True) -> None:
        """训练模式：教师保持 eval（继承原类）。"""
        super().train(mode)
        self.teacher.eval()
        self.teacher_obs_normalizer.eval()

    def update_normalization(self, obs: TensorDict) -> None:
        """更新归一化（继承原类）。"""
        if self.student_obs_normalization:
            student_obs = self.get_student_obs(obs)
            self.student_obs_normalizer.update(student_obs)

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        """
        加载状态字典：支持教师或完整 CVAE 参数（扩展原类）。
        """
        # 检查并加载教师参数（继承原类）
        if any("actor" in key for key in state_dict):  # 从 RL 训练加载
            teacher_state_dict = {}
            teacher_obs_normalizer_state_dict = {}
            for key, value in state_dict.items():
                if "actor." in key:
                    teacher_state_dict[key.replace("actor.", "")] = value
                if "actor_obs_normalizer." in key:
                    teacher_obs_normalizer_state_dict[
                        key.replace("actor_obs_normalizer.", "")
                    ] = value
            self.teacher.load_state_dict(teacher_state_dict, strict=strict)
            self.teacher_obs_normalizer.load_state_dict(
                teacher_obs_normalizer_state_dict, strict=strict
            )
            self.loaded_teacher = True
            self.teacher.eval()
            self.teacher_obs_normalizer.eval()
            return False  # 非恢复训练

        # 加载 CVAE 参数（新增）
        elif any(
            "encoder" in key or "prior" in key or "decoder" in key for key in state_dict
        ):
            super().load_state_dict(state_dict, strict=strict)
            self.loaded_teacher = True
            self.teacher.eval()
            self.teacher_obs_normalizer.eval()
            return True  # 恢复训练

        else:
            raise ValueError("state_dict 不包含教师或 CVAE 参数")

    # 新增：计算 KL 损失（在 Distillation.update 中调用，若需自定义）
    def compute_kl_loss(
        self,
        post_mu: torch.Tensor,
        post_logvar: torch.Tensor,
        prior_mu: torch.Tensor,
        prior_logvar: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算 KL 散度：D_{KL}(q(z|s_t^{priv}) || p(z|s_t^{dep}))。
        公式：\frac{1}{2} \sum (\log \sigma_p^2 - \log \sigma_q^2 + \frac{\sigma_q^2 + (\mu_q - \mu_p)^2}{\sigma_p^2} - 1)。
        """
        kl = -0.5 * torch.sum(
            1
            + post_logvar
            - prior_logvar
            - (torch.exp(post_logvar) + (post_mu - prior_mu) ** 2)
            / torch.exp(prior_logvar),
            dim=-1,
        )
        return self.kl_weight * kl.mean()
