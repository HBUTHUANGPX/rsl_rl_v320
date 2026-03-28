# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal
from typing import Any, NoReturn

from rsl_rl.networks import MLP, EmpiricalNormalization
from rsl_rl.modules.vqvae import FrameFSQVAE
import os
import onnx
import copy
from rsl_rl.modules import ActorCriticSingleFSQ

class ActorCriticSingleFSQDistillation(ActorCriticSingleFSQ,nn.Module):
    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        student_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        teacher_obs_normalization: bool = False, 
        student_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        critic_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        teacher_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs])
            )
        nn.Module.__init__(self)

        # Get the observation dimensions
        self.obs_groups = obs_groups
        num_student_obs = 0
        for obs_group in obs_groups["policy"]:
            assert (
                len(obs[obs_group].shape) == 2
            ), "The ActorCritic module only supports 1D observations."
            num_student_obs += obs[obs_group].shape[-1]
        num_critic_obs = 0
        for obs_group in obs_groups["critic"]:
            assert (
                len(obs[obs_group].shape) == 2
            ), "The ActorCritic module only supports 1D observations."
            num_critic_obs += obs[obs_group].shape[-1]
        num_student_fsq_obs = 0
        for obs_group in obs_groups["policy_window"]:
            assert (
                len(obs[obs_group].shape) == 2
            ), "The FSQ module only supports 1D observations."
            num_student_fsq_obs += obs[obs_group].shape[-1]
        num_critic_fsq_obs = 0
        for obs_group in obs_groups["critic_window"]:
            assert (
                len(obs[obs_group].shape) == 2
            ), "The FSQ module only supports 1D observations."
            num_critic_fsq_obs += obs[obs_group].shape[-1]
        num_teacher_obs = 0
        for obs_group in obs_groups["teacher"]:
            assert (
                len(obs[obs_group].shape) == 2
            ), "The Teacher module only supports 1D observations."
            num_teacher_obs += obs[obs_group].shape[-1]
        # Actor FSQ
        self.fsq_embedding_dim = 32
        self.fsq_hidden_dim = 256
        self.fsq_levels = 16
        self.student_fsq = FrameFSQVAE(
            encoder_input_dim=num_student_fsq_obs,
            decoder_condition_dim=0,
            target_dim=num_student_fsq_obs,
            embedding_dim=self.fsq_embedding_dim,
            hidden_dim=self.fsq_hidden_dim,
            fsq_levels=self.fsq_levels,
        )  # TODO: 重构FSQ模块
        print(f"Actor FSQ: {self.student_fsq}")
        # Actor FSQ observation normalization
        self.student_fsq_obs_normalization = student_obs_normalization
        if student_obs_normalization:
            self.student_fsq_obs_normalizer = EmpiricalNormalization(num_student_fsq_obs)
        else:
            self.student_fsq_obs_normalizer = torch.nn.Identity()

        # Actor
        self.state_dependent_std = state_dependent_std
        if self.state_dependent_std:
            self.student = MLP(
                num_student_obs + self.student_fsq.embedding_dim,
                [2, num_actions],
                student_hidden_dims,
                activation,
            )
        else:
            self.student = MLP(
                num_student_obs + self.student_fsq.embedding_dim,
                num_actions,
                student_hidden_dims,
                activation,
            )
        print(f"Actor MLP: {self.student}")

        # Actor observation normalization
        self.student_obs_normalization = student_obs_normalization
        if student_obs_normalization:
            self.student_obs_normalizer = EmpiricalNormalization(num_student_obs)
        else:
            self.student_obs_normalizer = torch.nn.Identity()

        # Critic FSQ
        self.critic_fsq = FrameFSQVAE(
            encoder_input_dim=num_critic_fsq_obs,
            decoder_condition_dim=0,
            target_dim=num_critic_fsq_obs,
            embedding_dim=32,
            hidden_dim=256,
            fsq_levels=16,
        )
        print(f"Critic FSQ: {self.critic_fsq}")

        # Critic fsq observation normalization
        self.critic_fsq_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_fsq_obs_normalizer = EmpiricalNormalization(num_critic_fsq_obs)
        else:
            self.critic_fsq_obs_normalizer = torch.nn.Identity()

        # Critic
        self.critic = MLP(
            num_critic_obs + self.critic_fsq.embedding_dim,
            1,
            critic_hidden_dims,
            activation,
        )
        print(f"Critic MLP: {self.critic}")

        # Critic observation normalization
        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()

        self.teacher = MLP(num_teacher_obs, num_actions, teacher_hidden_dims, activation)
        print(f"Teacher MLP: {self.teacher}")
        if teacher_obs_normalization:
            self.teacher_obs_normalizer = EmpiricalNormalization(num_teacher_obs)
        else:
            self.teacher_obs_normalizer = torch.nn.Identity()
        # Action noise
        self.noise_std_type = noise_std_type
        if self.state_dependent_std:
            torch.nn.init.zeros_(self.student[-2].weight[num_actions:])
            if self.noise_std_type == "scalar":
                torch.nn.init.constant_(
                    self.student[-2].bias[num_actions:], init_noise_std
                )
            elif self.noise_std_type == "log":
                torch.nn.init.constant_(
                    self.student[-2].bias[num_actions:],
                    torch.log(torch.tensor(init_noise_std + 1e-7)),
                )
            else:
                raise ValueError(
                    f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'"
                )
        else:
            if self.noise_std_type == "scalar":
                self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            elif self.noise_std_type == "log":
                self.log_std = nn.Parameter(
                    torch.log(init_noise_std * torch.ones(num_actions))
                )
            else:
                raise ValueError(
                    f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'"
                )

        # Action distribution
        # Note: Populated in update_distribution
        self.distribution = None
        self.teacher_distribution = None

        # Disable args validation for speedup
        Normal.set_default_validate_args(False)

    def reset(self, dones: torch.Tensor | None = None) -> None:
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
    def teacher_action_mean(self) -> torch.Tensor:
        return self.teacher_distribution.mean

    @property
    def teacher_action_std(self) -> torch.Tensor:
        return self.teacher_distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def kl_divergence(self, mu_s, sigma_s, mu_t, sigma_t):
        return torch.log(sigma_t / sigma_s) + (sigma_s**2 + (mu_s - mu_t)**2) / (2 * sigma_t**2) - 0.5
    
    def _update_distribution(self, obs: torch.Tensor) -> None:
        if torch.isnan(obs).any():
            raise ValueError(f"张量中存在 NaN 值")
        if self.state_dependent_std:
            # Compute mean and standard deviation
            mean_and_std = self.student(obs)
            if self.noise_std_type == "scalar":
                mean, std = torch.unbind(mean_and_std, dim=-2)
            elif self.noise_std_type == "log":
                mean, log_std = torch.unbind(mean_and_std, dim=-2)
                std = torch.exp(log_std)
            else:
                raise ValueError(
                    f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'"
                )
        else:
            # Compute mean
            mean = self.student(obs)
            # Compute standard deviation
            if self.noise_std_type == "scalar":
                std = self.std.expand_as(mean)
            elif self.noise_std_type == "log":
                std = torch.exp(self.log_std).expand_as(mean)
                # print("[INFO] use log_std")
            else:
                raise ValueError(
                    f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'"
                )
        # Create distribution
        self.distribution = Normal(mean, std)
    
    def _update_teacher_distribution(self, obs: torch.Tensor) -> None:
        if torch.isnan(obs).any():
            raise ValueError(f"张量中存在 NaN 值")
        if self.state_dependent_std:
            # Compute mean and standard deviation
            mean_and_std = self.teacher(obs)
            if self.noise_std_type == "scalar":
                mean, std = torch.unbind(mean_and_std, dim=-2)
            elif self.noise_std_type == "log":
                mean, log_std = torch.unbind(mean_and_std, dim=-2)
                std = torch.exp(log_std)
            else:
                raise ValueError(
                    f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'"
                )
        else:
            # Compute mean
            mean = self.teacher(obs)
            # Compute standard deviation
            if self.noise_std_type == "scalar":
                std = self.std.expand_as(mean)
            elif self.noise_std_type == "log":
                std = torch.exp(self.log_std).expand_as(mean)
                # print("[INFO] use log_std")
            else:
                raise ValueError(
                    f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'"
                )
        # Create distribution
        self.teacher_distribution = Normal(mean, std)

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        student_obs = self.get_student_obs(obs)
        student_obs = self.student_obs_normalizer(student_obs)
        teacher_obs = self.get_teacher_obs(obs)
        teacher_obs = self.teacher_obs_normalizer(teacher_obs)

        student_fsq_obs = self.get_student_fsq_obs(obs)
        student_fsq_obs = self.student_fsq_obs_normalizer(student_fsq_obs)

        if kwargs.get("reconstruct", False):
            fsq_out = self.student_fsq.forward(student_fsq_obs, decoder_condition=None)
            obs = torch.cat((student_obs, fsq_out["z_q"]), dim=-1)
            self._update_distribution(obs)
            return {
                "action": self.distribution.sample(),
                "fsq_out": self.student_fsq.loss_function(student_fsq_obs, fsq_out),
            }
        else:
            fsq_out = self.student_fsq.encoder_forward(student_fsq_obs)
            obs = torch.cat((student_obs, fsq_out["z_q"]), dim=-1)
            self._update_distribution(obs)
            self._update_teacher_distribution(teacher_obs)
            return self.distribution.sample()

    def act_inference(self, obs: TensorDict, only_action: bool = False) -> torch.Tensor:
        student_obs = self.get_student_obs(obs)
        student_obs = self.student_obs_normalizer(student_obs)
        student_fsq_obs = self.get_student_fsq_obs(obs)
        student_fsq_obs = self.student_fsq_obs_normalizer(student_fsq_obs)
        fsq_out = self.student_fsq.encoder_forward(student_fsq_obs)
        obs = torch.cat((student_obs, fsq_out["z_q"]), dim=-1)
        if self.state_dependent_std:
            return self.student(obs)[..., 0, :]
        else:
            return self.student(obs)

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        critic_obs = self.get_critic_obs(obs)
        critic_obs = self.critic_obs_normalizer(critic_obs)
        critic_fsq_obs = self.get_critic_fsq_obs(obs)
        critic_fsq_obs = self.critic_fsq_obs_normalizer(critic_fsq_obs)
        if kwargs.get("reconstruct", False):
            fsq_out = self.critic_fsq.forward(critic_fsq_obs,decoder_condition=None)
            obs = torch.cat((critic_obs, fsq_out["z_q"]), dim=-1)
            return {
                        "value": self.critic(obs),
                        "fsq_out": self.critic_fsq.loss_function(critic_fsq_obs, fsq_out),
                    }
        else:
            fsq_out = self.critic_fsq.encoder_forward(critic_fsq_obs)
            obs = torch.cat((critic_obs, fsq_out["z_q"]), dim=-1)
            return self.critic(obs)

    def get_student_obs(self, obs: TensorDict) -> torch.Tensor:
        return self.get_obs(obs, "policy")

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        return self.get_obs(obs, "critic")

    def get_student_fsq_obs(self, obs: TensorDict) -> torch.Tensor:
        return self.get_obs(obs, "policy_window")

    def get_critic_fsq_obs(self, obs: TensorDict) -> torch.Tensor:
        return self.get_obs(obs, "critic_window")

    def get_teacher_obs(self, obs: TensorDict) -> torch.Tensor:
        return self.get_obs(obs, "teacher")
    
    def get_obs(self, obs: TensorDict, name: str) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups[name]]
        return torch.cat(obs_list, dim=-1)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.student_obs_normalization:
            student_obs = self.get_student_obs(obs)
            self.student_obs_normalizer.update(student_obs)
        if self.critic_obs_normalization:
            critic_obs = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(critic_obs)
        if self.student_fsq_obs_normalization:
            student_fsq_obs = self.get_student_fsq_obs(obs)
            self.student_fsq_obs_normalizer.update(student_fsq_obs)
        if self.critic_fsq_obs_normalization:
            critic_fsq_obs = self.get_critic_fsq_obs(obs)
            self.critic_fsq_obs_normalizer.update(critic_fsq_obs)

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        if any("actor" in key for key in state_dict):  # Load parameters from rl training
            teacher_state_dict = {}
            teacher_obs_normalizer_state_dict = {}
            for key, value in state_dict.items():
                if "actor." in key:
                    teacher_state_dict[key.replace("actor.", "")] = value
                if "actor_obs_normalizer." in key:
                    teacher_obs_normalizer_state_dict[key.replace("actor_obs_normalizer.", "")] = value
            self.teacher.load_state_dict(teacher_state_dict, strict=strict)
            self.teacher_obs_normalizer.load_state_dict(teacher_obs_normalizer_state_dict, strict=strict)
            self.loaded_teacher = True
            self.teacher.eval()
            self.teacher_obs_normalizer.eval()
        return False

    def export_policy_as_onnx(
        self,
        env,
        path: str,
        filename: str = "policy.onnx",
        verbose: bool = False,
    ) -> None:

        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        class _OnnxPolicyExporter(torch.nn.Module):
            def __init__(self, env, student_critic: ActorCriticSingleFSQ, verbose=False):
                super().__init__()
                self.verbose = verbose
                self.student_input_dim = student_critic.student[0].in_features - student_critic.student_fsq.embedding_dim
                self.student_fsq_input_dim = student_critic.student_fsq.encoder_input_dim

                self.student_fsq_encoder = copy.deepcopy(student_critic.student_fsq.encoder)
                self.student = copy.deepcopy(student_critic.student)
                self.student_obs_normalizer = copy.deepcopy(
                    student_critic.student_obs_normalizer
                )
                self.student_fsq_obs_normalizer = copy.deepcopy(
                    student_critic.student_fsq_obs_normalizer
                )
                self.student_fsq_quantizer = copy.deepcopy(
                    student_critic.student_fsq.quantizer
                )

            def forward(self, student_obs, student_fsq_obs):
                student_obs = self.student_obs_normalizer(student_obs)
                student_fsq_obs = self.student_fsq_obs_normalizer(student_fsq_obs)
                fsq_z_e = self.student_fsq_encoder(student_fsq_obs)
                fsq_z_q = self.student_fsq_quantizer(fsq_z_e)["z_q"]
                obs = torch.cat((student_obs, fsq_z_q), dim=-1)
                actions = self.student(obs)
                return actions

            def export(self, path, filename):
                self.to("cpu")
                student_obs = torch.zeros(
                    1, self.student_input_dim
                ) 
                student_fsq_obs = torch.zeros(
                    1, self.student_fsq_input_dim
                ) 
                torch.onnx.export(
                    self,
                    (student_obs, student_fsq_obs),
                    os.path.join(path, filename),
                    export_params=True,
                    opset_version=11,
                    verbose=self.verbose,
                    input_names=["student_obs", "student_fsq_obs"],
                    output_names=[
                        "actions"
                    ],
                    dynamic_axes={},
                )

        exporter = _OnnxPolicyExporter(env, self, verbose)
        exporter.export(path, filename)
