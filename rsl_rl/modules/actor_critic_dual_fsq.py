# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import os
from typing import Any, NoReturn

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal

from rsl_rl.modules.dual_fsq_components import DualFSQAutoEncoder
from rsl_rl.networks import EmpiricalNormalization, MLP


class ActorCriticDualFSQ(nn.Module):
    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: tuple[int] | list[int] = (512, 256, 128),
        critic_hidden_dims: tuple[int] | list[int] = (512, 256, 128),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        latent_dim: int = 64,
        robot_encoder_hidden_dims: tuple[int] | list[int] = (512, 256),
        human_encoder_hidden_dims: tuple[int] | list[int] = (512, 256),
        decoder_hidden_dims: tuple[int] | list[int] = (256, 512),
        fsq_levels: int = 16,
        quantizer_type: str = "ifsq",
        ifsq_boundary_fn: str = "sigmoid",
        ifsq_boundary_scale: float = 1.6,
        dual_fsq_loss_weights: dict[str, float] | None = None,
        detach_fsq_latent_in_policy: bool = False,
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "ActorCriticDualFSQ.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs])
            )
        super().__init__()

        self.obs_groups = obs_groups
        self.actor_human_fsq_group_name = (
            "actor_human_fsq_window" if "actor_human_fsq_window" in obs_groups else "human_fsq_window"
        )
        self.actor_robot_fsq_group_name = (
            "actor_robot_fsq_window" if "actor_robot_fsq_window" in obs_groups else "robot_fsq_window"
        )
        self.critic_human_fsq_group_name = (
            "critic_human_fsq_window"
            if "critic_human_fsq_window" in obs_groups
            else self.actor_human_fsq_group_name
        )
        self.critic_robot_fsq_group_name = (
            "critic_robot_fsq_window"
            if "critic_robot_fsq_window" in obs_groups
            else self.actor_robot_fsq_group_name
        )
        self.detach_fsq_latent_in_policy = bool(detach_fsq_latent_in_policy)
        self.dual_fsq_loss_weights = dual_fsq_loss_weights or {
            "robot_recon": 1.0,
            "human_recon": 1.0,
            "latent_align": 1.0,
            "cycle_latent": 1.0,
        }

        num_actor_obs = self._count_obs_dim(obs, "policy")
        num_critic_obs = self._count_obs_dim(obs, "critic")
        num_actor_human_fsq_obs = self._count_obs_dim(obs, self.actor_human_fsq_group_name)
        num_actor_robot_fsq_obs = self._count_obs_dim(obs, self.actor_robot_fsq_group_name)
        num_critic_human_fsq_obs = self._count_obs_dim(obs, self.critic_human_fsq_group_name)
        num_critic_robot_fsq_obs = self._count_obs_dim(obs, self.critic_robot_fsq_group_name)

        self.actor_dual_fsq = DualFSQAutoEncoder(
            robot_input_dim=num_actor_robot_fsq_obs,
            human_input_dim=num_actor_human_fsq_obs,
            latent_dim=latent_dim,
            robot_encoder_hidden_dims=robot_encoder_hidden_dims,
            human_encoder_hidden_dims=human_encoder_hidden_dims,
            decoder_hidden_dims=decoder_hidden_dims,
            fsq_levels=fsq_levels,
            activation=activation,
            quantizer_type=quantizer_type,
            ifsq_boundary_fn=ifsq_boundary_fn,
            ifsq_boundary_scale=ifsq_boundary_scale,
        )
        print(f"Actor Dual FSQ: {self.actor_dual_fsq}")

        self.critic_dual_fsq = DualFSQAutoEncoder(
            robot_input_dim=num_critic_robot_fsq_obs,
            human_input_dim=num_critic_human_fsq_obs,
            latent_dim=latent_dim,
            robot_encoder_hidden_dims=robot_encoder_hidden_dims,
            human_encoder_hidden_dims=human_encoder_hidden_dims,
            decoder_hidden_dims=decoder_hidden_dims,
            fsq_levels=fsq_levels,
            activation=activation,
            quantizer_type=quantizer_type,
            ifsq_boundary_fn=ifsq_boundary_fn,
            ifsq_boundary_scale=ifsq_boundary_scale,
        )
        print(f"Critic Dual FSQ: {self.critic_dual_fsq}")

        self.actor_obs_normalization = actor_obs_normalization
        self.critic_obs_normalization = critic_obs_normalization
        self.actor_human_fsq_obs_normalization = actor_obs_normalization
        self.actor_robot_fsq_obs_normalization = actor_obs_normalization
        self.critic_human_fsq_obs_normalization = critic_obs_normalization
        self.critic_robot_fsq_obs_normalization = critic_obs_normalization

        self.actor_obs_normalizer = (
            EmpiricalNormalization(num_actor_obs) if actor_obs_normalization else torch.nn.Identity()
        )
        self.critic_obs_normalizer = (
            EmpiricalNormalization(num_critic_obs) if critic_obs_normalization else torch.nn.Identity()
        )
        self.actor_human_fsq_obs_normalizer = (
            EmpiricalNormalization(num_actor_human_fsq_obs)
            if self.actor_human_fsq_obs_normalization
            else torch.nn.Identity()
        )
        self.actor_robot_fsq_obs_normalizer = (
            EmpiricalNormalization(num_actor_robot_fsq_obs)
            if self.actor_robot_fsq_obs_normalization
            else torch.nn.Identity()
        )
        self.critic_human_fsq_obs_normalizer = (
            EmpiricalNormalization(num_critic_human_fsq_obs)
            if self.critic_human_fsq_obs_normalization
            else torch.nn.Identity()
        )
        self.critic_robot_fsq_obs_normalizer = (
            EmpiricalNormalization(num_critic_robot_fsq_obs)
            if self.critic_robot_fsq_obs_normalization
            else torch.nn.Identity()
        )

        self.state_dependent_std = state_dependent_std
        if self.state_dependent_std:
            self.actor = MLP(
                num_actor_obs + self.actor_dual_fsq.embedding_dim,
                [2, num_actions],
                actor_hidden_dims,
                activation,
            )
        else:
            self.actor = MLP(
                num_actor_obs + self.actor_dual_fsq.embedding_dim,
                num_actions,
                actor_hidden_dims,
                activation,
            )
        print(f"Actor MLP: {self.actor}")

        self.critic = MLP(
            num_critic_obs + self.critic_dual_fsq.embedding_dim,
            1,
            critic_hidden_dims,
            activation,
        )
        print(f"Critic MLP: {self.critic}")

        self.noise_std_type = noise_std_type
        if self.state_dependent_std:
            torch.nn.init.zeros_(self.actor[-2].weight[num_actions:])
            if self.noise_std_type == "scalar":
                torch.nn.init.constant_(self.actor[-2].bias[num_actions:], init_noise_std)
            elif self.noise_std_type == "log":
                torch.nn.init.constant_(
                    self.actor[-2].bias[num_actions:],
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
                self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
            else:
                raise ValueError(
                    f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'"
                )

        self.distribution = None
        self._dual_fsq_cache_obs_id: int | None = None
        self._dual_fsq_cache: dict[str, Any] | None = None

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
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def _count_obs_dim(self, obs: TensorDict, name: str) -> int:
        num_obs = 0
        for obs_group in self.obs_groups[name]:
            assert len(obs[obs_group].shape) == 2, "ActorCriticDualFSQ 只支持一维观测。"
            num_obs += obs[obs_group].shape[-1]
        return num_obs

    def _update_distribution(self, obs: torch.Tensor) -> None:
        if torch.isnan(obs).any():
            raise ValueError("张量中存在 NaN 值")
        if self.state_dependent_std:
            mean_and_std = self.actor(obs)
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
            mean = self.actor(obs)
            if self.noise_std_type == "scalar":
                std = self.std.expand_as(mean)
            elif self.noise_std_type == "log":
                std = torch.exp(self.log_std).expand_as(mean)
            else:
                raise ValueError(
                    f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'"
                )
        self.distribution = Normal(mean, std)

    def _pack_fsq_out(
        self,
        prefix: str,
        loss: torch.Tensor,
        output: Any,
        terms: dict[str, torch.Tensor],
    ) -> dict[str, Any]:
        return {
            "loss": loss,
            "terms": terms,
            "output": output,
            "q_robot": output.q_robot,
            "q_human": output.q_human,
            "q_cycle": output.q_cycle,
            "robot_recon_from_robot": output.robot_recon_from_robot,
            "robot_recon_from_human": output.robot_recon_from_human,
            "robot_aux": output.robot_aux,
            "human_aux": output.human_aux,
            "cycle_aux": output.cycle_aux,
            "prefix": prefix,
        }

    def _compute_dual_fsq_out(self, obs: TensorDict, force: bool = False) -> dict[str, Any]:
        obs_id = id(obs)
        if not force and self._dual_fsq_cache_obs_id == obs_id and self._dual_fsq_cache is not None:
            return self._dual_fsq_cache

        actor_robot_fsq_obs = self.get_actor_robot_fsq_obs_normalized(obs)
        actor_human_fsq_obs = self.get_actor_human_fsq_obs_normalized(obs)
        critic_robot_fsq_obs = self.get_critic_robot_fsq_obs_normalized(obs)
        critic_human_fsq_obs = self.get_critic_human_fsq_obs_normalized(obs)
        actor_loss, actor_output, actor_terms = self.actor_dual_fsq.compute_loss(
            actor_robot_fsq_obs,
            actor_human_fsq_obs,
            weights=self.dual_fsq_loss_weights,
        )
        critic_loss, critic_output, critic_terms = self.critic_dual_fsq.compute_loss(
            critic_robot_fsq_obs,
            critic_human_fsq_obs,
            weights=self.dual_fsq_loss_weights,
        )
        terms = {}
        for name in actor_terms:
            terms[name] = 0.5 * (actor_terms[name] + critic_terms[name])
        cache = {
            "loss": 0.5 * (actor_loss + critic_loss),
            "terms": terms,
            "actor": self._pack_fsq_out("actor", actor_loss, actor_output, actor_terms),
            "critic": self._pack_fsq_out("critic", critic_loss, critic_output, critic_terms),
            "q_human": actor_output.q_human,
            "q_robot": critic_output.q_robot,
        }
        self._dual_fsq_cache_obs_id = obs_id
        self._dual_fsq_cache = cache
        return cache

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor | dict[str, Any]:
        actor_obs = self.actor_obs_normalizer(self.get_actor_obs(obs))
        if kwargs.get("reconstruct", False):
            fsq_out = self._compute_dual_fsq_out(obs, force=True)
            actor_latent = fsq_out["actor"]["q_human"].detach() if self.detach_fsq_latent_in_policy else fsq_out["actor"]["q_human"]
            actor_input = torch.cat((actor_obs, actor_latent), dim=-1)
            self._update_distribution(actor_input)
            return {"action": self.distribution.sample(), "fsq_out": fsq_out}

        human_fsq_obs = self.get_actor_human_fsq_obs_normalized(obs)
        actor_latent = self.actor_dual_fsq.latent_from_human(
            human_fsq_obs,
            detach=self.detach_fsq_latent_in_policy,
        )
        actor_input = torch.cat((actor_obs, actor_latent), dim=-1)
        self._update_distribution(actor_input)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict, only_action: bool = False) -> torch.Tensor:
        actor_obs = self.actor_obs_normalizer(self.get_actor_obs(obs))
        human_fsq_obs = self.get_actor_human_fsq_obs_normalized(obs)
        actor_latent = self.actor_dual_fsq.latent_from_human(
            human_fsq_obs,
            detach=self.detach_fsq_latent_in_policy,
        )
        actor_input = torch.cat((actor_obs, actor_latent), dim=-1)
        if self.state_dependent_std:
            return self.actor(actor_input)[..., 0, :]
        return self.actor(actor_input)

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor | dict[str, Any]:
        critic_obs = self.critic_obs_normalizer(self.get_critic_obs(obs))
        if kwargs.get("reconstruct", False):
            fsq_out = self._compute_dual_fsq_out(obs, force=False)
            critic_input = torch.cat((critic_obs, fsq_out["critic"]["q_robot"]), dim=-1)
            return {"value": self.critic(critic_input), "fsq_out": fsq_out}

        robot_fsq_obs = self.get_critic_robot_fsq_obs_normalized(obs)
        critic_latent = self.critic_dual_fsq.latent_from_robot(robot_fsq_obs, detach=False)
        critic_input = torch.cat((critic_obs, critic_latent), dim=-1)
        return self.critic(critic_input)

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        return self.get_obs(obs, "policy")

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        return self.get_obs(obs, "critic")

    def get_actor_human_fsq_obs(self, obs: TensorDict) -> torch.Tensor:
        return self.get_obs(obs, self.actor_human_fsq_group_name)

    def get_actor_robot_fsq_obs(self, obs: TensorDict) -> torch.Tensor:
        return self.get_obs(obs, self.actor_robot_fsq_group_name)

    def get_critic_human_fsq_obs(self, obs: TensorDict) -> torch.Tensor:
        return self.get_obs(obs, self.critic_human_fsq_group_name)

    def get_critic_robot_fsq_obs(self, obs: TensorDict) -> torch.Tensor:
        return self.get_obs(obs, self.critic_robot_fsq_group_name)

    def get_actor_human_fsq_obs_normalized(self, obs: TensorDict) -> torch.Tensor:
        return self.actor_human_fsq_obs_normalizer(self.get_actor_human_fsq_obs(obs))

    def get_actor_robot_fsq_obs_normalized(self, obs: TensorDict) -> torch.Tensor:
        return self.actor_robot_fsq_obs_normalizer(self.get_actor_robot_fsq_obs(obs))

    def get_critic_human_fsq_obs_normalized(self, obs: TensorDict) -> torch.Tensor:
        return self.critic_human_fsq_obs_normalizer(self.get_critic_human_fsq_obs(obs))

    def get_critic_robot_fsq_obs_normalized(self, obs: TensorDict) -> torch.Tensor:
        return self.critic_robot_fsq_obs_normalizer(self.get_critic_robot_fsq_obs(obs))

    def get_obs(self, obs: TensorDict, name: str) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups[name]]
        return torch.cat(obs_list, dim=-1)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.actor_obs_normalization:
            self.actor_obs_normalizer.update(self.get_actor_obs(obs))
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(self.get_critic_obs(obs))
        if self.actor_human_fsq_obs_normalization:
            self.actor_human_fsq_obs_normalizer.update(self.get_actor_human_fsq_obs(obs))
        if self.actor_robot_fsq_obs_normalization:
            self.actor_robot_fsq_obs_normalizer.update(self.get_actor_robot_fsq_obs(obs))
        if self.critic_human_fsq_obs_normalization:
            self.critic_human_fsq_obs_normalizer.update(self.get_critic_human_fsq_obs(obs))
        if self.critic_robot_fsq_obs_normalization:
            self.critic_robot_fsq_obs_normalizer.update(self.get_critic_robot_fsq_obs(obs))

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        super().load_state_dict(state_dict, strict=strict)
        return True

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
            def __init__(self, actor_critic: ActorCriticDualFSQ, verbose: bool = False):
                super().__init__()
                self.verbose = verbose
                self.actor_input_dim = actor_critic.actor[0].in_features - actor_critic.actor_dual_fsq.embedding_dim
                self.human_fsq_input_dim = actor_critic.actor_dual_fsq.human_input_dim
                self.human_encoder = copy.deepcopy(actor_critic.actor_dual_fsq.human_encoder)
                self.quantizer = copy.deepcopy(actor_critic.actor_dual_fsq.quantizer)
                self.actor = copy.deepcopy(actor_critic.actor)
                self.actor_obs_normalizer = copy.deepcopy(actor_critic.actor_obs_normalizer)
                self.human_fsq_obs_normalizer = copy.deepcopy(actor_critic.actor_human_fsq_obs_normalizer)

            def forward(self, actor_obs: torch.Tensor, human_fsq_obs: torch.Tensor) -> torch.Tensor:
                actor_obs = self.actor_obs_normalizer(actor_obs)
                human_fsq_obs = self.human_fsq_obs_normalizer(human_fsq_obs)
                z_human = self.human_encoder(human_fsq_obs)
                q_human = self.quantizer(z_human)["z_q"]
                actor_input = torch.cat((actor_obs, q_human), dim=-1)
                return self.actor(actor_input)

            def export(self, path: str, filename: str) -> None:
                self.to("cpu")
                actor_obs = torch.zeros(1, self.actor_input_dim)
                human_fsq_obs = torch.zeros(1, self.human_fsq_input_dim)
                torch.onnx.export(
                    self,
                    (actor_obs, human_fsq_obs),
                    os.path.join(path, filename),
                    export_params=True,
                    opset_version=11,
                    verbose=self.verbose,
                    input_names=["actor_obs", "human_fsq_obs"],
                    output_names=["actions"],
                    dynamic_axes={},
                )

        exporter = _OnnxPolicyExporter(self, verbose)
        exporter.export(path, filename)
