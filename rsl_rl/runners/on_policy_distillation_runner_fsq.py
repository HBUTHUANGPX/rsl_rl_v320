# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import time
import torch
import warnings
from tensordict import TensorDict

from rsl_rl.algorithms import PPOSingleFSQ, PPOSingleFSQDistillation
from rsl_rl.env import VecEnv
from rsl_rl.modules import (
    ActorCriticSingleFSQ,
    ActorCriticSingleFSQDistillation,
    resolve_rnd_config,
    resolve_symmetry_config,
)
from rsl_rl.runners import OnPolicyRunnerFSQ
from rsl_rl.storage import RLDistillationRolloutStorage
from rsl_rl.utils import resolve_obs_groups
from rsl_rl.utils.logger import Logger


class OnPolicyDisstillationRunnerFSQ(OnPolicyRunnerFSQ):
    """On-policy runner for training and evaluation of actor-critic methods."""

    def __init__(
        self,
        env: VecEnv,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = "cpu",
    ) -> None:
        super().__init__(env, train_cfg, log_dir, device)

    def learn(
        self, num_learning_iterations: int, init_at_random_ep_len: bool = False
    ) -> None:
        # Check if teacher is loaded
        if not self.alg.policy.loaded_teacher:
            raise ValueError(
                "Teacher model parameters not loaded. Please load a teacher model to distill."
            )

        super().learn(num_learning_iterations, init_at_random_ep_len)

    def load(
        self, path: str, load_optimizer: bool = True, map_location: str | None = None
    ) -> dict:
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        # Load model
        resumed_training = self.alg.policy.load_state_dict(
            loaded_dict["model_state_dict"]
        )
        # Load RND model if used
        if self.alg_cfg["rnd_cfg"]:
            self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])
        # Load optimizer if used
        if load_optimizer and resumed_training:
            # Algorithm optimizer
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            # RND optimizer if used
            if self.alg_cfg["rnd_cfg"]:
                self.alg.rnd_optimizer.load_state_dict(
                    loaded_dict["rnd_optimizer_state_dict"]
                )
        # Load current learning iteration
        if resumed_training:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def _construct_algorithm(
        self, obs: TensorDict
    ) -> PPOSingleFSQ | PPOSingleFSQDistillation:
        """Construct the actor-critic algorithm."""
        # Resolve RND config if used
        self.alg_cfg = resolve_rnd_config(
            self.alg_cfg, obs, self.cfg["obs_groups"], self.env
        )

        # Resolve symmetry config if used
        self.alg_cfg = resolve_symmetry_config(self.alg_cfg, self.env)

        # Initialize the policy
        actor_critic_class = eval(self.policy_cfg.pop("class_name"))
        actor_critic: ActorCriticSingleFSQ | ActorCriticSingleFSQDistillation = (
            actor_critic_class(
                obs, self.cfg["obs_groups"], self.env.num_actions, **self.policy_cfg
            ).to(self.device)
        )

        # Initialize the storage
        storage = RLDistillationRolloutStorage(
            "rl_distillation",
            self.env.num_envs,
            self.cfg["num_steps_per_env"],
            obs,
            [self.env.num_actions],
            self.device,
        )

        # Initialize the algorithm
        alg_class = eval(self.alg_cfg.pop("class_name"))
        alg: PPOSingleFSQ | PPOSingleFSQDistillation = alg_class(
            actor_critic,
            storage,
            device=self.device,
            **self.alg_cfg,
            multi_gpu_cfg=self.multi_gpu_cfg,
        )

        return alg
