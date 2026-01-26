# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import torch
import os
from tensordict import TensorDict
from collections.abc import Sequence
from rsl_rl.algorithms import PPO_Distil
from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner, DistillationRunner
from rsl_rl.storage import RolloutStorage
from rsl_rl.modules import (
    ActorCritic_CVAE,
    resolve_rnd_config,
    resolve_symmetry_config,
)
import warnings


class MultiTeacherDistillationRunner(DistillationRunner):
    """Distillation runner for training and evaluation of teacher-student methods."""

    def __init__(
        self,
        env: VecEnv,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = "cpu",
        motion_run_names: list[str] = [""],
        teacher_names: list[str] = [""],
    ) -> None:
        if isinstance(teacher_names, str):
            teacher_names = [teacher_names]
        self.motion_run_names = motion_run_names
        self.teacher_num = len(teacher_names)
        print(f"[INFO]: Number of teachers: {self.teacher_num}")
        super().__init__(env, train_cfg, log_dir, device)

    def load(
        self,
        paths: str | Sequence[str] | list[str],
        load_optimizer: bool = True,
        map_location: str | None = None,
        is_eval: bool = False,
    ) -> dict:
        if isinstance(paths, str):
            paths = [paths]
        loaded_dicts: dict[dict | None] = {
            os.path.basename(os.path.dirname(p)): torch.load(p, weights_only=False, map_location=map_location)
            for p in paths
        }
        if is_eval:
            resumed_training = self.alg.policy.load_state_dicts_play(loaded_dicts)
        else:
            resumed_training = self.alg.policy.load_state_dicts(loaded_dicts)

        # Load RND model if used
        if self.alg_cfg["rnd_cfg"]:
            self.alg.rnd.load_state_dict(next(iter(loaded_dicts.values()))["rnd_state_dict"])
        # Load optimizer if used
        if load_optimizer and resumed_training:
            # Algorithm optimizer
            self.alg.optimizer.load_state_dict(next(iter(loaded_dicts.values()))["optimizer_state_dict"])
            # RND optimizer if used
            if self.alg_cfg["rnd_cfg"]:
                self.alg.rnd_optimizer.load_state_dict(
                    next(iter(loaded_dicts.values()))["rnd_optimizer_state_dict"]
                )
        # Load current learning iteration
        if resumed_training:
            self.current_learning_iteration = next(iter(loaded_dicts.values()))["iter"]
        print(f"[INFO]: Loaded checkpoint from :\r\n{paths}")
        return next(iter(loaded_dicts.values()))["infos"]

    def _construct_algorithm(self, obs: TensorDict) -> PPO_Distil:
        """Construct the distillation algorithm."""
        # Initialize the policy
        student_teacher_class = eval(self.policy_cfg.pop("class_name"))
        student_teacher: ActorCritic_CVAE = student_teacher_class(
            obs = obs,
            obs_groups = self.cfg["obs_groups"],
            num_actions = self.env.num_actions,
            motion_run_names = self.motion_run_names,
            **self.policy_cfg,
        ).to(self.device)

        # Initialize the storage
        storage = RolloutStorage(
            "rl",
            self.env.num_envs,
            self.cfg["num_steps_per_env"],
            obs,
            [self.env.num_actions],
            self.device,
        )

        # Initialize the algorithm
        alg_class = eval(self.alg_cfg.pop("class_name"))
        alg: PPO_Distil = alg_class(
            policy = student_teacher,
            storage = storage,
            device=self.device,
            **self.alg_cfg,
            multi_gpu_cfg=self.multi_gpu_cfg,
        )

        # Set RND configuration to None as it does not apply to distillation
        self.cfg["algorithm"]["rnd_cfg"] = None

        return alg
