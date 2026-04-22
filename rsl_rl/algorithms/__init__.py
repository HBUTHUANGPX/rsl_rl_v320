# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implementation of different learning algorithms."""

from .distillation import Distillation
from .ppo import PPO
from .multi_teacher_distillation import MultiTeacherDistillation
from .ppo_multi_teacher_distillation import PPO_Distil
from .ppo_dual_fsq import PPODualFSQ
from .ppo_single_fsq import PPOSingleFSQ
from .ppo_single_fsq_distillation import PPOSingleFSQDistillation
__all__ = ["PPO", "PPOSingleFSQ", "PPODualFSQ", "Distillation", "MultiTeacherDistillation", "PPO_Distil", "PPOSingleFSQDistillation"]
