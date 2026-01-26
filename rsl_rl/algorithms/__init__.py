# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implementation of different learning algorithms."""

from .distillation import Distillation
from .ppo import PPO
from .multi_teacher_distillation import MultiTeacherDistillation
from .ppo_multi_teacher_distillation import PPO_Distil
__all__ = ["PPO", "Distillation", "MultiTeacherDistillation","PPO_Distil"]
