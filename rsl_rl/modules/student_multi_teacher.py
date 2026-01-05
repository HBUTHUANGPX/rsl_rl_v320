from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal
from typing import Any, NoReturn

from rsl_rl.networks import MLP, EmpiricalNormalization, HiddenState
from rsl_rl.modules.student_teacher import StudentTeacher

class StudentMultiTeacher(StudentTeacher):
    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        teacher_num: int = 1,
        motion_run_names: list[str] = [""],
        student_obs_normalization: bool = False,
        teacher_obs_normalization: bool = False,
        student_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        teacher_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        activation: str = "elu",
        init_noise_std: float = 0.1,
        noise_std_type: str = "scalar",
        **kwargs: dict[str, Any],
    ) -> None:
        super().__init__(obs, obs_groups, num_actions,
                         student_obs_normalization=student_obs_normalization,
                         teacher_obs_normalization=teacher_obs_normalization,
                         student_hidden_dims=student_hidden_dims,
                         teacher_hidden_dims=teacher_hidden_dims,
                         activation=activation,
                         init_noise_std=init_noise_std,
                         noise_std_type=noise_std_type,
                         **kwargs)
        self.teacher_num = teacher_num
        self.motion_run_names = motion_run_names
        self.num_actions = num_actions
        
        num_teacher_obs = 0
        for obs_group in obs_groups["teacher"]:
            assert len(obs[obs_group].shape) == 2, "仅支持 1D 观测。"
            num_teacher_obs += obs[obs_group].shape[-1]

        # 多教师支持
        self.teacher = nn.ModuleList()
        self.teacher_obs_normalizer = nn.ModuleList()
        self.teacher_num = teacher_num
        for i in range(teacher_num):
            # 教师网络（保持原样，用于生成监督动作）
            teacher_net = MLP(num_teacher_obs, num_actions, teacher_hidden_dims, activation)
            self.teacher.append(teacher_net)
            print(f"Teacher MLP {i}: {teacher_net}")

            # 教师观测规范化
            if teacher_obs_normalization:
                teacher_obs_norm = EmpiricalNormalization(num_teacher_obs)
            else:
                teacher_obs_norm = torch.nn.Identity()
            self.teacher_obs_normalizer.append(teacher_obs_norm)
    
    def load_state_dicts(self, state_dicts: dict[dict | None], strict: bool = True) -> bool:
        """加载参数（兼容教师和学生）。
        
        Args:
            state_dict: 状态字典。
            strict: 是否严格匹配。
        
        Returns:
            是否恢复训练。
        """
        # 与原类相同，略微调整以包含 CVAE 组件
        if not state_dicts:
            raise ValueError("state_dicts 为空列表。")
        missing_clips = [run_name for run_name in self.motion_run_names if run_name not in state_dicts]
        if missing_clips:
            raise ValueError(f"以下 clip 缺少对应的状态字典: {', '.join(missing_clips)}。每个 clip 必须有对应的 teacher 网络。")# 每个clip必须要有对应的teacher网络，但teacher网络可以没有对应的clip
        # 对state_dicts进行查询，提取出每个clip对应的teacher网络
        for i, run_name in enumerate(self.motion_run_names):
            state_dict = state_dicts.get(run_name)
            if state_dict is None:
                continue  # 虽然前检查已确保不为空，但为鲁棒性保留
            
            if "model_state_dict" not in state_dict:
                raise ValueError(f"policy {run_name}' 的状态字典缺少 'model_state_dict' 键。")
            if any("actor" in key for key in state_dict["model_state_dict"]):  # 从 RL 加载教师
                teacher_state_dict = {}
                teacher_obs_normalizer_state_dict = {}
                for key, value in state_dict["model_state_dict"].items():
                    if "actor." in key:
                        teacher_state_dict[key.replace("actor.", "")] = value
                    if "actor_obs_normalizer." in key:
                        teacher_obs_normalizer_state_dict[key.replace("actor_obs_normalizer.", "")] = value
                self.teacher[i].load_state_dict(teacher_state_dict, strict=strict)
                self.teacher_obs_normalizer[i].load_state_dict(teacher_obs_normalizer_state_dict, strict=strict)
                self.teacher[i].eval()
                self.teacher_obs_normalizer[i].eval()

        self.loaded_teacher = True
        return False

    def load_state_dicts_play(self, state_dicts: dict[dict | None], strict: bool = True) -> bool:
        # 与原类相同，略微调整以包含 CVAE 组件
        if not state_dicts:
            raise ValueError("state_dicts 为空列表。")
        clips_num = len(self.motion_run_names)
        if len(state_dicts) ==1:
            state_dict = state_dicts[next(iter(state_dicts))]
        else:
            raise ValueError(f"提供的 state_dicts 数量 ({len(state_dicts)})应该为1。")
        keys = [key for key in state_dict["model_state_dict"]]
        pattern = r'^teacher_obs_normalizer\.\d+\.count$'  # 使用^和$确保完全匹配
        import re
        count = sum(1 for s in keys if re.fullmatch(pattern, s))
        if count != clips_num:
            raise ValueError(f"提供的 state_dicts 中教师数量 ({count}) 与 motion_run_names 数量 ({clips_num}) 不匹配。")
        if any("actor" in key for key in state_dict["model_state_dict"]):  # 从 RL 加载教师
            for i in range(count):
                teacher_state_dict = {}
                teacher_obs_normalizer_state_dict = {}
                for key, value in state_dict["model_state_dict"].items():
                    if "teacher." in key:
                        teacher_state_dict[key.replace("teacher."+str(i), "")] = value
                    if "teacher_obs_normalizer." in key:
                        teacher_obs_normalizer_state_dict[key.replace("teacher_obs_normalizer."+str(i), "")] = value
                self.teacher[i].load_state_dict(teacher_state_dict, strict=strict)
                self.teacher_obs_normalizer[i].load_state_dict(teacher_obs_normalizer_state_dict, strict=strict)
                self.teacher[i].eval()
                self.teacher_obs_normalizer[i].eval()
        self.loaded_teacher = True
        # 对state_dicts进行查询，提取出每个clip对应的teacher网络
        for i, run_name in enumerate(self.motion_run_names):
            state_dict = state_dicts.get(run_name)
            if state_dict is None:
                continue  # 虽然前检查已确保不为空，但为鲁棒性保留
            
            if "model_state_dict" not in state_dict:
                raise ValueError(f"policy {run_name}' 的状态字典缺少 'model_state_dict' 键。")
            if any("actor" in key for key in state_dict["model_state_dict"]):  # 从 RL 加载教师
                teacher_state_dict = {}
                teacher_obs_normalizer_state_dict = {}
                for key, value in state_dict["model_state_dict"].items():
                    if "actor." in key:
                        teacher_state_dict[key.replace("actor.", "")] = value
                    if "actor_obs_normalizer." in key:
                        teacher_obs_normalizer_state_dict[key.replace("actor_obs_normalizer.", "")] = value
                self.teacher[i].load_state_dict(teacher_state_dict, strict=strict)
                self.teacher_obs_normalizer[i].load_state_dict(teacher_obs_normalizer_state_dict, strict=strict)
                self.teacher[i].eval()
                self.teacher_obs_normalizer[i].eval()
        self.loaded_teacher = True
        return False
    
    def get_motion_id(self, obs: TensorDict) -> torch.Tensor:
        """获取 motion_id 观测（如果存在）。"""
        if "motion_id" in self.obs_groups and self.obs_groups["motion_id"]:
            obs_list = []
            for obs_group in self.obs_groups["motion_id"]:
                obs_list.append(obs[obs_group])
            # obs_list = [obs[obs_group] for obs_group in self.obs_groups["motion_id"]]
            return torch.cat(obs_list, dim=-1).to(torch.int64)
        else:
            raise ValueError("观测组中未定义 'motion_id'")

    def evaluate(self, obs: TensorDict) -> torch.Tensor:
        """教师评估（生成监督动作）。
        
        Args:
            obs: 观测字典。
        
        Returns:
            教师动作。
        """
        teacher_obs = self.get_teacher_obs(obs)
        motion_id = self.get_motion_id(obs).squeeze(1)# 形状从 (4096, 1) 转换为 (4096,)
        # 识别唯一教师索引
        unique_ids, inverse_indices = torch.unique(motion_id, return_inverse=True)

        # 预分配动作张量（假设动作维度为 action_dim，根据实际替换）
        actions = torch.zeros(teacher_obs.shape[0], self.num_actions, device=teacher_obs.device)
        # 根据 motion_id 选择对应教师
        with torch.no_grad():
            for uid in unique_ids:
                mask = (motion_id == uid)  # 布尔掩码，形状 (4096,)
                sub_obs = teacher_obs[mask]  # 子批次观测，形状 (sub_batch_size, 100)
                if sub_obs.numel() == 0:
                    continue
                # 应用对应教师的观测归一化
                normalized_sub_obs = self.teacher_obs_normalizer[uid](sub_obs)
                # 前向传播生成动作
                sub_actions = self.teacher[uid](normalized_sub_obs)
                # 将子批次动作填充回整体张量
                actions[mask] = sub_actions
            return actions 