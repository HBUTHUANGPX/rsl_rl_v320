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
    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        teacher_num: int = 1,
        motion_run_names: list[str] = [""],
        latent_dim: int = 64,  # CVAE 潜在维度（从配置传入）
        beta_kl: float = 0.1,  # KL 损失权重（从配置传入）
        student_obs_normalization: bool = False,
        teacher_obs_normalization: bool = False,
        student_hidden_dims: tuple[int] | list[int] = [256, 256, 256],  # 用于 prior 和 decoder
        teacher_hidden_dims: tuple[int] | list[int] = [256, 256, 256],  # 用于 teacher 和 encoder
        activation: str = "elu",
        init_noise_std: float = 0.1,
        noise_std_type: str = "scalar",
        normalize_mu: bool = False,  # 新参数，从配置传入
        z_scale_factor: float = 1.0,  # 新增：z 的缩放因子（默认 1.0，不缩放；从配置传入）
        prior_hidden_dims = [1024, 512, 128],
        encoder_hidden_dims = [512, 256, 128],
        **kwargs: dict[str, Any],
    ) -> None:
        """初始化 CVAE-based 学生-教师模块。
        
        Args:
            obs: 观测字典。
            obs_groups: 观测组映射。
            num_actions: 动作维度。
            latent_dim: 潜在变量维度（论文推荐 64）。
            beta_kl: KL 损失权重（论文推荐 0.1）。
            student_obs_normalization: 是否规范化学生观测。
            teacher_obs_normalization: 是否规范化教师观测。
            student_hidden_dims: 先验和解码器的隐藏层维度。
            teacher_hidden_dims: 编码器和教师的隐藏层维度。
            activation: 激活函数。
            init_noise_std: 初始动作噪声标准差。
            noise_std_type: 噪声类型 ('scalar' 或 'log')。
            normalize_mu: 是否对潜在均值 mu 进行经验规范化。
            z_scale_factor: z 的缩放因子（默认 1.0）。
            kwargs: 忽略的额外参数。
        """
        if kwargs:
            print(
                "StudentTeacher_CVAE.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs])
            )
        super().__init__()

        self.loaded_teacher = False  # 表示教师是否已加载
        self.latent_dim = latent_dim
        self.beta_kl = beta_kl  # KL 权重，用于训练
        self.z_scale_factor = z_scale_factor  # z 缩放因子
        self.num_actions = num_actions
        self.motion_run_names = motion_run_names
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

        # CVAE 组件（修改：prior 和 encoder 各用一个 MLP 输出 2*latent_dim，然后分割）
        # 先验网络：从学生观测到 [mu_p, logvar_p]
        self.prior_network = MLP(num_student_obs, 2 * latent_dim, prior_hidden_dims, activation)
        print(f"Prior Network: {self.prior_network}")

        # 编码器：从教师观测到 [mu_e, logvar_e]
        self.encoder_network = MLP(num_teacher_obs, 2 * latent_dim, encoder_hidden_dims, activation)
        print(f"Encoder Network: {self.encoder_network}")

        # 如果启用，对 mu 使用经验规范化
        self.normalize_mu = normalize_mu
        if normalize_mu:
            self.mu_normalizer = EmpiricalNormalization(latent_dim)  # 潜在维度作为形状
        else:
            self.mu_normalizer = nn.Identity()

        # 解码器：从学生观测 + 潜在 z 到动作均值
        self.decoder = MLP(num_student_obs + latent_dim, num_actions, student_hidden_dims, activation)
        print(f"Decoder MLP: {self.decoder}")

        # 学生观测规范化
        self.student_obs_normalization = student_obs_normalization
        if student_obs_normalization:
            self.student_obs_normalizer = EmpiricalNormalization(num_student_obs)
        else:
            self.student_obs_normalizer = torch.nn.Identity()
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

        # 动作噪声
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"未知噪声类型: {self.noise_std_type}. 应为 'scalar' 或 'log'")

        # 动作分布（在 update_distribution 中填充）
        self.distribution = None

        # 禁用分布验证以加速
        Normal.set_default_validate_args(False)

    def reset(
        self, dones: torch.Tensor | None = None, hidden_states: tuple[HiddenState, HiddenState] = (None, None)
    ) -> None:
        pass  # 无循环状态，保持为空

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

    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """重参数化技巧采样潜在 z，并应用缩放。
        
        Args:
            mu: 均值。
            logvar: log 方差。
        
        Returns:
            采样 z = (mu + eps * std) * scale_factor。
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return (mu + eps * std) * self.z_scale_factor

    def _compute_latent_dist(
        self, student_obs: torch.Tensor, teacher_obs: torch.Tensor, use_prior_only: bool = False, need_kl: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """计算潜在分布（后验或先验），并返回 KL。
        
        Args:
            student_obs: 学生观测。
            teacher_obs: 教师观测。
            use_prior_only: 如果 True，仅使用先验（用于部署推理）。
        
        Returns:
            mu, logvar, z, kl (如果适用，否则 0)。
        """
        # 先验参数（从单一 MLP 输出分割）
        prior_out = self.prior_network(student_obs)
        mu_p, logvar_p = prior_out.split(self.latent_dim, dim=-1)
        logvar_p = logvar_p.clamp(min=-10.0, max=2.0)

        # 采样 z
        if use_prior_only: # train inference和 eval inference 都会走这里,使用prior构建的latent
            z = self._reparameterize(mu_p, logvar_p)
        else: # rollout时使用后验构建的latent
            # 编码器残差参数（从单一 MLP 输出分割）
            encoder_out = self.encoder_network(teacher_obs)
            mu_e, logvar_e = encoder_out.split(self.latent_dim, dim=-1)
            logvar_e = logvar_e.clamp(min=-10.0, max=2.0)

            # 后验 mu（residual 设计）
            mu = mu_p + mu_e
            mu = self.mu_normalizer(mu)  # 规范化（如果启用）
            z = self._reparameterize(mu, logvar_e)

        if need_kl: # train inference更新kl、
            # 编码器残差参数（从单一 MLP 输出分割）
            encoder_out = self.encoder_network(teacher_obs)
            mu_e, logvar_e = encoder_out.split(self.latent_dim, dim=-1)
            logvar_e = logvar_e.clamp(min=-10.0, max=2.0)
            kl = 0.5 * (
                logvar_p - logvar_e + 
                torch.exp(logvar_e) / torch.exp(logvar_p) + 
                mu_e**2 / torch.exp(logvar_p) - 1 # mu - mu_p = mu_e 这里简化了，参考residual设计
            ).sum(-1).mean()
        else: #  eval inference 和 rollout不更新kl
            kl = torch.zeros_like(mu_p.mean())  # 无 KL

        return None, None, z, kl  # 返回后验参数

    def _update_distribution(self, student_obs: torch.Tensor, z: torch.Tensor) -> None:
        """更新动作分布。
        
        Args:
            student_obs: 学生观测。
            z: 采样潜在变量。
        """
        # 连接学生观测和 z
        decoder_input = torch.cat([student_obs, z], dim=-1)
        # 计算动作均值
        mean = self.decoder(decoder_input)
        # 计算 std
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"未知噪声类型: {self.noise_std_type}.")
        self.distribution = Normal(mean, std)

    def act(self, obs: TensorDict) -> torch.Tensor:
        """生成动作（rollout 时，使用后验采样 z 以获得高质量动作）。
        
        Args:
            obs: 观测字典。
        
        Returns:
            采样动作。
        """
        student_obs = self.get_student_obs(obs)
        student_obs = self.student_obs_normalizer(student_obs)
        teacher_obs = self.get_teacher_obs(obs)
        motion_id = self.get_motion_id(obs).squeeze(1)# 形状从 (4096, 1) 转换为 (4096,)
        unique_ids, _ = torch.unique(motion_id, return_inverse=True)
        _teacher_obs = torch.zeros_like(teacher_obs)
        for uid in unique_ids:
            mask = (motion_id == uid)  # 布尔掩码，形状 (4096,)
            sub_obs = teacher_obs[mask]  # 子批次观测，形状 (sub_batch_size, 100)
            if sub_obs.numel() == 0:
                continue
            _teacher_obs[mask] = self.teacher_obs_normalizer[uid](sub_obs)

        # 1. rollout 时使用后验构建 latent,不更新kl
        _, _, z, _ = self._compute_latent_dist(student_obs, _teacher_obs, use_prior_only=False, need_kl=False)
        self._update_distribution(student_obs, z)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict, need_kl: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """确定性推理（用于损失计算，返回动作均值和 KL）。
        
        Args:
            obs: 观测字典。
            only_action: 如果 True，仅返回动作均值（用于部署）。
        
        Returns:
            动作均值, KL 值（用于蒸馏损失，如果 only_action=False）。
        """
        student_obs = self.get_student_obs(obs)
        student_obs = self.student_obs_normalizer(student_obs)
        if need_kl: # 2. train inference,使用prior构建latent
            teacher_obs = self.get_teacher_obs(obs)
            motion_id = self.get_motion_id(obs).squeeze(1)# 形状从 (4096, 1) 转换为 (4096,)
            unique_ids, _ = torch.unique(motion_id, return_inverse=True)
            _teacher_obs = torch.zeros_like(teacher_obs)
            for uid in unique_ids:
                mask = (motion_id == uid)  # 布尔掩码，形状 (4096,)
                sub_obs = teacher_obs[mask]  # 子批次观测，形状 (sub_batch_size, 100)
                if sub_obs.numel() == 0:
                    continue
                _teacher_obs[mask] = self.teacher_obs_normalizer[uid](sub_obs)
            _, _, z, kl = self._compute_latent_dist(student_obs, teacher_obs, use_prior_only=True,need_kl=need_kl)
        else: # 3. eval inference ,使用prior构建latent
            _, _, z, kl = self._compute_latent_dist(student_obs, None, use_prior_only=True,need_kl=need_kl)


        # 计算动作均值
        decoder_input = torch.cat([student_obs, z], dim=-1)
        action_mean = self.decoder(decoder_input)
        if not need_kl:
            return action_mean
        else:
            return action_mean, kl

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
        # obs = self.teacher_obs_normalizer(obs)
        # with torch.no_grad():
        #     return self.teacher(obs)

    def get_student_obs(self, obs: TensorDict) -> torch.Tensor:
        """获取学生观测。"""
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["policy"]]
        return torch.cat(obs_list, dim=-1)

    def get_teacher_obs(self, obs: TensorDict) -> torch.Tensor:
        """获取教师观测。"""
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["teacher"]]
        return torch.cat(obs_list, dim=-1)

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
        
    def get_hidden_states(self) -> tuple[HiddenState, HiddenState]:
        return None, None

    def detach_hidden_states(self, dones: torch.Tensor | None = None) -> None:
        pass

    def train(self, mode: bool = True) -> None:
        super().train(mode)
        # 确保教师在评估模式
        self.teacher.eval()
        self.teacher_obs_normalizer.eval()

    def update_normalization(self, obs: TensorDict) -> None:
        if self.student_obs_normalization:
            student_obs = self.get_student_obs(obs)
            self.student_obs_normalizer.update(student_obs)
            
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

        # for i, state_dict in enumerate(state_dicts):
        #     if any("actor" in key for key in state_dict["model_state_dict"]):  # 从 RL 加载教师
        #         teacher_state_dict = {}
        #         teacher_obs_normalizer_state_dict = {}
        #         for key, value in state_dict["model_state_dict"].items():
        #             if "actor." in key:
        #                 teacher_state_dict[key.replace("actor.", "")] = value
        #             if "actor_obs_normalizer." in key:
        #                 teacher_obs_normalizer_state_dict[key.replace("actor_obs_normalizer.", "")] = value
        #         self.teacher[i].load_state_dict(teacher_state_dict, strict=strict)
        #         self.teacher_obs_normalizer[i].load_state_dict(teacher_obs_normalizer_state_dict, strict=strict)
        #         self.teacher[i].eval()
        #         self.teacher_obs_normalizer[i].eval()
        self.loaded_teacher = True
        return False