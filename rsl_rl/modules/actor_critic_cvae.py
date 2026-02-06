from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal
from typing import Any, NoReturn, Tuple

from rsl_rl.networks import MLP, EmpiricalNormalization


class ActorCritic_CVAE(nn.Module):
    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        motion_run_names: list[str] = [""], 
        latent_dim: int = 64,  # CVAE 潜在空间维度
        beta_kl: float = 0.1,  # KL 损失权重
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        teacher_obs_normalization: bool = False, 
        actor_hidden_dims: tuple[int] | list[int] = [256, 256, 256],    # 用于 decoder(actor)
        critic_hidden_dims: tuple[int] | list[int] = [256, 256, 256],   # 用于 critic
        teacher_hidden_dims: tuple[int] | list[int] = [256, 256, 256],  # 用于 teacher
        prior_hidden_dims: tuple[int] | list[int] = [1024, 512, 128],   # 用于 prior
        encoder_hidden_dims: tuple[int] | list[int] = [512, 256, 128],  # 用于 encoder
        activation: str = "elu",
        init_noise_std: float = 0.1,
        noise_std_type: str = "scalar",
        normalize_mu: bool = False,  
        z_scale_factor: float = 1.0,  # z 的缩放因子
        **kwargs: dict[str, Any],
    ) -> None:
        """初始化 CVAE-based actor-critic 模块，用于 PPO_Distil。
        
        修改点：
        - 融合 ActorCritic 和 actorTeacher_CVAE：Actor 使用 CVAE 生成动作，Critic 保持原样。
        - 新增教师网络，用于蒸馏中的 DKL 计算。
        - 支持规范化教师观测。

        Args:
            obs: 观测字典。
            obs_groups: 观测组映射。
            num_actions: 动作维度。
            motion_run_names: motion group name列表,代表teacher的数量
            latent_dim: 潜在变量维度（论文推荐 64）。
            beta_kl: KL 损失权重（论文推荐 0.1）。
            actor_obs_normalization: 是否规范化 actor 观测。
            critic_obs_normalization: 是否规范化 critic 观测。
            teacher_obs_normalization: 是否规范化教师观测。
            actor_hidden_dims: 解码器(学生)的隐藏层维度。
            critic_hidden_dims: Critic 的隐藏层维度。
            teacher_hidden_dims: 教师的隐藏层维度。
            prior_hidden_dims: prior的隐藏层维度。
            encoder_hidden_dims: encoder的隐藏层维度。
            activation: 激活函数。
            init_noise_std: 初始动作噪声标准差。
            noise_std_type: 噪声类型 ('scalar' 或 'log')。
            normalize_mu: 是否对潜在均值 mu 进行经验规范化。
            z_scale_factor: z 的缩放因子。
            kwargs: 忽略的额外参数。
        """
        if kwargs:
            print(
                "ActorCritic_CVAE.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs])
            )
        super().__init__()

        self.loaded_teacher = False  # 表示教师是否已加载
        self.latent_dim = latent_dim
        self.beta_kl = beta_kl  # KL 权重，用于训练
        self.z_scale_factor = z_scale_factor  # z 缩放因子
        self.num_actions = num_actions
        self.motion_run_names = motion_run_names
        self.obs_groups = obs_groups

        # =========== 初始化各个网络模块的obs尺寸 ================

        # actor 模块 obs尺寸 
        num_actor_obs = 0
        for obs_group in obs_groups["policy"]:
            assert len(obs[obs_group].shape) == 2, "仅支持 1D 观测。"
            num_actor_obs += obs[obs_group].shape[-1]

        # teacher 模块 obs尺寸 
        num_teacher_obs = 0
        for obs_group in obs_groups["teacher"]:
            assert len(obs[obs_group].shape) == 2, "仅支持 1D 观测。"
            num_teacher_obs += obs[obs_group].shape[-1]

        # critic 模块 obs尺寸 
        num_critic_obs = 0
        for obs_group in obs_groups["critic"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            num_critic_obs += obs[obs_group].shape[-1]

        # =========== 创建各个网络模块 ============
        # 创建prior网络，input 尺寸为 actor 模块的 obs尺寸，output 尺寸为 2 * latent_dim -> [mu_p, logvar_p]
        self.prior_network = MLP(num_actor_obs, 2 * latent_dim, prior_hidden_dims, activation)
        print(f"Prior Network: {self.prior_network}")

        # 创建encoder网络，input 尺寸为 teacher 模块的 obs尺寸，output 尺寸为 2 * latent_dim -> [mu_e, logvar_e]
        self.encoder_network = MLP(num_teacher_obs, 2 * latent_dim, encoder_hidden_dims, activation)
        print(f"Encoder Network: {self.encoder_network}")
        
        # 创建actor(decoder)网络，input 尺寸为 actor 模块的 obs尺寸 + 潜在变量尺寸，output 尺寸为 动作维度
        self.actor = MLP(num_actor_obs + latent_dim, num_actions, actor_hidden_dims, activation)
        print(f"Actor(Decoder) MLP: {self.actor}")

        # 创建critic网络，input 尺寸为 critic 模块的 obs尺寸，output 尺寸为 1
        self.critic = MLP(num_critic_obs, 1, critic_hidden_dims, activation)
        print(f"Critic MLP: {self.critic}")

        # actor(decoder) observation normalization
        self.actor_obs_normalization = actor_obs_normalization
        if self.actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()

        # Critic observation normalization
        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()

        self.normalize_mu = normalize_mu
        if normalize_mu:
            self.mu_normalizer = EmpiricalNormalization(latent_dim)  # 潜在维度作为形状
        else:
            self.mu_normalizer = nn.Identity()
        
        # 多教师支持
        self.teacher = nn.ModuleList()
        self.teacher_obs_normalizer = nn.ModuleList()
        self.teacher_num = len(motion_run_names)
        for i in range(self.teacher_num):
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

        # =========================Action noise ====================
        self.noise_std_type = noise_std_type
        
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            self.teacher_std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
            self.teacher_log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution
        # Note: Populated in update_distribution
        self.distribution = None

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

    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return (mu + eps * std) * self.z_scale_factor
    
    def _update_distribution(self, actor_obs: torch.Tensor, z: torch.Tensor) -> None:
        if torch.isnan(actor_obs).any():
            raise ValueError(f"张量中存在 NaN 值")
        # 连接学生观测和 z
        actor_input = torch.cat([actor_obs, z], dim=-1)
        # 计算动作均值
        mean = self.actor(actor_input)
        # Compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # Create distribution
        self.distribution = Normal(mean, std)

    def _update_teacher_distribution(self, obs: TensorDict) -> None:
        mean = self.evaluate_privileged_actions(obs)
        # Compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.teacher_std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.teacher_log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # Create distribution
        self.teacher_distribution = Normal(mean, std)

    def _compute_latent_dist(
        self, student_obs: torch.Tensor, teacher_obs: torch.Tensor, use_prior_only: bool = False, need_kl: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        actor_obs = self.get_actor_obs(obs)
        actor_obs = self.actor_obs_normalizer(actor_obs)
        teacher_obs = self.get_teacher_obs(obs)
        motion_group = self.get_motion_group(obs).squeeze(1)# 形状从 (4096, 1) 转换为 (4096,)
        unique_ids, _ = torch.unique(motion_group, return_inverse=True)
        _teacher_obs = torch.zeros_like(teacher_obs)
        for uid in unique_ids:
            mask = (motion_group == uid)  # 布尔掩码，形状 (4096,)
            sub_obs = teacher_obs[mask]  # 子批次观测，形状 (sub_batch_size, 100)
            if sub_obs.numel() == 0:
                continue
            _teacher_obs[mask] = self.teacher_obs_normalizer[uid](sub_obs)

        # 1. rollout 时使用后验构建 latent,不更新kl
        _, _, z, _ = self._compute_latent_dist(actor_obs, _teacher_obs, use_prior_only=False, need_kl=False)
        self._update_distribution(actor_obs, z)
        self._update_teacher_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict, need_kl: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        actor_obs = self.get_actor_obs(obs)
        actor_obs = self.actor_obs_normalizer(actor_obs)
        if need_kl: # 2. train inference,使用prior构建latent
            teacher_obs = self.get_teacher_obs(obs)
            motion_group = self.get_motion_group(obs).squeeze(1)# 形状从 (4096, 1) 转换为 (4096,)
            unique_ids, _ = torch.unique(motion_group, return_inverse=True)
            _teacher_obs = torch.zeros_like(teacher_obs)
            for uid in unique_ids:
                mask = (motion_group == uid)  # 布尔掩码，形状 (4096,)
                sub_obs = teacher_obs[mask]  # 子批次观测，形状 (sub_batch_size, 100)
                if sub_obs.numel() == 0:
                    continue
                _teacher_obs[mask] = self.teacher_obs_normalizer[uid](sub_obs)
            _, _, z, kl = self._compute_latent_dist(actor_obs, _teacher_obs, use_prior_only=True,need_kl=need_kl)
        else: # 3. eval inference ,使用prior构建latent
            _, _, z, kl = self._compute_latent_dist(actor_obs, None, use_prior_only=True,need_kl=need_kl)

        # 计算动作均值
        actor_input = torch.cat([actor_obs, z], dim=-1)
        action_mean = self.actor(actor_input)
        if not need_kl:
            return action_mean
        else:
            return action_mean, kl
        
    def evaluate_privileged_actions(self, obs: TensorDict) -> torch.Tensor:
        teacher_obs = self.get_teacher_obs(obs)
        critic_obs = self.get_critic_obs(obs)
        critic_obs = self.critic_obs_normalizer(critic_obs)
        motion_group = self.get_motion_group(obs).squeeze(1)# 形状从 (4096, 1) 转换为 (4096,)
        # 识别唯一教师索引
        unique_ids, inverse_indices = torch.unique(motion_group, return_inverse=True)

        # 预分配动作张量（假设动作维度为 action_dim，根据实际替换）
        actions = torch.zeros(teacher_obs.shape[0], self.num_actions, device=teacher_obs.device)
        # 根据 motion_group 选择对应教师
        with torch.no_grad():
            for uid in unique_ids:
                mask = (motion_group == uid)  # 布尔掩码，形状 (4096,)
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
        
    def evaluate_values(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        critic_obs = self.get_critic_obs(obs)
        critic_obs = self.critic_obs_normalizer(critic_obs)
        return self.critic(critic_obs)
    
    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = []
        for obs_group in self.obs_groups["policy"]:
            if torch.isnan(obs[obs_group]).any():
                raise ValueError(f"{obs_group} 张量中存在 NaN 值")
            obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1)

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["critic"]]
        return torch.cat(obs_list, dim=-1)
    
    def get_teacher_obs(self, obs: TensorDict) -> torch.Tensor:
        """获取教师观测。"""
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["teacher"]]
        return torch.cat(obs_list, dim=-1)

    def get_motion_group(self, obs: TensorDict) -> torch.Tensor:
        """获取 motion_group 观测（如果存在）。"""
        if "motion_group" in self.obs_groups and self.obs_groups["motion_group"]:
            obs_list = []
            for obs_group in self.obs_groups["motion_group"]:
                obs_list.append(obs[obs_group])
            # obs_list = [obs[obs_group] for obs_group in self.obs_groups["motion_group"]]
            return torch.cat(obs_list, dim=-1).to(torch.int64)
        else:
            raise ValueError("观测组中未定义 'motion_group'")
        
    def train(self, mode: bool = True) -> None:
        super().train(mode)
        # 确保教师在评估模式
        self.teacher.eval()
        self.teacher_obs_normalizer.eval()

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)
    
    def update_normalization(self, obs: TensorDict) -> None:
        if self.actor_obs_normalization:
            actor_obs = self.get_actor_obs(obs)
            self.actor_obs_normalizer.update(actor_obs)
        if self.critic_obs_normalization:
            critic_obs = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(critic_obs)

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
        if any("teacher" in key for key in state_dict["model_state_dict"]):  # 从 RL 加载教师
            for i in range(count):
                teacher_state_dict = {}
                teacher_obs_normalizer_state_dict = {}
                for key, value in state_dict["model_state_dict"].items():
                    if ("teacher."+str(i)+".") in key:
                        teacher_state_dict[key.replace("teacher."+str(i)+".", "")] = value
                    if ("teacher_obs_normalizer."+str(i)+".") in key:
                        teacher_obs_normalizer_state_dict[key.replace("teacher_obs_normalizer."+str(i)+".", "")] = value
                self.teacher[i].load_state_dict(teacher_state_dict, strict=strict)
                self.teacher_obs_normalizer[i].load_state_dict(teacher_obs_normalizer_state_dict, strict=strict)
                self.teacher[i].eval()
                self.teacher_obs_normalizer[i].eval()
            self.loaded_teacher = True
        if any("actor_obs_normalizer" in key for key in state_dict["model_state_dict"]):  # Load parameters from distillation training
            actor_obs_normalizer_state_dict = {}
            for key, value in state_dict["model_state_dict"].items():
                if "actor_obs_normalizer." in key:
                    actor_obs_normalizer_state_dict[key.replace("actor_obs_normalizer.", "")] = value
            self.actor_obs_normalizer.load_state_dict(actor_obs_normalizer_state_dict, strict=strict)
        else:
            print("警告：未在 state_dict 中找到 'actor_obs_normalizer' 参数。")
        if any("actor" in key for key in state_dict["model_state_dict"]):  # Load parameters from distillation training
            actor_state_dict = {}
            for key, value in state_dict["model_state_dict"].items():
                if "actor." in key:
                    actor_state_dict[key.replace("actor.", "")] = value
            self.actor.load_state_dict(actor_state_dict, strict=strict)
        if any("prior_network" in key for key in state_dict["model_state_dict"]):  # Load parameters from distillation training
            prior_network_state_dict = {}
            for key, value in state_dict["model_state_dict"].items():
                if "prior_network." in key:
                    prior_network_state_dict[key.replace("prior_network.", "")] = value
            self.prior_network.load_state_dict(prior_network_state_dict, strict=strict)
        return False
    
    def load_state_dicts(self, state_dicts: dict[dict | None], strict: bool = True) -> bool:
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