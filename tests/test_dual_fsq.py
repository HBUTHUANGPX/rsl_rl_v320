from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.modules.actor_critic_dual_fsq import ActorCriticDualFSQ


def _make_obs(batch_size: int = 4) -> TensorDict:
    return TensorDict(
        {
            "policy": torch.randn(batch_size, 6),
            "critic": torch.randn(batch_size, 7),
            "human_fsq_window": torch.randn(batch_size, 11),
            "robot_fsq_window": torch.randn(batch_size, 9),
        },
        batch_size=[batch_size],
    )


def _make_policy() -> ActorCriticDualFSQ:
    obs = _make_obs()
    obs_groups = {
        "policy": ["policy"],
        "critic": ["critic"],
        "human_fsq_window": ["human_fsq_window"],
        "robot_fsq_window": ["robot_fsq_window"],
    }
    return ActorCriticDualFSQ(
        obs=obs,
        obs_groups=obs_groups,
        num_actions=3,
        latent_dim=8,
        robot_encoder_hidden_dims=[16],
        human_encoder_hidden_dims=[16],
        decoder_hidden_dims=[16],
        fsq_levels=5,
        actor_hidden_dims=[16],
        critic_hidden_dims=[16],
        activation="elu",
    )


def test_dual_fsq_policy_returns_actions_values_and_losses() -> None:
    policy = _make_policy()
    obs = _make_obs()

    actor_out = policy.act(obs, reconstruct=True)
    critic_out = policy.evaluate(obs, reconstruct=True)

    assert actor_out["action"].shape == (4, 3)
    assert critic_out["value"].shape == (4, 1)
    assert actor_out["fsq_out"]["loss"].ndim == 0
    assert critic_out["fsq_out"]["loss"] is actor_out["fsq_out"]["loss"]
    assert set(actor_out["fsq_out"]["terms"]) == {
        "robot_recon",
        "human_recon",
        "latent_align",
        "cycle_latent",
    }


def test_policy_gradient_uses_human_encoder_without_touching_robot_decoder_path() -> None:
    policy = _make_policy()
    obs = _make_obs()

    policy.act(obs)
    policy.action_mean.sum().backward()

    human_grads = [
        parameter.grad
        for parameter in policy.dual_fsq.human_encoder.parameters()
        if parameter.requires_grad
    ]
    robot_grads = [
        parameter.grad
        for parameter in policy.dual_fsq.robot_encoder.parameters()
        if parameter.requires_grad
    ]
    decoder_grads = [
        parameter.grad
        for parameter in policy.dual_fsq.decoder.parameters()
        if parameter.requires_grad
    ]

    assert any(grad is not None for grad in human_grads)
    assert all(grad is None for grad in robot_grads)
    assert all(grad is None for grad in decoder_grads)
