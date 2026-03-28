from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.modules.actor_critic_single_fsq_distillation import (
    ActorCriticSingleFSQDistillation,
)
from rsl_rl.modules.fsq_components import FSQAutoEncoder


def _make_obs(batch_size: int = 4) -> TensorDict:
    return TensorDict(
        {
            "policy": torch.randn(batch_size, 3),
            "critic": torch.randn(batch_size, 2),
            "policy_window": torch.randn(batch_size, 5),
            "critic_window": torch.randn(batch_size, 4),
            "teacher": torch.randn(batch_size, 6),
        },
        batch_size=[batch_size],
    )


def _make_policy() -> ActorCriticSingleFSQDistillation:
    obs = _make_obs()
    obs_groups = {
        "policy": ["policy"],
        "critic": ["critic"],
        "policy_window": ["policy_window"],
        "critic_window": ["critic_window"],
        "teacher": ["teacher"],
    }
    return ActorCriticSingleFSQDistillation(
        obs=obs,
        obs_groups=obs_groups,
        num_actions=2,
        detach_fsq_latent_in_policy=True,
    )


def test_fsq_autoencoder_can_detach_policy_latent() -> None:
    model = FSQAutoEncoder(encoder_input_dim=5, target_dim=5)
    encoder_input = torch.randn(4, 5)
    latent = model.latent_for_policy(encoder_input, detach=True)
    assert latent.requires_grad is False
    encoded = model.encoder_forward(encoder_input)
    assert encoded.z_q.shape == (4, model.embedding_dim)


def test_policy_loss_does_not_backprop_into_student_fsq() -> None:
    policy = _make_policy()
    obs = _make_obs()

    policy.act(obs)
    policy.action_mean.sum().backward()

    fsq_grads = [
        parameter.grad
        for parameter in policy.student_fsq.parameters()
        if parameter.requires_grad
    ]
    assert all(grad is None for grad in fsq_grads)


def test_fsq_losses_still_backprop_into_student_fsq() -> None:
    policy = _make_policy()
    obs = _make_obs()

    fsq_losses = policy.compute_fsq_losses(obs)
    fsq_losses["actor"].loss.backward()

    fsq_grads = [
        parameter.grad
        for parameter in policy.student_fsq.parameters()
        if parameter.requires_grad
    ]
    assert any(grad is not None for grad in fsq_grads)
