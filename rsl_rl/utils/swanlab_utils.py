# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from dataclasses import asdict, is_dataclass
from typing import Any

from torch.utils.tensorboard import SummaryWriter

try:
    import swanlab
except ModuleNotFoundError:
    raise ModuleNotFoundError("swanlab package is required to log to SwanLab.") from None


class SwanLabSummaryWriter(SummaryWriter):
    """Summary writer for SwanLab."""

    def __init__(self, log_dir: str, flush_secs: int, cfg: dict) -> None:
        super().__init__(log_dir, flush_secs)

        self.run_name = os.path.split(log_dir)[-1]
        self.project = cfg.get("swanlab_project") or cfg.get("wandb_project") or "rsl_rl"
        self.workspace = cfg.get("swanlab_workspace") or os.environ.get("SWANLAB_WORKSPACE")
        self.mode = cfg.get("swanlab_mode") or os.environ.get("SWANLAB_MODE")
        self.swanlab_log_dir = cfg.get("swanlab_log_dir") or os.path.join(log_dir, "swanlab")

        init_kwargs = {
            "project": self.project,
            "workspace": self.workspace,
            "experiment_name": self.run_name,
            "config": {"log_dir": log_dir},
            "logdir": self.swanlab_log_dir,
        }
        if self.mode:
            init_kwargs["mode"] = self.mode

        self.run = swanlab.init(**init_kwargs)

    def store_config(self, env_cfg: dict | object, train_cfg: dict) -> None:
        self.run.config.update(
            {
                "runner_cfg": train_cfg,
                "policy_cfg": train_cfg["policy"],
                "alg_cfg": train_cfg["algorithm"],
                "env_cfg": self._to_config_dict(env_cfg),
            }
        )

    def add_scalar(
        self,
        tag: str,
        scalar_value: float,
        global_step: int | None = None,
        walltime: float | None = None,
        new_style: bool = False,
    ) -> None:
        super().add_scalar(
            tag,
            scalar_value,
            global_step=global_step,
            walltime=walltime,
            new_style=new_style,
        )
        swanlab.log({tag: self._to_scalar(scalar_value)}, step=global_step)

    def stop(self) -> None:
        self.close()
        swanlab.finish()

    def save_model(self, model_path: str, it: int) -> None:
        self.save_file(model_path)

    def save_file(self, path: str) -> None:
        swanlab.save(path, base_path=os.path.dirname(path), policy="now")

    @staticmethod
    def _to_config_dict(cfg: dict | object) -> dict[str, Any]:
        if isinstance(cfg, dict):
            return cfg
        if hasattr(cfg, "to_dict"):
            return cfg.to_dict()
        if is_dataclass(cfg):
            return asdict(cfg)
        if hasattr(cfg, "__dict__"):
            return dict(vars(cfg))
        return {"value": str(cfg)}

    @staticmethod
    def _to_scalar(value: Any) -> Any:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "item"):
            try:
                return value.item()
            except ValueError:
                pass
        return value
