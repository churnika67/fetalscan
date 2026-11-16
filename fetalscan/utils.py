from __future__ import annotations

import json
import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(pref: str = "auto") -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


class MaybeMLflow:
    """Logs to MLflow if installed and enabled; otherwise a no-op, so MLflow stays optional."""

    def __init__(self, enabled: bool, experiment: str, params: dict):
        self.on = False
        if not enabled:
            return
        try:
            import mlflow  # noqa: F401
        except ImportError:
            print("mlflow not installed; skipping experiment tracking")
            return
        import mlflow

        mlflow.set_experiment(experiment)
        mlflow.start_run()
        mlflow.log_params(params)
        self.mlflow, self.on = mlflow, True

    def log(self, metrics: dict, step: int | None = None):
        if self.on:
            self.mlflow.log_metrics({k: float(v) for k, v in metrics.items()}, step=step)

    def end(self):
        if self.on:
            self.mlflow.end_run()
