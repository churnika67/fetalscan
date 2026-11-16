"""Optuna hyperparameter search for the plane classifier (short runs, maximize val macro F1).

    python -m fetalscan.tune --data /content/FETAL_PLANES_ZENODO --trials 12 --epochs 4 --limit 400
"""
from __future__ import annotations

import argparse
import os

from .train_cls import parser as cls_parser
from .train_cls import run as cls_run
from .utils import save_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--trials", type=int, default=12)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--limit", type=int, default=400)
    p.add_argument("--out", default="runs/tune")
    p.add_argument("--no-pretrained", action="store_true")
    a = p.parse_args()

    import optuna

    def objective(trial: "optuna.Trial") -> float:
        args = cls_parser().parse_args(["--data", a.data])
        args.epochs, args.limit, args.patience = a.epochs, a.limit, a.epochs
        args.no_pretrained = a.no_pretrained
        args.lr = trial.suggest_float("lr", 1e-5, 3e-3, log=True)
        args.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        args.batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        args.size = trial.suggest_categorical("size", [160, 224])
        args.out = os.path.join(a.out, f"trial_{trial.number}")
        return cls_run(args)["best_val_macro_f1"]

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=a.trials)
    print("best", study.best_value, study.best_params)
    save_json({"best_val_macro_f1": study.best_value, "best_params": study.best_params,
               "trials": [{"params": t.params, "value": t.value} for t in study.trials]},
              os.path.join(a.out, "study.json"))


if __name__ == "__main__":
    main()
