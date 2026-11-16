"""Train the ResNet-18 fetal plane classifier on FETAL_PLANES_DB.

Example (Colab GPU):
    python -m fetalscan.train_cls --data /content/FETAL_PLANES_ZENODO --epochs 15 --out runs/cls
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .datasets import PLANE_CLASSES, PlanesDataset, patient_split, read_planes_csv
from .metrics import classification_report
from .models import build_classifier
from .utils import MaybeMLflow, pick_device, save_json, seed_everything


def class_weights(labels, n: int) -> torch.Tensor:
    counts = np.bincount(np.asarray(labels), minlength=n).astype(float)
    w = counts.sum() / (n * np.maximum(counts, 1))
    return torch.tensor(w, dtype=torch.float32)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        logits = model(x.to(device))
        ps.extend(logits.argmax(1).cpu().tolist())
        ys.extend(y.tolist())
    return ys, ps


def run(args) -> dict:
    seed_everything(args.seed)
    device = pick_device(args.device)
    df = read_planes_csv(args.data)
    if args.limit:
        df = df.groupby("label", group_keys=False).head(args.limit)
    tr, va, te = patient_split(df, seed=args.seed)
    print(f"train={len(tr)} val={len(va)} test={len(te)} device={device}")

    mk = lambda d, train: DataLoader(  # noqa: E731
        PlanesDataset(d, args.size, train=train, seed=args.seed),
        batch_size=args.batch_size, shuffle=train, num_workers=args.workers, pin_memory=device.type == "cuda",
    )
    dl_tr, dl_va, dl_te = mk(tr, True), mk(va, False), mk(te, False)

    n = len(PLANE_CLASSES)
    model = build_classifier(n, pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(tr["label"], n).to(device), label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    tracker = MaybeMLflow(args.mlflow, "fetalscan-cls", vars(args))

    os.makedirs(args.out, exist_ok=True)
    best, best_path, bad = -1.0, os.path.join(args.out, "best.pt"), 0
    last_path, start = os.path.join(args.out, "last.pt"), 1
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=device)["model"])
        print(f"initialized weights from {args.init}")
    if args.resume and os.path.exists(last_path):
        st = torch.load(last_path, map_location=device)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        best, bad, start = st["best"], st["bad"], st["epoch"] + 1
        print(f"resumed from epoch {st['epoch']} (best={best:.4f})")
    for epoch in range(start, args.epochs + 1):
        model.train()
        t0, total = time.time(), 0.0
        for x, y in dl_tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item() * len(y)
        sched.step()
        ys, ps = predict(model, dl_va, device)
        rep = classification_report(ys, ps, n)
        train_loss = total / max(len(tr), 1)
        print(f"epoch {epoch:02d} loss={train_loss:.4f} val_acc={rep['accuracy']:.4f} "
              f"val_macroF1={rep['macro_f1']:.4f} ({time.time() - t0:.0f}s)")
        tracker.log({"train_loss": train_loss, "val_acc": rep["accuracy"], "val_macro_f1": rep["macro_f1"]}, epoch)
        if rep["macro_f1"] > best:
            best, bad = rep["macro_f1"], 0
            torch.save({"model": model.state_dict(), "classes": PLANE_CLASSES, "size": args.size}, best_path)
        else:
            bad += 1
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": epoch, "best": best, "bad": bad}, last_path)
        if bad >= args.patience:
            print("early stopping")
            break

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    ys, ps = predict(model, dl_te, device)
    test = classification_report(ys, ps, n)
    result = {"best_val_macro_f1": best, "test": test, "classes": PLANE_CLASSES, "n_test": len(te)}
    save_json(result, os.path.join(args.out, "metrics.json"))
    tracker.log({"test_acc": test["accuracy"], "test_macro_f1": test["macro_f1"]})
    tracker.end()
    _plot_cm(np.array(test["confusion_matrix"]), os.path.join(args.out, "confusion_matrix.png"))
    print(f"TEST accuracy={test['accuracy']:.4f} macro_F1={test['macro_f1']:.4f}")
    return result


def _plot_cm(cm: np.ndarray, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    short = [c.replace("Fetal ", "").replace("Maternal ", "") for c in PLANE_CLASSES]
    ax.set_xticks(range(len(short)), short, rotation=45, ha="right")
    ax.set_yticks(range(len(short)), short)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="folder with FETAL_PLANES_DB_data.csv and Images/")
    p.add_argument("--out", default="runs/cls")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--limit", type=int, default=0, help="max images per class (quick experiments)")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--mlflow", action="store_true")
    p.add_argument("--resume", action="store_true", help="continue from <out>/last.pt if it exists")
    p.add_argument("--init", default="", help="start from these model weights (e.g. a previous best.pt)")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
