"""Train the U-Net fetal head segmenter on HC18 and measure head circumference.

Example (Colab GPU):
    python -m fetalscan.train_seg --data /content/HC18 --epochs 40 --out runs/seg
"""
from __future__ import annotations

import argparse
import os
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from .biometry import draw_ellipse, fit_ellipse, head_circumference_mm
from .datasets import HC18Dataset, random_split, read_hc18_csv
from .metrics import bootstrap_ci, dice
from .models import UNet, seg_loss
from .utils import MaybeMLflow, pick_device, save_json, seed_everything


@torch.no_grad()
def evaluate(model, ds: HC18Dataset, device, threshold: float = 0.5, vis_dir: str | None = None, n_vis: int = 8):
    model.eval()
    dices, errs, fails = [], [], 0
    for i in range(len(ds)):
        x, y, pix_mm, hc_ref = ds[i]
        prob = torch.sigmoid(model(x[None].to(device)))[0, 0].cpu().numpy()
        pred = (prob > threshold).astype(np.uint8)
        dices.append(dice(pred, y[0].numpy()))
        hc = head_circumference_mm(pred, pix_mm)
        if hc is None:
            fails += 1
        else:
            errs.append(abs(hc - hc_ref))
        if vis_dir and i < n_vis:
            os.makedirs(vis_dir, exist_ok=True)
            img = (x[0].numpy() * 255).astype(np.uint8)
            e = fit_ellipse(pred)
            vis = draw_ellipse(img, e) if e else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(os.path.join(vis_dir, f"sample_{i:03d}.png"), vis)
    out = {
        "dice_mean": float(np.mean(dices)) if dices else float("nan"),
        "dice_ci95": bootstrap_ci(dices),
        "hc_mae_mm": float(np.mean(errs)) if errs else float("nan"),
        "hc_mae_ci95": bootstrap_ci(errs),
        "hc_within_3mm": float(np.mean(np.array(errs) <= 3.0)) if errs else float("nan"),
        "ellipse_fit_failures": fails,
        "n": len(ds),
    }
    return out


def run(args) -> dict:
    seed_everything(args.seed)
    device = pick_device(args.device)
    df = read_hc18_csv(args.data)
    if args.limit:
        df = df.head(args.limit)
    tr, va, te = random_split(df, seed=args.seed)
    print(f"train={len(tr)} val={len(va)} test={len(te)} device={device}")

    ds_tr = HC18Dataset(tr, args.size, train=True, seed=args.seed)
    ds_va, ds_te = HC18Dataset(va, args.size), HC18Dataset(te, args.size)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                       pin_memory=device.type == "cuda")

    model = UNet(base=args.base).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    tracker = MaybeMLflow(args.mlflow, "fetalscan-seg", vars(args))

    os.makedirs(args.out, exist_ok=True)
    best, best_path, bad = -1.0, os.path.join(args.out, "best.pt"), 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total = time.time(), 0.0
        for x, y, _, _ in dl_tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(x)
            loss = seg_loss(logits.float(), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item() * len(x)
        sched.step()
        val = evaluate(model, ds_va, device)
        train_loss = total / max(len(tr), 1)
        print(f"epoch {epoch:02d} loss={train_loss:.4f} val_dice={val['dice_mean']:.4f} "
              f"val_HC_MAE={val['hc_mae_mm']:.2f}mm ({time.time() - t0:.0f}s)")
        tracker.log({"train_loss": train_loss, "val_dice": val["dice_mean"], "val_hc_mae_mm": val["hc_mae_mm"]}, epoch)
        if val["dice_mean"] > best:
            best, bad = val["dice_mean"], 0
            torch.save({"model": model.state_dict(), "size": args.size, "base": args.base}, best_path)
        else:
            bad += 1
            if bad >= args.patience:
                print("early stopping")
                break

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    test = evaluate(model, ds_te, device, vis_dir=os.path.join(args.out, "examples"))
    result = {"best_val_dice": best, "test": test}
    save_json(result, os.path.join(args.out, "metrics.json"))
    tracker.log({"test_dice": test["dice_mean"], "test_hc_mae_mm": test["hc_mae_mm"]})
    tracker.end()
    print(f"TEST dice={test['dice_mean']:.4f} HC_MAE={test['hc_mae_mm']:.2f} mm "
          f"(95% CI {test['hc_mae_ci95'][0]:.2f}-{test['hc_mae_ci95'][1]:.2f}) within3mm={test['hc_within_3mm']:.2%}")
    return result


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="HC18 folder with training_set/ and the pixel-size CSV")
    p.add_argument("--out", default="runs/seg")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--base", type=int, default=32, help="U-Net base channels")
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--mlflow", action="store_true")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
