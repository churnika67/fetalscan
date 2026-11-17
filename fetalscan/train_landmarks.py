"""Train the 4-point fetal head landmark detector (BPD and OFD endpoints) on HC18.

    python -m fetalscan.train_landmarks --data data/HC18 --init-seg runs/seg/best.pt --out runs/landmarks
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .datasets import random_split, read_hc18_csv
from .landmarks import LANDMARKS, HC18LandmarkDataset, decode_heatmaps, heatmap_loss
from .metrics import bootstrap_ci
from .models import UNet
from .utils import pick_device, save_json, seed_everything


@torch.no_grad()
def evaluate(model, ds, device) -> dict:
    model.eval()
    radial, bpd_err, ofd_err = [], [], []
    for i in range(len(ds)):
        x, _, gt, pix = ds[i]
        hm = model(x[None].to(device))[0].cpu().numpy()
        pred = decode_heatmaps(hm)
        gt = gt.numpy()
        radial.append(np.linalg.norm(pred - gt, axis=1) * pix)
        length = lambda p, a, b: np.linalg.norm(p[a] - p[b]) * pix  # noqa: E731
        bpd_err.append(abs(length(pred, 0, 1) - length(gt, 0, 1)))
        ofd_err.append(abs(length(pred, 2, 3) - length(gt, 2, 3)))
    radial = np.array(radial)  # N x 4, mm
    return {
        "mean_radial_error_mm": float(radial.mean()),
        "mre_ci95": bootstrap_ci(radial.mean(1)),
        "per_landmark_mre_mm": dict(zip(LANDMARKS, radial.mean(0).round(3).tolist())),
        "success_rate_2mm": float((radial <= 2).mean()),
        "success_rate_4mm": float((radial <= 4).mean()),
        "bpd_length_mae_mm": float(np.mean(bpd_err)),
        "ofd_length_mae_mm": float(np.mean(ofd_err)),
        "n": len(ds),
    }


def run(args) -> dict:
    seed_everything(args.seed)
    device = pick_device(args.device)
    df = read_hc18_csv(args.data)
    if args.limit:
        df = df.head(args.limit)
    tr, va, te = random_split(df, seed=args.seed)  # same split as segmentation
    ds_tr = HC18LandmarkDataset(tr, args.size, train=True, sigma=args.sigma, seed=args.seed)
    ds_va, ds_te = HC18LandmarkDataset(va, args.size, sigma=args.sigma), HC18LandmarkDataset(te, args.size, sigma=args.sigma)
    dl = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    model = UNet(out_ch=len(LANDMARKS), base=args.base).to(device)
    if args.init_seg:  # transfer the encoder/decoder learned for head segmentation; only the 1x1 head is new
        sd = {k: v for k, v in torch.load(args.init_seg, map_location=device)["model"].items() if not k.startswith("head.")}
        print("warm start:", model.load_state_dict(sd, strict=False))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))
    os.makedirs(args.out, exist_ok=True)
    best, best_path, last_path, start, bad = float("inf"), os.path.join(args.out, "best.pt"), os.path.join(args.out, "last.pt"), 1, 0
    if args.resume and os.path.exists(last_path):
        st = torch.load(last_path, map_location=device)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])  # noqa: E702
        best, bad, start = st["best"], st["bad"], st["epoch"] + 1
        print(f"resumed from epoch {st['epoch']}")
    print(f"train={len(tr)} val={len(va)} test={len(te)} device={device}")
    for epoch in range(start, args.epochs + 1):
        model.train()
        t0, total = time.time(), 0.0
        for x, hm, _, _ in dl:
            x, hm = x.to(device), hm.to(device)
            opt.zero_grad(set_to_none=True)
            loss = heatmap_loss(model(x), hm)
            loss.backward()
            opt.step()
            total += loss.item() * len(x)
        sched.step()
        val = evaluate(model, ds_va, device)
        print(f"epoch {epoch:02d} loss={total / len(tr):.5f} val_MRE={val['mean_radial_error_mm']:.2f}mm "
              f"val_BPD_MAE={val['bpd_length_mae_mm']:.2f}mm ({time.time() - t0:.0f}s)", flush=True)
        if val["mean_radial_error_mm"] < best:
            best, bad = val["mean_radial_error_mm"], 0
            torch.save({"model": model.state_dict(), "size": args.size, "base": args.base, "landmarks": LANDMARKS}, best_path)
        else:
            bad += 1
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": epoch, "best": best, "bad": bad}, last_path)
        if bad >= args.patience:
            print("early stopping")
            break
    model.load_state_dict(torch.load(best_path, map_location=device)["model"])
    test = evaluate(model, ds_te, device)
    save_json({"best_val_mre_mm": best, "test": test}, os.path.join(args.out, "metrics.json"))
    print(f"TEST MRE={test['mean_radial_error_mm']:.2f}mm SDR@2mm={test['success_rate_2mm']:.1%} "
          f"SDR@4mm={test['success_rate_4mm']:.1%} BPD_MAE={test['bpd_length_mae_mm']:.2f}mm OFD_MAE={test['ofd_length_mae_mm']:.2f}mm")
    return {"test": test}


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--out", default="runs/landmarks")
    p.add_argument("--init-seg", default="", help="segmentation checkpoint to warm-start from")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--base", type=int, default=32)
    p.add_argument("--sigma", type=float, default=4.0)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
