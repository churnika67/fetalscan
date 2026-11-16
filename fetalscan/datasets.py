"""PyTorch datasets for FETAL_PLANES_DB (classification) and HC18 (segmentation).

FETAL_PLANES_DB (Burgos-Artizzu et al., 2020, Zenodo record 3904280):
    FETAL_PLANES_DB_data.csv (';'-separated) with columns
    Image_name;Patient_num;Plane;Brain_plane;Operator;US_Machine;Train
    and images in Images/<Image_name>.png
HC18 (van den Heuvel et al., 2018, Zenodo record 1327317):
    training_set/NNN_HC.png, training_set/NNN_HC_Annotation.png,
    training_set_pixel_size_and_HC.csv with columns filename, pixel size(mm), head circumference (mm)
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .biometry import outline_to_filled_mask
from .preprocess import apply_clahe, letterbox, letterbox_mask, load_gray, to_tensor_input

PLANE_CLASSES = [
    "Fetal abdomen",
    "Fetal brain",
    "Fetal femur",
    "Fetal thorax",
    "Maternal cervix",
    "Other",
]


# --------------------------------------------------------------------------- splits
def read_planes_csv(root: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(root, "FETAL_PLANES_DB_data.csv"), sep=";")
    df.columns = [c.strip() for c in df.columns]
    df["Plane"] = df["Plane"].str.strip()
    unknown = set(df["Plane"]) - set(PLANE_CLASSES)
    if unknown:
        raise ValueError(f"Unexpected plane labels: {sorted(unknown)}")
    df["label"] = df["Plane"].map(PLANE_CLASSES.index)
    df["path"] = df["Image_name"].map(lambda n: os.path.join(root, "Images", f"{n}.png"))
    return df


def patient_split(df: pd.DataFrame, val_frac: float = 0.15, seed: int = 0):
    """Use the official Train flag for test, then split train into train/val BY PATIENT.

    Splitting by patient (not by image) avoids leakage: frames from the same scan look alike.
    """
    train_all = df[df["Train"] == 1]
    test = df[df["Train"] == 0]
    patients = sorted(train_all["Patient_num"].unique())
    rng = random.Random(seed)
    rng.shuffle(patients)
    n_val = max(1, int(len(patients) * val_frac))
    val_p = set(patients[:n_val])
    val = train_all[train_all["Patient_num"].isin(val_p)]
    train = train_all[~train_all["Patient_num"].isin(val_p)]
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def read_hc18_csv(root: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(root, "training_set_pixel_size_and_HC.csv"))
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"pixel size(mm)": "pixel_mm", "head circumference (mm)": "hc_mm"})
    df["img_path"] = df["filename"].map(lambda f: os.path.join(root, "training_set", f))
    df["ann_path"] = df["filename"].map(
        lambda f: os.path.join(root, "training_set", f.replace(".png", "_Annotation.png"))
    )
    return df


def random_split(df: pd.DataFrame, fracs=(0.7, 0.15, 0.15), seed: int = 0):
    idx = np.random.default_rng(seed).permutation(len(df))
    n1 = int(len(df) * fracs[0])
    n2 = n1 + int(len(df) * fracs[1])
    pick = lambda s: df.iloc[idx[s]].reset_index(drop=True)  # noqa: E731
    return pick(slice(0, n1)), pick(slice(n1, n2)), pick(slice(n2, None))


# --------------------------------------------------------------------------- augmentation
def _augment_pair(img: np.ndarray, mask: np.ndarray | None, rng: np.random.Generator):
    """Light, ultrasound-safe augmentation: small rotation/scale/shift, flip, brightness/contrast."""
    h, w = img.shape
    angle = rng.uniform(-15, 15)
    scale = rng.uniform(0.9, 1.1)
    tx, ty = rng.uniform(-0.05, 0.05) * w, rng.uniform(-0.05, 0.05) * h
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    m[:, 2] += (tx, ty)
    img = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    if mask is not None:
        mask = cv2.warpAffine(mask, m, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    if rng.random() < 0.5:
        img = img[:, ::-1]
        mask = mask[:, ::-1] if mask is not None else None
    alpha, beta = rng.uniform(0.85, 1.15), rng.uniform(-15, 15)
    img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(img), (np.ascontiguousarray(mask) if mask is not None else None)


# --------------------------------------------------------------------------- datasets
class PlanesDataset(Dataset):
    def __init__(self, df: pd.DataFrame, size: int = 224, train: bool = False, clahe: bool = True, seed: int = 0):
        self.df, self.size, self.train, self.clahe = df, size, train, clahe
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = load_gray(row["path"])
        if self.clahe:
            img = apply_clahe(img)
        img, _, _ = letterbox(img, self.size)
        if self.train:
            img, _ = _augment_pair(img, None, self.rng)
        x = torch.from_numpy(to_tensor_input(img))
        return x, int(row["label"])


@dataclass
class SegMeta:
    pixel_mm_resized: float
    hc_mm: float


class HC18Dataset(Dataset):
    """Returns (image 1xSxS, mask 1xSxS, pixel size in mm at size S, reference HC in mm)."""

    def __init__(self, df: pd.DataFrame, size: int = 256, train: bool = False, clahe: bool = True, seed: int = 0):
        self.df, self.size, self.train, self.clahe = df, size, train, clahe
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = load_gray(row["img_path"])
        mask = outline_to_filled_mask(load_gray(row["ann_path"]))
        if self.clahe:
            img = apply_clahe(img)
        img, scale, _ = letterbox(img, self.size)
        mask = letterbox_mask(mask, self.size)
        if self.train:
            img, mask = _augment_pair(img, mask, self.rng)
        x = torch.from_numpy(to_tensor_input(img))
        y = torch.from_numpy(mask.astype(np.float32))[None, ...]
        # augmentation changes geometry, so HC targets are only meaningful for eval (train=False)
        return x, y, float(row["pixel_mm"]) / scale, float(row["hc_mm"])
