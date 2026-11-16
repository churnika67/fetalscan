"""End-to-end smoke tests on synthetic data in the real dataset layouts."""
import subprocess
import sys

import numpy as np
import pytest

from fetalscan.datasets import (HC18Dataset, PlanesDataset, patient_split, random_split, read_hc18_csv,
                                read_planes_csv)


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    out = tmp_path_factory.mktemp("synth")
    subprocess.check_call([sys.executable, "scripts/make_synthetic.py", "--out", str(out),
                           "--n-per-class", "8", "--n-hc", "12"])
    return out


def test_planes_patient_split_no_leakage(synth):
    df = read_planes_csv(str(synth / "planes"))
    tr, va, te = patient_split(df)
    assert set(tr.Patient_num).isdisjoint(va.Patient_num)
    assert set(tr.Patient_num).isdisjoint(te.Patient_num)
    assert len(tr) + len(va) + len(te) == len(df)
    x, y = PlanesDataset(tr, size=64, train=True)[0]
    assert x.shape == (1, 64, 64) and 0 <= y < 6


def test_hc18_dataset_and_reference_hc(synth):
    df = read_hc18_csv(str(synth / "hc18"))
    tr, va, te = random_split(df)
    assert len(tr) + len(va) + len(te) == len(df)
    ds = HC18Dataset(df, size=128)
    x, y, pix, hc = ds[0]
    assert x.shape == y.shape == (1, 128, 128)
    assert set(np.unique(y.numpy())) <= {0.0, 1.0}
    # HC measured from the ground-truth mask should match the CSV reference closely
    from fetalscan.biometry import head_circumference_mm

    assert head_circumference_mm(y[0].numpy().astype(np.uint8), pix) == pytest.approx(hc, rel=0.05)


def test_train_export_and_serve(synth, tmp_path):
    from fetalscan import train_cls, train_seg
    from fetalscan.export import benchmark, export_classifier, export_segmenter

    a = train_cls.parser().parse_args(["--data", str(synth / "planes"), "--epochs", "1", "--size", "64",
                                       "--batch-size", "8", "--workers", "0", "--no-pretrained",
                                       "--out", str(tmp_path / "cls")])
    assert "test" in train_cls.run(a)
    b = train_seg.parser().parse_args(["--data", str(synth / "hc18"), "--epochs", "1", "--size", "64",
                                       "--base", "8", "--workers", "0", "--out", str(tmp_path / "seg")])
    assert "test" in train_seg.run(b)

    cls_onnx = export_classifier(str(tmp_path / "cls" / "best.pt"), str(tmp_path))
    seg_onnx = export_segmenter(str(tmp_path / "seg" / "best.pt"), str(tmp_path))
    assert benchmark(cls_onnx, runs=3)["p50_ms"] > 0

    import os

    os.environ["FETALSCAN_CLS"], os.environ["FETALSCAN_SEG"] = cls_onnx, seg_onnx
    from fastapi.testclient import TestClient

    import app.serve as serve

    serve._rt = None
    client = TestClient(serve.app)
    assert client.get("/health").json() == {"classifier": True, "segmenter": True}
    img = next((synth / "hc18" / "training_set").glob("*_HC.png")).read_bytes()
    r = client.post("/classify", files={"file": ("x.png", img, "image/png")})
    assert r.status_code == 200 and r.json()["plane"]
    r = client.post("/measure-head?pixel_size_mm=0.2", files={"file": ("x.png", img, "image/png")})
    assert r.status_code == 200 and "head_found" in r.json()
    r = client.post("/classify", files={"file": ("x.png", b"not an image", "image/png")})
    assert r.status_code == 400
