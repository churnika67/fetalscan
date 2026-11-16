# FetalScan

Deep learning for fetal ultrasound: **standard-plane classification** and **automated head circumference (HC) measurement**, built in PyTorch with OpenCV and PIL, on two public datasets.

> Research and learning project on public, de-identified datasets. Not a medical device and not for clinical use.

| Task | Dataset | Model | Metrics |
|---|---|---|---|
| Plane classification (6 classes: abdomen, brain, femur, thorax, maternal cervix, other) | FETAL_PLANES_DB, about 12,400 images | ResNet-18 (ImageNet transfer, 1-channel input) | accuracy, macro F1, confusion matrix |
| Fetal head segmentation + HC biometry | HC18, 999 labeled training images | U-Net + OpenCV ellipse fit | Dice, HC MAE (mm), % within 3 mm, bootstrap 95% CI |

## Results

_Fill in after running `colab_fetalscan.ipynb` on a GPU. Do not report numbers from the synthetic smoke-test data._

| Metric | Value |
|---|---|
| Plane classification, test accuracy | TBD |
| Plane classification, test macro F1 | TBD |
| Head segmentation, test Dice | TBD |
| Head circumference, test MAE (mm) [95% CI] | TBD |
| ONNX Runtime latency p50, classifier / segmenter (ms) | TBD |

## How it works

```
ultrasound frame ──PIL load──► grayscale ──OpenCV CLAHE──► letterbox resize (keeps aspect ratio)
   │
   ├─► ResNet-18 ──► plane label + probabilities ──► Grad-CAM heatmap (interpretability)
   │
   └─► U-Net ──► head mask ──► largest contour ──► cv2.fitEllipse ──► Ramanujan perimeter × pixel size ──► HC (mm)
```

Design choices worth knowing:
- **Patient-level splits** for FETAL_PLANES_DB: frames from one scan look alike, so validation patients never appear in training (the official `Train` flag defines the test set).
- **Grayscale transfer learning**: the RGB conv1 weights of ImageNet ResNet-18 are averaged into one channel instead of discarded.
- **Class-weighted cross-entropy + label smoothing** for the imbalanced plane classes; early stopping on validation macro F1.
- **BCE + Dice loss** for segmentation (the head is a minority of pixels).
- **Aspect-preserving letterbox**: stretching an image would distort the ellipse and bias HC; pixel size is rescaled with the image.
- **Ellipse fit on the largest region** ignores small false-positive blobs; HC uses Ramanujan's perimeter approximation.
- **Validation with uncertainty**: per-image Dice and HC error with bootstrap 95% confidence intervals.
- **Real-time path**: models export to ONNX and run in ONNX Runtime behind a FastAPI service and a Streamlit demo, with p50/p95 latency measured.

## Run it

**On Colab (recommended, free T4 GPU):** open `colab_fetalscan.ipynb`, set the runtime to GPU, run all cells.

**Locally:**
```bash
pip install -r requirements.txt
bash scripts/download_data.sh                      # FETAL_PLANES_DB + HC18 from Zenodo
python -m fetalscan.train_cls --data data/FETAL_PLANES_ZENODO --out runs/cls
python -m fetalscan.train_seg --data data/HC18 --out runs/seg
python -m fetalscan.tune --data data/FETAL_PLANES_ZENODO --trials 10      # optional Optuna search
python -m fetalscan.explain --ckpt runs/cls/best.pt --images some_image.png
python -m fetalscan.export --cls runs/cls/best.pt --seg runs/seg/best.pt --out exports

# serve
FETALSCAN_CLS=exports/classifier.onnx FETALSCAN_SEG=exports/segmenter.onnx uvicorn app.serve:app
curl -F file=@frame.png "localhost:8000/measure-head?pixel_size_mm=0.15"
streamlit run app/streamlit_app.py -- --cls exports/classifier.onnx --seg exports/segmenter.onnx
```
Add `--mlflow` to either training command to log runs with MLflow.

**Tests (CPU, no downloads):**
```bash
pytest -q
```
The suite covers ellipse math, mask filling, preprocessing, metrics, model shapes, Grad-CAM, and an end-to-end train → ONNX export → FastAPI run on synthetic images generated in the exact dataset layouts (`scripts/make_synthetic.py`).

## Layout
```
fetalscan/   preprocess.py  datasets.py  models.py  biometry.py  metrics.py  gradcam.py
             train_cls.py   train_seg.py tune.py    explain.py   export.py   inference.py
app/         serve.py (FastAPI)   streamlit_app.py
scripts/     download_data.sh   make_synthetic.py
tests/       unit + end-to-end tests
```

## Datasets and citations

Check each dataset's license on its Zenodo page before redistributing anything; this repo does not include any images.

- **FETAL_PLANES_DB**: X. P. Burgos-Artizzu et al., "Evaluation of deep convolutional neural networks for automatic classification of common maternal fetal ultrasound planes," *Scientific Reports* 10, 10200 (2020). Zenodo record 3904280.
- **HC18**: T. L. A. van den Heuvel et al., "Automated measurement of fetal head circumference using 2D ultrasound images," *PLOS ONE* 13(8): e0200412 (2018). Zenodo record 1327317.
- U-Net: Ronneberger et al., 2015. Grad-CAM: Selvaraju et al., 2017.

## License
MIT (code only).
