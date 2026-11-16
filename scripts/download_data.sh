#!/usr/bin/env bash
# Download the two public datasets from Zenodo into data/.
# If a URL 404s, open the record page and copy the current file link:
#   FETAL_PLANES_DB: https://zenodo.org/records/3904280
#   HC18:            https://zenodo.org/records/1327317
set -euo pipefail
mkdir -p data && cd data

if [ ! -d FETAL_PLANES_ZENODO ]; then
  wget -q --show-progress -O planes.zip "https://zenodo.org/api/records/3904280/files/FETAL_PLANES_ZENODO.zip/content"
  # the zip has no top-level folder, so extract into one
  mkdir -p FETAL_PLANES_ZENODO && unzip -q planes.zip -d FETAL_PLANES_ZENODO && rm planes.zip
fi

if [ ! -d HC18/training_set ]; then
  mkdir -p HC18 && cd HC18
  wget -q --show-progress -O training_set.zip "https://zenodo.org/api/records/1327317/files/training_set.zip/content"
  wget -q --show-progress -O training_set_pixel_size_and_HC.csv \
    "https://zenodo.org/api/records/1327317/files/training_set_pixel_size_and_HC.csv/content"
  unzip -q training_set.zip && rm training_set.zip
  cd ..
fi

# sanity checks: fail loudly if the layout is not what the loaders expect
test -f FETAL_PLANES_ZENODO/FETAL_PLANES_DB_data.csv || { echo "missing FETAL_PLANES_DB_data.csv"; exit 1; }
test -d FETAL_PLANES_ZENODO/Images || { echo "missing FETAL_PLANES_ZENODO/Images"; exit 1; }
test -f HC18/training_set_pixel_size_and_HC.csv || { echo "missing HC18 CSV"; exit 1; }
echo "planes images: $(ls FETAL_PLANES_ZENODO/Images | wc -l)   HC18 files: $(ls HC18/training_set | wc -l)"
