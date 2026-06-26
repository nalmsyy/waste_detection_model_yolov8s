# Panduan Download Dataset Roboflow

Dataset diarahkan dari Roboflow Universe:

```text
https://universe.roboflow.com/internship-rpdlx/deteksi-sampah-r4w18
```

## Cara 1 — Otomatis lewat script

1. Install dependency:

```bat
pip install -r requirements.txt
```

2. Set API key:

CMD:

```bat
set ROBOFLOW_API_KEY=ISI_API_KEY_KAMU
```

PowerShell:

```powershell
$env:ROBOFLOW_API_KEY="ISI_API_KEY_KAMU"
```

3. Download dataset:

```bat
python src/download_roboflow_dataset.py --overwrite
```

Default:

```text
workspace = internship-rpdlx
project   = deteksi-sampah-r4w18
version   = 5
format    = yolov8
output    = datasets/rf_deteksi_sampah_raw
```

## Cara 2 — Manual dari website Roboflow

1. Buka link dataset.
2. Klik Fork Dataset jika belum masuk workspace kamu.
3. Masuk ke project hasil fork.
4. Buka tab Versions / Download Dataset.
5. Pilih format YOLOv8.
6. Download ZIP.
7. Extract ke:

```text
datasets/rf_deteksi_sampah_raw/
```

Pastikan di folder tersebut ada file:

```text
data.yaml
train/images
train/labels
valid/images
valid/labels
test/images
test/labels
```

Setelah itu lanjut:

```bat
python src/inspect_classes.py --data datasets/rf_deteksi_sampah_raw/data.yaml
python src/merge_classes_to_5.py --overwrite
python src/check_dataset.py --data datasets/sampah_5kelas/data.yaml --show-label-stats
```
