# Deteksi Sampah YOLOv8 + OpenCV — 5 Kelas

Repo ini dibuat untuk proyek deteksi sampah menggunakan **YOLOv8 Object Detection** dan integrasi realtime dengan **kamera OpenCV**.

Dataset yang dipakai diarahkan dari Roboflow Universe:

```text
https://universe.roboflow.com/internship-rpdlx/deteksi-sampah-r4w18
```

> Catatan: dataset tidak dimasukkan langsung ke repo karena download Roboflow membutuhkan akun/API key. Repo ini sudah menyediakan script untuk download otomatis.

---

## 1. Kelas Final

Agar tidak membingungkan, kelas dibuat menjadi 5 saja:

```text
0: plastik
1: kertas
2: logam
3: organik
4: lainnya
```

Penjelasan kelas:

| Kelas | Contoh objek |
|---|---|
| plastik | botol plastik, kantong plastik, bungkus plastik |
| kertas | kertas, koran, kardus, karton |
| logam | kaleng, aluminium foil, tutup logam |
| organik | daun, sisa makanan, kulit buah, sampah alami |
| lainnya | kaca, gabus, kain, kemasan campuran, objek lain |

Mapping class Roboflow ke 5 kelas ini ada di:

```text
configs/class_mapping.yaml
```

---

## 2. Struktur Repo

```text
deteksi_sampah_yolov8_5kelas_repo/
├── src/
│   ├── download_roboflow_dataset.py
│   ├── inspect_classes.py
│   ├── merge_classes_to_5.py
│   ├── check_dataset.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict_image.py
│   ├── webcam_detection.py
│   ├── export_model.py
│   └── common.py
│
├── configs/
│   ├── classes_5.yaml
│   └── class_mapping.yaml
│
├── datasets/
│   ├── rf_deteksi_sampah_raw/
│   └── sampah_5kelas/
│
├── scripts/
│   ├── setup_windows.bat
│   ├── run_download_windows.bat
│   ├── run_merge_windows.bat
│   ├── run_train_windows.bat
│   └── run_camera_windows.bat
│
├── notebooks/
│   └── train_colab.ipynb
│
├── requirements.txt
├── DATASET_DOWNLOAD_INSTRUCTIONS.md
└── README.md
```

---

# LANGKAH SETUP DI WINDOWS

## Langkah 1 — Extract ZIP

Extract repo ZIP ke folder kerja kamu, misalnya:

```text
D:\Tugas Alif\Semester 4\ML\deteksi_sampah_yolov8_5kelas_repo
```

Lalu buka folder tersebut di VS Code / terminal.

---

## Langkah 2 — Buat virtual environment

Buka terminal di folder repo, lalu jalankan:

```bat
python -m venv .venv
```

Aktifkan virtual environment:

### CMD

```bat
.venv\Scripts\activate
```

### PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

Jika PowerShell menolak eksekusi script, jalankan:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

---

## Langkah 3 — Install library

```bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Cek instalasi YOLO:

```bat
yolo checks
```

---

# LANGKAH DOWNLOAD DATASET ROBOFLOW

## Langkah 4 — Ambil Roboflow API Key

1. Login ke Roboflow.
2. Masuk ke **Settings**.
3. Cari bagian **Roboflow API**.
4. Copy API key kamu.

Jangan upload API key ke GitHub.

---

## Langkah 5 — Set API Key

### Kalau pakai CMD

```bat
set ROBOFLOW_API_KEY=ISI_API_KEY_KAMU
```

### Kalau pakai PowerShell

```powershell
$env:ROBOFLOW_API_KEY="ISI_API_KEY_KAMU"
```

---

## Langkah 6 — Download dataset dari Roboflow

```bat
python src/download_roboflow_dataset.py --overwrite
```

Default script ini memakai:

```text
workspace : internship-rpdlx
project   : deteksi-sampah-r4w18
version   : 5
format    : yolov8
output    : datasets/rf_deteksi_sampah_raw
```

Kalau kamu mau mengganti version dataset:

```bat
python src/download_roboflow_dataset.py --version 4 --overwrite
```

Setelah berhasil, folder dataset mentah akan ada di:

```text
datasets/rf_deteksi_sampah_raw/
```

Strukturnya biasanya:

```text
train/images
train/labels
valid/images
valid/labels
test/images
test/labels
data.yaml
```

---

# LANGKAH MERAPIKAN CLASS KE 5 KELAS

## Langkah 7 — Lihat class asli Roboflow

```bat
python src/inspect_classes.py --data datasets/rf_deteksi_sampah_raw/data.yaml
```

Tujuannya untuk melihat nama class asli dari dataset.

---

## Langkah 8 — Merge/remap class menjadi 5 kelas

```bat
python src/merge_classes_to_5.py --overwrite
```

Script ini akan membuat dataset baru:

```text
datasets/sampah_5kelas/
```

Dengan class final:

```text
plastik
kertas
logam
organik
lainnya
```

---

## Langkah 9 — Cek dataset final

```bat
python src/check_dataset.py --data datasets/sampah_5kelas/data.yaml --show-label-stats
```

Pastikan outputnya menampilkan jumlah gambar dan label pada split train, valid, dan test.

---

# LANGKAH TRAINING YOLOV8

## Langkah 10 — Training model YOLOv8n

Untuk laptop biasa:

```bat
python src/train.py --data datasets/sampah_5kelas/data.yaml --model yolov8n.pt --epochs 100 --imgsz 640 --batch 8
```

Kalau laptop kamu kuat atau pakai GPU, boleh coba YOLOv8s:

```bat
python src/train.py --data datasets/sampah_5kelas/data.yaml --model yolov8s.pt --epochs 100 --imgsz 640 --batch 16
```

Saran awal:

```text
Gunakan yolov8n.pt dulu karena lebih ringan dan cocok untuk realtime kamera.
```

---

## Langkah 11 — Lokasi model hasil training

Setelah training selesai, model terbaik ada di:

```text
runs/detect/deteksi_sampah_5kelas/weights/best.pt
```

File penting:

```text
best.pt  -> model terbaik
last.pt  -> model epoch terakhir
```

Gunakan `best.pt` untuk kamera OpenCV.

---

# LANGKAH EVALUASI DAN TESTING

## Langkah 12 — Evaluasi model

```bat
python src/evaluate.py --model runs/detect/deteksi_sampah_5kelas/weights/best.pt --data datasets/sampah_5kelas/data.yaml
```

Hasil evaluasi biasanya tersimpan di folder `runs/detect/val`.

Perhatikan metrik:

```text
precision
recall
mAP50
mAP50-95
confusion matrix
```

---

## Langkah 13 — Prediksi gambar/folder/video

Contoh prediksi satu gambar:

```bat
python src/predict_image.py --model runs/detect/deteksi_sampah_5kelas/weights/best.pt --source path\ke\gambar.jpg
```

Contoh prediksi folder:

```bat
python src/predict_image.py --model runs/detect/deteksi_sampah_5kelas/weights/best.pt --source datasets/sampah_5kelas/test/images
```

Output tersimpan di:

```text
outputs/predict/hasil_prediksi
```

---

# LANGKAH INTEGRASI KE KAMERA OPENCV

## Langkah 14 — Jalankan kamera

```bat
python src/webcam_detection.py --model runs/detect/deteksi_sampah_5kelas/weights/best.pt --camera 0 --conf 0.25 --show-fps
```

Kalau kamera tidak muncul, coba:

```bat
python src/webcam_detection.py --model runs/detect/deteksi_sampah_5kelas/weights/best.pt --camera 1 --conf 0.25 --show-fps
```

Tekan tombol:

```text
q
```

untuk keluar dari kamera.

---

# OPSI CEPAT PAKAI FILE .BAT

Kalau tidak mau mengetik panjang, kamu bisa pakai file di folder `scripts`.

### Setup

```bat
scripts\setup_windows.bat
```

### Download dataset

Set API key dulu, lalu:

```bat
scripts\run_download_windows.bat
```

### Merge class

```bat
scripts\run_merge_windows.bat
```

### Training

```bat
scripts\run_train_windows.bat
```

### Kamera

```bat
scripts\run_camera_windows.bat
```

---

# MASALAH UMUM

## 1. `data.yaml tidak ditemukan`

Berarti dataset belum berhasil didownload atau foldernya tidak sesuai.

Cek:

```text
datasets/rf_deteksi_sampah_raw/data.yaml
```

Kalau tidak ada, download ulang dataset.

---

## 2. `ROBOFLOW_API_KEY belum ada`

Set API key dulu.

CMD:

```bat
set ROBOFLOW_API_KEY=ISI_API_KEY_KAMU
```

PowerShell:

```powershell
$env:ROBOFLOW_API_KEY="ISI_API_KEY_KAMU"
```

---

## 3. Kamera tidak terbuka

Coba ganti index kamera:

```bat
--camera 1
```

atau tutup aplikasi lain yang sedang memakai kamera.

---

## 4. Training lambat

Gunakan model ringan:

```text
yolov8n.pt
```

Kurangi batch:

```bat
--batch 4
```

Kurangi epoch untuk percobaan:

```bat
--epochs 30
```

---

# ALUR FINAL PROYEK

```text
Download dataset Roboflow
        ↓
Cek class asli
        ↓
Merge ke 5 kelas:
plastik, kertas, logam, organik, lainnya
        ↓
Training YOLOv8
        ↓
Evaluasi model
        ↓
Ambil best.pt
        ↓
Integrasi kamera OpenCV
        ↓
Deteksi sampah realtime
```
