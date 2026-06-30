# Deteksi Sampah Menggunakan YOLOv8 dan YOLO26

Project ini merupakan sistem deteksi objek sampah berbasis computer vision menggunakan model YOLO. Sistem dilatih menggunakan dataset deteksi sampah berformat YOLO dan dapat dijalankan secara real-time menggunakan kamera laptop maupun kamera HP yang terhubung ke PC.

Project ini dibuat untuk membandingkan beberapa eksperimen model deteksi objek, mulai dari YOLOv8s hingga YOLO26m.

---

## Deskripsi Project

Sistem ini digunakan untuk mendeteksi beberapa jenis objek sampah pada gambar atau video kamera. Model yang digunakan berbasis YOLO karena mampu melakukan deteksi objek secara real-time dengan menghasilkan class prediction dan bounding box.

Pada project ini dilakukan beberapa eksperimen:

1. Eksperimen 1 menggunakan dataset gabungan 5 kelas.
2. Eksperimen 2 menggunakan dataset final 4 kelas.
3. Eksperimen 3 menggunakan dataset asli Roboflow dengan model YOLOv8s.
4. Eksperimen 4 menggunakan dataset asli Roboflow dengan model YOLO26m.

---

## Fitur

- Training model deteksi sampah menggunakan YOLO.
- Evaluasi model menggunakan Precision, Recall, mAP50, dan mAP50-95.
- Menampilkan confusion matrix dan kurva evaluasi.
- Deteksi gambar menggunakan model hasil training.
- Deteksi real-time menggunakan OpenCV.
- Mendukung kamera laptop dan kamera HP sebagai webcam.
- Membandingkan beberapa eksperimen model.

---

## Dataset

Dataset yang digunakan berasal dari Roboflow Universe.

Dataset untuk Eksperimen 3 dan Eksperimen 4:

```text
Roboflow Universe
Project: latihan-deteksi-objek/deteksi-sampah-8p8u5
Format : YOLOv8
Version: 3
```

Command download dataset:

```bash
roboflow download -f yolov8 -l ./datasets/rf_exp3_deteksi_sampah_raw latihan-deteksi-objek/deteksi-sampah-8p8u5/3
```

Kelas asli dataset:

| No | Kelas |
|---:|---|
| 0 | kertas |
| 1 | logam |
| 2 | pakaian |
| 3 | plastik |
| 4 | tumbuhan |

---

## Eksperimen Model

### Eksperimen 1 - YOLOv8s Dataset 5 Kelas

Dataset awal terdiri dari 5 kelas:

| Kelas | Jumlah Instance |
|---|---:|
| plastik | 3.140 |
| kertas | 6.490 |
| logam | 2.891 |
| organik | 38 |
| lainnya | 3.979 |

Hasil evaluasi:

| Metric | Nilai |
|---|---:|
| Precision | 0.938 |
| Recall | 0.885 |
| mAP50 | 0.930 |
| mAP50-95 | 0.827 |

Catatan:

Kelas organik memiliki jumlah data yang sangat sedikit, yaitu hanya 38 instance. Hal ini menyebabkan model kurang stabil ketika diuji pada kamera real-time karena sering terjadi false positive pada kelas organik.

---

### Eksperimen 2 - YOLOv8s Dataset 4 Kelas

Dataset final 4 kelas terdiri dari:

| Kelas | Jumlah Instance |
|---|---:|
| plastik | 2.930 |
| kertas | 6.149 |
| logam | 2.730 |
| lainnya | 3.822 |

Hasil evaluasi:

| Metric | Nilai |
|---|---:|
| Precision | 0.985 |
| Recall | 0.964 |
| mAP50 | 0.979 |
| mAP50-95 | 0.894 |

Catatan:

Eksperimen 2 menghasilkan performa terbaik secara keseluruhan berdasarkan nilai mAP50-95. Model ini lebih stabil dibandingkan eksperimen 1 karena kelas organik yang bermasalah dihapus dari dataset final.

---

### Eksperimen 3 - YOLOv8s Dataset Asli Roboflow

Eksperimen 3 menggunakan dataset asli Roboflow tanpa penggabungan atau perubahan kelas.

Model yang digunakan:

```text
YOLOv8s
Pretrained weight: yolov8s.pt
```

Konfigurasi training:

| Konfigurasi | Nilai |
|---|---|
| Model | YOLOv8s |
| Epoch | 120 |
| Image size | 640 |
| Batch size | 16 |
| Device | CUDA GPU |
| Optimizer | AdamW |
| Dataset | rf_exp3_deteksi_sampah_raw |

Hasil evaluasi:

| Kelas | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| all | 524 | 526 | 0.969 | 0.952 | 0.978 | 0.822 |
| kertas | 106 | 108 | 0.935 | 0.933 | 0.942 | 0.869 |
| logam | 106 | 106 | 0.958 | 0.953 | 0.984 | 0.821 |
| pakaian | 104 | 104 | 1.000 | 0.993 | 0.995 | 0.825 |
| plastik | 104 | 104 | 0.950 | 0.904 | 0.974 | 0.760 |
| tumbuhan | 104 | 104 | 1.000 | 0.976 | 0.995 | 0.834 |

Model terbaik:

```text
runs/detect/eksperimen3_dataset_asli_roboflow_yolov8s/weights/best.pt
```

---

### Eksperimen 4 - YOLO26m Dataset Asli Roboflow

Eksperimen 4 menggunakan dataset yang sama dengan Eksperimen 3, tetapi model diganti menjadi YOLO26m.

Model yang digunakan:

```text
YOLO26m
Pretrained weight: yolo26m.pt
```

Konfigurasi training:

| Konfigurasi | Nilai |
|---|---|
| Model | YOLO26m |
| Epoch | 120 |
| Image size | 640 |
| Batch size | 8 |
| Device | CUDA GPU |
| Dataset | rf_exp3_deteksi_sampah_raw |
| Waktu training | 1.857 jam |

Hasil evaluasi:

| Kelas | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| all | 524 | 526 | 0.973 | 0.955 | 0.984 | 0.843 |
| kertas | 106 | 108 | 0.961 | 0.889 | 0.956 | 0.898 |
| logam | 106 | 106 | 0.946 | 0.989 | 0.991 | 0.841 |
| pakaian | 104 | 104 | 0.990 | 0.979 | 0.995 | 0.859 |
| plastik | 104 | 104 | 0.967 | 0.933 | 0.984 | 0.777 |
| tumbuhan | 104 | 104 | 1.000 | 0.986 | 0.995 | 0.838 |

Model terbaik:

```text
runs/detect/eksperimen4_dataset_asli_roboflow_yolo26m-2/weights/best.pt
```

---

## Perbandingan Eksperimen

| Eksperimen | Dataset | Model | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---:|---:|---:|---:|
| Eksperimen 1 | Dataset 5 kelas | YOLOv8s | 0.938 | 0.885 | 0.930 | 0.827 |
| Eksperimen 2 | Dataset 4 kelas | YOLOv8s | 0.985 | 0.964 | 0.979 | 0.894 |
| Eksperimen 3 | Dataset asli Roboflow | YOLOv8s | 0.969 | 0.952 | 0.978 | 0.822 |
| Eksperimen 4 | Dataset asli Roboflow | YOLO26m | 0.973 | 0.955 | 0.984 | 0.843 |

Kesimpulan:

- Eksperimen 2 memiliki nilai mAP50-95 tertinggi, yaitu 0.894.
- Eksperimen 4 lebih unggul dibanding Eksperimen 3 karena menggunakan YOLO26m.
- YOLO26m memberikan peningkatan akurasi dibanding YOLOv8s pada dataset yang sama.
- YOLOv8s lebih ringan dan lebih cepat untuk real-time camera.
- YOLO26m lebih kuat dari sisi akurasi, tetapi membutuhkan waktu inference lebih besar.

---

## Struktur Folder

```text
deteksi_sampah_yolov8_5kelas_repo/
├── configs/
├── datasets/                  # tidak diupload ke GitHub
├── models/
├── notebooks/
├── outputs/                   # tidak diupload ke GitHub
├── runs/                      # tidak diupload ke GitHub
├── scripts/
├── src/
│   ├── check_camera_index.py
│   ├── webcam_detection.py
│   ├── check_dataset.py
│   └── preview_yolo_labels.py
├── .gitignore
├── README.md
└── requirements.txt
```

Folder berikut tidak dimasukkan ke GitHub karena ukurannya besar:

```text
.venv/
datasets/
runs/
outputs/
*.pt
*.zip
```

---

## Instalasi

Clone repository:

```bash
git clone https://github.com/username/nama-repository.git
cd nama-repository
```

Buat virtual environment:

```bash
python -m venv .venv
```

Aktifkan virtual environment di Windows PowerShell:

```bash
.venv\Scripts\activate
```

Install library:

```bash
pip install -r requirements.txt
```

Install atau update Ultralytics:

```bash
pip install -U ultralytics
```

Cek instalasi:

```bash
yolo checks
```

Cek CUDA GPU:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

---

## Download Dataset Roboflow

Login Roboflow CLI:

```bash
roboflow login
```

Download dataset:

```bash
roboflow download -f yolov8 -l ./datasets/rf_exp3_deteksi_sampah_raw latihan-deteksi-objek/deteksi-sampah-8p8u5/3
```

Cek isi dataset:

```bash
type datasets\rf_exp3_deteksi_sampah_raw\data.yaml
```

---

## Training Model

### Training Eksperimen 3 - YOLOv8s

```bash
yolo task=detect mode=train model=yolov8s.pt data=datasets/rf_exp3_deteksi_sampah_raw/data.yaml epochs=120 imgsz=640 batch=16 device=0 workers=2 name=eksperimen3_dataset_asli_roboflow_yolov8s patience=30 optimizer=AdamW lr0=0.001 cos_lr=True close_mosaic=10 cache=False
```

### Training Eksperimen 4 - YOLO26m

```bash
yolo task=detect mode=train model=yolo26m.pt data=datasets/rf_exp3_deteksi_sampah_raw/data.yaml epochs=120 imgsz=640 batch=8 device=0 workers=2 name=eksperimen4_dataset_asli_roboflow_yolo26m patience=30 cos_lr=True close_mosaic=10 cache=False
```

---

## Evaluasi Model

### Evaluasi Eksperimen 3

```bash
yolo task=detect mode=val model=runs/detect/eksperimen3_dataset_asli_roboflow_yolov8s/weights/best.pt data=datasets/rf_exp3_deteksi_sampah_raw/data.yaml device=0 project=outputs/evaluasi name=eksperimen3_dataset_asli_roboflow
```

### Evaluasi Eksperimen 4

```bash
yolo task=detect mode=val model=runs/detect/eksperimen4_dataset_asli_roboflow_yolo26m-2/weights/best.pt data=datasets/rf_exp3_deteksi_sampah_raw/data.yaml device=0 project=outputs/evaluasi name=eksperimen4_dataset_asli_roboflow_yolo26m
```

Output evaluasi akan menghasilkan beberapa file penting:

```text
confusion_matrix.png
confusion_matrix_normalized.png
F1_curve.png
P_curve.png
R_curve.png
PR_curve.png
results.png
val_batch0_labels.jpg
val_batch0_pred.jpg
```

---

## Deteksi Real-Time Menggunakan Kamera

Cek index kamera:

```bash
python src/check_camera_index.py
```

Jika kamera HP terbaca pada index 1, jalankan:

```bash
python src/webcam_detection.py --model runs/detect/eksperimen4_dataset_asli_roboflow_yolo26m-2/weights/best.pt --camera 1 --conf 0.20 --show-fps
```

Jika ingin menggunakan model Eksperimen 3:

```bash
python src/webcam_detection.py --model runs/detect/eksperimen3_dataset_asli_roboflow_yolov8s/weights/best.pt --camera 1 --conf 0.20 --show-fps
```

Jika deteksi sulit muncul:

```bash
python src/webcam_detection.py --model runs/detect/eksperimen4_dataset_asli_roboflow_yolo26m-2/weights/best.pt --camera 1 --conf 0.10 --show-fps
```

Jika terlalu banyak false positive:

```bash
python src/webcam_detection.py --model runs/detect/eksperimen4_dataset_asli_roboflow_yolo26m-2/weights/best.pt --camera 1 --conf 0.35 --show-fps
```

Jika FPS terlalu rendah, gunakan resolusi lebih kecil:

```bash
python src/webcam_detection.py --model runs/detect/eksperimen4_dataset_asli_roboflow_yolo26m-2/weights/best.pt --camera 1 --conf 0.20 --width 640 --height 480 --show-fps
```

---

## Format Bounding Box YOLO

Dataset menggunakan format anotasi YOLO:

```text
class_id x_center y_center width height
```

Contoh:

```text
3 0.512 0.438 0.312 0.420
```

Keterangan:

| Nilai | Arti |
|---|---|
| class_id | ID kelas objek |
| x_center | posisi tengah bounding box pada sumbu X |
| y_center | posisi tengah bounding box pada sumbu Y |
| width | lebar bounding box |
| height | tinggi bounding box |

Semua nilai koordinat menggunakan normalisasi dari 0 sampai 1.

---

## Hasil Utama

Model terbaik berdasarkan mAP50-95:

```text
Eksperimen 2 - YOLOv8s Dataset 4 Kelas
mAP50-95: 0.894
```

Model terbaik pada dataset asli Roboflow:

```text
Eksperimen 4 - YOLO26m
mAP50: 0.984
mAP50-95: 0.843
```

Model yang lebih cepat untuk kamera real-time:

```text
Eksperimen 3 - YOLOv8s
Inference: sekitar 2.9 ms
```

Model yang lebih akurat pada dataset asli Roboflow:

```text
Eksperimen 4 - YOLO26m
Inference: sekitar 6.4 ms
```

---

## Catatan

File dataset, hasil training, hasil evaluasi, dan model weight tidak dimasukkan ke GitHub karena ukurannya besar. Jika ingin menjalankan project dari awal, pengguna perlu mendownload dataset dari Roboflow dan melakukan training ulang, atau menambahkan file model `best.pt` secara manual.

---

## Author

Nur Alif Maulana Syafrudin  
Universitas Dian Nuswantoro  
Program Studi Teknik Informatika