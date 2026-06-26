@echo off
call .venv\Scripts\activate
python src\train.py --data datasets\sampah_5kelas\data.yaml --model yolov8n.pt --epochs 100 --imgsz 640 --batch 8 --name deteksi_sampah_5kelas
pause
