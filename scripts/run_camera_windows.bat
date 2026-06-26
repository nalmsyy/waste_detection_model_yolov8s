@echo off
call .venv\Scripts\activate
python src\webcam_detection.py --model runs\detect\deteksi_sampah_5kelas\weights\best.pt --camera 0 --conf 0.25 --show-fps
pause
