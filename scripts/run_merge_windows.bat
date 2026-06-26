@echo off
call .venv\Scripts\activate
python src\inspect_classes.py --data datasets\rf_deteksi_sampah_raw\data.yaml
python src\merge_classes_to_5.py --overwrite
python src\check_dataset.py --data datasets\sampah_5kelas\data.yaml --show-label-stats
pause
