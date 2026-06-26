@echo off
echo ================================================
echo SETUP ENVIRONMENT DETEKSI SAMPAH YOLOV8
 echo ================================================
python -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
echo.
echo Setup selesai. Aktifkan venv dengan:
echo .venv\Scripts\activate
pause
