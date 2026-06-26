@echo off
call .venv\Scripts\activate
if "%ROBOFLOW_API_KEY%"=="" (
  echo ROBOFLOW_API_KEY belum diset.
  echo Contoh CMD: set ROBOFLOW_API_KEY=ISI_API_KEY_KAMU
  echo Contoh PowerShell: $env:ROBOFLOW_API_KEY="ISI_API_KEY_KAMU"
  pause
  exit /b 1
)
python src\download_roboflow_dataset.py --overwrite
pause
