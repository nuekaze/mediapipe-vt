@echo off
echo Create virtual environment...
python -m venv .
call Scripts\activate
echo Install dependencies...
pip install numpy opencv-python mediapipe
echo Done!
pause
