@echo off

echo Create virtual environment...
python -m venv mediapipe-vt-master\.
call mediapipe-vt-master\Scripts\activate

echo Install dependencies...
pip install numpy opencv-python mediapipe

echo Download facetracker task...
curl -Lo mediapipe-vt-master\face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task

echo Done!
