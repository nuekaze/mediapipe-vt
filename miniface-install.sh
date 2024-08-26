#!/bin/sh
echo "Creating environment..."
python -m venv mediapipe-vt-master/.
. mediapipe-vt-master/bin/activate

echo "Installing dependencies..."
pip install numpy opencv-python mediapipe

echo "Download facetracker tasks..."
curl -o mediapipe-vt-master/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task

echo "Done..."
