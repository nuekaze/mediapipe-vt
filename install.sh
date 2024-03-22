#!/bin/sh
echo "Creating environment..."
python -m venv .
. bin/activate
echo "Installing dependencies..."
pip install numpy opencv-python mediapipe
echo "Done..."
