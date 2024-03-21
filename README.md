# mediapipe-vt
Use MediaPipe facetracking for VTubing.

This is a very early version. I will work more on it.
- [x] Face tracking
- [ ] Body tracking
- [ ] Finger tracking

Program uses the VTube Studio format and pretends to be a 3rd party app like VTube Studio. In VSeeFace you can turn on 3rd party app and select VTube Studio. Once you start the tracker the character will start moving. This only supports ARKit.

Send a drink if you liked it!
https://www.buymeacoffee.com/nuemedia

## How to run
1. Install dependencies, prefferably in a venv. "pip install mediapipe opencv-python numpy"
2. Download facetracker task. https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
3. Adjust camera device and ip in tracker.py if needed.
