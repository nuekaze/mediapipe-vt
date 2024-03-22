import argparse
import json
import math
import os
import socket
import threading
import time

import cv2
import mediapipe as mp
import numpy as np

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

parser = argparse.ArgumentParser(
    prog="mediapipe-yt",
    description="Wrapper that uses Mediapipe to generate tracking data for vtuber apps.",
)
parser.add_argument(
    "-c", "--camera", default=0, type=int, help="the device index of the camera to use"
)
parser.add_argument(
    "--ip", default="127.0.0.1", type=str, help="the IP to send tracking data to"
)
parser.add_argument(
    "--target-fps", default=60, type=int, help="set the target speed of sending data"
)
parser.add_argument(
    "--enable-interpolation",
    default=False,
    action="store_true",
    help="enable experimental interpolation function",
)
parser.add_argument(
    "--interpolation-strength",
    default=0.4,
    type=float,
    help="set the power of the interpolation",
)

args = parser.parse_args()

TARGET_IP = args.ip  # If you run VSeeFace on same computer keep this.
CAMERA_DEVICE = args.camera

print("Get camera")
# Try figure out if windows or other os automatically.
if os.name == "nt":
    camera = cv2.VideoCapture(
        CAMERA_DEVICE, cv2.CAP_DSHOW
    )  # Force dshow if Windows because it is faster.
else:
    camera = cv2.VideoCapture(CAMERA_DEVICE)

data = {
    "FaceFound": True,
    "Position": {"x": 0.0, "y": 0.0, "z": 0.0},
    "Rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
    "BlendShapes": [
        {"k": "_neutral", "v": 0.0},
        {"k": "browDownLeft", "v": 0.0},
        {"k": "browDownRight", "v": 0.0},
        {"k": "browInnerUp", "v": 0.0},
        {"k": "browOuterUpLeft", "v": 0.0},
        {"k": "browOuterUpRight", "v": 0.0},
        {"k": "cheekPuff", "v": 0.0},
        {"k": "cheekSquintLeft", "v": 0.0},
        {"k": "cheekSquintRight", "v": 0.0},
        {"k": "eyeBlinkLeft", "v": 0.0},
        {"k": "eyeBlinkRight", "v": 0.0},
        {"k": "eyeLookDownLeft", "v": 0.0},
        {"k": "eyeLookDownRight", "v": 0.0},
        {"k": "eyeLookInLeft", "v": 0.0},
        {"k": "eyeLookInRight", "v": 0.0},
        {"k": "eyeLookOutLeft", "v": 0.0},
        {"k": "eyeLookOutRight", "v": 0.0},
        {"k": "eyeLookUpLeft", "v": 0.0},
        {"k": "eyeLookUpRight", "v": 0.0},
        {"k": "eyeSquintLeft", "v": 0.0},
        {"k": "eyeSquintRight", "v": 0.0},
        {"k": "eyeWideLeft", "v": 0.0},
        {"k": "eyeWideRight", "v": 0.0},
        {"k": "jawForward", "v": 0.0},
        {"k": "jawLeft", "v": 0.0},
        {"k": "jawOpen", "v": 0.0},
        {"k": "jawRight", "v": 0.0},
        {"k": "mouthClose", "v": 0.0},
        {"k": "mouthDimpleLeft", "v": 0.0},
        {"k": "mouthDimpleRight", "v": 0.0},
        {"k": "mouthFrownLeft", "v": 0.0},
        {"k": "mouthFrownRight", "v": 0.0},
        {"k": "mouthFunnel", "v": 0.0},
        {"k": "mouthLeft", "v": 0.0},
        {"k": "mouthLowerDownLeft", "v": 0.0},
        {"k": "mouthLowerDownRight", "v": 0.0},
        {"k": "mouthPressLeft", "v": 0.0},
        {"k": "mouthPressRight", "v": 0.0},
        {"k": "mouthPucker", "v": 0.0},
        {"k": "mouthRight", "v": 0.0},
        {"k": "mouthRollLower", "v": 0.0},
        {"k": "mouthRollUpper", "v": 0.0},
        {"k": "mouthShrugLower", "v": 0.0},
        {"k": "mouthShrugUpper", "v": 0.0},
        {"k": "mouthSmileLeft", "v": 0.0},
        {"k": "mouthSmileRight", "v": 0.0},
        {"k": "mouthStretchLeft", "v": 0.0},
        {"k": "mouthStretchRight", "v": 0.0},
        {"k": "mouthUpperUpLeft", "v": 0.0},
        {"k": "mouthUpperUpRight", "v": 0.0},
        {"k": "noseSneerLeft", "v": 0.0},
        {"k": "noseSneerRight", "v": 0.0},
        {"k": "tongueOut", "v": 0.0},
    ],
}

blenddiff = [0.0] * 53

running = True
inference_fps = float(args.target_fps)


def pred_callback(
    detection_result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int
):
    global data
    global blenddiff
    try:
        for i in range(len(detection_result.face_blendshapes[0])):
            score = detection_result.face_blendshapes[0][i].score
            blenddiff[i] = data["BlendShapes"][i]["v"] - score
            data["BlendShapes"][i]["v"] = score

        mat = np.array(detection_result.facial_transformation_matrixes[0])

        # Position is in forth column.
        data["Position"]["x"] = -mat[0][3]
        data["Position"]["y"] = mat[1][3]
        data["Position"]["z"] = mat[2][3]

        # Rotation matrix are the first 3x3 in matrix. Do some rotation matrix to euler angles magic.
        data["Rotation"]["x"] = (
            np.arctan2(-mat[2, 0], np.sqrt(mat[2, 1] ** 2 + mat[2, 2] ** 2))
            * 180
            / math.pi
        )
        data["Rotation"]["z"] = np.arctan2(mat[1, 0], mat[0, 0]) * 180 / math.pi
        data["Rotation"]["y"] = np.arctan2(mat[2, 1], mat[2, 2]) * 180 / math.pi

    except IndexError as a:
        print("Face not found.")


def data_send_thread():
    # Send data at target fps.
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    fps = 1.0 / args.target_fps

    while running:
        now = time.time()
        sock.sendto(bytes(json.dumps(data), "utf-8"), (TARGET_IP, 50506))
        time.sleep((now + fps) - now)


def interpolation():
    global data
    while running:
        now = time.time()
        fps = 1.0 / inference_fps

        iterations = float(args.target_fps) / inference_fps
        for i in range(len(blenddiff)):
            inter = (
                data["BlendShapes"][i]["v"]
                + (blenddiff[i] / iterations) * args.interpolation_strength
            )
            inter = max(inter, 0.0)
            inter = min(inter, 1.0)
            data["BlendShapes"][i]["v"] = inter

        time.sleep((now + fps) - now)


print("Init mediapipe")
model = BaseOptions(model_asset_path="face_landmarker.task")  # This is the task file.
options = FaceLandmarkerOptions(
    base_options=model,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    num_faces=1,
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=pred_callback,
)

# Start thread for sending data.
if args.enable_interpolation:
    interpolation_thread = threading.Thread(target=interpolation, daemon=True)
    interpolation_thread.start()

data_thread = threading.Thread(target=data_send_thread, daemon=True)
data_thread.start()

# The main loop.
with FaceLandmarker.create_from_options(options) as detector:
    tlast = 0
    while running:
        try:
            start_time = time.time()
            # Start timer for frame.
            t = round(time.time() * 1000)
            if t == tlast:
                t += 1
            tlast = t

            # Get image and mirror.
            _, image = camera.read()
            if image is None:
                print("Failed to get image from camera. Try another camera index.")
                running = False
                break

            image = cv2.flip(image, 1)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

            # Run inference.
            detector.detect_async(mp_image, t)

            inference_fps = 1.0 / (time.time() - start_time)

        except KeyboardInterrupt:
            running = False

        except Exception as e:
            print(e)
            running = False

camera.release()

if args.enable_interpolation:
    interpolation_thread.join()

data_thread.join()
