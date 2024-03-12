import cv2
import numpy as np
import time
import math
import socket, json
import mediapipe as mp
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

TARGET_IP = "127.0.0.1" # If you run VSeeFace on same computer keep this.
CAMERA_DEVICE = 0 # zero is usually the default webcam.

print("Get camera")
camera = cv2.VideoCapture(CAMERA_DEVICE, cv2.CAP_DSHOW) # I had to force DSHOW mode or it would take minutes to start up

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) 

face_model = {
    "nose_tip": 1,
    "left_eye": 33,
    "right_eye": 263,
    "left_mouth": 91,
    "right_mouth": 321,
    "center_of_face": 152
}

def calculate_normal_and_roll(A, B, C):
    AB = B - A
    AC = C - A
    
    normal = np.cross(AB, AC)
    normal = normal / np.linalg.norm(normal)
    
    x = (A[0] + B[0]) / 2
    y = (A[1] + B[1]) / 2
    
    dx = x - C[0]
    dy = y - C[1]
    
    roll = math.atan2(dy, dx)
    
    return (
        normal[0] * 90, # These 90 multiplications controls the amount of rotation of the head.
        -normal[1] * 90,
        roll * 90
    )

data = {"FaceFound": True, "Position": {"x":0.0,"y":0.0,"z":0.0}, "Rotation": {"x":0.0,"y":0.0,"z":0.0}, "BlendShapes": []}

def pred_callback(detection_result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    try:
        data["BlendShapes"] = []
        for shape in detection_result.face_blendshapes[0]:
            t = {}
            t["k"] = shape.category_name
            t["v"] = shape.score
            data["BlendShapes"].append(t)
        
        landmarks = detection_result.face_landmarks[0]
        
        data["Position"]["x"] = -landmarks[face_model["center_of_face"]].x * 100
        data["Position"]["y"] = -landmarks[face_model["center_of_face"]].y * 100
        
        dx = (landmarks[face_model["left_eye"]].x - landmarks[face_model["right_eye"]].x)**2
        dy = (landmarks[face_model["left_eye"]].z - landmarks[face_model["right_eye"]].z)**2
        data["Position"]["z"] = -(math.sqrt(dx + dy)) * 100
        
        rot = calculate_normal_and_roll(
            np.array([
                landmarks[face_model["left_eye"]].x,
                landmarks[face_model["left_eye"]].y,
                landmarks[face_model["left_eye"]].z
            ]),
            np.array([
                landmarks[face_model["right_eye"]].x,
                landmarks[face_model["right_eye"]].y,
                landmarks[face_model["right_eye"]].z
            ]),
            np.array([
                landmarks[face_model["center_of_face"]].x,
                landmarks[face_model["center_of_face"]].y,
                landmarks[face_model["center_of_face"]].z
            ]))
            
        data["Rotation"]["x"] = -rot[0]
        data["Rotation"]["y"] = rot[1]
        data["Rotation"]["z"] = -rot[2]
        
        sock.sendto(bytes(json.dumps(data), "utf-8"), (TARGET_IP, 50506))
    
    except IndexError as a:
        print("Face not found.")

print("Init mediapipe")
model = BaseOptions(model_asset_path='face_landmarker.task') # This is the task file.
options = FaceLandmarkerOptions(
    base_options=model,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    num_faces=1,
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=pred_callback
    )

# The main loop.
with FaceLandmarker.create_from_options(options) as detector:
    while 1:
        try:
            # Get image
            _, image = camera.read()
            
            # Run inference
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            detector.detect_async(mp_image, int(time.time()*1000))

        except KeyboardInterrupt:
            break
            
        except Exception as e:
            print(e)
            exit()

camera.release()
cv2.destroyAllWindows()
