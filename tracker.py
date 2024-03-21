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

data = {"FaceFound": True, "Position": {"x":0.0,"y":0.0,"z":0.0}, "Rotation": {"x":0.0,"y":0.0,"z":0.0}, "BlendShapes": []}

def pred_callback(detection_result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    try:
        data["BlendShapes"] = []
        for shape in detection_result.face_blendshapes[0]:
            t = {}
            t["k"] = shape.category_name
            t["v"] = shape.score
            data["BlendShapes"].append(t)

        mat = np.array(detection_result.facial_transformation_matrixes[0])

        # Position is in forth column.
        data["Position"]["x"] = -mat[0][3]
        data["Position"]["y"] = mat[1][3]
        data["Position"]["z"] = mat[2][3]

        # Rotation matrix are the first 3x3 in matrix. Do some rotation matrix to euler angles magic.
        data["Rotation"]["x"] = np.arctan2(-mat[2, 0], np.sqrt(mat[2, 1]**2 + mat[2, 2]**2)) * 180 / math.pi
        data["Rotation"]["z"] = np.arctan2(mat[1, 0], mat[0, 0]) * 180 / math.pi
        data["Rotation"]["y"] = np.arctan2(mat[2, 1], mat[2, 2]) * 180 / math.pi
        
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
    tlast = 0
    while 1:
        try:
            # Get image
            _, image = camera.read()
            image = cv2.flip(image, 1) # Add a flip because most vtuber programs expect this.
            
            # Run inference
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            t = round(time.time() * 1000)
            if t == tlast:
                t += 1
            detector.detect_async(mp_image, t)
            tlast = t
        
        except KeyboardInterrupt:
            break
        
        except Exception as e:
            print(e)
            exit()

camera.release()
cv2.destroyAllWindows()
