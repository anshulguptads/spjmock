import cv2
import numpy as np

def analyze_frame(frame):
    # Simple face detection and "centered" check
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                         'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    is_centered = False
    looking_forward = False

    if len(faces) > 0:
        x, y, w, h = faces[0]
        # Check if the face is roughly centered
        img_center_x = frame.shape[1] // 2
        face_center_x = x + w // 2
        if abs(img_center_x - face_center_x) < frame.shape[1] * 0.20:
            is_centered = True
        # Dummy logic: if width/height ratio ~1 (face straight), "looking forward"
        if 0.7 < w/h < 1.3:
            looking_forward = True

    return {
        "face_detected": len(faces) > 0,
        "centered": is_centered,
        "looking_forward": looking_forward
    }

