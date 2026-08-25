import cv2
import mediapipe as mp
import socket
import json
import math
import sys
import time

def log(msg):
    with open("python_log.txt", "a") as f:
        f.write(f"{time.strftime('%H:%M:%S')} - {msg}\n")
    print(msg)

open("python_log.txt", "w").close() # Clear old log
log("Starting Python Eye Tracker...")

try:
    UDP_IP = "127.0.0.1"
    UDP_PORT = 5005
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    log(f"UDP Socket created on {UDP_IP}:{UDP_PORT}")

    mp_face_mesh = mp.solutions.face_mesh
    # LOWERED confidence to 0.2 so it detects faces even in low light!
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.2, min_tracking_confidence=0.2)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        log("ERROR: Cannot open webcam")
        sys.exit()

    log("Webcam opened successfully. Waiting for frames...")
    
    frames_sent = 0
    frames_empty = 0
    
    while True:
        success, image = cap.read()
        if not success:
            continue
            
        image = cv2.flip(image, 1)
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if results.multi_face_landmarks:
            frames_empty = 0 # Reset empty frames
            landmarks = results.multi_face_landmarks[0].landmark
            
            def distance(p1, p2): return math.hypot(p1.x - p2.x, p1.y - p2.y)
            
            right_ear = distance(landmarks[159], landmarks[145]) / (distance(landmarks[33], landmarks[133]) + 1e-6)
            left_ear = distance(landmarks[386], landmarks[374]) / (distance(landmarks[362], landmarks[263]) + 1e-6)
            
            left_blink = bool(left_ear < 0.22)
            right_blink = bool(right_ear < 0.22)
            
            left_iris_center = landmarks[473]
            left_eye_inner = landmarks[362]
            left_eye_outer = landmarks[263]
            left_eye_top = landmarks[386]
            left_eye_bottom = landmarks[374]
            
            eye_width = abs(left_eye_inner.x - left_eye_outer.x)
            rel_x = (left_iris_center.x - left_eye_outer.x) / (eye_width + 1e-6)
            
            eye_height = abs(left_eye_bottom.y - left_eye_top.y)
            rel_y = (left_iris_center.y - left_eye_top.y) / (eye_height + 1e-6)
            
            data = {"x": rel_x, "y": rel_y, "lb": left_blink, "rb": right_blink}
            
            # Send JSON string with a newline so MATLAB can split it easily
            payload = json.dumps(data) + "\n"
            sock.sendto(payload.encode(), (UDP_IP, UDP_PORT))
            
            frames_sent += 1
            if frames_sent == 1:
                log(f"SUCCESS! First packet sent successfully to MATLAB!")
                
            if frames_sent % 300 == 0:
                log(f"Still running smoothly. Sent {frames_sent} frames so far.")
        else:
            frames_empty += 1
            if frames_empty == 1:
                log("Warning: Camera is active, but I cannot see a face. Please make sure your face is well-lit and in frame!")
            elif frames_empty % 150 == 0:
                log("Still searching for a face...")
                
except Exception as e:
    log(f"FATAL ERROR: {str(e)}")
finally:
    if 'cap' in locals():
        cap.release()
