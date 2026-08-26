import time
import math
import logging
import cv2
import numpy as np
import os
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from .base import GazeSource, GazeSample, EyeGeometry, Point2D
from engine.features.blink import compute_ear
from engine.features.depth import compute_ipd_px

logger = logging.getLogger(__name__)

class WebcamSource(GazeSource):
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        # Resolve model path relative to this module file
        self.model_path = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
        self.cap = None
        self.detector = None
        self.seq = 0
        self.is_running = False
        self.dropped_frames = 0
        self.frames_processed = 0
        self.start_time = 0.0

    def _ensure_model(self):
        expected_checksum = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
        if os.path.exists(self.model_path):
            import hashlib
            with open(self.model_path, 'rb') as f:
                if hashlib.sha256(f.read()).hexdigest() == expected_checksum:
                    return
            logger.info("Existing model checksum mismatch, re-downloading...")
            
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        logger.info(f"Downloading face landmarker model to {self.model_path}")
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(self.model_path, 'wb') as out_file:
            out_file.write(response.read())
        
        import hashlib
        with open(self.model_path, 'rb') as f:
            if hashlib.sha256(f.read()).hexdigest() != expected_checksum:
                raise RuntimeError("Model checksum verification failed.")
        logger.info("Model download and verification complete.")

    def start(self) -> None:
        self._ensure_model()
        logger.info(f"Starting WebcamSource on camera {self.camera_index}")
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {self.camera_index}")

        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
        self.is_running = True
        self.start_time = time.monotonic()
        self.seq = 0

    def stop(self) -> None:
        self.is_running = False
        if self.detector:
            self.detector.close()
        if self.cap:
            self.cap.release()
        logger.info(f"WebcamSource stopped. Processed {self.frames_processed} frames, {self.dropped_frames} dropped.")

    def _get_iris_centroid(self, landmarks, indices) -> Point2D:
        cx = sum(landmarks[i].x for i in indices) / len(indices)
        cy = sum(landmarks[i].y for i in indices) / len(indices)
        return Point2D(x=cx, y=cy)

    def iter_samples(self):
        while self.is_running:
            cap_time = time.monotonic()
            ret, frame = self.cap.read()
            self.seq += 1

            if not ret:
                self.dropped_frames += 1
                logger.warning("Dropped frame from camera")
                continue

            self.frames_processed += 1
            
            # Detect low light
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray)
            condition = "low_light" if mean_brightness < 30 else None
            if condition:
                logger.warning(f"Low light condition detected (brightness {mean_brightness:.1f})")

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            detection_result = self.detector.detect(mp_image)

            sample = GazeSample(
                t=cap_time,
                seq=self.seq,
                ok=False,
                condition=condition
            )

            if detection_result.face_landmarks:
                sample.ok = True
                landmarks = detection_result.face_landmarks[0]
                
                left_iris = self._get_iris_centroid(landmarks, [474, 475, 476, 477])
                left_inner = Point2D(landmarks[362].x, landmarks[362].y)
                left_outer = Point2D(landmarks[263].x, landmarks[263].y)
                left_top = Point2D(landmarks[386].x, landmarks[386].y)
                left_bottom = Point2D(landmarks[374].x, landmarks[374].y)

                right_iris = self._get_iris_centroid(landmarks, [469, 470, 471, 472])
                right_inner = Point2D(landmarks[133].x, landmarks[133].y)
                right_outer = Point2D(landmarks[33].x, landmarks[33].y)
                right_top = Point2D(landmarks[159].x, landmarks[159].y)
                right_bottom = Point2D(landmarks[145].x, landmarks[145].y)

                sample.eyes = {
                    "left": EyeGeometry(iris=left_iris, inner=left_inner, outer=left_outer, top=left_top, bottom=left_bottom),
                    "right": EyeGeometry(iris=right_iris, inner=right_inner, outer=right_outer, top=right_top, bottom=right_bottom)
                }

                h_img, w_img, _ = frame.shape
                
                sample.ear = {
                    "left": compute_ear(left_top, left_bottom, left_inner, left_outer, w_img, h_img),
                    "right": compute_ear(right_top, right_bottom, right_inner, right_outer, w_img, h_img)
                }

                sample.ipd_px = compute_ipd_px(left_iris, right_iris, w_img, h_img)
                sample.conf = 1.0

            yield sample
