"""
calibration — Camera & Gaze Calibration Module
===============================================
Camera intrinsics (checkerboard) + gaze-to-screen mapping (Ridge regression).

Modules:
    camera.py       — Camera intrinsic calibration (OpenCV checkerboard)
    gaze_calib.py   — Calibration GUI: grid → collect faces → fit Ridge → calib.json
    predictor.py    — OfflineGazePredictor: features + calib.json → screen (u, v)

Usage:
    from calibration import OfflineGazePredictor

    predictor = OfflineGazePredictor(calib_path="calib.json")
    result = predictor.predict_gaze_point(face_image)
    # result['gaze_point'] — (u, v) screen coordinates
"""
from .predictor import OfflineGazePredictor
