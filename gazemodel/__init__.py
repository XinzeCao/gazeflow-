"""
gazemodel — Gaze Estimation Module
===================================
Face image → gaze direction vector.

To swap in a different gaze model, implement a class with the same
``extract(bgr_frame) -> dict`` interface as GazeFeatureExtractor.

Usage:
    from gazemodel import GazeFeatureExtractor

    extractor = GazeFeatureExtractor(
        model_cfg_path="gazemodel/config.yaml",
        ckpt_path="gazemodel/weights/model.pth.tar",
    )
    result = extractor.extract(face_bgr_image)
    # result['x']              — (8,) feature vector
    # result['gaze_cam']       — (3,) gaze direction in camera coords
    # result['yaw_pitch_cam']  — (2,) pitch, yaw angles
"""
from .extractor import GazeFeatureExtractor
