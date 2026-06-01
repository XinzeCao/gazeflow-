# -*- coding: utf-8 -*-
"""
Gaze math utilities — merged from UniGaze gazelib.

Pitch/yaw ↔ vector conversions, angular error, cos similarity.
Head pose estimation and data normalization (ETRA 2018 method).

Original license: CC BY-NC-SA 4.0
Cite: "Revisiting Data Normalization for Appearance-Based Gaze Estimation"
      Xucong Zhang, Yusuke Sugano, Andreas Bulling, ETRA 2018
"""

import cv2
import numpy as np
import torch

# ============================================================
#  Pitch/Yaw ↔ 3D Vector  (from gaze_utils.py)
# ============================================================

def pitchyaw_to_vector(pitchyaws):
    """Convert yaw (θ) and pitch (φ) angles to unit gaze vectors."""
    if isinstance(pitchyaws, np.ndarray):
        return pitchyaw_to_vector_numpy(pitchyaws)
    elif isinstance(pitchyaws, torch.Tensor):
        return pitchyaw_to_vector_torch(pitchyaws)
    else:
        raise ValueError("Unsupported input type. Only numpy arrays and torch tensors are supported.")


def pitchyaw_to_vector_numpy(pitchyaws):
    n = pitchyaws.shape[0]
    sin = np.sin(pitchyaws)
    cos = np.cos(pitchyaws)
    out = np.empty((n, 3))
    out[:, 0] = np.multiply(cos[:, 0], sin[:, 1])
    out[:, 1] = sin[:, 0]
    out[:, 2] = np.multiply(cos[:, 0], cos[:, 1])
    return out


def pitchyaw_to_vector_torch(pitchyaws):
    n = pitchyaws.size()[0]
    sin = torch.sin(pitchyaws)
    cos = torch.cos(pitchyaws)
    out = torch.empty((n, 3), device=pitchyaws.device)
    out[:, 0] = torch.mul(cos[:, 0], sin[:, 1])
    out[:, 1] = sin[:, 0]
    out[:, 2] = torch.mul(cos[:, 0], cos[:, 1])
    return out


def vector_to_pitchyaw(vectors):
    """Convert gaze vectors to pitch (θ) and yaw (φ) angles."""
    if isinstance(vectors, np.ndarray):
        return vector_to_pitchyaw_numpy(vectors)
    elif isinstance(vectors, torch.Tensor):
        return vector_to_pitchyaw_torch(vectors)
    else:
        raise ValueError("Unsupported input type. Only numpy arrays and torch tensors are supported.")


def vector_to_pitchyaw_numpy(vectors):
    n = vectors.shape[0]
    vectors = vectors / np.linalg.norm(vectors, axis=1).reshape(n, 1)
    out = np.empty((n, 2))
    out[:, 0] = np.arcsin(vectors[:, 1])  # theta (pitch)
    out[:, 1] = np.arctan2(vectors[:, 0], vectors[:, 2])  # phi (yaw)
    return out


def vector_to_pitchyaw_torch(vectors):
    n = vectors.size()[0]
    vectors = vectors / torch.norm(vectors, dim=1).reshape(n, 1)
    out = torch.empty((n, 2), device=vectors.device)
    out[:, 0] = torch.asin(vectors[:, 1])
    out[:, 1] = torch.atan2(vectors[:, 0], vectors[:, 2])
    return out


def angular_error(a, b):
    """Angular error via cosine similarity (degrees)."""
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return angular_error_numpy(a, b)
    elif isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return angular_error_torch(a, b)
    else:
        raise ValueError("Input type mismatch. Both inputs should be either numpy arrays or torch tensors.")


def angular_error_numpy(a, b):
    a = pitchyaw_to_vector(a) if a.shape[1] == 2 else a
    b = pitchyaw_to_vector(b) if b.shape[1] == 2 else b
    ab = np.sum(np.multiply(a, b), axis=1)
    a_norm = np.clip(np.linalg.norm(a, axis=1), a_min=1e-7, a_max=None)
    b_norm = np.clip(np.linalg.norm(b, axis=1), a_min=1e-7, a_max=None)
    similarity = np.divide(ab, np.multiply(a_norm, b_norm))
    return np.arccos(similarity) * 180.0 / np.pi


def angular_error_torch(a, b):
    a = pitchyaw_to_vector(a) if a.size()[1] == 2 else a
    b = pitchyaw_to_vector(b) if b.size()[1] == 2 else b
    ab = torch.sum(a * b, dim=1)
    a_norm = torch.clamp(torch.norm(a, dim=1), min=1e-7)
    b_norm = torch.clamp(torch.norm(b, dim=1), min=1e-7)
    similarity = ab / (a_norm * b_norm)
    return torch.acos(similarity) * 180.0 / np.pi


def cos_similarity(a, b):
    """Cosine similarity between gaze vectors."""
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return cos_similarity_numpy(a, b)
    elif isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return cos_similarity_torch(a, b)
    else:
        raise ValueError("Input type mismatch.")


def cos_similarity_numpy(a, b):
    a = pitchyaw_to_vector(a) if a.shape[1] == 2 else a
    b = pitchyaw_to_vector(b) if b.shape[1] == 2 else b
    ab = np.sum(np.multiply(a, b), axis=1)
    a_norm = np.clip(np.linalg.norm(a, axis=1), a_min=1e-7, a_max=None)
    b_norm = np.clip(np.linalg.norm(b, axis=1), a_min=1e-7, a_max=None)
    similarity = np.clip(np.divide(ab, np.multiply(a_norm, b_norm)), 0., 1.)
    return similarity


def cos_similarity_torch(a, b):
    a = pitchyaw_to_vector(a) if a.size()[1] == 2 else a
    b = pitchyaw_to_vector(b) if b.size()[1] == 2 else b
    ab = torch.sum(a * b, dim=1)
    a_norm = torch.clamp(torch.norm(a, dim=1), min=1e-7)
    b_norm = torch.clamp(torch.norm(b, dim=1), min=1e-7)
    similarity = torch.clamp(ab / (a_norm * b_norm), 0., 1.)
    return similarity


# ============================================================
#  Head Pose Estimation & Data Normalization  (from normalize.py)
# ============================================================

def estimateHeadPose(landmarks, face_model, camera, distortion, iterate=True):
    """Estimate head pose from 2D landmarks and 3D face model via PnP."""
    ret, rvec, tvec = cv2.solvePnP(
        face_model, landmarks, camera, distortion, flags=cv2.SOLVEPNP_EPNP
    )
    if iterate:
        ret, rvec, tvec = cv2.solvePnP(
            face_model, landmarks, camera, distortion, rvec, tvec, True
        )
    return rvec, tvec


def normalize(img, landmarks, focal_norm, distance_norm, roi_size,
              center, hr, ht, cam, gc=None):
    """Normalize face image to canonical view (ETRA 2018 method)."""
    center = center.reshape(3, 1)
    hR = cv2.Rodrigues(hr)[0]

    distance = np.linalg.norm(center)
    z_scale = distance_norm / distance
    cam_norm = np.array([
        [focal_norm, 0, roi_size[0] / 2],
        [0, focal_norm, roi_size[1] / 2],
        [0, 0, 1.0],
    ])
    S = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, z_scale],
    ])

    hRx = hR[:, 0]
    forward = (center / distance).reshape(3)
    down = np.cross(forward, hRx)
    down /= np.linalg.norm(down)
    right = np.cross(down, forward)
    right /= np.linalg.norm(right)
    R = np.c_[right, down, forward].T

    W = np.dot(np.dot(cam_norm, S), np.dot(R, np.linalg.inv(cam)))

    img_warped = cv2.warpPerspective(img, W, roi_size)
    hR_norm = np.dot(R, hR)

    gc_normalized = None
    num_point = landmarks.shape[0]
    landmarks_warped = cv2.perspectiveTransform(
        landmarks.reshape(-1, 1, 2).astype('float32'), W
    )
    landmarks_warped = landmarks_warped.reshape(num_point, 2)

    if gc is not None:
        gc_normalized = gc.reshape((3, 1)) - center
        gc_normalized = np.dot(R, gc_normalized)
        gc_normalized = gc_normalized / np.linalg.norm(gc_normalized)

    return [img_warped, R, hR_norm, gc_normalized, landmarks_warped, W]


# ============================================================
#  Face Center / Landmark Utilities  (from label_utils.py)
# ============================================================

def get_face_center_by_nose(hR, ht, face_model_load):
    """Compute 3D face center from head pose and face model landmarks (eye corners + nose)."""
    # Select eye-corner and nose landmarks (indices depending on 50- vs 68-point model)
    if face_model_load.shape[0] == 50:
        lm_6_idx = [20, 23, 26, 29, 15, 19]
    elif face_model_load.shape[0] == 68:
        lm_6_idx = [36, 39, 42, 45, 31, 35]
    else:
        lm_6_idx = [20, 23, 26, 29, 15, 19]  # default to 50-pt

    face_model = face_model_load[lm_6_idx, :]  # (6, 3)
    Fc = np.dot(hR, face_model.T) + ht         # 3D positions of facial landmarks
    # Face center = mean of eye centers and nose center
    two_eye_center = np.mean(Fc[:, 0:4], axis=1).reshape(3, 1)
    nose_center = np.mean(Fc[:, 4:6], axis=1).reshape(3, 1)
    face_center = np.mean(np.concatenate((two_eye_center, nose_center), axis=1), axis=1).reshape(3, 1)
    return face_center, Fc
