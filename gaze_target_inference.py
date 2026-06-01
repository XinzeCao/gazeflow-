#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gaze target inference algorithms (M1 ~ M4).

Implements four ablation methods described in Section 3.4:
  M1: pure centroid distance matching
  M2: edge distance matching with exponential decay
  M3: edge distance + fixed-weight semantic fusion (lambda = 1.0)
  M4: edge distance + entropy-driven adaptive semantic fusion

Input format (candidates list):
  [
    {
      "target_id": str,           # unique identifier
      "class_name": str,          # Cityscapes class
      "mask": np.ndarray (H, W),  # binary mask (uint8, 0 or 255)
      "centroid": (cx, cy),       # geometric centroid
    },
    ...
  ]

Output format:
  {
    "predicted_id": str,
    "predicted_class": str,
    "scores": {target_id: final_score},
    "geometric_scores": {target_id: geo_score},
    "method": "M1" | "M2" | "M3" | "M4",
    "lambda_used": float,         # only for M3 / M4
    "h_norm": float,              # only for M4
    "candidates_considered": int, # number of effective candidates
  }
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import cv2


# ---------------------------------------------------------------------
# Cityscapes 19-class semantic weights (Section 3.4.2 Table 3.1)
# ---------------------------------------------------------------------
SEMANTIC_WEIGHTS: Dict[str, float] = {
    # Critical traffic targets (W > 1.0)
    "person":        1.6,
    "rider":         1.6,
    "car":           1.4,
    "truck":         1.3,
    "bus":           1.3,
    "train":         1.3,
    "motorcycle":    1.5,
    "bicycle":       1.5,
    "traffic light": 1.5,
    "traffic sign":  1.4,

    # Driving environment context (W ~ 1.0)
    "road":          1.0,
    "sidewalk":      0.9,

    # Background structures (W < 1.0)
    "building":      0.8,
    "wall":          0.8,
    "fence":         0.8,
    "pole":          0.8,
    "vegetation":    0.7,
    "terrain":       0.7,
    "sky":           0.6,
}

DEFAULT_SEMANTIC_WEIGHT = 1.0


# ---------------------------------------------------------------------
# Geometric utilities
# ---------------------------------------------------------------------
def _compute_centroid(mask: np.ndarray) -> Tuple[float, float]:
    """Compute the geometric centroid of a binary mask."""
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if mask.max() <= 1:
        binary = (mask > 0).astype(np.uint8)
    else:
        binary = (mask > 127).astype(np.uint8)

    moments = cv2.moments(binary)
    if moments["m00"] < 1e-6:
        ys, xs = np.where(binary > 0)
        if len(xs) == 0:
            h, w = binary.shape[:2]
            return (w / 2.0, h / 2.0)
        return (float(xs.mean()), float(ys.mean()))
    return (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])


def _compute_centroid_distance(gaze_point: Tuple[float, float],
                               centroid: Tuple[float, float]) -> float:
    """Euclidean distance from gaze point to mask centroid."""
    u, v = gaze_point
    cx, cy = centroid
    return float(np.sqrt((u - cx) ** 2 + (v - cy) ** 2))


def _compute_edge_distance(gaze_point: Tuple[float, float],
                           mask: np.ndarray) -> float:
    """Distance from gaze point to mask edge.

    If the gaze point falls inside the mask, returns 0.
    Otherwise returns the Euclidean distance to the closest contour point.
    """
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if mask.max() <= 1:
        binary = (mask > 0).astype(np.uint8) * 255
    else:
        binary = mask.copy()

    h, w = binary.shape[:2]
    u, v = gaze_point
    u_int = int(round(u))
    v_int = int(round(v))

    # Check if gaze point is inside the mask
    if 0 <= v_int < h and 0 <= u_int < w:
        if binary[v_int, u_int] > 0:
            return 0.0

    # Compute distance to closest contour point
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return float("inf")

    min_distance = float("inf")
    for contour in contours:
        dist = cv2.pointPolygonTest(contour, (float(u), float(v)), True)
        abs_dist = abs(dist)
        if abs_dist < min_distance:
            min_distance = abs_dist
    return min_distance


# ---------------------------------------------------------------------
# Method M1: Pure centroid distance matching
# ---------------------------------------------------------------------
def predict_m1(gaze_point: Tuple[float, float],
               candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """M1: select the candidate with the smallest centroid distance.

    The most naive geometric baseline. No semantic information is used.
    """
    if not candidates:
        return {"predicted_id": None, "predicted_class": None, "method": "M1",
                "scores": {}, "geometric_scores": {}, "candidates_considered": 0}

    best_id = None
    best_class = None
    best_distance = float("inf")
    distance_scores: Dict[str, float] = {}

    for cand in candidates:
        centroid = cand.get("centroid")
        if centroid is None:
            centroid = _compute_centroid(cand["mask"])
        distance = _compute_centroid_distance(gaze_point, centroid)
        # Score = inverse distance (smaller distance => higher score)
        distance_scores[cand["target_id"]] = 1.0 / (distance + 1e-6)
        if distance < best_distance:
            best_distance = distance
            best_id = cand["target_id"]
            best_class = cand["class_name"]

    return {
        "predicted_id": best_id,
        "predicted_class": best_class,
        "method": "M1",
        "scores": distance_scores,
        "geometric_scores": distance_scores,
        "candidates_considered": len(candidates),
    }


# ---------------------------------------------------------------------
# Method M2: Edge distance matching (exponential decay)
# ---------------------------------------------------------------------
def predict_m2(gaze_point: Tuple[float, float],
               candidates: List[Dict[str, Any]],
               tau: float = 80.0,
               truncation_factor: float = 5.0) -> Dict[str, Any]:
    """M2: select the candidate with the highest geometric score.

    S_geo(i) = exp(-d_edge(i) / tau)

    Candidates with d_edge > tau * truncation_factor are excluded
    from the effective candidate set.

    Args:
        gaze_point: (u, v) screen coordinates
        candidates: list of candidate target dicts
        tau: decay scale parameter (in pixels)
        truncation_factor: candidates beyond tau * factor are excluded
    """
    if not candidates:
        return {"predicted_id": None, "predicted_class": None, "method": "M2",
                "scores": {}, "geometric_scores": {}, "candidates_considered": 0}

    truncation_distance = tau * truncation_factor

    # Compute edge distances and effective candidates
    edge_distances: Dict[str, float] = {}
    effective_candidates: List[Dict[str, Any]] = []

    for cand in candidates:
        d_edge = _compute_edge_distance(gaze_point, cand["mask"])
        edge_distances[cand["target_id"]] = d_edge
        if d_edge <= truncation_distance:
            effective_candidates.append(cand)

    # If no effective candidates, fall back to closest by edge distance
    if not effective_candidates:
        best_cand = min(candidates,
                        key=lambda c: edge_distances[c["target_id"]])
        return {
            "predicted_id": best_cand["target_id"],
            "predicted_class": best_cand["class_name"],
            "method": "M2",
            "scores": {best_cand["target_id"]: 1.0},
            "geometric_scores": {tid: np.exp(-d / tau)
                                 for tid, d in edge_distances.items()},
            "candidates_considered": 0,
            "fallback_used": True,
        }

    # Geometric scores via exponential decay
    geo_scores: Dict[str, float] = {}
    for cand in effective_candidates:
        d_edge = edge_distances[cand["target_id"]]
        geo_scores[cand["target_id"]] = float(np.exp(-d_edge / tau))

    # Normalize within effective set
    total = sum(geo_scores.values())
    if total > 1e-12:
        geo_scores_norm = {tid: s / total for tid, s in geo_scores.items()}
    else:
        geo_scores_norm = {tid: 1.0 / len(geo_scores) for tid in geo_scores}

    # Pick the candidate with highest score
    best_id = max(geo_scores_norm, key=geo_scores_norm.get)
    best_class = next(c["class_name"] for c in effective_candidates
                      if c["target_id"] == best_id)

    return {
        "predicted_id": best_id,
        "predicted_class": best_class,
        "method": "M2",
        "scores": geo_scores_norm,
        "geometric_scores": geo_scores_norm,
        "edge_distances": edge_distances,
        "candidates_considered": len(effective_candidates),
    }


# ---------------------------------------------------------------------
# Method M3: Edge distance + fixed-weight semantic fusion
# ---------------------------------------------------------------------
def predict_m3(gaze_point: Tuple[float, float],
               candidates: List[Dict[str, Any]],
               tau: float = 80.0,
               truncation_factor: float = 5.0,
               lambda_fixed: float = 1.0,
               semantic_weights: Optional[Dict[str, float]] = None
               ) -> Dict[str, Any]:
    """M3: combine geometric score with fixed-weight semantic prior.

    S_final(i) = S_geo_norm(i) * W_sem(c_i)^lambda_fixed

    Args:
        lambda_fixed: fixed fusion coefficient (default 1.0 = lambda_max)
        semantic_weights: class -> weight mapping (defaults to SEMANTIC_WEIGHTS)
    """
    if semantic_weights is None:
        semantic_weights = SEMANTIC_WEIGHTS

    if not candidates:
        return {"predicted_id": None, "predicted_class": None, "method": "M3",
                "scores": {}, "geometric_scores": {}, "candidates_considered": 0}

    # First obtain geometric component via M2
    m2_result = predict_m2(gaze_point, candidates, tau, truncation_factor)
    geo_scores = m2_result.get("geometric_scores", {})
    edge_distances = m2_result.get("edge_distances", {})

    if m2_result.get("fallback_used", False):
        return {**m2_result, "method": "M3", "lambda_used": lambda_fixed}

    # Apply semantic weights
    final_scores: Dict[str, float] = {}
    class_lookup = {c["target_id"]: c["class_name"] for c in candidates}
    for tid, geo_score in geo_scores.items():
        cls = class_lookup.get(tid, "unknown")
        w_sem = semantic_weights.get(cls, DEFAULT_SEMANTIC_WEIGHT)
        final_scores[tid] = geo_score * (w_sem ** lambda_fixed)

    best_id = max(final_scores, key=final_scores.get)
    best_class = class_lookup[best_id]

    return {
        "predicted_id": best_id,
        "predicted_class": best_class,
        "method": "M3",
        "scores": final_scores,
        "geometric_scores": geo_scores,
        "edge_distances": edge_distances,
        "candidates_considered": m2_result["candidates_considered"],
        "lambda_used": lambda_fixed,
    }


# ---------------------------------------------------------------------
# Method M4: Edge distance + entropy-driven adaptive fusion
# ---------------------------------------------------------------------
def _compute_normalized_entropy(geo_scores_norm: Dict[str, float]) -> float:
    """Compute normalized Shannon entropy of geometric probability distribution.

    H_norm = H(p) / log(N_eff)
    where N_eff is the number of effective candidates.

    Returns a value in [0, 1]:
      0 = sharp distribution (one target dominates)
      1 = flat distribution (all candidates equally likely)
    """
    n_eff = len(geo_scores_norm)
    if n_eff <= 1:
        return 0.0

    probs = np.array(list(geo_scores_norm.values()), dtype=np.float64)
    probs = probs[probs > 1e-12]
    if len(probs) <= 1:
        return 0.0

    entropy = -np.sum(probs * np.log(probs))
    max_entropy = np.log(n_eff)
    if max_entropy < 1e-12:
        return 0.0
    return float(entropy / max_entropy)


def _compute_adaptive_lambda(h_norm: float,
                             lambda_min: float = 0.2,
                             lambda_max: float = 1.0) -> float:
    """Linearly map normalized entropy to fusion coefficient.

    lambda = lambda_min + (lambda_max - lambda_min) * H_norm
    """
    h_norm = max(0.0, min(1.0, h_norm))
    return lambda_min + (lambda_max - lambda_min) * h_norm


def predict_m4(gaze_point: Tuple[float, float],
               candidates: List[Dict[str, Any]],
               tau: float = 80.0,
               truncation_factor: float = 5.0,
               lambda_min: float = 0.2,
               lambda_max: float = 1.0,
               semantic_weights: Optional[Dict[str, float]] = None
               ) -> Dict[str, Any]:
    """M4: complete method with entropy-driven adaptive fusion.

    Pipeline:
      1. Compute geometric scores (S_geo) and effective candidates
      2. Compute normalized entropy H_norm of geometric distribution
      3. Determine lambda via linear mapping from H_norm
      4. Final score: S_final(i) = S_geo_norm(i) * W_sem(c_i)^lambda
    """
    if semantic_weights is None:
        semantic_weights = SEMANTIC_WEIGHTS

    if not candidates:
        return {"predicted_id": None, "predicted_class": None, "method": "M4",
                "scores": {}, "geometric_scores": {}, "candidates_considered": 0}

    m2_result = predict_m2(gaze_point, candidates, tau, truncation_factor)
    geo_scores = m2_result.get("geometric_scores", {})
    edge_distances = m2_result.get("edge_distances", {})

    if m2_result.get("fallback_used", False):
        return {**m2_result, "method": "M4",
                "lambda_used": lambda_min, "h_norm": 0.0}

    # Compute normalized entropy
    h_norm = _compute_normalized_entropy(geo_scores)

    # Compute adaptive lambda
    lambda_adaptive = _compute_adaptive_lambda(h_norm, lambda_min, lambda_max)

    # Apply semantic weights with adaptive lambda
    final_scores: Dict[str, float] = {}
    class_lookup = {c["target_id"]: c["class_name"] for c in candidates}
    for tid, geo_score in geo_scores.items():
        cls = class_lookup.get(tid, "unknown")
        w_sem = semantic_weights.get(cls, DEFAULT_SEMANTIC_WEIGHT)
        final_scores[tid] = geo_score * (w_sem ** lambda_adaptive)

    best_id = max(final_scores, key=final_scores.get)
    best_class = class_lookup[best_id]

    return {
        "predicted_id": best_id,
        "predicted_class": best_class,
        "method": "M4",
        "scores": final_scores,
        "geometric_scores": geo_scores,
        "edge_distances": edge_distances,
        "candidates_considered": m2_result["candidates_considered"],
        "lambda_used": lambda_adaptive,
        "h_norm": h_norm,
    }


# ---------------------------------------------------------------------
# Convenience: run all four methods at once
# ---------------------------------------------------------------------
def predict_all(gaze_point: Tuple[float, float],
                candidates: List[Dict[str, Any]],
                tau: float = 80.0,
                truncation_factor: float = 5.0,
                lambda_min: float = 0.2,
                lambda_max: float = 1.0,
                semantic_weights: Optional[Dict[str, float]] = None
                ) -> Dict[str, Dict[str, Any]]:
    """Run M1 ~ M4 on the same input and return all predictions.

    Useful for batch evaluation.
    """
    return {
        "M1": predict_m1(gaze_point, candidates),
        "M2": predict_m2(gaze_point, candidates, tau, truncation_factor),
        "M3": predict_m3(gaze_point, candidates, tau, truncation_factor,
                         lambda_fixed=lambda_max,
                         semantic_weights=semantic_weights),
        "M4": predict_m4(gaze_point, candidates, tau, truncation_factor,
                         lambda_min=lambda_min, lambda_max=lambda_max,
                         semantic_weights=semantic_weights),
    }


# ---------------------------------------------------------------------
# Self-test (for development)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Generate synthetic test case: 3 candidate targets on a 1000x1000 screen
    H, W = 1000, 1000

    # Target A: a small "person" mask near (300, 500)
    mask_a = np.zeros((H, W), dtype=np.uint8)
    cv2.rectangle(mask_a, (280, 470), (320, 530), 255, -1)

    # Target B: a large "building" mask covering left half
    mask_b = np.zeros((H, W), dtype=np.uint8)
    cv2.rectangle(mask_b, (0, 0), (450, H), 255, -1)
    mask_b[470:530, 280:320] = 0  # exclude person area

    # Target C: a "car" mask near (700, 500)
    mask_c = np.zeros((H, W), dtype=np.uint8)
    cv2.rectangle(mask_c, (650, 450), (800, 550), 255, -1)

    candidates = [
        {"target_id": "A", "class_name": "person",   "mask": mask_a,
         "centroid": _compute_centroid(mask_a)},
        {"target_id": "B", "class_name": "building", "mask": mask_b,
         "centroid": _compute_centroid(mask_b)},
        {"target_id": "C", "class_name": "car",      "mask": mask_c,
         "centroid": _compute_centroid(mask_c)},
    ]

    # Test 1: gaze on the person
    print("=" * 60)
    print("Test 1: Gaze point at (300, 500) — person")
    print("=" * 60)
    results = predict_all((300, 500), candidates, tau=50.0)
    for method, r in results.items():
        print(f"  {method}: predicted={r['predicted_id']} "
              f"({r.get('predicted_class')})", end="")
        if "lambda_used" in r:
            print(f"  lambda={r['lambda_used']:.3f}", end="")
        if "h_norm" in r:
            print(f"  H_norm={r['h_norm']:.3f}", end="")
        print()

    # Test 2: gaze near the boundary between person and building
    print("=" * 60)
    print("Test 2: Gaze point at (340, 500) — ambiguous boundary")
    print("=" * 60)
    results = predict_all((340, 500), candidates, tau=50.0)
    for method, r in results.items():
        print(f"  {method}: predicted={r['predicted_id']} "
              f"({r.get('predicted_class')})", end="")
        if "lambda_used" in r:
            print(f"  lambda={r['lambda_used']:.3f}", end="")
        if "h_norm" in r:
            print(f"  H_norm={r['h_norm']:.3f}", end="")
        print()

    # Test 3: gaze far from all targets
    print("=" * 60)
    print("Test 3: Gaze point at (900, 100) — far from all targets")
    print("=" * 60)
    results = predict_all((900, 100), candidates, tau=50.0)
    for method, r in results.items():
        print(f"  {method}: predicted={r['predicted_id']} "
              f"({r.get('predicted_class')})", end="")
        if "lambda_used" in r:
            print(f"  lambda={r['lambda_used']:.3f}", end="")
        if "h_norm" in r:
            print(f"  H_norm={r['h_norm']:.3f}", end="")
        if r.get("fallback_used"):
            print("  [fallback]", end="")
        print()