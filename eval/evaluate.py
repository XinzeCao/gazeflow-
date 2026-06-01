#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch evaluation script for gaze target identification (Section 5.4.1).

Workflow:
  1. Load the test dataset directory produced by mask2former_gaze_collection.py
  2. For each sample (face_image, ground-truth target_id):
     a. Predict screen gaze point using OfflineGazePredictor (Section 5.3 module)
     b. Load all candidate targets for that source image
     c. Run M1, M2, M3, M4 inference algorithms
     d. Compare predictions against ground truth
  3. Aggregate per-class accuracies and produce Table 5-X output

Output:
  - results_per_sample.csv         (per-sample raw predictions for all methods)
  - results_per_class.csv          (per-class accuracy table — aligns with Table 5-X)
  - results_summary.json           (high-level summary)

Usage:
  python evaluate_gaze_target.py \
      --dataset-dir gaze_dataset_mask2former_test \
      --calib-path calib.json \
      --output-dir results/eval \
      --tau 80
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Make sure inference module is importable from the same directory
_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
if _CUR_DIR not in sys.path:
    sys.path.insert(0, _CUR_DIR)

from gaze_target_inference import (
    SEMANTIC_WEIGHTS, predict_all, _compute_centroid,
)


# ---------------------------------------------------------------------
# Cityscapes class taxonomy for Table 5-X
# ---------------------------------------------------------------------
KEY_CLASSES = {
    "car", "person", "rider", "truck", "bus", "train",
    "motorcycle", "bicycle", "traffic light", "traffic sign",
}
BACKGROUND_CLASSES = {
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "vegetation", "terrain", "sky",
}


# ---------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------
def load_metadata(dataset_dir: Path) -> List[Dict[str, Any]]:
    """Read metadata.jsonl produced by mask2former_gaze_collection.py."""
    meta_path = dataset_dir / "metadata.jsonl"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.jsonl not found at {meta_path}")

    samples = []
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] failed to parse metadata line: {e}")
    return samples


def index_samples_by_image(samples: List[Dict[str, Any]]
                           ) -> Dict[str, List[Dict[str, Any]]]:
    """Group samples by their source image filename."""
    by_image: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in samples:
        image_name = s.get("source_image_filename") or s.get("original_image")
        if image_name is None:
            continue
        by_image[image_name].append(s)
    return by_image


def load_candidates_for_image(dataset_dir: Path,
                               image_name: str,
                               samples_for_image: List[Dict[str, Any]]
                               ) -> List[Dict[str, Any]]:
    """Build the candidate target list for one source image.

    PRIORITY:
      1. Load from panoptic/<image_stem>/all_targets.json if it exists.
         This contains ALL targets detected by the segmenter, fulfilling the
         Section 3.4 methodology of choosing from the full scene context.
      2. Fallback: Aggregate from instance_masks_screen/ associated with
         the collected samples. (Legacy/fallback mode).
    """
    image_stem = Path(image_name).stem
    all_targets_path = dataset_dir / "panoptic" / image_stem / "all_targets.json"

    if all_targets_path.exists():
        try:
            with all_targets_path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            candidates = []
            for t in data.get("targets", []):
                # screen_mask_path is relative to dataset root
                mask_path = dataset_dir / t["screen_mask_path"]
                if not mask_path.exists():
                    continue

                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is None or mask.max() == 0:
                    continue

                candidates.append({
                    "target_id": str(t["target_id"]),
                    "class_name": t["class_name"],
                    "mask": mask,
                    "centroid": t["centroid"],
                })

            if candidates:
                return candidates
        except Exception as e:
            print(f"[WARN] Failed to load all_targets.json for {image_name}: {e}")

    # Fallback to legacy sample aggregation
    screen_mask_dir = dataset_dir / "instance_masks_screen"

    candidates: List[Dict[str, Any]] = []
    seen_ids = set()

    for s in samples_for_image:
        target_dict = s.get("target", {})
        target_class = (target_dict.get("class_name")
                        or s.get("target_info", {}).get("class_name")
                        or s.get("class_name"))
        # target_id may sit under different keys depending on collector version
        target_id = (target_dict.get("target_id")
                     or s.get("target_info", {}).get("target_id")
                     or s.get("target_id"))

        if target_id in seen_ids:
            continue
        seen_ids.add(target_id)

        screen_mask_name = s.get("instance_mask_screen")
        if screen_mask_name is None:
            continue
        screen_mask_path = screen_mask_dir / screen_mask_name
        if not screen_mask_path.exists():
            print(f"[WARN] missing screen mask: {screen_mask_path}")
            continue

        mask = cv2.imread(str(screen_mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.max() == 0:
            continue

        centroid = _compute_centroid(mask)

        candidates.append({
            "target_id": str(target_id),
            "class_name": target_class or "unknown",
            "mask": mask,
            "centroid": centroid,
            "_sample_id": s.get("sample_id"),
        })

    return candidates


# ---------------------------------------------------------------------
# Gaze point prediction
# ---------------------------------------------------------------------
def init_gaze_predictor(calib_path: str,
                        camera_calib_path: str = "camera_calib.json",
                        model_cfg_path: Optional[str] = None,
                        ckpt_path: Optional[str] = None):
    """Initialize the OfflineGazePredictor (Section 5.3 module)."""
    from calibration.predictor import OfflineGazePredictor

    return OfflineGazePredictor(
        calib_path=calib_path,
        model_cfg_path=model_cfg_path,
        ckpt_path=ckpt_path,
        camera_calib_path=camera_calib_path,
    )


def predict_gaze_for_sample(predictor,
                            face_image_path: Path
                            ) -> Optional[Tuple[float, float]]:
    """Run the gaze predictor on one face image and return (u, v)."""
    if not face_image_path.exists():
        return None
    face_image = cv2.imread(str(face_image_path))
    if face_image is None:
        return None
    result = predictor.predict_gaze_point(face_image)
    if not result.get("success"):
        return None
    return tuple(result["gaze_point"])


# ---------------------------------------------------------------------
# Per-class accuracy aggregation
# ---------------------------------------------------------------------
class AccuracyAggregator:
    """Collect per-class hit counts for each method."""

    def __init__(self):
        # method -> class -> [hits, total]
        self.stats: Dict[str, Dict[str, List[int]]] = defaultdict(
            lambda: defaultdict(lambda: [0, 0]))

    def record(self, method: str, true_class: str, predicted_class: str):
        self.stats[method][true_class][1] += 1  # total
        if predicted_class == true_class:
            self.stats[method][true_class][0] += 1  # hits

    def per_class_table(self, methods: List[str]) -> Dict[str, Any]:
        # Collect all classes that appear at least once
        all_classes = set()
        for m in methods:
            all_classes.update(self.stats[m].keys())
        all_classes = sorted(all_classes)

        rows = []
        for cls in all_classes:
            row = {"class": cls,
                   "samples": self.stats[methods[0]][cls][1] if cls in self.stats[methods[0]] else 0}
            for m in methods:
                hits, total = self.stats[m].get(cls, [0, 0])
                acc = (hits / total * 100) if total > 0 else 0.0
                row[f"{m}_hits"] = hits
                row[f"{m}_total"] = total
                row[f"{m}_acc"] = round(acc, 2)
            rows.append(row)
        return rows

    def aggregate_groups(self, methods: List[str]) -> Dict[str, Any]:
        """Compute aggregate stats for KEY / BACKGROUND / OVERALL groups."""
        out = {}
        for group_name, class_set in [
            ("key", KEY_CLASSES),
            ("background", BACKGROUND_CLASSES),
        ]:
            group_stats = {}
            for m in methods:
                hits = sum(s[0] for cls, s in self.stats[m].items()
                           if cls in class_set)
                total = sum(s[1] for cls, s in self.stats[m].items()
                            if cls in class_set)
                acc = (hits / total * 100) if total > 0 else 0.0
                group_stats[m] = {"hits": hits, "total": total,
                                  "acc": round(acc, 2)}
            out[group_name] = group_stats

        # Overall
        overall = {}
        for m in methods:
            hits = sum(s[0] for s in self.stats[m].values())
            total = sum(s[1] for s in self.stats[m].values())
            acc = (hits / total * 100) if total > 0 else 0.0
            overall[m] = {"hits": hits, "total": total, "acc": round(acc, 2)}
        out["overall"] = overall

        return out


# ---------------------------------------------------------------------
# Main evaluation pipeline
# ---------------------------------------------------------------------
def evaluate(dataset_dir: Path,
             calib_path: Path,
             output_dir: Path,
             camera_calib_path: str = "camera_calib.json",
             model_cfg_path: Optional[str] = None,
             ckpt_path: Optional[str] = None,
             tau: float = 80.0,
             truncation_factor: float = 5.0,
             lambda_min: float = 0.2,
             lambda_max: float = 1.0,
             cached_predictions: Optional[Path] = None,
             save_predictions: Optional[Path] = None,
             ) -> Dict[str, Any]:
    """Run the complete evaluation pipeline.

    Args:
        dataset_dir:        directory produced by mask2former_gaze_collection.py
        calib_path:         path to calib.json (Section 5.3)
        output_dir:         where evaluation results will be written
        cached_predictions: optional JSON file with pre-computed gaze points
                            keyed by sample_id (skips model inference)
        save_predictions:   if set, dumps predicted gaze points to this path
                            for reuse in sensitivity / ablation experiments
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------- Load metadata --------------------
    samples = load_metadata(dataset_dir)
    print(f"[INFO] Loaded {len(samples)} samples from {dataset_dir}")
    samples_by_image = index_samples_by_image(samples)
    print(f"[INFO] Source images: {len(samples_by_image)}")

    # -------------------- Pre-compute candidates per image --------------------
    candidates_by_image: Dict[str, List[Dict[str, Any]]] = {}
    for image_name, image_samples in samples_by_image.items():
        cands = load_candidates_for_image(dataset_dir, image_name, image_samples)
        candidates_by_image[image_name] = cands
        print(f"[INFO]   {image_name}: {len(cands)} candidate targets")

    # -------------------- Gaze prediction --------------------
    cached: Dict[str, List[float]] = {}
    if cached_predictions and cached_predictions.exists():
        with cached_predictions.open("r", encoding="utf-8") as f:
            cached = json.load(f)
        print(f"[INFO] Loaded {len(cached)} cached gaze predictions")

    if cached:
        predictor = None
    else:
        print("[INFO] Initializing gaze predictor...")
        predictor = init_gaze_predictor(
            str(calib_path), camera_calib_path,
            model_cfg_path, ckpt_path)

    # -------------------- Iterate samples --------------------
    methods = ["M1", "M2", "M3", "M4"]
    aggregator = AccuracyAggregator()
    per_sample_records: List[Dict[str, Any]] = []
    new_predictions: Dict[str, List[float]] = {}

    n_processed = 0
    n_failed = 0
    t_start = time.time()

    for s in samples:
        sample_id = s.get("sample_id")
        image_name = s.get("source_image_filename") or s.get("original_image")
        if image_name not in candidates_by_image:
            continue

        # Resolve true target_id and class
        target_dict = s.get("target", {})
        true_id = (target_dict.get("target_id")
                   or s.get("target_info", {}).get("target_id")
                   or s.get("target_id"))
        true_class = (target_dict.get("class_name")
                      or s.get("target_info", {}).get("class_name")
                      or s.get("class_name") or "unknown")

        # Determine gaze point
        if sample_id in cached:
            gaze_point = tuple(cached[sample_id])
        else:
            face_path = dataset_dir / "face_images" / s.get("face_image", "")
            gaze_point = predict_gaze_for_sample(predictor, face_path)
            if gaze_point is None:
                n_failed += 1
                continue
            new_predictions[sample_id] = list(gaze_point)

        candidates = candidates_by_image[image_name]
        if not candidates:
            continue

        # Run all four methods
        results = predict_all(
            gaze_point, candidates,
            tau=tau,
            truncation_factor=truncation_factor,
            lambda_min=lambda_min,
            lambda_max=lambda_max,
        )

        # Record per-sample data
        record = {
            "sample_id": sample_id,
            "image": image_name,
            "true_id": str(true_id),
            "true_class": true_class,
            "gaze_u": gaze_point[0],
            "gaze_v": gaze_point[1],
        }
        for m in methods:
            r = results[m]
            pred_class = r.get("predicted_class") or "unknown"
            record[f"{m}_pred_id"] = r.get("predicted_id")
            record[f"{m}_pred_class"] = pred_class
            record[f"{m}_correct"] = int(pred_class == true_class)
            if "lambda_used" in r:
                record[f"{m}_lambda"] = r["lambda_used"]
            if "h_norm" in r:
                record[f"{m}_h_norm"] = r["h_norm"]
            aggregator.record(m, true_class, pred_class)

        per_sample_records.append(record)
        n_processed += 1
        if n_processed % 50 == 0:
            print(f"[INFO]   processed {n_processed} samples...")

    elapsed = time.time() - t_start
    print(f"[INFO] Done: {n_processed} processed, {n_failed} failed "
          f"({elapsed:.1f}s)")

    # -------------------- Save outputs --------------------
    # Save predictions cache
    if save_predictions and new_predictions:
        merged = {**cached, **new_predictions}
        save_predictions.parent.mkdir(parents=True, exist_ok=True)
        with save_predictions.open("w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Saved {len(merged)} cached predictions to {save_predictions}")

    # Per-sample CSV
    per_sample_csv = output_dir / "results_per_sample.csv"
    if per_sample_records:
        keys = list(per_sample_records[0].keys())
        with per_sample_csv.open("w", encoding="utf-8") as f:
            f.write(",".join(keys) + "\n")
            for rec in per_sample_records:
                f.write(",".join(str(rec.get(k, "")) for k in keys) + "\n")
        print(f"[INFO] Per-sample results: {per_sample_csv}")

    # Per-class CSV (Table 5-X)
    per_class_rows = aggregator.per_class_table(methods)
    aggregates = aggregator.aggregate_groups(methods)

    per_class_csv = output_dir / "results_per_class.csv"
    with per_class_csv.open("w", encoding="utf-8") as f:
        header = ["class", "samples"]
        for m in methods:
            header += [f"{m}_hits", f"{m}_total", f"{m}_acc"]
        f.write(",".join(header) + "\n")
        for row in per_class_rows:
            f.write(",".join(str(row.get(k, "")) for k in header) + "\n")
        # Append aggregate rows
        for group in ["key", "background", "overall"]:
            line = [group, sum(aggregates[group][m]["total"] for m in [methods[0]])]
            line = [group, aggregates[group][methods[0]]["total"]]
            for m in methods:
                stats = aggregates[group][m]
                line += [stats["hits"], stats["total"], stats["acc"]]
            f.write(",".join(str(x) for x in line) + "\n")
    print(f"[INFO] Per-class table:   {per_class_csv}")

    # Summary JSON
    summary = {
        "n_samples_processed": n_processed,
        "n_samples_failed": n_failed,
        "elapsed_seconds": round(elapsed, 2),
        "hyperparameters": {
            "tau": tau, "truncation_factor": truncation_factor,
            "lambda_min": lambda_min, "lambda_max": lambda_max,
        },
        "per_class": per_class_rows,
        "aggregates": aggregates,
    }
    summary_path = output_dir / "results_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Summary JSON:      {summary_path}")

    # -------------------- Print human-readable table --------------------
    print()
    print("=" * 78)
    print("Table 5-X — Gaze target identification accuracy")
    print("=" * 78)
    print(f"{'Class':<18}{'N':>6}", end="")
    for m in methods:
        print(f"  {m+' acc':>10}", end="")
    print()
    print("-" * 78)
    for row in per_class_rows:
        print(f"{row['class']:<18}{row['samples']:>6}", end="")
        for m in methods:
            print(f"  {row[f'{m}_acc']:>9.2f}%", end="")
        print()
    print("-" * 78)
    for group in ["key", "background", "overall"]:
        label = {"key": "Key targets", "background": "Background",
                 "overall": "Overall"}[group]
        total = aggregates[group][methods[0]]["total"]
        print(f"{label:<18}{total:>6}", end="")
        for m in methods:
            print(f"  {aggregates[group][m]['acc']:>9.2f}%", end="")
        print()
    print("=" * 78)

    return summary


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Batch evaluation for gaze target identification (M1~M4)"
    )
    parser.add_argument("--dataset-dir", type=str, required=True,
                        help="Directory produced by mask2former_gaze_collection.py")
    parser.add_argument("--calib-path", type=str, required=True,
                        help="Path to calib.json")
    parser.add_argument("--output-dir", type=str, default="./eval_results",
                        help="Where to write evaluation outputs")
    parser.add_argument("--camera-calib-path", type=str,
                        default="camera_calib.json")
    parser.add_argument("--model-cfg-path", type=str, default=None)
    parser.add_argument("--ckpt-path", type=str, default=None)
    parser.add_argument("--tau", type=float, default=80.0,
                        help="Decay scale for edge-distance score")
    parser.add_argument("--truncation-factor", type=float, default=5.0)
    parser.add_argument("--lambda-min", type=float, default=0.2)
    parser.add_argument("--lambda-max", type=float, default=1.0)
    parser.add_argument("--cached-predictions", type=str, default=None,
                        help="Optional cached gaze predictions JSON")
    parser.add_argument("--save-predictions", type=str, default=None,
                        help="If set, dump computed gaze points here")
    args = parser.parse_args()

    evaluate(
        dataset_dir=Path(args.dataset_dir),
        calib_path=Path(args.calib_path),
        output_dir=Path(args.output_dir),
        camera_calib_path=args.camera_calib_path,
        model_cfg_path=args.model_cfg_path,
        ckpt_path=args.ckpt_path,
        tau=args.tau,
        truncation_factor=args.truncation_factor,
        lambda_min=args.lambda_min,
        lambda_max=args.lambda_max,
        cached_predictions=Path(args.cached_predictions)
            if args.cached_predictions else None,
        save_predictions=Path(args.save_predictions)
            if args.save_predictions else None,
    )


if __name__ == "__main__":
    main()