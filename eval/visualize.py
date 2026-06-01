#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone visualization script for gaze target identification evaluation.

Reads results_per_sample.csv produced by evaluate_gaze_target.py and
generates per-sample diagnostic visualizations. Images are organized
into subdirectories based on which methods predicted correctly.

Usage:
    # Visualize all "interesting" samples (M1~M4 disagree):
    python visualize_evaluation.py \
        --results-csv eval_results/results_per_sample.csv \
        --dataset-dir gaze_dataset \
        --output-dir eval_results/visualizations \
        --mode interesting

    # Visualize only M4 failure cases:
    python visualize_evaluation.py \
        --results-csv eval_results/results_per_sample.csv \
        --dataset-dir gaze_dataset \
        --output-dir eval_results/visualizations \
        --mode failures
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, Rectangle


# ---------------------------------------------------------------------
# Visual style constants
# ---------------------------------------------------------------------
PLT_STYLE = {
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "savefig.dpi": 120,
    "savefig.bbox": "tight",
}

METHOD_COLORS = {
    "M1": "#F5C842",   # yellow — centroid distance
    "M2": "#F08C2E",   # orange — edge distance
    "M3": "#E84A4A",   # red — fixed semantic fusion
    "M4": "#7E3CB8",   # purple — adaptive fusion
}

GT_COLOR = "#15B5B0"      # cyan — ground truth marker
GAZE_COLOR = "#E84A4A"    # red — predicted gaze point cross


# ---------------------------------------------------------------------
# Categorize samples by method correctness pattern
# ---------------------------------------------------------------------
def categorize_sample(record: Dict[str, Any]) -> str:
    """Decide which subdirectory a sample belongs to."""
    m1 = int(record.get("M1_correct", 0))
    m2 = int(record.get("M2_correct", 0))
    m3 = int(record.get("M3_correct", 0))
    m4 = int(record.get("M4_correct", 0))

    pattern = (m1, m2, m3, m4)

    if pattern == (1, 1, 1, 1):
        return "all_correct"
    if pattern == (0, 0, 0, 0):
        return "all_wrong"
    if pattern == (0, 0, 0, 1):
        return "m4_only_correct"
    if pattern == (0, 0, 1, 1):
        return "m3_m4_correct"
    if pattern == (0, 1, 1, 1):
        return "m2_m3_m4_correct"
    return "other"


# ---------------------------------------------------------------------
# Load results CSV produced by evaluate_gaze_target.py
# ---------------------------------------------------------------------
def load_results(results_csv: Path) -> List[Dict[str, Any]]:
    records = []
    with results_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            for key in list(row.keys()):
                v = row[key]
                if v == "":
                    row[key] = None
                    continue
                if key in ("gaze_u", "gaze_v",
                           "M3_lambda", "M4_lambda", "M4_h_norm"):
                    try:
                        row[key] = float(v)
                    except (ValueError, TypeError):
                        row[key] = None
                elif key.endswith("_correct"):
                    try:
                        row[key] = int(v)
                    except (ValueError, TypeError):
                        row[key] = 0
            records.append(row)
    return records


# ---------------------------------------------------------------------
# Filter samples by mode
# ---------------------------------------------------------------------
def filter_records(records: List[Dict[str, Any]],
                   mode: str,
                   max_samples: int = 500,
                   seed: int = 42) -> List[Dict[str, Any]]:
    if mode == "all":
        selected = records
    elif mode == "interesting":
        selected = [r for r in records if not (
            r.get("M1_correct") == r.get("M2_correct") ==
            r.get("M3_correct") == r.get("M4_correct"))]
    elif mode == "failures":
        selected = [r for r in records if r.get("M4_correct") == 0]
    elif mode.startswith("sample"):
        parts = mode.split()
        n = int(parts[1]) if len(parts) > 1 else 50
        rng = random.Random(seed)
        selected = rng.sample(records, min(n, len(records)))
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if len(selected) > max_samples:
        rng = random.Random(seed)
        selected = rng.sample(selected, max_samples)
    return selected


# ---------------------------------------------------------------------
# Resource loading helpers
# ---------------------------------------------------------------------
def find_original_image(dataset_dir: Path, image_name: str) -> Optional[Path]:
    """Search for the original scene image."""
    candidates = [
        dataset_dir / "images" / image_name,
        dataset_dir / "images" / f"{Path(image_name).stem}.png",
        dataset_dir / "images" / f"{Path(image_name).stem}.jpg",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def find_color_map(dataset_dir: Path, image_stem: str) -> Optional[Path]:
    """Search for the panoptic color map for that image."""
    candidates = [
        dataset_dir / "instance_color_maps" / f"{image_stem}_colormap.png",
        dataset_dir / "instance_color_maps" / f"{image_stem}.png",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def load_all_targets(dataset_dir: Path,
                     image_stem: str) -> List[Dict[str, Any]]:
    """Load the all_targets.json describing every panoptic-detected target."""
    p = dataset_dir / "panoptic" / image_stem / "all_targets.json"
    if not p.exists():
        return []
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("targets", [])
    except Exception:
        return []


def lookup_target(targets: List[Dict[str, Any]],
                  target_id: str) -> Optional[Dict[str, Any]]:
    """Find a target by its target_id."""
    if target_id is None:
        return None
    for t in targets:
        if str(t.get("target_id")) == str(target_id):
            return t
    return None


def load_screen_mask(dataset_dir: Path,
                     screen_mask_path: str) -> Optional[np.ndarray]:
    """Load a screen-space mask given the relative path stored in JSON."""
    if not screen_mask_path:
        return None
    full = dataset_dir / screen_mask_path
    if not full.exists():
        return None
    return cv2.imread(str(full), cv2.IMREAD_GRAYSCALE)


def compute_centroid_from_mask(mask: np.ndarray) -> Tuple[float, float]:
    """Geometric centroid of a binary mask in screen coordinates."""
    if mask is None or mask.max() == 0:
        return (0.0, 0.0)
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    binary = (mask > 127).astype(np.uint8) if mask.max() > 1 else (mask > 0).astype(np.uint8)
    moments = cv2.moments(binary)
    if moments["m00"] < 1e-6:
        ys, xs = np.where(binary > 0)
        if len(xs) == 0:
            h, w = binary.shape[:2]
            return (w / 2.0, h / 2.0)
        return (float(xs.mean()), float(ys.mean()))
    return (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])


# ---------------------------------------------------------------------
# Build per-sample visualization
# ---------------------------------------------------------------------
def render_visualization(record: Dict[str, Any],
                          dataset_dir: Path,
                          output_path: Path) -> bool:
    """Render the 2x2 diagnostic figure for one sample."""
    image_name = record.get("image")
    if not image_name:
        return False
    image_stem = Path(image_name).stem

    # ---- Load resources ----
    original_path = find_original_image(dataset_dir, image_name)
    color_map_path = find_color_map(dataset_dir, image_stem)
    all_targets = load_all_targets(dataset_dir, image_stem)

    if original_path is None or not all_targets:
        return False

    original_img = cv2.imread(str(original_path))
    if original_img is None:
        return False
    original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

    color_map_img = None
    if color_map_path and color_map_path.exists():
        cm = cv2.imread(str(color_map_path))
        if cm is not None:
            color_map_img = cv2.cvtColor(cm, cv2.COLOR_BGR2RGB)

    # Resolve true target
    true_id = record.get("true_id")
    true_target = lookup_target(all_targets, true_id)

    # Resolve predicted targets per method
    pred_targets = {}
    for method in ("M1", "M2", "M3", "M4"):
        pid = record.get(f"{method}_pred_id")
        pred_targets[method] = lookup_target(all_targets, pid)

    # ---- Build figure ----
    plt.rcParams.update(PLT_STYLE)
    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05],
                          hspace=0.32, wspace=0.18)

    # === Top-left: Original image with sample metadata ===
    ax_tl = fig.add_subplot(gs[0, 0])
    ax_tl.imshow(original_img_rgb)
    ax_tl.set_title("(a) Original Scene", loc="left", fontweight="bold")
    ax_tl.axis("off")
    info_text = (
        f"Sample: {record.get('sample_id', 'N/A')}\n"
        f"Source: {image_name}\n"
        f"True class: {record.get('true_class', 'N/A')}\n"
        f"True ID: {true_id}"
    )
    ax_tl.text(0.02, 0.98, info_text, transform=ax_tl.transAxes,
               fontsize=8, va="top", ha="left", color="white",
               bbox=dict(boxstyle="round,pad=0.4",
                         facecolor="black", alpha=0.7, edgecolor="none"))

    # === Top-right: Panoptic color map ===
    ax_tr = fig.add_subplot(gs[0, 1])
    if color_map_img is not None:
        ax_tr.imshow(color_map_img)
    else:
        ax_tr.imshow(original_img_rgb)
        ax_tr.text(0.5, 0.5, "[color map unavailable]",
                   transform=ax_tr.transAxes,
                   ha="center", va="center", color="white", fontsize=12,
                   bbox=dict(facecolor="black", alpha=0.6))
    ax_tr.set_title(f"(b) Panoptic Segmentation  "
                    f"({len(all_targets)} candidates)",
                    loc="left", fontweight="bold")
    ax_tr.axis("off")

    # === Bottom-left: Decision visualization ===
    ax_bl = fig.add_subplot(gs[1, 0])
    ax_bl.imshow(original_img_rgb)
    ax_bl.set_title("(c) Predictions Compared",
                    loc="left", fontweight="bold")
    ax_bl.axis("off")

    img_h, img_w = original_img_rgb.shape[:2]

    # The gaze point and centroids in `all_targets.json` are stored in the
    # ORIGINAL IMAGE coordinate system (for centroids) or SCREEN coordinate
    # system (for gaze_u/gaze_v). To overlay them on the original image, we
    # transform screen-space gaze to image-space via the letterbox formula
    # used by the collector. Without explicit display_rect data here, we
    # approximate by assuming centroids in all_targets.json are in IMAGE
    # coordinates, which matches the collector implementation.
    #
    # For the gaze point: it's in screen coords (2194x1234). We need to
    # transform back to image coords. Using letterbox: scale = min(sw/iw,
    # sh/ih); offset_x = (sw - iw*scale)/2; offset_y = (sh - ih*scale)/2.

    # Reconstruct letterbox transform (default screen 2194x1234)
    # Better: read from session.json if available
    screen_w, screen_h = 2194, 1234
    session_path = dataset_dir / "session.json"
    if session_path.exists():
        try:
            with session_path.open("r", encoding="utf-8") as f:
                session = json.load(f)
            screen_w = session.get("screen", {}).get("width", screen_w)
            screen_h = session.get("screen", {}).get("height", screen_h)
        except Exception:
            pass

    scale = min(screen_w / img_w, screen_h / img_h)
    disp_w = img_w * scale
    disp_h = img_h * scale
    off_x = (screen_w - disp_w) / 2
    off_y = (screen_h - disp_h) / 2

    def screen_to_image(u: float, v: float) -> Tuple[float, float]:
        ix = (u - off_x) / scale
        iy = (v - off_y) / scale
        return (ix, iy)

    # Mark ground truth target with cyan ellipse (around its centroid)
    if true_target is not None:
        cx, cy = true_target.get("centroid", (img_w / 2, img_h / 2))
        ellipse = Ellipse((cx, cy), width=img_w * 0.06, height=img_h * 0.10,
                          fill=False, edgecolor=GT_COLOR, linewidth=3,
                          linestyle="-", zorder=4)
        ax_bl.add_patch(ellipse)
        ax_bl.text(cx, cy - img_h * 0.06,
                   f"GT: {record.get('true_class', '?')}",
                   color=GT_COLOR, fontsize=8, fontweight="bold",
                   ha="center", va="bottom",
                   bbox=dict(boxstyle="round,pad=0.2",
                             facecolor="white", alpha=0.85, edgecolor=GT_COLOR))

    # Plot gaze point (transformed to image coords)
    gaze_u = record.get("gaze_u")
    gaze_v = record.get("gaze_v")
    if gaze_u is not None and gaze_v is not None:
        gx, gy = screen_to_image(gaze_u, gaze_v)
        cross_size = max(img_w, img_h) * 0.018
        ax_bl.plot([gx - cross_size, gx + cross_size], [gy, gy],
                   color=GAZE_COLOR, linewidth=2.5, zorder=5)
        ax_bl.plot([gx, gx], [gy - cross_size, gy + cross_size],
                   color=GAZE_COLOR, linewidth=2.5, zorder=5)
        ax_bl.scatter([gx], [gy], s=60, marker="o", facecolor="none",
                      edgecolor=GAZE_COLOR, linewidth=2, zorder=5)

    # Plot each method's predicted target centroid
    method_offsets = [(0.02, 0.0), (-0.02, 0.0), (0.0, 0.02), (0.0, -0.02)]
    for i, method in enumerate(("M1", "M2", "M3", "M4")):
        t = pred_targets[method]
        if t is None:
            continue
        cx, cy = t.get("centroid", (img_w / 2, img_h / 2))
        # Add small perturbation so overlapping markers are distinguishable
        dx, dy = method_offsets[i]
        cx_p = cx + dx * img_w
        cy_p = cy + dy * img_h
        correct = bool(record.get(f"{method}_correct", 0))
        marker_edge = METHOD_COLORS[method]
        ax_bl.scatter([cx_p], [cy_p], s=140,
                      marker="o", facecolor=marker_edge,
                      edgecolor="white" if correct else "black",
                      linewidth=2 if correct else 1.5,
                      alpha=0.85, zorder=6)
        ax_bl.text(cx_p, cy_p, method, fontsize=7,
                   color="white", ha="center", va="center",
                   fontweight="bold", zorder=7)

    # Bottom-left legend
    legend_lines = []
    legend_labels = []
    for method, color in METHOD_COLORS.items():
        legend_lines.append(plt.Line2D([0], [0], marker="o",
                                       color="w", markerfacecolor=color,
                                       markersize=10, markeredgecolor="black"))
        legend_labels.append(method)
    legend_lines.append(plt.Line2D([0], [0], marker="o",
                                   color="w", markerfacecolor="none",
                                   markersize=10, markeredgecolor=GT_COLOR,
                                   markeredgewidth=2))
    legend_labels.append("Ground truth")
    legend_lines.append(plt.Line2D([0], [0], color=GAZE_COLOR,
                                   linewidth=2.5, marker="+", markersize=10))
    legend_labels.append("Gaze point")
    ax_bl.legend(legend_lines, legend_labels,
                 loc="lower right", fontsize=7,
                 framealpha=0.9, ncol=3)

    # === Bottom-right: Decision summary table ===
    ax_br = fig.add_subplot(gs[1, 1])
    ax_br.axis("off")
    ax_br.set_title("(d) Method Decisions", loc="left", fontweight="bold")

    table_data = []
    for method in ("M1", "M2", "M3", "M4"):
        t = pred_targets[method]
        pred_class = record.get(f"{method}_pred_class") or "—"
        pred_id = record.get(f"{method}_pred_id") or "—"
        # Truncate long IDs for display
        pred_id_short = (pred_id[:18] + "..." if len(str(pred_id)) > 18
                          else pred_id)
        correct = bool(record.get(f"{method}_correct", 0))
        mark = "✓" if correct else "✗"
        row = [method, pred_class, pred_id_short, mark]
        table_data.append(row)

    # Render table
    cell_colors = []
    for row in table_data:
        method = row[0]
        correct = row[3] == "✓"
        bg = "#E8F8E8" if correct else "#FBE6E6"
        method_bg = METHOD_COLORS[method] + "33"  # transparent suffix won't work
        cell_colors.append([METHOD_COLORS[method] + "55", bg, bg, bg])

    table = ax_br.table(
        cellText=table_data,
        colLabels=["Method", "Pred Class", "Pred ID", "Result"],
        loc="upper center",
        cellLoc="center",
        colWidths=[0.13, 0.32, 0.40, 0.13],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.6)

    # Style header row
    for j in range(4):
        cell = table[0, j]
        cell.set_facecolor("#D0D0D0")
        cell.set_text_props(fontweight="bold")

    # Style each data row
    for i, row in enumerate(table_data, start=1):
        method = row[0]
        correct = row[3] == "✓"
        method_color = METHOD_COLORS[method]
        # Method column
        table[i, 0].set_facecolor(method_color)
        table[i, 0].set_text_props(color="white", fontweight="bold")
        # Other columns
        bg = "#E8F8E8" if correct else "#FBE6E6"
        for j in (1, 2, 3):
            table[i, j].set_facecolor(bg)
        if correct:
            table[i, 3].set_text_props(color="#1A7B1A", fontweight="bold")
        else:
            table[i, 3].set_text_props(color="#A02020", fontweight="bold")

    # Add metadata box below table (H_norm and lambda)
    h_norm = record.get("M4_h_norm")
    lam = record.get("M4_lambda")
    meta_lines = []
    if h_norm is not None:
        ambiguity = ("low" if h_norm < 0.4 else
                     "medium" if h_norm < 0.7 else "high")
        meta_lines.append(f"Scene ambiguity: H_norm = {h_norm:.3f}  ({ambiguity})")
    if lam is not None:
        meta_lines.append(f"Adaptive λ (M4) = {lam:.3f}")
    if meta_lines:
        ax_br.text(0.5, 0.02, "\n".join(meta_lines),
                   transform=ax_br.transAxes,
                   ha="center", va="bottom", fontsize=8.5,
                   bbox=dict(boxstyle="round,pad=0.5",
                             facecolor="#F4F6FA", edgecolor="0.7"))

    # ---- Suptitle ----
    category = categorize_sample(record)
    cat_label = {
        "all_correct": "All methods correct",
        "all_wrong": "All methods failed",
        "m4_only_correct": "Only M4 correct (adaptive fusion key value)",
        "m3_m4_correct": "M3 & M4 correct (semantic prior key value)",
        "m2_m3_m4_correct": "M2~M4 correct (edge distance key value)",
        "other": "Mixed pattern",
    }.get(category, category)
    fig.suptitle(f"{record.get('sample_id', '')}  —  {cat_label}",
                 fontsize=11, fontweight="bold", y=0.985)

    # ---- Save ----
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return True


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate per-sample visualizations from "
                    "evaluate_gaze_target.py results"
    )
    parser.add_argument("--results-csv", type=str, required=True,
                        help="Path to results_per_sample.csv")
    parser.add_argument("--dataset-dir", type=str, required=True,
                        help="Original dataset directory (with panoptic/, "
                             "images/, instance_color_maps/, session.json)")
    parser.add_argument("--output-dir", type=str, default="./visualizations",
                        help="Output directory for visualizations")
    parser.add_argument("--mode", type=str, default="interesting",
                        help="Selection mode: 'all' | 'interesting' | "
                             "'failures' | 'sample N'")
    parser.add_argument("--max-samples", type=int, default=500,
                        help="Hard cap on the number of generated images "
                             "(default 500)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results_csv = Path(args.results_csv)
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)

    if not results_csv.exists():
        print(f"[ERROR] CSV file not found: {results_csv}")
        sys.exit(1)
    if not dataset_dir.exists():
        print(f"[ERROR] Dataset directory not found: {dataset_dir}")
        sys.exit(1)

    print(f"[INFO] Loading results from: {results_csv}")
    records = load_results(results_csv)
    print(f"[INFO] Total records: {len(records)}")

    print(f"[INFO] Filtering by mode: {args.mode}")
    selected = filter_records(records, args.mode,
                              max_samples=args.max_samples,
                              seed=args.seed)
    print(f"[INFO] Selected for visualization: {len(selected)}")

    # Pre-create category subdirectories
    categories = ["all_correct", "all_wrong", "m4_only_correct",
                  "m3_m4_correct", "m2_m3_m4_correct", "other"]
    for cat in categories:
        (output_dir / cat).mkdir(parents=True, exist_ok=True)

    # Render each sample
    n_success = 0
    n_failed = 0
    cat_counts = {c: 0 for c in categories}

    for i, record in enumerate(selected):
        sample_id = record.get("sample_id", f"sample_{i}")
        true_class = record.get("true_class", "unknown")
        category = categorize_sample(record)
        cat_counts[category] += 1

        out_name = f"{sample_id}__{true_class}.png".replace("/", "_")
        out_path = output_dir / category / out_name

        try:
            ok = render_visualization(record, dataset_dir, out_path)
            if ok:
                n_success += 1
            else:
                n_failed += 1
                print(f"[WARN] Skipped {sample_id}: missing resources")
        except Exception as e:
            n_failed += 1
            print(f"[ERROR] Failed to render {sample_id}: {e}")

        if (i + 1) % 50 == 0:
            print(f"[INFO]   processed {i + 1}/{len(selected)}...")

    print()
    print("=" * 60)
    print(f"[DONE] Success: {n_success}, Failed: {n_failed}")
    print(f"[DONE] Output directory: {output_dir.resolve()}")
    print("[DONE] Distribution by category:")
    for cat, count in cat_counts.items():
        print(f"         {cat:<22}: {count}")
    print("=" * 60)


if __name__ == "__main__":
    main()