#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Collect gaze-target trials with Mask2Former panoptic targets.

Workflow:
  1. Load all images from a folder.
  2. Run Cityscapes-style Mask2Former panoptic segmentation.
  3. Show each image fullscreen with aspect-fit scaling.
  4. Highlight each candidate target one by one.
  5. Press Space to save the current camera frame and trial metadata.

Controls:
  Space      save one gaze/frame sample for the highlighted target
  N / Right  skip to next target
  B / Left   go back one target
  S          skip the rest of the current image
  Esc / Q    quit

Example:
    python mmdetection/mask2former_gaze_collection.py ^
        --images-dir mmsegmentation/image ^
        --model-dir mmdetection/checkpoints ^
        --output-dir gaze_dataset_mask2former_test
"""

import argparse
import colorsys
import json
import os
import shutil
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageTk
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation


CITYSCAPES_THING_CLASSES = {
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
    "traffic light",
    "traffic sign",
}

SPLIT_COMPONENT_CLASSES = CITYSCAPES_THING_CLASSES | {
    "building",
    "vegetation",
}

EXCLUDED_CLASSES = {
    "pole",
}


CITYSCAPES_PALETTE = {
    "road": (128, 64, 128),
    "sidewalk": (244, 35, 232),
    "building": (70, 70, 70),
    "wall": (102, 102, 156),
    "fence": (190, 153, 153),
    "pole": (153, 153, 153),
    "traffic light": (250, 170, 30),
    "traffic sign": (220, 220, 0),
    "vegetation": (107, 142, 35),
    "terrain": (152, 251, 152),
    "sky": (70, 130, 180),
    "person": (220, 20, 60),
    "rider": (255, 0, 0),
    "car": (0, 0, 142),
    "truck": (0, 0, 70),
    "bus": (0, 60, 100),
    "train": (0, 80, 100),
    "motorcycle": (0, 0, 230),
    "bicycle": (119, 11, 32),
}


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
RESAMPLE_BILINEAR = getattr(Image, "Resampling", Image).BILINEAR
RESAMPLE_NEAREST = getattr(Image, "Resampling", Image).NEAREST


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Fullscreen target-highlighting data collection with Mask2Former."
    )
    parser.add_argument(
        "--images-dir",
        default=str(repo_root / "mmsegmentation" / "image"),
        help="Folder containing source images.",
    )
    parser.add_argument(
        "--model-dir",
        default=str(repo_root / "mmdetection" / "checkpoints"),
        help="Local Hugging Face Mask2Former model directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Dataset output directory. Default: gaze_dataset_mask2former_<timestamp>.",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0.")
    parser.add_argument("--camera-id", type=int, default=0, help="OpenCV camera id.")
    parser.add_argument("--cam-width", type=int, default=1280, help="Camera capture width.")
    parser.add_argument("--cam-height", type=int, default=720, help="Camera capture height.")
    parser.add_argument("--mirror-camera", action="store_true", help="Horizontally flip saved camera frames.")
    parser.add_argument("--min-area", type=int, default=1200, help="Drop tiny panoptic targets.")
    parser.add_argument(
        "--target-mode",
        choices=["things", "stuff", "all"],
        default="all",
        help="Which panoptic targets to present.",
    )
    parser.add_argument(
        "--max-targets-per-image",
        type=int,
        default=0,
        help="0 means no limit. Useful for quick pilot sessions.",
    )
    parser.add_argument("--highlight-alpha", type=float, default=0.58, help="Target mask opacity.")
    parser.add_argument("--dim-alpha", type=float, default=0.18, help="Non-target dimming opacity.")
    parser.add_argument(
        "--copy-calib",
        default=None,
        help="Optional calibration json to copy into the dataset and reference in metadata.",
    )
    parser.add_argument(
        "--warmup-ms",
        type=int,
        default=250,
        help="Delay after drawing a new target before Space is accepted.",
    )
    return parser.parse_args()


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def id2label_from_model(model) -> Dict[int, str]:
    raw = getattr(model.config, "id2label", {})
    return {int(k): str(v) for k, v in raw.items()}


def stable_color(segment_id: int, class_name: str, instance_index: int = 0) -> Tuple[int, int, int]:
    if class_name in CITYSCAPES_PALETTE:
        base = CITYSCAPES_PALETTE[class_name]
        h, s, v = colorsys.rgb_to_hsv(base[0] / 255.0, base[1] / 255.0, base[2] / 255.0)
        rng = np.random.default_rng(segment_id * 1009 + 17)
        if class_name in CITYSCAPES_THING_CLASSES:
            hue_offsets = [-0.105, -0.070, -0.035, 0.0, 0.040, 0.080, 0.120]
            sat_levels = [0.58, 0.78, 0.96, 1.0]
            val_levels = [0.48, 0.68, 0.86, 1.0]
            h = (h + hue_offsets[instance_index % len(hue_offsets)]) % 1.0
            s = max(0.45, sat_levels[(instance_index // len(hue_offsets)) % len(sat_levels)])
            v = max(0.58, val_levels[(instance_index // (len(hue_offsets) * len(sat_levels))) % len(val_levels)])
        else:
            h = (h + float(rng.uniform(-0.018, 0.018))) % 1.0
            s = float(np.clip(s * rng.uniform(0.78, 1.10), 0.45, 1.0))
            v = float(np.clip(v * rng.uniform(0.82, 1.20), 0.45, 1.0))
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return int(round(r * 255)), int(round(g * 255)), int(round(b * 255))
    rng = np.random.default_rng(segment_id * 9173 + 31)
    color = rng.integers(40, 235, size=3)
    return int(color[0]), int(color[1]), int(color[2])


def bbox_from_mask(mask: np.ndarray) -> List[int]:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def centroid_from_mask(mask: np.ndarray) -> List[float]:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return [0.0, 0.0]
    return [float(xs.mean()), float(ys.mean())]


def split_instance_like_components(mask: np.ndarray, min_area: int) -> List[np.ndarray]:
    mask_u8 = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    components = []
    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area >= min_area:
            components.append(labels == label_idx)
    return components or [mask]


def public_target(target: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in target.items() if not key.startswith("_")}


def list_images(images_dir: Path) -> List[Path]:
    files = [
        path for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(files)


class PanopticSegmenter:
    def __init__(self, model_dir: str, device_arg: str, min_area: int):
        self.model_dir = str(Path(model_dir).resolve())
        self.device = choose_device(device_arg)
        self.min_area = min_area
        print(f"[INFO] Loading Mask2Former from {self.model_dir}")
        print(f"[INFO] Device: {self.device}")
        self.processor = AutoImageProcessor.from_pretrained(self.model_dir, local_files_only=True)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            self.model_dir,
            local_files_only=True,
        )
        self.model.to(self.device)
        self.model.eval()
        self.id2label = id2label_from_model(self.model)

    def segment(self, image_path: Path, output_image_dir: Path) -> Dict[str, Any]:
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        result = self.processor.post_process_panoptic_segmentation(
            outputs,
            target_sizes=[(image.height, image.width)],
        )[0]

        segmentation = result["segmentation"].detach().cpu().numpy().astype(np.int32)
        segments_info = result.get("segments_info", [])

        image_key = image_path.stem
        masks_dir = output_image_dir / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_image_dir / f"{image_key}_panoptic_ids.npy", segmentation)

        targets = self._build_targets(segmentation, segments_info, masks_dir)
        targets_json = {
            "source_image": str(image_path.resolve()),
            "image_filename": image_path.name,
            "image_size": {"width": image.width, "height": image.height},
            "model_dir": self.model_dir,
            "num_targets": len(targets),
            "targets": [public_target(target) for target in targets],
        }
        with (output_image_dir / f"{image_key}_targets.json").open("w", encoding="utf-8") as f:
            json.dump(targets_json, f, ensure_ascii=False, indent=2)

        overlay = self.make_context_overlay(image, segmentation, targets, alpha=0.45)
        overlay.save(output_image_dir / f"{image_key}_panoptic_overlay.png")

        return {
            "image": image,
            "segmentation": segmentation,
            "targets": targets,
            "targets_json": targets_json,
        }

    def _build_targets(
        self,
        segmentation: np.ndarray,
        segments_info: List[Dict[str, Any]],
        masks_dir: Path,
    ) -> List[Dict[str, Any]]:
        targets = []
        per_class_counts: Dict[str, int] = {}

        for segment in segments_info:
            segment_id = int(segment["id"])
            label_id = int(segment.get("label_id", segment.get("category_id", -1)))
            class_name = self.id2label.get(label_id, f"class_{label_id}")
            if class_name in EXCLUDED_CLASSES:
                continue
            segment_mask = segmentation == segment_id
            if class_name in SPLIT_COMPONENT_CLASSES:
                component_masks = split_instance_like_components(segment_mask, self.min_area)
            else:
                component_masks = [segment_mask]

            for component_index, mask in enumerate(component_masks):
                area = int(mask.sum())
                if area < self.min_area:
                    continue

                per_class_counts[class_name] = per_class_counts.get(class_name, 0) + 1
                class_index = per_class_counts[class_name] - 1
                target_id = f"{class_name.replace(' ', '_')}_{class_index:02d}_seg{segment_id}_c{component_index}"
                mask_name = f"{target_id}_mask.png"
                cv2.imwrite(str(masks_dir / mask_name), mask.astype(np.uint8) * 255)
                color = stable_color(segment_id, class_name, class_index)

                targets.append(
                    {
                        "target_id": target_id,
                        "target_instance_id": f"seg{segment_id}_c{component_index}",
                        "segment_id": segment_id,
                        "component_index": component_index,
                        "label_id": label_id,
                        "class_name": class_name,
                        "is_thing": class_name in CITYSCAPES_THING_CLASSES,
                        "score": float(segment.get("score", 0.0)),
                        "area": area,
                        "bbox": bbox_from_mask(mask),
                        "centroid": centroid_from_mask(mask),
                        "mask_path": str(Path("masks") / mask_name),
                        "overlay_color": list(color),
                        "_mask": mask,
                    }
                )

        targets.sort(key=lambda item: (not item["is_thing"], item["class_name"], -item["area"]))
        return targets

    @staticmethod
    def make_context_overlay(
        image: Image.Image,
        segmentation: np.ndarray,
        targets: List[Dict[str, Any]],
        alpha: float,
    ) -> Image.Image:
        base = np.asarray(image).astype(np.float32)
        color_layer = np.zeros_like(base)
        covered = np.zeros(segmentation.shape, dtype=bool)
        for target in targets:
            mask = segmentation == int(target["segment_id"])
            color_layer[mask] = np.array(target["overlay_color"], dtype=np.float32)
            covered |= mask
        blended = base.copy()
        blended[covered] = base[covered] * (1.0 - alpha) + color_layer[covered] * alpha
        return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))

    @staticmethod
    def make_color_map(segmentation: np.ndarray, targets: List[Dict[str, Any]]) -> Image.Image:
        color_map = np.zeros((*segmentation.shape, 3), dtype=np.uint8)
        for target in targets:
            mask = target.get("_mask")
            if mask is None:
                mask = segmentation == int(target["segment_id"])
            color_map[mask] = np.array(target["overlay_color"], dtype=np.uint8)
        return Image.fromarray(color_map)


class GazeCollectionApp:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.images_dir = Path(args.images_dir).resolve()
        if args.output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = Path.cwd() / f"gaze_dataset_mask2former_{timestamp}"
        else:
            self.output_dir = Path(args.output_dir).resolve()

        self.images_out_dir = self.output_dir / "images"
        self.panoptic_out_dir = self.output_dir / "panoptic"
        self.face_out_dir = self.output_dir / "face_images"
        self.color_map_out_dir = self.output_dir / "instance_color_maps"
        self.mask_out_dir = self.output_dir / "instance_masks"
        self.screen_mask_out_dir = self.output_dir / "instance_masks_screen"
        self.trial_vis_dir = self.output_dir / "trial_views"
        self.images_out_dir.mkdir(parents=True, exist_ok=True)
        self.panoptic_out_dir.mkdir(parents=True, exist_ok=True)
        self.face_out_dir.mkdir(parents=True, exist_ok=True)
        self.color_map_out_dir.mkdir(parents=True, exist_ok=True)
        self.mask_out_dir.mkdir(parents=True, exist_ok=True)
        self.screen_mask_out_dir.mkdir(parents=True, exist_ok=True)
        self.trial_vis_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.output_dir / "metadata.jsonl"

        self.segmenter = PanopticSegmenter(args.model_dir, args.device, args.min_area)
        self.image_paths = list_images(self.images_dir)
        if not self.image_paths:
            raise RuntimeError(f"No images found in {self.images_dir}")

        self.calib_copy_path = None
        if args.copy_calib:
            src = Path(args.copy_calib).resolve()
            if src.exists():
                self.calib_copy_path = self.output_dir / src.name
                shutil.copy2(src, self.calib_copy_path)
            else:
                print(f"[WARN] Calibration file not found: {src}")

        self.camera = cv2.VideoCapture(args.camera_id, cv2.CAP_DSHOW)
        if not self.camera.isOpened():
            self.camera = cv2.VideoCapture(args.camera_id)
        if not self.camera.isOpened():
            raise RuntimeError(f"Cannot open camera {args.camera_id}")
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.cam_width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cam_height)
        self.camera.set(cv2.CAP_PROP_FPS, 30)

        self.root = tk.Tk()
        self.root.title("Mask2Former Gaze Target Collection")
        self.root.attributes("-fullscreen", True)
        self.root.configure(bg="black")
        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()
        self.canvas = tk.Canvas(
            self.root,
            width=self.screen_width,
            height=self.screen_height,
            bg="black",
            highlightthickness=0,
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.root.bind("<space>", self.on_space)
        self.root.bind("<Right>", self.on_next)
        self.root.bind("n", self.on_next)
        self.root.bind("N", self.on_next)
        self.root.bind("<Left>", self.on_back)
        self.root.bind("b", self.on_back)
        self.root.bind("B", self.on_back)
        self.root.bind("s", self.on_skip_image)
        self.root.bind("S", self.on_skip_image)
        self.root.bind("<Escape>", self.on_quit)
        self.root.bind("q", self.on_quit)
        self.root.bind("Q", self.on_quit)

        self.current_image_index = -1
        self.current_target_index = -1
        self.current_image_path: Optional[Path] = None
        self.current_image: Optional[Image.Image] = None
        self.current_segmentation: Optional[np.ndarray] = None
        self.current_targets: List[Dict[str, Any]] = []
        self.current_image_output_dir: Optional[Path] = None
        self.current_display_rect: Optional[Dict[str, int]] = None
        self.current_color_map_name: Optional[str] = None
        self.tk_image = None
        self.target_ready_at = 0.0
        self.saved_count = 0

        session = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "images_dir": str(self.images_dir),
            "model_dir": str(Path(args.model_dir).resolve()),
            "output_dir": str(self.output_dir),
            "screen": {"width": self.screen_width, "height": self.screen_height},
            "camera": {
                "id": args.camera_id,
                "width": args.cam_width,
                "height": args.cam_height,
                "mirror_camera": bool(args.mirror_camera),
            },
            "target_mode": args.target_mode,
            "min_area": args.min_area,
            "calibration_file": str(self.calib_copy_path) if self.calib_copy_path else None,
        }
        with (self.output_dir / "session.json").open("w", encoding="utf-8") as f:
            json.dump(session, f, ensure_ascii=False, indent=2)

    def run(self) -> None:
        print(f"[INFO] Dataset output: {self.output_dir}")
        print(f"[INFO] Images: {len(self.image_paths)}")
        self.load_next_image()
        self.root.mainloop()

    def filter_targets(self, targets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.args.target_mode == "things":
            targets = [target for target in targets if target["is_thing"]]
        elif self.args.target_mode == "stuff":
            targets = [target for target in targets if not target["is_thing"]]

        if self.args.max_targets_per_image > 0:
            targets = targets[: self.args.max_targets_per_image]
        return targets

    def load_next_image(self) -> None:
        self.current_image_index += 1
        self.current_target_index = -1

        while self.current_image_index < len(self.image_paths):
            image_path = self.image_paths[self.current_image_index]
            print(f"[INFO] Segmenting {image_path.name} ({self.current_image_index + 1}/{len(self.image_paths)})")
            image_output_dir = self.panoptic_out_dir / image_path.stem
            image_output_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, self.images_out_dir / image_path.name)

            panoptic = self.segmenter.segment(image_path, image_output_dir)
            targets = self.filter_targets(panoptic["targets"])
            print(f"[INFO] Targets kept for presentation: {len(targets)}")

            if targets:
                # Save all candidates for evaluation to align with paper's Section 3.4
                all_targets_info = []
                display_rect = self.compute_display_rect(panoptic["image"].width, panoptic["image"].height)
                self.current_display_rect = display_rect  # Temporarily set for save_screen_mask

                all_masks_screen_dir = image_output_dir / "masks_screen"
                all_masks_screen_dir.mkdir(parents=True, exist_ok=True)

                for t in targets:
                    t_mask_bool = t.get("_mask")
                    if t_mask_bool is None:
                        t_mask_bool = panoptic["segmentation"] == int(t["segment_id"])

                    t_mask_u8 = t_mask_bool.astype(np.uint8) * 255
                    screen_mask_name = f"{t['target_id']}_screen_mask.png"
                    screen_mask_path = all_masks_screen_dir / screen_mask_name
                    self.save_screen_mask(t_mask_u8, screen_mask_path)

                    all_targets_info.append({
                        "target_id": t["target_id"],
                        "class_name": t["class_name"],
                        "segment_id": t["segment_id"],
                        "centroid": t["centroid"],
                        "screen_mask_path": f"panoptic/{image_path.stem}/masks_screen/{screen_mask_name}",
                    })

                with (image_output_dir / "all_targets.json").open("w", encoding="utf-8") as f:
                    json.dump({"targets": all_targets_info}, f, ensure_ascii=False, indent=2)

                color_map_name = f"{image_path.stem}_colormap.png"
                color_map = self.segmenter.make_color_map(panoptic["segmentation"], targets)
                color_map.save(self.color_map_out_dir / color_map_name)

                self.current_image_path = image_path
                self.current_image = panoptic["image"]
                self.current_segmentation = panoptic["segmentation"]
                self.current_targets = targets
                self.current_image_output_dir = image_output_dir
                self.current_color_map_name = color_map_name
                self.show_target(0)
                return

            self.current_image_index += 1

        self.finish()

    def show_target(self, target_index: int) -> None:
        if not self.current_targets:
            self.load_next_image()
            return

        target_index = max(0, min(target_index, len(self.current_targets) - 1))
        self.current_target_index = target_index
        self.target_ready_at = time.time() + self.args.warmup_ms / 1000.0

        target = self.current_targets[self.current_target_index]
        rendered = self.render_trial_view(self.current_image, self.current_segmentation, target)
        self.tk_image = ImageTk.PhotoImage(rendered)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.tk_image, anchor=tk.NW)

        label = (
            f"{self.current_image_index + 1}/{len(self.image_paths)}  "
            f"target {self.current_target_index + 1}/{len(self.current_targets)}  "
            f"{target['class_name']}  {target['target_id']}  "
            f"Space=save  N=next  S=skip image  Esc=quit"
        )
        self.canvas.create_rectangle(
            18,
            18,
            min(self.screen_width - 18, 1040),
            56,
            outline="",
            fill="black",
        )
        self.canvas.create_text(
            24,
            24,
            text=label,
            anchor=tk.NW,
            fill="white",
            font=("Arial", 16, "bold"),
        )

    def compute_display_rect(self, image_width: int, image_height: int) -> Dict[str, int]:
        scale = min(self.screen_width / image_width, self.screen_height / image_height)
        display_w = int(round(image_width * scale))
        display_h = int(round(image_height * scale))
        x0 = (self.screen_width - display_w) // 2
        y0 = (self.screen_height - display_h) // 2
        return {
            "x": x0,
            "y": y0,
            "width": display_w,
            "height": display_h,
            "screen_width": self.screen_width,
            "screen_height": self.screen_height,
            "scale": float(scale),
        }

    def render_trial_view(
        self,
        image: Image.Image,
        segmentation: np.ndarray,
        target: Dict[str, Any],
    ) -> Image.Image:
        image_rgb = np.asarray(image).astype(np.float32)
        target_mask = target.get("_mask")
        if target_mask is None:
            target_mask = segmentation == int(target["segment_id"])

        dark = image_rgb * (1.0 - self.args.dim_alpha)
        highlight = dark.copy()
        color = np.array(target["overlay_color"], dtype=np.float32)
        a = self.args.highlight_alpha
        highlight[target_mask] = image_rgb[target_mask] * (1.0 - a) + color * a
        rendered = Image.fromarray(np.clip(highlight, 0, 255).astype(np.uint8))

        draw = ImageDraw.Draw(rendered)
        bbox = target["bbox"]
        draw.rectangle(tuple(bbox), outline=(255, 255, 255), width=5)
        cx, cy = target["centroid"]
        r = 12
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(255, 255, 255), width=4)

        self.current_display_rect = self.compute_display_rect(image.width, image.height)
        display_w = self.current_display_rect["width"]
        display_h = self.current_display_rect["height"]
        resized = rendered.resize((display_w, display_h), RESAMPLE_BILINEAR)

        canvas_img = Image.new("RGB", (self.screen_width, self.screen_height), (0, 0, 0))
        canvas_img.paste(resized, (self.current_display_rect["x"], self.current_display_rect["y"]))

        trial_name = self.make_trial_id(target)
        canvas_img.save(self.trial_vis_dir / f"{trial_name}_display.png")
        return canvas_img

    def make_trial_id(self, target: Dict[str, Any]) -> str:
        image_stem = self.current_image_path.stem if self.current_image_path else "image"
        return f"{image_stem}_{self.current_target_index:04d}_{target['target_id']}"

    def on_space(self, event=None) -> None:
        if time.time() < self.target_ready_at:
            return
        self.save_current_sample()
        self.advance_target()

    def on_next(self, event=None) -> None:
        self.advance_target()

    def on_back(self, event=None) -> None:
        if self.current_target_index > 0:
            self.show_target(self.current_target_index - 1)

    def on_skip_image(self, event=None) -> None:
        self.load_next_image()

    def on_quit(self, event=None) -> None:
        self.finish()

    def advance_target(self) -> None:
        next_index = self.current_target_index + 1
        if next_index < len(self.current_targets):
            self.show_target(next_index)
        else:
            self.load_next_image()

    def capture_frame(self) -> Optional[np.ndarray]:
        # Read a few frames so the saved one reflects the current moment.
        frame = None
        for _ in range(3):
            ok, candidate = self.camera.read()
            if ok:
                frame = candidate
        if frame is None:
            return None
        if self.args.mirror_camera:
            frame = cv2.flip(frame, 1)
        return frame

    def save_current_sample(self) -> None:
        if self.current_image_path is None or self.current_display_rect is None:
            return
        target = self.current_targets[self.current_target_index]
        trial_id = self.make_trial_id(target)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        sample_id = f"{trial_id}_{timestamp}"

        target_mask_bool = target.get("_mask")
        if target_mask_bool is None:
            target_mask_bool = self.current_segmentation == int(target["segment_id"])
        target_mask = target_mask_bool.astype(np.uint8) * 255
        original_mask_name = f"{sample_id}_mask.png"
        original_mask_path = self.mask_out_dir / original_mask_name
        cv2.imwrite(str(original_mask_path), target_mask)

        screen_mask_name = f"{sample_id}_screen_mask.png"
        screen_mask_path = self.screen_mask_out_dir / screen_mask_name
        self.save_screen_mask(target_mask, screen_mask_path)

        frame = self.capture_frame()
        face_image_name = f"{sample_id}_face.jpg"
        face_image_path = self.face_out_dir / face_image_name
        camera_ok = frame is not None
        if camera_ok:
            cv2.imwrite(str(face_image_path), frame)

        metadata = {
            "sample_id": sample_id,
            "created_at": datetime.now().isoformat(timespec="milliseconds"),
            "original_image": self.current_image_path.name,
            "face_image": face_image_name if camera_ok else None,
            "instance_color_map": self.current_color_map_name,
            "instance_mask_original": original_mask_name,
            "instance_mask_screen": screen_mask_name,
            "source_image": str(self.current_image_path.resolve()),
            "source_image_filename": self.current_image_path.name,
            "copied_image": str((self.images_out_dir / self.current_image_path.name).resolve()),
            "panoptic_dir": str(self.current_image_output_dir.resolve()) if self.current_image_output_dir else None,
            "face_image_path": str(face_image_path.resolve()) if camera_ok else None,
            "camera_frame_saved": camera_ok,
            "calibration_file": str(self.calib_copy_path.resolve()) if self.calib_copy_path else None,
            "display_rect": self.current_display_rect,
            "image_size": {
                "width": int(self.current_image.width),
                "height": int(self.current_image.height),
            },
            "target_index": self.current_target_index,
            "num_targets_in_image": len(self.current_targets),
            "target": public_target(target),
            "object_info": {
                "class_id": target["label_id"],
                "object_id": target["segment_id"],
                "component_index": target.get("component_index", 0),
                "class_name": target["class_name"],
                "is_thing": target["is_thing"],
                "palette_color": target["overlay_color"],
                "bbox": target["bbox"],
                "centroid": target["centroid"],
                "area": target["area"],
                "mask_path": target["mask_path"],
            },
            "keys": {
                "save": "space",
                "next": "n/right",
                "skip_image": "s",
                "quit": "esc/q",
            },
        }

        with self.metadata_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(metadata, ensure_ascii=False) + "\n")

        self.saved_count += 1
        print(f"[SAVED] {sample_id}")

    def save_screen_mask(self, target_mask: np.ndarray, output_path: Path) -> None:
        rect = self.current_display_rect
        if rect is None:
            cv2.imwrite(str(output_path), target_mask)
            return

        mask_img = Image.fromarray(target_mask)
        resized = mask_img.resize((rect["width"], rect["height"]), RESAMPLE_NEAREST)
        screen_mask = Image.new("L", (self.screen_width, self.screen_height), 0)
        screen_mask.paste(resized, (rect["x"], rect["y"]))
        screen_mask.save(output_path)

    def finish(self) -> None:
        print(f"[INFO] Saved samples: {self.saved_count}")
        print(f"[INFO] Metadata: {self.metadata_path}")
        try:
            if self.camera is not None:
                self.camera.release()
        finally:
            try:
                self.root.destroy()
            except Exception:
                pass


def main() -> int:
    args = parse_args()
    app = GazeCollectionApp(args)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
