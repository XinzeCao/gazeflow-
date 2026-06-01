# Gazeflow — Driver Gaze Target Identification

A modular pipeline for **driver gaze target identification** in driving scenes: given a face image from an in-cabin camera, predict which object on the road the driver is looking at.

## Pipeline

```
Scene Image → Panoptic Segmentation (Mask2Former) → Instance Extraction
                                                              ↓
Face Image → Landmarks (MediaPipe) → Gaze Features (MAE-L16) → Gaze-to-Screen Mapping (Ridge)
                                                              ↓
                                               Pixel-level Hit Check → Target Inference (M1–M4)
                                                              ↓
                                               Evaluation Report + Visualization
```

## Project Structure

```
gazeflow/
├── gazemodel/                     Gaze estimation: face → gaze direction
│   ├── extractor.py                GazeFeatureExtractor (main class)
│   ├── gaze_math.py                Pitch/yaw math, head pose, face geometry
│   ├── model_utils.py              Dynamic model loading + image transforms
│   ├── config.yaml                 MAE_Gaze model config
│   ├── face_model.txt              ★ 3D face landmarks (download separately)
│   ├── models/
│   │   ├── mae.py                  Vision Transformer (MAE)
│   │   └── mae_gaze.py             MAE_Gaze wrapper
│   └── weights/
│       └── .gitkeep                ★ Put unigaze_l16_joint.pth.tar here
│
├── calibration/                   Camera + gaze calibration
│   ├── camera.py                   Camera intrinsics (OpenCV checkerboard)
│   ├── camera_calib.json           Example camera intrinsics
│   ├── gaze_calib.py               Calibration GUI (5×5 grid → calib.json)
│   └── predictor.py                OfflineGazePredictor: features → screen (u,v)
│
├── scene_understanding/           Scene segmentation
│   └── mask2former_gaze_collection.py  Mask2Former panoptic data collection GUI
│
├── eval/                          Evaluation & visualization
│   ├── evaluate.py                 M1–M4 batch evaluation (per-sample/class/summary)
│   └── visualize.py                Diagnostic 2×2 visualizations
│
├── lbw_validation/                External cross-validation (LookBothWays dataset)
│   ├── validate_on_lbw.py          Main LBW validation entry
│   ├── lbw_adapter.py              LBW dataset adapter
│   ├── lbw_cached.py               Cached inference variant
│   ├── lbw_dataset.py              LBW dataset loader
│   ├── lbw_end2end.py              End-to-end LBW evaluation
│   ├── lbw_sensitivity.py          τ/λ sensitivity analysis on LBW
│   ├── lbw_visualize.py            LBW result visualization
│   └── gaze_to_scene_calibrator.py Gaze-to-scene calibration for LBW
│
├── gaze_target_inference.py       M1–M4 target inference algorithms (core library)
├── 简介.txt                        Project description (Chinese)
├── 运行指令.txt                     Run commands reference (Chinese)
└── README.md                       This file
```

## Core Models

| Model | Role |
|---|---|
| **UniGaze (MAE-L16)** | Face gaze feature extraction (768-dim) |
| **Mask2Former** | Panoptic scene segmentation |
| **MediaPipe FaceMesh** | 478-point face landmark detection |
| **Ridge Regression** | Polynomial features → screen gaze coordinates |

External dependencies:

```bash
pip install mmdet mmseg mmengine face_alignment mediapipe sklearn omegaconf
```

### Required Weights (download separately)

- `gazemodel/weights/unigaze_l16_joint.pth.tar` — UniGaze MAE-L16 checkpoint (~1.2 GB)
- `gazemodel/face_model.txt` — 3D face model points (from UniGaze upstream)

## M1–M4 Target Inference Algorithms

Implements four ablation methods for identifying which scene object the driver is looking at:

| Method | Description |
|---|---|
| **M1** | Pure centroid distance — pick the closest target center |
| **M2** | Edge distance with exponential decay (`S = exp(-d_edge / τ)`) |
| **M3** | M2 + fixed-weight semantic fusion (`S_final = S_geo × W_sem^λ`) |
| **M4** | M2 + **entropy-driven adaptive fusion** — λ adapts to geometric ambiguity |

### How M4 works

1. Compute geometric scores `S_geo` for all candidates in the effective set
2. Calculate normalized Shannon entropy `H_norm` of the score distribution
   - **Low** `H_norm` → one target clearly dominates → rely on geometry (`λ → λ_min`)
   - **High** `H_norm` → ambiguous (e.g., boundary between car/building) → boost semantic prior (`λ → λ_max`)
3. `λ = λ_min + (λ_max − λ_min) × H_norm`
4. Final score: `S_final(i) = S_geo_norm(i) × W_sem(class_i)^λ`

## Quick Start

### 0. Verify Environment

```bash
ls gazemodel/weights/unigaze_l16_joint.pth.tar
ls gazemodel/face_model.txt
ls calibration/camera_calib.json
```

### 1. Gaze-to-Screen Calibration

The participant sits in front of the screen. Run the 5×5 grid calibration GUI:

```bash
python calibration/gaze_calib.py \
    --screen-width 2194 --screen-height 1234 \
    --grid-cols 5 --grid-rows 5 --camera-id 0 \
    --camera-calib calibration/camera_calib.json \
    --calibration-margin-ratio 0.10 \
    --output-dir calib_runs/run_001
```

Verify calibration error (should be < 30 px):

```bash
python -c "import json; d=json.load(open('calib_runs/run_001/calib.json','r',encoding='utf-8')); s=d['statistics']; print(f'Mean: {s[\"mean_error_px\"]:.2f}px  Median: {s[\"median_error_px\"]:.2f}px')"
```

### 2. Data Collection

Run Mask2Former panoptic segmentation to collect labeled samples:

```bash
python scene_understanding/mask2former_gaze_collection.py \
    --images-dir <images_dir> \
    --model-dir <model_dir> \
    --output-dir gaze_dataset \
    --target-mode all --camera-id 0 \
    --copy-calib calib_runs/run_001/calib.json
```

### 3. Main Evaluation (M1–M4)

```bash
python eval/evaluate.py \
    --dataset-dir gaze_dataset \
    --calib-path calib_runs/run_001/calib.json \
    --output-dir eval_results/main \
    --camera-calib-path calibration/camera_calib.json \
    --tau 80 --lambda-min 0.2 --lambda-max 1.0 \
    --save-predictions eval_results/gaze_predictions.json
```

### 4. Sensitivity Analysis (Optional)

**τ (decay scale)** — reuses cached gaze predictions:

```bash
for tau in 40 60 120 160; do
    python eval/evaluate.py --dataset-dir gaze_dataset \
        --calib-path calib_runs/run_001/calib.json \
        --output-dir eval_results/tau_${tau} \
        --cached-predictions eval_results/gaze_predictions.json \
        --tau ${tau} --lambda-min 0.2 --lambda-max 1.0
done
```

**λ (fusion range)** — reuses cached gaze predictions:

```bash
# C2: λ ∈ [0.0, 1.0]
python eval/evaluate.py ... --cached-predictions eval_results/gaze_predictions.json \
    --tau 80 --lambda-min 0.0 --lambda-max 1.0 --output-dir eval_results/lambda_C2

# C3: λ ∈ [0.4, 1.0]
python eval/evaluate.py ... --cached-predictions eval_results/gaze_predictions.json \
    --tau 80 --lambda-min 0.4 --lambda-max 1.0 --output-dir eval_results/lambda_C3

# C4: λ ∈ [0.2, 0.8]
python eval/evaluate.py ... --cached-predictions eval_results/gaze_predictions.json \
    --tau 80 --lambda-min 0.2 --lambda-max 0.8 --output-dir eval_results/lambda_C4

# C5: λ ∈ [0.2, 1.2]
python eval/evaluate.py ... --cached-predictions eval_results/gaze_predictions.json \
    --tau 80 --lambda-min 0.2 --lambda-max 1.2 --output-dir eval_results/lambda_C5
```

### 5. Visualization

```bash
python eval/visualize.py \
    --results-csv eval_results/main/results_per_sample.csv \
    --dataset-dir gaze_dataset \
    --output-dir eval_results/visualizations \
    --mode interesting --max-samples 500
```

### 6. LBW External Validation (Optional)

```bash
python lbw_validation/validate_on_lbw.py \
    --lbw-root lbw_dataset \
    --project-root . \
    --calib-path calib_runs/run_001/calib.json \
    --output-dir lbw_results
```

## Evaluation Outputs

Running `eval/evaluate.py` produces:

| File | Content |
|---|---|
| `results_per_sample.csv` | Per-sample raw predictions (all methods) |
| `results_per_class.csv` | Per-class accuracy table (key targets / background / overall) |
| `results_summary.json` | High-level summary with hyperparameters |

### Cityscapes Semantic Weights

The system uses a 19-class Cityscapes taxonomy with driving-optimized semantic weights:

| Priority | Classes | Weight |
|---|---|---|
| **Critical** | person, rider, car, truck, bus, train, motorcycle, bicycle, traffic light, traffic sign | 1.3–1.6 |
| **Context** | road, sidewalk | 0.9–1.0 |
| **Background** | building, wall, fence, pole, vegetation, terrain, sky | 0.6–0.8 |

## Module APIs

### `gazemodel` — Gaze Feature Extraction

```python
from gazemodel import GazeFeatureExtractor

extractor = GazeFeatureExtractor(
    model_cfg_path="gazemodel/config.yaml",
    ckpt_path="gazemodel/weights/unigaze_l16_joint.pth.tar",
)
result = extractor.extract(face_bgr_image)
# result['x']              → (8,) feature vector [pitch, yaw, head_roll, head_pitch, head_yaw, fc_x, fc_y, fc_z]
# result['gaze_cam']       → (3,) gaze direction in camera coordinates
# result['yaw_pitch_cam']  → (2,) pitch & yaw angles
```

### `calibration` — Gaze Prediction on Screen

```python
from calibration import OfflineGazePredictor

predictor = OfflineGazePredictor(calib_path="calib.json")
result = predictor.predict_gaze_point(face_image)
# result['gaze_point'] → (u, v) screen pixel coordinates
```

### `gaze_target_inference` — Target Identification

```python
from gaze_target_inference import predict_all

results = predict_all(gaze_point=(u, v), candidates=candidates, tau=80)
# results['M1']['predicted_id']
# results['M4']['lambda_used'], results['M4']['h_norm']
```

## Important Notes

1. **Model weights** (`unigaze_l16_joint.pth.tar`) and `face_model.txt` must be downloaded separately and placed in `gazemodel/`
2. **Consistent seating** — calibration (step 1) and data collection (step 2) require the same participant posture
3. **Cached predictions** — sensitivity analyses (step 4) reuse the gaze prediction cache from step 3 to avoid redundant model inference
4. **Missing files** — `evaluate_mouse_proxy.py`, `mouse_proxy_collection.py`, `mmsegmentation_image_mapping.py`, and `pspnet_config.py` were lost in .git migration; recover from backup if needed

## License

This project incorporates code from [UniGaze](https://github.com/...), which is distributed under the **CC BY-NC-SA 4.0** license. All derived and modified files in this repository inherit those terms. See individual source files for attribution headers.
