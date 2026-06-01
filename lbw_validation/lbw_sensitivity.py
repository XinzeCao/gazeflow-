"""
lbw_sensitivity.py
------------------
从上一次跑完保存的 lbw_per_sample.csv 出发，
不重新跑 Mask2Former，直接对不同超参数做敏感性分析。

需要先修改 lbw_adapter.py，在保存 results_per_sample 时额外把
候选目标掩码路径也存进去——但那样会很大。

所以这里换一个思路：直接读 lbw_results/lbw_per_sample.csv，
对 M2 的"正确了但 M3/M4 错了"的样本做分析，
同时对 lambda_max 和权重压缩系数做网格搜索。

用法：
    python lbw_sensitivity.py \
        --csv-path   D:/gazeflow/lbw_results/lbw_per_sample.csv \
        --project-root D:/gazeflow \
        --lbw-root   D:/gazeflow/lbw_dataset \
        --model-dir  D:/gazeflow/mmdetection/checkpoints \
        --output-dir D:/gazeflow/lbw_results/sensitivity
"""

import argparse
import json
import sys
import os
import csv
import numpy as np
import cv2
import warnings
from pathlib import Path
from collections import defaultdict

# ------------------------------------------------------------------ #
#  把原项目路径加进来
# ------------------------------------------------------------------ #
def setup_path(project_root):
    for p in [project_root]:
        if p not in sys.path:
            sys.path.insert(0, p)


# ------------------------------------------------------------------ #
#  复用原项目的核心推理逻辑，但允许覆盖超参数和语义权重
# ------------------------------------------------------------------ #
def identify_target_custom(
    gaze_point,
    candidates,
    tau=80.0,
    lambda_min=0.2,
    lambda_max=1.0,
    semantic_weights=None,    # dict: class_name -> weight
    weight_compress=1.0,      # 把权重压缩到 [1 - (w-1)*compress, 1 + (w-1)*compress] 附近
):
    """
    和原项目 M4 逻辑完全一致，但支持自定义 lambda_max 和权重压缩。
    weight_compress=1.0 → 用原始权重
    weight_compress=0.5 → 权重差距缩小一半
    weight_compress=0.0 → 所有权重变成 1.0（等于纯几何，即 M2）
    """
    if not candidates:
        return None

    # 默认权重（和你论文 Table I 一致）
    DEFAULT_WEIGHTS = {
        'person': 1.5, 'car': 1.4, 'bicycle': 1.3,
        'traffic sign': 1.3, 'traffic light': 1.3,
        'road': 1.0, 'sidewalk': 0.9, 'vegetation': 0.8,
        'building': 0.7, 'sky': 0.5,
        # 补充其他类别
        'truck': 1.4, 'bus': 1.4, 'train': 1.4,
        'motorcycle': 1.3, 'rider': 1.4,
        'terrain': 0.8, 'wall': 0.7, 'fence': 0.7,
    }
    if semantic_weights is not None:
        DEFAULT_WEIGHTS.update(semantic_weights)

    # 压缩权重：w_new = 1 + (w_orig - 1) * compress
    def get_weight(class_name):
        w = DEFAULT_WEIGHTS.get(class_name, 1.0)
        return 1.0 + (w - 1.0) * weight_compress

    u, v = gaze_point
    CUTOFF = 5.0 * tau

    geo_scores = []
    valid_cands = []
    for obj in candidates:
        mask = obj['mask']
        dist_map = cv2.distanceTransform(
            (1 - mask).astype(np.uint8), cv2.DIST_L2, 5)
        vi = int(np.clip(v, 0, mask.shape[0] - 1))
        ui = int(np.clip(u, 0, mask.shape[1] - 1))
        d_edge = float(dist_map[vi, ui])
        if d_edge <= CUTOFF:
            geo_scores.append(np.exp(-d_edge / tau))
            valid_cands.append(obj)

    if not valid_cands:
        dists = [np.linalg.norm(
            np.array(o['centroid']) - np.array([u, v])) for o in candidates]
        return candidates[int(np.argmin(dists))]['target_id']

    geo_scores = np.array(geo_scores)
    geo_prob   = geo_scores / (geo_scores.sum() + 1e-12)

    n_eff = len(valid_cands)
    h_norm = (-np.sum(geo_prob * np.log(geo_prob + 1e-12)) / np.log(n_eff)
              if n_eff > 1 else 0.0)
    lam = lambda_min + (lambda_max - lambda_min) * h_norm

    sem_w = np.array([get_weight(o['class_name']) for o in valid_cands])
    final = geo_prob * (sem_w ** lam)
    return valid_cands[int(np.argmax(final))]['target_id']


# ------------------------------------------------------------------ #
#  从 LBW 数据集 + Mask2Former 重建候选目标（带缓存）
# ------------------------------------------------------------------ #
def load_mask2former(model_dir, device_str='auto'):
    import torch
    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
    device = (torch.device('cuda' if torch.cuda.is_available() else 'cpu')
              if device_str == 'auto' else torch.device(device_str))
    processor = AutoImageProcessor.from_pretrained(model_dir)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_dir).to(device)
    model.eval()
    return processor, model, device


def segment_scene(scene_path, processor, model, device, min_area=1200):
    import torch
    from PIL import Image as PILImage
    EXCLUDED = {"pole"}

    img_pil = PILImage.open(scene_path).convert("RGB")
    W, H = img_pil.size
    inputs = {k: v.to(device)
              for k, v in processor(images=img_pil, return_tensors="pt").items()}
    with torch.no_grad():
        outputs = model(**inputs)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*label_ids_to_fuse.*")
        result = processor.post_process_panoptic_segmentation(
            outputs, target_sizes=[(H, W)],
            threshold=0.5, mask_threshold=0.5,
            overlap_mask_area_threshold=0.8,
        )[0]

    seg_map = result["segmentation"].cpu().numpy()
    id2label = {int(k): str(v) for k, v in model.config.id2label.items()}
    candidates = []
    for seg in result["segments_info"]:
        cls = id2label.get(int(seg["label_id"]), "unknown")
        if cls in EXCLUDED:
            continue
        mask = (seg_map == int(seg["id"])).astype(np.uint8)
        if mask.sum() < min_area:
            continue
        ys, xs = np.where(mask)
        candidates.append({
            "target_id": str(seg["id"]),
            "class_name": cls,
            "mask": mask,
            "centroid": [float(xs.mean()), float(ys.mean())],
        })
    return candidates


def get_gt_id(gaze_loc_2d, candidates):
    u, v = int(round(float(gaze_loc_2d[0]))), int(round(float(gaze_loc_2d[1])))
    matched = [o for o in candidates
               if 0 <= v < o['mask'].shape[0]
               and 0 <= u < o['mask'].shape[1]
               and o['mask'][v, u]]
    if not matched:
        return None
    matched.sort(key=lambda o: int(o['mask'].sum()))
    return matched[0]['target_id']


# ------------------------------------------------------------------ #
#  主流程
# ------------------------------------------------------------------ #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lbw-root',    required=True)
    parser.add_argument('--model-dir',   required=True)
    parser.add_argument('--project-root', required=True)
    parser.add_argument('--output-dir',  default='./lbw_sensitivity')
    parser.add_argument('--calib-ratio', type=float, default=0.3)
    args = parser.parse_args()

    setup_path(args.project_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据集
    sys.path.insert(0, str(Path(__file__).parent))
    from lbw_dataset import LBWDataset
    from gaze_to_scene_calibrator import GazeToSceneCalibrator

    dataset = LBWDataset(args.lbw_root)
    all_s   = [dataset[i] for i in range(len(dataset))]
    n_calib = max(10, int(len(all_s) * args.calib_ratio))
    calib_s, test_s = all_s[:n_calib], all_s[n_calib:]

    calib = GazeToSceneCalibrator()
    calib.fit(calib_s, verbose=False)

    # 加载 Mask2Former
    proc, seg_model, device = load_mask2former(args.model_dir)
    print(f"[INFO] Mask2Former 加载完成，开始分割 {len(test_s)} 个测试样本...")

    # 分割 + 收集有效样本（带缓存）
    seg_cache = {}
    valid_samples = []   # list of (pred_uv, gaze_loc_2d, candidates, gt_id, gt_class)

    for i, s in enumerate(test_s):
        pred_u, pred_v = calib.predict(s)
        sp = s['scene_path']
        if sp not in seg_cache:
            try:
                seg_cache[sp] = segment_scene(sp, proc, seg_model, device)
            except Exception as e:
                seg_cache[sp] = []
            if (i + 1) % 50 == 0:
                print(f"  已处理 {i+1}/{len(test_s)}，缓存场景数={len(seg_cache)}")
        cands = seg_cache[sp]
        if not cands:
            continue
        gt_id = get_gt_id(s['gaze_loc_2d'], cands)
        gt_cls = next((c['class_name'] for c in cands
                       if c['target_id'] == gt_id), None) if gt_id else None
        valid_samples.append((pred_u, pred_v, cands, gt_id, gt_cls))

    print(f"[INFO] 有效样本: {len(valid_samples)}")

    # ------------------------------------------------------------------ #
    #  三维网格搜索：tau x lambda_max x weight_compress
    # ------------------------------------------------------------------ #
    # tau: 原始80px对应2194px宽屏幕，LBW是942px，按比例约34px，搜索20~80
    tau_values           = [20, 34, 50, 65, 80]           # px，原始是80
    lambda_max_values    = [0.3, 0.5, 0.7, 1.0]           # 原始是1.0
    weight_compress_vals = [0.0, 0.25, 0.5, 0.75, 1.0]    # 0=纯几何, 1=原始权重

    KEY = {"car","truck","bus","person","rider","motorcycle",
           "bicycle","traffic light","traffic sign","train"}

    grid_results = []
    total_configs = len(tau_values) * len(lambda_max_values) * len(weight_compress_vals)
    print(f"\n[INFO] 开始三维网格搜索，共 {total_configs} 个配置...\n")

    for tau in tau_values:
        print(f"\n===== tau = {tau}px =====")
        print(f"{'lambda_max':>12} {'compress':>10} {'总体':>10} {'road':>10} {'关键目标':>10}")
        print("-" * 56)
        for lam_max in lambda_max_values:
            for compress in weight_compress_vals:
                hits_all, total_all = 0, 0
                hits_road, total_road = 0, 0
                hits_key, total_key = 0, 0

                for (pu, pv, cands, gt_id, gt_cls) in valid_samples:
                    if gt_id is None or gt_cls is None:
                        continue
                    pred_id = identify_target_custom(
                        (pu, pv), cands,
                        tau=tau, lambda_min=0.2,
                        lambda_max=lam_max,
                        weight_compress=compress,
                    )
                    correct = int(pred_id == gt_id)
                    hits_all  += correct; total_all  += 1
                    if gt_cls == "road":
                        hits_road += correct; total_road += 1
                    if gt_cls in KEY:
                        hits_key  += correct; total_key  += 1

                acc_all  = hits_all  / total_all  * 100 if total_all  else 0
                acc_road = hits_road / total_road * 100 if total_road else 0
                acc_key  = hits_key  / total_key  * 100 if total_key  else 0

                print(f"{lam_max:>12.1f} {compress:>10.2f} "
                      f"{acc_all:>9.1f}% {acc_road:>9.1f}% {acc_key:>9.1f}%")
                grid_results.append({
                    "tau": tau,
                    "lambda_max": lam_max,
                    "weight_compress": compress,
                    "acc_overall": round(acc_all, 2),
                    "acc_road":    round(acc_road, 2),
                    "acc_key":     round(acc_key, 2),
                    "n_total":     total_all,
                })

    # 保存
    with open(out_dir / "sensitivity_grid_3d.json", "w", encoding="utf-8") as f:
        json.dump(grid_results, f, indent=2)
    print(f"\n结果已保存到 {out_dir / 'sensitivity_grid_3d.json'}")

    # 找最优配置
    best = max(grid_results, key=lambda r: r["acc_overall"])
    print(f"\n最优配置: tau={best['tau']}px, lambda_max={best['lambda_max']}, "
          f"compress={best['weight_compress']} -> 总体准确率={best['acc_overall']}%")

    # 原始配置作为对比基准
    orig = next((r for r in grid_results
                 if r["tau"]==80 and r["lambda_max"]==1.0 and r["weight_compress"]==1.0), None)
    if orig:
        print(f"原始配置(tau=80, lam=1.0, comp=1.0): 总体={orig['acc_overall']}%, "
              f"road={orig['acc_road']}%, 关键={orig['acc_key']}%")



if __name__ == '__main__':
    main()