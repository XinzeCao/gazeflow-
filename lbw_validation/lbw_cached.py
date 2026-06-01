"""
lbw_cached.py
-------------
两阶段验证，缓存候选目标，之后调参秒出结果。

阶段1 (--stage build_cache):
    跑 UniGaze 模型 + Mask2Former，把以下内容缓存到磁盘:
      - 每个测试样本的预测注视点 (pred_u, pred_v)
      - 每个场景图的候选目标（mask 用 RLE 压缩存储）
      - 真值目标 ID
    这一步慢（要跑模型和分割），只需做一次。

阶段2 (--stage eval):
    从缓存读取，用指定的权重/τ/λ 跑 M1~M4，秒级出结果。
    可以反复跑，每次换不同参数。

用法:
  # 第一次：构建缓存（慢，~3分钟）
  python lbw_validation/lbw_cached.py --stage build_cache \
      --lbw-root D:/gazeflow/lbw_dataset --project-root D:/gazeflow \
      --calib-path D:/gazeflow/calib_runs/run_004/calib.json \
      --camera-calib-path D:/gazeflow/calibration/camera_calib.json \
      --model-dir D:/gazeflow/mmdetection/checkpoints \
      --cache-dir D:/gazeflow/lbw_results/cache \
      --calib-ratio 0.3

  # 之后：用默认权重评估（秒级）
  python lbw_validation/lbw_cached.py --stage eval \
      --cache-dir D:/gazeflow/lbw_results/cache --tau 34

  # 改 building 权重为 1.0 再评估（秒级）
  python lbw_validation/lbw_cached.py --stage eval \
      --cache-dir D:/gazeflow/lbw_results/cache --tau 34 \
      --weight-override building=1.0
"""

import argparse
import json
import os
import sys
import time
import pickle
import warnings
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict


def setup_project_path(project_root: str):
    root = Path(project_root).resolve()
    for p in [str(root)]:
        if p not in sys.path:
            sys.path.insert(0, p)


# ================================================================== #
#  mask 的 RLE 压缩（节省缓存空间）
# ================================================================== #
def mask_to_rle(mask):
    """bool/uint8 mask (H,W) → RLE dict"""
    m = np.asarray(mask, dtype=np.uint8).reshape(-1)
    # 找变化点
    diffs = np.diff(m)
    change_idx = np.where(diffs != 0)[0] + 1
    starts = np.concatenate([[0], change_idx])
    ends   = np.concatenate([change_idx, [len(m)]])
    lengths = ends - starts
    values  = m[starts]
    return {
        'shape': mask.shape,
        'starts': starts.tolist(),
        'lengths': lengths.tolist(),
        'values': values.tolist(),
    }


def rle_to_mask(rle):
    H, W = rle['shape']
    m = np.zeros(H * W, dtype=np.uint8)
    for s, l, v in zip(rle['starts'], rle['lengths'], rle['values']):
        if v:
            m[s:s+l] = 1
    return m.reshape(H, W)


# ================================================================== #
#  Mask2Former
# ================================================================== #
def load_mask2former(model_dir, device_str='auto'):
    import torch
    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
    device = (torch.device('cuda' if torch.cuda.is_available() else 'cpu')
              if device_str == 'auto' else torch.device(device_str))
    processor = AutoImageProcessor.from_pretrained(model_dir)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_dir).to(device)
    model.eval()
    print(f"[Mask2Former] 加载完成，device={device}")
    return processor, model, device


def run_mask2former(scene_img_path, processor, model, device, min_area=1200):
    import torch
    from PIL import Image as PILImage
    EXCLUDED = {"pole"}
    img_pil = PILImage.open(scene_img_path).convert("RGB")
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
            overlap_mask_area_threshold=0.8)[0]
    seg_map = result["segmentation"].cpu().numpy()
    id2label = {int(k): str(v) for k, v in model.config.id2label.items()}
    candidates = []
    for seg in result["segments_info"]:
        cls = id2label.get(int(seg["label_id"]), f"label_{seg['label_id']}")
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


def get_gt_target_id(gaze_loc_2d, candidates):
    u = int(round(float(gaze_loc_2d[0])))
    v = int(round(float(gaze_loc_2d[1])))
    matched = []
    for obj in candidates:
        mask = obj['mask']
        H, W = mask.shape
        if 0 <= v < H and 0 <= u < W and mask[v, u]:
            matched.append(obj)
    if not matched:
        return None, None
    matched.sort(key=lambda o: int(o['mask'].sum()))
    return matched[0]['target_id'], matched[0]['class_name']


# ================================================================== #
#  阶段1：构建缓存
# ================================================================== #
def build_cache(args):
    setup_project_path(args.project_root)
    from calibration.predictor import OfflineGazePredictor
    sys.path.insert(0, str(Path(__file__).parent))
    from lbw_dataset import LBWDataset
    from gaze_to_scene_calibrator import GazeToSceneCalibrator

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据集
    dataset = LBWDataset(args.lbw_root)
    all_samples = [dataset[i] for i in range(len(dataset))]
    n_calib = max(10, int(len(all_samples) * args.calib_ratio))
    calib_samples = all_samples[:n_calib]
    test_samples  = all_samples[n_calib:]
    if args.max_samples > 0:
        test_samples = test_samples[:args.max_samples]
    print(f"标定样本: {len(calib_samples)}，测试样本: {len(test_samples)}")

    # 加载模型
    predictor = OfflineGazePredictor(
        calib_path=args.calib_path,
        model_cfg_path=args.model_cfg_path,
        ckpt_path=args.ckpt_path,
        camera_calib_path=args.camera_calib_path,
    )
    print("[INFO] OfflineGazePredictor 就绪")

    def predict_dir(sample):
        """跑模型，返回对齐后的视线方向，失败返回 None"""
        face_img = cv2.imread(sample['face_path'])
        if face_img is None:
            return None
        result = predictor.predict_gaze_point(face_img)
        if not result.get('success'):
            return None
        g = np.asarray(result['raw_features']['gaze_cam'], dtype=np.float32).reshape(-1)
        n = np.linalg.norm(g)
        if n > 1e-6:
            g = g / n
        return -g  # 坐标系对齐

    # 标定集预测视线
    print(f"\n[INFO] 标定集预测视线...")
    calib_dirs = []
    for i, s in enumerate(calib_samples):
        calib_dirs.append(predict_dir(s))
        if (i+1) % 100 == 0:
            print(f"  {i+1}/{len(calib_samples)}")

    # 训练 calibrator
    calibrator = GazeToSceneCalibrator(alpha=0.05, degree=2)
    calibrator.fit_with_predicted_gaze(calib_samples, calib_dirs, verbose=True)

    # 加载分割模型
    proc, seg_model, device = load_mask2former(args.model_dir)

    # 逐测试样本：预测注视点 + 分割 + 真值目标
    print(f"\n[INFO] 处理测试样本并缓存...")
    seg_cache_rle = {}     # scene_path → [候选目标(RLE)]
    sample_records = []
    n_fail = 0
    t0 = time.time()

    for i, s in enumerate(test_samples):
        pred_dir = predict_dir(s)
        if pred_dir is None:
            n_fail += 1
            continue
        # 用 calibrator 映射
        pseudo = {
            'right_eye_loc3d': s['right_eye_loc3d'],
            'left_eye_loc3d' : s['left_eye_loc3d'],
            'right_gaze_dir' : pred_dir,
            'left_gaze_dir'  : pred_dir,
            'scene_size'     : s['scene_size'],
        }
        pred_u, pred_v = calibrator.predict(pseudo)

        sp = s['scene_path']
        if sp not in seg_cache_rle:
            try:
                cands = run_mask2former(sp, proc, seg_model, device)
            except Exception as e:
                print(f"  [WARN] 分割失败 {s['frame_id']}: {e}")
                cands = []
            # 转 RLE 存储
            seg_cache_rle[sp] = [{
                'target_id': c['target_id'],
                'class_name': c['class_name'],
                'centroid': c['centroid'],
                'mask_rle': mask_to_rle(c['mask']),
            } for c in cands]

        # 真值目标（需要 mask，临时解 RLE）
        cands_full = [{
            'target_id': c['target_id'],
            'class_name': c['class_name'],
            'centroid': c['centroid'],
            'mask': rle_to_mask(c['mask_rle']),
        } for c in seg_cache_rle[sp]]
        gt_id, gt_class = get_gt_target_id(s['gaze_loc_2d'], cands_full)

        sample_records.append({
            'frame_id': s['frame_id'],
            'scene_path': sp,
            'pred_u': float(pred_u), 'pred_v': float(pred_v),
            'gt_u': float(s['gaze_loc_2d'][0]), 'gt_v': float(s['gaze_loc_2d'][1]),
            'gt_target_id': gt_id, 'gt_class': gt_class,
        })

        if (i+1) % 100 == 0:
            print(f"  {i+1}/{len(test_samples)}, 失败={n_fail}, "
                  f"场景缓存={len(seg_cache_rle)}, 耗时={time.time()-t0:.0f}s")

    # 保存缓存
    with open(cache_dir / 'samples.pkl', 'wb') as f:
        pickle.dump(sample_records, f)
    with open(cache_dir / 'segments.pkl', 'wb') as f:
        pickle.dump(seg_cache_rle, f)

    gaze_errs = [np.sqrt((r['pred_u']-r['gt_u'])**2 + (r['pred_v']-r['gt_v'])**2)
                 for r in sample_records]
    print(f"\n[INFO] 缓存完成，保存到 {cache_dir}")
    print(f"  有效样本: {len(sample_records)}, 失败: {n_fail}")
    print(f"  注视点误差: mean={np.mean(gaze_errs):.1f}px, median={np.median(gaze_errs):.1f}px")
    print(f"\n现在可以用 --stage eval 反复调参，无需重跑模型。")


# ================================================================== #
#  阶段2：从缓存评估（秒级）
# ================================================================== #
def evaluate(args):
    sys.path.insert(0, str(Path(__file__).parent))
    setup_project_path(args.project_root) if args.project_root else None

    cache_dir = Path(args.cache_dir)
    with open(cache_dir / 'samples.pkl', 'rb') as f:
        sample_records = pickle.load(f)
    with open(cache_dir / 'segments.pkl', 'rb') as f:
        seg_cache_rle = pickle.load(f)
    print(f"[INFO] 加载缓存: {len(sample_records)} 样本, {len(seg_cache_rle)} 场景")

    # 默认权重（你论文 Table I）
    WEIGHTS = {
        'person': 1.5, 'car': 1.4, 'bicycle': 1.3,
        'traffic sign': 1.3, 'traffic light': 1.3,
        'road': 1.0, 'sidewalk': 0.9, 'vegetation': 0.8,
        'building': 0.7, 'sky': 0.5,
        'truck': 1.4, 'bus': 1.4, 'train': 1.4,
        'motorcycle': 1.3, 'rider': 1.4,
        'terrain': 0.8, 'wall': 0.7, 'fence': 0.7,
    }
    # 应用权重覆盖
    if args.weight_override:
        for kv in args.weight_override:
            k, v = kv.rsplit('=', 1)
            WEIGHTS[k.strip()] = float(v)
            print(f"  [权重覆盖] {k.strip()} = {float(v)}")

    tau = args.tau
    lambda_min, lambda_max = args.lambda_min, args.lambda_max
    CUTOFF = args.truncation_factor * tau

    def infer(gaze_point, candidates, method):
        """method ∈ {M1,M2,M3,M4}"""
        if not candidates:
            return None
        u, v = gaze_point

        if method == 'M1':
            # 纯质心距离
            dists = [np.linalg.norm(np.array(c['centroid']) - np.array([u, v]))
                     for c in candidates]
            return candidates[int(np.argmin(dists))]['target_id']

        # M2/M3/M4 用边缘距离
        geo_scores, valid = [], []
        for c in candidates:
            mask = c['mask']
            dist_map = cv2.distanceTransform((1-mask).astype(np.uint8), cv2.DIST_L2, 5)
            vi = int(np.clip(v, 0, mask.shape[0]-1))
            ui = int(np.clip(u, 0, mask.shape[1]-1))
            d = float(dist_map[vi, ui])
            if d <= CUTOFF:
                geo_scores.append(np.exp(-d / tau))
                valid.append(c)
        if not valid:
            dists = [np.linalg.norm(np.array(c['centroid']) - np.array([u, v]))
                     for c in candidates]
            return candidates[int(np.argmin(dists))]['target_id']

        geo = np.array(geo_scores)
        geo_prob = geo / (geo.sum() + 1e-12)

        if method == 'M2':
            return valid[int(np.argmax(geo_prob))]['target_id']

        # 语义权重
        sem = np.array([WEIGHTS.get(c['class_name'], 1.0) for c in valid])

        if method == 'M3':
            lam = lambda_max  # 固定
        else:  # M4 自适应
            n_eff = len(valid)
            h_norm = (-np.sum(geo_prob*np.log(geo_prob+1e-12))/np.log(n_eff)
                      if n_eff > 1 else 0.0)
            lam = lambda_min + (lambda_max - lambda_min) * h_norm

        final = geo_prob * (sem ** lam)
        return valid[int(np.argmax(final))]['target_id']

    # 逐样本评估
    methods = ['M1', 'M2', 'M3', 'M4']
    agg = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    n_no_gt = 0

    for r in sample_records:
        gt_id, gt_class = r['gt_target_id'], r['gt_class']
        if gt_id is None or gt_class is None:
            n_no_gt += 1
            continue
        # 解 RLE 得到候选目标
        cands = [{
            'target_id': c['target_id'],
            'class_name': c['class_name'],
            'centroid': c['centroid'],
            'mask': rle_to_mask(c['mask_rle']),
        } for c in seg_cache_rle[r['scene_path']]]

        for m in methods:
            pred_id = infer((r['pred_u'], r['pred_v']), cands, m)
            agg[m][gt_class][1] += 1
            agg[m][gt_class][0] += int(pred_id == gt_id)

    # 输出
    KEY = {"car","truck","bus","person","rider","motorcycle",
           "bicycle","traffic light","traffic sign","train"}
    all_classes = sorted(set(k for m in methods for k in agg[m]))
    print(f"\n===== 注视目标识别准确率（τ={tau}px）=====")
    print(f"  {'类别':<16}{'N':>5}", end="")
    for m in methods:
        print(f"  {m:>8}", end="")
    print()
    for cls in all_classes:
        n = agg[methods[0]][cls][1]
        print(f"  {cls:<16}{n:>5}", end="")
        for m in methods:
            h, t = agg[m][cls]
            print(f"  {h/t*100 if t else 0:>7.1f}%", end="")
        print()
    print()
    for grp, fn in [("关键目标", lambda c: c in KEY),
                    ("背景目标", lambda c: c not in KEY),
                    ("总体",     lambda c: True)]:
        tot = sum(agg[methods[0]][k][1] for k in all_classes if fn(k))
        print(f"  {grp:<16}{tot:>5}", end="")
        for m in methods:
            h = sum(agg[m][k][0] for k in all_classes if fn(k))
            t = sum(agg[m][k][1] for k in all_classes if fn(k))
            print(f"  {h/t*100 if t else 0:>7.1f}%", end="")
        print()


# ================================================================== #
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['build_cache', 'eval'], required=True)
    parser.add_argument('--cache-dir', required=True)
    # build_cache 需要的
    parser.add_argument('--lbw-root')
    parser.add_argument('--project-root')
    parser.add_argument('--calib-path')
    parser.add_argument('--camera-calib-path', default='camera_calib.json')
    parser.add_argument('--model-cfg-path', default=None)
    parser.add_argument('--ckpt-path', default=None)
    parser.add_argument('--model-dir')
    parser.add_argument('--calib-ratio', type=float, default=0.3)
    parser.add_argument('--max-samples', type=int, default=0)
    # eval 需要的
    parser.add_argument('--tau', type=float, default=34.0)
    parser.add_argument('--lambda-min', type=float, default=0.2)
    parser.add_argument('--lambda-max', type=float, default=1.0)
    parser.add_argument('--truncation-factor', type=float, default=5.0)
    parser.add_argument('--weight-override', nargs='*', default=None,
                        help='覆盖语义权重，格式 building=1.0 vegetation=1.0')
    args = parser.parse_args()

    if args.stage == 'build_cache':
        for req in ['lbw_root', 'project_root', 'calib_path', 'model_dir']:
            if getattr(args, req) is None:
                parser.error(f"build_cache 需要 --{req.replace('_','-')}")
        build_cache(args)
    else:
        evaluate(args)