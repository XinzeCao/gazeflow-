"""
lbw_visualize.py
----------------
从缓存中挑出错误样本，可视化保存:
  - 原始场景图
  - 全景分割掩码叠加（不同颜色）
  - 真值注视点（绿色十字）+ 它落在的真值目标（绿色边框）
  - 预测注视点（红色十字）+ M4 预测的目标（红色边框）

用法:
  python lbw_validation/lbw_visualize.py \
      --cache-dir D:/gazeflow/lbw_results/cache \
      --output-dir D:/gazeflow/lbw_results/vis \
      --tau 34 \
      --n 20 \
      --filter wrong          # wrong=只看错的, all=全部, class:building=只看某类
"""

import argparse
import pickle
import sys
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict


def rle_to_mask(rle):
    H, W = rle['shape']
    m = np.zeros(H * W, dtype=np.uint8)
    for s, l, v in zip(rle['starts'], rle['lengths'], rle['values']):
        if v:
            m[s:s+l] = 1
    return m.reshape(H, W)


# 固定调色板（每个类别一个颜色，BGR）
PALETTE = {
    'road': (128, 64, 128), 'sidewalk': (232, 35, 244), 'building': (70, 70, 70),
    'wall': (156, 102, 102), 'fence': (153, 153, 190), 'pole': (153, 153, 153),
    'traffic light': (30, 170, 250), 'traffic sign': (0, 220, 220),
    'vegetation': (35, 142, 107), 'terrain': (152, 251, 152), 'sky': (180, 130, 70),
    'person': (60, 20, 220), 'rider': (0, 0, 255), 'car': (142, 0, 0),
    'truck': (70, 0, 0), 'bus': (100, 60, 0), 'train': (100, 80, 0),
    'motorcycle': (230, 0, 0), 'bicycle': (32, 11, 119),
}


def color_for(cls):
    return PALETTE.get(cls, (200, 200, 200))


def infer_m4(gaze_point, candidates, tau, lambda_min, lambda_max,
             cutoff_factor, weights):
    """复制 lbw_cached.py 的 M4 逻辑，返回 (预测target_id, 有效候选数)"""
    if not candidates:
        return None, 0
    u, v = gaze_point
    geo_scores, valid = [], []
    for c in candidates:
        mask = c['mask']
        dist_map = cv2.distanceTransform((1-mask).astype(np.uint8), cv2.DIST_L2, 5)
        vi = int(np.clip(v, 0, mask.shape[0]-1))
        ui = int(np.clip(u, 0, mask.shape[1]-1))
        d = float(dist_map[vi, ui])
        if d <= cutoff_factor * tau:
            geo_scores.append(np.exp(-d / tau))
            valid.append(c)
    if not valid:
        dists = [np.linalg.norm(np.array(c['centroid'])-np.array([u,v])) for c in candidates]
        return candidates[int(np.argmin(dists))]['target_id'], 0
    geo = np.array(geo_scores)
    geo_prob = geo / (geo.sum() + 1e-12)
    sem = np.array([weights.get(c['class_name'], 1.0) for c in valid])
    n_eff = len(valid)
    h_norm = (-np.sum(geo_prob*np.log(geo_prob+1e-12))/np.log(n_eff)
              if n_eff > 1 else 0.0)
    lam = lambda_min + (lambda_max - lambda_min) * h_norm
    final = geo_prob * (sem ** lam)
    return valid[int(np.argmax(final))]['target_id'], n_eff


def draw_cross(img, x, y, color, size=12, thick=2):
    x, y = int(x), int(y)
    cv2.line(img, (x-size, y), (x+size, y), color, thick)
    cv2.line(img, (x, y-size), (x, y+size), color, thick)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--tau', type=float, default=34.0)
    parser.add_argument('--lambda-min', type=float, default=0.2)
    parser.add_argument('--lambda-max', type=float, default=1.0)
    parser.add_argument('--truncation-factor', type=float, default=5.0)
    parser.add_argument('--n', type=int, default=20, help='保存多少个样本')
    parser.add_argument('--filter', default='wrong',
                        help='wrong=只看M4错的, all=全部, class:building=只看某真值类')
    parser.add_argument('--weight-override', nargs='*', default=None)
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(cache_dir / 'samples.pkl', 'rb') as f:
        samples = pickle.load(f)
    with open(cache_dir / 'segments.pkl', 'rb') as f:
        seg_rle = pickle.load(f)

    WEIGHTS = {
        'person': 1.5, 'car': 1.4, 'bicycle': 1.3, 'traffic sign': 1.3,
        'traffic light': 1.3, 'road': 1.0, 'sidewalk': 0.9, 'vegetation': 0.8,
        'building': 0.7, 'sky': 0.5, 'truck': 1.4, 'bus': 1.4, 'train': 1.4,
        'motorcycle': 1.3, 'rider': 1.4, 'terrain': 0.8, 'wall': 0.7, 'fence': 0.7,
    }
    if args.weight_override:
        for kv in args.weight_override:
            k, v = kv.rsplit('=', 1)
            WEIGHTS[k.strip()] = float(v)

    # 筛选样本
    selected = []
    for r in samples:
        gt_id, gt_class = r['gt_target_id'], r['gt_class']
        if gt_id is None or gt_class is None:
            continue
        cands = [{
            'target_id': c['target_id'], 'class_name': c['class_name'],
            'centroid': c['centroid'], 'mask': rle_to_mask(c['mask_rle']),
        } for c in seg_rle[r['scene_path']]]
        pred_id, n_eff = infer_m4(
            (r['pred_u'], r['pred_v']), cands, args.tau,
            args.lambda_min, args.lambda_max, args.truncation_factor, WEIGHTS)
        correct = (pred_id == gt_id)

        keep = False
        if args.filter == 'wrong' and not correct:
            keep = True
        elif args.filter == 'all':
            keep = True
        elif args.filter.startswith('class:') and gt_class == args.filter.split(':')[1]:
            keep = True

        if keep:
            pred_class = next((c['class_name'] for c in cands if c['target_id']==pred_id), '?')
            selected.append((r, cands, gt_id, gt_class, pred_id, pred_class, n_eff))

    print(f"[INFO] 筛选出 {len(selected)} 个样本，保存前 {min(args.n, len(selected))} 个")

    for idx, (r, cands, gt_id, gt_class, pred_id, pred_class, n_eff) in enumerate(selected[:args.n]):
        scene = cv2.imread(r['scene_path'])
        if scene is None:
            print(f"  读图失败: {r['scene_path']}")
            continue
        H, W = scene.shape[:2]

        # 半透明分割叠加
        overlay = scene.copy()
        for c in cands:
            mask = c['mask'].astype(bool)
            overlay[mask] = color_for(c['class_name'])
        vis = cv2.addWeighted(scene, 0.5, overlay, 0.5, 0)

        # 真值目标边框（绿）
        for c in cands:
            if c['target_id'] == gt_id:
                contours, _ = cv2.findContours(c['mask'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)
        # 预测目标边框（红）
        for c in cands:
            if c['target_id'] == pred_id:
                contours, _ = cv2.findContours(c['mask'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis, contours, -1, (0, 0, 255), 2)

        # 注视点
        draw_cross(vis, r['gt_u'], r['gt_v'], (0, 255, 0), size=14, thick=2)   # 真值绿
        draw_cross(vis, r['pred_u'], r['pred_v'], (0, 0, 255), size=14, thick=2)  # 预测红
        # 连线
        cv2.line(vis, (int(r['gt_u']), int(r['gt_v'])),
                 (int(r['pred_u']), int(r['pred_v'])), (255, 255, 0), 1)

        # 文字信息
        err = np.sqrt((r['pred_u']-r['gt_u'])**2 + (r['pred_v']-r['gt_v'])**2)
        info = [
            f"frame={r['frame_id']}  err={err:.0f}px  n_eff={n_eff}",
            f"GT(green): {gt_class}",
            f"Pred(red): {pred_class}",
        ]
        y0 = 20
        for line in info:
            cv2.putText(vis, line, (8, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3)
            cv2.putText(vis, line, (8, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            y0 += 20

        fn = f"{idx:03d}_{r['frame_id']}_GT-{gt_class}_PRED-{pred_class}.png"
        cv2.imwrite(str(out_dir / fn), vis)

    print(f"[INFO] 已保存到 {out_dir}")
    print("图例: 绿=真值(点/目标边框)  红=预测(点/目标边框)  黄线=偏移  n_eff=有效候选数")


if __name__ == '__main__':
    main()
