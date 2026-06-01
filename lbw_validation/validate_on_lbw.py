"""
validate_on_lbw.py
------------------
在 LookBothWays 数据集上运行端到端验证。

两种验证模式（可通过命令行参数选择）：
  --mode gt_calib   用 LBW 真值视线方向训练映射，只验证【场景匹配模块】
  --mode model      用你自己的视线估计模型推理，做【端到端】验证

使用方法:
  python validate_on_lbw.py \
      --lbw_root /path/to/look_both_ways_dataset \
      --model_ckpt /path/to/your_gaze_model.pth \
      --mode gt_calib \
      --calib_ratio 0.3 \
      --output_dir ./lbw_results
"""

import argparse
import json
import os
import sys
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import cv2

# -------- 把你项目的路径加进来 --------
# 根据实际情况修改这两行，保证能 import 你的模块
# sys.path.insert(0, '/path/to/your_project')
# from your_project.gaze_model import GazeModel
# from your_project.panoptic import PanopticSegmentor
# from your_project.gaze_target import GazeTargetIdentifier
# ---------------------------------------

from lbw_dataset import LBWDataset
from gaze_to_scene_calibrator import GazeToSceneCalibrator


# ================================================================== #
#  Step 1: 全景分割（复用你项目中已有的 Mask2Former 模块）
#  下面是一个接口占位，替换成你的实际调用即可
# ================================================================== #
def run_panoptic_segmentation(scene_img_path: str, segmentor):
    """
    输入: 场景图路径
    输出: candidate_objects 列表，每个元素是字典:
        {
          'id'    : int,          # 唯一实例ID
          'class' : str,          # 语义类别名
          'mask'  : np.ndarray,   # bool mask，shape=(H, W)
          'center': np.ndarray,   # [cx, cy] 质心像素坐标
          'area'  : int,          # 掩码面积（像素数）
          'bbox'  : list,         # [x1, y1, x2, y2]
        }
    """
    # ---- 替换为你的实际调用 ----
    # img = cv2.imread(scene_img_path)
    # candidates = segmentor.predict(img)
    # return candidates
    raise NotImplementedError("请替换为你项目的全景分割调用")


# ================================================================== #
#  Step 2: 注视目标识别（复用你项目中已有的 GazeTargetIdentifier）
#  下面是一个自包含的简化实现，和你论文 3.4 节完全一致
# ================================================================== #

# Cityscapes 类别语义权重（和你论文 Table 3.1 一致，按需调整）
SEMANTIC_WEIGHTS = {
    'car': 3.0, 'truck': 3.0, 'bus': 3.0, 'motorcycle': 2.5,
    'bicycle': 2.5, 'person': 3.5, 'rider': 3.0,
    'traffic light': 3.0, 'traffic sign': 2.5,
    'road': 1.0, 'sidewalk': 1.0, 'building': 0.5,
    'wall': 0.5, 'fence': 0.5, 'vegetation': 0.5,
    'terrain': 0.5, 'sky': 0.3,
}

def identify_gaze_target(
    gaze_point: tuple,
    candidates: list,
    tau: float = 80.0,
    lambda_min: float = 0.2,
    lambda_max: float = 1.0,
    cutoff_factor: float = 5.0,
):
    """
    融合空间几何度量与语义先验（论文 3.4 节）。
    gaze_point : (u_px, v_px)
    candidates : 全景分割输出的候选目标列表
    返回: 预测注视目标的 id，或 None
    """
    if not candidates:
        return None

    u, v = gaze_point

    # --- 3.4.1 边缘距离 → 几何得分 ---
    geo_scores = []
    valid_cands = []
    for obj in candidates:
        mask = obj['mask'].astype(np.uint8)
        # 边缘距离：落在掩码内部为 0，外部为最近轮廓距离
        dist_map = cv2.distanceTransform(1 - mask, cv2.DIST_L2, 5)
        d_edge = float(dist_map[int(np.clip(v, 0, mask.shape[0]-1)),
                                int(np.clip(u, 0, mask.shape[1]-1))])
        if d_edge <= cutoff_factor * tau:
            score = np.exp(-d_edge / tau)
            geo_scores.append(score)
            valid_cands.append(obj)

    if not valid_cands:
        # 兜底：返回质心距离最近的目标
        dists = [np.linalg.norm(np.array(o['center']) - np.array([u, v]))
                 for o in candidates]
        return candidates[int(np.argmin(dists))]['id']

    geo_scores = np.array(geo_scores, dtype=np.float64)
    geo_prob   = geo_scores / (geo_scores.sum() + 1e-12)

    # --- 3.4.3 归一化熵 → 自适应 λ ---
    n_eff = len(valid_cands)
    if n_eff > 1:
        entropy  = -np.sum(geo_prob * np.log(geo_prob + 1e-12))
        h_norm   = entropy / np.log(n_eff)
    else:
        h_norm = 0.0
    lam = lambda_min + (lambda_max - lambda_min) * h_norm

    # --- 3.4.2 语义权重 ---
    sem_weights = np.array([
        SEMANTIC_WEIGHTS.get(o['class'], 1.0) for o in valid_cands
    ], dtype=np.float64)

    # --- 最终得分 ---
    final_scores = geo_prob * (sem_weights ** lam)
    best_idx     = int(np.argmax(final_scores))
    return valid_cands[best_idx]['id']


# ================================================================== #
#  Step 3: 真值目标 ID（把 Gaze_Loc_2D 打到分割结果里）
# ================================================================== #
def get_gt_target_id(gaze_loc_2d: np.ndarray, candidates: list):
    """
    用真值 2D 注视点查询它落在哪个实例的掩码里，作为真值目标 ID。
    若落在多个掩码的重叠区，取面积最小的（更精确的前景目标）。
    """
    u, v = int(round(gaze_loc_2d[0])), int(round(gaze_loc_2d[1]))
    matched = []
    for obj in candidates:
        mask = obj['mask']
        H, W = mask.shape
        if 0 <= v < H and 0 <= u < W and mask[v, u]:
            matched.append(obj)
    if not matched:
        return None
    # 取面积最小者（优先前景小目标）
    matched.sort(key=lambda o: o['area'])
    return matched[0]['id']


# ================================================================== #
#  主流程
# ================================================================== #
def run_validation(args):
    # 1. 加载数据集
    dataset = LBWDataset(args.lbw_root)
    all_samples = [dataset[i] for i in range(len(dataset))]

    n_calib = max(10, int(len(all_samples) * args.calib_ratio))
    calib_samples = all_samples[:n_calib]
    test_samples  = all_samples[n_calib:]
    print(f"Calibration: {len(calib_samples)} samples, Test: {len(test_samples)} samples")

    # 2. 训练 gaze→scene 映射
    calibrator = GazeToSceneCalibrator(alpha=0.05, degree=2)
    calibrator.fit(calib_samples, verbose=True)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        calibrator.save(str(Path(args.output_dir) / 'lbw_calibrator.pkl'))

    # 3. 加载你的视线估计模型（仅 model 模式需要）
    gaze_model = None
    if args.mode == 'model':
        # ---- 替换为你的实际模型加载 ----
        # gaze_model = GazeModel(...)
        # gaze_model.load_state_dict(torch.load(args.model_ckpt))
        # gaze_model.eval()
        raise NotImplementedError("请替换为你项目的视线估计模型加载代码")

    # 4. 加载全景分割模型
    # ---- 替换为你的实际模型加载 ----
    # segmentor = PanopticSegmentor(...)
    segmentor = None  # 占位

    # 5. 逐帧评估
    results = []
    n_correct = 0

    for s in test_samples:
        # ---- 获取预测注视点 ----
        if args.mode == 'gt_calib':
            # 直接用 LBW 真值视线训练的映射预测
            pred_u, pred_v = calibrator.predict(s)
        else:
            # 用你的模型推理 face_path → pitch/yaw → 映射
            # face_img = preprocess_face(s['face_path'])   # 你的几何归一化
            # with torch.no_grad():
            #     pitch, yaw = gaze_model(face_img, head_pose)
            # pred_u, pred_v = calibrator.predict_from_model_output(
            #     pitch, yaw, eye_loc_3d, s['scene_size'])
            raise NotImplementedError("请替换为你项目的视线估计推理代码")

        # ---- 全景分割（用你的 segmentor）----
        # candidates = run_panoptic_segmentation(s['scene_path'], segmentor)
        # 若暂无 segmentor，用如下占位直接跳过：
        print(f"[{s['frame_id']}] pred_gaze=({pred_u:.1f}, {pred_v:.1f}), "
              f"gt_gaze={s['gaze_loc_2d']}, scene={s['scene_size']}")
        results.append({
            'frame_id'  : s['frame_id'],
            'pred_u'    : float(pred_u),
            'pred_v'    : float(pred_v),
            'gt_u'      : float(s['gaze_loc_2d'][0]),
            'gt_v'      : float(s['gaze_loc_2d'][1]),
            'pixel_err' : float(np.sqrt(
                (pred_u - s['gaze_loc_2d'][0])**2 +
                (pred_v - s['gaze_loc_2d'][1])**2
            )),
        })

    # 6. 汇总
    if results:
        errs = [r['pixel_err'] for r in results]
        print("\n===== 注视点预测误差 =====")
        print(f"  样本数    : {len(errs)}")
        print(f"  平均误差  : {np.mean(errs):.1f} px")
        print(f"  中位数误差: {np.median(errs):.1f} px")
        print(f"  标准差    : {np.std(errs):.1f} px")
        print(f"  最大误差  : {np.max(errs):.1f} px")

        if args.output_dir:
            out_path = str(Path(args.output_dir) / f'results_{args.mode}.json')
            with open(out_path, 'w') as f:
                json.dump({'summary': {
                    'n': len(errs), 'mean_px': np.mean(errs),
                    'median_px': np.median(errs), 'std_px': np.std(errs),
                }, 'details': results}, f, indent=2)
            print(f"结果已保存到 {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lbw_root',   required=True,
                        help='LookBothWays 数据集根目录（含 face_ims/ 等子目录）')
    parser.add_argument('--model_ckpt', default=None,
                        help='你的视线估计模型权重路径（model 模式需要）')
    parser.add_argument('--mode', choices=['gt_calib', 'model'], default='gt_calib',
                        help='gt_calib: 用真值视线验证场景匹配; model: 端到端验证')
    parser.add_argument('--calib_ratio', type=float, default=0.3,
                        help='用于训练映射的样本比例（默认 0.3）')
    parser.add_argument('--output_dir', default='./lbw_results',
                        help='结果输出目录')
    args = parser.parse_args()
    run_validation(args)
