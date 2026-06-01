"""
lbw_end2end.py
--------------
端到端验证：用你自己的视线估计模型预测 LBW 数据集的注视目标。

核心流程:
  1. 加载 LBW 数据集
  2. 用 LBW 真值视线方向训练 GazeToSceneCalibrator（一次性，得到视线特征→场景坐标的映射）
  3. 对每个测试样本：
     a. 用 OfflineGazePredictor 跑面部图，拿到 raw_features
     b. raw_features → calibrator.predict_from_raw_features → 场景图注视点
     c. Mask2Former 跑场景图，得到候选目标
     d. predict_all() 跑 M1~M4
     e. 对比真值，统计准确率

运行方式:
  python lbw_validation/lbw_end2end.py \
      --lbw-root      D:/gazeflow/lbw_dataset \
      --project-root  D:/gazeflow \
      --calib-path    D:/gazeflow/calib.json \
      --camera-calib-path D:/gazeflow/camera_calib.json \
      --model-dir     D:/gazeflow/mmdetection/checkpoints \
      --output-dir    D:/gazeflow/lbw_results/end2end \
      --tau           34 \
      --calib-ratio   0.3 \
      --max-samples   100      # 可选，限制样本数做快速测试
"""

import argparse
import json
import os
import sys
import time
import warnings
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict

# ================================================================== #
#  把原项目根目录加入 sys.path
# ================================================================== #
def setup_project_path(project_root: str):
    root = Path(project_root).resolve()
    for p in [str(root)]:
        if p not in sys.path:
            sys.path.insert(0, p)

# 本地模块
from lbw_dataset import LBWDataset
from gaze_to_scene_calibrator import GazeToSceneCalibrator


# ================================================================== #
#  Mask2Former 全景分割
# ================================================================== #
def load_mask2former(model_dir: str, device_str: str = 'auto'):
    import torch
    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
    device = (torch.device('cuda' if torch.cuda.is_available() else 'cpu')
              if device_str == 'auto' else torch.device(device_str))
    processor = AutoImageProcessor.from_pretrained(model_dir)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_dir).to(device)
    model.eval()
    print(f"[Mask2Former] 加载完成，device={device}")
    return processor, model, device


def run_mask2former(scene_img_path: str, processor, model, device, min_area: int = 1200):
    """跑全景分割，返回候选目标列表（格式和原项目 evaluate_gaze_target 一致）"""
    import torch
    from PIL import Image as PILImage
    EXCLUDED_CLASSES = {"pole"}

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
            overlap_mask_area_threshold=0.8,
        )[0]

    seg_map = result["segmentation"].cpu().numpy()
    id2label = {int(k): str(v) for k, v in model.config.id2label.items()}
    candidates = []
    for seg in result["segments_info"]:
        cls = id2label.get(int(seg["label_id"]), f"label_{seg['label_id']}")
        if cls in EXCLUDED_CLASSES:
            continue
        mask = (seg_map == int(seg["id"])).astype(np.uint8)
        if mask.sum() < min_area:
            continue
        ys, xs = np.where(mask)
        candidates.append({
            "target_id" : str(seg["id"]),
            "class_name": cls,
            "mask"      : mask,
            "centroid"  : [float(xs.mean()), float(ys.mean())],
        })
    return candidates


# ================================================================== #
#  真值目标 ID：把 Gaze_Loc_2D 打到分割结果里
# ================================================================== #
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
#  主验证流程
# ================================================================== #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lbw-root',          required=True)
    parser.add_argument('--project-root',      required=True)
    parser.add_argument('--calib-path',        required=True, help='你的 calib.json（屏幕标定，OfflineGazePredictor 需要）')
    parser.add_argument('--camera-calib-path', default='camera_calib.json')
    parser.add_argument('--model-cfg-path',    default=None)
    parser.add_argument('--ckpt-path',         default=None)
    parser.add_argument('--model-dir',         required=True, help='Mask2Former 本地模型目录')
    parser.add_argument('--output-dir',        default='./lbw_end2end_results')
    parser.add_argument('--calib-ratio',       type=float, default=0.3, help='用于训练视线→场景映射的比例')
    parser.add_argument('--tau',               type=float, default=34.0, help='边缘距离衰减尺度（LBW 推荐 34）')
    parser.add_argument('--truncation-factor', type=float, default=5.0)
    parser.add_argument('--lambda-min',        type=float, default=0.2)
    parser.add_argument('--lambda-max',        type=float, default=1.0)
    parser.add_argument('--max-samples',       type=int, default=0, help='限制处理样本数，0=不限制')
    parser.add_argument('--skip-mask2former',  action='store_true', help='跳过全景分割，只验证视线预测精度')
    args = parser.parse_args()

    setup_project_path(args.project_root)

    # 导入原项目模块（不修改任何原有代码）
    from calibration.predictor import OfflineGazePredictor
    from gaze_target_inference import predict_all

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. 加载 LBW 数据集 ----
    dataset = LBWDataset(args.lbw_root)
    all_samples = [dataset[i] for i in range(len(dataset))]
    n_calib = max(10, int(len(all_samples) * args.calib_ratio))
    calib_samples = all_samples[:n_calib]
    test_samples  = all_samples[n_calib:]
    if args.max_samples > 0:
        test_samples = test_samples[:args.max_samples]
    print(f"标定样本: {len(calib_samples)}，测试样本: {len(test_samples)}")

    # ---- 2. 加载 OfflineGazePredictor（标定前先加载，标定要用）----
    print(f"\n[INFO] 加载 OfflineGazePredictor...")
    predictor = OfflineGazePredictor(
        calib_path=args.calib_path,
        model_cfg_path=args.model_cfg_path,
        ckpt_path=args.ckpt_path,
        camera_calib_path=args.camera_calib_path,
    )
    print(f"[INFO] OfflineGazePredictor 就绪")

    # ---- 3. 用模型在标定集上跑预测视线 ----
    print(f"\n[INFO] 在 {len(calib_samples)} 个标定样本上跑模型，收集预测视线方向...")
    predicted_calib_dirs = []
    n_calib_fail = 0
    for i, s in enumerate(calib_samples):
        face_img = cv2.imread(s['face_path'])
        if face_img is None:
            predicted_calib_dirs.append(None)
            n_calib_fail += 1
            continue
        result = predictor.predict_gaze_point(face_img)
        if not result.get('success'):
            predicted_calib_dirs.append(None)
            n_calib_fail += 1
            continue
        # 取模型预测的 gaze_cam，并对齐到 LBW 坐标系（整体取反）
        gaze_dir = np.asarray(result['raw_features']['gaze_cam'], dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(gaze_dir)
        if norm > 1e-6:
            gaze_dir = gaze_dir / norm
        gaze_dir = -gaze_dir   # OpenGL → OpenCV 坐标系
        predicted_calib_dirs.append(gaze_dir)
        if (i + 1) % 50 == 0:
            print(f"  标定集预测进度: {i+1}/{len(calib_samples)}, 失败={n_calib_fail}")
    print(f"[INFO] 标定集预测完成，有效样本={len(calib_samples) - n_calib_fail}/{len(calib_samples)}")

    # ---- 4. 用模型预测的视线方向训练 calibrator ----
    print(f"\n[INFO] 训练视线→场景图坐标映射（基于模型预测视线）")
    calibrator = GazeToSceneCalibrator(alpha=0.05, degree=2)
    calibrator.fit_with_predicted_gaze(calib_samples, predicted_calib_dirs, verbose=True)

    # ---- 5. 加载 Mask2Former ----
    seg_processor, seg_model, seg_device = None, None, None
    if not args.skip_mask2former:
        seg_processor, seg_model, seg_device = load_mask2former(args.model_dir)

    # ---- 5. 逐样本评估 ----
    methods = ["M1", "M2", "M3", "M4"]
    agg = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    seg_cache = {}
    results = []
    n_face_fail = 0   # 人脸检测失败
    n_no_gt    = 0   # 真值注视点不在任何掩码内
    gaze_errs  = []
    t_start = time.time()

    for i, s in enumerate(test_samples):
        # ---- A. 用 OfflineGazePredictor 跑面部图 ----
        face_img = cv2.imread(s['face_path'])
        if face_img is None:
            n_face_fail += 1
            continue
        result = predictor.predict_gaze_point(face_img)
        if not result.get('success'):
            n_face_fail += 1
            if i < 5:
                print(f"  [WARN] 样本 {s['frame_id']} 特征提取失败: {result.get('error_msg')}")
            continue

        raw = result['raw_features']

        # ---- B. raw_features → 场景图注视点 ----
        try:
            pred_u, pred_v = calibrator.predict_from_raw_features(raw, s)
        except Exception as e:
            print(f"  [WARN] 样本 {s['frame_id']} 映射失败: {e}")
            continue

        gt_u, gt_v = float(s['gaze_loc_2d'][0]), float(s['gaze_loc_2d'][1])
        px_err = float(np.sqrt((pred_u - gt_u)**2 + (pred_v - gt_v)**2))
        gaze_errs.append(px_err)

        record = {
            'frame_id'  : s['frame_id'],
            'pred_u'    : round(pred_u, 2),
            'pred_v'    : round(pred_v, 2),
            'gt_u'      : round(gt_u, 2),
            'gt_v'      : round(gt_v, 2),
            'pixel_err' : round(px_err, 2),
        }

        # ---- C. 全景分割 + 目标推理（如果启用）----
        if seg_model is not None:
            sp = s['scene_path']
            if sp not in seg_cache:
                try:
                    seg_cache[sp] = run_mask2former(sp, seg_processor, seg_model, seg_device)
                except Exception as e:
                    print(f"  [WARN] 分割 {s['frame_id']} 失败: {e}")
                    seg_cache[sp] = []
            candidates = seg_cache[sp]

            if candidates:
                gt_id, gt_class = get_gt_target_id(s['gaze_loc_2d'], candidates)
                record['gt_target_id'] = gt_id
                record['gt_class']     = gt_class
                if gt_id is None:
                    n_no_gt += 1
                else:
                    preds = predict_all(
                        gaze_point=(pred_u, pred_v),
                        candidates=candidates,
                        tau=args.tau,
                        truncation_factor=args.truncation_factor,
                        lambda_min=args.lambda_min,
                        lambda_max=args.lambda_max,
                    )
                    for m in methods:
                        r = preds[m]
                        pred_id  = r.get('predicted_id')
                        pred_cls = r.get('predicted_class', 'unknown')
                        correct = int(pred_id == gt_id)
                        record[f'{m}_pred_id']    = pred_id
                        record[f'{m}_pred_class'] = pred_cls
                        record[f'{m}_correct']    = correct
                        agg[m][gt_class][1] += 1
                        agg[m][gt_class][0] += correct

        results.append(record)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t_start
            print(f"  已处理 {i+1}/{len(test_samples)}，"
                  f"人脸失败={n_face_fail}, 无真值目标={n_no_gt}, "
                  f"耗时={elapsed:.1f}s")

    elapsed = time.time() - t_start
    print(f"\n[INFO] 完成，总耗时 {elapsed:.1f}s")
    print(f"  人脸/特征提取失败: {n_face_fail}/{len(test_samples)}")
    if seg_model is not None:
        print(f"  真值注视点未落入任何掩码: {n_no_gt}")

    # ---- 6. 汇总 ----
    print("\n===== 注视点预测误差 =====")
    if gaze_errs:
        print(f"  样本数    : {len(gaze_errs)}")
        print(f"  平均误差  : {np.mean(gaze_errs):.1f} px")
        print(f"  中位数误差: {np.median(gaze_errs):.1f} px")
        print(f"  标准差    : {np.std(gaze_errs):.1f} px")
        print(f"  最大误差  : {np.max(gaze_errs):.1f} px")

    if seg_model is not None and agg:
        print("\n===== 注视目标识别准确率（M1~M4，τ=%.0fpx）=====" % args.tau)
        KEY = {"car","truck","bus","person","rider","motorcycle",
               "bicycle","traffic light","traffic sign","train"}

        all_classes = sorted(set(k for m in methods for k in agg[m]))
        print(f"\n  {'类别':<18} {'N':>5}", end="")
        for m in methods:
            print(f"  {m:>8}", end="")
        print()
        for cls in all_classes:
            n = agg[methods[0]][cls][1]
            print(f"  {cls:<18} {n:>5}", end="")
            for m in methods:
                h, t = agg[m][cls]
                print(f"  {h/t*100 if t else 0:>7.1f}%", end="")
            print()

        # 关键目标 / 背景 / 总体
        print()
        for grp_name, in_grp in [("关键目标", lambda c: c in KEY),
                                  ("背景目标", lambda c: c not in KEY),
                                  ("总体",     lambda c: True)]:
            print(f"  {grp_name:<18}", end="")
            tot = sum(agg[methods[0]][k][1] for k in all_classes if in_grp(k))
            print(f" {tot:>5}", end="")
            for m in methods:
                h = sum(agg[m][k][0] for k in all_classes if in_grp(k))
                t = sum(agg[m][k][1] for k in all_classes if in_grp(k))
                print(f"  {h/t*100 if t else 0:>7.1f}%", end="")
            print()

    # ---- 7. 保存结果 ----
    summary = {
        'mode':                'end2end',
        'tau':                 args.tau,
        'lambda_min':          args.lambda_min,
        'lambda_max':          args.lambda_max,
        'n_test':              len(test_samples),
        'n_processed':         len(results),
        'n_face_fail':         n_face_fail,
        'n_no_gt':             n_no_gt,
        'gaze_err_mean':       float(np.mean(gaze_errs)) if gaze_errs else None,
        'gaze_err_median':     float(np.median(gaze_errs)) if gaze_errs else None,
        'gaze_err_std':        float(np.std(gaze_errs)) if gaze_errs else None,
        'target_acc': {m: {cls: {'hits': agg[m][cls][0], 'total': agg[m][cls][1]}
                           for cls in agg[m]}
                       for m in methods} if agg else {},
    }
    with open(out_dir / 'end2end_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if results:
        keys = list(results[0].keys())
        with open(out_dir / 'end2end_per_sample.csv', 'w', encoding='utf-8') as f:
            f.write(','.join(keys) + '\n')
            for r in results:
                f.write(','.join(str(r.get(k, '')) for k in keys) + '\n')

    print(f"\n结果已保存到 {out_dir}")


if __name__ == '__main__':
    main()