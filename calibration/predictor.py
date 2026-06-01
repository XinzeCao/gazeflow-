#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
精简版离线/实时注视点预测器
当前策略：
1. 仅保留 UniGaze 特征提取
2. 仅保留 ridge 主映射
3. 使用与标定一致的 8D gaze/head/face-center 特征
"""

import os
import json
import argparse
import cv2
import numpy as np
import torch
import sys
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CUR_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from gazemodel.extractor import GazeFeatureExtractor
from sklearn.preprocessing import PolynomialFeatures


class OfflineGazePredictor:
    """仅保留 ridge 的注视点预测器"""

    def __init__(self, calib_path: str, model_cfg_path: str = None, ckpt_path: str = None,
                 device: str = 'auto', camera_calib_path: str = 'camera_calib.json'):
        self.device = torch.device(device if device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.camera_calib_path = camera_calib_path
        print(f"[INFO] 使用设备: {self.device}")

        self.load_calibration(calib_path)

        if model_cfg_path is None:
            model_cfg_path = os.path.join('gazemodel', 'config.yaml')
        if ckpt_path is None:
            ckpt_path = os.path.join('gazemodel', 'weights', 'unigaze_l16_joint.pth.tar')

        self.feature_extractor = GazeFeatureExtractor(
            model_cfg_path=model_cfg_path,
            ckpt_path=ckpt_path,
            device=self.device,
            camera_calib_path=camera_calib_path
        )

        print("[INFO] 预测器初始化完成")
        print(f"  - 主映射: ridge ({self.mean.shape[0]}D)")
        print(f"  - 屏幕尺寸: {self.screen_width}x{self.screen_height}")

    def load_calibration(self, calib_path: str):
        if not os.path.exists(calib_path):
            raise FileNotFoundError(f"标定文件不存在: {calib_path}")

        with open(calib_path, 'r', encoding='utf-8') as f:
            calib = json.load(f)

        self.calib_path = calib_path
        self.calib_data = calib

        # ===== 仅加载 legacy ridge 参数 =====
        lr = calib.get('legacy_ridge', calib)

        mean_data = lr.get('mean', calib.get('mean'))
        std_data = lr.get('std', calib.get('std'))
        if mean_data is None or std_data is None:
            raise ValueError("标定文件缺少 legacy ridge 的 mean/std 参数")

        self.mean = torch.tensor(mean_data, dtype=torch.float32, device=self.device)
        self.std = torch.tensor(std_data, dtype=torch.float32, device=self.device)

        self.normalization_strategy = lr.get(
            'normalization_strategy',
            calib.get('normalization_strategy', 'standard_zscore')
        )

        med = lr.get('median', calib.get('median'))
        iqr_val = lr.get('iqr', calib.get('iqr'))
        if med is not None and iqr_val is not None:
            self.median = torch.tensor(med, dtype=torch.float32, device=self.device)
            self.iqr = torch.tensor(iqr_val, dtype=torch.float32, device=self.device)
        else:
            self.median = None
            self.iqr = None

        self.poly_degree = lr.get('poly_degree', calib.get('poly_degree', 1))

        coef_x = lr.get('coef_x', calib.get('coef_x'))
        intercept_x = lr.get('intercept_x', calib.get('intercept_x'))
        coef_y = lr.get('coef_y', calib.get('coef_y'))
        intercept_y = lr.get('intercept_y', calib.get('intercept_y'))

        if coef_x is None or intercept_x is None or coef_y is None or intercept_y is None:
            raise ValueError("标定文件缺少 legacy ridge 的回归参数")

        self.coef_x = torch.tensor(coef_x, dtype=torch.float32, device=self.device)
        self.intercept_x = torch.tensor(intercept_x, dtype=torch.float32, device=self.device)
        self.coef_y = torch.tensor(coef_y, dtype=torch.float32, device=self.device)
        self.intercept_y = torch.tensor(intercept_y, dtype=torch.float32, device=self.device)

        # ===== 屏幕和标定区域 =====
        self.screen_width = calib['screen']['width']
        self.screen_height = calib['screen']['height']
        self.calib_resize = calib.get('resize', 1.0)

        if 'calibration_region' in calib:
            cr = calib['calibration_region']
            self.calib_x_min = cr['x_min']
            self.calib_y_min = cr['y_min']
            self.calib_x_max = cr['x_max']
            self.calib_y_max = cr['y_max']
            self.calib_width = cr['width']
            self.calib_height = cr['height']
            print(f"[INFO] 标定区域: ({self.calib_x_min}, {self.calib_y_min}, {self.calib_x_max}, {self.calib_y_max})")
        else:
            self.calib_x_min = 0
            self.calib_y_min = 0
            self.calib_x_max = self.screen_width
            self.calib_y_max = self.screen_height
            self.calib_width = self.screen_width
            self.calib_height = self.screen_height
            print(f"[INFO] 标定区域: 使用全屏 ({self.screen_width} x {self.screen_height})")

        # 仅保留可配置 offset，不保留硬编码补丁
        self.offset_x = float(calib.get('offset_x', 0.0))
        self.offset_y = float(calib.get('offset_y', 0.0))
        self.screen_x_flipped = bool(calib.get('screen_x_flipped', False))
        if self.offset_x != 0 or self.offset_y != 0:
            print(f"[INFO] 偏移量校正: X={self.offset_x:.1f}px, Y={self.offset_y:.1f}px")

        if 'statistics' in calib:
            stats = calib['statistics']
            print(f"[INFO] 标定精度: {stats.get('mean_error_px', 0):.1f}±{stats.get('std_error_px', 0):.1f}px")

    def update_offset(self, offset_x: float, offset_y: float, save_to_file: bool = True):
        self.offset_x = float(offset_x)
        self.offset_y = float(offset_y)

        if save_to_file:
            self.calib_data['offset_x'] = self.offset_x
            self.calib_data['offset_y'] = self.offset_y
            with open(self.calib_path, 'w', encoding='utf-8') as f:
                json.dump(self.calib_data, f, indent=2, ensure_ascii=False)
            print(f"[INFO] 偏移量已保存: X={self.offset_x:.1f}px, Y={self.offset_y:.1f}px")

    def _predict_legacy(self, features: Dict[str, Any]) -> Optional[Tuple[float, float]]:
        """
        使用与标定文件维度一致的 ridge 特征进行预测。
        """
        x_features = features.get('x')
        if x_features is None:
            return None

        x_features = np.asarray(x_features, dtype=np.float32).reshape(-1)

        if x_features.shape[0] != self.mean.shape[0]:
            if self.mean.shape[0] == 2 and x_features.shape[0] > 2:
                x_features = x_features[:2]
                print("[WARN] 当前标定文件是旧版2D特征，已回退到pitch/yaw；建议重新标定以使用8D映射。")
            else:
                return None

        x_tensor = torch.tensor(x_features, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            if self.normalization_strategy == 'hybrid_zscore_robust' and self.median is not None and self.iqr is not None:
                x_zscore = (x_tensor - self.mean) / (self.std + 1e-6)
                x_robust = (x_tensor - self.median) / (self.iqr + 1e-6)
                x_normalized = 0.7 * x_zscore + 0.3 * x_robust
            else:
                x_normalized = (x_tensor - self.mean) / (self.std + 1e-6)

            x_np = x_normalized.cpu().numpy()
            poly = PolynomialFeatures(degree=self.poly_degree, include_bias=False)
            x_poly_np = poly.fit_transform(x_np.reshape(1, -1))[0]
            x_poly = torch.tensor(x_poly_np, dtype=torch.float32, device=self.device)

            u = (x_poly @ self.coef_x) + self.intercept_x
            v = (x_poly @ self.coef_y) + self.intercept_y
            u, v = float(u.item()), float(v.item())

        # 归一化坐标 -> 像素坐标
        u = u * self.calib_width + self.calib_x_min
        v = v * self.calib_height + self.calib_y_min

        if self.screen_x_flipped:
            u = self.screen_width - u

        # 仅保留可配置 offset
        u += self.offset_x
        v += self.offset_y

        u = max(0, min(self.screen_width - 1, u))
        v = max(0, min(self.screen_height - 1, v))

        return u, v

    def predict_gaze_point(self, face_image: np.ndarray, resize_factor: float = None) -> Dict[str, Any]:
        try:
            if resize_factor is None:
                resize_factor = self.calib_resize

            if abs(resize_factor - 1.0) > 1e-6:
                face_input = cv2.resize(face_image, None, fx=resize_factor, fy=resize_factor,
                                        interpolation=cv2.INTER_LINEAR)
            else:
                face_input = face_image.copy()

            features = self.feature_extractor.extract(face_input)
            if features is None:
                return {'success': False, 'error_msg': '特征提取失败，可能未检测到人脸'}

            result = self._predict_legacy(features)
            if result is None:
                return {'success': False, 'error_msg': 'legacy ridge 预测失败'}

            return {
                'success': True,
                'gaze_point': result,
                'features': features.get('x'),
                'raw_features': features,
                'method': 'ridge'
            }

        except Exception as e:
            return {'success': False, 'error_msg': f'预测过程出错: {str(e)}'}

    def check_instance_gaze(self, gaze_point: Tuple[float, float], instance_mask: np.ndarray,
                            tolerance_radius: int = 0, use_rounded: bool = True) -> Dict[str, Any]:
        u, v = gaze_point

        if use_rounded:
            u_int, v_int = int(round(u)), int(round(v))
        else:
            u_int, v_int = int(u), int(v)

        h, w = instance_mask.shape[:2]
        if v_int < 0 or v_int >= h or u_int < 0 or u_int >= w:
            return {
                'is_gazing': False,
                'reason': 'point_out_of_bounds',
                'gaze_point_int': (u_int, v_int),
                'distance_to_instance': float('inf')
            }

        if instance_mask.max() > 1:
            mask_binary = (instance_mask > 127).astype(np.uint8)
        else:
            mask_binary = instance_mask.astype(np.uint8)

        is_hit = bool(mask_binary[v_int, u_int] > 0)

        if not is_hit and tolerance_radius > 0:
            r = tolerance_radius
            y0 = max(0, v_int - r)
            y1 = min(h, v_int + r + 1)
            x0 = max(0, u_int - r)
            x1 = min(w, u_int + r + 1)

            y_grid, x_grid = np.ogrid[y0:y1, x0:x1]
            circle_mask = ((x_grid - u_int) ** 2 + (y_grid - v_int) ** 2) <= r ** 2

            neighborhood = mask_binary[y0:y1, x0:x1]
            is_hit = bool((neighborhood[circle_mask] > 0).any())

        distance_to_instance = self._compute_distance_to_instance(gaze_point, mask_binary)

        return {
            'is_gazing': is_hit,
            'reason': 'hit' if is_hit else 'miss',
            'gaze_point_int': (u_int, v_int),
            'distance_to_instance': distance_to_instance,
            'tolerance_used': tolerance_radius
        }

    def _compute_distance_to_instance(self, gaze_point: Tuple[float, float], mask_binary: np.ndarray) -> float:
        try:
            contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                return float('inf')

            u, v = gaze_point
            min_distance = float('inf')

            for contour in contours:
                dist = cv2.pointPolygonTest(contour, (u, v), True)
                abs_dist = abs(dist)
                if abs_dist < min_distance:
                    min_distance = abs_dist

            return min_distance

        except Exception:
            if (mask_binary > 0).any():
                dist_transform = cv2.distanceTransform(
                    (1 - mask_binary).astype(np.uint8),
                    cv2.DIST_L2, 5
                )
                u_int, v_int = int(round(gaze_point[0])), int(round(gaze_point[1]))
                h, w = mask_binary.shape
                if 0 <= v_int < h and 0 <= u_int < w:
                    return float(dist_transform[v_int, u_int])

            return float('inf')

    def process_single_image(self, face_image_path: str, instance_color_map_path: str,
                             output_dir: str = None, tolerance_radius: int = 0,
                             save_visualization: bool = True, instance_category: str = None,
                             palette_color: List[int] = None) -> Dict[str, Any]:
        if not os.path.exists(face_image_path):
            return {'success': False, 'error': f'面部图像不存在: {face_image_path}'}

        if not os.path.exists(instance_color_map_path):
            return {'success': False, 'error': f'实例彩色映射图不存在: {instance_color_map_path}'}

        face_image = cv2.imread(face_image_path)
        if face_image is None:
            return {'success': False, 'error': f'无法读取面部图像: {face_image_path}'}

        instance_color_map = cv2.imread(instance_color_map_path)
        if instance_color_map is None:
            return {'success': False, 'error': f'无法读取实例彩色映射图: {instance_color_map_path}'}
        instance_color_map = cv2.cvtColor(instance_color_map, cv2.COLOR_BGR2RGB)

        if palette_color is None:
            try:
                color_map_dir = os.path.dirname(instance_color_map_path)
                parent_dir = os.path.dirname(color_map_dir)
                metadata_path = os.path.join(parent_dir, 'metadata.jsonl')

                color_map_basename = os.path.basename(instance_color_map_path)
                if '_colormap.png' in color_map_basename:
                    sample_id = color_map_basename.replace('_colormap.png', '')
                else:
                    sample_id = color_map_basename.split('_colormap')[0]

                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            if not line.strip():
                                continue
                            try:
                                meta = json.loads(line)
                                if meta.get('sample_id') == sample_id:
                                    palette_color = meta.get('object_info', {}).get('palette_color')
                                    if palette_color:
                                        break
                            except json.JSONDecodeError:
                                continue
            except Exception:
                pass

        if palette_color is None:
            return {
                'success': False,
                'error': '缺少 palette_color，且无法从 metadata.jsonl 读取'
            }

        palette_color_array = np.array(palette_color, dtype=np.uint8)
        instance_mask = np.all(instance_color_map == palette_color_array, axis=-1).astype(np.uint8) * 255

        gaze_result = self.predict_gaze_point(face_image)
        if not gaze_result['success']:
            return {'success': False, 'error': f"注视点预测失败: {gaze_result['error_msg']}"}

        gaze_point = gaze_result['gaze_point']
        instance_result = self.check_instance_gaze(gaze_point, instance_mask, tolerance_radius)

        result = {
            'success': True,
            'face_image': face_image_path,
            'instance_color_map': instance_color_map_path,
            'palette_color': palette_color,
            'gaze_point': gaze_point,
            'is_gazing_instance': instance_result['is_gazing'],
            'distance_to_instance': instance_result['distance_to_instance'],
            'gaze_point_int': instance_result['gaze_point_int'],
            'tolerance_radius': tolerance_radius,
            'category': instance_category or "unknown"
        }

        if save_visualization and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            vis_path = os.path.join(output_dir, f"{Path(face_image_path).stem}_gaze_vis.png")
            self._create_visualization(
                face_image, instance_mask, gaze_point,
                instance_result, vis_path, instance_color_map
            )
            result['visualization'] = vis_path

        return result

    def _create_visualization(self, face_image: np.ndarray, instance_mask: np.ndarray,
                              gaze_point: Tuple[float, float], instance_result: Dict[str, Any],
                              vis_path: str, instance_color_map: Optional[np.ndarray] = None):
        try:
            if instance_color_map is not None:
                visualization = cv2.cvtColor(instance_color_map.copy(), cv2.COLOR_RGB2BGR)
            else:
                visualization = np.zeros((instance_mask.shape[0], instance_mask.shape[1], 3), dtype=np.uint8)
                mask_color = (0, 255, 0) if instance_result['is_gazing'] else (0, 0, 255)
                visualization[instance_mask > 0] = mask_color

            u, v = gaze_point
            h, w = visualization.shape[:2]
            u_int, v_int = int(round(u)), int(round(v))
            u_int = max(0, min(w - 1, u_int))
            v_int = max(0, min(h - 1, v_int))

            tolerance_radius = instance_result.get('tolerance_used', 0)
            max_radius = max(20, int(tolerance_radius * 1.5))

            for i in range(10, 0, -1):
                current_radius = int(max_radius * i / 10)
                alpha = 0.05 + (0.4 * i / 10)

                overlay = visualization.copy()
                red_intensity = int(200 * i / 10) + 55
                color = (0, 0, min(255, red_intensity))
                cv2.circle(overlay, (u_int, v_int), current_radius, color, -1)
                cv2.addWeighted(overlay, alpha, visualization, 1 - alpha, 0, visualization)

            overlay = visualization.copy()
            cv2.circle(overlay, (u_int, v_int), int(max_radius * 0.7), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.3, visualization, 0.7, 0, visualization)

            overlay = visualization.copy()
            cv2.circle(overlay, (u_int, v_int), int(max_radius), (0, 0, 0), 2)
            cv2.addWeighted(overlay, 1.0, visualization, 0.0, 0, visualization)

            cv2.circle(visualization, (u_int, v_int), 3, (0, 0, 255), -1)
            cv2.imwrite(vis_path, visualization)

        except Exception as e:
            print(f"[ERROR] 创建可视化失败: {e}")

    def process_batch(self, batch_list_path: str, output_dir: str, tolerance_radius: int = 0) -> Dict[str, Any]:
        try:
            if not os.path.exists(batch_list_path):
                return {'success': False, 'error': f'批量处理列表文件不存在: {batch_list_path}'}

            os.makedirs(output_dir, exist_ok=True)

            batch_results = []
            total_success = 0
            total_fail = 0
            total_gaze_hit = 0
            total_gaze_miss = 0

            with open(batch_list_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                try:
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) < 2:
                        continue

                    face_image_path = parts[0]
                    instance_color_map_path = parts[1]
                    instance_category = parts[2] if len(parts) >= 3 else "unknown"

                    palette_color = None
                    if len(parts) >= 6:
                        try:
                            palette_color = [int(parts[3]), int(parts[4]), int(parts[5])]
                        except ValueError:
                            pass

                    result = self.process_single_image(
                        face_image_path, instance_color_map_path,
                        output_dir, tolerance_radius,
                        save_visualization=True,
                        instance_category=instance_category,
                        palette_color=palette_color
                    )

                    batch_results.append(result)

                    if result['success']:
                        total_success += 1
                        if result['is_gazing_instance']:
                            total_gaze_hit += 1
                        else:
                            total_gaze_miss += 1
                    else:
                        total_fail += 1

                except Exception as e:
                    total_fail += 1
                    batch_results.append({'success': False, 'error': str(e), 'line': line})

            overall_hit_rate = (total_gaze_hit / total_success * 100) if total_success > 0 else 0.0

            results_json = os.path.join(output_dir, 'batch_results.json')
            with open(results_json, 'w', encoding='utf-8') as f:
                json.dump({
                    'total_samples': len(batch_results),
                    'success_count': total_success,
                    'fail_count': total_fail,
                    'hit_count': total_gaze_hit,
                    'miss_count': total_gaze_miss,
                    'overall_hit_rate': overall_hit_rate,
                    'results': batch_results
                }, f, ensure_ascii=False, indent=2)

            return {
                'success': True,
                'total_samples': len(batch_results),
                'success_count': total_success,
                'fail_count': total_fail,
                'hit_count': total_gaze_hit,
                'miss_count': total_gaze_miss,
                'overall_hit_rate': overall_hit_rate,
                'results_file': results_json
            }

        except Exception as e:
            return {'success': False, 'error': f'批量处理过程出错: {str(e)}'}


def main():
    parser = argparse.ArgumentParser(description='精简版离线注视点预测器')
    parser.add_argument('--calib', type=str, default='calib.json', help='标定文件路径')
    parser.add_argument('--model-cfg', type=str,
                        default=os.path.join('gazemodel', 'config.yaml'),
                        help='模型配置文件')
    parser.add_argument('--ckpt', type=str, default=None, help='模型权重文件')
    parser.add_argument('--device', type=str, default='auto', help='计算设备')
    parser.add_argument('--tolerance', type=int, default=0, help='命中判定容忍半径')
    parser.add_argument('--camera-calib', type=str, default='camera_calib.json',
                        help='OpenCV相机内参JSON文件路径')
    parser.add_argument('--face-image', type=str, help='面部图像路径')
    parser.add_argument('--instance-color-map', type=str, help='实例彩色映射图路径')
    parser.add_argument('--palette-color', type=str, help='实例颜色，格式 R,G,B')
    parser.add_argument('--batch-list', type=str, help='批量处理列表文件')
    parser.add_argument('--output', type=str, default='gaze_output', help='输出目录')
    parser.add_argument('--no-vis', action='store_true', help='不保存可视化')

    args = parser.parse_args()

    single_mode = args.face_image and args.instance_color_map
    batch_mode = args.batch_list

    if not (single_mode or batch_mode):
        print("[ERROR] 请指定处理模式")
        return 1

    try:
        predictor = OfflineGazePredictor(
            calib_path=args.calib,
            model_cfg_path=args.model_cfg,
            ckpt_path=args.ckpt,
            device=args.device,
            camera_calib_path=args.camera_calib
        )

        if single_mode:
            palette_color = None
            if args.palette_color:
                palette_color = [int(c.strip()) for c in args.palette_color.split(',')]

            result = predictor.process_single_image(
                args.face_image, args.instance_color_map,
                args.output, args.tolerance, not args.no_vis,
                palette_color=palette_color
            )

            if result['success']:
                print(f"[SUCCESS] 注视点: ({result['gaze_point'][0]:.1f}, {result['gaze_point'][1]:.1f})")
            else:
                print(f"[ERROR] {result['error']}")
                return 1

        elif batch_mode:
            result = predictor.process_batch(args.batch_list, args.output, args.tolerance)
            if not result['success']:
                print(f"[ERROR] {result['error']}")
                return 1

        return 0

    except Exception as e:
        print(f"[ERROR] 程序运行失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
