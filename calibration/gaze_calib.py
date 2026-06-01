#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
注视点标定程序（在线预览增强版）
用于建立 8D gaze/head/face-center 到屏幕坐标的 Ridge 回归映射
"""

import os
import sys
import json
import time
import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox, ttk
import threading
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional
from collections import defaultdict, deque

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()

# ---------------------------------------------------------------------
# matplotlib（可选）
# ---------------------------------------------------------------------
try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib import font_manager

    MATPLOTLIB_AVAILABLE = True

    available_fonts = {f.name for f in font_manager.fontManager.ttflist}
    preferred_fonts = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
    usable_fonts = [f for f in preferred_fonts if f in available_fonts]

    if usable_fonts:
        plt.rcParams["font.family"] = usable_fonts
    else:
        plt.rcParams["font.family"] = ["DejaVu Sans"]

    plt.rcParams["axes.unicode_minus"] = False

except ImportError:
    print("[WARN] matplotlib 未安装，无法提供可视化功能")
    MATPLOTLIB_AVAILABLE = False

# ---------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------
_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CUR_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
from gazemodel.extractor import GazeFeatureExtractor


class CalibrationProgram:
    def __init__(
        self,
        screen_width=2194,
        screen_height=1234,
        grid_size=(4, 4),
        output_dir=None,
        calibration_region=None,
        grid_mode=None,
        camera_id=0,
        camera_calib_path='camera_calib.json',
        calibration_margin_ratio=0.10
    ):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.output_dir = output_dir
        self.camera_id = camera_id
        self.camera_calib_path = camera_calib_path
        self.calibration_margin_ratio = calibration_margin_ratio

        if grid_mode is not None:
            grid_mode_map = {
                'quick': (3, 3),
                'balanced': (4, 4),
                'thorough': (5, 5)
            }
            grid_size = grid_mode_map.get(grid_mode, grid_size)
            print(f"[INFO] grid_mode='{grid_mode}' -> grid_size={grid_size}")

        self.grid_cols, self.grid_rows = grid_size

        if calibration_region is None:
            self.calibration_region = (0, 0, screen_width, screen_height)
        else:
            self.calibration_region = calibration_region

        self.calibration_points = self._generate_calibration_points()
        self.current_point_idx = 0

        # 已完成点的数据（正式训练集）
        self.collected_data: List[Dict[str, Any]] = []

        self.is_collecting = False

        self.camera = None
        self.camera_lock = threading.Lock()

        self.feature_extractor = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.root = None
        self.canvas = None
        self.status_label = None
        self.progress_var = None

        self.collection_thread = None
        self.thread_status_text = ""
        self.thread_progress_text = ""
        self.finish_scheduled = False

        self.collection_duration = 2.5
        self.samples_per_point = 15
        self.warmup_duration = 0.5

        # 临时预览模型
        self.temp_model_x = None
        self.temp_model_y = None
        self.temp_mean = None
        self.temp_std = None
        self.temp_poly_degree = 2
        self.temp_model_lock = threading.Lock()

        # 蓝点预览
        self.realtime_gaze_point: Optional[Tuple[float, float]] = None
        self.realtime_gaze_error_px: Optional[float] = None
        self.realtime_running = False
        self.preview_model_ready = False

        # 蓝点平滑
        self.preview_smooth_alpha = 0.25
        self.preview_smoothed_point: Optional[Tuple[float, float]] = None

        # 输入稳定性判定
        self.preview_feature_buffer = deque(maxlen=8)
        self.preview_min_completed_points = 4
        self.preview_min_samples = 40
        self.preview_std_threshold = 0.04

    # -----------------------------------------------------------------
    # 标定点生成
    # -----------------------------------------------------------------
    def _generate_calibration_points(self) -> List[Tuple[int, int]]:
        points = []

        x_min, y_min, x_max, y_max = self.calibration_region
        width_range = x_max - x_min
        height_range = y_max - y_min

        margin_ratio_x = self.calibration_margin_ratio
        margin_ratio_y = self.calibration_margin_ratio

        margin_x = int(round(width_range * margin_ratio_x))
        margin_y = int(round(height_range * margin_ratio_y))

        inner_x_min = x_min + margin_x
        inner_x_max = x_max - margin_x
        inner_y_min = y_min + margin_y
        inner_y_max = y_max - margin_y

        inner_width = inner_x_max - inner_x_min
        inner_height = inner_y_max - inner_y_min

        if inner_width <= 0 or inner_height <= 0:
            raise ValueError(
                f"标定区域过小或边距设置过大: "
                f"region=({x_min},{y_min},{x_max},{y_max}), "
                f"margin=({margin_x},{margin_y})"
            )

        if self.grid_cols == 1:
            xs = [int(round((inner_x_min + inner_x_max) / 2))]
        else:
            xs = [
                int(round(inner_x_min + i * inner_width / (self.grid_cols - 1)))
                for i in range(self.grid_cols)
            ]

        if self.grid_rows == 1:
            ys = [int(round((inner_y_min + inner_y_max) / 2))]
        else:
            ys = [
                int(round(inner_y_min + j * inner_height / (self.grid_rows - 1)))
                for j in range(self.grid_rows)
            ]

        for row_idx, y in enumerate(ys):
            row_xs = xs if row_idx % 2 == 0 else list(reversed(xs))
            for x in row_xs:
                points.append((x, y))

        print(f"[INFO] 生成标定点: {self.grid_cols}x{self.grid_rows} = {len(points)} 个点")
        print(f"[INFO] 原始标定区域: ({x_min}, {y_min}) -> ({x_max}, {y_max})")
        print(f"[INFO] 留边后区域: ({inner_x_min}, {inner_y_min}) -> ({inner_x_max}, {inner_y_max})")
        print(f"[INFO] 边距: margin_x={margin_x}px, margin_y={margin_y}px")
        print("[INFO] 蛇形采样顺序已启用")

        return points

    # -----------------------------------------------------------------
    # 初始化
    # -----------------------------------------------------------------
    def initialize_camera(self):
        try:
            self.camera = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
            if not self.camera.isOpened():
                self.camera = cv2.VideoCapture(self.camera_id)
            if not self.camera.isOpened():
                raise RuntimeError(f"无法打开摄像头 {self.camera_id}")

            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 30)

            print(f"[INFO] 摄像头初始化成功 (camera_id={self.camera_id})")
            return True
        except Exception as e:
            print(f"[ERROR] 摄像头初始化失败: {e}")
            return False

    def initialize_model(self, model_cfg_path=None, ckpt_path=None):
        try:
            if model_cfg_path is None:
                model_cfg_path = os.path.join(_PROJECT_ROOT, 'gazemodel', 'config.yaml')
            if ckpt_path is None:
                ckpt_path = os.path.join(_PROJECT_ROOT, 'gazemodel', 'weights', 'unigaze_l16_joint.pth.tar')

            self.feature_extractor = GazeFeatureExtractor(
                model_cfg_path=model_cfg_path,
                ckpt_path=ckpt_path,
                device=self.device,
                camera_calib_path=self.camera_calib_path
            )

            print("[INFO] 模型初始化成功")
            print("[INFO] 主特征模式: 8D gaze + head pose + face center")
            print(f"[INFO] 真实相机内参: {'已加载' if self.feature_extractor.has_real_intrinsics else '未加载(使用dummy)'}")
            return True
        except Exception as e:
            print(f"[ERROR] 模型初始化失败: {e}")
            return False

    # -----------------------------------------------------------------
    # GUI
    # -----------------------------------------------------------------
    def create_gui(self):
        self.root = tk.Tk()
        self.root.title("注视点标定程序")
        self.root.attributes('-fullscreen', True)
        self.root.configure(bg='black')
        self.root.bind('<Escape>', self.quit_program)
        self.root.bind('<space>', self.collect_current_point)

        self.canvas = tk.Canvas(
            self.root,
            width=self.screen_width,
            height=self.screen_height,
            bg='black',
            highlightthickness=0
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.status_label = tk.Label(
            self.root,
            text="准备开始标定...",
            fg='white',
            bg='black',
            font=('Arial', 16)
        )
        self.status_label.place(x=50, y=50)

        self.progress_var = tk.StringVar()
        self.progress_label = tk.Label(
            self.root,
            textvariable=self.progress_var,
            fg='yellow',
            bg='black',
            font=('Arial', 14),
            justify=tk.LEFT
        )
        self.progress_label.place(x=50, y=100)

        instruction_text = (
            "按空格键开始采集当前点\n"
            "按ESC键退出程序\n"
            f"屏幕分辨率: {self.screen_width}x{self.screen_height}\n"
            f"标定点总数: {len(self.calibration_points)}\n"
            "蓝点 = 当前模型对当前视线的实时预测"
        )

        instruction_label = tk.Label(
            self.root,
            text=instruction_text,
            fg='cyan',
            bg='black',
            font=('Arial', 12),
            justify=tk.LEFT
        )
        instruction_label.place(x=50, y=self.screen_height - 170)

        self.update_display()
        self.root.after(33, self._display_loop)

    def _display_loop(self):
        if self.root is None:
            return
        self.update_display()
        if self.current_point_idx <= len(self.calibration_points):
            self.root.after(33, self._display_loop)

    def update_display(self):
        self.canvas.delete("all")

        if self.current_point_idx < len(self.calibration_points):
            x, y = self.calibration_points[self.current_point_idx]

            # 红点
            self.canvas.create_oval(
                x - 30, y - 30, x + 30, y + 30,
                fill='red', outline='white', width=3
            )
            self.canvas.create_oval(
                x - 10, y - 10, x + 10, y + 10,
                fill='white'
            )

            self.canvas.create_text(
                x, y - 50,
                text=f"点 {self.current_point_idx + 1}/{len(self.calibration_points)}",
                fill='white',
                font=('Arial', 16)
            )

            status = f"请注视红色目标点 ({x}, {y})"
            if self.preview_model_ready:
                status += " | 蓝点=实时预测"
            else:
                status += " | 蓝点暂未启用（样本不足）"
            self.status_label.config(text=status)

            progress_lines = [
                f"进度: {self.current_point_idx}/{len(self.calibration_points)}",
                f"已完成样本: {len(self.collected_data)}",
            ]

            if self.preview_model_ready and self.realtime_gaze_error_px is not None:
                progress_lines.append(f"实时预览误差: {self.realtime_gaze_error_px:.1f}px")

            self.progress_var.set("\n".join(progress_lines))

            if self.preview_model_ready and self.realtime_gaze_point is not None:
                pred_x, pred_y = self.realtime_gaze_point

                self.canvas.create_oval(
                    pred_x - 18, pred_y - 18, pred_x + 18, pred_y + 18,
                    fill='#2040FF', outline='cyan', width=2, stipple='gray50'
                )
                self.canvas.create_oval(
                    pred_x - 8, pred_y - 8, pred_x + 8, pred_y + 8,
                    fill='cyan', outline=''
                )

                self.canvas.create_line(x, y, pred_x, pred_y, fill='#66FFFF', width=2, dash=(4, 2))

                if self.realtime_gaze_error_px is not None:
                    self.canvas.create_text(
                        pred_x, pred_y - 28,
                        text=f"{self.realtime_gaze_error_px:.0f}px",
                        fill='cyan',
                        font=('Arial', 11)
                    )

        else:
            self.canvas.create_text(
                self.screen_width // 2, self.screen_height // 2,
                text="标定完成！\n正在计算映射参数...",
                fill='green',
                font=('Arial', 24),
                justify=tk.CENTER
            )
            self.status_label.config(text="标定数据采集完成")
            if not self.finish_scheduled:
                self.finish_scheduled = True
                self.root.after(1000, self.finish_calibration)

    # -----------------------------------------------------------------
    # 摄像头图像
    # -----------------------------------------------------------------
    def capture_face_image(self) -> Optional[np.ndarray]:
        if self.camera is None:
            return None

        with self.camera_lock:
            ret, frame = self.camera.read()

        if not ret:
            return None

        frame = cv2.flip(frame, 1)
        return frame

    # -----------------------------------------------------------------
    # 采集
    # -----------------------------------------------------------------
    def collect_current_point(self, event=None):
        if self.current_point_idx >= len(self.calibration_points):
            return

        if self.is_collecting:
            print("[INFO] 正在采集中，请等待...")
            return

        self.is_collecting = True
        self.thread_status_text = "准备采集中..."
        self.thread_progress_text = ""

        self.collection_thread = threading.Thread(target=self._collect_point_thread, daemon=True)
        self.collection_thread.start()

        self.check_collection_status()

    def check_collection_status(self):
        self.status_label.config(text=self.thread_status_text)
        self.progress_var.set(self.thread_progress_text)

        if self.collection_thread is not None and self.collection_thread.is_alive():
            self.root.after(100, self.check_collection_status)
        else:
            if self.is_collecting:
                self.is_collecting = False

                # 切换到新点时，清空旧蓝点状态，避免残影
                self.realtime_gaze_point = None
                self.realtime_gaze_error_px = None
                self.preview_smoothed_point = None
                self.preview_feature_buffer.clear()

                self.current_point_idx += 1
                self.update_display()

    def _check_sample_quality(self, samples: List[Dict]) -> Tuple[float, bool]:
        if len(samples) < 3:
            return 0.0, False

        features = np.array([s['features'] for s in samples], dtype=np.float32)
        feature_std = np.std(features, axis=0)

        gaze_std = float(np.mean(feature_std[:2])) if features.shape[1] >= 2 else float(np.mean(feature_std))
        head_std = float(np.mean(feature_std[2:5])) if features.shape[1] >= 5 else 0.0
        center_std = float(np.linalg.norm(feature_std[5:8])) if features.shape[1] >= 8 else 0.0

        gaze_score = max(0.0, 1.0 - gaze_std / 0.035)
        head_score = max(0.0, 1.0 - head_std / 0.06)
        center_score = max(0.0, 1.0 - center_std / 80.0)

        quality_score = 0.55 * gaze_score + 0.25 * head_score + 0.20 * center_score
        is_good = quality_score > 0.55
        return quality_score, is_good

    def _collect_point_thread(self):
        point_x, point_y = self.calibration_points[self.current_point_idx]
        self.thread_status_text = f"正在采集 ({point_x}, {point_y})... 请保持注视"

        warmup_end = time.time() + self.warmup_duration
        self.thread_progress_text = "预热中... 请注视标定点"
        while time.time() < warmup_end:
            _ = self.capture_face_image()
            time.sleep(0.05)

        current_point_samples = []
        sample_interval = self.collection_duration / self.samples_per_point
        quality_score = 0.0

        for i in range(self.samples_per_point):
            remaining = self.collection_duration - i * sample_interval
            face_image = self.capture_face_image()

            if face_image is not None:
                try:
                    features = self.feature_extractor.extract(face_image)
                    if features is not None:
                        feature_vector = features['x'].copy()

                        sample_data = {
                            'point_idx': self.current_point_idx,
                            'screen_x': point_x,
                            'screen_y': point_y,
                            'features': feature_vector,
                            'raw_gaze': features['yaw_pitch_cam'].copy(),
                            'gaze_cam': features['gaze_cam'].copy(),
                            'face_center_cam': features['face_center_cam'].copy(),
                            'head_euler': features['head_euler'].copy(),
                            'camera_model_source': features.get('debug', {}).get('camera_model_source', 'unknown'),
                            'timestamp': time.time(),
                        }
                        current_point_samples.append(sample_data)
                except Exception as e:
                    print(f"[WARN] 特征提取失败: {e}")

            if len(current_point_samples) >= 3:
                quality_score, is_good = self._check_sample_quality(current_point_samples)
                if i == 2 and not is_good:
                    self.thread_progress_text = (
                        f"⚠️ 检测到不稳定，请保持注视并尽量不要移动头部！质量: {quality_score:.2f}"
                    )
                    time.sleep(0.5)

            if len(current_point_samples) < 3:
                self.thread_progress_text = (
                    f"采集中... 剩余时间: {remaining:.1f}s ({i + 1}/{self.samples_per_point})"
                )
            else:
                self.thread_progress_text = (
                    f"采集中... 剩余: {remaining:.1f}s ({i + 1}/{self.samples_per_point}) 质量: {quality_score:.2f}"
                )

            if i < self.samples_per_point - 1:
                time.sleep(sample_interval)

        if current_point_samples:
            final_quality, is_good = self._check_sample_quality(current_point_samples)

            # 当前点采完后，再并入正式训练集
            self.collected_data.extend(current_point_samples)

            print(f"[INFO] 点 {self.current_point_idx + 1} 采集完成, 质量评分: {final_quality:.3f} {'✓' if is_good else '⚠️'}")

            if self._preview_ready_condition():
                self._update_temp_model()

            if not is_good and len(current_point_samples) < self.samples_per_point * 0.8:
                self.thread_progress_text = f"⚠️ 样本质量较低 ({final_quality:.2f})，建议重新采集该点"
                time.sleep(1.2)
            else:
                self.thread_progress_text = (
                    f"✓ 采集完成！质量: {final_quality:.2f} ({len(current_point_samples)}/{self.samples_per_point})"
                )

    # -----------------------------------------------------------------
    # 在线预览模型
    # -----------------------------------------------------------------
    def _preview_ready_condition(self) -> bool:
        completed_points = len(set(d['point_idx'] for d in self.collected_data))
        enough_points = completed_points >= self.preview_min_completed_points
        enough_samples = len(self.collected_data) >= self.preview_min_samples
        return enough_points and enough_samples

    def _update_temp_model(self):
        try:
            if not self._preview_ready_condition():
                return

            X_list = []
            Y_list = []

            x_min, y_min, x_max, y_max = self.calibration_region
            calib_w = x_max - x_min
            calib_h = y_max - y_min

            for sample in self.collected_data:
                X_list.append(sample['features'])
                norm_x = (sample['screen_x'] - x_min) / calib_w
                norm_y = (sample['screen_y'] - y_min) / calib_h
                Y_list.append([norm_x, norm_y])

            X = np.array(X_list, dtype=np.float32)
            Y = np.array(Y_list, dtype=np.float32)

            mean = X.mean(axis=0)
            std = X.std(axis=0) + 1e-6
            X_norm = (X - mean) / std

            poly = PolynomialFeatures(degree=self.temp_poly_degree, include_bias=False)
            X_poly = poly.fit_transform(X_norm)

            model_x = Ridge(alpha=1.0).fit(X_poly, Y[:, 0])
            model_y = Ridge(alpha=1.0).fit(X_poly, Y[:, 1])

            with self.temp_model_lock:
                self.temp_model_x = model_x
                self.temp_model_y = model_y
                self.temp_mean = mean
                self.temp_std = std
                self.preview_model_ready = True

            completed_points = len(set(d['point_idx'] for d in self.collected_data))
            print(f"[INFO] 临时预览模型已更新: completed_points={completed_points}, samples={len(self.collected_data)}")

        except Exception as e:
            print(f"[WARN] 临时模型训练失败: {e}")

    def _predict_temp(self, features) -> Optional[Tuple[float, float]]:
        with self.temp_model_lock:
            if self.temp_model_x is None or self.temp_model_y is None:
                return None
            model_x = self.temp_model_x
            model_y = self.temp_model_y
            mean = self.temp_mean.copy()
            std = self.temp_std.copy()

        try:
            x = np.asarray(features['x'], dtype=np.float32).reshape(-1)

            x_norm = (x - mean) / std
            poly = PolynomialFeatures(degree=self.temp_poly_degree, include_bias=False)
            x_poly = poly.fit_transform(x_norm.reshape(1, -1))

            u_norm = model_x.predict(x_poly)[0]
            v_norm = model_y.predict(x_poly)[0]

            x_min, y_min, x_max, y_max = self.calibration_region
            u = u_norm * (x_max - x_min) + x_min
            v = v_norm * (y_max - y_min) + y_min

            u = float(np.clip(u, 0, self.screen_width - 1))
            v = float(np.clip(v, 0, self.screen_height - 1))
            return (u, v)
        except Exception:
            return None

    def _is_preview_input_stable(self) -> bool:
        if len(self.preview_feature_buffer) < self.preview_feature_buffer.maxlen:
            return False

        arr = np.array(self.preview_feature_buffer, dtype=np.float32)
        feature_std = np.std(arr, axis=0)
        gaze_std = float(np.mean(feature_std[:2])) if arr.shape[1] >= 2 else float(np.mean(feature_std))
        head_std = float(np.mean(feature_std[2:5])) if arr.shape[1] >= 5 else 0.0
        center_std = float(np.linalg.norm(feature_std[5:8])) if arr.shape[1] >= 8 else 0.0
        return gaze_std < 0.06 and head_std < 0.08 and center_std < 120.0

    def _smooth_preview_point(self, new_point: Tuple[float, float]) -> Tuple[float, float]:
        if self.preview_smoothed_point is None:
            self.preview_smoothed_point = new_point
            return new_point

        old_u, old_v = self.preview_smoothed_point
        new_u, new_v = new_point
        a = self.preview_smooth_alpha

        smooth_u = a * new_u + (1 - a) * old_u
        smooth_v = a * new_v + (1 - a) * old_v
        self.preview_smoothed_point = (smooth_u, smooth_v)
        return self.preview_smoothed_point

    def _realtime_predict_loop(self):
        while self.realtime_running:
            try:
                if not self.preview_model_ready or self.camera is None:
                    time.sleep(0.05)
                    continue

                frame = self.capture_face_image()
                if frame is None:
                    time.sleep(0.05)
                    continue

                features = self.feature_extractor.extract(frame)
                if features is None:
                    time.sleep(0.05)
                    continue

                feature_vector = np.asarray(features['x'], dtype=np.float32).reshape(-1)
                self.preview_feature_buffer.append(feature_vector)

                if not self._is_preview_input_stable():
                    self.realtime_gaze_error_px = None
                    time.sleep(0.05)
                    continue

                pred = self._predict_temp(features)
                if pred is not None:
                    pred = self._smooth_preview_point(pred)
                    self.realtime_gaze_point = pred

                    if self.current_point_idx < len(self.calibration_points):
                        tx, ty = self.calibration_points[self.current_point_idx]
                        self.realtime_gaze_error_px = float(
                            np.sqrt((pred[0] - tx) ** 2 + (pred[1] - ty) ** 2)
                        )

                time.sleep(0.05)

            except Exception:
                time.sleep(0.1)

    # -----------------------------------------------------------------
    # 标定主计算
    # -----------------------------------------------------------------
    def _filter_outliers(self, X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        point_indices = np.array([sample['point_idx'] for sample in self.collected_data])
        mask = np.ones(len(X), dtype=bool)

        for point_idx in np.unique(point_indices):
            point_mask = (point_indices == point_idx)
            point_features = X[point_mask]

            if len(point_features) < 5:
                continue

            center = np.mean(point_features, axis=0)
            distances = np.linalg.norm(point_features - center, axis=1)

            q1, q3 = np.percentile(distances, [25, 75])
            iqr = q3 - q1
            threshold = q3 + 1.5 * iqr

            outliers = distances > threshold
            global_indices = np.where(point_mask)[0]
            mask[global_indices[outliers]] = False

            if np.any(outliers):
                print(f"[INFO] 点 {point_idx}: 过滤了 {np.sum(outliers)} 个异常样本")

        filtered_X = X[mask]
        filtered_Y = Y[mask]
        print(f"[INFO] 异常值过滤: {len(X)} -> {len(filtered_X)} 样本")
        return filtered_X, filtered_Y, mask

    def _compute_calibration_mapping(self) -> Dict[str, Any]:
        try:
            features_list = []
            coordinates_list = []
            point_indices = []

            x_min, y_min, x_max, y_max = self.calibration_region
            calib_width = x_max - x_min
            calib_height = y_max - y_min

            print(f"[INFO] 使用标定区域: ({x_min}, {y_min}, {x_max}, {y_max})")
            print(f"[INFO] 标定区域尺寸: {calib_width} x {calib_height}")

            for sample in self.collected_data:
                features_list.append(sample['features'])
                norm_x = (sample['screen_x'] - x_min) / calib_width
                norm_y = (sample['screen_y'] - y_min) / calib_height
                coordinates_list.append([norm_x, norm_y])
                point_indices.append(sample['point_idx'])

            X = np.array(features_list, dtype=np.float32)
            Y = np.array(coordinates_list, dtype=np.float32)
            point_indices = np.array(point_indices)

            print("\n[INFO] ===== 开始标定计算 =====")
            feature_names = self.feature_extractor.feature_names() if self.feature_extractor else []
            print(f"[INFO] 特征模式: 8D gaze + head pose + face center")
            print(f"[INFO] 特征名称: {feature_names}")
            print(f"[INFO] 原始数据: 特征维度={X.shape[1]}, 样本数={X.shape[0]}")

            X_filtered, Y_filtered, valid_mask = self._filter_outliers(X, Y)
            point_indices_filtered = point_indices[valid_mask]

            mean = np.mean(X_filtered, axis=0)
            std = np.std(X_filtered, axis=0) + 1e-6
            X_normalized = (X_filtered - mean) / std

            pixel_coords = np.zeros_like(Y_filtered)
            pixel_coords[:, 0] = Y_filtered[:, 0] * calib_width + x_min
            pixel_coords[:, 1] = Y_filtered[:, 1] * calib_height + y_min

            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            distances = np.sqrt((pixel_coords[:, 0] - center_x) ** 2 + (pixel_coords[:, 1] - center_y) ** 2)
            max_distance = np.sqrt((calib_width / 2) ** 2 + (calib_height / 2) ** 2)
            distance_ratios = distances / max_distance
            edge_weights = 1 + 0.6 * distance_ratios

            best_degree = 1
            best_alpha = 1.0
            best_score = -np.inf

            print("[INFO] 开始模型选择（按标定点分组交叉验证）...")

            for degree in range(1, 4):
                for alpha in [0.05, 0.1, 0.5, 1.0, 5.0, 10.0]:
                    poly = PolynomialFeatures(degree=degree, include_bias=False)
                    X_poly = poly.fit_transform(X_normalized)

                    try:
                        from sklearn.model_selection import GroupKFold, KFold

                        unique_groups = np.unique(point_indices_filtered)
                        if len(unique_groups) >= 2:
                            n_splits = min(5, len(unique_groups))
                            splitter = GroupKFold(n_splits=n_splits)
                            split_iter = splitter.split(X_poly, Y_filtered, groups=point_indices_filtered)
                        else:
                            n_splits = min(3, max(2, len(Y_filtered) // 10))
                            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)
                            split_iter = splitter.split(X_poly)

                        scores = []
                        for train_idx, val_idx in split_iter:
                            model_x_cv = Ridge(alpha=alpha)
                            model_y_cv = Ridge(alpha=alpha)
                            model_x_cv.fit(X_poly[train_idx], Y_filtered[train_idx, 0],
                                           sample_weight=edge_weights[train_idx])
                            model_y_cv.fit(X_poly[train_idx], Y_filtered[train_idx, 1],
                                           sample_weight=edge_weights[train_idx])

                            pred_x = model_x_cv.predict(X_poly[val_idx])
                            pred_y = model_y_cv.predict(X_poly[val_idx])
                            mse_x = np.mean((pred_x - Y_filtered[val_idx, 0]) ** 2)
                            mse_y = np.mean((pred_y - Y_filtered[val_idx, 1]) ** 2)
                            scores.append(-(mse_x + mse_y) / 2)

                        score = float(np.mean(scores))
                    except Exception as e:
                        print(f"[WARN] 模型评估出错 (degree={degree}, alpha={alpha}): {e}")
                        continue

                    if score > best_score:
                        best_score = score
                        best_degree = degree
                        best_alpha = alpha

            print(f"[INFO] 最佳模型: 多项式阶数={best_degree}, α={best_alpha}")

            poly = PolynomialFeatures(degree=best_degree, include_bias=False)
            X_poly = poly.fit_transform(X_normalized)

            model_x = Ridge(alpha=best_alpha)
            model_y = Ridge(alpha=best_alpha)

            model_x.fit(X_poly, Y_filtered[:, 0], sample_weight=edge_weights)
            model_y.fit(X_poly, Y_filtered[:, 1], sample_weight=edge_weights)

            Y_pred_x = model_x.predict(X_poly)
            Y_pred_y = model_y.predict(X_poly)
            Y_pred = np.column_stack((Y_pred_x, Y_pred_y))

            Y_pred_pixels = Y_pred * np.array([calib_width, calib_height]) + np.array([x_min, y_min])
            Y_true_pixels = Y_filtered * np.array([calib_width, calib_height]) + np.array([x_min, y_min])

            errors = np.linalg.norm(Y_pred_pixels - Y_true_pixels, axis=1)
            mean_error = float(np.mean(errors))
            std_error = float(np.std(errors))
            max_error = float(np.max(errors))
            median_error = float(np.median(errors))

            print("\n[INFO] ===== 标定误差统计 =====")
            print(f"[INFO] 平均误差: {mean_error:.2f}px")
            print(f"[INFO] 中位数误差: {median_error:.2f}px")
            print(f"[INFO] 标准差: {std_error:.2f}px")
            print(f"[INFO] 最大误差: {max_error:.2f}px")

            point_errors = defaultdict(list)
            point_offsets = defaultdict(list)
            filtered_sample_indices = np.where(valid_mask)[0]

            for i, sample_idx in enumerate(filtered_sample_indices):
                sample = self.collected_data[sample_idx]
                point_errors[sample['point_idx']].append(errors[i])
                point_offsets[sample['point_idx']].append(
                    (Y_pred_pixels[i] - Y_true_pixels[i]).tolist()
                )

            point_stats = {}
            for point_idx, point_errs in point_errors.items():
                x, y = self.calibration_points[point_idx]
                offsets_arr = np.array(point_offsets[point_idx])

                point_stats[f"point_{point_idx}"] = {
                    'screen_coord': [x, y],
                    'mean_error': float(np.mean(point_errs)),
                    'std_error': float(np.std(point_errs)),
                    'sample_count': len(point_errs),
                    'mean_offset': offsets_arr.mean(axis=0).tolist(),
                }

            result = {
                'version': '3.0_camera_calib_8d',
                'calibration_method': 'ridge_regression',
                'has_real_camera_intrinsics': getattr(self.feature_extractor, 'has_real_intrinsics', False),
                'camera_calib_path': self.camera_calib_path,
                'calibration_margin_ratio': float(self.calibration_margin_ratio),

                'mean': mean.tolist(),
                'std': std.tolist(),
                'normalization_strategy': 'standard_zscore',
                'poly_degree': best_degree,
                'ridge_alpha': best_alpha,
                'poly_features': poly.get_feature_names_out().tolist(),
                'coef_x': model_x.coef_.tolist(),
                'intercept_x': float(model_x.intercept_),
                'coef_y': model_y.coef_.tolist(),
                'intercept_y': float(model_y.intercept_),

                'legacy_ridge': {
                    'mean': mean.tolist(),
                    'std': std.tolist(),
                    'normalization_strategy': 'standard_zscore',
                    'poly_degree': best_degree,
                    'ridge_alpha': best_alpha,
                    'coef_x': model_x.coef_.tolist(),
                    'intercept_x': float(model_x.intercept_),
                    'coef_y': model_y.coef_.tolist(),
                    'intercept_y': float(model_y.intercept_),
                },

                'screen': {
                    'width': self.screen_width,
                    'height': self.screen_height
                },
                'calibration_region': {
                    'x_min': int(x_min),
                    'y_min': int(y_min),
                    'x_max': int(x_max),
                    'y_max': int(y_max),
                    'width': int(calib_width),
                    'height': int(calib_height)
                },
                'calibration_points': self.calibration_points,
                'statistics': {
                    'total_samples': len(self.collected_data),
                    'valid_samples': len(X_filtered),
                    'filtered_samples': len(X) - len(X_filtered),
                    'mean_error_px': mean_error,
                    'median_error_px': median_error,
                    'std_error_px': std_error,
                    'max_error_px': max_error,
                    'per_sample_errors': errors.tolist(),
                    'point_stats': point_stats
                },
                'timestamp': datetime.now().isoformat(),
                'feature_dimension': int(X.shape[1]),
                'feature_mode': 'gaze_head_face_center_8d',
                'feature_names': feature_names,
                'grid_size': [self.grid_cols, self.grid_rows],
                'screen_x_flipped': False,
                'offset_x': 0.0,
                'offset_y': 0.0,
            }

            return result

        except Exception as e:
            raise RuntimeError(f"标定计算失败: {e}")

    # -----------------------------------------------------------------
    # 完成标定
    # -----------------------------------------------------------------
    def finish_calibration(self):
        expected_points = len(self.calibration_points)
        min_samples_required = expected_points * 5

        print("\n[INFO] 标定数据统计:")
        print(f"  - 总样本数: {len(self.collected_data)}")
        print(f"  - 覆盖的标定点: {len(set(d['point_idx'] for d in self.collected_data))}/{expected_points}")
        print(f"  - 最小要求: {min_samples_required} 个样本")

        if len(self.collected_data) < min_samples_required:
            error_msg = (
                f"标定数据不足！\n\n"
                f"当前样本数: {len(self.collected_data)}\n"
                f"需要至少: {min_samples_required} 个样本\n"
                f"覆盖的标定点: {len(set(d['point_idx'] for d in self.collected_data))}/{expected_points}\n\n"
                f"请重新进行标定。"
            )
            messagebox.showerror("错误", error_msg)
            print(f"[ERROR] {error_msg}")
            return

        try:
            print("\n[INFO] 使用 Ridge 回归标定方法...")
            calibration_result = self._compute_calibration_mapping()

            self._save_calibration_data(calibration_result, self.output_dir)

            messagebox.showinfo(
                "成功",
                f"标定完成！\n"
                f"方法: Ridge回归\n"
                f"总样本数: {len(self.collected_data)}\n"
                f"有效标定点: {len(set(d['point_idx'] for d in self.collected_data))}\n"
                f"标定文件已保存为 calib.json"
            )

            self._visualize_calibration_results(calibration_result)
            self._generate_publication_figures(calibration_result)

        except Exception as e:
            messagebox.showerror("错误", f"标定计算失败: {e}")
            import traceback
            traceback.print_exc()

        self.quit_program()

    # -----------------------------------------------------------------
    # 可视化
    # -----------------------------------------------------------------
    def _visualize_calibration_results(self, mapping_params: Dict[str, Any]) -> None:
        if not MATPLOTLIB_AVAILABLE:
            print("[INFO] matplotlib 不可用，跳过结果可视化")
            return

        try:
            viz_window = tk.Toplevel(self.root)
            viz_window.title("标定结果可视化")
            viz_window.geometry("1000x700")
            viz_window.attributes('-topmost', True)

            fig = plt.Figure(figsize=(12, 10), dpi=80)

            ax1 = fig.add_subplot(221)
            self._plot_error_distribution(ax1, mapping_params)
            ax1.set_title('各标定点的误差分布')
            ax1.set_xlabel('X坐标')
            ax1.set_ylabel('Y坐标')
            ax1.grid(True, alpha=0.3)

            ax2 = fig.add_subplot(222)
            self._plot_edge_vs_center_comparison(ax2, mapping_params)
            ax2.set_title('边缘点vs中心点误差对比')
            ax2.set_ylabel('平均误差 (像素)')
            ax2.grid(True, alpha=0.3)

            ax3 = fig.add_subplot(212)
            self._plot_error_histogram(ax3, mapping_params)
            ax3.set_title('标定误差直方图')
            ax3.set_xlabel('误差 (像素)')
            ax3.set_ylabel('频率')
            ax3.grid(True, alpha=0.3)

            fig.tight_layout(pad=3.0)

            canvas = FigureCanvasTkAgg(fig, master=viz_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            self._save_visualization_figure(fig, mapping_params)

            close_button = ttk.Button(viz_window, text="关闭", command=viz_window.destroy)
            close_button.pack(pady=10)

        except Exception as e:
            print(f"[ERROR] 创建可视化时出错: {e}")
            messagebox.showerror("可视化错误", f"无法创建标定结果可视化: {e}")

    def _plot_error_distribution(self, ax, mapping_params: Dict[str, Any]) -> None:
        point_stats = mapping_params['statistics']['point_stats']
        screen_width = mapping_params['screen']['width']
        screen_height = mapping_params['screen']['height']

        x_coords = []
        y_coords = []
        errors = []

        for _, stats in point_stats.items():
            x, y = stats['screen_coord']
            error = stats['mean_error']
            matplotlib_y = screen_height - y
            x_coords.append(x)
            y_coords.append(matplotlib_y)
            errors.append(error)

        scatter = ax.scatter(
            x_coords, y_coords, c=errors, cmap='viridis',
            s=100, alpha=0.8, edgecolors='k', linewidths=0.5,
            vmin=0, vmax=max(errors) if errors else 10
        )
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('平均误差 (像素)')
        ax.set_xlim(0, screen_width)
        ax.set_ylim(0, screen_height)

    def _plot_edge_vs_center_comparison(self, ax, mapping_params: Dict[str, Any]) -> None:
        point_stats = mapping_params['statistics']['point_stats']
        grid_size = mapping_params.get('grid_size', [4, 4])
        cols, _ = grid_size

        row_errors = defaultdict(list)
        for i, (_, stats) in enumerate(sorted(point_stats.items())):
            row_idx = i // cols
            row_errors[row_idx].append(stats['mean_error'])

        row_means = [np.mean(row_errors[r]) for r in sorted(row_errors)]
        ax.bar(range(len(row_means)), row_means, alpha=0.7, color='#4878CF')
        ax.set_xlabel('行')
        ax.set_ylabel('平均误差 (像素)')

        overall_mean = mapping_params['statistics']['mean_error_px']
        ax.axhline(y=overall_mean, color='r', linestyle='--', label=f'整体平均: {overall_mean:.1f}px')
        ax.legend()

    def _plot_error_histogram(self, ax, mapping_params: Dict[str, Any]) -> None:
        per_sample = mapping_params['statistics'].get('per_sample_errors')
        if per_sample:
            errors = per_sample
        else:
            errors = [s['mean_error'] for s in mapping_params['statistics']['point_stats'].values()]

        ax.hist(errors, bins=15, alpha=0.7, color='#4878CF', edgecolor='white', density=False)

        mean_error = mapping_params['statistics']['mean_error_px']
        std_error = mapping_params['statistics']['std_error_px']
        max_error = mapping_params['statistics']['max_error_px']
        median_error = mapping_params['statistics']['median_error_px']

        stats_text = (
            f'平均误差: {mean_error:.2f}px\n'
            f'中位数: {median_error:.2f}px\n'
            f'标准差: {std_error:.2f}px\n'
            f'最大误差: {max_error:.2f}px'
        )

        ax.text(
            0.95, 0.95, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

        ax.axvline(x=mean_error, color='r', linestyle='--', label='平均误差')
        ax.axvline(x=median_error, color='blue', linestyle='--', label='中位数')
        ax.legend()

    def _save_visualization_figure(self, fig, mapping_params: Dict[str, Any]) -> None:
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_dir = self.output_dir or os.getcwd()
            os.makedirs(save_dir, exist_ok=True)

            png_path = os.path.join(save_dir, f'calibration_visualization_{timestamp}.png')
            pdf_path = os.path.join(save_dir, f'calibration_visualization_{timestamp}.pdf')

            fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
            fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')

            print("[INFO] 可视化图表已保存:")
            print(f"  - PNG格式: {png_path}")
            print(f"  - PDF格式: {pdf_path}")

        except Exception as e:
            print(f"[WARN] 保存可视化图表失败: {e}")

    # -----------------------------------------------------------------
    # 保存
    # -----------------------------------------------------------------
    def _save_calibration_data(self, calibration_result: Dict[str, Any], output_dir=None):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        if output_dir is None:
            output_dir = os.path.join(os.getcwd(), f'gaze_dataset_{timestamp}')

        os.makedirs(output_dir, exist_ok=True)

        calib_path = os.path.join(output_dir, 'calib.json')
        with open(calib_path, 'w', encoding='utf-8') as f:
            json.dump(calibration_result, f, ensure_ascii=False, indent=2)

        detailed_data = {
            'calibration_result': calibration_result,
            'raw_samples': []
        }

        for sample in self.collected_data:
            sample_record = {
                'point_idx': sample['point_idx'],
                'screen_x': sample['screen_x'],
                'screen_y': sample['screen_y'],
                'features': sample['features'].tolist() if hasattr(sample['features'], 'tolist') else list(sample['features']),
                'timestamp': sample['timestamp']
            }

            for key in ('raw_gaze', 'gaze_cam', 'face_center_cam', 'head_euler', 'camera_model_source'):
                if key in sample and sample[key] is not None:
                    val = sample[key]
                    sample_record[key] = val.tolist() if hasattr(val, 'tolist') else val

            detailed_data['raw_samples'].append(sample_record)

        detailed_path = os.path.join(output_dir, f'calibration_detailed_{timestamp}.json')
        with open(detailed_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_data, f, ensure_ascii=False, indent=2)

        print("[INFO] 标定文件已保存:")
        print(f"  - 主文件: {calib_path}")
        print(f"  - 详细数据: {detailed_path}")

    def _generate_publication_figures(self, calibration_result: Dict[str, Any]):
        try:
            import subprocess

            script_path = os.path.join(_CUR_DIR, 'visualize_calibration.py')
            if not os.path.isfile(script_path):
                print(f"[WARN] 可视化脚本不存在: {script_path}")
                return

            output_dir = self.output_dir or os.getcwd()
            calib_path = os.path.join(output_dir, 'calib.json')
            fig_dir = os.path.join(output_dir, 'figures')

            if os.path.isfile(calib_path):
                cmd = [sys.executable, script_path, '--calib', calib_path, '--output', fig_dir]
                print(f"[INFO] 生成论文级图表: {' '.join(cmd)}")
                subprocess.run(cmd, check=True)
            else:
                print("[WARN] calib.json 未找到，跳过论文级图表生成")
        except Exception as e:
            print(f"[WARN] 论文级图表生成失败: {e}")

    # -----------------------------------------------------------------
    # 退出
    # -----------------------------------------------------------------
    def quit_program(self, event=None):
        self.realtime_running = False

        if self.camera is not None:
            with self.camera_lock:
                self.camera.release()

        if self.root is not None:
            self.root.quit()
            self.root.destroy()

    # -----------------------------------------------------------------
    # 主运行
    # -----------------------------------------------------------------
    def run(self):
        print("=== 注视点标定程序 ===")
        print(f"屏幕分辨率: {self.screen_width} x {self.screen_height}")
        print(f"标定点网格: {self.grid_cols} x {self.grid_rows} = {len(self.calibration_points)} 个点")

        if not self.initialize_camera():
            print("[ERROR] 摄像头初始化失败，程序退出")
            return False

        if not self.initialize_model():
            print("[ERROR] 模型初始化失败，程序退出")
            return False

        self.create_gui()

        self.realtime_running = True
        self.realtime_thread = threading.Thread(target=self._realtime_predict_loop, daemon=True)
        self.realtime_thread.start()

        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            print("\n[INFO] 用户中断程序")
        except Exception as e:
            print(f"[ERROR] 程序运行错误: {e}")
        finally:
            if self.camera is not None:
                with self.camera_lock:
                    self.camera.release()

        return True


def main():
    import argparse

    if sys.platform == 'win32':
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
        os.environ['OPENCV_OPENCL_RUNTIME'] = ''

    parser = argparse.ArgumentParser(description='注视点标定程序')
    parser.add_argument('--screen-width', type=int, default=2194, help='屏幕宽度')
    parser.add_argument('--screen-height', type=int, default=1234, help='屏幕高度')
    parser.add_argument('--grid-cols', type=int, default=4, help='标定点网格列数')
    parser.add_argument('--grid-rows', type=int, default=4, help='标定点网格行数')
    parser.add_argument('--grid-mode', type=str, default=None, choices=['quick', 'balanced', 'thorough'],
                        help='标定网格模式: quick(3x3), balanced(4x4), thorough(5x5)')
    parser.add_argument('--camera-id', type=int, default=0, help='摄像头ID')
    parser.add_argument('--output-dir', type=str, default=None, help='标定结果输出目录')
    parser.add_argument('--calibration-region', type=str, default='0,0,2194,1234',
                        help='标定区域，格式: x_min,y_min,x_max,y_max')
    parser.add_argument('--calibration-margin-ratio', type=float, default=0.10,
                        help='标定点距离标定区域边缘的比例，默认0.10')
    parser.add_argument('--camera-calib', type=str, default='camera_calib.json',
                        help='OpenCV相机内参JSON文件路径')

    args = parser.parse_args()

    calibration_region = None
    if args.calibration_region:
        try:
            parts = [int(x.strip()) for x in args.calibration_region.split(',')]
            if len(parts) == 4:
                calibration_region = tuple(parts)
                print(f"[INFO] 使用自定义标定区域: {calibration_region}")
        except ValueError:
            print("[WARN] 标定区域解析失败，使用默认值")

    calibrator = CalibrationProgram(
        screen_width=args.screen_width,
        screen_height=args.screen_height,
        grid_size=(args.grid_cols, args.grid_rows),
        output_dir=args.output_dir,
        calibration_region=calibration_region,
        grid_mode=args.grid_mode,
        camera_id=args.camera_id,
        camera_calib_path=args.camera_calib,
        calibration_margin_ratio=args.calibration_margin_ratio
    )

    print("[INFO] 标定方法: Ridge回归 (8D gaze + head pose + face center)")
    print("[INFO] 蓝点预览策略: 基于已完成点训练，当前点实时泛化预测")

    success = calibrator.run()

    if success:
        print("[INFO] 标定程序正常结束")
        return 0
    else:
        print("[ERROR] 标定程序异常结束")
        return 1


if __name__ == '__main__':
    sys.exit(main())
