#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
相机内参标定工具
使用棋盘格图案标定相机真实内参，替代虚假的 focal = w * 4.0

用法:
  GUI模式:  python camera_calibration.py
  CLI参数:  python camera_calibration.py --camera-id 0 --board-size 9x6 --square-size 25
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 默认参数
# ---------------------------------------------------------------------------
DEFAULT_BOARD_COLS = 8       # 棋盘格内角点列数
DEFAULT_BOARD_ROWS = 5       # 棋盘格内角点行数
DEFAULT_SQUARE_SIZE = 27.0   # 棋盘格方块边长 (mm)
MIN_CAPTURES = 15            # 最少拍摄张数
RECOMMENDED_CAPTURES = 20    # 推荐拍摄张数
MAX_CAPTURES = 40            # 最大拍摄张数


def calibrate_camera_from_images(image_points_list, board_size, square_size_mm, image_size):
    """
    根据已检测到的角点进行相机标定。

    Args:
        image_points_list: list of (N, 1, 2) ndarray - 每张图的角点像素坐标
        board_size: (cols, rows) 内角点数
        square_size_mm: float - 方块实际边长 (mm)
        image_size: (width, height)

    Returns:
        dict: 标定结果 或 None (失败)
    """
    cols, rows = board_size
    # 构造世界坐标 (z=0 平面)
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size_mm

    obj_points = [objp for _ in image_points_list]

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, image_points_list, image_size, None, None
    )

    if not ret:
        return None

    # 计算重投影误差
    total_error = 0
    total_points = 0
    per_image_errors = []
    for i in range(len(obj_points)):
        projected, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i],
                                          camera_matrix, dist_coeffs)
        error = cv2.norm(image_points_list[i], projected, cv2.NORM_L2)
        n = len(obj_points[i])
        per_image_errors.append(error / n)
        total_error += error * error
        total_points += n
    rms_error = np.sqrt(total_error / total_points)
    mean_error = float(np.mean(per_image_errors))

    w, h = image_size
    result = {
        'camera_matrix': camera_matrix.tolist(),
        'dist_coeffs': dist_coeffs.flatten().tolist(),
        'image_size': [w, h],
        'reprojection_error': round(mean_error, 4),
        'rms_error': round(rms_error, 4),
        'calibration_date': datetime.now().isoformat(),
        'checkerboard_size': [cols, rows],
        'square_size_mm': square_size_mm,
        'num_images_used': len(image_points_list),
    }
    return result


def save_calibration(result, output_path):
    """保存标定结果到 JSON 文件。"""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 标定结果已保存: {output_path}")


# =========================================================================
# GUI 模式
# =========================================================================
class CameraCalibrationGUI:
    """tkinter GUI：实时预览、空格拍照、c 开始计算标定。"""

    def __init__(self, camera_id=0, board_size=(DEFAULT_BOARD_COLS, DEFAULT_BOARD_ROWS),
                 square_size=DEFAULT_SQUARE_SIZE, output_path=None):
        import tkinter as tk
        self.tk = tk

        self.camera_id = camera_id
        self.board_size = board_size
        self.square_size = square_size
        self.output_path = output_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'camera_calib.json')

        self.captures = []          # list of (corners, image_size)
        self.last_frame = None
        self.running = True

        # 打开摄像头
        self.cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开摄像头 {camera_id}")

        # 读取一帧获取尺寸
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("无法读取摄像头帧")
        self.img_h, self.img_w = frame.shape[:2]

        # 创建窗口
        self.root = tk.Tk()
        self.root.title("相机内参标定")
        self.root.configure(bg='black')

        # 状态标签
        self.status_var = tk.StringVar(value=self._status_text())
        self.status_label = tk.Label(self.root, textvariable=self.status_var,
                                      font=('Consolas', 14), bg='black', fg='white',
                                      anchor='w', justify='left')
        self.status_label.pack(side='top', fill='x', padx=10, pady=5)

        # 帮助标签
        help_text = ("操作: [空格] 拍照  |  [c] 开始计算  |  [u] 撤销上张  |  [ESC] 退出\n"
                     f"棋盘格: {self.board_size[0]}x{self.board_size[1]}, "
                     f"方格边长: {self.square_size}mm")
        self.help_label = tk.Label(self.root, text=help_text,
                                    font=('Consolas', 11), bg='black', fg='#aaaaaa',
                                    anchor='w', justify='left')
        self.help_label.pack(side='top', fill='x', padx=10)

        # 画布 - 用于显示视频
        canvas_w = min(self.img_w, 960)
        canvas_h = int(canvas_w * self.img_h / self.img_w)
        self.canvas_size = (canvas_w, canvas_h)
        self.canvas = tk.Canvas(self.root, width=canvas_w, height=canvas_h, bg='black')
        self.canvas.pack(padx=10, pady=10)

        # 绑定按键
        self.root.bind('<space>', self._on_capture)
        self.root.bind('<Escape>', self._on_quit)
        self.root.bind('c', self._on_calibrate)
        self.root.bind('C', self._on_calibrate)
        self.root.bind('u', self._on_undo)
        self.root.bind('U', self._on_undo)

        # 启动视频循环
        self._update_frame()

    def _status_text(self):
        n = len(self.captures)
        if n < MIN_CAPTURES:
            return f"已拍摄: {n}/{MIN_CAPTURES} (最少需要{MIN_CAPTURES}张，建议{RECOMMENDED_CAPTURES}张)"
        elif n < RECOMMENDED_CAPTURES:
            return f"已拍摄: {n} (已达标！建议继续拍到{RECOMMENDED_CAPTURES}张。按 c 开始标定)"
        else:
            return f"已拍摄: {n} (充足！按 c 开始标定)"

    def _update_frame(self):
        if not self.running:
            return
        ret, frame = self.cap.read()
        if ret:
            self.last_frame = frame.copy()
            # 检测棋盘格角点并绘制
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(
                gray, self.board_size,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
            )
            if found:
                cv2.drawChessboardCorners(frame, self.board_size, corners, found)
                # 绿色指示
                cv2.putText(frame, "Checkerboard DETECTED", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "No checkerboard", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            # 缩放到画布大小
            display = cv2.resize(frame, self.canvas_size)
            display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)

            from PIL import Image, ImageTk
            img_pil = Image.fromarray(display_rgb)
            self._photo = ImageTk.PhotoImage(img_pil)
            self.canvas.create_image(0, 0, anchor='nw', image=self._photo)

        self.root.after(30, self._update_frame)

    def _on_capture(self, event=None):
        if self.last_frame is None:
            return
        if len(self.captures) >= MAX_CAPTURES:
            self.status_var.set(f"已达最大拍摄数 {MAX_CAPTURES}，请按 c 开始标定")
            return

        gray = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray, self.board_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        if not found:
            self.status_var.set("未检测到棋盘格，请调整角度后重试！")
            return

        # 亚像素精细化
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        h, w = gray.shape
        self.captures.append((corners_refined, (w, h)))
        self.status_var.set(self._status_text())
        print(f"[INFO] 拍摄第 {len(self.captures)} 张，角点已精细化")

    def _on_undo(self, event=None):
        if self.captures:
            self.captures.pop()
            self.status_var.set(self._status_text() + "  (已撤销上一张)")
            print(f"[INFO] 撤销，剩余 {len(self.captures)} 张")

    def _on_calibrate(self, event=None):
        if len(self.captures) < MIN_CAPTURES:
            self.status_var.set(f"拍摄数不足！当前 {len(self.captures)}/{MIN_CAPTURES}")
            return

        self.status_var.set("正在计算标定参数... 请稍候")
        self.root.update()

        corners_list = [c for c, _ in self.captures]
        _, image_size = self.captures[0]

        result = calibrate_camera_from_images(
            corners_list, self.board_size, self.square_size, image_size
        )
        if result is None:
            self.status_var.set("标定失败！请拍摄更多不同角度的图片后重试")
            return

        save_calibration(result, self.output_path)

        fx = result['camera_matrix'][0][0]
        fy = result['camera_matrix'][1][1]
        cx = result['camera_matrix'][0][2]
        cy = result['camera_matrix'][1][2]
        err = result['reprojection_error']

        summary = (f"标定完成！重投影误差: {err:.4f}px\n"
                   f"焦距: fx={fx:.1f}, fy={fy:.1f}\n"
                   f"主点: cx={cx:.1f}, cy={cy:.1f}\n"
                   f"已保存: {self.output_path}")
        self.status_var.set(summary)
        print(f"\n{'='*50}")
        print(f"[INFO] {summary}")
        print(f"{'='*50}")

        if err > 0.5:
            print("[WARN] 重投影误差 > 0.5px，建议重新标定（拍摄更多角度/改善光照）")

    def _on_quit(self, event=None):
        self.running = False
        self.cap.release()
        self.root.quit()
        self.root.destroy()

    def run(self):
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            pass
        finally:
            self.running = False
            if self.cap.isOpened():
                self.cap.release()


# =========================================================================
# 加载标定结果的工具函数 (供其他模块使用)
# =========================================================================
def load_camera_calibration(calib_path=None):
    """
    加载相机标定参数。

    Args:
        calib_path: JSON 文件路径。为 None 时尝试默认路径。

    Returns:
        (camera_matrix, dist_coeffs, image_size) 或 None (未找到/加载失败)
    """
    if calib_path is None:
        # 默认路径：calibration/camera_calib.json
        calib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'camera_calib.json')

    if not os.path.isfile(calib_path):
        return None

    try:
        with open(calib_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        camera_matrix = np.array(data['camera_matrix'], dtype=np.float64)
        dist_coeffs = np.array(data['dist_coeffs'], dtype=np.float64).reshape(1, -1)
        image_size = tuple(data['image_size'])  # (w, h)
        return camera_matrix, dist_coeffs, image_size
    except Exception as e:
        print(f"[WARN] 加载相机标定文件失败 ({calib_path}): {e}")
        return None


# =========================================================================
# CLI 入口
# =========================================================================
def main():
    parser = argparse.ArgumentParser(description='相机内参标定工具')
    parser.add_argument('--camera-id', type=int, default=0, help='摄像头ID')
    parser.add_argument('--board-size', type=str, default=f'{DEFAULT_BOARD_COLS}x{DEFAULT_BOARD_ROWS}',
                        help='棋盘格内角点数，格式: 列x行 (默认: 9x6)')
    parser.add_argument('--square-size', type=float, default=DEFAULT_SQUARE_SIZE,
                        help='棋盘格方块边长 (mm，默认: 25)')
    parser.add_argument('--output', type=str, default=None,
                        help='输出文件路径 (默认: calibration/camera_calib.json)')
    args = parser.parse_args()

    # 解析 board size
    try:
        cols, rows = [int(x) for x in args.board_size.split('x')]
    except ValueError:
        print(f"[ERROR] 无效的棋盘格尺寸格式: {args.board_size}，应为 列x行 如 9x6")
        return 1

    print(f"[INFO] 相机标定工具")
    print(f"  摄像头ID: {args.camera_id}")
    print(f"  棋盘格: {cols}x{rows}")
    print(f"  方格边长: {args.square_size}mm")

    gui = CameraCalibrationGUI(
        camera_id=args.camera_id,
        board_size=(cols, rows),
        square_size=args.square_size,
        output_path=args.output
    )
    gui.run()
    return 0


if __name__ == '__main__':
    sys.exit(main())
