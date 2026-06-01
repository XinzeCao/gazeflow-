"""
gaze_to_scene_calibrator.py
---------------------------
用 LBW 数据集中已知的 (眼睛3D位置, 视线方向) → Gaze_Loc_2D
训练一个多项式岭回归映射，和原项目 3.3 节标定模块思路完全一致。

输入特征（8维，和原项目对齐）：
    [eye_x, eye_y, eye_z,      # 右眼/左眼平均 3D 位置
     dir_x, dir_y, dir_z,      # 平均视线方向
     dir_x/dir_z, dir_y/dir_z] # 透视除法（线性化投影关系）

输出：归一化场景坐标 [u', v']，u'∈[0,1], v'∈[0,1]
"""

import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
import joblib
from pathlib import Path


def build_feature(sample: dict) -> np.ndarray:
    """
    从一个 LBW 样本构造 8 维特征向量。
    若左右眼 3D 位置都是零向量（无效），则只用右眼。
    """
    r_loc = sample['right_eye_loc3d']
    l_loc = sample['left_eye_loc3d']
    r_dir = sample['right_gaze_dir']
    l_dir = sample['left_gaze_dir']

    # 左右眼平均（若某眼无效即全零则只用另一眼）
    r_valid = np.linalg.norm(r_loc) > 1e-6
    l_valid = np.linalg.norm(l_loc) > 1e-6

    if r_valid and l_valid:
        eye_loc = (r_loc + l_loc) / 2.0
        gaze_dir = (r_dir + l_dir) / 2.0
    elif r_valid:
        eye_loc, gaze_dir = r_loc, r_dir
    else:
        eye_loc, gaze_dir = l_loc, l_dir

    # 归一化视线方向
    norm = np.linalg.norm(gaze_dir)
    if norm > 1e-6:
        gaze_dir = gaze_dir / norm

    # 透视除法（若 dir_z 接近 0 则置 0）
    dz = gaze_dir[2]
    persp_x = gaze_dir[0] / dz if abs(dz) > 1e-6 else 0.0
    persp_y = gaze_dir[1] / dz if abs(dz) > 1e-6 else 0.0

    feat = np.array([
        eye_loc[0], eye_loc[1], eye_loc[2],
        gaze_dir[0], gaze_dir[1], gaze_dir[2],
        persp_x, persp_y
    ], dtype=np.float32)
    return feat


class GazeToSceneCalibrator:
    """
    训练: fit(samples)  — samples 是 LBWDataset 返回的字典列表
    预测: predict(sample) → (u_px, v_px) 场景图像素坐标
    保存/加载: save / load
    """

    def __init__(self, alpha: float = 0.05, degree: int = 2):
        self.alpha  = alpha
        self.degree = degree
        self._pipe_u = None
        self._pipe_v = None

    def _make_pipe(self):
        return Pipeline([
            ('scaler', StandardScaler()),
            ('poly',   PolynomialFeatures(degree=self.degree, include_bias=False)),
            ('ridge',  Ridge(alpha=self.alpha)),
        ])

    # ------------------------------------------------------------------ #
    #  训练 - 使用模型预测的视线方向（推荐用于端到端验证）
    # ------------------------------------------------------------------ #
    def fit_with_predicted_gaze(
        self,
        samples: list,
        predicted_gaze_dirs: list,
        val_ratio: float = 0.2,
        verbose: bool = True,
    ):
        """
        用模型预测的视线方向训练映射（推荐用于端到端验证）。

        关键区别（对比 fit）:
          - fit():  输入特征用 LBW 真值 gaze_dir → 学习"理想视线 → 场景坐标"
          - 本方法: 输入特征用模型预测 gaze_cam   → 学习"模型视线 → 场景坐标"

        后者能让映射直接吸收模型的系统性偏差（比如相机内参不匹配导致的 y 方向偏移），
        从而避免端到端预测时出现固定偏移。

        参数:
          samples            : 标定样本列表（LBWDataset.__getitem__ 输出）
          predicted_gaze_dirs: 长度和 samples 一致的列表，每项是 (3,) numpy 数组，
                               对应该样本模型预测的 gaze 方向（已经过坐标系对齐，
                               即和 LBW gaze_dir 同坐标系）
          val_ratio          : 验证集比例
        """
        assert len(samples) == len(predicted_gaze_dirs), \
            f"样本数不一致: {len(samples)} vs {len(predicted_gaze_dirs)}"

        X, y_u, y_v = [], [], []
        for s, pred_dir in zip(samples, predicted_gaze_dirs):
            if pred_dir is None:
                continue
            W, H = s['scene_size']
            loc  = s['gaze_loc_2d']

            # 构造 pseudo sample：模型预测的视线方向 + LBW 真值眼睛位置
            r_loc = np.asarray(s['right_eye_loc3d'], dtype=np.float32)
            l_loc = np.asarray(s['left_eye_loc3d'],  dtype=np.float32)
            r_valid = np.linalg.norm(r_loc) > 1e-6
            l_valid = np.linalg.norm(l_loc) > 1e-6
            if r_valid and l_valid:
                eye_loc = (r_loc + l_loc) / 2.0
            elif r_valid:
                eye_loc = r_loc
            else:
                eye_loc = l_loc

            pseudo = {
                'right_eye_loc3d': eye_loc,
                'left_eye_loc3d' : np.zeros(3, dtype=np.float32),
                'right_gaze_dir' : pred_dir.astype(np.float32),
                'left_gaze_dir'  : pred_dir.astype(np.float32),
            }
            feat = build_feature(pseudo)
            X.append(feat)
            y_u.append(loc[0] / W)
            y_v.append(loc[1] / H)

        if len(X) == 0:
            raise ValueError("没有有效的标定样本（predicted_gaze_dirs 全部为 None）")

        X   = np.array(X,   dtype=np.float32)
        y_u = np.array(y_u, dtype=np.float32)
        y_v = np.array(y_v, dtype=np.float32)

        n_val = max(1, int(len(X) * val_ratio))
        X_tr, X_val = X[:-n_val],   X[-n_val:]
        u_tr, u_val = y_u[:-n_val], y_u[-n_val:]
        v_tr, v_val = y_v[:-n_val], y_v[-n_val:]

        self._pipe_u = self._make_pipe()
        self._pipe_v = self._make_pipe()
        self._pipe_u.fit(X_tr, u_tr)
        self._pipe_v.fit(X_tr, v_tr)

        if verbose:
            W_ref, H_ref = samples[0]['scene_size']
            u_pred = self._pipe_u.predict(X_val) * W_ref
            v_pred = self._pipe_v.predict(X_val) * H_ref
            u_gt   = u_val * W_ref
            v_gt   = v_val * H_ref
            pixel_err = np.sqrt((u_pred - u_gt)**2 + (v_pred - v_gt)**2)
            print(f"[Calibrator-PredictedGaze] 有效标定样本={len(X)}, "
                  f"val={n_val}, mean pixel error={pixel_err.mean():.1f}px, "
                  f"median={np.median(pixel_err):.1f}px")

    # ------------------------------------------------------------------ #
    #  训练 - 使用 LBW 真值视线方向（仅用于真值-验证基准）
    # ------------------------------------------------------------------ #
    def fit(self, samples: list, val_ratio: float = 0.2, verbose: bool = True):
        """
        samples : LBWDataset 样本列表
        val_ratio: 按帧 ID 顺序切末尾 val_ratio 作验证集
        """
        X, y_u, y_v = [], [], []
        for s in samples:
            W, H = s['scene_size']
            loc  = s['gaze_loc_2d']   # [u_px, v_px]
            feat = build_feature(s)
            X.append(feat)
            y_u.append(loc[0] / W)    # 归一化到 [0,1]
            y_v.append(loc[1] / H)

        X   = np.array(X,   dtype=np.float32)
        y_u = np.array(y_u, dtype=np.float32)
        y_v = np.array(y_v, dtype=np.float32)

        n_val = max(1, int(len(X) * val_ratio))
        X_tr, X_val = X[:-n_val],   X[-n_val:]
        u_tr, u_val = y_u[:-n_val], y_u[-n_val:]
        v_tr, v_val = y_v[:-n_val], y_v[-n_val:]

        self._pipe_u = self._make_pipe()
        self._pipe_v = self._make_pipe()
        self._pipe_u.fit(X_tr, u_tr)
        self._pipe_v.fit(X_tr, v_tr)

        if verbose:
            # 验证误差（像素级需要乘回尺寸，用首个样本的尺寸近似）
            W_ref, H_ref = samples[0]['scene_size']
            u_pred = self._pipe_u.predict(X_val) * W_ref
            v_pred = self._pipe_v.predict(X_val) * H_ref
            u_gt   = u_val * W_ref
            v_gt   = v_val * H_ref
            pixel_err = np.sqrt((u_pred - u_gt)**2 + (v_pred - v_gt)**2)
            print(f"[Calibrator] val samples={n_val}, "
                  f"mean pixel error={pixel_err.mean():.1f}px, "
                  f"median={np.median(pixel_err):.1f}px")

        # 记录每帧尺寸（预测时需要反归一化）
        self._scene_sizes = [(s['scene_size'][0], s['scene_size'][1]) for s in samples]

    # ------------------------------------------------------------------ #
    #  预测：输入单个样本，输出像素坐标
    # ------------------------------------------------------------------ #
    def predict(self, sample: dict):
        """
        返回 (u_px, v_px) —— 场景图上的预测注视点像素坐标。
        需要提供 sample['scene_size'] 用于反归一化。
        """
        assert self._pipe_u is not None, "请先调用 fit()"
        W, H = sample['scene_size']
        feat = build_feature(sample).reshape(1, -1)
        u_norm = float(self._pipe_u.predict(feat)[0])
        v_norm = float(self._pipe_v.predict(feat)[0])
        u_px = np.clip(u_norm * W, 0, W - 1)
        v_px = np.clip(v_norm * H, 0, H - 1)
        return u_px, v_px

    # ------------------------------------------------------------------ #
    #  你自己模型的 pitch/yaw → 构造伪 sample → 预测
    # ------------------------------------------------------------------ #
    def predict_from_model_output(
        self,
        pitch: float, yaw: float,
        eye_loc_3d: np.ndarray,
        scene_size: tuple
    ):
        """
        将你的视线估计模型输出的 pitch/yaw 转成 3D 方向向量，
        然后走同一套映射，得到场景图坐标。

        pitch, yaw : 弧度制（和你模型输出一致）
        eye_loc_3d : (3,) 头部中心或眼睛位置，可从几何归一化中取
        scene_size : (W, H)
        """
        # pitch/yaw → 单位向量（和 ETH-XGaze 约定一致）
        x = -np.cos(pitch) * np.sin(yaw)
        y = -np.sin(pitch)
        z = -np.cos(pitch) * np.cos(yaw)
        gaze_dir = np.array([x, y, z], dtype=np.float32)

        pseudo_sample = {
            'right_eye_loc3d': eye_loc_3d.astype(np.float32),
            'left_eye_loc3d' : np.zeros(3, dtype=np.float32),  # 标记左眼无效
            'right_gaze_dir' : gaze_dir,
            'left_gaze_dir'  : gaze_dir,
            'scene_size'     : scene_size,
        }
        return self.predict(pseudo_sample)

    # ------------------------------------------------------------------ #
    #  接收 GazeFeatureExtractor.extract() 的原始输出，预测场景图坐标
    # ------------------------------------------------------------------ #
    def predict_from_raw_features(self, raw_features: dict, lbw_sample: dict):
        """
        混合模式：你的模型只贡献视线方向（pitch/yaw 等价于 gaze_cam），
        眼睛 3D 位置使用 LBW 标签提供的真值。

        理由:
          - LBW 的 face_ims 是已裁好的脸部图，你的模型训练时见的是完整摄像头帧，
            模型在裁剪图上估算的 face_center_cam 会有偏差。
          - LBW 标签的 Right/Left_3D_Eye_Loc 是数据集标注的精确位置，更可靠。
          - 这种"提供 ground-truth context"的做法能干净地隔离误差来源——
            评估出来的误差几乎完全反映你视线估计模型的跨域能力。
          - roll 角度对场景注视点几乎无贡献，所以只取 pitch/yaw 对应的 3D 方向向量。

        参数:
          raw_features: OfflineGazePredictor 返回的 'raw_features' 字段
                        其中只用 'gaze_cam'（3D 视线方向单位向量，相机坐标系）
          lbw_sample  : LBWDataset.__getitem__ 返回的样本字典
                        其中用 'right_eye_loc3d'、'left_eye_loc3d' 和 'scene_size'

        返回: (u_px, v_px) 场景图像素坐标
        """
        # 1. 视线方向：来自你的模型（pitch/yaw 编码在 gaze_cam 这个 3D 单位向量里）
        gaze_dir = np.asarray(raw_features['gaze_cam'], dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(gaze_dir)
        if norm > 1e-6:
            gaze_dir = gaze_dir / norm

        # 坐标系对齐：UniGaze 的 gaze_cam 用 OpenGL 风格相机坐标系（z 朝相机后方），
        # 而 LBW 用 OpenCV 风格（z 朝场景前方），整体取反即可对齐。
        gaze_dir = -gaze_dir

        # 2. 眼睛 3D 位置：直接用 LBW 标签（左右眼平均，单位 m）
        r_loc = np.asarray(lbw_sample['right_eye_loc3d'], dtype=np.float32)
        l_loc = np.asarray(lbw_sample['left_eye_loc3d'],  dtype=np.float32)
        r_valid = np.linalg.norm(r_loc) > 1e-6
        l_valid = np.linalg.norm(l_loc) > 1e-6
        if r_valid and l_valid:
            eye_loc = (r_loc + l_loc) / 2.0
        elif r_valid:
            eye_loc = r_loc
        else:
            eye_loc = l_loc

        # 3. 复用 predict() 走同一套多项式岭回归映射
        pseudo_sample = {
            'right_eye_loc3d': eye_loc,
            'left_eye_loc3d' : np.zeros(3, dtype=np.float32),
            'right_gaze_dir' : gaze_dir,
            'left_gaze_dir'  : gaze_dir,
            'scene_size'     : lbw_sample['scene_size'],
        }
        return self.predict(pseudo_sample)

    # ------------------------------------------------------------------ #
    #  保存 / 加载
    # ------------------------------------------------------------------ #
    def save(self, path: str):
        joblib.dump({'pipe_u': self._pipe_u, 'pipe_v': self._pipe_v}, path)
        print(f"[Calibrator] saved → {path}")

    def load(self, path: str):
        obj = joblib.load(path)
        self._pipe_u = obj['pipe_u']
        self._pipe_v = obj['pipe_v']
        print(f"[Calibrator] loaded ← {path}")