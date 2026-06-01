"""
LookBothWays (LBW) 数据集加载器
数据集结构:
    lbw_root/
        face_ims/       00000190_face.jpg ...
        scene_ims/      00000190_scene.jpg ...
        gaze_info/      00000190_gaze      (文本文件，无扩展名)
        scene_depth/    (可选，不使用)
"""

import os
import re
import numpy as np
from pathlib import Path
from PIL import Image


def parse_gaze_file(filepath):
    """
    解析单个 gaze_info 文件，返回字典。
    支持字段:
        Gaze_Loc_2D, Gaze_Loc_3D,
        Right_Gaze_Dir, Right_2D_Eye_Loc, Right_3D_Eye_Loc,
        Left_Gaze_Dir,  Left_2D_Eye_Loc,  Left_3D_Eye_Loc
    """
    data = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue
            key, val = line.split(':', 1)
            key = key.strip()
            # 提取所有数字（含负号和小数点）
            nums = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', val)
            if nums:
                arr = np.array([float(x) for x in nums])
                data[key] = arr[0] if len(arr) == 1 else arr
    return data


class LBWDataset:
    """
    LookBothWays 数据集，逐样本迭代。
    每个样本返回:
        {
          'frame_id'      : str,           # '00000190'
          'face_path'     : str,           # 面部图像路径
          'scene_path'    : str,           # 场景图像路径
          'gaze_loc_2d'   : np.ndarray,    # [u, v] 真值注视点（场景图坐标）
          'gaze_loc_3d'   : np.ndarray,    # 3D 注视点
          'right_gaze_dir': np.ndarray,    # 右眼视线方向（单位向量）
          'right_eye_loc3d': np.ndarray,   # 右眼 3D 位置
          'left_gaze_dir' : np.ndarray,    # 左眼视线方向
          'left_eye_loc3d': np.ndarray,    # 左眼 3D 位置
          'scene_size'    : tuple,         # (W, H) 场景图尺寸
        }
    """

    def __init__(self, root: str):
        self.root = Path(root)
        self.face_dir  = self.root / 'face_ims'
        self.scene_dir = self.root / 'scene_ims'
        self.gaze_dir  = self.root / 'gaze_info'

        # 找所有有 gaze 文件的帧（以 gaze_info 为主键）
        gaze_files = sorted(self.gaze_dir.glob('*_gaze*'))
        self.frame_ids = []
        for g in gaze_files:
            # 从文件名中提取 frame_id，处理可能的扩展名（如 .txt）
            name = g.name
            if '_gaze' in name:
                fid = name[:name.find('_gaze')]  # '00000190'
            else:
                continue
            # 检查对应的 face / scene 是否存在（支持 jpg/png）
            face_ok  = self._find_image(self.face_dir,  fid + '_face')
            scene_ok = self._find_image(self.scene_dir, fid + '_scene')
            if face_ok and scene_ok:
                self.frame_ids.append(fid)

        print(f"[LBWDataset] root={root}, valid samples={len(self.frame_ids)}")

    def _find_image(self, folder: Path, stem: str):
        """查找 stem.jpg 或 stem.png，返回路径或 None"""
        for ext in ['.jpg', '.jpeg', '.png']:
            p = folder / (stem + ext)
            if p.exists():
                return str(p)
        return None

    def __len__(self):
        return len(self.frame_ids)

    def __getitem__(self, idx):
        fid = self.frame_ids[idx]
        # 查找带扩展名的 gaze 文件
        gaze_file = None
        for ext in ['', '.txt']:
            p = self.gaze_dir / (fid + '_gaze' + ext)
            if p.exists():
                gaze_file = p
                break
        if gaze_file is None:
            raise FileNotFoundError(f"Gaze file not found for frame {fid}")
        gaze_data = parse_gaze_file(gaze_file)

        face_path  = self._find_image(self.face_dir,  fid + '_face')
        scene_path = self._find_image(self.scene_dir, fid + '_scene')

        # 读场景图尺寸（用于归一化）
        with Image.open(scene_path) as img:
            scene_size = img.size  # (W, H)

        return {
            'frame_id'       : fid,
            'face_path'      : face_path,
            'scene_path'     : scene_path,
            'gaze_loc_2d'    : gaze_data.get('Gaze_Loc_2D',    np.zeros(2)),
            'gaze_loc_3d'    : gaze_data.get('Gaze_Loc_3D',    np.zeros(3)),
            'right_gaze_dir' : gaze_data.get('Right_Gaze_Dir', np.zeros(3)),
            'right_eye_loc3d': gaze_data.get('Right_3D_Eye_Loc', np.zeros(3)),
            'left_gaze_dir'  : gaze_data.get('Left_Gaze_Dir',  np.zeros(3)),
            'left_eye_loc3d' : gaze_data.get('Left_3D_Eye_Loc', np.zeros(3)),
            'scene_size'     : scene_size,
        }

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
