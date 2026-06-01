import os
import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_CUR_DIR)

from .model_utils import instantiate_from_cfg
from .model_utils import wrap_transforms
from .gaze_math import pitchyaw_to_vector, vector_to_pitchyaw, estimateHeadPose, normalize, get_face_center_by_nose
import face_alignment

# Camera intrinsics (optional dependency on calibration module)
try:
    from calibration.camera import load_camera_calibration
except ImportError:
    load_camera_calibration = None


def set_dummy_camera_model(image=None):
    """构造一个简易相机模型；如有真实标定参数请替换。"""
    if image is None:
        h, w = 480, 640
    else:
        h, w = image.shape[:2]
    focal_length = w * 4.0
    center = (w // 2, h // 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype=np.double
    )
    camera_distortion = np.zeros((1, 5), dtype=np.double)
    return camera_matrix, camera_distortion


def resolve_camera_calib_path(calib_path=None):
    """Resolve camera_calib.json from common project locations."""
    if calib_path is None:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'camera_calib.json')

    if os.path.isabs(calib_path):
        return calib_path

    candidates = [
        os.path.abspath(calib_path),
        os.path.join(_project_root, calib_path),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), calib_path),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return candidates[-1]


def load_camera_intrinsics(calib_path=None):
    """
    加载真实相机内参。

    Returns:
        (camera_matrix, dist_coeffs, image_size, resolved_path) 或 None
    """
    resolved_path = resolve_camera_calib_path(calib_path)
    result = load_camera_calibration(resolved_path)
    if result is None:
        return None
    camera_matrix, dist_coeffs, image_size = result
    return camera_matrix, dist_coeffs, image_size, resolved_path


def load_checkpoint(model, ckpt_key, ckpt_path):
    assert os.path.isfile(ckpt_path), f"checkpoint 不存在: {ckpt_path}"
    weights = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if ckpt_key in weights:
        state = weights[ckpt_key]
    else:
        # 兼容直存state_dict的情况
        state = weights
    first_key = next(iter(state))
    if first_key.startswith('module.'):
        state = {k[7:]: v for k, v in state.items()}
    model.load_state_dict(state, strict=False)


def denormalize_predicted_gaze(gaze_yaw_pitch, R_inv):
    vec_norm = pitchyaw_to_vector(gaze_yaw_pitch.reshape(1, 2)).reshape(3, 1)
    vec_cam = (R_inv @ vec_norm)
    vec_cam = vec_cam / (np.linalg.norm(vec_cam) + 1e-8)
    yaw_pitch_cam = vector_to_pitchyaw(vec_cam.reshape(1, 3))
    return vec_cam.reshape(-1), yaw_pitch_cam.reshape(-1)


def denormalize_predicted_gaze_torch(gaze_yaw_pitch_t, R_inv_t):
    """torch版本反归一化：输入[pitch,yaw]与R_inv，在device上完成向量变换."""
    if gaze_yaw_pitch_t.ndim == 1:
        g = gaze_yaw_pitch_t.unsqueeze(0)
    else:
        g = gaze_yaw_pitch_t
    vec_norm = pitchyaw_to_vector(g)  # [1,3]
    vec_cam = (R_inv_t @ vec_norm.transpose(0, 1)).squeeze(1)  # [3]
    vec_cam = vec_cam / (torch.norm(vec_cam) + 1e-8)
    yaw_pitch_cam = vector_to_pitchyaw(vec_cam.unsqueeze(0)).squeeze(0)  # [2]
    return vec_cam, yaw_pitch_cam


def rotmat_to_euler_xyz(R):
    """将旋转矩阵转为XYZ顺序的欧拉角(弧度)：roll(x), pitch(y), yaw(z)。"""
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0
    return np.array([roll, pitch, yaw], dtype=float)


class GazeFeatureExtractor:
    """
    负责：
      - 人脸关键点检测（face_alignment）
      - 头姿估计与归一化
      - UniGaze前向得到[pitch,yaw]_norm
      - 反归一化为相机系gaze向量与角度
      - 返回供路线B回归使用的特征x
    """
    def __init__(self, model_cfg_path=None, ckpt_path=None, device=None,
                 camera_calib_path='camera_calib.json', camera_is_mirrored=True):
        # --- 添加详细调试信息 ---
        print("[DEBUG][GazeFeatureExtractor] 开始初始化...")

        # (保留您原来的调试信息)
        print(f"[DEBUG] 当前文件路径: {__file__}")
        cur_dir = os.path.dirname(__file__)
        print(f"[DEBUG] 当前目录: {cur_dir}")

        if model_cfg_path is None:
            model_cfg_path = os.path.join(cur_dir, 'config.yaml')
        if ckpt_path is None:
            ckpt_path = os.path.join(cur_dir, 'weights', 'unigaze_l16_joint.pth.tar')
            print(f"[DEBUG] 使用用户指定的权重路径: {ckpt_path}")
        else:
            print(f"[DEBUG] 用户提供的权重路径: {ckpt_path}")
        
        print(f"[DEBUG] 权重文件是否存在: {os.path.isfile(ckpt_path)}")
        print(f"[DEBUG] 配置文件是否存在: {os.path.isfile(model_cfg_path)}")
        
        assert os.path.isfile(model_cfg_path), f"找不到模型配置: {model_cfg_path}"
        assert os.path.isfile(ckpt_path), f"找不到权重: {ckpt_path}"

        print("[DEBUG][GazeFeatureExtractor] 准备设置设备...")
        self.device = torch.device(device) if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[DEBUG][GazeFeatureExtractor] 设备设置为: {self.device}")

        # 模型
        print("[DEBUG][GazeFeatureExtractor] 准备加载模型配置...")
        net_cfg = OmegaConf.load(model_cfg_path)['net_config']
        print("[DEBUG][GazeFeatureExtractor] 模型配置加载成功。")

        if 'params' in net_cfg and 'custom_pretrained_path' in net_cfg['params']:
            net_cfg['params']['custom_pretrained_path'] = None
        
        print("[DEBUG][GazeFeatureExtractor] 准备实例化模型 (instantiate_from_cfg)...")
        self.model = instantiate_from_cfg(net_cfg)
        print("[DEBUG][GazeFeatureExtractor] 模型实例化成功。")

        print("[DEBUG][GazeFeatureExtractor] 准备加载模型权重 (load_checkpoint)...")
        load_checkpoint(self.model, 'model_state', ckpt_path)
        print("[DEBUG][GazeFeatureExtractor] 模型权重加载成功。")

        print("[DEBUG][GazeFeatureExtractor] 准备将模型移至设备 ( .to(device) )...")
        self.model.eval().to(self.device)
        print("[DEBUG][GazeFeatureExtractor] 模型已移至设备。")

        print("[DEBUG][GazeFeatureExtractor] 准备创建图像转换 (wrap_transforms)...")
        self.tfm = wrap_transforms('basic_imagenet', image_size=224)
        print("[DEBUG][GazeFeatureExtractor] 图像转换创建成功。")

        # 人脸对齐器
        print("[DEBUG][GazeFeatureExtractor] 准备初始化人脸对齐器 (face_alignment)...")
        try:
            lt = getattr(face_alignment.LandmarksType, 'TWO_D', None)
            if lt is None:
                lt = getattr(face_alignment.LandmarksType, '_2D', None)
            if lt is None:
                raise AttributeError('No TWO_D/_2D in LandmarksType')
            try:
                print("[DEBUG][GazeFeatureExtractor] 尝试使用 blazeface...")
                self.fa = face_alignment.FaceAlignment(lt, device='cuda' if torch.cuda.is_available() else 'cpu', flip_input=False, face_detector='blazeface')
                print("[DEBUG][GazeFeatureExtractor] blazeface 初始化成功。")
            except Exception:
                print("[DEBUG][GazeFeatureExtractor] blazeface 失败，尝试默认检测器...")
                self.fa = face_alignment.FaceAlignment(lt, device='cuda' if torch.cuda.is_available() else 'cpu', flip_input=False)
                print("[DEBUG][GazeFeatureExtractor] 默认检测器初始化成功。")
        except Exception:
            print("[DEBUG][GazeFeatureExtractor] 2D/3D LandmarksType 失败，尝试最后的备用方案...")
            self.fa = face_alignment.FaceAlignment(1, device='cuda' if torch.cuda.is_available() else 'cpu', flip_input=False)
            print("[DEBUG][GazeFeatureExtractor] 备用方案初始化成功。")
        print("[DEBUG][GazeFeatureExtractor] 人脸对齐器初始化完成。")

        # 头姿/归一化参数
        self.focal_norm = 960
        self.distance_norm = 600
        self.roi_size = (224, 224)
        self.face_model_path = os.path.join(cur_dir, 'face_model.txt')
        self.face_model_load = np.loadtxt(self.face_model_path)
        self.face_model = self.face_model_load[[20, 23, 26, 29, 15, 19], :]
        self.facePts = self.face_model.reshape(6, 1, 3)

        # 加载真实相机内参（如果可用）
        self.camera_is_mirrored = camera_is_mirrored
        intrinsics = load_camera_intrinsics(camera_calib_path)
        if intrinsics is not None:
            self.camera_matrix = intrinsics[0].copy()
            self.camera_distortion = intrinsics[1].copy()
            self.camera_image_size = intrinsics[2]
            self.camera_calib_path = intrinsics[3]
            self.has_real_intrinsics = True
            print(f"[INFO][GazeFeatureExtractor] 已加载真实相机内参: "
                  f"{self.camera_calib_path}, fx={self.camera_matrix[0,0]:.1f}, fy={self.camera_matrix[1,1]:.1f}")
        else:
            self.camera_matrix = None
            self.camera_distortion = None
            self.camera_image_size = None
            self.camera_calib_path = None
            self.has_real_intrinsics = False
            print("[INFO][GazeFeatureExtractor] 未找到相机标定文件，将使用dummy内参")

        print("[DEBUG][GazeFeatureExtractor] 初始化完成！")
        # --- 详细调试信息结束 ---

    @staticmethod
    def feature_names():
        return [
            'gaze_pitch',       # [0] pitch (vertical angle)
            'gaze_yaw',         # [1] yaw (horizontal angle)
            'head_roll',        # [2] head roll
            'head_pitch',       # [3] head pitch
            'head_yaw',         # [4] head yaw
            'face_center_x',    # [5] face center X (mm, camera coord)
            'face_center_y',    # [6] face center Y (mm, camera coord)
            'face_center_z',    # [7] face center Z (mm, camera coord)
        ]

    def _camera_model_for_crop(self, frame_shape, crop_xyxy, face_img):
        if not self.has_real_intrinsics:
            camera_matrix, camera_distortion = set_dummy_camera_model(image=face_img)
            return camera_matrix, camera_distortion, 'dummy'

        frame_h, frame_w = frame_shape[:2]
        camera_matrix = self.camera_matrix.copy()
        camera_distortion = self.camera_distortion.copy()

        if self.camera_image_size:
            calib_w, calib_h = self.camera_image_size
            if calib_w > 0 and calib_h > 0 and (calib_w != frame_w or calib_h != frame_h):
                sx = frame_w / float(calib_w)
                sy = frame_h / float(calib_h)
                camera_matrix[0, 0] *= sx
                camera_matrix[0, 2] *= sx
                camera_matrix[1, 1] *= sy
                camera_matrix[1, 2] *= sy

        if self.camera_is_mirrored:
            camera_matrix[0, 2] = (frame_w - 1) - camera_matrix[0, 2]

        x_min, y_min, _, _ = crop_xyxy
        camera_matrix[0, 2] -= float(x_min)
        camera_matrix[1, 2] -= float(y_min)
        return camera_matrix, camera_distortion, 'camera_calib'

    def extract(self, bgr_frame):
        """
        输入: BGR图像
        输出: dict 或 None
          - x: (8,) 特征向量 [pitch/yaw, head_euler, face_center]
          - yaw_pitch_cam: (2,) 归一化相机系下[pitch,yaw]
          - gaze_cam: (3,) 注视向量（相机坐标系）
          - head_euler: (3,) roll,pitch,yaw（弧度），相机系
          - face_center_cam: (3,) 面部中心坐标
          - debug: 附带可视化/ROI等
        """
        if bgr_frame is None:
            return None
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        preds = self.fa.get_landmarks(rgb)
        if preds is None or len(preds) == 0:
            return None
        lm = preds[0]

        # ROI
        x_min = int(lm[:, 0].min()); x_max = int(lm[:, 0].max())
        y_min = int(lm[:, 1].min()); y_max = int(lm[:, 1].max())
        cx = (x_min + x_max) // 2; cy = (y_min + y_max) // 2
        bw = x_max - x_min; bh = y_max - y_min
        scale = 2.0
        x_min = max(0, cx - int(bw * scale // 2))
        x_max = min(bgr_frame.shape[1], cx + int(bw * scale // 2))
        y_min = max(0, cy - int(bh * scale // 2))
        y_max = min(bgr_frame.shape[0], cy + int(bh * scale // 2))
        face_img = bgr_frame[y_min:y_max, x_min:x_max]
        if face_img.size == 0:
            return None
        lm_local = lm - np.array([x_min, y_min])

        camera_matrix, camera_distortion, camera_model_source = self._camera_model_for_crop(
            bgr_frame.shape, (x_min, y_min, x_max, y_max), face_img
        )

        # 头姿与归一化
        lm6 = lm_local[[36, 39, 42, 45, 31, 35], :].astype(float).reshape(6, 1, 2)
        hr, ht = estimateHeadPose(lm6, self.facePts, camera_matrix, camera_distortion)
        hR = cv2.Rodrigues(hr)[0]
        face_center_camera_cord, _ = get_face_center_by_nose(hR=hR, ht=ht, face_model_load=self.face_model_load)

        img_norm, R, hR_norm, _, _, _ = normalize(
            face_img, lm_local, self.focal_norm, self.distance_norm, self.roi_size,
            face_center_camera_cord, hr, ht, camera_matrix, gc=None)

        # 极端姿态屏蔽
        hr_norm = np.array([np.arcsin(hR_norm[1, 2]), np.arctan2(hR_norm[0, 2], hR_norm[2, 2])])
        if np.linalg.norm(hr_norm) > 80 * np.pi / 180:
            return None

        # 模型前向（GPU）
        inp_rgb = img_norm[:, :, [2, 1, 0]]
        tensor = self.tfm(inp_rgb).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            ret = self.model(tensor)
        pred_gaze_t = ret["pred_gaze"][0].detach()  # [2] on device

        # 反归一化到原始相机系（GPU）
        R_t = torch.from_numpy(R).to(device=self.device, dtype=torch.float32)
        R_inv_t = torch.linalg.inv(R_t)
        gaze_cam_vec_t, yaw_pitch_cam_t = denormalize_predicted_gaze_torch(pred_gaze_t, R_inv_t)

        # 转回numpy以便下游使用/显示
        gaze_cam_vec = gaze_cam_vec_t.detach().cpu().numpy().astype(np.float32)
        yaw_pitch_cam = yaw_pitch_cam_t.detach().cpu().numpy().astype(np.float32)

        head_euler = rotmat_to_euler_xyz(hR).astype(np.float32)
        face_center = face_center_camera_cord.reshape(-1).astype(np.float32)

        # 8维特征向量: [pitch, yaw, head_roll, head_pitch, head_yaw, fc_x, fc_y, fc_z]
        x_feat = np.concatenate([
            yaw_pitch_cam.reshape(-1),          # [0:2]  pitch, yaw
            head_euler.reshape(-1),              # [2:5]  roll, pitch, yaw
            face_center.reshape(-1),             # [5:8]  x, y, z (mm)
        ]).astype(np.float32)

        debug = {
            'bbox_xyxy': (x_min, y_min, x_max, y_max),
            'img_norm': img_norm,
            'pred_gaze_norm': pred_gaze_t.detach().cpu().numpy(),
            'camera_model_source': camera_model_source,
            'camera_is_mirrored': self.camera_is_mirrored,
            'camera_matrix_crop': camera_matrix.copy(),
        }
        return {
            'x': x_feat,                                     # (8,) 完整特征向量
            'gaze_cam': gaze_cam_vec.reshape(-1),             # (3,) 视线方向向量
            'yaw_pitch_cam': yaw_pitch_cam.reshape(-1),       # (2,) [pitch, yaw]
            'head_euler': head_euler.reshape(-1),              # (3,) [roll, pitch, yaw]
            'face_center_cam': face_center.reshape(-1),       # (3,) 面部中心坐标
            'debug': debug
        }
