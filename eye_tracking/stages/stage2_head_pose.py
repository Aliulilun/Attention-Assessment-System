"""
第二階段：頭部姿態估計
Stage 2: Head Pose Estimation

使用 OpenCV solvePnP + scipy Rotation
簡化版實作，直接從旋轉矩陣計算 Euler 角
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple
from scipy.spatial.transform import Rotation as R


class HeadPoseEstimator:
    """
    使用 OpenCV solvePnP 進行頭部姿態估計
    
    核心邏輯：
    - MediaPipe 3D 模型 + 2D landmarks → solvePnP → rvec/tvec
    - rvec → rotation matrix → Euler angles (YXZ 順序)
    """
    
    def __init__(self, config: Dict = None):
        """
        初始化頭部姿態估計器
        
        Args:
            config: 配置字典，包含以下鍵值：
                - face_model_path: 3D 人臉模型路徑
        """
        if config is None:
            config = {}
        
        # 3D 人臉模型路徑（使用 ETH-XGaze 6 點模型）
        self.face_model_path = config.get(
            'face_model_path',
            'models/face_model_ethxgaze.txt'
        )
        
        # 載入 3D 人臉模型
        self.face_model_3d = self._load_face_model()
        
        print(f"✅ 頭部姿態估計器初始化完成")
        print(f"  - 3D 模型點數: {self.face_model_3d.shape[0]}")
    
    def get_default_camera_matrix(self, width: int, height: int) -> np.ndarray:
        """
        根據圖像尺寸自動生成估算內參矩陣 (Estimated Camera Matrix)
        學術原理：假設焦距等於圖像寬度，光心位於圖像中心。
        """
        focal_length = width 
        center_x = width / 2
        center_y = height / 2

        camera_matrix = np.array([
            [focal_length, 0, center_x],
            [0, focal_length, center_y],
            [0, 0, 1]
        ], dtype=np.float32)
        
        return camera_matrix
    
    def _load_face_model(self) -> np.ndarray:
        face_model = np.loadtxt(self.face_model_path, comments='#')
        face_model = face_model.astype(np.float32)

        # 核心修改：模型轉向
        # 1. Z 軸取反：讓臉朝向相機 (原本是遠離臉部)
        face_model[:, 2] *= -1 
        # 2. X 軸取反：為了維持右手坐標系，防止左右鏡像顛倒
        face_model[:, 0] *= -1 

        # 3. 官方歸一化建議：將原點設為雙眼中心 (原本可能在鼻尖或模型中心)
        # 索引說明：0:左外, 1:左內, 2:右內, 3:右外
        eye_center = np.mean(face_model[0:4, :], axis=0)
        face_model -= eye_center # 將旋轉支點移至雙眼中心
        
        return face_model
    
    def estimate(self, 
                    landmarks_2d: np.ndarray, 
                    camera_matrix: np.ndarray = None, # 改為可選
                    distortion_coeffs: np.ndarray = None,
                    image_size: Tuple[int, int] = (1280, 720)) -> Dict:
            """
            估計頭部姿態 (重構版：符合 ETH-XGaze 幾何規範)
            """
            # --- 邏輯 A: 自動檢查/生成內參 ---
            if camera_matrix is None:
                camera_matrix = self.get_default_camera_matrix(image_size[0], image_size[1])
            
            if distortion_coeffs is None:
                distortion_coeffs = np.zeros((4, 1), dtype=np.float32)

            # --- 邏輯 B: 穩健 PnP 求解 ---
            # 1. RANSAC 階段：過濾 MediaPipe 噪點
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                self.face_model_3d,
                landmarks_2d,
                camera_matrix,
                distortion_coeffs,
                reprojectionError=20.0, # 提高容忍度，減少 "求解失敗" 發生
                iterationsCount=100,
                flags=cv2.SOLVEPNP_EPNP
            )

            if success:
                # 2. Iterative 階段：精細優化
                success, rvec, tvec = cv2.solvePnP(
                    self.face_model_3d,
                    landmarks_2d,
                    camera_matrix,
                    distortion_coeffs,
                    rvec=rvec,
                    tvec=tvec,
                    useExtrinsicGuess=True,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
            
            if not success:
                return {'success': False} # 保持簡潔的回傳
            
            # --- 邏輯 C: 歐拉角分解 (不再寫死 Roll) ---
            R_cam, _ = cv2.Rodrigues(rvec)
            
            # 提取模型 Z 軸向量
            bz = R_cam[:, 2] 

            # ETH-XGaze 標準分解公式
            yaw_rad = np.arctan2(bz[0], bz[2])
            pitch_rad = np.arcsin(-bz[1])
            # 分解 Roll (歪頭角度)
            roll_rad = np.arctan2(R_cam[1, 0], R_cam[1, 1])
            
            return {
                'rvec': rvec,
                'tvec': tvec,
                'rotation_matrix': R_cam,
                'euler_angles': {
                    'pitch': float(np.degrees(pitch_rad)),
                    'yaw': float(np.degrees(yaw_rad)),
                    'roll': float(np.degrees(roll_rad)) # 現在 Roll 有值了
                },
                'success': True
            }
    
    def draw_axes(self, 
                  image: np.ndarray, 
                  rvec: np.ndarray, 
                  tvec: np.ndarray,
                  camera_matrix: np.ndarray,
                  distortion_coeffs: np.ndarray = None,
                  axis_length: float = 50.0) -> np.ndarray:
        """
        在圖像上繪製 3D 坐標軸
        
        Args:
            image: 輸入圖像
            rvec: 旋轉向量
            tvec: 平移向量
            camera_matrix: 相機內參矩陣
            distortion_coeffs: 畸變係數
            axis_length: 坐標軸長度（像素）
        
        Returns:
            繪製了坐標軸的圖像
        """
        if distortion_coeffs is None:
            distortion_coeffs = np.zeros((4, 1), dtype=np.float32)
        
        # 定義 3D 坐標軸端點
        axis_points_3d = np.array([
            [0, 0, 0],              # 原點
            [axis_length, 0, 0],    # X 軸（紅色）
            [0, axis_length, 0],    # Y 軸（綠色）
            [0, 0, axis_length]     # Z 軸（藍色）
        ], dtype=np.float32)
        
        # 投影到 2D
        axis_points_2d, _ = cv2.projectPoints(
            axis_points_3d,
            rvec,
            tvec,
            camera_matrix,
            distortion_coeffs
        )
        
        axis_points_2d = axis_points_2d.reshape(-1, 2).astype(int)
        
        # 繪製坐標軸
        origin = tuple(axis_points_2d[0])
        image = cv2.line(image, origin, tuple(axis_points_2d[1]), (0, 0, 255), 3)  # X - 紅色
        image = cv2.line(image, origin, tuple(axis_points_2d[2]), (0, 255, 0), 3)  # Y - 綠色
        image = cv2.line(image, origin, tuple(axis_points_2d[3]), (255, 0, 0), 3)  # Z - 藍色
        
        return image
    
    def get_head_direction_vector(self, rotation_matrix: np.ndarray) -> np.ndarray:
        """
        從旋轉矩陣獲取頭部方向向量（視線方向）
        
        Args:
            rotation_matrix: 旋轉矩陣 (3, 3)
        
        Returns:
            direction_vector: 3D 單位向量 (3,)
        """
        # Z 軸方向即為頭部朝向
        direction_vector = rotation_matrix[:, 2]
        return direction_vector / np.linalg.norm(direction_vector)


def create_head_pose_estimator(config: Dict = None) -> HeadPoseEstimator:
    """
    工廠函數：創建頭部姿態估計器實例
    """
    return HeadPoseEstimator(config)


# 使用範例
if __name__ == "__main__":
    print("Stage 2: 頭部姿態估計模組")
    print("=" * 50)
    
    # 創建估計器
    estimator = create_head_pose_estimator()
    
    print("\n模組測試完成 ✅")