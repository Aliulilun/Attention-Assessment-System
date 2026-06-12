"""
第三階段：圖像正規化 (ETH-XGaze 官方標準)
Stage 3: Image Normalization

實作
1. 統一單位為 cm（與 stage2 一致）
2. Z 軸：從相機指向人臉中心（tvec 方向）
3. Y 軸：Z × 頭部 X 軸（確保眼睛水平線保持水平）
4. X 軸：Y × Z（右手座標系）
5. focal_norm = 960（ETH-XGaze 標準）
6. distance_norm = 60 cm（ETH-XGaze 標準）

"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional


class ImageNormalizer:
    """
    ETH-XGaze 官方標準的圖像正規化器
    完全遵循官方 demo.py 的實作方式
    
    核心步驟（與官方一致）：
    1. 從 solvePnP 結果獲得 R_head, t_head（頭部在相機座標系）
    2. 建立正規化座標系：
       - Z 軸 (forward): 從相機指向人臉中心（tvec 方向）
       - Y 軸 (down): Z × 頭部 X 軸
       - X 軸 (right): Y × Z
    3. 計算縮放係數，將 face 移動到固定距離 distance_norm
    4. 計算 homography 矩陣 W = K_norm @ S @ R_norm @ R_head^T @ K_orig^(-1)
    5. warpPerspective 得到正規化圖像
    """
    
    def __init__(self, config: Dict = None):
        """
        初始化圖像正規化器
        
        Args:
            config: 配置字典：
                - output_size: (width, height)，默認 (448, 448)
                - focal_norm: 正規化焦距，默認 960
                - distance_norm: 正規化距離（cm），默認 60
                - face_model_path: 3D 人臉模型路徑
        """
        if config is None:
            config = {}
        
        # 輸出參數
        self.output_size = config.get('output_size', (448, 448))
        
        # 正規化參數（ETH-XGaze 標準）
        self.focal_norm = config.get('focal_norm', 960.0)  # ETH-XGaze 標準
        self.distance_norm = config.get('distance_norm', 60.0)  # cm
        
        # 載入 3D 人臉模型（使用 ETH-XGaze 6 點模型，單位：cm）
        face_model_path = config.get('face_model_path', 'models/face_model_ethxgaze.txt')
        self.face_model_3d = self._load_face_model(face_model_path)
        
        # 計算正規化相機矩陣
        self.camera_matrix_norm = self._get_normalized_camera_matrix()
        
        print(f"✅ ImageNormalizer 初始化完成")
        print(f"  - 輸出尺寸: {self.output_size}")
        print(f"  - 正規化焦距: {self.focal_norm}")
        print(f"  - 正規化距離: {self.distance_norm} cm")
        print(f"  - 3D 模型點數: {self.face_model_3d.shape[0]}")
    
    def _load_face_model(self, model_path: str) -> np.ndarray:
        """
        載入 3D 人臉模型（單位：cm）
        
        重要：保持與 stage2 一致的單位（cm）
        """
        try:
            face_model = np.loadtxt(model_path, comments='#')
            if face_model.ndim == 1:
                face_model = face_model.reshape(-1, 3)
            
            print(f"  - 載入 3D 模型: {model_path}")
            print(f"  - 模型點數: {face_model.shape[0]}")
            print(f"  - 單位: cm")
            
            return face_model.astype(np.float32)
        except Exception as e:
            raise FileNotFoundError(f"無法載入 3D 人臉模型: {e}")
    
    def _get_normalized_camera_matrix(self) -> np.ndarray:
        """
        構建正規化相機內參矩陣
        """
        cx_norm = self.output_size[0] / 2.0
        cy_norm = self.output_size[1] / 2.0
        
        K_norm = np.array([
            [self.focal_norm, 0, cx_norm],
            [0, self.focal_norm, cy_norm],
            [0, 0, 1]
        ], dtype=np.float32)
        
        return K_norm
    
    def normalize(self,
                 image: np.ndarray,
                 rotation_vector: np.ndarray,
                 translation_vector: np.ndarray,
                 camera_matrix: np.ndarray) -> Dict:
        """
        修正版：完全對齊 ETH-XGaze 官方 normalization 幾何邏輯
        """
        try:
            hR, _ = cv2.Rodrigues(rotation_vector)
            ht = translation_vector.reshape(3, 1)
            distance = np.linalg.norm(ht)

            # Z 軸 (forward): 相機指向臉部
            forward = (ht / distance).flatten()
            
            # 取得頭部 X 軸 (Stage 2 中已修正為指向臉部左側)
            hRx = hR[:, 0]

            # --- 修正：確保 Y 軸向下 ---
            # 原本是 np.cross(forward, hRx) 導致向上
            # 將順序反轉，或者對結果取負，強制讓 Y 軸指向下方 (下巴方向)
            y_axis = np.cross(hRx, forward) # 反轉外積順序：X cross Z = -Y (向下)
            y_axis /= np.linalg.norm(y_axis)

            # 重新計算 X 軸 (Right) 以維持正交右手系
            x_axis = np.cross(y_axis, forward)
            x_axis /= np.linalg.norm(x_axis)

            # 構建旋轉矩陣 R_n
            R_n = np.vstack([x_axis, y_axis, forward])

            # 5. 計算 Homography W
            z_scale = self.distance_norm / distance
            S = np.diag([1.0, 1.0, z_scale])
            K_norm = self.camera_matrix_norm
            K_inv = np.linalg.inv(camera_matrix)
            
            # 官方標準 W 矩陣
            W = K_norm @ S @ R_n @ K_inv
            
            # 6. 執行 Warp (移除 WARP_INVERSE_MAP 標誌進行測試)
            # 如果還是顛倒，請嘗試移除或加入標誌，通常標準 W 不需要 INVERSE
            normalized_image = cv2.warpPerspective(
                image, W, self.output_size,
                flags=cv2.INTER_LINEAR
            )

            # 7. 計算歸一化後的姿態標籤 (給 Stage 4 使用)
            # R_head_norm = R_n @ hR
            hR_norm = R_n @ hR
            
            return {
                'normalized_image': normalized_image,
                'warp_matrix': W.astype(np.float32),
                'head_rot_norm': hR_norm,
                'face_center_distance': distance,
                'scale_factor': z_scale,
                'success': True
            }
            
        except Exception as e:
            print(f"❌ 正規化失敗: {e}")
            return self._fallback(image) 
    
    def _fallback(self, image: np.ndarray) -> Dict:
        """
        失敗時的備用方案
        """
        fallback_image = cv2.resize(image, self.output_size)
        return {
            'normalized_image': fallback_image,
            'warp_matrix': np.eye(3, dtype=np.float32),
            'head_rot_norm': np.eye(3, dtype=np.float32),
            'gaze_rot_norm': np.eye(3, dtype=np.float32),
            'scale_factor': 1.0,
            'face_center_distance': 0.0,
            'success': False
        }
    
    def visualize_normalization(self,
                               original_image: np.ndarray,
                               normalized_image: np.ndarray,
                               warp_matrix: np.ndarray,
                               face_center_distance: float = None,
                               scale_factor: float = None) -> np.ndarray:
        """
        可視化正規化結果
        
        在原圖上繪製正規化區域的邊界
        """
        h_norm, w_norm = normalized_image.shape[:2]
        
        # 正規化圖像的四個角點
        corners_norm = np.array([
            [0, 0],
            [w_norm - 1, 0],
            [w_norm - 1, h_norm - 1],
            [0, h_norm - 1]
        ], dtype=np.float32).reshape(-1, 1, 2)
        
        # 反變換到原圖
        try:
            W_inv = np.linalg.inv(warp_matrix)
            corners_orig = cv2.perspectiveTransform(corners_norm, W_inv)
            corners_orig = corners_orig.reshape(-1, 2).astype(np.int32)
            
            # 繪製邊界
            vis = original_image.copy()
            for i in range(4):
                pt1 = tuple(corners_orig[i])
                pt2 = tuple(corners_orig[(i + 1) % 4])
                cv2.line(vis, pt1, pt2, (0, 255, 0), 2)
            
            # 繪製對角線（幫助理解變換）
            cv2.line(vis, tuple(corners_orig[0]), tuple(corners_orig[2]), 
                    (0, 255, 0), 1)
            cv2.line(vis, tuple(corners_orig[1]), tuple(corners_orig[3]), 
                    (0, 255, 0), 1)
            
        except:
            vis = original_image.copy()
        
        # 縮放並並排顯示
        vis_resized = cv2.resize(vis, (w_norm, h_norm))
        comparison = np.hstack([vis_resized, normalized_image])
        
        # 添加標籤
        cv2.putText(comparison, 'Original + ROI', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(comparison, 'Normalized', (w_norm + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 添加資訊
        if face_center_distance is not None:
            cv2.putText(comparison, f'Distance: {face_center_distance:.1f} cm', 
                       (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if scale_factor is not None:
            cv2.putText(comparison, f'Scale: {scale_factor:.3f}', 
                       (10, 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 分隔線
        cv2.line(comparison, (w_norm, 0), (w_norm, h_norm), (255, 255, 255), 2)
        
        return comparison
    
    def get_normalization_params(self) -> Dict:
        """
        獲取正規化參數
        """
        return {
            'output_size': self.output_size,
            'focal_norm': self.focal_norm,
            'distance_norm': self.distance_norm,
            'camera_matrix_norm': self.camera_matrix_norm.tolist()
        }


def create_image_normalizer(config: Dict = None) -> ImageNormalizer:
    """
    工廠函數：創建圖像正規化器
    """
    return ImageNormalizer(config)


# ==================== 測試代碼 ====================

def test_normalizer():
    """
    測試正規化器
    """
    print("=" * 70)
    print("測試 ETH-XGaze Head Orientation Normalization")
    print("=" * 70)
    
    # 創建正規化器
    config = {
        'output_size': (448, 448),
        'focal_norm': 960.0,
        'distance_norm': 60.0,  # cm
        'face_model_path': 'models/face_model_ethxgaze.txt'
    }
    
    try:
        normalizer = create_image_normalizer(config)
    except FileNotFoundError:
        print("⚠️ 找不到 3D 模型文件，使用默認配置")
        config['face_model_path'] = None
        normalizer = create_image_normalizer(config)
    
    # 創建測試圖像
    print("\n創建測試圖像...")
    image = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)
    
    # 繪製測試圖案
    cv2.rectangle(image, (220, 140), (420, 340), (255, 200, 100), -1)
    cv2.circle(image, (280, 200), 20, (0, 0, 255), -1)  # 左眼
    cv2.circle(image, (360, 200), 20, (0, 0, 255), -1)  # 右眼
    cv2.ellipse(image, (320, 280), (40, 20), 0, 0, 180, (0, 0, 0), 2)  # 嘴
    
    # 模擬頭部姿態（輕微向右轉）
    rvec = np.array([[0.1], [-0.2], [0.05]], dtype=np.float32)  # 弧度
    tvec = np.array([[0], [0], [60]], dtype=np.float32)  # cm
    
    # 相機矩陣
    K = np.array([
        [500, 0, 320],
        [0, 500, 240],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # 執行正規化
    print("\n執行正規化...")
    result = normalizer.normalize(image, rvec, tvec, K)
    
    if result['success']:
        print("\n✅ 正規化成功！")
        
        # 可視化
        vis = normalizer.visualize_normalization(
            image,
            result['normalized_image'],
            result['warp_matrix'],
            result['face_center_distance'],
            result['scale_factor']
        )
        
        # 保存結果
        from pathlib import Path
        output_dir = Path('output')
        output_dir.mkdir(exist_ok=True)
        
        cv2.imwrite(str(output_dir / 'test_normalized.jpg'), 
                   result['normalized_image'])
        cv2.imwrite(str(output_dir / 'test_comparison.jpg'), vis)
        
        print(f"✅ 結果已保存到 output/")
    else:
        print("\n❌ 正規化失敗")
    
    print("\n" + "=" * 70)


if __name__ == '__main__':
    test_normalizer()