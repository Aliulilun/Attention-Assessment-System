"""
第五階段：視線向量轉換
Stage 5: Gaze Vector Conversion

將視線角度（pitch, yaw）轉換為 3D 單位向量：
- 簡單的三角函數計算
- 向量歸一化
- 雙向轉換支援
"""

import numpy as np
from typing import Dict, Tuple, Optional
import cv2


class GazeVectorConverter:
    """
    視線向量轉換器
    
    將 2D 視線角度（pitch, yaw）與 3D 視線向量相互轉換
    """
    
    def __init__(self, config: Dict = None):
        """
        初始化視線向量轉換器
        
        Args:
            config: 配置字典（預留，目前不需要配置）
        """
        if config is None:
            config = {}
        
        print(f"✓ GazeVectorConverter 初始化完成")
    
    def angles_to_vector(self, pitch: float, yaw: float) -> np.ndarray:
        # 修正為符合 ETH-XGaze 與 OpenCV 座標系的標準推導
        x = -np.cos(pitch) * np.sin(yaw)
        y = -np.sin(pitch)
        z = -np.cos(pitch) * np.cos(yaw)
        
        gaze_vector = np.array([x, y, z], dtype=np.float32)
        return gaze_vector / (np.linalg.norm(gaze_vector) + 1e-6)
    
    def vector_to_angles(self, gaze_vector: np.ndarray) -> Tuple[float, float]:
        """
        將 3D 視線向量轉換為視線角度
        
        Args:
            gaze_vector: 3D 視線向量 (3,) [x, y, z]
        
        Returns:
            pitch: 俯仰角（弧度）
            yaw: 偏航角（弧度）
        
        公式:
            pitch = arcsin(-y)
            yaw = arctan2(-x, -z)
        """
        # 歸一化向量
        norm = np.linalg.norm(gaze_vector)
        if norm > 1e-6:
            gaze_vector = gaze_vector / norm
        
        x, y, z = gaze_vector
        
        # 計算角度
        pitch = np.arcsin(-y)
        yaw = np.arctan2(-x, -z)
        
        return pitch, yaw
    
    def convert(self, 
                pitch: Optional[float] = None, 
                yaw: Optional[float] = None,
                gaze_vector: Optional[np.ndarray] = None,
                output_degrees: bool = False) -> Dict:
        """
        統一的轉換介面（自動判斷轉換方向）
        
        Args:
            pitch: 俯仰角（弧度），如果提供則進行 angles → vector 轉換
            yaw: 偏航角（弧度）
            gaze_vector: 3D 視線向量，如果提供則進行 vector → angles 轉換
            output_degrees: 是否輸出角度為度（而非弧度）
        
        Returns:
            result: 包含轉換結果的字典
        """
        if pitch is not None and yaw is not None:
            # 角度 → 向量
            vector = self.angles_to_vector(pitch, yaw)
            
            result = {
                'gaze_vector': vector,
                'pitch': pitch,
                'yaw': yaw,
                'conversion': 'angles_to_vector'
            }
            
            if output_degrees:
                result['pitch_deg'] = np.rad2deg(pitch)
                result['yaw_deg'] = np.rad2deg(yaw)
        
        elif gaze_vector is not None:
            # 向量 → 角度
            pitch, yaw = self.vector_to_angles(gaze_vector)
            
            result = {
                'gaze_vector': gaze_vector,
                'pitch': pitch,
                'yaw': yaw,
                'conversion': 'vector_to_angles'
            }
            
            if output_degrees:
                result['pitch_deg'] = np.rad2deg(pitch)
                result['yaw_deg'] = np.rad2deg(yaw)
        
        else:
            raise ValueError("必須提供 (pitch, yaw) 或 gaze_vector")
        
        result['success'] = True
        return result
    
    def batch_angles_to_vectors(self, angles: np.ndarray) -> np.ndarray:
        """
        批量將角度轉換為向量
        
        Args:
            angles: 角度陣列 (N, 2) - [pitch, yaw] 弧度
        
        Returns:
            vectors: 向量陣列 (N, 3) - [x, y, z]
        """
        pitch = angles[:, 0]
        yaw = angles[:, 1]
        
        x = -np.cos(pitch) * np.sin(yaw)
        y = -np.sin(pitch)
        z = -np.cos(pitch) * np.cos(yaw)
        
        vectors = np.stack([x, y, z], axis=1).astype(np.float32)
        
        # 批量歸一化
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / (norms + 1e-6)
        
        return vectors
    
    def batch_vectors_to_angles(self, vectors: np.ndarray) -> np.ndarray:
        """
        批量將向量轉換為角度
        
        Args:
            vectors: 向量陣列 (N, 3) - [x, y, z]
        
        Returns:
            angles: 角度陣列 (N, 2) - [pitch, yaw] 弧度
        """
        # 批量歸一化
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / (norms + 1e-6)
        
        x = vectors[:, 0]
        y = vectors[:, 1]
        z = vectors[:, 2]
        
        pitch = np.arcsin(-y)
        yaw = np.arctan2(-x, -z)
        
        angles = np.stack([pitch, yaw], axis=1).astype(np.float32)
        
        return angles
    
    def draw_gaze_vector_3d(self,
                           image: np.ndarray,
                           gaze_vector: np.ndarray,
                           origin: Tuple[int, int] = None,
                           length: float = 100.0,
                           thickness: int = 2,
                           color: Tuple[int, int, int] = (0, 0, 255)) -> np.ndarray:
        """
        在圖像上繪製 3D 視線向量（投影到 2D）
        
        Args:
            image: 輸入圖像
            gaze_vector: 3D 視線向量 (3,)
            origin: 起點（默認為圖像中心）
            length: 箭頭長度（像素）
            thickness: 線條粗細
            color: 箭頭顏色 (B, G, R)
        
        Returns:
            image_out: 繪製後的圖像
        """
        image_out = image.copy()
        h, w = image.shape[:2]
        
        # 設置起點
        if origin is None:
            origin = (w // 2, h // 2)
        
        # 將 3D 向量投影到 2D（簡單正交投影）
        # 忽略 Z 分量，使用 X, Y
        dx = length * gaze_vector[0]
        dy = length * gaze_vector[1]
        
        end_point = (
            int(origin[0] + dx),
            int(origin[1] + dy)
        )
        
        # 繪製箭頭
        cv2.arrowedLine(
            image_out,
            origin,
            end_point,
            color,
            thickness,
            cv2.LINE_AA,
            tipLength=0.2
        )
        
        return image_out
    
    def calculate_angular_error(self,
                               pred_vector: np.ndarray,
                               gt_vector: np.ndarray) -> float:
        """
        計算兩個視線向量之間的角度誤差
        
        Args:
            pred_vector: 預測的視線向量
            gt_vector: 真實的視線向量（ground truth）
        
        Returns:
            error_deg: 角度誤差（度）
        """
        # 歸一化
        pred_vector = pred_vector / (np.linalg.norm(pred_vector) + 1e-6)
        gt_vector = gt_vector / (np.linalg.norm(gt_vector) + 1e-6)
        
        # 計算點積（cos(角度)）
        dot_product = np.dot(pred_vector, gt_vector)
        
        # 限制在 [-1, 1] 範圍（避免數值誤差）
        dot_product = np.clip(dot_product, -1.0, 1.0)
        
        # 計算角度
        angle_rad = np.arccos(dot_product)
        angle_deg = np.rad2deg(angle_rad)
        
        return angle_deg
    
    def get_gaze_direction_label(self, pitch: float, yaw: float) -> str:
        """
        根據視線角度獲取方向標籤（用於可視化）
        
        Args:
            pitch: 俯仰角（度）
            yaw: 偏航角（度）
        
        Returns:
            label: 方向標籤（中文）
        """
        # 定義閾值（度）
        threshold = 15
        
        # 判斷垂直方向
        if pitch > threshold:
            vertical = "向上"
        elif pitch < -threshold:
            vertical = "向下"
        else:
            vertical = ""
        
        # 判斷水平方向
        if yaw > threshold:
            horizontal = "向右"
        elif yaw < -threshold:
            horizontal = "向左"
        else:
            horizontal = ""
        
        # 組合標籤
        if vertical and horizontal:
            label = f"{vertical}{horizontal}"
        elif vertical:
            label = vertical
        elif horizontal:
            label = horizontal
        else:
            label = "正前方"
        
        return label


# ==================== 測試和示例代碼 ====================

def test_gaze_vector_converter():
    """
    測試視線向量轉換器
    """
    print("=" * 70)
    print("測試第五階段：視線向量轉換")
    print("=" * 70)
    
    # 創建轉換器
    converter = GazeVectorConverter()
    
    # ==================== 測試 1：角度 → 向量 ====================
    print("\n" + "=" * 70)
    print("測試 1：角度 → 向量")
    print("=" * 70)
    
    # 測試案例
    test_cases = [
        (0.0, 0.0, "正前方"),
        (0.0, np.deg2rad(30), "向右看 30°"),
        (0.0, np.deg2rad(-30), "向左看 30°"),
        (np.deg2rad(20), 0.0, "向上看 20°"),
        (np.deg2rad(-20), 0.0, "向下看 20°"),
        (np.deg2rad(15), np.deg2rad(15), "向右上看"),
    ]
    
    for pitch, yaw, description in test_cases:
        vector = converter.angles_to_vector(pitch, yaw)
        
        print(f"\n{description}:")
        print(f"  輸入: Pitch={np.rad2deg(pitch):.1f}°, Yaw={np.rad2deg(yaw):.1f}°")
        print(f"  輸出向量: [{vector[0]:.4f}, {vector[1]:.4f}, {vector[2]:.4f}]")
        print(f"  向量長度: {np.linalg.norm(vector):.6f} (應為 1.0)")
    
    # ==================== 測試 2：向量 → 角度 ====================
    print("\n" + "=" * 70)
    print("測試 2：向量 → 角度")
    print("=" * 70)
    
    # 使用前面生成的向量
    for pitch, yaw, description in test_cases:
        vector = converter.angles_to_vector(pitch, yaw)
        pitch_back, yaw_back = converter.vector_to_angles(vector)
        
        print(f"\n{description}:")
        print(f"  原始角度: Pitch={np.rad2deg(pitch):.1f}°, Yaw={np.rad2deg(yaw):.1f}°")
        print(f"  轉換向量: [{vector[0]:.4f}, {vector[1]:.4f}, {vector[2]:.4f}]")
        print(f"  還原角度: Pitch={np.rad2deg(pitch_back):.1f}°, Yaw={np.rad2deg(yaw_back):.1f}°")
        print(f"  誤差: Pitch={abs(np.rad2deg(pitch - pitch_back)):.6f}°, "
              f"Yaw={abs(np.rad2deg(yaw - yaw_back)):.6f}°")
    
    # ==================== 測試 3：批量轉換 ====================
    print("\n" + "=" * 70)
    print("測試 3：批量轉換")
    print("=" * 70)
    
    # 創建批量測試數據
    angles_batch = np.array([
        [0.0, 0.0],
        [np.deg2rad(10), np.deg2rad(20)],
        [np.deg2rad(-15), np.deg2rad(-10)],
    ])
    
    print(f"\n批量輸入角度 ({angles_batch.shape[0]} 個):")
    for i, (p, y) in enumerate(angles_batch):
        print(f"  #{i+1}: Pitch={np.rad2deg(p):.1f}°, Yaw={np.rad2deg(y):.1f}°")
    
    vectors_batch = converter.batch_angles_to_vectors(angles_batch)
    print(f"\n批量輸出向量:")
    for i, vec in enumerate(vectors_batch):
        print(f"  #{i+1}: [{vec[0]:.4f}, {vec[1]:.4f}, {vec[2]:.4f}]")
    
    angles_back = converter.batch_vectors_to_angles(vectors_batch)
    print(f"\n批量還原角度:")
    for i, (p, y) in enumerate(angles_back):
        print(f"  #{i+1}: Pitch={np.rad2deg(p):.1f}°, Yaw={np.rad2deg(y):.1f}°")
    
    # ==================== 測試 4：方向標籤 ====================
    print("\n" + "=" * 70)
    print("測試 4：方向標籤")
    print("=" * 70)
    
    directions = [
        (0, 0),
        (20, 0),
        (-20, 0),
        (0, 20),
        (0, -20),
        (15, 15),
        (-15, -15),
    ]
    
    for pitch_deg, yaw_deg in directions:
        label = converter.get_gaze_direction_label(pitch_deg, yaw_deg)
        print(f"  Pitch={pitch_deg:>4}°, Yaw={yaw_deg:>4}° → {label}")
    
    # ==================== 測試 5：角度誤差計算 ====================
    print("\n" + "=" * 70)
    print("測試 5：角度誤差計算")
    print("=" * 70)
    
    vec1 = np.array([0, 0, 1])  # 正前方
    vec2 = np.array([0.5, 0, 0.866])  # 偏右約 30°
    
    error = converter.calculate_angular_error(vec1, vec2)
    print(f"\n向量 1: {vec1}")
    print(f"向量 2: {vec2}")
    print(f"角度誤差: {error:.2f}°")
    
    print("\n" + "=" * 70)
    print("測試完成！")
    print("=" * 70)


if __name__ == '__main__':
    test_gaze_vector_converter()

