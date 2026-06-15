"""
可視化工具
Visualization Utilities

提供各種可視化功能，包括視線向量繪製
"""

import cv2
import numpy as np
from typing import Tuple, Optional

# 創建一個全局的 GazeVectorConverter 實例，避免重複初始化
_gaze_converter = None

def _get_gaze_converter():
    """獲取 GazeVectorConverter 單例"""
    global _gaze_converter
    if _gaze_converter is None:
        from stages.stage5_gaze_vector import GazeVectorConverter
        _gaze_converter = GazeVectorConverter()
    return _gaze_converter


def draw_gaze_arrow(image: np.ndarray,
                   center: Tuple[int, int],
                   pitch: float,
                   yaw: float,
                   gaze_vector: Optional[np.ndarray] = None, # 新增此參數
                   length: float = 150.0,
                   thickness: int = 3,
                   color: Tuple[int, int, int] = (0, 0, 255),
                   tip_length: float = 0.3) -> np.ndarray:
    """
    在圖像上繪製視線方向箭頭
    
    優化邏輯：
    1. 優先使用 gaze_vector 的分量，避免重複計算三角函數。
    2. 確保 2D 投影座標 (dx, dy) 與 3D 空間座標系完全自洽。
    """
    image_out = image.copy()
    
    # 確保是彩色圖像
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
    
    # --- 核心計算區塊 ---
    if gaze_vector is not None:
        # 方案 A (推薦)：直接從 3D 單位向量提取投影分量
        # 由於 V_x 正值代表向右，V_y 負值代表向上（符合 OpenCV 座標系）
        dx = length * gaze_vector[0]
        dy = length * gaze_vector[1]
    else:
        # 方案 B (備用)：若無預算好的向量，則執行已修正符號的三角函數
        # 此處公式須與 Stage 5 的修正版完全對齊
        dx = -length * np.cos(pitch) * np.sin(yaw)
        dy = -length * np.sin(pitch)
    
    end_point = (
        int(center[0] + dx),
        int(center[1] + dy)
    )
    
    # 繪製箭頭
    cv2.arrowedLine(
        image_out,
        center,
        end_point,
        color,
        thickness,
        cv2.LINE_AA,
        tipLength=tip_length
    )
    
    # 在起點繪製一個小圓點
    cv2.circle(image_out, center, 5, color, -1)
    
    return image_out
    
    # 繪製箭頭
    cv2.arrowedLine(
        image_out,
        center,
        end_point,
        color,
        thickness,
        cv2.LINE_AA,
        tipLength=tip_length
    )
    
    # 在起點繪製一個小圓點（表示視線起點）
    cv2.circle(image_out, center, 5, color, -1)
    
    return image_out


def draw_gaze_with_face_box(image: np.ndarray,
                            face_bbox: Tuple[int, int, int, int],
                            pitch: float,
                            yaw: float,
                            gaze_vector: Optional[np.ndarray] = None,
                            nose_tip: Optional[Tuple[int, int]] = None,
                            left_eye: Optional[Tuple[int, int]] = None,
                            right_eye: Optional[Tuple[int, int]] = None,
                            confidence: Optional[float] = None, 
                            show_angles: bool = True,
                            show_direction_label: bool = True,
                            show_gaze_vector: bool = True,
                            bbox_format: str = 'xywh') -> np.ndarray:
    """
    繪製人臉框和視線箭頭（完整版，像你的附圖）
    
    Args:
        image: 輸入圖像
        face_bbox: 人臉邊界框
            - 如果 bbox_format='xywh': (x, y, width, height)
            - 如果 bbox_format='xyxy': (x_min, y_min, x_max, y_max)
        pitch: 俯仰角（弧度）
        yaw: 偏航角（弧度）
        gaze_vector: 3D 視線向量 [x, y, z]（可選）
        nose_tip: 鼻尖座標 (x, y)（可選）
        left_eye: 左眼中心座標 (x, y)（推薦）
        right_eye: 右眼中心座標 (x, y)（推薦）
        show_angles: 是否顯示角度數值
        show_direction_label: 是否顯示方向標籤
        show_gaze_vector: 是否顯示 3D 視線向量
        bbox_format: 邊界框格式 ('xywh' 或 'xyxy')
    
    Returns:
        繪製後的圖像
    """
    image_out = image.copy()
    
    # 解析邊界框座標
    if bbox_format == 'xyxy':
        # [x_min, y_min, x_max, y_max] 格式
        x_min, y_min, x_max, y_max = face_bbox
        x_min, y_min = int(x_min), int(y_min)
        x_max, y_max = int(x_max), int(y_max)
        x, y = x_min, y_min
        w, h = x_max - x_min, y_max - y_min
    else:
        # [x, y, width, height] 格式
        x, y, w, h = face_bbox
        x, y, w, h = int(x), int(y), int(w), int(h)
        x_min, y_min = x, y
        x_max, y_max = x + w, y + h
    
    # 1. 繪製人臉框（綠色，像附圖那樣）
    cv2.rectangle(image_out, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    
    # 2. 繪製雙眼視線箭頭（左眼和右眼各一個）
    arrow_length = min(w, h) * 0.8  # 箭頭更小
    arrow_thickness = 2  # 更細的箭頭
    
    if left_eye is not None and right_eye is not None:
        # 在左眼和右眼各繪製一個箭頭
        left_eye_pos = (int(left_eye[0]), int(left_eye[1]))
        right_eye_pos = (int(right_eye[0]), int(right_eye[1]))
        
        # 左眼箭頭
        image_out = draw_gaze_arrow(
            image_out,
            left_eye_pos,
            pitch,
            yaw,
            length=arrow_length,
            thickness=arrow_thickness,
            color=(0, 0, 255)  # 紅色箭頭
        )
        
        # 右眼箭頭
        image_out = draw_gaze_arrow(
            image_out,
            right_eye_pos,
            pitch,
            yaw,
            length=arrow_length,
            thickness=arrow_thickness,
            color=(0, 0, 255)  # 紅色箭頭
        )
    elif nose_tip is not None:
        # 備用方案：如果沒有眼睛座標，使用鼻尖
        arrow_start = (int(nose_tip[0]), int(nose_tip[1]) - int(h * 0.1))
        image_out = draw_gaze_arrow(
            image_out,
            arrow_start,
            pitch,
            yaw,
            length=arrow_length,
            thickness=arrow_thickness,
            color=(0, 0, 255)
        )
    else:
        # 最後備用：使用人臉框中心
        face_center_x = x_min + w // 2
        face_center_y = y_min + int(h * 0.35)
        arrow_start = (face_center_x, face_center_y)
        image_out = draw_gaze_arrow(
            image_out,
            arrow_start,
            pitch,
            yaw,
            length=arrow_length,
            thickness=arrow_thickness,
            color=(0, 0, 255)
        )
    
    # 3. 顯示角度信息和 3D 視線向量（在人臉框上方）
    if show_angles or show_gaze_vector:
        pitch_deg = np.rad2deg(pitch)
        yaw_deg = np.rad2deg(yaw)
        
        # 組合顯示文字
        text_lines = []
        if confidence is not None:
            text_lines.append(f"Conf:{confidence:.2f}") 
        if show_angles:
            text_lines.append(f"P:{pitch_deg:.1f} Y:{yaw_deg:.1f}")
        if show_gaze_vector and gaze_vector is not None:
            text_lines.append(f"Vec:[{gaze_vector[0]:.2f},{gaze_vector[1]:.2f},{gaze_vector[2]:.2f}]")
        
        # 顯示第一行（角度）
        if len(text_lines) > 0:
            text = text_lines[0]
            text_pos = (x_min, y_min - 30 if len(text_lines) > 1 else y_min - 10)
        
            # 添加文字背景（讓文字更清晰）
            (text_width, text_height), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                image_out,
                (text_pos[0] - 2, text_pos[1] - text_height - 2),
                (text_pos[0] + text_width + 2, text_pos[1] + 2),
                (0, 0, 0),
                -1
            )
            
            cv2.putText(
                image_out,
                text,
                text_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA
            )
        
        # 顯示第二行（3D 向量）
        if len(text_lines) > 1:
            text2 = text_lines[1]
            text_pos2 = (x_min, y_min - 10)
            
            (text_width2, text_height2), _ = cv2.getTextSize(
                text2, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
            )
            cv2.rectangle(
                image_out,
                (text_pos2[0] - 2, text_pos2[1] - text_height2 - 2),
                (text_pos2[0] + text_width2 + 2, text_pos2[1] + 2),
                (0, 0, 0),
                -1
            )
            
            cv2.putText(
                image_out,
                text2,
                text_pos2,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 0),  # 青色
                1,
                cv2.LINE_AA
            )
    
    # 4. 顯示方向標籤（在人臉框下方）- 使用英文避免中文亂碼
    if show_direction_label:
        pitch_deg = np.rad2deg(pitch)
        yaw_deg = np.rad2deg(yaw)
        threshold = 15
        
        # 判斷方向（英文）
        if pitch_deg > threshold:
            vertical = "Up"
        elif pitch_deg < -threshold:
            vertical = "Down"
        else:
            vertical = ""
        
        if yaw_deg > threshold:
            horizontal = "Right"
        elif yaw_deg < -threshold:
            horizontal = "Left"
        else:
            horizontal = ""
        
        # 組合標籤
        if vertical and horizontal:
            direction = f"{vertical}-{horizontal}"
        elif vertical:
            direction = vertical
        elif horizontal:
            direction = horizontal
        else:
            direction = "Center"
        
        text_pos = (x_min, y_max + 25)
        
        # 添加文字背景
        (text_width, text_height), _ = cv2.getTextSize(
            direction, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )
        cv2.rectangle(
            image_out,
            (text_pos[0] - 2, text_pos[1] - text_height - 2),
            (text_pos[0] + text_width + 2, text_pos[1] + 2),
            (0, 0, 0),
            -1
        )
        
        cv2.putText(
            image_out,
            direction,
            text_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),  # 黃色
            2,
            cv2.LINE_AA
        )
    
    return image_out


def draw_multiple_gazes(image: np.ndarray,
                       faces_data: list) -> np.ndarray:
    """
    繪製多個人臉的視線（支援多人場景，像你的附圖有兩個孩子）
    
    Args:
        image: 輸入圖像
        faces_data: 人臉數據列表，每個元素包含：
            {
                'bbox': (x, y, w, h),
                'pitch': float,
                'yaw': float,
                'gaze_vector': np.ndarray (可選)
            }
    
    Returns:
        繪製後的圖像
    """
    image_out = image.copy()
    
    for i, face_data in enumerate(faces_data):
        bbox = face_data['bbox']
        pitch = face_data['pitch']
        yaw = face_data['yaw']
        gaze_vector = face_data.get('gaze_vector', None)
        
        image_out = draw_gaze_with_face_box(
            image_out,
            bbox,
            pitch,
            yaw,
            gaze_vector,
            show_angles=True,
            show_direction_label=True
        )
        
        # 在左上角標註人臉編號
        cv2.putText(
            image_out,
            f"Face {i+1}",
            (bbox[0], bbox[1] - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),  # 青色
            2,
            cv2.LINE_AA
        )
    
    return image_out


def create_gaze_visualization_grid(original_image: np.ndarray,
                                   face_bbox: Tuple[int, int, int, int],
                                   normalized_image: np.ndarray,
                                   pitch: float,
                                   yaw: float,
                                   gaze_vector: np.ndarray) -> np.ndarray:
    """
    創建視線可視化網格（原始圖像 + 正規化圖像）
    
    Args:
        original_image: 原始圖像
        face_bbox: 人臉框
        normalized_image: 正規化圖像
        pitch: 俯仰角
        yaw: 偏航角
        gaze_vector: 3D 視線向量
    
    Returns:
        組合後的可視化圖像
    """
    # 在原始圖像上繪製
    vis_original = draw_gaze_with_face_box(
        original_image,
        face_bbox,
        pitch,
        yaw
    )
    
    # 在正規化圖像上繪製（居中）
    h, w = normalized_image.shape[:2]
    center = (w // 2, h // 2)
    vis_normalized = draw_gaze_arrow(
        normalized_image,
        center,
        pitch,
        yaw,
        length=80,
        thickness=2,
        color=(0, 0, 255)
    )
    
    # 添加標題
    cv2.putText(vis_normalized, 'Normalized View', (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # 調整尺寸並組合
    h_orig = vis_original.shape[0]
    vis_normalized_resized = cv2.resize(vis_normalized, (h_orig, h_orig))
    
    # 並排顯示
    combined = np.hstack([vis_original, vis_normalized_resized])
    
    return combined


# ==================== 舊版向後兼容函數 ====================
# 這些函數保留以支持現有代碼

def draw_landmarks(image: np.ndarray,
                  landmarks: np.ndarray,
                  color: Tuple[int, int, int] = (0, 255, 0),
                  radius: int = 2) -> np.ndarray:
    """
    在圖像上繪製特徵點
    
    Args:
        image: 輸入圖像
        landmarks: 特徵點座標 (N, 2)
        color: 顏色 (B, G, R)
        radius: 點的半徑
    
    Returns:
        繪製後的圖像
    """
    image_out = image.copy()
    
    for landmark in landmarks:
        x, y = int(landmark[0]), int(landmark[1])
        cv2.circle(image_out, (x, y), radius, color, -1)
    
    return image_out


def draw_head_pose_axes(image: np.ndarray,
                       rvec: np.ndarray,
                       tvec: np.ndarray,
                       camera_matrix: np.ndarray,
                       length: float = 50.0) -> np.ndarray:
    """
    繪製頭部姿態 3D 坐標軸
    
    Args:
        image: 輸入圖像
        rvec: 旋轉向量
        tvec: 平移向量
        camera_matrix: 相機內參矩陣
        length: 坐標軸長度
    
    Returns:
        繪製後的圖像
    """
    image_out = image.copy()
    
    # 定義 3D 坐標軸點
    axis_points_3d = np.float32([
        [0, 0, 0],           # 原點
        [length, 0, 0],      # X 軸（紅色）
        [0, length, 0],      # Y 軸（綠色）
        [0, 0, length]       # Z 軸（藍色）
    ])
    
    # 投影到 2D
    dist_coeffs = np.zeros((4, 1))
    axis_points_2d, _ = cv2.projectPoints(
        axis_points_3d,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs
    )
    
    # 轉換為整數座標
    origin = tuple(axis_points_2d[0].ravel().astype(int))
    x_axis = tuple(axis_points_2d[1].ravel().astype(int))
    y_axis = tuple(axis_points_2d[2].ravel().astype(int))
    z_axis = tuple(axis_points_2d[3].ravel().astype(int))
    
    # 繪製坐標軸
    cv2.line(image_out, origin, x_axis, (0, 0, 255), 3)  # X 軸：紅色
    cv2.line(image_out, origin, y_axis, (0, 255, 0), 3)  # Y 軸：綠色
    cv2.line(image_out, origin, z_axis, (255, 0, 0), 3)  # Z 軸：藍色
    
    return image_out


def create_result_grid(images: list,
                      titles: Optional[list] = None,
                      grid_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    創建圖像網格（用於並排顯示多個結果）
    
    Args:
        images: 圖像列表
        titles: 標題列表（可選）
        grid_size: 網格尺寸 (rows, cols)，如果為 None 則自動計算
    
    Returns:
        組合後的網格圖像
    """
    if not images:
        raise ValueError("圖像列表不能為空")
    
    n_images = len(images)
    
    # 自動計算網格尺寸
    if grid_size is None:
        cols = int(np.ceil(np.sqrt(n_images)))
        rows = int(np.ceil(n_images / cols))
        grid_size = (rows, cols)
    else:
        rows, cols = grid_size
    
    # 確保所有圖像大小相同
    target_h, target_w = images[0].shape[:2]
    resized_images = []
    
    for img in images:
        if img.shape[:2] != (target_h, target_w):
            img = cv2.resize(img, (target_w, target_h))
        
        # 確保是 3 通道
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        resized_images.append(img)
    
    # 添加標題（如果提供）
    if titles is not None:
        for i, (img, title) in enumerate(zip(resized_images, titles)):
            cv2.putText(
                img, title, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2
            )
    
    # 創建網格
    grid_rows = []
    for i in range(rows):
        row_images = []
        for j in range(cols):
            idx = i * cols + j
            if idx < n_images:
                row_images.append(resized_images[idx])
            else:
                # 填充空白圖像
                blank = np.zeros_like(resized_images[0])
                row_images.append(blank)
        
        grid_rows.append(np.hstack(row_images))
    
    grid = np.vstack(grid_rows)
    
    return grid
