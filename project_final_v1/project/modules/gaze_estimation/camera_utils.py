"""
相機參數處理工具
Camera Parameter Utilities
"""

import numpy as np
import cv2
from pathlib import Path


def get_default_camera_matrix(width, height, focal_length_factor=1.0):
    """
    生成默認相機內參矩陣
    
    Args:
        width: 圖像寬度
        height: 圖像高度
        focal_length_factor: 焦距係數（焦距 = width * factor）
    
    Returns:
        camera_matrix: 3x3 相機內參矩陣
    """
    focal_length = width * focal_length_factor
    center_x = width / 2.0
    center_y = height / 2.0
    
    camera_matrix = np.array([
        [focal_length, 0, center_x],
        [0, focal_length, center_y],
        [0, 0, 1]
    ], dtype=np.float32)
    
    return camera_matrix


def load_camera_calibration(calibration_file):
    """
    從 OpenCV XML 文件載入相機校正參數
    
    Args:
        calibration_file: 相機校正文件路徑 (.xml)
    
    Returns:
        dict: 包含 'camera_matrix' 和 'distortion_coeffs' 的字典
    """
    calibration_file = Path(calibration_file)
    
    if not calibration_file.exists():
        raise FileNotFoundError(f"相機校正文件不存在: {calibration_file}")
    
    fs = cv2.FileStorage(str(calibration_file), cv2.FILE_STORAGE_READ)
    
    camera_matrix = fs.getNode('Camera_Matrix').mat()
    distortion_coeffs = fs.getNode('Distortion_Coefficients').mat()
    
    fs.release()
    
    if camera_matrix is None:
        raise ValueError(f"無法從文件中讀取 Camera_Matrix: {calibration_file}")
    
    return {
        'camera_matrix': camera_matrix,
        'distortion_coeffs': distortion_coeffs
    }


def estimate_camera_matrix_from_fov(width, height, fov_degrees=60):
    """
    根據視野角度估計相機內參矩陣
    
    Args:
        width: 圖像寬度
        height: 圖像高度
        fov_degrees: 水平視野角度（度）
    
    Returns:
        camera_matrix: 3x3 相機內參矩陣
    """
    fov_rad = np.deg2rad(fov_degrees)
    focal_length = width / (2 * np.tan(fov_rad / 2))
    
    camera_matrix = np.array([
        [focal_length, 0, width / 2.0],
        [0, focal_length, height / 2.0],
        [0, 0, 1]
    ], dtype=np.float32)
    
    return camera_matrix


def print_camera_info(camera_matrix):
    """
    打印相機參數資訊
    
    Args:
        camera_matrix: 3x3 相機內參矩陣
    """
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    
    print("相機內參矩陣:")
    print(f"  焦距 fx: {fx:.2f} pixels")
    print(f"  焦距 fy: {fy:.2f} pixels")
    print(f"  主點 cx: {cx:.2f} pixels")
    print(f"  主點 cy: {cy:.2f} pixels")

