"""
視線估計流程整合
Gaze Estimation Pipeline

將 5 個階段封裝成單一的流程類，方便在 main.py 中調用
"""

import cv2
import numpy as np
from pathlib import Path

from .stage1_face_detection import FaceDetector
from .stage2_head_pose import HeadPoseEstimator
from .stage3_normalization import ImageNormalizer
from .stage4_gaze_network import GazeEstimator
from .stage5_gaze_vector import GazeVectorConverter
from .camera_utils import get_default_camera_matrix


class GazeEstimationPipeline:
    """視線估計完整流程"""
    
    def __init__(self, config=None):
        """
        初始化視線估計流程
        
        Args:
            config: 配置字典，包含各階段的配置
        """
        if config is None:
            config = self._get_default_config()
        
        print(">>> [GazeEstimation] 初始化視線估計模組...")
        
        # 初始化各個階段
        self.face_detector = FaceDetector(config=config.get('face_detection', {}))
        self.head_pose_estimator = HeadPoseEstimator(config=config.get('head_pose', {}))
        self.image_normalizer = ImageNormalizer(config=config.get('normalization', {}))
        self.gaze_estimator = GazeEstimator(config=config.get('model', {}))
        self.gaze_converter = GazeVectorConverter()
        
        print(">>> [GazeEstimation] 視線估計模組載入完成！")
    
    def _get_default_config(self):
        """獲取默認配置"""
        return {
            'face_detection': {
                'min_confidence': 0.3,
                'yolo_model_path': 'model/gaze/nano.pt',
                'face_landmarker_task': 'model/gaze/face_landmarker.task'
            },
            'head_pose': {
                'face_model_path': 'model/gaze/face_model_ethxgaze.txt',
                'use_iterative': True
            },
            'normalization': {
                'output_size': (224, 224),
                'focal_norm': 960.0,
                'distance_norm': 60.0,
                'face_model_path': 'model/gaze/face_model_ethxgaze.txt'
            },
            'model': {
                'model_path': 'model/gaze/epoch_24_ckpt.pth.tar',
                'use_gpu': True
            }
        }
    
    def estimate(self, frame, camera_matrix=None):
        """
        執行完整的視線估計流程
        
        Args:
            frame: 輸入圖像 (BGR)
            camera_matrix: 相機內參矩陣（可選）
        
        Returns:
            result: 視線估計結果字典，包含：
                - success: 是否成功
                - gaze_angles: [pitch, yaw] 視線角度（弧度）
                - gaze_angles_deg: [pitch, yaw] 視線角度（度）
                - gaze_vector: [x, y, z] 3D 視線向量
                - face_bbox: 人臉邊界框
                - confidence: 人臉檢測置信度
                - ... 其他中間結果
        """
        try:
            h, w = frame.shape[:2]
            
            # 生成相機內參矩陣
            if camera_matrix is None:
                camera_matrix = get_default_camera_matrix(w, h)
            
            # Stage 1: 人臉檢測
            face_result = self.face_detector.detect(frame)
            if face_result is None:
                return {'success': False, 'error': 'Face detection failed'}
            
            # Stage 2: 頭部姿態估計
            pose_result = self.head_pose_estimator.estimate(
                landmarks_2d=face_result['landmarks_2d_selected'],
                camera_matrix=camera_matrix
            )
            if not pose_result['success']:
                return {'success': False, 'error': 'Head pose estimation failed'}
            
            # Stage 3: 圖像正規化
            norm_result = self.image_normalizer.normalize(
                image=frame,
                rotation_vector=pose_result['rvec'],
                translation_vector=pose_result['tvec'],
                camera_matrix=camera_matrix
            )
            if not norm_result['success']:
                return {'success': False, 'error': 'Image normalization failed'}
            
            # Stage 4: 神經網絡推理
            gaze_result = self.gaze_estimator.estimate(norm_result['normalized_image'])
            if not gaze_result['success']:
                return {'success': False, 'error': 'Gaze estimation failed'}
            
            # Stage 5: 視線向量轉換
            gaze_vector = self.gaze_converter.angles_to_vector(
                pitch=gaze_result['gaze_angles'][0],
                yaw=gaze_result['gaze_angles'][1]
            )
            
            # 計算左右眼中心位置（用於視線箭頭繪製）
            # ETH-XGaze 6點: [33, 133, 362, 263, 61, 291]
            # 索引: [右眼外, 右眼內, 左眼外, 左眼內, 左嘴角, 右嘴角]
            landmarks_selected = face_result['landmarks_2d_selected']
            right_eye_center = ((landmarks_selected[0] + landmarks_selected[1]) / 2).astype(int)
            left_eye_center = ((landmarks_selected[2] + landmarks_selected[3]) / 2).astype(int)
            
            # 組裝完整結果
            result = {
                'success': True,
                'gaze_angles': gaze_result['gaze_angles'],
                'gaze_angles_deg': gaze_result['gaze_angles_deg'],
                'gaze_vector': gaze_vector,
                'face_bbox': face_result['bbox'],
                'confidence': face_result.get('confidence', 1.0),
                'head_pose': pose_result['euler_angles'],
                'landmarks': face_result['landmarks_2d_selected'],
                'left_eye': tuple(left_eye_center),   # 新增：左眼中心
                'right_eye': tuple(right_eye_center)  # 新增：右眼中心
            }
            
            return result
        
        except Exception as e:
            return {'success': False, 'error': str(e)}