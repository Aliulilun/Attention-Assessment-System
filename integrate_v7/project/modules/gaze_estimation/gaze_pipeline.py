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
        
        # ==================================================
        #  新增：載入基準校正參數 (Baseline Calibration)
        # 直接使用傳入的 config 字典，保持架構簡潔
        # ==================================================
        calib_config = config.get('calibration', {})
        self.pitch_offset_deg = calib_config.get('pitch_offset_deg', 0.0) # μ_pitch (單位：度)
        self.pitch_offset_rad = np.radians(self.pitch_offset_deg)         # 轉為弧度
        
        if self.pitch_offset_deg != 0.0:
            print(f">>> [GazeEstimation] 已啟用基準校正：Pitch 偏移量 = {self.pitch_offset_deg}°")
        
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
            },
            #  新增：防呆預設校正參數區塊
            'calibration': {
                'pitch_offset_deg': 0.0  
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
            if camera_matrix is None:
                camera_matrix = get_default_camera_matrix(w, h)
            
            # Stage 1: 人臉檢測
            face_result = self.face_detector.detect(frame)
            if face_result is None:
                return {'success': False, 'error': 'YOLO failed'}
            
            #  如果特徵點丟失，直接中斷幾何運算，但傳出 face_result 供狀態機盲追蹤
            if face_result.get('landmarks_2d_selected') is None:
                return {
                    'success': False, 
                    'error': 'Landmarks lost but head locked', 
                    'face_result': face_result
                }
            
            # Stage 2: 頭部姿態估計
            pose_result = self.head_pose_estimator.estimate(
                landmarks_2d=face_result['landmarks_2d_selected'],
                camera_matrix=camera_matrix
            )

            #  【核心修復一】將 Stage 3~5 移回正常的 try 區塊內，確保管線暢通
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
            # ==================================================
            #  【核心邏輯】：基準校正 (Baseline Calibration)
            # 數學公式: Pitch_calibrated = Pitch_raw - μ_pitch
            # ==================================================
            if self.pitch_offset_rad != 0.0:
                # 扣除弧度偏移 (pitch 為索引 0)
                gaze_result['gaze_angles'][0] -= self.pitch_offset_rad
                # 同步扣除角度偏移
                gaze_result['gaze_angles_deg'][0] -= self.pitch_offset_deg
            # ==================================================
            
            # Stage 5: 視線向量轉換
            gaze_vector_norm = self.gaze_converter.angles_to_vector(
                pitch=gaze_result['gaze_angles'][0],
                yaw=gaze_result['gaze_angles'][1]
            )
            
            # 幾何還原：將視線向量從歸一化空間逆旋轉回原始相機空間
            if 'R_n' in norm_result:
                R_n = norm_result['R_n']
                gaze_vector_orig = np.dot(R_n.T, gaze_vector_norm)
                gaze_vector = gaze_vector_orig / (np.linalg.norm(gaze_vector_orig) + 1e-6)
            else:
                gaze_vector = gaze_vector_norm
            
            # 計算左右眼中心位置
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
                'left_eye': tuple(left_eye_center),
                'right_eye': tuple(right_eye_center),
                #  【核心修復二】將 Stage 1 的關鍵特徵欄位同步封裝至成功字典，供下游狀態機讀取
                'yolo_head_bbox': face_result.get('yolo_head_bbox'),
                'num_landmarks': face_result.get('num_landmarks', 468)
            }
            
            return result
            
        except Exception as e:
            return {'success': False, 'error': str(e)}