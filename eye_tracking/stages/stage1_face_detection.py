"""
第一階段：人臉與特徵點檢測 (YOLO + MediaPipe Tasks API 兩階段級聯架構)
Stage 1: Face Detection (Two-stage Cascade with Modern Tasks API)

架構邏輯：
1. 使用 YOLO (nano.pt) 鎖定頭部全域 ROI。
2. 進行動態填充 (Dynamic Padding) 並裁切影像。
3. 使用 MediaPipe Tasks API (face_landmarker.task) 在高解析度 ROI 上提取特徵點。
4. 執行仿射平移變換，將局部座標映射回全域座標系，確保 Stage 2/3 幾何一致性。
"""

import cv2
import numpy as np
from typing import Dict, Optional
from pathlib import Path
import warnings

# 抑制部分底層 C++ 警告
warnings.filterwarnings("ignore", category=UserWarning)

# 導入新版 MediaPipe Tasks API
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("請確保已安裝 ultralytics 套件: pip install ultralytics")

"""
第一階段：人臉與特徵點檢測 (YOLO + MediaPipe Tasks API - 完備資料結構版)
Stage 1: Face Detection (Two-stage Cascade with Comprehensive Return Schema)

針對高解析度大圖與下游視覺化優化：
1. 自動進行「中央區域前置裁切」(空間先驗)，確保小目標解像力並維持 640 運算速度。
2. YOLO (nano.pt) 提取頭部位置與置信度 (Confidence)。
3. MediaPipe Tasks 進行高精度局部特徵點回歸。
4. 補全完備的回傳資料結構，完全滿足下游 Stage 2/3 與視覺化繪圖模組的需求。
"""

import cv2
import numpy as np
from typing import Dict, Optional
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("請確保已安裝 ultralytics 套件: pip install ultralytics")

class FaceDetector:
    def __init__(self, config: Dict = None):
        if config is None:
            config = {}
            
        # 1. 初始化 YOLO 模型
        yolo_path = config.get('yolo_model_path', 'models/nano.pt')
        if not Path(yolo_path).exists():
            raise FileNotFoundError(f"找不到 YOLO 模型檔案: {yolo_path}")
        self.yolo_model = YOLO(yolo_path)
        self.yolo_conf = config.get('yolo_conf', 0.4)
        
        # 2. 初始化 MediaPipe Tasks
        mp_task_path = config.get('face_landmarker_task', 'models/face_landmarker.task')
        if not Path(mp_task_path).exists():
            raise FileNotFoundError(f"找不到 MediaPipe 模型檔案: {mp_task_path}")
            
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=mp_task_path),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=config.get('min_confidence', 0.5),
            min_face_presence_confidence=0.5
        )
        self.face_landmarker = FaceLandmarker.create_from_options(options)
        
        # 3. 系統常數與空間先驗配置
        self.ethxgaze_indices = [33, 133, 362, 263, 61, 291]
        self.crop_padding_ratio = config.get('crop_padding_ratio', 0.2)
        
        # 預設中央聚焦區域尺寸
        self.target_center_w = config.get('target_center_w', 1600)
        self.target_center_h = config.get('target_center_h', 960)
        
        print("Stage 1 (空間先驗 + 完備回傳版) 初始化完成")

    def detect(self, image: np.ndarray) -> Optional[Dict]:
        img_h, img_w = image.shape[:2]
        
        # ==========================================
        # 階段 0: 自動計算中央空間先驗裁切範圍
        # ==========================================
        if img_w > self.target_center_w and img_h > self.target_center_h:
            center_x_offset = (img_w - self.target_center_w) // 2
            center_y_offset = (img_h - self.target_center_h) // 2
            
            center_x_max = center_x_offset + self.target_center_w
            center_y_max = center_y_offset + self.target_center_h
            
            focused_image = image[center_y_offset:center_y_max, center_x_offset:center_x_max]
            focused_h, focused_w = focused_image.shape[:2]
        else:
            center_x_offset = 0
            center_y_offset = 0
            focused_image = image
            focused_h, focused_w = img_h, img_w
        
        # ==========================================
        # 階段 1: YOLO 多頭部偵測與目標篩選 (幼兒優先)
        # ==========================================
        results = self.yolo_model.predict(focused_image, conf=self.yolo_conf, imgsz=640, verbose=False)
        
        if len(results) == 0 or len(results[0].boxes) == 0:
            return None
        
        # 提取所有被偵測到的 BBoxes 資訊
        all_boxes = results[0].boxes.xyxy.cpu().numpy()  # 格式: [N, 4] -> [x_min, y_min, x_max, y_max]
        all_confs = results[0].boxes.conf.cpu().numpy()  # 格式: [N]
        
        best_box_local = None
        min_area = float('inf')
        chosen_confidence = 0.0
        
        # 遍歷所有偵測到的頭部，執行「最小面積過濾」
        for box, conf in zip(all_boxes, all_confs):
            x_min_l, y_min_l, x_max_l, y_max_l = box
            
            # 計算該邊界框的像素面積 (Scale Measurement)
            box_w = x_max_l - x_min_l
            box_h = y_max_l - y_min_l
            area = box_w * box_h
            
            # 學術安全防護：排除異常微小（像素面積小於 40x40）的背景雜訊誤檢
            if box_w < 40 or box_h < 40:
                continue
                
            # 核心邏輯：尋找面積最小的邊界框（理論上為幼兒）
            if area < min_area:
                min_area = area
                best_box_local = box
                chosen_confidence = float(conf)
        
        # 若經過過濾後沒有合格的 BBox，則終止
        if best_box_local is None:
            return None
            
        x_min_local, y_min_local, x_max_local, y_max_local = best_box_local
        confidence = chosen_confidence
        
        #  將選定的 YOLO 偵測框精確還原回原始大圖的全域座標系
        x_min = x_min_local + center_x_offset
        y_min = y_min_local + center_y_offset
        x_max = x_max_local + center_x_offset
        y_max = y_max_local + center_y_offset
        
        box_w = x_max_local - x_min_local
        box_h = y_max_local - y_min_local
        
        # ==========================================
        # 階段 1.5: 動態 Padding 與頭部 ROI 裁切
        # ==========================================
        pad_x = int(box_w * self.crop_padding_ratio)
        pad_y = int(box_h * self.crop_padding_ratio)
        
        crop_x_min_local = max(0, int(x_min_local - pad_x))
        crop_y_min_local = max(0, int(y_min_local - pad_y))
        crop_x_max_local = min(focused_w, int(x_max_local + pad_x))
        crop_y_max_local = min(focused_h, int(y_max_local + pad_y))
        
        roi_image = focused_image[crop_y_min_local:crop_y_max_local, crop_x_min_local:crop_x_max_local]
        roi_h, roi_w = roi_image.shape[:2]
        
        if roi_h < 10 or roi_w < 10:
            return None
        
        # ==========================================
        # 階段 2: MediaPipe Tasks 局部特徵點回歸
        # ==========================================
        roi_rgb = cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=roi_rgb)
        mp_results = self.face_landmarker.detect(mp_image)
        
        if not mp_results.face_landmarks:
            return None
            
        face_landmarks = mp_results.face_landmarks[0]
        
        # ==========================================
        # 階段 3: 雙重平移座標全域回歸與結構解構
        # ==========================================
        local_landmarks = np.array([
            [lm.x * roi_w, lm.y * roi_h] for lm in face_landmarks
        ])
        
        # 雙重平移補償：加上 YOLO 局部起點與空間先驗固定起點
        total_offset_x = crop_x_min_local + center_x_offset
        total_offset_y = crop_y_min_local + center_y_offset
        global_landmarks = local_landmarks + np.array([total_offset_x, total_offset_y])
        
        # 提取標準 6 點
        selected_landmarks = global_landmarks[self.ethxgaze_indices]
        
        # 計算人臉全域邊界框 (Face BBox)
        face_x_min = np.min(global_landmarks[:, 0])
        face_y_min = np.min(global_landmarks[:, 1])
        face_x_max = np.max(global_landmarks[:, 0])
        face_y_max = np.max(global_landmarks[:, 1])
        
        # ==========================================
        # 階段 4: 返回完全向後相容的資料結構
        # ==========================================
        return {
            'num_landmarks': 468,  # MediaPipe Face Mesh 標準特徵點數量
            'landmarks_468': global_landmarks.astype(np.float32),
            'landmarks_2d_selected': selected_landmarks.astype(np.float32),
            'bbox': [face_x_min, face_y_min, face_x_max, face_y_max],
            'yolo_head_bbox': [x_min, y_min, x_max, y_max], # 附加原始 YOLO 框供視覺化參考
            'confidence': confidence # YOLO 偵測信心值
        }
    
    def visualize_landmarks(self, image: np.ndarray, 
                          landmarks_468: np.ndarray = None,
                          landmarks_selected: np.ndarray = None,
                          bbox: list = None,
                          show_all: bool = False) -> np.ndarray:
        """
        在圖像上可視化特徵點和邊界框
        
        Args:
            image: 輸入圖像
            landmarks_468: 所有 468 個特徵點（可選）
            landmarks_selected: 選定的關鍵點（可選）
            bbox: 人臉邊界框（可選）
            show_all: 是否顯示所有 468 個點（默認只顯示選定的點）
        
        Returns:
            標註後的圖像
        """
        vis_image = image.copy()
        
        # 繪製邊界框
        if bbox is not None:
            x_min, y_min, x_max, y_max = map(int, bbox)
            cv2.rectangle(vis_image, (x_min, y_min), (x_max, y_max), 
                         (0, 255, 0), 2)
            cv2.putText(vis_image, 'Face', (x_min, y_min - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 繪製所有 468 個點（如果需要）
        if show_all and landmarks_468 is not None:
            for point in landmarks_468:
                x, y = int(point[0]), int(point[1])
                cv2.circle(vis_image, (x, y), 1, (200, 200, 200), -1)
        
        # 繪製選定的關鍵點（更明顯）
        if landmarks_selected is not None:
            for idx, point in enumerate(landmarks_selected):
                x, y = int(point[0]), int(point[1])
                cv2.circle(vis_image, (x, y), 4, (0, 255, 0), -1)
                cv2.circle(vis_image, (x, y), 5, (0, 0, 255), 1)
        
        return vis_image

# ==================== 測試代碼 ====================
if __name__ == "__main__":
    print("==================================================")
    print("執行模組獨立測試: Stage 1 (YOLO + Tasks API)")
    print("==================================================")
    
    # 建立設定檔 (請確保路徑符合您的實際目錄結構)
    test_config = {
        'yolo_model_path': 'models/nano.pt',
        'face_landmarker_task': 'models/face_landmarker.task'
    }
    
    try:
        detector = FaceDetector(test_config)
        
        # 創建虛擬測試圖像 (H=720, W=1280, 3通道)
        test_img = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        # 執行一次空測試以預熱模型 (Warm-up)
        print("\n正在執行預熱推論...")
        res = detector.detect(test_img)
        
        print("\n模組建構與預熱成功！準備就緒。")
        
    except FileNotFoundError as e:
        print(f"\n 錯誤: {e}")
        print("請檢查 models/ 目錄下是否具備 nano.pt 與 face_landmarker.task")
    except Exception as e:
        print(f"\n 初始化發生未預期錯誤: {e}")