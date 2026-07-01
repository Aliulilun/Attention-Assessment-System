import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO

class ModelManager:
    def __init__(self, model_dir):
        print(">>> [ModelManager] 初始化階段適應性物件偵測器...")
        self.model_dir = model_dir
        self.models = {}

        # ==========================================
        # 🌟 硬體優化邏輯
        # ==========================================
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            print(f">>> [ModelManager] 偵測到 GPU：{torch.cuda.get_device_name(0)}，啟用 CUDA 加速！")
            torch.backends.cudnn.benchmark = True # 讓 cuDNN 自動尋找最適合的卷積演算法
        else:
            print(">>> [ModelManager] ⚠️ 沒有偵測到 CUDA，將使用 CPU 進行推論，速度會較慢。")

        # ==========================================
        # 🌟 階段與模型映射表
        # ==========================================
        self.stage_model_map = {
            1: "front_model.pt",
            2: "front_model.pt",
            3: "background_model.pt",
            4: "background_model.pt",
            5: "balloon_model.pt",
            6: "doll_model.pt",        
            7: "toy_model.pt",
            9: "tablet_model.pt",
            11: "front_model.pt",       # ✅ 新增：第 11~12 階段 (前方物品)
            12: "front_model.pt",
            13: "background_model.pt",  # ✅ 新增：第 13~14 階段 (背景物品)
            14: "background_model.pt"
        }

    def _get_model(self, model_name):
        """
        惰性載入模型 (Lazy Loading)：避免一次載入太多模型導致顯存 (VRAM) 爆炸
        """
        if model_name not in self.models:
            model_path = os.path.join(self.model_dir, model_name)
            if not os.path.exists(model_path):
                print(f"⚠️ [ModelManager] 警告：找不到模型檔案 {model_path}")
                return None
            
            print(f">>> [ModelManager] 正在將模型載入顯存：{model_name}")
            self.models[model_name] = YOLO(model_path)
            
        return self.models[model_name]

    def detect_objects(self, frame, stage):
        """
        執行 YOLO 偵測。
        - 階段 1~8：回傳目標物 boxes (單一 list)
        - 階段 9~14：回傳 (target_boxes, robot_boxes) 雙框 tuple，提供主程式做 TH 判定
        """

        # ==========================================
        # 🤖 階段 9 ~ 14：雙模型並行派發 (目標物 + 機器人)
        # ==========================================
        if stage >= 9:
            target_boxes = []
            robot_boxes = []

            # 1. 取得當前階段的「目標物」模型與框 (平板、前方物件或背景物件)
            target_model_name = self.stage_model_map.get(stage)
            if target_model_name:
                target_model = self._get_model(target_model_name)
                if target_model:
                    results = target_model.predict(
                        source=frame, conf=0.75, imgsz=960, 
                        device=0 if self.use_cuda else "cpu", half=self.use_cuda, verbose=False
                    )
                    if results and len(results) > 0:
                        for box in results[0].boxes.xyxy.cpu().numpy():
                            target_boxes.append((int(box[0]), int(box[1]), int(box[2]), int(box[3])))

            # 2. 獨立取得「機器人」模型與框 (供 TH 判定用)
            robot_model = self._get_model("robot_model.pt") # ✅ 導入新創立的純偵測模型
            if robot_model:
                robot_results = robot_model.predict(
                    source=frame, conf=0.6, imgsz=960, # 機器人信心度設為 0.6，可視情況微調
                    device=0 if self.use_cuda else "cpu", half=self.use_cuda, verbose=False
                )
                if robot_results and len(robot_results) > 0:
                    for box in robot_results[0].boxes.xyxy.cpu().numpy():
                        robot_boxes.append((int(box[0]), int(box[1]), int(box[2]), int(box[3])))
                        
            # 回傳兩個獨立的框陣列
            return target_boxes, robot_boxes

        # ==========================================
        # 🎈 階段 1 ~ 8：單目標物偵測 (原版邏輯完整保留)
        # ==========================================
        else:
            model_name = self.stage_model_map.get(stage)
            if not model_name:
                return [] # 包含已被拔除模型的第 8 階段，會直接安全地回傳空陣列

            model = self._get_model(model_name)
            if not model:
                return []

            # 核心修改：客製化各階段的信心度門檻
            if stage in [1, 2, 6, 7]:
                current_conf = 0.7
            else:
                current_conf = 0.75
            
            results = model.predict(
                source=frame,
                conf=current_conf,                
                imgsz=960,                
                device=0 if self.use_cuda else "cpu",
                half=self.use_cuda,        
                verbose=False              
            )

            boxes = []
            if results and len(results) > 0:
                boxes_data = results[0].boxes.xyxy.cpu().numpy()
                
                frame_h, frame_w = frame.shape[:2]
                frame_area = frame_w * frame_h
                
                for box in boxes_data:
                    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    
                    box_area = (x2 - x1) * (y2 - y1)
                    area_ratio = box_area / frame_area
                    
                    # 面積過濾防線：確保人體被濾掉
                    if stage in [1, 2, 6, 7] and area_ratio > 0.40:
                        continue
                        
                    boxes.append((x1, y1, x2, y2))
            
            return boxes