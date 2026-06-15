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
            8: "robot_point_model.pt"  
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
        執行 YOLO 偵測，並回傳物件框的座標陣列
        """
        model_name = self.stage_model_map.get(stage)
        if not model_name:
            return []

        model = self._get_model(model_name)
        if not model:
            return []

        # ==========================================
        # 🤖 第八階段：機器人專屬指向判定與繪圖
        # ==========================================
        if stage == 8:
            results = model.predict(
                source=frame, 
                conf=0.6, 
                iou=0.5, 
                imgsz=960, 
                device=0 if self.use_cuda else "cpu",
                half=self.use_cuda,
                verbose=False
            )

            if results and len(results) > 0:
                r = results[0]
                if r.keypoints is not None and r.boxes is not None:
                    boxes = r.boxes.xyxy.cpu().numpy()
                    confs = r.boxes.conf.cpu().numpy()
                    kpts = r.keypoints.xy.cpu().numpy()
                    
                    for i in range(len(boxes)):
                        bx1, by1, bx2, by2 = boxes[i]
                        box_conf = confs[i]
                        
                        cx = int((bx1 + bx2) / 2)
                        cy = int((by1 + by2) / 2)
                        
                        cv2.rectangle(frame, (int(bx1), int(by1)), (int(bx2), int(by2)), (255, 100, 100), 2)
                        cv2.putText(frame, f"Robot Conf: {box_conf:.2f}", (int(bx1), int(by1)-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 2)

                        if i < len(kpts) and len(kpts[i]) > 0:
                            tx, ty = int(kpts[i][0][0]), int(kpts[i][0][1])
                            if tx > 0 and ty > 0:
                                cv2.circle(frame, (cx, cy), 6, (0, 255, 255), -1) 
                                cv2.circle(frame, (tx, ty), 6, (0, 0, 255), -1)   
                                
                                vec = np.array([tx, ty]) - np.array([cx, cy])
                                if np.linalg.norm(vec) > 0:
                                    unit_vec = vec / np.linalg.norm(vec)
                                    end_point = np.array([tx, ty]) + unit_vec * 1500 
                                    
                                    cv2.line(frame, (cx, cy), (int(end_point[0]), int(end_point[1])), (0, 255, 255), 4)
            
            return []

        # ==========================================
        # 🎈 其他階段 (1~7)：目標物偵測
        # ==========================================
        else:
            # 🌟 核心修改：客製化各階段的信心度門檻
            if stage in [1, 2]:
                # 狗模型特徵較不明顯，我們將門檻下殺到 0.30 把它逼出來！
                current_conf = 0.7
            elif stage in [6, 7]:
                current_conf = 0.7
            else:
                # 其他掛在牆上或明顯的物件維持嚴格標準
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
                    
                    # 🌟 面積過濾防線：
                    # 既然我們把第 1,2 階段的信心度降得這麼低，就必須確保「人體」被濾掉。
                    # 設定 0.40 (佔畫面 40%) 為極限，超過這個大小的一律當作人類或背景誤判。
                    if stage in [1, 2, 6, 7] and area_ratio > 0.40:
                        continue
                        
                    boxes.append((x1, y1, x2, y2))
            
            return boxes