import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
from ultralytics import YOLO
import urllib.request

class InteractionEngine:
    def __init__(self, pose_model_path=None, sma_window=5):
        """
        初始化指向判定引擎 (模組化設計)
        """
        print(">>> [InteractionEngine] 載入人類骨架模型 (YOLO-Pose) 與互動判定模組...")
        
        # 1. 載入 YOLO Pose 模型
        if pose_model_path and os.path.exists(pose_model_path):
            self.model_human = YOLO(pose_model_path)
        else:
            self.model_human = YOLO('yolo11n-pose.pt') # 容錯預設路徑

        # 2. 初始化 MediaPipe Hands (新版 Tasks API)
        hand_model_path = self._get_hand_model_path()
        base_options = python.BaseOptions(model_asset_path=hand_model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=4,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.hand_landmarker = vision.HandLandmarker.create_from_options(options)
        
        # 3. 色票定義
        self.C_CHILD = (0, 255, 0)           # 小孩：綠色
        self.C_TESTER = (0, 165, 255)        # 施測者：橘色
        self.C_WRIST_FALLBACK = (200, 0, 200) # YOLO 手腕代償：粉紫色
        
        self.sma_window = sma_window
    
    def _get_hand_model_path(self):
        """下載並返回 hand_landmarker.task 模型路徑"""
        model_dir = os.path.join(os.path.dirname(__file__), '..', 'model', 'gaze')
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, 'hand_landmarker.task')
        
        if not os.path.exists(model_path):
            print(f">>> [InteractionEngine] 下載 hand_landmarker.task 模型...")
            url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
            try:
                urllib.request.urlretrieve(url, model_path)
                print(f">>> [InteractionEngine] 模型下載完成: {model_path}")
            except Exception as e:
                print(f">>> [InteractionEngine] ⚠️ 模型下載失敗: {e}")
                raise
        
        return model_path

    @staticmethod
    def is_valid_pointing(landmarks, frame_w, frame_h):
        """
        嚴格過濾：判斷手部是否「真的」伸出食指指向，沒抓穩或沒伸直絕對不連線
        注意：新版 MediaPipe Tasks API，landmarks 是 list，不是 LandmarkList 對象
        """
        wri = landmarks[0]  # 手腕
        tip = landmarks[8]  # 食指尖
        mcp = landmarks[5]  # 食指根部
        
        # 1. 確保關鍵點都在畫面合理範圍內 (避免邊緣誤判)
        if not (0 <= wri.x <= 1 and 0 <= wri.y <= 1 and 0 <= tip.x <= 1 and 0 <= tip.y <= 1):
            return False
            
        wri_pt = np.array([wri.x * frame_w, wri.y * frame_h])
        tip_pt = np.array([tip.x * frame_w, tip.y * frame_h])
        mcp_pt = np.array([mcp.x * frame_w, mcp.y * frame_h])
        
        # 2. 幾何判定：食指必須是「伸直」的狀態
        # 條件：指尖到手腕的距離 > 指根到手腕的距離
        dist_tip_wri = np.linalg.norm(tip_pt - wri_pt)
        dist_mcp_wri = np.linalg.norm(mcp_pt - wri_pt)
        
        # 3. 確保手指不是微微彎曲的狀態
        dist_tip_mcp = np.linalg.norm(tip_pt - mcp_pt)
        
        if dist_tip_wri > dist_mcp_wri and dist_tip_mcp > 30:
            return True
        return False

    @staticmethod
    def ray_intersects_box(origin, dir_vec, box):
        """
        精準碰撞：Ray-AABB 射線與矩形邊界交集演算法
        只有當射線「實體穿過」指定的 box 時才會回傳 True
        """
        ox, oy = origin
        dx, dy = dir_vec
        x1, y1, x2, y2 = box
        
        # 避免除以零的錯誤
        dx = dx if dx != 0 else 1e-5
        dy = dy if dy != 0 else 1e-5
        
        tx1 = (x1 - ox) / dx
        tx2 = (x2 - ox) / dx
        ty1 = (y1 - oy) / dy
        ty2 = (y2 - oy) / dy
        
        tmin = max(min(tx1, tx2), min(ty1, ty2))
        tmax = min(max(tx1, tx2), max(ty1, ty2))
        
        # tmax >= 0 確保是往前指，且 tmax >= tmin 代表有交集
        return tmax >= max(0, tmin)

    @staticmethod
    def calculate_arm_link_score(mp_wrist, kpts, confs, side="right"):
        """
        輔助演算法：計算 MediaPipe 手腕與 YOLO 骨架關節的匹配分數 (防盜用機制)
        """
        s_idx, e_idx, w_idx = (6, 8, 10) if side == "right" else (5, 7, 9)
        
        if confs[w_idx] > 0.4: return np.linalg.norm(np.array(mp_wrist) - np.array(kpts[w_idx]))
        if confs[e_idx] > 0.4 and confs[s_idx] > 0.4:
            S, E, W_mp = np.array(kpts[s_idx]), np.array(kpts[e_idx]), np.array(mp_wrist)
            arm_vec = E - S 
            arm_len = np.linalg.norm(arm_vec)
            if arm_len == 0: return float('inf')
            
            unit_arm = arm_vec / arm_len
            elbow_to_mp = W_mp - E
            proj_len = np.dot(elbow_to_mp, unit_arm) 
            if proj_len < -30: return float('inf') 
            
            dist_to_line = np.abs(np.cross(unit_arm, elbow_to_mp))
            return dist_to_line + (proj_len * 0.1)
            
        if confs[e_idx] > 0.4: return np.linalg.norm(np.array(mp_wrist) - np.array(kpts[e_idx])) + 50
        if confs[s_idx] > 0.4: return np.linalg.norm(np.array(mp_wrist) - np.array(kpts[s_idx])) + 100
        return float('inf')

    def draw_dashed_rectangle(self, img, pt1, pt2, color, thickness=1, style='dotted'):
        """自定義繪製虛線防觸擊框"""
        x1, y1 = pt1
        x2, y2 = pt2
        # 簡單畫出四個角來表示防觸擊範圍
        length = 15
        cv2.line(img, (x1, y1), (x1+length, y1), color, thickness)
        cv2.line(img, (x1, y1), (x1, y1+length), color, thickness)
        cv2.line(img, (x2, y1), (x2-length, y1), color, thickness)
        cv2.line(img, (x2, y1), (x2, y1+length), color, thickness)
        cv2.line(img, (x1, y2), (x1+length, y2), color, thickness)
        cv2.line(img, (x1, y2), (x1, y2-length), color, thickness)
        cv2.line(img, (x2, y2), (x2-length, y2), color, thickness)
        cv2.line(img, (x2, y2), (x2, y2-length), color, thickness)

    def analyze_interaction(self, frame, yolo_boxes):
        """
        核心多模態互動判定
        """
        FRAME_H, FRAME_W = frame.shape[:2]
        DIVIDER_X = int(FRAME_W * 0.35) 
        MAX_ARM_LENGTH = FRAME_W * 0.25 
        child_hit_target = False

        # --- 畫出場地界線 ---
        cv2.line(frame, (DIVIDER_X, 0), (DIVIDER_X, FRAME_H), (255, 255, 255), 2)
        cv2.putText(frame, "TESTER ZONE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.C_TESTER, 2)
        cv2.putText(frame, "CHILD ZONE", (DIVIDER_X + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.C_CHILD, 2)

        # 🌟 任務 0：明確畫出「原本物品框」與「外圍防觸擊框」
        for (yx1, yy1, yx2, yy2) in yolo_boxes:
            # 1. 畫出原本物品的框 (實線白色)
            cv2.rectangle(frame, (int(yx1), int(yy1)), (int(yx2), int(yy2)), (255, 255, 255), 2)
            # 2. 畫出外圍防觸擊區 (向外擴 20px，灰色虛線角)
            self.draw_dashed_rectangle(frame, (int(yx1)-20, int(yy1)-20), (int(yx2)+20, int(yy2)+20), (150, 150, 150), 2)

        # ====================================================
        # 🟢 任務一：YOLO 人體追蹤與「絕對身分判定」
        # ====================================================
        yolo_people = []    
        pose_results = self.model_human.track(frame, persist=True, imgsz=640, conf=0.5, verbose=False)
        
        if pose_results and len(pose_results) > 0 and pose_results[0].keypoints is not None:
            kpts_all = pose_results[0].keypoints.xy.cpu().numpy()
            conf_all = pose_results[0].keypoints.conf.cpu().numpy()
            boxes_all = pose_results[0].boxes.xyxy.cpu().numpy()
            
            for i in range(len(boxes_all)):
                kpts, confs = kpts_all[i], conf_all[i]
                
                if confs[5] > 0.4 and confs[6] > 0.4:
                    true_body_x = (kpts[5][0] + kpts[6][0]) / 2
                    shoulder_y = (kpts[5][1] + kpts[6][1]) / 2
                elif confs[5] > 0.4:
                    true_body_x, shoulder_y = kpts[5][0], kpts[5][1]
                elif confs[6] > 0.4:
                    true_body_x, shoulder_y = kpts[6][0], kpts[6][1]
                else:
                    true_body_x = (boxes_all[i][0] + boxes_all[i][2]) / 2
                    shoulder_y = (boxes_all[i][1] + boxes_all[i][3]) / 2
                
                owner = "Tester" if true_body_x < DIVIDER_X else "Child"
                yolo_people.append({"owner": owner, "box": boxes_all[i], "kpts": kpts, "confs": confs})
                
                color = self.C_TESTER if owner == "Tester" else self.C_CHILD
                bx1, by1, bx2, by2 = map(int, boxes_all[i])
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)
                cv2.putText(frame, f"Body: {owner}", (bx1, by1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                if owner == "Child":
                    for wrist_idx in [9, 10]:
                        if confs[wrist_idx] > 0.4: 
                            dist = np.linalg.norm(np.array([kpts[wrist_idx][0], kpts[wrist_idx][1]]) - np.array([true_body_x, shoulder_y]))
                            if dist < MAX_ARM_LENGTH:
                                cv2.circle(frame, (int(kpts[wrist_idx][0]), int(kpts[wrist_idx][1])), 10, self.C_WRIST_FALLBACK, -1)

        # ====================================================
        # 🟢 任務二：人類手勢判定與精準指向
        # ====================================================
        # 使用新版 MediaPipe Tasks API
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        mp_results = self.hand_landmarker.detect(mp_image)
        
        if mp_results.hand_landmarks:
            for hand_lms in mp_results.hand_landmarks:
                
                # 🌟 嚴格過濾：如果沒有確實伸出食指，就跳過不畫射線
                if not self.is_valid_pointing(hand_lms, FRAME_W, FRAME_H):
                    continue
                    
                p_wri = (int(hand_lms[0].x * FRAME_W), int(hand_lms[0].y * FRAME_H)) 
                p_idx = (int(hand_lms[8].x * FRAME_W), int(hand_lms[8].y * FRAME_H)) 
                
                best_owner, best_score = None, float('inf') 
                for person in yolo_people:
                    score = min(self.calculate_arm_link_score(p_wri, person["kpts"], person["confs"], "left"),
                                self.calculate_arm_link_score(p_wri, person["kpts"], person["confs"], "right"))
                    if score < best_score:
                        best_score, best_owner = score, person["owner"]
                        
                vec_x = hand_lms[0].x - hand_lms[9].x 
                
                if best_score > 300 or best_owner is None:
                    best_owner = "Tester" if p_wri[0] < DIVIDER_X else "Child"

                if best_owner == "Child" and vec_x < -0.02: best_owner = "Tester" 
                elif best_owner == "Tester" and vec_x > 0.02: best_owner = "Child"  

                hand_color = self.C_CHILD if best_owner == "Child" else self.C_TESTER
                cv2.putText(frame, f"[{best_owner}]", (p_wri[0], p_wri[1]+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, hand_color, 2)
                
                # 畫出指關節
                for pt in [4, 8, 12, 16, 20]: 
                    cv2.circle(frame, (int(hand_lms[pt].x * FRAME_W), int(hand_lms[pt].y * FRAME_H)), 5, hand_color, -1)
                
                # 計算發射向量
                vec = np.array(p_idx) - np.array(p_wri)
                if np.linalg.norm(vec) > 0:
                    unit_vec = vec / np.linalg.norm(vec)
                    end_point = np.array(p_idx) + unit_vec * 1000 
                    
                    # 畫出雷射
                    cv2.line(frame, p_wri, (int(end_point[0]), int(end_point[1])), hand_color, 3)

                    # 🌟 精準碰撞判定：只有射線切過原本的內層框才算
                    for (yx1, yy1, yx2, yy2) in yolo_boxes:
                        # 傳入原本物品的最準確的邊界框
                        exact_box = (yx1, yy1, yx2, yy2)
                        
                        if self.ray_intersects_box(p_wri, unit_vec, exact_box):
                            # HIT！在上方顯示
                            cv2.putText(frame, f"{best_owner} HIT!", (p_wri[0], p_wri[1] - 40), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, hand_color, 3)
                            # 將命中的「原本物品框」加粗亮起
                            cv2.rectangle(frame, (int(yx1), int(yy1)), (int(yx2), int(yy2)), hand_color, 5)
                            
                            if best_owner == "Child":
                                child_hit_target = True
                            break

        return child_hit_target