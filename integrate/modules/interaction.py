import os
import cv2
import numpy as np
import mediapipe as mp
# 🌟 導入合作人建議的新版 Tasks API
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
from collections import deque
from ultralytics import YOLO

class InteractionEngine:
    def __init__(self, pose_model_path=None, sma_window=5):
        """
        初始化指向判定引擎 (模組化設計)
        """
        print(">>> [InteractionEngine] 載入人類骨架模型 (YOLO-Pose) 與互動判定模組...")
        
        if pose_model_path and os.path.exists(pose_model_path):
            self.model_human = YOLO(pose_model_path)
        else:
            self.model_human = YOLO('yolo11n-pose.pt')

        # 🌟 採用合作人建議：升級為 MediaPipe Tasks API 解決閃退問題
        print(">>> [InteractionEngine] 載入 MediaPipe Hand Landmarker (Tasks API)...")
        # 請確認 hand_landmarker.task 放在 model 資料夾下
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model', 'gaze', 'hand_landmarker.task')
        
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=4,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3
        )
        self.mp_hands = vision.HandLandmarker.create_from_options(options)
        
        self.C_CHILD = (0, 255, 0)           # 小孩：綠色
        self.C_TESTER = (0, 165, 255)        # 施測者：橘色
        
        self.sma_window = sma_window
        self.ray_history = {
            "Child": {"origin": deque(maxlen=sma_window), "vector": deque(maxlen=sma_window)},
            "Tester": {"origin": deque(maxlen=sma_window), "vector": deque(maxlen=sma_window)}
        }

    @staticmethod
    def get_angle(v1, v2):
        """計算兩個二維向量的夾角 (度數)"""
        unit_v1 = v1 / (np.linalg.norm(v1) + 1e-6)
        unit_v2 = v2 / (np.linalg.norm(v2) + 1e-6)
        dot_product = np.clip(np.dot(unit_v1, unit_v2), -1.0, 1.0)
        return np.degrees(np.arccos(dot_product))

    def is_valid_pointing(self, landmarks, frame_w, frame_h, owner):
        """
        🌟 絕對防禦拓樸學：導入神經網路幻覺防禦，封殺所有不合理的手勢
        """
        def get_pt(idx):
            return np.array([landmarks.landmark[idx].x * frame_w, landmarks.landmark[idx].y * frame_h])

        wri = get_pt(0)      
        idx_mcp = get_pt(5)  
        idx_pip = get_pt(6)  
        idx_tip = get_pt(8)  
        mid_pip = get_pt(10)
        mid_tip = get_pt(12)
        rng_pip = get_pt(14)
        rng_tip = get_pt(16)
        pnk_tip = get_pt(20)

        # 🛡️ 條件 1：確保關鍵節點都在畫面內
        margin = 0.02 
        for node in [0, 5, 6, 8, 10, 12, 14, 16, 20]:
            lm = landmarks.landmark[node]
            if not (margin <= lm.x <= 1-margin and margin <= lm.y <= 1-margin):
                return False

        # 🛡️ 條件 2：物理長度限制 (太短代表手握拳或面對鏡頭)
        dist_idx_tip_wri = np.linalg.norm(idx_tip - wri)
        if dist_idx_tip_wri < 40: 
            return False
            
        # 基本伸直判斷 (指尖必須大於第二指節)
        is_idx_extended = dist_idx_tip_wri > np.linalg.norm(idx_pip - wri) + 10
        if not is_idx_extended:
            return False

        # 🌟 條件 3：神經網路幻覺防禦 (Anti-Illusion)
        if np.linalg.norm(idx_tip - mid_tip) < 20 or np.linalg.norm(idx_tip - rng_tip) < 20:
            return False # 手指重疊黏在一起，絕對是標錯了

        if owner == "Child":
            # 🛡️ 條件 4：【對小孩極度嚴格】中指與無名指必須確實彎曲收起
            is_mid_folded = np.linalg.norm(mid_tip - wri) < np.linalg.norm(mid_pip - wri) + 15
            is_rng_folded = np.linalg.norm(rng_tip - wri) < np.linalg.norm(rng_pip - wri) + 15
            
            if not (is_mid_folded and is_rng_folded):
                return False
                
            # 🛡️ 條件 5：向量夾角判定：食指必須筆直
            vec_wri_to_mcp = idx_mcp - wri
            vec_mcp_to_tip = idx_tip - idx_mcp
            angle = self.get_angle(vec_wri_to_mcp, vec_mcp_to_tip)
            
            if angle > 35:
                return False
        else:
            # 大人寬鬆：只要食指有往前伸，手掌攤開也無妨
            pass
            
        return True

    @staticmethod
    def ray_intersects_box(origin, dir_vec, box):
        """精準碰撞演算法"""
        ox, oy = origin
        dx, dy = dir_vec
        x1, y1, x2, y2 = box
        
        dx = dx if dx != 0 else 1e-5
        dy = dy if dy != 0 else 1e-5
        
        tx1 = (x1 - ox) / dx
        tx2 = (x2 - ox) / dx
        ty1 = (y1 - oy) / dy
        ty2 = (y2 - oy) / dy
        
        tmin = max(min(tx1, tx2), min(ty1, ty2))
        tmax = min(max(tx1, tx2), max(ty1, ty2))
        
        return tmax >= max(0, tmin)

    @staticmethod
    def trace_hand_to_body(mp_wrist, kpts, confs):
        """利用 YOLO 骨架「往回看身體」：手腕 -> 手肘 -> 肩膀"""
        arms = {"left": [5, 7, 9], "right": [6, 8, 10]}
        best_score = float('inf')
        
        for side, indices in arms.items():
            s_idx, e_idx, w_idx = indices
            
            if confs[w_idx] > 0.4:
                dist = np.linalg.norm(np.array(mp_wrist) - np.array(kpts[w_idx]))
                best_score = min(best_score, dist)
                continue
                
            if confs[e_idx] > 0.4 and confs[s_idx] > 0.4:
                S = np.array(kpts[s_idx])
                E = np.array(kpts[e_idx])
                W_mp = np.array(mp_wrist)
                
                arm_vec = E - S 
                arm_len = np.linalg.norm(arm_vec)
                if arm_len > 0:
                    unit_arm = arm_vec / arm_len
                    elbow_to_mp = W_mp - E
                    
                    proj_len = np.dot(elbow_to_mp, unit_arm) 
                    
                    if proj_len > -20: 
                        dist_to_line = np.abs(np.cross(unit_arm, elbow_to_mp))
                        score = dist_to_line + (np.abs(proj_len) * 0.3)
                        best_score = min(best_score, score + 30) 
                        continue

            if confs[e_idx] > 0.4:
                dist = np.linalg.norm(np.array(mp_wrist) - np.array(kpts[e_idx]))
                best_score = min(best_score, dist + 80)
                
            if confs[s_idx] > 0.4:
                dist = np.linalg.norm(np.array(mp_wrist) - np.array(kpts[s_idx]))
                best_score = min(best_score, dist + 150)

        return best_score

    def draw_dashed_rectangle(self, img, pt1, pt2, color, thickness=1):
        x1, y1 = pt1
        x2, y2 = pt2
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
        FRAME_H, FRAME_W = frame.shape[:2]
        DIVIDER_X = int(FRAME_W * 0.35) 
        child_hit_target = False

        # --- 畫出場地界線 ---
        cv2.line(frame, (DIVIDER_X, 0), (DIVIDER_X, FRAME_H), (255, 255, 255), 2)
        cv2.putText(frame, "TESTER ZONE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.C_TESTER, 2)
        cv2.putText(frame, "CHILD ZONE", (DIVIDER_X + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.C_CHILD, 2)

        for (yx1, yy1, yx2, yy2) in yolo_boxes:
            cv2.rectangle(frame, (int(yx1), int(yy1)), (int(yx2), int(yy2)), (255, 255, 255), 2)
            self.draw_dashed_rectangle(frame, (int(yx1)-20, int(yy1)-20), (int(yx2)+20, int(yy2)+20), (150, 150, 150), 2)

        # ====================================================
        # 🟢 任務一：YOLO 人體追蹤與「骨架記錄」
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
                    body_cx = (kpts[5][0] + kpts[6][0]) / 2
                else:
                    body_cx = (boxes_all[i][0] + boxes_all[i][2]) / 2
                
                owner = "Tester" if body_cx < DIVIDER_X else "Child"
                yolo_people.append({"owner": owner, "box": boxes_all[i], "kpts": kpts, "confs": confs})
                
                color = self.C_TESTER if owner == "Tester" else self.C_CHILD
                bx1, by1, bx2, by2 = map(int, boxes_all[i])
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)
                cv2.putText(frame, f"Body: {owner}", (bx1, by1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # ====================================================
        # 🟢 任務二：人類手勢判定 (採用 Tasks API 處理)
        # ====================================================
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        hand_result = self.mp_hands.detect(mp_image)
        
        pointing_owners = set() 
        
        # 🌟 相容性包裝器：讓新 API 的資料結構偽裝成舊版的樣子，避免改動你的防禦演算法
        class LegacyHandLms:
            def __init__(self, lms):
                self.landmark = lms

        invalid_hand_indices = set()
        if hand_result.hand_landmarks:
            mp_hands_info = []
            for idx, hand_landmarks_list in enumerate(hand_result.hand_landmarks):
                w_x = int(hand_landmarks_list[0].x * FRAME_W)
                w_y = int(hand_landmarks_list[0].y * FRAME_H)
                mp_hands_info.append({"idx": idx, "pos": np.array([w_x, w_y])})
                
            for i in range(len(mp_hands_info)):
                for j in range(i + 1, len(mp_hands_info)):
                    dist = np.linalg.norm(mp_hands_info[i]["pos"] - mp_hands_info[j]["pos"])
                    if dist < FRAME_W * 0.12: 
                        invalid_hand_indices.add(i)
                        invalid_hand_indices.add(j)

        if hand_result.hand_landmarks:
            for idx, hand_landmarks_list in enumerate(hand_result.hand_landmarks):
                # 將新資料結構包裝起來
                hand_lms = LegacyHandLms(hand_landmarks_list)
                
                p_wri = (int(hand_lms.landmark[0].x * FRAME_W), int(hand_lms.landmark[0].y * FRAME_H)) 
                p_idx = (int(hand_lms.landmark[8].x * FRAME_W), int(hand_lms.landmark[8].y * FRAME_H)) 
                
                best_owner = None
                min_score = float('inf')
                
                for person in yolo_people:
                    score = self.trace_hand_to_body(p_wri, person["kpts"], person["confs"])
                    if score < min_score:
                        min_score = score
                        best_owner = person["owner"]
                        
                if min_score > 300 or best_owner is None:
                    best_owner = "Tester" if p_wri[0] < DIVIDER_X else "Child"

                hand_color = self.C_CHILD if best_owner == "Child" else self.C_TESTER
                
                cv2.circle(frame, p_wri, 8, hand_color, -1)
                cv2.circle(frame, p_wri, 10, (255, 255, 255), 2)
                for pt in [4, 8, 12, 16, 20]: 
                    cv2.circle(frame, (int(hand_lms.landmark[pt].x * FRAME_W), int(hand_lms.landmark[pt].y * FRAME_H)), 5, hand_color, -1)
                
                if idx in invalid_hand_indices:
                    continue
                    
                if not self.is_valid_pointing(hand_lms, FRAME_W, FRAME_H, best_owner):
                    continue
                
                pointing_owners.add(best_owner)
                raw_vec = np.array(p_idx) - np.array(p_wri)
                
                self.ray_history[best_owner]["origin"].append(p_wri)
                self.ray_history[best_owner]["vector"].append(raw_vec)

        # ====================================================
        # 🟢 任務三：取出平滑化後的射線，並進行碰撞判定
        # ====================================================
        for owner in ["Tester", "Child"]:
            if owner in pointing_owners and len(self.ray_history[owner]["vector"]) > 0:
                
                avg_origin = np.mean(self.ray_history[owner]["origin"], axis=0)
                avg_vector = np.mean(self.ray_history[owner]["vector"], axis=0)
                
                avg_ox, avg_oy = int(avg_origin[0]), int(avg_origin[1])
                hand_color = self.C_CHILD if owner == "Child" else self.C_TESTER

                if np.linalg.norm(avg_vector) > 0:
                    unit_vec = avg_vector / np.linalg.norm(avg_vector)
                    end_point = np.array([avg_ox, avg_oy]) + unit_vec * 1500 
                    
                    cv2.line(frame, (avg_ox, avg_oy), (int(end_point[0]), int(end_point[1])), hand_color, 3)

                    for (yx1, yy1, yx2, yy2) in yolo_boxes:
                        exact_box = (yx1, yy1, yx2, yy2)
                        
                        if self.ray_intersects_box((avg_ox, avg_oy), unit_vec, exact_box):
                            if owner == "Tester":
                                cv2.rectangle(frame, (int(yx1), int(yy1)), (int(yx2), int(yy2)), hand_color, 5)
                            elif owner == "Child":
                                cv2.putText(frame, f"Child HIT!", (avg_ox, avg_oy - 40), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, hand_color, 3)
                                cv2.rectangle(frame, (int(yx1), int(yy1)), (int(yx2), int(yy2)), hand_color, 5)
                                child_hit_target = True
                            break 
            else:
                self.ray_history[owner]["origin"].clear()
                self.ray_history[owner]["vector"].clear()

        return child_hit_target