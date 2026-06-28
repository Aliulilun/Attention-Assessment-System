import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
from collections import deque
from ultralytics import YOLO
from typing import Optional, List, Dict, Tuple, Any

# ==========================================
# 4. 效能優化：將 LegacyHandLms 移至模組層級，避免每幀重複建立類別
# ==========================================
class LegacyHandLms:
    """封裝物件，用於相容舊版 MediaPipe Landmark 格式"""
    def __init__(self, lms: Any):
        self.landmark = lms

class InteractionEngine:
    
    # ==========================================
    # 8. 消除魔數：將骨架關鍵點索引提取為類別常數
    # ==========================================
    WRIST = 0
    IDX_MCP = 5
    IDX_PIP = 6
    IDX_TIP = 8
    MID_MCP = 9
    MID_PIP = 10
    MID_TIP = 12
    RNG_MCP = 13
    RNG_PIP = 14
    RNG_TIP = 16
    PNK_MCP = 17
    PNK_PIP = 18
    PNK_TIP = 20

    def __init__(
        self, 
        pose_model_path: Optional[str] = None, 
        sma_window: int = 5,
        dwell_frames: int = 3,       # 6. 配置化：DWELL_FRAMES 參數化
        divider_ratio: float = 0.35  # 6. 配置化：分割線比例參數化
    ):
        """
        初始化指向判定引擎 (Tasks API 影片追蹤模式 + 絕對身分防禦 + 寧缺勿濫幾何)
        """
        self.DWELL_FRAMES: int = dwell_frames
        self.divider_ratio: float = divider_ratio
        
        print(">>> [InteractionEngine] 載入人類骨架模型 (YOLO-Pose)...")
        if pose_model_path and os.path.exists(pose_model_path):
            self.model_human = YOLO(pose_model_path)
        else:
            self.model_human = YOLO('yolo11n-pose.pt')

        print(">>> [InteractionEngine] 載入 MediaPipe Hand Landmarker (Tasks API VIDEO MODE)...")
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model', 'gaze', 'hand_landmarker.task')
        
        # 10. 防呆機制：明確報錯 FileNotFoundError
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ 找不到 MediaPipe 模型檔案: {model_path}，請確認路徑設定！")
        
        base_options = python.BaseOptions(model_asset_path=model_path)
        
        # 🌟 啟用 VIDEO 模式解鎖連續追蹤能力
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=4,
            running_mode=vision.RunningMode.VIDEO, 
            min_hand_detection_confidence=0.1,
            min_hand_presence_confidence=0.1,
            min_tracking_confidence=0.1
        )
        self.mp_hands = vision.HandLandmarker.create_from_options(options)
        
        # 用於記錄影片時間戳 (Tasks API VIDEO 模式必須)
        self.timestamp_ms: int = 0
        
        self.C_CHILD: Tuple[int, int, int] = (0, 255, 0)
        self.C_TESTER: Tuple[int, int, int] = (0, 165, 255)
        
        self.dwell_counters: Dict[str, int] = {"Child": 0, "Tester": 0}

        self.ray_history: Dict[str, Dict[str, deque]] = {
            "Child": {"origin": deque(maxlen=sma_window), "vector": deque(maxlen=sma_window)},
            "Tester": {"origin": deque(maxlen=sma_window), "vector": deque(maxlen=sma_window)}
        }

    @staticmethod
    def get_angle(v1: np.ndarray, v2: np.ndarray) -> float:
        # 11. 補上型別標注
        unit_v1 = v1 / (np.linalg.norm(v1) + 1e-9) # 9. 改為 1e-9 減少浮點誤差
        unit_v2 = v2 / (np.linalg.norm(v2) + 1e-9)
        dot_product = np.clip(np.dot(unit_v1, unit_v2), -1.0, 1.0)
        return float(np.degrees(np.arccos(dot_product)))

    # ==========================================
    # 1. 光暈效果修復：第二條中心線改為白色
    # ==========================================
    @staticmethod
    def draw_25d_laser(frame: np.ndarray, start_pt: Tuple[int, int], end_pt: Tuple[int, int], color: Tuple[int, int, int]) -> None:
        overlay = frame.copy()
        # 外層光暈 (色彩)
        cv2.line(overlay, start_pt, end_pt, color, 16, lineType=cv2.LINE_AA)
        # 內層核心 (強制白色才能在 addWeighted 透出光芒)
        cv2.line(overlay, start_pt, end_pt, (255, 255, 255), 6, lineType=cv2.LINE_AA)
        
        alpha = 0.5
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        # 繪製發射點裝飾
        cv2.line(frame, start_pt, end_pt, (255, 255, 255), 2, lineType=cv2.LINE_AA)
        cv2.circle(frame, start_pt, 6, (255, 255, 255), -1)
        cv2.circle(frame, start_pt, 9, color, 2)

    # ==========================================
    # 2. 回傳值改進：改為 Optional[float]，符合則回傳角度，不符合回 None
    # ==========================================
    def is_valid_pointing(self, landmarks: LegacyHandLms, frame_w: int, frame_h: int, owner: str) -> Optional[float]:
        """
        🌟 相對指長鑑別法 (寧缺勿濫版)
        """
        def get_pt(idx: int) -> np.ndarray:
            return np.array([landmarks.landmark[idx].x * frame_w, landmarks.landmark[idx].y * frame_h])

        wri = get_pt(self.WRIST)
        idx_mcp = get_pt(self.IDX_MCP)
        idx_pip = get_pt(self.IDX_PIP)
        idx_tip = get_pt(self.IDX_TIP)
        mid_mcp = get_pt(self.MID_MCP)
        mid_pip = get_pt(self.MID_PIP)
        mid_tip = get_pt(self.MID_TIP)
        rng_mcp = get_pt(self.RNG_MCP)
        rng_pip = get_pt(self.RNG_PIP)
        rng_tip = get_pt(self.RNG_TIP)
        pnk_mcp = get_pt(self.PNK_MCP)
        pnk_pip = get_pt(self.PNK_PIP)
        pnk_tip = get_pt(self.PNK_TIP)

        # 1. 確保關鍵節點都在畫面內
        margin = 0.01 
        for node in [self.WRIST, self.IDX_MCP, self.IDX_PIP, self.IDX_TIP, self.MID_PIP, self.MID_TIP, self.RNG_PIP, self.RNG_TIP]:
            lm = landmarks.landmark[node]
            if not (margin <= lm.x <= 1-margin and margin <= lm.y <= 1-margin):
                return None

        dist = lambda p1, p2: float(np.linalg.norm(p1 - p2))

        # 2. 食指絕對伸直判定
        if dist(idx_tip, idx_mcp) <= dist(idx_pip, idx_mcp) + 5: return None
        if dist(idx_tip, wri) < 25: return None

        if owner == "Tester":
            # 🛡️ 施測者特化防禦
            dist_idx = dist(idx_tip, idx_mcp)
            if dist_idx <= dist(mid_tip, mid_mcp) or dist_idx <= dist(rng_tip, rng_mcp) or dist_idx <= dist(pnk_tip, pnk_mcp):
                return None 
            if dist(idx_tip, wri) < dist(mid_tip, wri) + 20: return None 
            if (idx_tip[0] - wri[0]) < 10: return None 
        else:
            # 🛡️ 兒童特化防禦
            if dist(idx_tip, wri) <= dist(mid_tip, wri) + 10: return None
            is_mid_folded = dist(mid_tip, wri) < dist(mid_pip, wri) + 15
            is_rng_folded = dist(rng_tip, wri) < dist(rng_pip, wri) + 15
            if not (is_mid_folded and is_rng_folded): return None
            if self.get_angle(idx_mcp - wri, idx_tip - idx_mcp) > 40: return None
            
        # 若條件皆符合，計算並回傳「食指的直度 (角度)」
        idx_angle = self.get_angle(idx_pip - idx_mcp, idx_tip - idx_pip)
        return idx_angle

    @staticmethod
    def calculate_arm_link_score(mp_wrist: Tuple[int, int], kpts: np.ndarray, confs: np.ndarray, side: str = "right") -> float:
        s_idx, e_idx, w_idx = (6, 8, 10) if side == "right" else (5, 7, 9)
        if confs[w_idx] > 0.4: return float(np.linalg.norm(np.array(mp_wrist) - np.array(kpts[w_idx])))
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
            return float(dist_to_line + (proj_len * 0.1))
        if confs[e_idx] > 0.4: return float(np.linalg.norm(np.array(mp_wrist) - np.array(kpts[e_idx])) + 50)
        if confs[s_idx] > 0.4: return float(np.linalg.norm(np.array(mp_wrist) - np.array(kpts[s_idx])) + 100)
        return float('inf')

    # ==========================================
    # 9. 數值誤差修正：1e-5 改為 1e-9
    # ==========================================
    @staticmethod
    def ray_intersects_box(origin: Tuple[float, float], dir_vec: np.ndarray, box: Tuple[float, float, float, float]) -> bool:
        ox, oy = origin; dx, dy = dir_vec; x1, y1, x2, y2 = box
        dx = dx if dx != 0 else 1e-9; dy = dy if dy != 0 else 1e-9
        tx1, tx2 = (x1 - ox) / dx, (x2 - ox) / dx
        ty1, ty2 = (y1 - oy) / dy, (y2 - oy) / dy
        tmin = max(min(tx1, tx2), min(ty1, ty2)); tmax = min(max(tx1, tx2), max(ty1, ty2))
        return bool(tmax >= max(0, tmin))

    def draw_dashed_rectangle(self, img: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int], color: Tuple[int, int, int], thickness: int = 1) -> None:
        x1, y1 = pt1; x2, y2 = pt2; length = 15
        lines = [((x1, y1), (x1+length, y1)), ((x1, y1), (x1, y1+length)),
                 ((x2, y1), (x2-length, y1)), ((x2, y1), (x2, y1+length)),
                 ((x1, y2), (x1+length, y2)), ((x1, y2), (x1, y2-length)),
                 ((x2, y2), (x2-length, y2)), ((x2, y2), (x2, y2-length))]
        for pt_a, pt_b in lines: cv2.line(img, pt_a, pt_b, color, thickness)

    # ==========================================
    # 7. 支援外部傳入實際時間戳 elapsed_ms
    # ==========================================
    def analyze_interaction(self, frame: np.ndarray, yolo_boxes: List[Tuple[float, float, float, float]], elapsed_ms: Optional[int] = None) -> bool:
        FRAME_H, FRAME_W = frame.shape[:2]
        DIVIDER_X = int(FRAME_W * self.divider_ratio) # 使用設定好的比例
        child_hit_target = False

        cv2.line(frame, (DIVIDER_X, 0), (DIVIDER_X, FRAME_H), (255, 255, 255), 2)
        cv2.putText(frame, "TESTER ZONE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.C_TESTER, 2)
        cv2.putText(frame, "CHILD ZONE", (DIVIDER_X + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.C_CHILD, 2)

        for (yx1, yy1, yx2, yy2) in yolo_boxes:
            cv2.rectangle(frame, (int(yx1), int(yy1)), (int(yx2), int(yy2)), (255, 255, 255), 2)
            self.draw_dashed_rectangle(frame, (int(yx1)-20, int(yy1)-20), (int(yx2)+20, int(yy2)+20), (150, 150, 150), 2)

        # 任務一：YOLO 人體追蹤與實體存在檢查
        yolo_people = []    
        pose_results = self.model_human.track(frame, persist=True, imgsz=640, conf=0.5, verbose=False)
        
        tester_present = False
        child_present = False
        
        if pose_results and len(pose_results)>0 and pose_results[0].keypoints is not None:
            kpts_all = pose_results[0].keypoints.xy.cpu().numpy()
            confs_all = pose_results[0].keypoints.conf.cpu().numpy()
            boxes_all = pose_results[0].boxes.xyxy.cpu().numpy()
            
            for i in range(len(boxes_all)):
                kpts, confs = kpts_all[i], confs_all[i]
                
                if confs[5] > 0.4 and confs[6] > 0.4:
                    true_body_x = (kpts[5][0] + kpts[6][0]) / 2
                elif confs[5] > 0.4:
                    true_body_x = kpts[5][0]
                elif confs[6] > 0.4:
                    true_body_x = kpts[6][0]
                else:
                    true_body_x = (boxes_all[i][0] + boxes_all[i][2]) / 2
                
                owner = "Tester" if true_body_x < DIVIDER_X else "Child"
                yolo_people.append({"owner": owner, "box": boxes_all[i], "kpts": kpts, "confs": confs})
                
                if owner == "Tester": tester_present = True
                if owner == "Child": child_present = True
                
                color = self.C_TESTER if owner == "Tester" else self.C_CHILD
                bx1, by1, bx2, by2 = map(int, boxes_all[i])
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)
                cv2.putText(frame, f"Body: {owner}", (bx1, by1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # ==========================================
        # 5. 提前返回：畫面中完全沒人時，不用跑耗時的手部判定
        # ==========================================
        if not tester_present and not child_present:
            return child_hit_target

        # 任務二：MediaPipe 手勢判定
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # 處理外部傳入真實時間戳
        if elapsed_ms is not None:
            self.timestamp_ms = elapsed_ms
        else:
            self.timestamp_ms += 33 
            
        hand_result = self.mp_hands.detect_for_video(mp_image, self.timestamp_ms)
        
        current_frame_pointing = {"Child": False, "Tester": False}
        candidate_rays = {}
        pointing_owners = set() 

        if hand_result.hand_landmarks:
            for hand_landmarks_list in hand_result.hand_landmarks:
                p_wri = (int(hand_landmarks_list[self.WRIST].x * FRAME_W), int(hand_landmarks_list[self.WRIST].y * FRAME_H)) 
                p_mcp = (int(hand_landmarks_list[self.IDX_MCP].x * FRAME_W), int(hand_landmarks_list[self.IDX_MCP].y * FRAME_H))
                p_idx = (int(hand_landmarks_list[self.IDX_TIP].x * FRAME_W), int(hand_landmarks_list[self.IDX_TIP].y * FRAME_H)) 
                
                best_owner, min_score = None, float('inf') 
                for person in yolo_people:
                    score = min(self.calculate_arm_link_score(p_wri, person["kpts"], person["confs"], "left"),
                                self.calculate_arm_link_score(p_wri, person["kpts"], person["confs"], "right"))
                    if score < min_score:
                        min_score = score
                        best_owner = person["owner"]
                        
                # 絕對防呆機制 (Presence Lock)
                if min_score > 300 or best_owner is None:
                    if p_wri[0] < DIVIDER_X:
                        best_owner = "Tester" if tester_present else "Child"
                    else:
                        best_owner = "Child"
                
                if not tester_present and child_present:
                    best_owner = "Child"
                elif not child_present and tester_present:
                    best_owner = "Tester"

                hand_color = self.C_CHILD if best_owner == "Child" else self.C_TESTER
                cv2.putText(frame, f"[{best_owner}]", (p_wri[0], p_wri[1]+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, hand_color, 2)
                for pt in [4, 8, 12, 16, 20]: 
                    cv2.circle(frame, (int(hand_landmarks_list[pt].x * FRAME_W), int(hand_landmarks_list[pt].y * FRAME_H)), 5, hand_color, -1)
                
                hand_lms = LegacyHandLms(hand_landmarks_list)
                
                # ==========================================
                # 3. 多隻手競爭機制：同一人有兩隻手時，保留「食指角度最直 (角度最小)」的那隻手
                # ==========================================
                idx_angle = self.is_valid_pointing(hand_lms, FRAME_W, FRAME_H, best_owner)
                if idx_angle is None:
                    continue  # 不符合有效指向，略過這隻手
                
                current_frame_pointing[best_owner] = True
                raw_vec = np.array(p_idx) - np.array(p_mcp)
                
                if best_owner in candidate_rays:
                    # 如果該身分已經有登錄的手，比較誰更直 (角度更小)
                    if idx_angle < candidate_rays[best_owner]['angle']:
                        candidate_rays[best_owner] = {'origin': p_mcp, 'vec': raw_vec, 'angle': idx_angle}
                else:
                    candidate_rays[best_owner] = {'origin': p_mcp, 'vec': raw_vec, 'angle': idx_angle}

        # 💡 時間滯留防抖
        for owner in ["Tester", "Child"]:
            if current_frame_pointing[owner]:
                self.dwell_counters[owner] += 1
            else:
                self.dwell_counters[owner] = max(0, self.dwell_counters[owner] - 2) 

            if self.dwell_counters[owner] >= self.DWELL_FRAMES:
                pointing_owners.add(owner)
                self.dwell_counters[owner] = min(self.dwell_counters[owner], self.DWELL_FRAMES + 2)
                if owner in candidate_rays:
                    # 取出勝利的那隻手放入歷史軌跡中
                    self.ray_history[owner]["origin"].append(candidate_rays[owner]['origin'])
                    self.ray_history[owner]["vector"].append(candidate_rays[owner]['vec'])
            else:
                if self.dwell_counters[owner] == 0:
                    self.ray_history[owner]["origin"].clear()
                    self.ray_history[owner]["vector"].clear()

        # 🟢 取出射線判定碰撞
        for owner in ["Tester", "Child"]:
            if owner in pointing_owners and len(self.ray_history[owner]["vector"]) > 0:
                avg_origin = np.mean(self.ray_history[owner]["origin"], axis=0)
                avg_vector = np.mean(self.ray_history[owner]["vector"], axis=0)
                avg_ox, avg_oy = int(avg_origin[0]), int(avg_origin[1])
                hand_color = self.C_CHILD if owner == "Child" else self.C_TESTER

                if np.linalg.norm(avg_vector) > 0:
                    unit_vec = avg_vector / np.linalg.norm(avg_vector)
                    end_point = np.array([avg_ox, avg_oy]) + unit_vec * 1500 
                    
                    self.draw_25d_laser(frame, (avg_ox, avg_oy), (int(end_point[0]), int(end_point[1])), hand_color)

                    for (yx1, yy1, yx2, yy2) in yolo_boxes:
                        exact_box = (yx1, yy1, yx2, yy2)
                        if self.ray_intersects_box((avg_ox, avg_oy), unit_vec, exact_box):
                            if owner == "Tester":
                                cv2.rectangle(frame, (int(yx1), int(yy1)), (int(yx2), int(yy2)), hand_color, 5)
                            elif owner == "Child":
                                cv2.putText(frame, f"Child HIT!", (avg_ox, avg_oy - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, hand_color, 3)
                                cv2.rectangle(frame, (int(yx1), int(yy1)), (int(yx2), int(yy2)), hand_color, 5)
                                child_hit_target = True
                            break 

        return child_hit_target