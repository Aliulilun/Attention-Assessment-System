# 檔案路徑：utils/state_manager.py
import numpy as np

class GazeFSMManager:
    def __init__(self, history_len=5, max_hold_frames=30):
        """
        初始化時序有限狀態機
        Args:
            history_len: 用於判定運動趨勢的歷史幀緩衝長度
            max_hold_frames: 特徵點丟失後，極端姿勢最長維持的訊框數（盲追蹤安全閥）
        """
        self.history_len = history_len
        self.max_hold_frames = max_hold_frames
        
        self.history_pose = []      # 儲存歷史歐拉角姿態 [(pitch, yaw), ...]
        self.hold_counter = 0       # 盲追蹤計數器
        self.current_state = "FRONTAL"
        self.gaze_target = "NONE"
        
        # 高級幾何約束：記錄特徵點消失前最後一個合法的頭部 2D 質心
        self.last_valid_centroid = None

    def update(self, face_result, pose_result=None) -> str:
        """
        更新有限狀態機狀態，並回傳當前視線意圖
        Returns:
            "LEFT_BACK_UPPER", "RIGHT_BACK_UPPER", "NONE"
        """
        # 計算當前訊框的 YOLO 頭部質心 (Centroid)
        current_centroid = None
        if face_result is not None and face_result.get('yolo_head_bbox') is not None:
            x1, y1, x2, y2 = face_result['yolo_head_bbox']
            current_centroid = ((x1 + x2) / 2, (y1 + y2) / 2)

        # ─── 情況 1：常規追蹤期（MediaPipe 正常迴歸特徵點且 PnP 解算成功） ───
        if face_result is not None and face_result['num_landmarks'] > 0 and pose_result and pose_result['success']:
            self.current_state = "FRONTAL"
            self.hold_counter = 0
            
            pitch = pose_result['euler_angles']['pitch']
            yaw = pose_result['euler_angles']['yaw']
            
            # 更新時序緩衝隊列
            self.history_pose.append((pitch, yaw))
            if len(self.history_pose) > self.history_len:
                self.history_pose.pop(0)
                
            if current_centroid:
                self.last_valid_centroid = current_centroid
                
            # 常規幾何空間量測
            if pitch < 10:  # 向上看（在你的系統坐標系中，頭部向上為負值）
                if yaw > 35: 
                    return "RIGHT_BACK_UPPER"
                elif yaw < -35: 
                    return "LEFT_BACK_UPPER"
            return "NONE"

        # ─── 情況 2：盲追蹤期（人臉遮蔽，特徵點消失，PnP 無法求解） ───
        else:
            # 【節點 A：狀態瞬時轉移】從 FRONTAL 驟然丟失的第一幀
            if self.current_state == "FRONTAL" and len(self.history_pose) > 0:
                last_pitch, last_yaw = self.history_pose[-1]
                
                # 依據盲區前一影格的運動向量外推意圖
                # 臨界條件：Yaw 偏航超過 35° 且 Pitch 呈現上揚趨勢
                if last_yaw > 35 and last_pitch < 10:
                    self.current_state = "EXTREME_TURNING"
                    self.gaze_target = "RIGHT_BACK_UPPER"
                    self.hold_counter = 1
                    print(f"⚠️ 臨界狀態轉移：測不到人臉，但消失前姿態為 Yaw:{last_yaw:.1f}°, Pitch:{last_pitch:.1f}° -> 進入右後上方盲追蹤")
                
                elif last_yaw < -35 and last_pitch < 10:
                    self.current_state = "EXTREME_TURNING"
                    self.gaze_target = "LEFT_BACK_UPPER"
                    self.hold_counter = 1
                    print(f"⚠️ 臨界狀態轉移：測不到人臉，但消失前姿態為 Yaw:{last_yaw:.1f}°, Pitch:{last_pitch:.1f}° -> 進入左後上方盲追蹤")
                else:
                    self.current_state = "LOST"
                    self.gaze_target = "NONE"

            # 【節點 B：盲追蹤狀態維持】完全不計算幾何歐拉角，僅依賴 YOLO 存在性證明
            elif self.current_state == "EXTREME_TURNING":
                has_yolo_box = (current_centroid is not None)
                
                # 幾何安全性約束 (防止小孩只是單純低頭或轉向下方)
                # 影像坐標系下，Y 軸正向朝下。如果當前頭部質心 Y 座標比消失前顯著增大（如下沉超過 120 像素），說明其視線偏離後上方圖片
                centroid_valid = True
                if has_yolo_box and self.last_valid_centroid:
                    if current_centroid[1] - self.last_valid_centroid[1] > 120:
                        centroid_valid = False
                        print("❌ 狀態終止：檢測到頭部質心大幅下沉，取消後上方 GAZING 判定")
                
                if has_yolo_box and centroid_valid and self.hold_counter < self.max_hold_frames:
                    self.hold_counter += 1
                    if current_centroid:
                        self.last_valid_centroid = current_centroid
                    return self.gaze_target  # 核心：直接返回鎖定的歷史意圖
                else:
                    # 超時、YOLO 目標完全丟失或幾何約束破裂
                    self.current_state = "LOST"
                    self.gaze_target = "NONE"
                    self.history_pose.clear()
            
            return self.gaze_target