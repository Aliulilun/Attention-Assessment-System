import cv2
import numpy as np
import easyocr
from collections import deque, Counter

class SignboardTracker:
    def __init__(self, allowlist='12345678'):
        print(">>> [SignboardTracker] 正在載入 EasyOCR 引擎...")
        self.reader = easyocr.Reader(['en'])
        self.allowlist = allowlist
        
        self.current_stage = 0
        self.last_valid_box = None
        self.lost_patience = 0
        
        # 🌟 修改 1：擴大動態追蹤結界 (40 -> 80)。
        # 換牌子到後期 (7階段) 時手容易晃，給予更大的緩衝區防止跟丟
        self.TRACKING_PAD = 80 
        self.PATIENCE_THRESHOLD = 15 
        
        self.history_results = deque(maxlen=7)
        self.UPGRADE_THRESHOLD = 2             
        
        self.roi_x, self.roi_y, self.roi_w, self.roi_h = 0, 0, 0, 0
        self.current_crop_coords = (0, 0, 0, 0)

    def initialize_roi(self, first_frame):
        print(">>> [SignboardTracker] 請框選牌子可能出現的「大範圍區域」。")
        cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Select ROI", 1280, 720) 
        roi = cv2.selectROI("Select ROI", first_frame, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow("Select ROI")

        self.roi_x, self.roi_y, self.roi_w, self.roi_h = roi
        if self.roi_w == 0 or self.roi_h == 0:
            h, w = first_frame.shape[:2]
            self.roi_x, self.roi_y, self.roi_w, self.roi_h = 0, 0, w, h

    def detect_stage(self, frame):
        # ==========================================
        # 🔒 死鎖跟蹤邏輯：不亂放大，丟失太久才退回大桌面
        # ==========================================
        if self.last_valid_box and self.lost_patience <= self.PATIENCE_THRESHOLD:
            bx, by, bw, bh = self.last_valid_box
            crop_x1 = max(self.roi_x, bx - self.TRACKING_PAD)
            crop_y1 = max(self.roi_y, by - self.TRACKING_PAD)
            crop_x2 = min(self.roi_x + self.roi_w, bx + bw + self.TRACKING_PAD)
            crop_y2 = min(self.roi_y + self.roi_h, by + bh + self.TRACKING_PAD)
        else:
            crop_x1, crop_y1 = self.roi_x, self.roi_y
            crop_x2, crop_y2 = self.roi_x + self.roi_w, self.roi_y + self.roi_h

        self.current_crop_coords = (crop_x1, crop_y1, crop_x2, crop_y2)
        crop_img = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        
        # 安全防護：避免視窗縮小到 0 導致崩潰
        if crop_img.shape[0] == 0 or crop_img.shape[1] == 0:
            return None

        gray_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        
        found_number = False
        detected_stage = None
        best_bbox = None
        best_stage_num = -1

        # ==========================================
        # 🛡️ 統一安檢站：防呆與亮度驗證 (針對 7 號特製版)
        # ==========================================
        def passes_strict_filter(stage_val, prob_val, ox, oy, ow, oh, min_prob):
            # 🌟 修改 2：【7 號專屬 VVIP 通道】
            # 因為 7 的筆畫最少，信心度容易偏低。如果是 7，門檻直接打 6 折！
            actual_min_prob = min_prob * 0.6 if stage_val == 7 else min_prob
            if prob_val < actual_min_prob: return False
            
            if oh < 20 or oh > 250: return False         
            if ow > oh * 1.1: return False               
            if ow < oh * 0.10: return False              
            if stage_val == 1 and ow > oh * 0.6: return False 
            
            roi_gray = gray_img[max(0, oy):oy+oh, max(0, ox):ox+ow]
            if roi_gray.size > 0:
                # 🌟 修改 3：取消對 7 的亮度封殺
                # 7 的白色背景佔比太高，如果有一點陰影就會被 np.mean < 85 誤殺
                if stage_val != 7 and np.mean(roi_gray) < 85: 
                    return False 
            return True

        # ==========================================
        # 🚀 裝甲四軌道辨識系統 (Quad-Track Recognition)
        # ==========================================
        tracks = []
        pad_size = 20
        
        # 軌道 1：原始灰階
        img_1 = cv2.copyMakeBorder(gray_img, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT, value=255)
        tracks.append(("Normal", img_1, 1.0, 0.40))
        
        # 軌道 2：放大與對比度強化
        scale = 2.0
        resized_img = cv2.resize(gray_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl_img = clahe.apply(resized_img)
        pad_2 = int(pad_size * scale)
        img_2 = cv2.copyMakeBorder(cl_img, pad_2, pad_2, pad_2, pad_2, cv2.BORDER_CONSTANT, value=255)
        tracks.append(("CLAHE", img_2, scale, 0.45))
        
        # 🌟 修改 4：局部自適應二值化 (Adaptive Threshold) -> 專剋陰影與反光
        # 取代容易被陰影干擾的 Otsu，無論牌子處於什麼光線都能看清
        blur_img = cv2.GaussianBlur(cl_img, (5, 5), 0)
        adaptive_thresh = cv2.adaptiveThreshold(blur_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5)
        img_3 = cv2.copyMakeBorder(adaptive_thresh, pad_2, pad_2, pad_2, pad_2, cv2.BORDER_CONSTANT, value=255)
        tracks.append(("Adaptive", img_3, scale, 0.45))

        # 🌟 修改 5：字體侵蝕加粗 (Bold Erosion) -> 專剋 7 的細線條斷裂
        # 在白底黑字的影像中，對整體做侵蝕 (erode) 會讓「黑色的字體變粗」！
        kernel = np.ones((3,3), np.uint8)
        bold_img = cv2.erode(adaptive_thresh, kernel, iterations=1)
        img_4 = cv2.copyMakeBorder(bold_img, pad_2, pad_2, pad_2, pad_2, cv2.BORDER_CONSTANT, value=255)
        tracks.append(("Bold", img_4, scale, 0.45))

        # 執行四軌道掃描，不提早中斷，取最高合法的數字！
        for (track_name, track_img, current_scale, min_prob) in tracks:
            ocr_results = self.reader.readtext(track_img, allowlist=self.allowlist)
            
            for (bbox, text, prob) in ocr_results:
                if text.isdigit():
                    stage_num = int(text)
                    if 1 <= stage_num <= 8:
                        current_pad = int(pad_size * current_scale)
                        orig_x = int((bbox[0][0] - current_pad) / current_scale)
                        orig_y = int((bbox[0][1] - current_pad) / current_scale)
                        orig_w = int((bbox[1][0] - bbox[0][0]) / current_scale)
                        orig_h = int((bbox[2][1] - bbox[1][1]) / current_scale)
                        
                        if passes_strict_filter(stage_num, prob, max(0, orig_x), max(0, orig_y), orig_w, orig_h, min_prob):
                            if stage_num > best_stage_num:
                                best_stage_num = stage_num
                                best_bbox = [[orig_x, orig_y], [orig_x+orig_w, orig_y], 
                                             [orig_x+orig_w, orig_y+orig_h], [orig_x, orig_y+orig_h]]

        # ==========================================
        # 🏁 階段推進與繪圖邏輯
        # ==========================================
        if best_stage_num != -1 and best_stage_num >= self.current_stage:
            orig_x, orig_y = int(best_bbox[0][0]), int(best_bbox[0][1])
            orig_w = int(best_bbox[1][0] - best_bbox[0][0])
            orig_h = int(best_bbox[2][1] - best_bbox[1][1])

            pad_box = 15 
            self.last_valid_box = (
                max(0, orig_x + crop_x1 - pad_box), 
                max(0, orig_y + crop_y1 - pad_box), 
                orig_w + (pad_box * 2), 
                orig_h + (pad_box * 2)
            )
            self.lost_patience = 0
            found_number = True
            
            self.history_results.append(best_stage_num)
            
            stage_counts = Counter(self.history_results)
            upgrade_triggered = False
            
            for s_num, count in stage_counts.items():
                if s_num > self.current_stage and count >= self.UPGRADE_THRESHOLD:
                    detected_stage = s_num
                    self.current_stage = s_num
                    self.history_results.clear()
                    upgrade_triggered = True
                    break
                    
            if not upgrade_triggered and best_stage_num == self.current_stage:
                detected_stage = self.current_stage
                        
        if not found_number:
            self.lost_patience += 1
            if self.lost_patience > self.PATIENCE_THRESHOLD:
                self.last_valid_box = None

        return detected_stage

    def draw_boxes(self, frame, override_stage=None):
        display_stage = override_stage if override_stage is not None else self.current_stage
        cx1, cy1, cx2, cy2 = self.current_crop_coords
        
        cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (0, 255, 255), 2)

        if self.last_valid_box and display_stage > 0:
            bx, by, bw, bh = self.last_valid_box
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
            cv2.putText(frame, f"Sign: {display_stage}", (bx, by - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    def force_stage(self, target_stage):
        if target_stage > self.current_stage:
            print(f">>> [SignboardTracker] 觸發強制代償！系統從階段 {self.current_stage} 強制切換為 {target_stage}")
            self.current_stage = target_stage
            self.history_results.clear()
            self.lost_patience = 0
            self.last_valid_box = None