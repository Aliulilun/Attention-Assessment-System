import cv2
import easyocr

class SignboardTracker:
    def __init__(self, allowlist='12345678'):
        print(">>> [SignboardTracker] 正在載入 EasyOCR 引擎...")
        self.reader = easyocr.Reader(['en'])
        self.allowlist = allowlist
        
        self.current_stage = 0
        self.last_valid_box = None
        self.lost_patience = 0
        
        self.SEARCH_PAD_NORMAL = 60
        self.SEARCH_PAD_LOST = 250
        self.current_pad = self.SEARCH_PAD_NORMAL
        
        # 🌟 修改 1：新增「防呆耐心值」參數，從原本寫死的 10 改為 3。
        # 只要連續 3 幀抓不到，黃色結界瞬間放大抓牌子，大幅降低卡頓感！
        self.PATIENCE_THRESHOLD = 3 
        
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
        # 🌟 應用新的耐心值
        if self.lost_patience > self.PATIENCE_THRESHOLD:
            self.current_pad = self.SEARCH_PAD_LOST
            
        if self.last_valid_box:
            bx, by, bw, bh = self.last_valid_box
            crop_x1 = max(self.roi_x, bx - self.current_pad)
            crop_y1 = max(self.roi_y, by - self.current_pad)
            crop_x2 = min(self.roi_x + self.roi_w, bx + bw + self.current_pad)
            crop_y2 = min(self.roi_y + self.roi_h, by + bh + self.current_pad)
        else:
            crop_x1, crop_y1 = self.roi_x, self.roi_y
            crop_x2, crop_y2 = self.roi_x + self.roi_w, self.roi_y + self.roi_h

        self.current_crop_coords = (crop_x1, crop_y1, crop_x2, crop_y2)

        crop_img = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        ocr_results = self.reader.readtext(crop_img, allowlist=self.allowlist)
        
        found_number = False
        detected_stage = None
        
        for (bbox, text, prob) in ocr_results:
            # 🌟 修改 2：將信心度門檻稍微降低 (0.5 -> 0.45)
            # 避免影片前幾幀因為輕微模糊而漏掉牌子，提升初次抓取速度
            if prob > 0.45 and text.isdigit(): 
                stage_num = int(text)
                if 1 <= stage_num <= 8:
                    # 🛡️ 核心防呆機制：單向鎖定 (只能前進或不動，不能後退)
                    if stage_num >= self.current_stage:
                        detected_stage = stage_num
                        self.current_stage = stage_num
                        
                        self.last_valid_box = (
                            int(bbox[0][0]) + crop_x1, 
                            int(bbox[0][1]) + crop_y1, 
                            int(bbox[1][0] - bbox[0][0]), 
                            int(bbox[2][1] - bbox[1][1])
                        )
                        self.lost_patience = 0
                        self.current_pad = self.SEARCH_PAD_NORMAL
                        found_number = True
                        break
                    else:
                        # 辨識到較小的數字，視為雜訊幻覺，直接略過！
                        pass
                        
        if not found_number:
            self.lost_patience += 1

        return detected_stage

    def draw_boxes(self, frame, override_stage=None):
        display_stage = override_stage if override_stage is not None else self.current_stage

        cx1, cy1, cx2, cy2 = self.current_crop_coords
        # 讓黃色的搜索結界稍微粗一點 (thickness=2)，方便你觀察它擴大縮小的反應速度
        cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (0, 255, 255), 2)

        if self.last_valid_box and display_stage > 0:
            bx, by, bw, bh = self.last_valid_box
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
            cv2.putText(frame, f"Sign: {display_stage}", (bx, by - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)