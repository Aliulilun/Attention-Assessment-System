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

        # \u{1F31F} 修改：擴大追蹤結界 (80 -> 180)：防止攝影機微動時框跟不上牌子
        self.TRACKING_PAD = 180
        # 縮短 patience (15 -> 10)：更快回到全域對照
        self.PATIENCE_THRESHOLD = 10

        # \u{1F31F} 新增：定期全域重錨機制—每 40 幀強制做一次全域 ROI 扫描，重新錨定牌子位置
        self.FULL_SCAN_INTERVAL = 40
        self.frame_counter = 0

        # 🌟 新增：OCR 跳幀降載 — 每 OCR_FRAME_INTERVAL 幀才執行完整五軸 OCR
        # 五條軸道 readtext 成本高，逐幀全跑會吃滿 GPU/CPU；其餘幀沿用目前階段。
        # 數值可調：=1 等同每幀都跑（最靈敏）、=2/3 越省資源但偵測延遲略增。
        self.OCR_FRAME_INTERVAL = 2

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

    def _run_ocr_on_crop(self, frame, crop_x1, crop_y1, crop_x2, crop_y2, gray_img_in=None):
        """
        對給定區域執行七軸掃描，回傳 (best_stage_num, best_bbox_in_crop) 或 (-1, None)
        """
        crop_img = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        if crop_img.shape[0] == 0 or crop_img.shape[1] == 0:
            return -1, None

        gray_img = gray_img_in if gray_img_in is not None else cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

        def passes_strict_filter(stage_val, prob_val, ox, oy, ow, oh, min_prob):
            # \u{1F31F} 7 號專屬通道：問閣打 0.45 折 (0.6 折 -> 0.45 折)，再次降低
            actual_min_prob = min_prob * 0.55 if stage_val == 7 else min_prob
            if prob_val < actual_min_prob:
                return False
            # 🌟 修改：字高門檻 20->35，並加入最小面積 600px² 過濾
            # 真實牌子數字在畫面中最少 35px 高；背景雜訊/細小字元通常更小
            # 🌟 修改：7號最小字高 25px（EasyOCR 對橫槓7識別率低，不能太嚴）
            # 其他數字維持 35px，背景雜訊通常在 25px 以下
            min_oh = 25 if stage_val == 7 else 35
            if oh < min_oh or oh > 250:
                return False
            if ow * oh < 600:
                return False
            # 🌟 7 的寬高比放寬至 1.5（保留基本過濾，但比其他數字 1.1 寬鬆）
            if ow > oh * (1.5 if stage_val == 7 else 1.1):
                return False
            if ow < oh * 0.10:
                return False
            if stage_val == 1 and ow > oh * 0.6:
                return False
            roi_gray = gray_img[max(0, oy):oy+oh, max(0, ox):ox+ow]
            if roi_gray.size > 0:
                if stage_val != 7 and np.mean(roi_gray) < 85:
                    return False
            return True

        pad_size = 20
        tracks = []

        # 軸道 1：原始灰階
        img_1 = cv2.copyMakeBorder(gray_img, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT, value=255)
        tracks.append(("Normal", img_1, 1.0, 0.50))

        # 軸道 2：放大 + CLAHE
        scale = 2.0
        resized_img = cv2.resize(gray_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl_img = clahe.apply(resized_img)
        pad_2 = int(pad_size * scale)
        img_2 = cv2.copyMakeBorder(cl_img, pad_2, pad_2, pad_2, pad_2, cv2.BORDER_CONSTANT, value=255)
        tracks.append(("CLAHE", img_2, scale, 0.45))

        # 軸道 3：適應性二値化 (Adaptive Threshold)
        blur_img = cv2.GaussianBlur(cl_img, (5, 5), 0)
        adaptive_thresh = cv2.adaptiveThreshold(blur_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5)
        img_3 = cv2.copyMakeBorder(adaptive_thresh, pad_2, pad_2, pad_2, pad_2, cv2.BORDER_CONSTANT, value=255)
        tracks.append(("Adaptive", img_3, scale, 0.45))

        # 軸道 4：字體侵蒒加粗 (Bold Erosion) - 專克 7 的細線條斷裂
        kernel = np.ones((3, 3), np.uint8)
        bold_img = cv2.erode(adaptive_thresh, kernel, iterations=1)
        img_4 = cv2.copyMakeBorder(bold_img, pad_2, pad_2, pad_2, pad_2, cv2.BORDER_CONSTANT, value=255)
        tracks.append(("Bold", img_4, scale, 0.45))

        # \u{1F31F} 軸道 5 (新增)：锐化強化 (Sharpen) - 專門強化 7 的水平橫筆畫和斜線
        sharpen_kernel = np.array([[-1, -1, -1],
                                   [-1,  9, -1],
                                   [-1, -1, -1]], dtype=np.float32)
        sharp_img = cv2.filter2D(resized_img, -1, sharpen_kernel)
        sharp_img = np.clip(sharp_img, 0, 255).astype(np.uint8)
        img_5 = cv2.copyMakeBorder(sharp_img, pad_2, pad_2, pad_2, pad_2, cv2.BORDER_CONSTANT, value=255)
        tracks.append(("Sharpen", img_5, scale, 0.50))

        # 🌟 新增軌道 6：Otsu 全域二值化 — 對雙峰直方圖(白底黑字)效果好
        # 特別針對「7」：橫槓式7在全域閾值下比 Adaptive 更清晰
        _, otsu_img = cv2.threshold(resized_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img_6 = cv2.copyMakeBorder(otsu_img, pad_2, pad_2, pad_2, pad_2, cv2.BORDER_CONSTANT, value=255)
        tracks.append(("Otsu", img_6, scale, 0.40))

        # 🌟 新增軌道 7：Inverted（反白）— EasyOCR 有時對白字深底識別更穩定
        inv_img = cv2.bitwise_not(otsu_img)
        img_7 = cv2.copyMakeBorder(inv_img, pad_2, pad_2, pad_2, pad_2, cv2.BORDER_CONSTANT, value=0)
        tracks.append(("Inverted", img_7, scale, 0.40))

        best_stage_num = -1
        best_bbox = None
        # 🌟 修改：改以「OCR 信心(prob)最高」者為準
        best_prob = -1.0

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
                            # 🌟 修改：原本「stage_num > best_stage_num」會偏好畫面中最大的數字，
                            # 背景被誤讀成大數字時容易選錯造成跳階；改為在通過嚴格過濾的候選中
                            # 取「信心最高」者，較不受雜訊干擾。逐階防跳的保護在 detect_stage 升階處。
                            if prob > best_prob:
                                best_prob = prob
                                best_stage_num = stage_num
                                best_bbox = [[orig_x, orig_y], [orig_x+orig_w, orig_y],
                                             [orig_x+orig_w, orig_y+orig_h], [orig_x, orig_y+orig_h]]

        # ============================================================
        # 🌟 新增：7號專獵模式 (Seven-Hunt Mode)
        # 觸發條件：current_stage == 6（正在等待升到第7關）
        #
        # 根本問題：EasyOCR 把帶橫槓的歐式「7」在 allowlist='1234567' 下
        # 映射成「1」→ stage_num=1 < current_stage=6 → 被過濾掉，永遠升不了階。
        #
        # 解法：用 allowlist='7' 強制 EasyOCR 只能輸出「7」，
        # 搭配 3x 放大讓細節更清晰，prob 門檻再降低（幾何過濾仍在）。
        # 只在 current_stage==6 時啟動，避免其他階段誤觸發。
        # ============================================================
        if self.current_stage == 6 and best_stage_num != 7:
            scale_3 = 3.0
            pad_3 = int(pad_size * scale_3)
            resized_3x = cv2.resize(gray_img, None, fx=scale_3, fy=scale_3,
                                    interpolation=cv2.INTER_CUBIC)
            clahe_3x = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8)).apply(resized_3x)
            _, otsu_3x = cv2.threshold(resized_3x, 0, 255,
                                        cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            inv_3x = cv2.bitwise_not(otsu_3x)

            seven_hunt_tracks = [
                (clahe_3x, 255, 0.25),   # CLAHE 3x，白底
                (otsu_3x,  255, 0.22),   # Otsu 3x，白底
                (inv_3x,   0,   0.22),   # 反白 3x，黑底
            ]
            for hunt_img, border_val, min_p7 in seven_hunt_tracks:
                bordered = cv2.copyMakeBorder(hunt_img, pad_3, pad_3, pad_3, pad_3,
                                              cv2.BORDER_CONSTANT, value=border_val)
                # allowlist='7'：強制任何匹配形狀輸出為「7」
                hunt_results = self.reader.readtext(bordered, allowlist='7')
                for (bbox, text, prob) in hunt_results:
                    if text.strip() != '7':
                        continue
                    h_pad = pad_3
                    orig_x = int((bbox[0][0] - h_pad) / scale_3)
                    orig_y = int((bbox[0][1] - h_pad) / scale_3)
                    orig_w = int((bbox[1][0] - bbox[0][0]) / scale_3)
                    orig_h = int((bbox[2][1] - bbox[1][1]) / scale_3)
                    if passes_strict_filter(7, prob, max(0, orig_x), max(0, orig_y),
                                            orig_w, orig_h, min_p7):
                        if prob > best_prob:
                            best_prob = prob
                            best_stage_num = 7
                            best_bbox = [[orig_x, orig_y], [orig_x + orig_w, orig_y],
                                         [orig_x + orig_w, orig_y + orig_h],
                                         [orig_x, orig_y + orig_h]]

        return best_stage_num, best_bbox

    def detect_stage(self, frame):
        self.frame_counter += 1

        # 🌟 新增：OCR 跳幀降載 — 非掃描幀直接回傳 None（沿用目前階段），不動 patience/history
        if self.frame_counter % self.OCR_FRAME_INTERVAL != 0:
            return None

        # ==========================================
        # \u{1F31F} 雙區域扫描策略
        # - 正常模式：使用追蹤結界（tracking zone）快速扫描
        # - 每 40 幀一次 OR tracking 失敗時：立即備援全域 ROI 扫描，防止攝影機小幅移動导致跟个
        # ==========================================
        full_roi_x1 = self.roi_x
        full_roi_y1 = self.roi_y
        full_roi_x2 = self.roi_x + self.roi_w
        full_roi_y2 = self.roi_y + self.roi_h

        # 決定主要扫描區域
        force_full_scan = (self.frame_counter % self.FULL_SCAN_INTERVAL == 0)
        use_tracking = (self.last_valid_box is not None
                        and self.lost_patience <= self.PATIENCE_THRESHOLD
                        and not force_full_scan)

        if use_tracking:
            bx, by, bw, bh = self.last_valid_box
            crop_x1 = max(self.roi_x, bx - self.TRACKING_PAD)
            crop_y1 = max(self.roi_y, by - self.TRACKING_PAD)
            crop_x2 = min(full_roi_x2, bx + bw + self.TRACKING_PAD)
            crop_y2 = min(full_roi_y2, by + bh + self.TRACKING_PAD)
        else:
            crop_x1, crop_y1, crop_x2, crop_y2 = full_roi_x1, full_roi_y1, full_roi_x2, full_roi_y2

        self.current_crop_coords = (crop_x1, crop_y1, crop_x2, crop_y2)

        # 首先嘗試主要區域
        best_stage_num, best_bbox = self._run_ocr_on_crop(frame, crop_x1, crop_y1, crop_x2, crop_y2)

        # 🌟 修改：備援全域掃描改為「純位置重錨」
        # 追蹤框掃不到牌子時，補做全域掃描找回位置 → 只更新 last_valid_box，
        # 不把結果加入 history_results，防止 ROI 背景元素造成誤判升階。
        # 下一幀追蹤框會以更新後的位置繼續掃描，才真正計入歷史。
        if best_stage_num == -1 and use_tracking:
            backup_stage_num, backup_bbox = self._run_ocr_on_crop(frame, full_roi_x1, full_roi_y1, full_roi_x2, full_roi_y2)
            if backup_stage_num != -1 and backup_stage_num >= self.current_stage:
                bx2 = int(backup_bbox[0][0]) + full_roi_x1
                by2 = int(backup_bbox[0][1]) + full_roi_y1
                bw2 = int(backup_bbox[1][0] - backup_bbox[0][0])
                bh2 = int(backup_bbox[2][1] - backup_bbox[1][1])
                pad_b = 15
                self.last_valid_box = (max(0, bx2 - pad_b), max(0, by2 - pad_b),
                                       bw2 + pad_b*2, bh2 + pad_b*2)
                self.lost_patience = 0
                self.current_crop_coords = (full_roi_x1, full_roi_y1, full_roi_x2, full_roi_y2)
                # best_stage_num 仍為 -1 → 本幀不加入 history_results

        # ==========================================
        # \u{1F3C1} 階段推進與繪圖邏輯
        # ==========================================
        found_number = False
        detected_stage = None

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

            # 🌟 修改：逐階前進防跳機制
            # 原本遍歷 Counter 並 break 在「第一個」達標者，順序不確定，可能直接跳到被誤讀的高數字。
            # 改為：在所有「票數達門檻(2幀) 且 大於目前階段」的候選中，取「最接近的下一階(最小者)」。
            # 仍保留 2 幀確認（含 7 號）；逐階前進能避免單一雜訊把階段一次跳到高位且回不來。
            qualified = sorted(
                s for s, c in stage_counts.items()
                if s > self.current_stage and c >= self.UPGRADE_THRESHOLD
            )
            if qualified:
                detected_stage = qualified[0]
                self.current_stage = qualified[0]
                self.history_results.clear()
                upgrade_triggered = True

        else:
            # best_stage_num == -1：本幀未偵測到牌子
            self.lost_patience += 1

        return detected_stage

    def draw_boxes(self, frame, current_stage):
        """
        在畫面上繪製：
        1. 當前掃描框（黃色細框）— 顯示這幀實際掃了哪個區域
        2. 上次有效偵測框（綠色粗框）＋階段標籤
        """
        import cv2  # 確保 cv2 在此作用域可用

        # 繪製當前掃描區域（黃色）
        cx1, cy1, cx2, cy2 = self.current_crop_coords
        if cx2 > cx1 and cy2 > cy1:
            cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (0, 255, 255), 1)

        # 繪製上次有效偵測框（綠色）＋階段文字
        if self.last_valid_box is not None:
            bx, by, bw, bh = self.last_valid_box
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
            cv2.putText(frame, f"Sign:{current_stage}",
                        (bx, max(10, by - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
