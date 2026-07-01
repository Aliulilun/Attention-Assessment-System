import os
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

        # 新增：OCR 跳幀降載 — 每 OCR_FRAME_INTERVAL 幀才執行完整五軸 OCR
        # 五條軸道 readtext 成本高，逐幀全跑會吃滿 GPU/CPU；其餘幀沿用目前階段。
        # 數值可調：=1 等同每幀都跑（最靈敏）、=2/3 越省資源但偵測延遲略增。
        self.OCR_FRAME_INTERVAL = 2

        self.history_results = deque(maxlen=7)
        self.UPGRADE_THRESHOLD = 2

        self.roi_x, self.roi_y, self.roi_w, self.roi_h = 0, 0, 0, 0
        self.current_crop_coords = (0, 0, 0, 0)

        self.ROI_DEAD_ZONE = 2   # 低於此像素的位移視為抖動，不更新
        self.ROI_EMA_ALPHA  = 0.8 # 每幀移向目標的比例（0=不動，1=即時跟隨）
        self.last_tm_score  = 0.0 # 最近一次模板比對信心值（供外部監控）
        self.tracker        = None # KCF 物體追蹤器（OCR 跳幀時維持框線跟隨）
        self._clahe         = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        self.no_detect_frames = 0  # 連續未偵測到數字的幀數；>= NO_DETECT_FALLBACK 啟用應急框
        self.NO_DETECT_FALLBACK = 30
        self._stage7_consec          = 0  # Stage 6→7 連續符合幀計數（TM 7 信心 > OCR 6 信心）
        self.STAGE7_CONSEC_THRESHOLD = 3  # 需連續幾幀才接受 TM 升階至 Stage 7
        self.RECENTER_MIN   = 5   # re-center 死區：位移小於此值不更新
        self.RECENTER_MAX   = 10  # re-center 上限：位移超過此值視為異常，忽略
        # OpenCV KCF API 版本偵測（只做一次，避免 _init_tracker 每次 try/except）
        self._kcf_create    = cv2.TrackerKCF_create if hasattr(cv2, 'TrackerKCF_create') \
                              else cv2.TrackerKCF.create

        # 模板比對：專用於 Stage 6→7 的「7」辨識（取代 Seven-Hunt OCR 模式）
        # 載入 signboardphoto/ 目錄下所有以「7」開頭的 .png 模板
        _tmpl_dir = os.path.join(os.path.dirname(__file__), '..', 'model', 'signboardphoto')
        _tmpl_files = sorted(f for f in os.listdir(_tmpl_dir) if f.startswith('7') and f.endswith('.png'))
        self.seven_templates = []
        for _fname in _tmpl_files:
            self.seven_templates.extend(self._load_and_prepare_template(os.path.join(_tmpl_dir, _fname)))
        print(f">>> [SignboardTracker] Stage 7 共載入 {len(_tmpl_files)} 張模板，{len(self.seven_templates)} 個旋轉版本")

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

    def _run_ocr_on_crop(self, frame, crop_x1, crop_y1, crop_x2, crop_y2, gray_img_in=None, min_stage=1):
        """
        對給定區域執行三軸掃描，回傳 (best_stage_num, best_bbox_in_crop) 或 (-1, None)
        min_stage：只接受 >= 此值的偵測結果（防止已升階後誤讀低數字）
        """
        crop_img = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        if crop_img.shape[0] == 0 or crop_img.shape[1] == 0:
            return -1, None, 0.0

        gray_img = gray_img_in if gray_img_in is not None else cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        h_img, w_img = gray_img.shape[:2]  # 在 closure 外部計算一次，供 passes_strict_filter 使用

        def passes_strict_filter(stage_val, prob_val, ox, oy, ow, oh, min_prob):
            # \u{1F31F} 7 號專屬通道：問閣打 0.45 折 (0.6 折 -> 0.45 折)，再次降低
            actual_min_prob = min_prob * 0.55 if stage_val == 7 else min_prob
            if prob_val < actual_min_prob:
                return False
            # 修改：字高門檻 20->35，並加入最小面積 600px² 過濾
            # 真實牌子數字在畫面中最少 35px 高；背景雜訊/細小字元通常更小
            # 修改：7號最小字高 25px（EasyOCR 對橫槓7識別率低，不能太嚴）
            # 其他數字維持 32px，背景雜訊通常在 25px 以下
            min_oh = 25 if stage_val == 7 else 32
            if oh < min_oh or oh > 250:
                return False
            if ow * oh < 600:
                return False
            # 7 的寬高比放寬至 1.5（保留基本過濾，但比其他數字 1.1 寬鬆）
            if ow > oh * (1.5 if stage_val == 7 else 1.1):
                return False
            if ow < oh * 0.10:
                return False
            if stage_val == 1 and ow > oh * 0.6:
                return False
            roi_gray = gray_img[max(0, oy):min(h_img, oy+oh), max(0, ox):min(w_img, ox+ow)]
            if roi_gray.size > 0:
                if stage_val == 1 and np.std(roi_gray) < 20:
                    return False
                if stage_val != 7 and np.mean(roi_gray) < 75:
                    return False
                if stage_val == 7 and np.std(roi_gray) < 15:
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
        cl_img = self._clahe.apply(resized_img)
        pad_2 = int(pad_size * scale)
        img_2 = cv2.copyMakeBorder(cl_img, pad_2, pad_2, pad_2, pad_2, cv2.BORDER_CONSTANT, value=255)
        tracks.append(("CLAHE", img_2, scale, 0.45))

        # 軸道 3：適應性二値化 (Adaptive Threshold)
        blur_img = cv2.GaussianBlur(cl_img, (5, 5), 0)
        adaptive_thresh = cv2.adaptiveThreshold(blur_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5)
        img_3 = cv2.copyMakeBorder(adaptive_thresh, pad_2, pad_2, pad_2, pad_2, cv2.BORDER_CONSTANT, value=255)
        tracks.append(("Adaptive", img_3, scale, 0.45))


        # 收集所有通過嚴格過濾的候選物件 (prob, stage_num, bbox)
        candidates = []
        for (track_name, track_img, current_scale, min_prob) in tracks:
            ocr_results = self.reader.readtext(track_img, allowlist=self.allowlist)
            for (bbox, text, prob) in ocr_results:
                if text.isdigit():
                    stage_num = int(text)
                    if min_stage <= stage_num <= 8:
                        current_pad = int(pad_size * current_scale)
                        orig_x = int((bbox[0][0] - current_pad) / current_scale)
                        orig_y = int((bbox[0][1] - current_pad) / current_scale)
                        orig_w = int((bbox[1][0] - bbox[0][0]) / current_scale)
                        orig_h = int((bbox[2][1] - bbox[1][1]) / current_scale)
                        if passes_strict_filter(stage_num, prob, max(0, orig_x), max(0, orig_y), orig_w, orig_h, min_prob):
                            candidates.append((prob, stage_num, [
                                [orig_x, orig_y], [orig_x + orig_w, orig_y],
                                [orig_x + orig_w, orig_y + orig_h], [orig_x, orig_y + orig_h]
                            ]))

        if not candidates:
            return -1, None, 0.0

        # 跨 track NMS：依信心由高到低排序後，同 stage 且 IoU > 0.4 的後續視為重複
        candidates.sort(key=lambda c: c[0], reverse=True)
        kept = []
        for cand in candidates:
            if not any(self._iou(cand[2], k[2]) > 0.4 and cand[1] == k[1] for k in kept):
                kept.append(cand)

        # 多個不同物件時，取信心最高者
        best_prob, best_stage_num, best_bbox = kept[0]
        return best_stage_num, best_bbox, best_prob

    @staticmethod
    def _iou(b1, b2):
        ax1, ay1 = b1[0]; ax2, ay2 = b1[2]
        bx1, by1 = b2[0]; bx2, by2 = b2[2]
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
        return inter / union if union > 0 else 0.0

    def _update_roi_ema(self, sign_cx, sign_cy):
        target_x = max(0, sign_cx - self.roi_w // 2)
        target_y = max(0, sign_cy - self.roi_h // 2)
        if abs(target_x - self.roi_x) > self.ROI_DEAD_ZONE or \
           abs(target_y - self.roi_y) > self.ROI_DEAD_ZONE:
            self.roi_x = int(self.roi_x + self.ROI_EMA_ALPHA * (target_x - self.roi_x))
            self.roi_y = int(self.roi_y + self.ROI_EMA_ALPHA * (target_y - self.roi_y))

    def _make_tracking_zone(self, bx, by, bw, bh):
        """以偵測框為中心展開 TRACKING_PAD，並限制在 ROI 範圍內"""
        return (
            max(self.roi_x,              bx - self.TRACKING_PAD),
            max(self.roi_y,              by - self.TRACKING_PAD),
            min(self.roi_x + self.roi_w, bx + bw + self.TRACKING_PAD),
            min(self.roi_y + self.roi_h, by + bh + self.TRACKING_PAD),
        )

    def _init_tracker(self, frame, box):
        bx, by, bw, bh = [int(v) for v in box]
        t = self._kcf_create()
        t.init(frame, (bx, by, bw, bh))
        self.tracker = t

    def _load_and_prepare_template(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f">>> [SignboardTracker] 警告：找不到模板圖片 {os.path.basename(path)}，略過此檔")
            return []
        # Otsu 自動找閾值，適用任何背景顏色（白/灰/藍均可）
        _, digit_mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(digit_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
            pad = 5
            img = img[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad]
        # 正規化為二值圖（白底黑字），消除背景色差異
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return [(angle, self._rotate_template(img, angle)) for angle in range(-20, 25, 10)]

    def _rotate_template(self, img, angle):
        h, w = img.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, float(angle), 1.0)
        cos, sin = abs(M[0, 0]), abs(M[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        return cv2.warpAffine(img, M, (new_w, new_h), borderValue=255)

    def _match_seven(self, frame, crop_x1, crop_y1, crop_x2, crop_y2):
        """多尺度 + 旋轉模板比對，專門偵測數字「7」，回傳 (7, bbox) 或 (-1, None)"""
        if not self.seven_templates:
            return -1, None
        roi = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        if roi.size == 0:
            return -1, None
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        _, roi_gray = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        roi_h, roi_w = roi_gray.shape
        best_score = 0.0
        best_bbox  = None
        THRESHOLD  = 0.65
        EARLY_EXIT = 0.90
        scales = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.5]
        for _, tmpl in self.seven_templates:
            th, tw = tmpl.shape
            for scale in scales:
                new_w, new_h = int(tw * scale), int(th * scale)
                if new_w < 10 or new_h < 10 or new_w > roi_w or new_h > roi_h:
                    continue
                scaled = cv2.resize(tmpl, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                result = cv2.matchTemplate(roi_gray, scaled, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                if max_val > best_score:
                    best_score = max_val
                    best_bbox  = [
                        [max_loc[0], max_loc[1]],
                        [max_loc[0] + new_w, max_loc[1]],
                        [max_loc[0] + new_w, max_loc[1] + new_h],
                        [max_loc[0], max_loc[1] + new_h]
                    ]
                if best_score >= EARLY_EXIT:
                    break
            if best_score >= EARLY_EXIT:
                break
        self.last_tm_score = best_score
        if best_score >= THRESHOLD:
            return 7, best_bbox
        return -1, None

    def detect_stage(self, frame):
        if self.current_stage >= 8:
            return None

        self.frame_counter += 1
        self.no_detect_frames += 1  # 每幀遞增；偵測成功時歸零

        # 每幀更新 KCF 追蹤器，讓框線在 OCR 跳幀時也能持續跟隨牌子
        if self.tracker is not None:
            ok, tracked_box = self.tracker.update(frame)
            if ok:
                tx, ty, tw, th = [int(v) for v in tracked_box]
                self.last_valid_box = (tx, ty, tw, th)
                self._update_roi_ema(tx + tw // 2, ty + th // 2)
            else:
                self.tracker = None  # 追蹤失敗，等下次 OCR 重新初始化

        # 每幀檢查：若黃色掃描框中心偏離 sign 超過 5px，立即重新置中
        if self.last_valid_box is not None and self.current_crop_coords != (0, 0, 0, 0):
            cx1, cy1, cx2, cy2 = self.current_crop_coords
            box_cx = (cx1 + cx2) // 2
            box_cy = (cy1 + cy2) // 2
            bx, by, bw, bh = self.last_valid_box
            sign_cx = bx + bw // 2
            sign_cy = by + bh // 2
            dx = abs(box_cx - sign_cx)
            dy = abs(box_cy - sign_cy)
            # RECENTER_MIN 以下：死區不動；MIN~MAX：正常跟隨；>MAX：單次跳動過大，忽略
            if (dx > self.RECENTER_MIN or dy > self.RECENTER_MIN) and (dx <= self.RECENTER_MAX and dy <= self.RECENTER_MAX):
                half_w = (cx2 - cx1) // 2
                half_h = (cy2 - cy1) // 2
                new_x1 = max(self.roi_x, sign_cx - half_w)
                new_y1 = max(self.roi_y, sign_cy - half_h)
                new_x2 = min(self.roi_x + self.roi_w, sign_cx + half_w)
                new_y2 = min(self.roi_y + self.roi_h, sign_cy + half_h)
                self.current_crop_coords = (new_x1, new_y1, new_x2, new_y2)

        # 連續 NO_DETECT_FALLBACK 幀未偵測 → 重設掃描框至 ROI 中心 1.5 倍，並清除 tracker
        if self.no_detect_frames >= self.NO_DETECT_FALLBACK and self.current_stage != 7:
            fh, fw = frame.shape[:2]
            cx = self.roi_x + self.roi_w // 2
            cy = self.roi_y + self.roi_h // 2
            hw = self.roi_w * 3 // 4
            hh = self.roi_h * 3 // 4
            self.current_crop_coords = (
                max(0, cx - hw), max(0, cy - hh),
                min(fw, cx + hw), min(fh, cy + hh),
            )
            self.tracker = None

        # 新增：OCR 跳幀降載 — 非掃描幀直接回傳 None（沿用目前階段），不動 patience/history
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

        # 黃色框即時同步掃描範圍：直接沿用已由 KCF / re-center 維護的 current_crop_coords，
        # 不在 OCR 幀重新推算，確保「顯示框 = 掃描範圍」始終一致。
        # 只有 current_crop_coords 尚未初始化 或 強制全域掃描 時，才退回全 ROI。
        if use_tracking and self.current_crop_coords != (0, 0, 0, 0):
            crop_x1, crop_y1, crop_x2, crop_y2 = self.current_crop_coords
        else:
            crop_x1, crop_y1, crop_x2, crop_y2 = full_roi_x1, full_roi_y1, full_roi_x2, full_roi_y2
            self.current_crop_coords = (crop_x1, crop_y1, crop_x2, crop_y2)

        # 首先嘗試主要區域
        best_stage_num, best_bbox, best_prob = self._run_ocr_on_crop(frame, crop_x1, crop_y1, crop_x2, crop_y2, min_stage=self.current_stage)

        # 修改：備援全域掃描改為「純位置重錨」
        # 追蹤框掃不到牌子時，補做全域掃描找回位置 → 只更新 last_valid_box，
        # 不把結果加入 history_results，防止 ROI 背景元素造成誤判升階。
        # 下一幀追蹤框會以更新後的位置繼續掃描，才真正計入歷史。
        if best_stage_num == -1 and use_tracking:
            backup_stage_num, backup_bbox, _ = self._run_ocr_on_crop(frame, full_roi_x1, full_roi_y1, full_roi_x2, full_roi_y2, min_stage=self.current_stage)
            if backup_stage_num != -1 and backup_stage_num >= self.current_stage:
                bx2 = int(backup_bbox[0][0]) + full_roi_x1
                by2 = int(backup_bbox[0][1]) + full_roi_y1
                bw2 = int(backup_bbox[1][0] - backup_bbox[0][0])
                bh2 = int(backup_bbox[2][1] - backup_bbox[1][1])
                pad_b = 15
                self.last_valid_box = (max(0, bx2 - pad_b), max(0, by2 - pad_b),
                                       bw2 + pad_b*2, bh2 + pad_b*2)
                self.lost_patience = 0
                self._init_tracker(frame, self.last_valid_box)
                # 重錨後立即縮回追蹤結界，讓掃描框大小與正常偵測路徑一致
                self.current_crop_coords = self._make_tracking_zone(*self.last_valid_box)
                # best_stage_num 仍為 -1 → 本幀不加入 history_results

        # Stage 6→7：模板比對補掃，OCR 未偵測到「7」時啟動
        # 防誤判：需連續 STAGE7_CONSEC_THRESHOLD 幀均符合「TM 7 信心 > OCR 6 信心」才升階
        if self.current_stage == 6 and best_stage_num != 7:
            tm_stage, tm_bbox = self._match_seven(frame, crop_x1, crop_y1, crop_x2, crop_y2)
            if tm_stage == 7:
                ocr_6_conf = best_prob if best_stage_num == 6 else 0.0
                if ocr_6_conf > self.last_tm_score:
                    self._stage7_consec = 0  # OCR 的 6 信心更高，重置連續計數
                else:
                    self._stage7_consec += 1
                    if self._stage7_consec >= self.STAGE7_CONSEC_THRESHOLD:
                        best_stage_num = 7
                        best_bbox      = tm_bbox
            else:
                self._stage7_consec = 0  # TM 未達門檻，重置連續計數

        # ==========================================
        # \u{1F3C1} 階段推進與繪圖邏輯
        # ==========================================
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
            self.no_detect_frames = 0
            self._init_tracker(frame, self.last_valid_box)

            # 偵測成功後立即把掃描框擴展到新的追蹤結界，確保下一輪 current_crop_coords 大小正確
            self.current_crop_coords = self._make_tracking_zone(*self.last_valid_box)

            # ROI 跟隨 Sign（Dead Zone + EMA）
            bx_n, by_n, bw_n, bh_n = self.last_valid_box
            self._update_roi_ema(bx_n + bw_n // 2, by_n + bh_n // 2)

            self.history_results.append(best_stage_num)
            stage_counts = Counter(self.history_results)

            qualified = sorted(
                s for s, c in stage_counts.items()
                if s > self.current_stage and c >= self.UPGRADE_THRESHOLD
            )
            if qualified:
                detected_stage = qualified[0]
                self.current_stage = qualified[0]
                self.history_results.clear()

        else:
            # best_stage_num == -1：本幀未偵測到牌子
            self.lost_patience += 1

        # force_full_scan 且本幀無偵測：恢復追蹤結界，避免下一幀沿用全 ROI 大小的掃描框
        if force_full_scan and best_stage_num == -1 and self.last_valid_box is not None:
            self.current_crop_coords = self._make_tracking_zone(*self.last_valid_box)

        return detected_stage

    def draw_boxes(self, frame, current_stage):
        """
        在畫面上繪製：
        1. 當前掃描框（黃色細框）— 顯示這幀實際掃了哪個區域
        2. 上次有效偵測框（綠色粗框）＋階段標籤
        """
        # 繪製當前掃描區域（黃色）
        cx1, cy1, cx2, cy2 = self.current_crop_coords
        use_fallback = (not (cx2 > cx1 and cy2 > cy1) or (self.no_detect_frames >= self.NO_DETECT_FALLBACK)) and current_stage != 7
        if use_fallback:
            # 掃描框無效 或 連續 20 幀未偵測 → 應急框：以當前 ROI 中心為基準擴大 1.5 倍，偵測到後自動恢復
            fh, fw = frame.shape[:2]
            cx = self.roi_x + self.roi_w // 2
            cy = self.roi_y + self.roi_h // 2
            hw = self.roi_w * 3 // 4   # 半寬 = roi_w * 0.75，總寬 = roi_w * 1.5
            hh = self.roi_h * 3 // 4
            cx1 = max(0, cx - hw)
            cy1 = max(0, cy - hh)
            cx2 = min(fw, cx + hw)
            cy2 = min(fh, cy + hh)
        if cx2 > cx1 and cy2 > cy1:
            cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (0, 255, 255), 2)

        # 繪製上次有效偵測框（綠色）＋階段文字
        if self.last_valid_box is not None:
            bx, by, bw, bh = self.last_valid_box
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
            cv2.putText(frame, f"Sign:{current_stage}",
                        (bx, max(10, by - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)