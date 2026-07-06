import os
import cv2
import numpy as np
import easyocr
from collections import deque, Counter

class SignboardTracker:

    # ══════════════════════════════════════════════════════════════════════════
    # 初始化
    # ══════════════════════════════════════════════════════════════════════════

    def __init__(self, allowlist='12345678'):
        print(">>> [SignboardTracker] 正在載入 EasyOCR 引擎...")
        self.reader = easyocr.Reader(['en'])
        self.allowlist = allowlist

        # ── 基本狀態 ──────────────────────────────────────────────────────────
        self.current_stage   = 0     # 目前判定階段（0 = 尚未開始）
        self.frame_counter   = 0     # 累積幀數，用於跳幀與定期全域掃描
        self.history_results = deque(maxlen=7)  # 升階投票佇列（最近 7 筆 OCR 結果）

        # ── OCR 偵測設定 ───────────────────────────────────────────────────────
        self.OCR_FRAME_INTERVAL        = 2   # Stage 1–6 OCR 執行間隔（每 N 幀執行一次）
        self.OCR_FRAME_INTERVAL_STAGE7 = 5   # Stage 7 TM 執行間隔（降低頻率節省效能）
        self.UPGRADE_THRESHOLD         = 2   # 升階所需最低得票數（history_results 內）
        self.FULL_SCAN_INTERVAL        = 40  # 每隔幾幀強制全域 ROI 掃描以重新錨定牌子
        self.PATIENCE_THRESHOLD        = 10  # 連續未偵測幾幀後放棄追蹤，改做全域掃描
        self.lost_patience             = 0   # 當前連續未偵測幀數（計數器）
        self._clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))  # CLAHE 影像增強器

        # ── Stage 7 模板比對設定 ──────────────────────────────────────────────
        self.last_tm_score           = 0.0  # 最近一次模板比對信心值（供 Stage 6→7 比較用）
        self._stage7_consec          = 0    # 連續符合「TM 7 > OCR 6 信心」的幀數計數
        self.STAGE7_CONSEC_THRESHOLD = 3    # 需連續幾幀符合才允許升至 Stage 7

        # ── 黃框（掃描範圍）設定 ──────────────────────────────────────────────
        self.roi_x, self.roi_y, self.roi_w, self.roi_h = 0, 0, 0, 0
        self.current_crop_coords = (0, 0, 0, 0)  # 黃色掃描框目前座標 (x1,y1,x2,y2)
        self.TRACKING_PAD    = 180  # 掃描框在 sign 框外擴展的像素數
        self.ROI_DEAD_ZONE   = 2    # ROI 中心位移低於此值（px）視為抖動，不更新
        self.ROI_EMA_ALPHA   = 0.8  # ROI 跟隨平滑係數（0=不動，1=即時跟隨）
        self.RECENTER_MIN    = 5    # 黃框置中死區：位移低於此值（px）不重新置中
        self.RECENTER_MAX    = 10   # 黃框置中上限：單幀位移超過此值視為異常，忽略
        self.no_detect_frames   = 0   # 連續未偵測到數字的幀數（計數器）
        self.NO_DETECT_FALLBACK = 25  # 達到此幀數後觸發應急框（重設至 ROI 中心 1.5 倍）
        self._fallback_active     = False  # 是否正在應急框模式（遲滯防閃爍）
        self.FALLBACK_EXIT_STREAK = 3      # 連續幾次 OCR 成功才退出應急框
        self._fallback_exit_count = 0      # 當前連續 OCR 成功次數（退出計數器）

        # ── Sign 框（KCF 追蹤）設定 ───────────────────────────────────────────
        self.last_valid_box  = None  # 綠色 sign 框目前座標 (bx, by, bw, bh)
        self.tracker         = None  # KCF 追蹤器實例（OCR 跳幀期間維持框跟隨）
        self.KCF_JITTER_ZONE = 3     # KCF 微抖動死區：中心位移低於此值（px）不更新 sign 框
        self.KCF_MAX_JUMP    = 40    # KCF 單幀最大位移（px）；超過視為誤跳拒絕（應急框模式下跳過）
        self._appear_snapshot     = None  # OCR 確認時儲存的 sign 外觀直方圖（遮擋偵測基準）
        self.APPEAR_SIM_THRESHOLD = 0.55  # 直方圖相關係數低於此值視為手遮擋，凍結 sign 框
        self._low_sim_streak      = 0     # 連續低相似度幀數計數器
        self.LOW_SIM_STREAK_MAX   = 3     # 需連續幾幀低相似度才觸發凍結（避免單幀誤判）
        self._kcf_create = cv2.TrackerKCF_create if hasattr(cv2, 'TrackerKCF_create') \
                           else cv2.TrackerKCF.create  # API 版本相容偵測（只執行一次）

        # 載入 Stage 7 模板：signboardphoto/ 目錄下所有以「7」開頭的 .png
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
        self.current_crop_coords = (
            self.roi_x, self.roi_y,
            self.roi_x + self.roi_w, self.roi_y + self.roi_h,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # 純工具函式
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _iou(b1, b2):
        """計算兩個 bbox（[[x1,y1],[x2,y1],[x2,y2],[x1,y2]] 格式）的 IoU"""
        ax1, ay1 = b1[0]; ax2, ay2 = b1[2]
        bx1, by1 = b2[0]; bx2, by2 = b2[2]
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
        return inter / union if union > 0 else 0.0

    def _compute_hist(self, frame, bx, by, bw, bh):
        """計算指定區域的灰階直方圖（正規化），供外觀相似度比對使用"""
        crop = frame[by:by+bh, bx:bx+bw]
        if crop.size == 0:
            return None
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
        hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
        cv2.normalize(hist, hist)
        return hist

    @staticmethod
    def _passes_strict_filter(stage_val, prob_val, ox, oy, ow, oh, min_prob, gray_img):
        """OCR 候選框的嚴格過濾器（幾何比例 + 像素統計）"""
        actual_min_prob = min_prob * 0.65 if stage_val == 7 else min_prob
        if prob_val < actual_min_prob:
            return False
        min_oh = 25 if stage_val == 7 else 32
        if oh < min_oh or oh > 250:
            return False
        if ow * oh < 600:
            return False
        if ow > oh * (1.5 if stage_val == 7 else 1.1):
            return False
        if ow < oh * 0.10:
            return False
        if stage_val == 1 and ow > oh * 0.6:
            return False
        h_img, w_img = gray_img.shape[:2]
        roi_gray = gray_img[max(0, oy):min(h_img, oy+oh), max(0, ox):min(w_img, ox+ow)]
        if roi_gray.size > 0:
            if stage_val == 1 and np.std(roi_gray) < 20:
                return False
            if stage_val != 7 and np.mean(roi_gray) < 75:
                return False
            if stage_val == 7 and np.std(roi_gray) < 15:
                return False
        return True

    # ══════════════════════════════════════════════════════════════════════════
    # 黃框（掃描範圍）管理
    # ══════════════════════════════════════════════════════════════════════════

    def _update_roi_ema(self, sign_cx, sign_cy):
        """以 EMA 平滑方式讓 ROI 跟隨 sign 中心（含死區防抖）"""
        target_x = max(0, sign_cx - self.roi_w // 2)
        target_y = max(0, sign_cy - self.roi_h // 2)
        if abs(target_x - self.roi_x) > self.ROI_DEAD_ZONE or \
           abs(target_y - self.roi_y) > self.ROI_DEAD_ZONE:
            self.roi_x = int(self.roi_x + self.ROI_EMA_ALPHA * (target_x - self.roi_x))
            self.roi_y = int(self.roi_y + self.ROI_EMA_ALPHA * (target_y - self.roi_y))

    def _make_tracking_zone(self, bx, by, bw, bh):
        """以 sign 框為中心向外展開 TRACKING_PAD，限制在 ROI 範圍內，作為黃框掃描範圍"""
        return (
            max(self.roi_x,              bx - self.TRACKING_PAD),
            max(self.roi_y,              by - self.TRACKING_PAD),
            min(self.roi_x + self.roi_w, bx + bw + self.TRACKING_PAD),
            min(self.roi_y + self.roi_h, by + bh + self.TRACKING_PAD),
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Sign 框 / KCF 追蹤管理
    # ══════════════════════════════════════════════════════════════════════════

    def _init_tracker(self, frame, box):
        """以指定 box 初始化 KCF 追蹤器"""
        bx, by, bw, bh = [int(v) for v in box]
        t = self._kcf_create()
        t.init(frame, (bx, by, bw, bh))
        self.tracker = t

    # ══════════════════════════════════════════════════════════════════════════
    # OCR 偵測
    # ══════════════════════════════════════════════════════════════════════════

    def _run_ocr_on_crop(self, frame, crop_x1, crop_y1, crop_x2, crop_y2, gray_img_in=None, min_stage=1):
        """
        對給定區域執行三軌掃描（Normal / CLAHE / Adaptive），
        回傳 (best_stage_num, best_bbox, best_prob) 或 (-1, None, 0.0)。
        min_stage：只接受 >= 此值的偵測結果（防止已升階後誤讀低數字）
        """
        crop_img = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        if crop_img.shape[0] == 0 or crop_img.shape[1] == 0:
            return -1, None, 0.0

        gray_img = gray_img_in if gray_img_in is not None else cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

        pad_size = 20
        tracks = []

        # 軌道 1：原始灰階
        img_1 = cv2.copyMakeBorder(gray_img, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT, value=255)
        tracks.append(("Normal", img_1, 1.0, 0.50))

        # 軌道 2：放大 + CLAHE
        scale = 2.0
        resized_img = cv2.resize(gray_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        cl_img = self._clahe.apply(resized_img)
        pad_2 = int(pad_size * scale)
        img_2 = cv2.copyMakeBorder(cl_img, pad_2, pad_2, pad_2, pad_2, cv2.BORDER_CONSTANT, value=255)
        tracks.append(("CLAHE", img_2, scale, 0.45))

        # 軌道 3：適應性二值化
        blur_img = cv2.GaussianBlur(cl_img, (5, 5), 0)
        adaptive_thresh = cv2.adaptiveThreshold(blur_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5)
        img_3 = cv2.copyMakeBorder(adaptive_thresh, pad_2, pad_2, pad_2, pad_2, cv2.BORDER_CONSTANT, value=255)
        tracks.append(("Adaptive", img_3, scale, 0.45))

        candidates = []
        for (_, track_img, current_scale, min_prob) in tracks:
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
                        if SignboardTracker._passes_strict_filter(stage_num, prob, max(0, orig_x), max(0, orig_y), orig_w, orig_h, min_prob, gray_img):
                            candidates.append((prob, stage_num, [
                                [orig_x, orig_y], [orig_x + orig_w, orig_y],
                                [orig_x + orig_w, orig_y + orig_h], [orig_x, orig_y + orig_h]
                            ]))

        if not candidates:
            return -1, None, 0.0

        # 跨軌道 NMS：同 stage 且 IoU > 0.4 視為重複，保留信心最高者
        candidates.sort(key=lambda c: c[0], reverse=True)
        kept = []
        for cand in candidates:
            if not any(self._iou(cand[2], k[2]) > 0.4 and cand[1] == k[1] for k in kept):
                kept.append(cand)

        best_prob, best_stage_num, best_bbox = kept[0]
        return best_stage_num, best_bbox, best_prob

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 7 模板比對
    # ══════════════════════════════════════════════════════════════════════════

    def _rotate_template(self, img, angle):
        """旋轉模板影像（不裁切，自動擴展畫布）"""
        h, w = img.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, float(angle), 1.0)
        cos, sin = abs(M[0, 0]), abs(M[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        return cv2.warpAffine(img, M, (new_w, new_h), borderValue=255)

    def _load_and_prepare_template(self, path):
        """載入模板圖片，裁切數字區域並產生 5 個旋轉版本（-20° ~ +20°）"""
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f">>> [SignboardTracker] 警告：找不到模板圖片 {os.path.basename(path)}，略過此檔")
            return []
        _, digit_mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(digit_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
            pad = 5
            img = img[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad]
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return [(angle, self._rotate_template(img, angle)) for angle in range(-20, 25, 10)]

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
        scales = [1.0, 0.9, 1.2, 0.8, 1.4, 0.7, 1.6, 0.6, 1.8, 0.5, 2.0, 0.4, 2.2, 0.3, 2.5]
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

    # ══════════════════════════════════════════════════════════════════════════
    # 主流程
    # ══════════════════════════════════════════════════════════════════════════

    def detect_stage(self, frame):
        if self.current_stage >= 8:
            return None

        self.frame_counter += 1
        self.no_detect_frames += 1

        # ── 1. KCF 追蹤更新（每幀）────────────────────────────────────────────
        # 在 OCR 跳幀期間持續更新 sign 框位置；含三道過濾：外觀相似度、大幅誤跳、微抖動死區
        if self.tracker is not None:
            ok, tracked_box = self.tracker.update(frame)
            if ok:
                tx, ty, tw, th = [int(v) for v in tracked_box]
                accept = True
                # 外觀相似度：需連續 LOW_SIM_STREAK_MAX 幀低相似才觸發凍結
                if self._appear_snapshot is not None:
                    hist = self._compute_hist(frame, tx, ty, tw, th)
                    if hist is not None:
                        sim = cv2.compareHist(self._appear_snapshot, hist, cv2.HISTCMP_CORREL)
                        if sim < self.APPEAR_SIM_THRESHOLD:
                            self._low_sim_streak += 1
                        else:
                            self._low_sim_streak = 0
                        accept = self._low_sim_streak < self.LOW_SIM_STREAK_MAX
                if accept and self.last_valid_box is not None:
                    old_bx, old_by, old_bw, old_bh = self.last_valid_box
                    dx = abs((tx + tw // 2) - (old_bx + old_bw // 2))
                    dy = abs((ty + th // 2) - (old_by + old_bh // 2))
                    in_fallback = self._fallback_active
                    # 大幅誤跳拒絕（應急框模式下跳過，允許重新定位）
                    if not in_fallback and (dx > self.KCF_MAX_JUMP or dy > self.KCF_MAX_JUMP):
                        accept = False
                    # 微抖動死區
                    elif dx <= self.KCF_JITTER_ZONE and dy <= self.KCF_JITTER_ZONE:
                        accept = False
                if accept:
                    self.last_valid_box = (tx, ty, tw, th)
                    self._update_roi_ema(tx + tw // 2, ty + th // 2)
            else:
                self.tracker = None

        # ── 2. 黃框置中（每幀）────────────────────────────────────────────────
        # 若黃框中心偏離 sign 框超過 RECENTER_MIN，立即修正（上限 RECENTER_MAX）
        if self.last_valid_box is not None and self.current_crop_coords != (0, 0, 0, 0):
            cx1, cy1, cx2, cy2 = self.current_crop_coords
            bx, by, bw, bh = self.last_valid_box
            sign_cx = bx + bw // 2
            sign_cy = by + bh // 2
            dx = abs((cx1 + cx2) // 2 - sign_cx)
            dy = abs((cy1 + cy2) // 2 - sign_cy)
            if (dx > self.RECENTER_MIN or dy > self.RECENTER_MIN) and \
               (dx <= self.RECENTER_MAX and dy <= self.RECENTER_MAX):
                half_w = (cx2 - cx1) // 2
                half_h = (cy2 - cy1) // 2
                self.current_crop_coords = (
                    max(self.roi_x, sign_cx - half_w),
                    max(self.roi_y, sign_cy - half_h),
                    min(self.roi_x + self.roi_w, sign_cx + half_w),
                    min(self.roi_y + self.roi_h, sign_cy + half_h),
                )

        # ── 3. 應急框管理（每幀）──────────────────────────────────────────────
        # 首次達閾值時進入應急框模式（清 tracker）；保持到連續 FALLBACK_EXIT_STREAK 次成功才退出
        if (self.no_detect_frames >= self.NO_DETECT_FALLBACK
                and not self._fallback_active and self.current_stage != 7):
            self._fallback_active = True
            self._fallback_exit_count = 0
            self.tracker = None
        if self._fallback_active and self.current_stage != 7:
            fh, fw = frame.shape[:2]
            cx = self.roi_x + self.roi_w // 2
            cy = self.roi_y + self.roi_h // 2
            hw = self.roi_w * 3 // 4
            hh = self.roi_h * 3 // 4
            self.current_crop_coords = (
                max(0, cx - hw), max(0, cy - hh),
                min(fw, cx + hw), min(fh, cy + hh),
            )

        # ── 4. OCR 跳幀降載────────────────────────────────────────────────────
        interval = self.OCR_FRAME_INTERVAL_STAGE7 if self.current_stage == 7 else self.OCR_FRAME_INTERVAL
        if self.frame_counter % interval != 0:
            return None

        # ── 5. 決定掃描範圍────────────────────────────────────────────────────
        full_roi_x1 = self.roi_x
        full_roi_y1 = self.roi_y
        full_roi_x2 = self.roi_x + self.roi_w
        full_roi_y2 = self.roi_y + self.roi_h

        force_full_scan = (self.frame_counter % self.FULL_SCAN_INTERVAL == 0)
        use_tracking = (self.last_valid_box is not None
                        and self.lost_patience <= self.PATIENCE_THRESHOLD
                        and not force_full_scan)

        if use_tracking and self.current_crop_coords != (0, 0, 0, 0):
            crop_x1, crop_y1, crop_x2, crop_y2 = self.current_crop_coords
        else:
            crop_x1, crop_y1, crop_x2, crop_y2 = full_roi_x1, full_roi_y1, full_roi_x2, full_roi_y2
            if not self._fallback_active:  # 應急模式期間不覆寫 step 3 設好的應急框
                self.current_crop_coords = (crop_x1, crop_y1, crop_x2, crop_y2)

        # ── 6. 主要區域 OCR────────────────────────────────────────────────────
        _got_ocr_hit = False
        best_stage_num, best_bbox, best_prob = self._run_ocr_on_crop(
            frame, crop_x1, crop_y1, crop_x2, crop_y2, min_stage=self.current_stage)

        # ── 7. 備援全域掃描（純位置重錨）──────────────────────────────────────
        # 追蹤框掃不到時補做全域掃描，只更新 last_valid_box，不計入 history_results
        if best_stage_num == -1 and use_tracking:
            backup_stage_num, backup_bbox, _ = self._run_ocr_on_crop(
                frame, full_roi_x1, full_roi_y1, full_roi_x2, full_roi_y2, min_stage=self.current_stage)
            if backup_stage_num != -1 and backup_stage_num >= self.current_stage:
                bx2 = int(backup_bbox[0][0]) + full_roi_x1
                by2 = int(backup_bbox[0][1]) + full_roi_y1
                bw2 = int(backup_bbox[1][0] - backup_bbox[0][0])
                bh2 = int(backup_bbox[2][1] - backup_bbox[1][1])
                pad_b = 15
                self.last_valid_box = (max(0, bx2 - pad_b), max(0, by2 - pad_b),
                                       bw2 + pad_b*2, bh2 + pad_b*2)
                self.lost_patience = 0
                self.no_detect_frames = 0
                _got_ocr_hit = True
                self._init_tracker(frame, self.last_valid_box)
                if self._fallback_active:
                    self._fallback_exit_count += 1
                    if self._fallback_exit_count >= self.FALLBACK_EXIT_STREAK:
                        self._fallback_active = False
                        self._fallback_exit_count = 0
                        self.current_crop_coords = self._make_tracking_zone(*self.last_valid_box)
                    # else：current_crop_coords 由 step 3 維持應急框，不更動
                else:
                    self.current_crop_coords = self._make_tracking_zone(*self.last_valid_box)

        # ── 8. Stage 6→7 模板比對補掃─────────────────────────────────────────
        # 需連續 STAGE7_CONSEC_THRESHOLD 幀 TM 信心 > OCR 6 信心才升階
        if self.current_stage == 6 and best_stage_num != 7:
            tm_stage, tm_bbox = self._match_seven(frame, crop_x1, crop_y1, crop_x2, crop_y2)
            if tm_stage == 7:
                ocr_6_conf = best_prob if best_stage_num == 6 else 0.0
                if ocr_6_conf > self.last_tm_score:
                    self._stage7_consec = 0
                else:
                    self._stage7_consec += 1
                    if self._stage7_consec >= self.STAGE7_CONSEC_THRESHOLD:
                        best_stage_num = 7
                        best_bbox      = tm_bbox
            else:
                self._stage7_consec = 0

        # ── 9. 升階判定────────────────────────────────────────────────────────
        detected_stage = None

        if best_stage_num != -1 and best_stage_num >= self.current_stage:
            orig_x, orig_y = int(best_bbox[0][0]), int(best_bbox[0][1])
            orig_w = int(best_bbox[1][0] - best_bbox[0][0])
            orig_h = int(best_bbox[2][1] - best_bbox[1][1])
            pad_box = 15
            self.last_valid_box = (
                max(0, orig_x + crop_x1 - pad_box),
                max(0, orig_y + crop_y1 - pad_box),
                orig_w + pad_box * 2,
                orig_h + pad_box * 2,
            )
            self.lost_patience = 0
            self.no_detect_frames = 0
            self._init_tracker(frame, self.last_valid_box)
            # 更新外觀快照（遮擋偵測基準）
            bx_s, by_s, bw_s, bh_s = self.last_valid_box
            snap = self._compute_hist(frame, bx_s, by_s, bw_s, bh_s)
            if snap is not None:
                self._appear_snapshot = snap
            # 應急框遲滯：累積連續成功次數，達標才退出
            if self._fallback_active:
                self._fallback_exit_count += 1
                if self._fallback_exit_count >= self.FALLBACK_EXIT_STREAK:
                    self._fallback_active = False
                    self._fallback_exit_count = 0
                    self.current_crop_coords = self._make_tracking_zone(*self.last_valid_box)
                # else：current_crop_coords 由 step 3 維持應急框，不更動
            else:
                self.current_crop_coords = self._make_tracking_zone(*self.last_valid_box)
            self._update_roi_ema(self.last_valid_box[0] + self.last_valid_box[2] // 2,
                                 self.last_valid_box[1] + self.last_valid_box[3] // 2)

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
            self.lost_patience += 1
            if self._fallback_active and not _got_ocr_hit:
                self._fallback_exit_count = 0  # primary + backup 均失敗，連續計數歸零

        # force_full_scan 且本幀無偵測：恢復追蹤結界，避免下一幀沿用全 ROI 掃描框
        # 應急模式期間跳過，讓 step 3 保持應急框
        if force_full_scan and best_stage_num == -1 and self.last_valid_box is not None and not self._fallback_active:
            self.current_crop_coords = self._make_tracking_zone(*self.last_valid_box)

        return detected_stage

    def draw_boxes(self, frame, current_stage):
        """繪製黃色掃描框與綠色 sign 框"""
        cx1, cy1, cx2, cy2 = self.current_crop_coords
        if cx2 > cx1 and cy2 > cy1:
            cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (0, 255, 255), 2)

        if self.last_valid_box is not None:
            bx, by, bw, bh = self.last_valid_box
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
            cv2.putText(frame, f"Sign:{current_stage}",
                        (bx, max(10, by - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
