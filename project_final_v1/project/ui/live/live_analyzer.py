# -*- coding: utf-8 -*-
"""
LiveAnalyzer：即時體驗頁的逐幀推論
==================================
每幀流程：

    YOLO-Pose 人框＋關鍵點 → TargetSelector 選體驗者
      → 視線：用視線管線自己的頭部 YOLO 找出所有頭 → 挑出含體驗者臉部關鍵點的那顆
        → 其餘的頭塗黑 → 整張畫面送進專題 GazeEstimationPipeline.estimate
        （專題人臉偵測在多顆頭時挑面積最小的；塗黑後畫面只剩體驗者一顆頭。
          用同一個頭部模型找頭，管線看到的頭與我們塗掉的頭必定一致。
          不裁成頭部特寫：頭部 YOLO 用施測影片訓練，頭在畫面中太大時信心會掉到門檻下）
      → 指向：體驗者人框外擴 15% 裁切 → HandDetector（MediaPipe）＋ live_pointing 規則
      → 座標加回裁切偏移 → 畫在 display_frame

只唯讀使用專題模組的公開介面（GazeEstimationPipeline、config.yaml 的
gaze_estimation 區塊、model/ 權重），不修改任何專題程式。

推論與繪圖分離：所有模型吃乾淨的 frame，標註一律畫在 display = frame.copy()。
中文文字（「體驗者」、遊戲提示）由 overlay.py 用 QPainter 疊上，cv2 畫不出中文。
"""

import copy
import os
import sys

import cv2
import numpy as np

from .geometry import (expand_box, shift_camera_matrix, gaze_screen_angles, direction_text,
                       pick_target_head)
from .live_pointing import HandDetector, PointingStabilizer, is_pointing_pose, pointing_ray
from .target_person import TargetSelector

PERSON_CONF = 0.5
CROP_RATIO = 0.15
GAZE_EMA = 0.6          # 視線角度平滑（新值權重）
POSE_EVERY = 3          # YOLO-Pose 與頭部 YOLO 每幾幀跑一次（人移動慢，中間沿用；實測各 ~35–50 ms）
# 視線與手部「交錯」執行：偶數幀跑視線、奇數幀跑手部，各自沿用上次結果。
# 實測兩者用執行緒同時跑會在 CPU 上互搶（手部 35 → 68 ms、人臉 47 → 68 ms），反而更慢。
# 只開其中一項時該項每幀都跑。
HAND_MAX_SIDE = 480     # 手部偵測輸入的最長邊（縮圖省 CPU）
HEAD_MIN_SIDE = 40      # 與專題 stage1 相同：小於 40×40 的頭框視為雜訊
MASK_PAD = 0.10         # 塗黑其他人的頭時外擴比例

# BGR，與 theme.py 色票一致
C_TARGET = (255, 163, 77)     # ACCENT 藍
C_OTHER = (120, 110, 100)
C_GAZE = (87, 200, 255)       # WARN 黃
C_HAND = (132, 220, 61)       # OK 綠
C_RAY = (67, 159, 255)        # ROBOT 橘

HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
              (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
              (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


def _absolutize_paths(cfg, project_dir):
    """把設定中相對路徑（*_path、*_task）轉成以專案根目錄為基準的絕對路徑。"""
    if isinstance(cfg, dict):
        out = {}
        for k, v in cfg.items():
            if isinstance(v, str) and (k.endswith("_path") or k.endswith("_task")) \
                    and not os.path.isabs(v):
                out[k] = os.path.join(project_dir, v)
            else:
                out[k] = _absolutize_paths(v, project_dir)
        return out
    return cfg


class LiveAnalyzer:

    def __init__(self, project_dir, gaze_cfg):
        if project_dir not in sys.path:
            sys.path.insert(0, project_dir)
        from ultralytics import YOLO
        from modules.gaze_estimation import GazeEstimationPipeline
        from modules.gaze_estimation.camera_utils import get_default_camera_matrix

        self._camera_matrix_fn = get_default_camera_matrix
        self.pose = YOLO(os.path.join(project_dir, "model", "yolo11n-pose.pt"))
        live_cfg = _absolutize_paths(copy.deepcopy(gaze_cfg or {}), project_dir)
        # config.yaml 的 pitch_offset_deg（-12.5°）是依實驗室攝影機架設角度校正的，
        # 套在筆電 webcam 上會讓視線整體偏「往上看」；只在即時體驗的設定複本裡歸零。
        live_cfg.setdefault("calibration", {})["pitch_offset_deg"] = 0.0
        self.gaze = GazeEstimationPipeline(config=live_cfg)
        self.hands = HandDetector(os.path.join(project_dir, "model", "gaze", "hand_landmarker.task"))
        self.selector = TargetSelector()
        self.stabilizer = PointingStabilizer()
        self._h = self._v = None
        self._frame_no = 0
        self._people_cache = []
        self._heads_cache = []
        self._gaze_cache = None
        self._hands_cache = []

    def close(self):
        self.hands.close()
        self.pose = self.gaze = self.hands = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # ────────────────────────────────────────────
    def _people(self, frame):
        """回傳 [{"box", "kpts", "confs"}]；關鍵點為 COCO 17 點（缺時為 None）。"""
        res = self.pose.predict(frame, conf=PERSON_CONF, classes=[0], verbose=False)
        if not res or res[0].boxes is None or len(res[0].boxes) == 0:
            return []
        boxes = res[0].boxes.xyxy.cpu().numpy()
        kp = res[0].keypoints
        xy = kp.xy.cpu().numpy() if kp is not None else None
        cf = kp.conf.cpu().numpy() if kp is not None and kp.conf is not None else None
        people = []
        for i, b in enumerate(boxes):
            people.append({"box": tuple(map(float, b)),
                           "kpts": xy[i] if xy is not None else None,
                           "confs": cf[i] if cf is not None else None})
        return people

    def _heads(self, frame):
        """用視線管線的頭部 YOLO（唯讀使用）找出畫面中所有頭，門檻與專題 stage1 相同。"""
        fd = self.gaze.face_detector
        res = fd.yolo_model.predict(frame, conf=fd.yolo_conf, imgsz=640, verbose=False)
        if not res or len(res[0].boxes) == 0:
            return []
        return [tuple(map(float, b)) for b in res[0].boxes.xyxy.cpu().numpy()
                if b[2] - b[0] >= HEAD_MIN_SIDE and b[3] - b[1] >= HEAD_MIN_SIDE]

    def _detect_hands(self, frame, crop, ts_ms):
        """在裁切區偵測手部（縮圖後），回傳 [(全畫面像素座標, world landmarks)]。"""
        x1, y1, x2, y2 = crop
        roi = frame[y1:y2, x1:x2]
        scale = min(1.0, HAND_MAX_SIDE / float(max(roi.shape[:2])))
        if scale < 1.0:
            roi = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        # world landmarks 是以手為中心的公尺座標，與縮圖、裁切無關，不需換算
        return [(lm / scale + (x1, y1), world) for lm, world in self.hands.detect(roi, ts_ms)]

    def _estimate_gaze(self, frame, crop_box, mask_boxes=()):
        """crop_box：送進管線的範圍（目前用整張畫面）。mask_boxes：要塗黑的其他人頭部（全畫面座標）。

        專題的人臉偵測在多顆頭時挑「面積最小」的，裁切內只要殘留背景其他人
        的頭就會選錯人（實測：背後同學的後腦被選中 → 抓不到臉部特徵點）。
        """
        x1, y1, x2, y2 = crop_box
        h, w = frame.shape[:2]
        roi = frame[y1:y2, x1:x2]
        if mask_boxes:
            roi = roi.copy()          # 不污染共用的乾淨 frame（手部偵測同時在讀）
            for mx1, my1, mx2, my2 in mask_boxes:
                ax1, ay1 = max(int(mx1) - x1, 0), max(int(my1) - y1, 0)
                ax2, ay2 = min(int(mx2) - x1, x2 - x1), min(int(my2) - y1, y2 - y1)
                if ax2 > ax1 and ay2 > ay1:
                    roi[ay1:ay2, ax1:ax2] = 0
        K = shift_camera_matrix(self._camera_matrix_fn(w, h), x1, y1)
        res = self.gaze.estimate(roi, camera_matrix=K)
        if not res.get("success"):
            return None
        fx1, fy1, fx2, fy2 = res["face_bbox"]
        return {
            "face_box": (fx1 + x1, fy1 + y1, fx2 + x1, fy2 + y1),
            "vector": np.asarray(res["gaze_vector"], dtype=np.float64),
        }

    def _smooth(self, h, v):
        if self._h is None:
            self._h, self._v = h, v
        else:
            self._h = GAZE_EMA * h + (1 - GAZE_EMA) * self._h
            self._v = GAZE_EMA * v + (1 - GAZE_EMA) * self._v
        return self._h, self._v

    # ────────────────────────────────────────────
    def process(self, frame, now, show_gaze=True, show_pointing=True):
        """回傳 (display_frame, state)。frame 應已水平翻轉為鏡像。"""
        h, w = frame.shape[:2]
        display = frame.copy()
        state = {"person_found": False, "face_found": False, "h_deg": None, "v_deg": None,
                 "gaze_text": "—", "pointing": False, "ray": None, "frame_size": (w, h),
                 "person_box": None, "other_boxes": []}

        frame_no = self._frame_no
        self._frame_no += 1
        if frame_no % POSE_EVERY == 0:
            self._people_cache = self._people(frame)
            self._heads_cache = self._heads(frame) if show_gaze else []
        people = self._people_cache
        idx = self.selector.update([p["box"] for p in people], w, h)
        others = [p for i, p in enumerate(people) if i != idx]
        state["other_boxes"] = [p["box"] for p in others]
        for b in state["other_boxes"]:
            cv2.rectangle(display, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), C_OTHER, 1)

        if idx is None:
            self.stabilizer.reset()
            self._h = self._v = None
            self._gaze_cache = None
            self._hands_cache = []
            return display, state

        target = people[idx]
        person = target["box"]
        state["person_found"] = True
        state["person_box"] = person
        cv2.rectangle(display, (int(person[0]), int(person[1])),
                      (int(person[2]), int(person[3])), C_TARGET, 2)
        crop = expand_box(person, CROP_RATIO, w, h)

        both = show_gaze and show_pointing
        if show_gaze:
            if not both or frame_no % 2 == 0 or self._gaze_cache is None:
                self._gaze_cache = self._infer_gaze(frame, target, person)
            self._apply_gaze(self._gaze_cache, display, state)
        else:
            self._h = self._v = None
            self._gaze_cache = None
        if show_pointing:
            if not both or frame_no % 2 == 1:
                self._hands_cache = self._detect_hands(frame, crop, now * 1000.0)
            self._run_pointing(self._hands_cache, frame, display, state)
        else:
            self._hands_cache = []
            self.stabilizer.reset()
        return display, state

    def _infer_gaze(self, frame, target, person):
        """塗黑其他人的頭後跑視線管線。回傳 {"face_box", "vector", "h", "v"} 或 None。"""
        h, w = frame.shape[:2]
        heads = self._heads_cache
        mine = pick_target_head(heads, target["kpts"], target["confs"], person)
        if mine is None:                        # 找不到體驗者的頭 → 不讓管線去挑別人
            self._h = self._v = None
            return None
        masks = [expand_box(hb, MASK_PAD, w, h) for i, hb in enumerate(heads) if i != mine]
        g = self._estimate_gaze(frame, (0, 0, w, h), masks)
        if g is None:
            self._h = self._v = None
            return None
        g["h"], g["v"] = self._smooth(*gaze_screen_angles(g["vector"]))
        return g

    def _apply_gaze(self, g, display, state):
        if g is None:
            return
        hd, vd = g["h"], g["v"]
        state.update(face_found=True, h_deg=hd, v_deg=vd, gaze_text=direction_text(hd, vd))
        fx1, fy1, fx2, fy2 = (int(v) for v in g["face_box"])
        cv2.rectangle(display, (fx1, fy1), (fx2, fy2), C_GAZE, 2)
        cx, cy = (fx1 + fx2) // 2, (fy1 + fy2) // 2
        length = max(80, int((fx2 - fx1) * 1.8))
        vec = g["vector"]
        end = (int(cx + length * vec[0]), int(cy + length * vec[1]))
        cv2.arrowedLine(display, (cx, cy), end, C_GAZE, 4, cv2.LINE_AA, tipLength=0.25)

    def _run_pointing(self, hands, frame, display, state):
        pointing_hand = None
        for lm, world in hands:
            for a, b in HAND_EDGES:
                cv2.line(display, tuple(lm[a].astype(int)), tuple(lm[b].astype(int)),
                         C_HAND, 2, cv2.LINE_AA)
            if pointing_hand is None and is_pointing_pose(lm, world):
                pointing_hand = lm
        stable = self.stabilizer.update(pointing_hand is not None)
        if not stable:
            return
        origin, direction = pointing_ray(pointing_hand)
        h, w = frame.shape[:2]
        end = origin + direction * float(np.hypot(w, h))
        cv2.line(display, tuple(origin.astype(int)), tuple(end.astype(int)), C_RAY, 4, cv2.LINE_AA)
        cv2.circle(display, tuple(origin.astype(int)), 8, C_RAY, -1, cv2.LINE_AA)
        state["pointing"] = True
        state["ray"] = (tuple(origin), tuple(direction))
