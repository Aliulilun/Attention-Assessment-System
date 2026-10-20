# -*- coding: utf-8 -*-
"""
即時體驗專用的指向偵測（展示用簡化版）
======================================
不用專題的 InteractionEngine：它內建兒童／施測者分區、手部歸屬平滑等
施測情境邏輯，不適合單人體驗，而且專題程式不能為此修改。

規則：
- 食指伸直：2D 影像座標下 PIP、DIP 兩個關節夾角都 > 160°
- 中指、無名指、小指彎曲：
  有 3D world landmarks 時 → 各指「指尖到 MCP」3D 長度 < 食指同長度 × 0.75（EXT_RATIO）
  只有 2D 時（單元測試）    → 指尖到手腕距離 < 該指 MCP 到手腕距離 × 1.1
- 連續 3 幀成立才算（PointingStabilizer）

為什麼彎曲改用 3D（2026-10-01 實測）：側著指向左右時，鏡頭看到的是拳頭側面，
彎起的手指在 2D 上反而凸到指根前面，舊的 2D 判斷比值衝到 1.4–1.9 而誤判「沒彎」。
用有標籤的 1622 筆資料（指左/右/上、張開、握拳、比 YA、放鬆）掃門檻，扣掉每段
動作開頭 1 秒切換期後，0.70–0.80 之間結果幾乎一樣（指向 84–100% 成立、張開/握拳/
YA/放鬆 0–1%），取中間值 0.75 保留餘裕。食指伸直仍用 2D：側著指時 2D 很準，
而 3D 直線度反而讓指向成立率下降。
射線：食指 MCP(5) → 指尖(8) 的方向，從指尖往外延伸。
"""

import math

import numpy as np

STRAIGHT_DEG = 160.0
CURL_RATIO = 1.1        # 只有 2D 時的後備規則
EXT_RATIO = 0.75        # 3D：其他手指長度 / 食指長度

# MediaPipe 21 點索引
WRIST = 0
INDEX = (5, 6, 7, 8)
CURLED_FINGERS = ((9, 12), (13, 16), (17, 20))   # (MCP, TIP)


def _angle_deg(a, b, c):
    """以 b 為頂點的夾角。"""
    v1, v2 = np.asarray(a, float) - b, np.asarray(c, float) - b
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    if n < 1e-9:
        return 0.0
    return math.degrees(math.acos(float(np.clip(np.dot(v1, v2) / n, -1.0, 1.0))))


def is_pointing_pose(lm, world=None):
    """lm：21×2 像素座標；world：21×3 MediaPipe world landmarks（可省略）。

    食指伸直、其餘三指彎曲才回傳 True。
    """
    lm = np.asarray(lm, float)
    mcp, pip, dip, tip = (lm[i] for i in INDEX)
    if _angle_deg(mcp, pip, dip) <= STRAIGHT_DEG or _angle_deg(pip, dip, tip) <= STRAIGHT_DEG:
        return False
    if world is not None:
        w = np.asarray(world, float)
        index_len = np.linalg.norm(w[INDEX[3]] - w[INDEX[0]])
        if index_len < 1e-9:
            return False
        return all(np.linalg.norm(w[t] - w[m]) < index_len * EXT_RATIO for m, t in CURLED_FINGERS)
    wrist = lm[WRIST]
    for mcp_i, tip_i in CURLED_FINGERS:
        if np.linalg.norm(lm[tip_i] - wrist) >= np.linalg.norm(lm[mcp_i] - wrist) * CURL_RATIO:
            return False
    return True


def pointing_ray(lm):
    """回傳 (起點=食指尖, 單位方向向量 MCP→指尖)。"""
    lm = np.asarray(lm, float)
    origin = lm[INDEX[3]]
    d = origin - lm[INDEX[0]]
    n = np.linalg.norm(d)
    return origin, (d / n if n > 1e-9 else np.zeros(2))


class PointingStabilizer:
    """連續 frames 幀成立才回傳 True，任一幀不成立就歸零。"""

    def __init__(self, frames=3):
        self.frames = frames
        self._streak = 0

    def update(self, flag):
        self._streak = self._streak + 1 if flag else 0
        return self._streak >= self.frames

    def reset(self):
        self._streak = 0


class HandDetector:
    """MediaPipe HandLandmarker（VIDEO 模式）薄包裝，回傳像素座標關鍵點。"""

    def __init__(self, model_path, num_hands=2):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        self._mp = mp
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_ts = -1

    def detect(self, frame_bgr, ts_ms):
        """回傳 list[(ndarray(21, 2) 像素座標, ndarray(21, 3) world landmarks)]。"""
        import cv2
        # VIDEO 模式要求時間戳嚴格遞增
        ts_ms = max(int(ts_ms), self._last_ts + 1)
        self._last_ts = ts_ms
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, ts_ms)
        hands = []
        worlds = result.hand_world_landmarks or []
        for i, landmarks in enumerate(result.hand_landmarks or []):
            lm = np.array([[p.x * w, p.y * h] for p in landmarks], dtype=np.float64)
            world = (np.array([[p.x, p.y, p.z] for p in worlds[i]], dtype=np.float64)
                     if i < len(worlds) else None)
            hands.append((lm, world))
        return hands

    def close(self):
        try:
            self._landmarker.close()
        except Exception:
            pass
