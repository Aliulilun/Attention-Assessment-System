# -*- coding: utf-8 -*-
"""
即時體驗頁的幾何工具（純 numpy，不依賴 Qt／模型）
==================================================
- 2D 射線與矩形相交（指向小遊戲判定）
- 框外擴與夾邊（裁切體驗者）
- 相機內參主點位移（裁切後送視線管線，幾何維持與全畫面一致）
- 視線向量 → 畫面上的左右／上下角度與白話方向

刻意不 import hurry/main.py 的同名函式：那支檔案載入時會拉進整條分析管線，
而且即時體驗與專題程式必須隔離。
"""

import math

import numpy as np


def ray_hits_box(origin, direction, box):
    """2D Ray-AABB（slab 法）。只算射線前方（t ≥ 0）；起點在框內視為命中。"""
    ox, oy = float(origin[0]), float(origin[1])
    dx, dy = float(direction[0]), float(direction[1])
    if dx == 0.0 and dy == 0.0:
        return False
    x1, y1, x2, y2 = box
    t_min, t_max = 0.0, math.inf
    for o, d, lo, hi in ((ox, dx, x1, x2), (oy, dy, y1, y2)):
        if d == 0.0:
            if o < lo or o > hi:
                return False
            continue
        t1, t2 = (lo - o) / d, (hi - o) / d
        if t1 > t2:
            t1, t2 = t2, t1
        t_min, t_max = max(t_min, t1), min(t_max, t2)
        if t_min > t_max:
            return False
    return True


def expand_box(box, ratio, frame_w, frame_h):
    """框四邊各外擴 寬/高 × ratio，夾在畫面內，回傳整數座標。"""
    x1, y1, x2, y2 = box
    pw, ph = (x2 - x1) * ratio, (y2 - y1) * ratio
    return (int(max(0, math.floor(x1 - pw))), int(max(0, math.floor(y1 - ph))),
            int(min(frame_w, math.ceil(x2 + pw))), int(min(frame_h, math.ceil(y2 + ph))))


def shift_camera_matrix(K, dx, dy):
    """裁切畫面後的相機內參：焦距不變，主點減去裁切起點。回傳新矩陣。"""
    K2 = np.array(K, dtype=np.float64, copy=True)
    K2[0, 2] -= dx
    K2[1, 2] -= dy
    return K2


def gaze_screen_angles(gaze_vector):
    """相機座標視線向量 → (水平角, 垂直角)，單位度。

    水平角：正 = 畫面右；垂直角：正 = 畫面上。
    直接用向量分量判斷方向，不必理會 ETH-XGaze 的 yaw 正負號慣例。
    """
    gx, gy, gz = (float(v) for v in gaze_vector[:3])
    depth = -gz if gz < 0 else max(abs(gz), 1e-6)
    return math.degrees(math.atan2(gx, depth)), math.degrees(math.atan2(-gy, depth))


def direction_text(h_deg, v_deg, dead=8.0):
    """角度 → 白話方向，例如「往左上看」。死區內視為看著鏡頭。"""
    if h_deg is None or v_deg is None:
        return "—"
    horiz = "右" if h_deg > dead else ("左" if h_deg < -dead else "")
    vert = "上" if v_deg > dead else ("下" if v_deg < -dead else "")
    if not horiz and not vert:
        return "看著鏡頭"
    return "往%s%s看" % (horiz, vert)


# YOLO-Pose（COCO 17 點）：0 鼻、1 左眼、2 右眼、3 左耳、4 右耳
FACE_KPTS = (0, 1, 2, 3, 4)
KPT_CONF = 0.3


def pick_target_head(heads, kpts, confs, person_box):
    """從頭部 YOLO 的所有頭框中挑出體驗者的那一顆，回傳索引；找不到回傳 None。

    - 有臉部關鍵點：挑「包含最多臉部關鍵點」的頭框（至少 1 點）
    - 沒有臉部關鍵點（例如背對）：挑中心落在體驗者人框內、面積最大的頭框
    """
    if not heads:
        return None
    pts = []
    if kpts is not None and confs is not None:
        pts = [kpts[i] for i in FACE_KPTS if confs[i] >= KPT_CONF]
    if pts:
        counts = [sum(1 for x, y in pts if b[0] <= x <= b[2] and b[1] <= y <= b[3]) for b in heads]
        best = max(range(len(heads)), key=counts.__getitem__)
        return best if counts[best] > 0 else None
    px1, py1, px2, py2 = person_box
    inside = [i for i, b in enumerate(heads)
              if px1 <= (b[0] + b[2]) / 2.0 <= px2 and py1 <= (b[1] + b[3]) / 2.0 <= py2]
    if not inside:
        return None
    return max(inside, key=lambda i: (heads[i][2] - heads[i][0]) * (heads[i][3] - heads[i][1]))
