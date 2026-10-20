# -*- coding: utf-8 -*-
"""即時體驗頁指向規則（ui/live/live_pointing.py）的單元測試。

用合成的 21 點手部關鍵點，不載入 MediaPipe 模型。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ui"))

from live.live_pointing import is_pointing_pose, pointing_ray, PointingStabilizer, EXT_RATIO

FINGER_X = {"index": -30, "middle": -10, "ring": 10, "pinky": 30}
FINGER_BASE = {"index": 5, "middle": 9, "ring": 13, "pinky": 17}


def make_hand(straight):
    """手腕在 (0, 0)，手指朝上（y 負）。straight 為伸直的手指集合。"""
    lm = np.zeros((21, 2), dtype=np.float64)
    lm[1:5] = [(-40, -20), (-50, -40), (-55, -55), (-58, -65)]   # 拇指，規則不看
    for name, base in FINGER_BASE.items():
        x = FINGER_X[name]
        if name in straight:
            ys = [-80, -120, -150, -180]
            xs = [x, x, x, x]
        else:
            ys = [-80, -100, -85, -70]          # PIP 往上、DIP/TIP 折回手掌
            xs = [x, x + 5, x + 8, x + 6]
        lm[base:base + 4] = list(zip(xs, ys))
    return lm


def make_world(straight):
    """3D world landmarks（公尺級尺度不重要，只看長度比）：用 2D 版加 z=0。"""
    return np.hstack([make_hand(straight), np.zeros((21, 1))])


def test_index_extended_others_curled_is_pointing():
    assert is_pointing_pose(make_hand({"index"}))


def test_open_palm_is_not_pointing():
    assert not is_pointing_pose(make_hand({"index", "middle", "ring", "pinky"}))


def test_fist_is_not_pointing():
    assert not is_pointing_pose(make_hand(set()))


def test_two_fingers_is_not_pointing():
    assert not is_pointing_pose(make_hand({"index", "middle"}))


def test_bent_index_is_not_pointing():
    lm = make_hand({"index"})
    lm[7] = (-30 + 25, -150)   # DIP 往旁邊折
    assert not is_pointing_pose(lm)


def test_pointing_ray_from_tip_along_index():
    origin, direction = pointing_ray(make_hand({"index"}))
    assert tuple(origin) == (-30, -180)
    assert np.allclose(direction, (0, -1))


def test_stabilizer_needs_consecutive_frames():
    st = PointingStabilizer(frames=3)
    assert not st.update(True)
    assert not st.update(True)
    assert st.update(True)
    assert not st.update(False)
    assert not st.update(True)


# ── 3D 判斷其他手指彎曲（側著指時 2D 判斷會失效）──
def test_world_landmarks_pointing():
    assert is_pointing_pose(make_hand({"index"}), make_world({"index"}))


def test_world_landmarks_open_palm_rejected():
    all4 = {"index", "middle", "ring", "pinky"}
    assert not is_pointing_pose(make_hand(all4), make_world(all4))


def test_world_landmarks_v_sign_rejected():
    assert not is_pointing_pose(make_hand({"index", "middle"}), make_world({"index", "middle"}))


def test_side_view_uses_3d_curl():
    """側面看握拳：2D 上彎曲手指的指尖凸到指根前面（舊規則誤判為沒彎），3D 長度仍短。"""
    lm2d = make_hand({"index"})
    for base in (9, 13, 17):
        lm2d[base + 3] = lm2d[base] + (0, -40)        # 2D：指尖比指根離手腕更遠
    assert not is_pointing_pose(lm2d)                  # 只有 2D → 舊規則判失敗
    assert is_pointing_pose(lm2d, make_world({"index"}))


def test_bent_index_rejected_even_with_world():
    lm = make_hand({"index"})
    lm[7] = (-30 + 25, -150)
    assert not is_pointing_pose(lm, make_world({"index"}))


def test_ext_ratio_threshold_value():
    assert EXT_RATIO == 0.75
