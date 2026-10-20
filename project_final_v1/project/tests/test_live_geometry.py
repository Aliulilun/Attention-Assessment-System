# -*- coding: utf-8 -*-
"""即時體驗頁幾何工具（ui/live/geometry.py）的單元測試。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ui"))

from live.geometry import (ray_hits_box, expand_box, shift_camera_matrix,
                           gaze_screen_angles, direction_text)


BOX = (100, 100, 200, 200)


def test_ray_hits_box_straight_on():
    assert ray_hits_box((0, 150), (1, 0), BOX)


def test_ray_pointing_away_misses():
    assert not ray_hits_box((0, 150), (-1, 0), BOX)


def test_parallel_ray_outside_slab_misses():
    assert not ray_hits_box((0, 50), (1, 0), BOX)


def test_origin_inside_box_hits():
    assert ray_hits_box((150, 150), (0, 1), BOX)


def test_diagonal_ray_hits_and_misses():
    assert ray_hits_box((0, 0), (1, 1), BOX)
    assert not ray_hits_box((0, 0), (1, -1), BOX)


def test_zero_direction_never_hits():
    assert not ray_hits_box((0, 150), (0, 0), BOX)


def test_expand_box_clamps_to_frame():
    assert expand_box((10, 10, 110, 110), 0.5, 120, 1000) == (0, 0, 120, 160)


def test_expand_box_returns_ints():
    x1, y1, x2, y2 = expand_box((10.4, 10.4, 20.6, 20.6), 0.0, 100, 100)
    assert all(isinstance(v, int) for v in (x1, y1, x2, y2))


def test_shift_camera_matrix_moves_principal_point_only():
    K = np.array([[640.0, 0, 320], [0, 640.0, 240], [0, 0, 1]])
    K2 = shift_camera_matrix(K, 100, 50)
    assert K2[0, 2] == 220 and K2[1, 2] == 190
    assert K2[0, 0] == 640 and K2[1, 1] == 640
    assert K[0, 2] == 320  # 原矩陣不變


def test_gaze_screen_angles_signs():
    # 相機座標：看向鏡頭時 z 為負；x 正 = 畫面右；y 負 = 畫面上
    h, v = gaze_screen_angles(np.array([0.0, 0.0, -1.0]))
    assert abs(h) < 1e-6 and abs(v) < 1e-6
    h, v = gaze_screen_angles(np.array([1.0, 0.0, -1.0]))
    assert abs(h - 45) < 1e-6
    h, v = gaze_screen_angles(np.array([0.0, -1.0, -1.0]))
    assert abs(v - 45) < 1e-6


def test_direction_text():
    assert direction_text(0, 0) == "看著鏡頭"
    assert direction_text(-20, 0) == "往左看"
    assert direction_text(20, 15) == "往右上看"
    assert direction_text(3, -12) == "往下看"
    assert direction_text(None, None) == "—"


def _kpts(points, conf=0.9):
    """points: {索引: (x, y)}，其餘關鍵點信心 0。"""
    k = np.zeros((17, 2)); c = np.zeros(17)
    for i, (x, y) in points.items():
        k[i] = (x, y); c[i] = conf
    return k, c


# ── 從頭部 YOLO 結果挑出體驗者的頭（YOLO-Pose COCO 關鍵點）──
from live.geometry import pick_target_head

USER_HEAD = (572, 406, 814, 717)
BACK_HEAD = (437, 444, 511, 523)
USER_FACE = {0: (690, 560), 1: (720, 520), 2: (660, 520), 3: (760, 540), 5: (900, 720), 6: (480, 720)}


def test_pick_head_containing_face_keypoints():
    k, c = _kpts(USER_FACE)
    assert pick_target_head([BACK_HEAD, USER_HEAD], k, c, (439, 405, 976, 720)) == 1


def test_pick_head_none_when_no_head_matches_keypoints():
    k, c = _kpts(USER_FACE)
    assert pick_target_head([BACK_HEAD], k, c, (439, 405, 976, 720)) is None


def test_pick_head_without_keypoints_uses_largest_head_inside_person_box():
    k, c = _kpts({})
    assert pick_target_head([BACK_HEAD, USER_HEAD], k, c, (439, 380, 976, 720)) == 1


def test_pick_head_empty_list():
    k, c = _kpts(USER_FACE)
    assert pick_target_head([], k, c, (439, 405, 976, 720)) is None
