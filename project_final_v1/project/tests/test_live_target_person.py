# -*- coding: utf-8 -*-
"""即時體驗頁體驗者選定（ui/live/target_person.py）的單元測試。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ui"))

from live.target_person import score_box, TargetSelector

W, H = 1000, 1000
CENTER_BIG = (300, 200, 700, 1000)     # 置中大框
EDGE_BIG = (0, 100, 400, 1000)         # 貼左邊的大框
CENTER_SMALL = (450, 400, 550, 600)    # 置中小框（遠處路人）
RIGHT_MID = (600, 200, 1000, 1000)


def test_score_prefers_big_and_centered():
    s_center_big = score_box(CENTER_BIG, W, H)
    assert s_center_big > score_box(EDGE_BIG, W, H)
    assert s_center_big > score_box(CENTER_SMALL, W, H)


def test_single_person_selected_immediately():
    sel = TargetSelector()
    assert sel.update([CENTER_BIG], W, H) == 0


def test_no_people_returns_none():
    sel = TargetSelector()
    assert sel.update([], W, H) is None


def test_picks_best_among_many():
    sel = TargetSelector()
    assert sel.update([EDGE_BIG, CENTER_SMALL, CENTER_BIG], W, H) == 2


def test_tracks_target_when_order_changes():
    sel = TargetSelector()
    sel.update([CENTER_BIG, CENTER_SMALL], W, H)
    assert sel.update([CENTER_SMALL, CENTER_BIG], W, H) == 1


def test_challenger_needs_consecutive_frames():
    sel = TargetSelector(switch_frames=5)
    assert sel.update([RIGHT_MID], W, H) == 0
    for _ in range(4):
        assert sel.update([RIGHT_MID, CENTER_BIG], W, H) == 0
    assert sel.update([RIGHT_MID, CENTER_BIG], W, H) == 1


def test_challenger_streak_resets_when_interrupted():
    sel = TargetSelector(switch_frames=5)
    sel.update([RIGHT_MID], W, H)
    for _ in range(4):
        sel.update([RIGHT_MID, CENTER_BIG], W, H)
    sel.update([RIGHT_MID], W, H)          # 挑戰者消失一幀 → 重新累計
    for _ in range(4):
        assert sel.update([RIGHT_MID, CENTER_BIG], W, H) == 0


def test_lost_target_waits_before_reselecting():
    sel = TargetSelector(lost_frames=10)
    sel.update([CENTER_BIG], W, H)
    for _ in range(10):
        assert sel.update([EDGE_BIG], W, H) is None
    assert sel.update([EDGE_BIG], W, H) == 0


def test_target_returns_within_lost_window():
    sel = TargetSelector(lost_frames=10)
    sel.update([CENTER_BIG], W, H)
    for _ in range(5):
        sel.update([], W, H)
    assert sel.update([EDGE_BIG, CENTER_BIG], W, H) == 1


def test_reset_forgets_target():
    sel = TargetSelector(lost_frames=10)
    sel.update([CENTER_BIG], W, H)
    sel.reset()
    assert sel.update([EDGE_BIG], W, H) == 0
