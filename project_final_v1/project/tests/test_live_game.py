# -*- coding: utf-8 -*-
"""即時體驗頁反應小遊戲（ui/live/reaction_game.py）的單元測試。

時間與隨機來源都由測試注入，結果完全確定。
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ui"))

from live.reaction_game import ReactionGame

FRAME = (1000, 1000)
DT = 0.05   # 20 FPS


class FakeRng:
    """target_side 依序取 sides；等待時間固定 wait 秒。"""

    def __init__(self, sides=("right",), wait=1.0):
        self.sides = list(sides)
        self.i = 0
        self.wait = wait

    def choice(self, _seq):
        side = self.sides[self.i % len(self.sides)]
        self.i += 1
        return side

    def uniform(self, _a, _b):
        return self.wait


def obs(h=0.0, face=True, person=True, pointing=False, ray=None):
    return {"person_found": person, "face_found": face, "h_deg": h if face else None,
            "pointing": pointing, "ray": ray, "frame_size": FRAME}


class Clock:
    def __init__(self):
        self.t = 0.0

    def run(self, game, seconds, o):
        snap = None
        for _ in range(int(round(seconds / DT))):
            self.t += DT
            snap = game.update(self.t, o)
        return snap

    def step(self, game, o):
        self.t += DT
        return game.update(self.t, o)


def calibrated(sides=("right",), baseline=0.0):
    """開局並完成校正，停在 WAIT。"""
    game = ReactionGame(rng=FakeRng(sides))
    clk = Clock()
    game.start(clk.t)
    clk.run(game, 2.05, obs(h=baseline))
    return game, clk


def to_target(game, clk, o=None):
    """從 WAIT 等到目標出現，回傳出現時的快照。"""
    o = o or obs()
    for _ in range(200):
        snap = clk.step(game, o)
        if snap["phase"] == "TARGET":
            return snap
    raise AssertionError("目標沒有出現")


RIGHT_RAY = ((500.0, 500.0), (1.0, 0.0))
LEFT_RAY = ((500.0, 500.0), (-1.0, 0.0))


# ── 校正 ──
def test_starts_idle():
    assert ReactionGame().update(0.0, obs())["phase"] == "IDLE"


def test_calibration_uses_median_baseline():
    game = ReactionGame(rng=FakeRng())
    game.start(0.0)
    t = 0.0
    values = [5.0] * 36 + [40.0] * 5     # 少數離群值不影響中位數
    for v in values:
        t += DT
        game.update(t, obs(h=v))
    assert game.baseline == pytest.approx(5.0)
    assert game.update(t + DT, obs(h=5.0))["phase"] == "WAIT"


def test_calibration_extends_when_too_few_face_frames():
    game = ReactionGame(rng=FakeRng())
    clk = Clock()
    game.start(clk.t)
    snap = clk.run(game, 2.5, obs(face=False))
    assert snap["phase"] == "CALIBRATING"
    assert "看著鏡頭" in snap["message"]
    snap = clk.run(game, 0.6, obs(h=0.0))   # 補足 12 幀有臉
    assert snap["phase"] == "WAIT"


# ── 第一關：看 ──
def test_look_hit_records_time_from_first_matching_frame():
    game, clk = calibrated(sides=("right",))
    shown = to_target(game, clk)
    assert shown["target_side"] == "right" and shown["stage"] == "look"
    clk.run(game, 0.5, obs(h=0.0))
    first = clk.t + DT
    snap = clk.run(game, 3 * DT, obs(h=25.0))
    assert snap["phase"] == "RETURN"
    assert snap["results"]["look"][0] == pytest.approx(first - (clk.t - 0.5 - 3 * DT), abs=1e-6) \
        or snap["results"]["look"][0] == pytest.approx(0.55, abs=DT + 1e-6)


def test_look_hit_uses_baseline():
    game, clk = calibrated(sides=("right",), baseline=10.0)
    to_target(game, clk, obs(h=10.0))
    snap = clk.run(game, 3 * DT, obs(h=20.0))       # 只偏 10° → 不算
    assert snap["phase"] == "TARGET"
    snap = clk.run(game, 3 * DT, obs(h=30.0))       # 偏 20° → 命中
    assert snap["phase"] == "RETURN"


def test_wrong_side_does_not_stop_timer():
    game, clk = calibrated(sides=("right",))
    to_target(game, clk)
    snap = clk.run(game, 1.0, obs(h=-30.0))
    assert snap["phase"] == "TARGET"


def test_look_hit_needs_three_consecutive_frames():
    game, clk = calibrated(sides=("left",))
    to_target(game, clk)
    clk.run(game, 2 * DT, obs(h=-25.0))
    snap = clk.step(game, obs(h=0.0))
    assert snap["phase"] == "TARGET"


def test_look_timeout_records_miss():
    game, clk = calibrated()
    to_target(game, clk)
    snap = clk.run(game, 3.1, obs(h=-30.0))    # 一直看錯邊 → 超時
    assert snap["phase"] == "RETURN"
    assert snap["results"]["look"] == [None]


def test_missing_face_pauses_timer():
    game, clk = calibrated()
    to_target(game, clk)
    clk.run(game, 1.0, obs(face=False))
    snap = clk.run(game, 2.5, obs(h=0.0))        # 有效時間只過 2.5 秒 → 尚未超時
    assert snap["phase"] == "TARGET"
    snap = clk.run(game, 0.6, obs(h=0.0))
    assert snap["results"]["look"] == [None]


def test_missing_face_message_and_time_subtracted():
    game, clk = calibrated(sides=("right",))
    to_target(game, clk)
    snap = clk.run(game, 1.0, obs(face=False))
    assert "找不到臉" in snap["message"]
    snap = clk.run(game, 3 * DT, obs(h=25.0))
    assert snap["results"]["look"][0] == pytest.approx(DT, abs=1e-6)


def test_return_to_center_required_before_next_round():
    game, clk = calibrated()
    to_target(game, clk)
    snap = clk.run(game, 3 * DT, obs(h=25.0))
    assert snap["phase"] == "RETURN" and "看回鏡頭" in snap["message"]
    snap = clk.run(game, 1.0, obs(h=25.0))
    assert snap["phase"] == "RETURN"
    snap = clk.run(game, 3 * DT, obs(h=2.0))
    assert snap["phase"] == "WAIT" and snap["round"] == 2


def play_look_stage(game, clk):
    for _ in range(5):
        to_target(game, clk)
        side = game.update(clk.t, obs())["target_side"]
        clk.run(game, 3 * DT, obs(h=25.0 if side == "right" else -25.0))
        snap = clk.run(game, 3 * DT, obs(h=0.0))
    return snap


def test_after_five_look_rounds_goes_to_intro_then_point_stage():
    game, clk = calibrated(sides=("right", "left"))
    snap = play_look_stage(game, clk)
    assert snap["phase"] == "INTRO" and "第二關" in snap["message"]
    assert len(snap["results"]["look"]) == 5
    snap = clk.run(game, 3.05, obs())
    assert snap["phase"] == "WAIT" and snap["stage"] == "point" and snap["round"] == 1


# ── 第二關：指 ──
def at_point_stage(sides=("right",)):
    game, clk = calibrated(sides=sides)
    play_look_stage(game, clk)
    clk.run(game, 3.05, obs())
    return game, clk


def test_point_target_box_on_correct_side():
    game, clk = at_point_stage(sides=("right",))
    snap = to_target(game, clk)
    x1, y1, x2, y2 = snap["target_box"]
    assert x1 == pytest.approx(700) and x2 == pytest.approx(1000)
    assert y1 == pytest.approx(200) and y2 == pytest.approx(800)


def test_point_hit_with_ray_into_box():
    game, clk = at_point_stage(sides=("right",))
    to_target(game, clk)
    clk.run(game, 1.0, obs(pointing=False))
    snap = clk.run(game, 3 * DT, obs(pointing=True, ray=RIGHT_RAY))
    assert snap["phase"] == "RETURN"
    assert snap["results"]["point"][0] == pytest.approx(1.0 + DT, abs=1e-6)


def test_point_wrong_side_does_not_stop_timer():
    game, clk = at_point_stage(sides=("right",))
    to_target(game, clk)
    snap = clk.run(game, 1.0, obs(pointing=True, ray=LEFT_RAY))
    assert snap["phase"] == "TARGET"


def test_point_timeout_is_five_seconds():
    game, clk = at_point_stage()
    to_target(game, clk)
    snap = clk.run(game, 4.5, obs())
    assert snap["phase"] == "TARGET"
    snap = clk.run(game, 0.6, obs())
    assert snap["results"]["point"] == [None]


def test_point_missing_hand_does_not_pause():
    game, clk = at_point_stage()
    to_target(game, clk)
    snap = clk.run(game, 5.1, obs(face=False, person=True))
    assert snap["results"]["point"] == [None]


def test_point_missing_person_pauses():
    game, clk = at_point_stage()
    to_target(game, clk)
    snap = clk.run(game, 2.0, obs(person=False, face=False))
    assert "回到鏡頭前" in snap["message"]
    snap = clk.run(game, 4.5, obs())
    assert snap["phase"] == "TARGET"


def test_point_return_requires_hand_down():
    game, clk = at_point_stage(sides=("right",))
    to_target(game, clk)
    clk.run(game, 3 * DT, obs(pointing=True, ray=RIGHT_RAY))
    snap = clk.run(game, 1.0, obs(pointing=True, ray=RIGHT_RAY))
    assert snap["phase"] == "RETURN" and "把手放下" in snap["message"]
    snap = clk.run(game, 3 * DT, obs(pointing=False))
    assert snap["phase"] == "WAIT" and snap["round"] == 2


def play_point_stage(game, clk, hit=True):
    for _ in range(5):
        to_target(game, clk)
        side = game.update(clk.t, obs())["target_side"]
        if hit:
            clk.run(game, 3 * DT, obs(pointing=True, ray=RIGHT_RAY if side == "right" else LEFT_RAY))
        else:
            clk.run(game, 5.1, obs())
        snap = clk.run(game, 3 * DT, obs(pointing=False))
    return snap


def test_full_game_reaches_result_with_summary_and_best():
    game, clk = at_point_stage(sides=("right", "left"))
    snap = play_point_stage(game, clk)
    assert snap["phase"] == "RESULT"
    assert len(snap["results"]["point"]) == 5
    s = snap["summary"]
    assert s["look"]["avg"] is not None and s["point"]["fastest"] is not None
    assert game.best["look"] == pytest.approx(s["look"]["avg"])
    assert game.best["point"] == pytest.approx(s["point"]["avg"])


def test_average_excludes_misses_and_all_miss_gives_none():
    game, clk = at_point_stage()
    snap = play_point_stage(game, clk, hit=False)
    assert snap["summary"]["point"]["avg"] is None
    assert game.best["point"] is None
    assert snap["summary"]["look"]["misses"] == 0
    assert snap["summary"]["point"]["misses"] == 5


def test_best_keeps_lowest_average_across_games():
    game, clk = at_point_stage()
    play_point_stage(game, clk)
    first_best = game.best["look"]
    game.start(clk.t)
    clk.run(game, 2.05, obs())
    for _ in range(5):          # 第二局看關每回合都慢 1 秒
        to_target(game, clk)
        side = game.update(clk.t, obs())["target_side"]
        clk.run(game, 1.0, obs())
        clk.run(game, 3 * DT, obs(h=25.0 if side == "right" else -25.0))
        clk.run(game, 3 * DT, obs(h=0.0))
    assert game.best["look"] == pytest.approx(first_best)


def test_cancel_returns_to_idle():
    game, clk = calibrated()
    to_target(game, clk)
    game.cancel()
    assert game.update(clk.t + DT, obs())["phase"] == "IDLE"


def test_start_from_result_restarts_calibration():
    game, clk = at_point_stage()
    play_point_stage(game, clk)
    game.start(clk.t)
    snap = clk.step(game, obs())
    assert snap["phase"] == "CALIBRATING"
    assert snap["results"] == {"look": [], "point": []}
