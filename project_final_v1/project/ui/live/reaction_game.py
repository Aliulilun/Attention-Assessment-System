# -*- coding: utf-8 -*-
"""
反應小遊戲：「先看再指」
========================
第一關「看」：星星出現 → 視線轉向星星那一側 → 停錶
第二關「指」：星星出現 → 食指射線指中星星 → 停錶
每關 5 回合，只記錄反應秒數（不算 TB / TH）。

狀態機（時間一律由呼叫端傳入，隨機來源可注入，方便測試）：

    IDLE ─start→ CALIBRATING ─2 秒→ WAIT ─1~3 秒→ TARGET
    TARGET ─命中連 3 幀→ RETURN（記秒數）／超時→ RETURN（記錯過）
    RETURN ─回中立連 3 幀→ WAIT／INTRO（換第二關）／RESULT
    INTRO ─3 秒→ WAIT
    任意 ─cancel→ IDLE

觀測 obs（每幀由 LiveAnalyzer 提供）：
    person_found, face_found, h_deg（水平視線角，右正）, pointing, ray, frame_size
"""

import random
import statistics

from .geometry import ray_hits_box

CALIB_SEC = 2.0
CALIB_MIN_FRAMES = 10
WAIT_MIN, WAIT_MAX = 1.0, 3.0
LOOK_THRESH = 15.0
CENTER_THRESH = 8.0
HOLD_FRAMES = 3
LOOK_TIMEOUT = 3.0
POINT_TIMEOUT = 5.0
INTRO_SEC = 3.0
TARGET_W, TARGET_H = 0.30, 0.60

STAGES = ("look", "point")

MSG = {
    "calib": "請看著鏡頭，校正中…",
    "calib_more": "請看著鏡頭（還需要再看一下）",
    "wait": "準備…",
    "look_target": "你看！轉頭看向星星那一邊",
    "point_target": "指給我看！用食指指向星星",
    "look_return": "很好！請看回鏡頭",
    "point_return": "很好！請把手放下",
    "no_face": "找不到臉，請靠近鏡頭",
    "no_person": "請回到鏡頭前",
    "intro": "第二關：用手指指星星！",
    "result": "我們的系統用同樣的方法，測量孩子聽到『你看』之後，"
              "多快把視線轉向目標、會不會主動用手指東西給大人看。",
}


def summarize(times):
    hits = [t for t in times if t is not None]
    return {
        "avg": statistics.mean(hits) if hits else None,
        "fastest": min(hits) if hits else None,
        "misses": len(times) - len(hits),
    }


def target_box_for(side, frame_size):
    if frame_size is None:
        return None
    w, h = frame_size
    bw, bh = w * TARGET_W, h * TARGET_H
    y1 = (h - bh) / 2.0
    x1 = 0.0 if side == "left" else w - bw
    return (x1, y1, x1 + bw, y1 + bh)


class ReactionGame:

    def __init__(self, rng=None, rounds=5):
        self.rng = rng or random.Random()
        self.rounds = rounds
        self.best = {"look": None, "point": None}
        self.cancel()

    # ── 控制 ──
    def cancel(self):
        self.phase = "IDLE"
        self.stage = "look"
        self.round_idx = 0
        self.results = {"look": [], "point": []}
        self.baseline = None
        self._calib = []
        self._phase_start = 0.0
        self._wait = 0.0
        self.target_side = None
        self._shown_at = 0.0
        self._paused = 0.0
        self._pause_from = None
        self._streak = 0
        self._first_hit = None
        self._message = ""
        self._elapsed = 0.0

    def start(self, now):
        best = self.best
        self.cancel()
        self.best = best
        self._enter("CALIBRATING", now, MSG["calib"])

    # ── 內部 ──
    def _enter(self, phase, now, message=""):
        self.phase = phase
        self._phase_start = now
        self._streak = 0
        self._message = message

    def _enter_wait(self, now):
        self._wait = self.rng.uniform(WAIT_MIN, WAIT_MAX)
        self._enter("WAIT", now, MSG["wait"])

    def _finish_stage(self):
        avg = summarize(self.results[self.stage])["avg"]
        if avg is not None and (self.best[self.stage] is None or avg < self.best[self.stage]):
            self.best[self.stage] = avg

    def _is_hit(self, obs):
        if self.stage == "look":
            h = obs.get("h_deg")
            if not obs.get("face_found") or h is None:
                return False
            d = h - self.baseline
            return d > LOOK_THRESH if self.target_side == "right" else d < -LOOK_THRESH
        ray = obs.get("ray")
        box = target_box_for(self.target_side, obs.get("frame_size"))
        return bool(obs.get("pointing") and ray is not None and box is not None
                    and ray_hits_box(ray[0], ray[1], box))

    def _is_neutral(self, obs):
        if self.stage == "look":
            h = obs.get("h_deg")
            return bool(obs.get("face_found") and h is not None
                        and abs(h - self.baseline) < CENTER_THRESH)
        return not obs.get("pointing")

    def _absent(self, obs):
        if self.stage == "look":
            return not obs.get("face_found")
        return not obs.get("person_found")

    # ── 每幀 ──
    def update(self, now, obs):
        handler = getattr(self, "_update_" + self.phase.lower(), None)
        if handler is not None:
            handler(now, obs)
        return self.snapshot(obs)

    def _update_calibrating(self, now, obs):
        if obs.get("face_found") and obs.get("h_deg") is not None:
            self._calib.append(float(obs["h_deg"]))
        if now - self._phase_start < CALIB_SEC:
            return
        if len(self._calib) >= CALIB_MIN_FRAMES:
            self.baseline = statistics.median(self._calib)
            self._enter_wait(now)
        else:
            self._message = MSG["calib_more"]

    def _update_wait(self, now, _obs):
        if now - self._phase_start >= self._wait:
            self.target_side = self.rng.choice(("left", "right"))
            self._shown_at = now
            self._paused = 0.0
            self._pause_from = None
            self._first_hit = None
            self._elapsed = 0.0
            self._enter("TARGET", now, MSG[self.stage + "_target"])

    def _update_target(self, now, obs):
        if self._absent(obs):
            if self._pause_from is None:
                self._pause_from = now
            self._streak = 0
            self._message = MSG["no_face" if self.stage == "look" else "no_person"]
            return
        if self._pause_from is not None:
            self._paused += now - self._pause_from
            self._pause_from = None
            self._message = MSG[self.stage + "_target"]

        self._elapsed = now - self._shown_at - self._paused
        if self._is_hit(obs):
            if self._streak == 0:
                self._first_hit = self._elapsed
            self._streak += 1
            if self._streak >= HOLD_FRAMES:
                self._record(now, self._first_hit)
            return
        self._streak = 0
        timeout = LOOK_TIMEOUT if self.stage == "look" else POINT_TIMEOUT
        if self._elapsed >= timeout:
            self._record(now, None)

    def _record(self, now, value):
        self.results[self.stage].append(value)
        self._enter("RETURN", now, MSG[self.stage + "_return"])

    def _update_return(self, now, obs):
        self._streak = self._streak + 1 if self._is_neutral(obs) else 0
        if self._streak < HOLD_FRAMES:
            return
        self.round_idx += 1
        if self.round_idx < self.rounds:
            self._enter_wait(now)
            return
        self._finish_stage()
        if self.stage == "look":
            self.stage = "point"
            self.round_idx = 0
            self._enter("INTRO", now, MSG["intro"])
        else:
            self._enter("RESULT", now, MSG["result"])

    def _update_intro(self, now, _obs):
        if now - self._phase_start >= INTRO_SEC:
            self._enter_wait(now)

    # ── 快照 ──
    def snapshot(self, obs=None):
        in_target = self.phase == "TARGET"
        frame_size = (obs or {}).get("frame_size")
        return {
            "phase": self.phase,
            "stage": self.stage,
            "round": min(self.round_idx + 1, self.rounds),
            "rounds": self.rounds,
            "target_side": self.target_side if in_target else None,
            "target_box": target_box_for(self.target_side, frame_size) if in_target else None,
            "results": {k: list(v) for k, v in self.results.items()},
            "message": self._message,
            "elapsed": self._elapsed if in_target else None,
            "summary": ({s: summarize(self.results[s]) for s in STAGES}
                        if self.phase == "RESULT" else None),
            "best": dict(self.best),
        }
