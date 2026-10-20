# -*- coding: utf-8 -*-
"""
LiveWorker：即時體驗頁的攝影機執行緒
====================================
攝影機讀取、模型推論、小遊戲狀態機全部在這條執行緒跑，GUI 不會卡住。
小遊戲物件由頁面持有（「本場最佳」才能跨多次開關鏡頭保留），但只有
這條執行緒會呼叫它；頁面透過 start_game() / cancel_game() 送旗標。
"""

import os
import threading
import time

import cv2
import yaml

from qt_compat import QThread, Signal, QImage
from workers import bgr_to_qimage
from .live_analyzer import LiveAnalyzer
from .overlay import paint_overlay

CAMERA_INDEX = 0
MAX_READ_FAILS = 30
ACTIVE_PHASES = ("CALIBRATING", "WAIT", "TARGET", "RETURN", "INTRO")


class LiveWorker(QThread):

    frame_ready = Signal(QImage, dict)
    status = Signal(str)
    failed = Signal(str)

    def __init__(self, project_dir, game, parent=None):
        super().__init__(parent)
        self.project_dir = project_dir
        self.game = game
        self._stop = False
        self._lock = threading.Lock()
        self._show_gaze = True
        self._show_pointing = True
        self._game_cmd = None

    # ── 由 GUI 執行緒呼叫 ──
    def request_stop(self):
        self._stop = True

    def set_options(self, gaze, pointing):
        with self._lock:
            self._show_gaze, self._show_pointing = bool(gaze), bool(pointing)

    def start_game(self):
        with self._lock:
            self._game_cmd = "start"

    def cancel_game(self):
        with self._lock:
            self._game_cmd = "cancel"

    # ── 執行緒本體 ──
    def run(self):
        cap = None
        analyzer = None
        try:
            self.status.emit("開啟攝影機…")
            cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
            if not cap.isOpened():
                self.failed.emit("找不到攝影機，請確認已接上，且沒有被其他程式（視訊會議等）占用。")
                return
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            self.status.emit("載入 AI 模型中（約 15 秒）…")
            with open(os.path.join(self.project_dir, "config.yaml"), encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            analyzer = LiveAnalyzer(self.project_dir, cfg.get("gaze_estimation", {}))
            if self._stop:
                return
            self.status.emit("")
            self._loop(cap, analyzer)
        except Exception as exc:   # 模型缺檔、CUDA 錯誤等：回報給頁面，不讓程式閃退
            self.failed.emit("即時分析發生錯誤：%s" % exc)
        finally:
            if cap is not None:
                cap.release()
            if analyzer is not None:
                analyzer.close()

    def _loop(self, cap, analyzer):
        fails = 0
        fps, last = 0.0, time.monotonic()
        while not self._stop:
            ok, frame = cap.read()
            if not ok or frame is None:
                fails += 1
                if fails >= MAX_READ_FAILS:
                    self.failed.emit("攝影機訊號中斷，請檢查連接線後重新開啟鏡頭。")
                    return
                time.sleep(0.02)
                continue
            fails = 0
            frame = cv2.flip(frame, 1)       # 鏡像：左右方向符合體驗者直覺
            now = time.monotonic()

            with self._lock:
                show_gaze, show_pointing = self._show_gaze, self._show_pointing
                cmd, self._game_cmd = self._game_cmd, None
            if cmd == "start":
                self.game.start(now)
            elif cmd == "cancel":
                self.game.cancel()
            if self.game.phase in ACTIVE_PHASES:
                show_gaze = show_pointing = True   # 遊戲進行中兩種偵測都要開

            display, state = analyzer.process(frame, now, show_gaze, show_pointing)
            state["game"] = self.game.update(now, state)

            dt = now - last
            last = now
            if dt > 0:
                fps = 1.0 / dt if fps == 0 else fps * 0.9 + (1.0 / dt) * 0.1
            state["fps"] = fps

            qimg = bgr_to_qimage(display)
            paint_overlay(qimg, state, frame.shape[1])
            state.pop("ray", None)            # numpy 物件不必跨執行緒傳
            self.frame_ready.emit(qimg, state)
