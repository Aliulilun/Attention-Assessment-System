# -*- coding: utf-8 -*-
"""
背景執行緒 Worker
==================
分析一支影片要跑好幾分鐘的 GPU 推論。若在主執行緒做，Qt 事件迴圈會被
卡住、視窗變成「沒有回應」——報告現場最不想看到的畫面。因此整條管線
放進 QThread，透過 signal 把畫面與統計送回 GUI 執行緒。

三個關鍵設計：
1. **stdout 轉接**：pipeline 與 modules/ 內大量使用 print()，與其逐一改寫，
   不如在 worker 執行期間把 sys.stdout 導向 signal，讓所有既有訊息自動
   出現在介面的主控台。只有一條 worker 執行緒，不會有交錯問題。
2. **畫面節流**：1080p 影格每秒 30 張全部丟給 GUI 會讓介面追不上而積壓
   記憶體。畫面以固定上限（預設 20 fps）取樣送出，統計數值則每幀都送
   （體積小），所以數字不會漏跳。
3. **中止旗標**：should_stop() 每幀檢查一次。中止後仍走 finally，
   已分析的部分照樣寫出報告與影片，不會白跑。
"""

import os
import sys
import time
import traceback

import cv2
import numpy as np

from qt_compat import QThread, Signal, QImage
import pipeline_api


class _StreamRelay:
    """把 print() 的輸出轉成 Qt signal，同時保留原本的終端機輸出。"""

    def __init__(self, emit_fn, original):
        self._emit = emit_fn
        self._original = original
        self._buffer = ""

    def write(self, text):
        if self._original is not None:
            try:
                self._original.write(text)
            except Exception:
                pass
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip():
                try:
                    self._emit(line.rstrip())
                except Exception:
                    pass

    def flush(self):
        if self._original is not None:
            try:
                self._original.flush()
            except Exception:
                pass


def bgr_to_qimage(frame_bgr, max_width=960):
    """OpenCV BGR ndarray → QImage（在 worker 執行緒先縮圖，減輕 GUI 負擔）。

    QImage 建立時會參照原本的記憶體，ndarray 被回收就會變成花畫面，
    因此最後一定要 .copy() 讓 Qt 自己持有一份。
    """
    h, w = frame_bgr.shape[:2]
    if w > max_width:
        scale = max_width / float(w)
        frame_bgr = cv2.resize(frame_bgr, (max_width, int(h * scale)),
                               interpolation=cv2.INTER_AREA)
        h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)
    return QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888).copy()


class AnalysisWorker(QThread):
    """對單一影片執行完整分析流程（Whisper → 載入模型 → 逐幀分析 → 輸出）。"""

    log = Signal(str)               # 主控台訊息
    phase = Signal(str)             # 目前階段描述（語音辨識 / 載入模型 / 分析中…）
    frame_ready = Signal(object)    # QImage
    stats_ready = Signal(dict)      # 每幀統計（不含影像）
    progress = Signal(int, int)     # (已處理幀, 總幀數；總幀數 0 = 未知)
    finished_ok = Signal(str, str)  # (報告 txt 路徑, 輸出影片路徑)
    failed = Signal(str)

    def __init__(self, video_path, output_dir, settings, config, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.output_dir = output_dir
        self.settings = dict(settings or {})
        self.config = config or {}
        self._stop_requested = False
        self._last_frame_emit = 0.0
        self._min_frame_interval = 1.0 / max(1, int(self.settings.get("preview_fps", 20)))

    # ── 外部控制 ────────────────────────────────
    def request_stop(self):
        self._stop_requested = True

    def _should_stop(self):
        return self._stop_requested

    # ── 每幀回呼（在 worker 執行緒中被 pipeline 呼叫）──
    def _on_frame(self, payload):
        # payload['frame'] 是 pipeline 內部的緩衝區本體，不是複本。
        # 必須在這個函式回傳前用完 —— bgr_to_qimage() 會產生獨立的 QImage，
        # 之後就不再參照原陣列。pop 出來也避免它被塞進 stats signal。
        frame = payload.pop("frame", None)

        now = time.time()
        if frame is not None and (now - self._last_frame_emit) >= self._min_frame_interval:
            self._last_frame_emit = now
            try:
                self.frame_ready.emit(
                    bgr_to_qimage(frame, int(self.settings.get("preview_width", 960)))
                )
            except Exception:
                pass  # 畫面轉檔失敗不該中斷分析

        self.stats_ready.emit(payload)
        self.progress.emit(payload.get("frame_count", 0), payload.get("total_frames", 0))

    # ── 主流程 ──────────────────────────────────
    def run(self):
        original_stdout, original_stderr = sys.stdout, sys.stderr
        relay = _StreamRelay(self.log.emit, original_stdout)
        sys.stdout = relay
        sys.stderr = relay
        try:
            hm = pipeline_api.apply_runtime_settings(self.settings)

            # pipeline 內大量使用相對路徑（model/gaze/… 等），
            # 與命令列版本一樣切到專案根目錄，避免路徑解析不一致。
            os.chdir(pipeline_api.PROJECT_DIR)
            os.makedirs(self.output_dir, exist_ok=True)

            self.phase.emit("檢查硬體")
            hm.enable_cuda_optimizations()
            self.log.emit(">>> 影片：%s" % self.video_path)
            self.log.emit(">>> 輸出：%s" % self.output_dir)
            self.log.emit(">>> 量測階段：%s" % sorted(hm.ACTIVE_STAGES))
            self.log.emit(">>> 跳幀設定：YOLO=%d GAZE=%d OCR=%d"
                          % (hm.YOLO_SKIP, hm.GAZE_SKIP, hm.OCR_SKIP))

            if self._should_stop():
                raise RuntimeError("使用者在開始前取消")

            # ── Step 0：Whisper 語音辨識（必須在 GPU 視覺模型載入前）──
            self.phase.emit("語音辨識（Whisper）")
            self.log.emit("=" * 50)
            self.log.emit("Step 0：語音辨識 — 首次執行需要數十秒至數分鐘，之後走快取")
            hm.prepare_speech_cache(self.video_path, self.output_dir)

            if self._should_stop():
                raise RuntimeError("使用者取消")

            # ── Step 1：載入視覺模型（已載入且設定未變則直接沿用）──
            self.phase.emit("載入視覺模型")
            self.log.emit("=" * 50)
            self.log.emit("Step 1：載入 YOLO / 視線估計 / EasyOCR")
            comp = pipeline_api.get_components(self.settings, self.config)

            if self._should_stop():
                raise RuntimeError("使用者取消")

            # ── Step 2：逐幀分析 ──
            self.phase.emit("逐幀分析中")
            self.log.emit("=" * 50)
            self.log.emit("Step 2：開始逐幀分析")
            hm.process_single_video(
                self.video_path,
                self.output_dir,
                comp["model_manager"],
                comp["interaction"],
                comp["gaze_pipeline"],
                comp["gaze_config"],
                comp["sign_tracker"],
                frame_callback=self._on_frame,
                should_stop=self._should_stop,
            )

            out_mp4, out_txt = pipeline_api.output_paths(self.video_path, self.output_dir)
            if not os.path.exists(out_txt):
                raise RuntimeError("分析結束但找不到報告檔：%s" % out_txt)

            self.phase.emit("已完成" if not self._stop_requested else "已中止（保留部分結果）")
            self.finished_ok.emit(out_txt, out_mp4 if os.path.exists(out_mp4) else "")

        except Exception as e:
            traceback.print_exc()
            self.phase.emit("發生錯誤")
            self.failed.emit(str(e))
        finally:
            sys.stdout, sys.stderr = original_stdout, original_stderr
