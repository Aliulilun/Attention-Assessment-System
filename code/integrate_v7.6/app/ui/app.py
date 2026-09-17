# -*- coding: utf-8 -*-
"""
兒童注意力監測與量化分析平台 — 圖形化介面
===========================================
執行方式（務必用專案的 conda 環境）：

    conda activate mediapipe_py39
    cd C:\\project
    python ui\\app.py

或直接雙擊 C:\\project\\run_ui.bat

四個頁面：
    1. 影片載入與參數設定
    2. 即時監測主畫面
    3. 單次結果報表
    4. 歷史統計與比較

介面本身不做任何分析判斷，所有計分都由 hurry/main.py 與 modules/ 負責，
確保介面上的數字與命令列跑出來的完全一致。
"""

import os
import sys

# 讓 ui/ 內的模組彼此可以直接 import（ui/app.py 可能從任何工作目錄啟動）
UI_DIR = os.path.dirname(os.path.abspath(__file__))
if UI_DIR not in sys.path:
    sys.path.insert(0, UI_DIR)


# ──────────────────────────────────────────────────────
# pythonw.exe 沒有主控台，sys.stdout / sys.stderr 會是 None，
# 此時任何 print() 都會丟 AttributeError: 'NoneType' has no attribute 'write'。
# 分析管線與 modules/ 內大量使用 print()，不補這一段的話，
# 用 run_ui.bat（走 pythonw，不留黑視窗）啟動就會在載入設定時直接崩潰。
# 訊息不會消失——AnalysisWorker 會把 stdout 轉接到介面的主控台面板。
# ──────────────────────────────────────────────────────
class _NullStream:
    """吞掉輸出的替身，介面自己的主控台面板才是真正的輸出管道。"""

    def write(self, _text):
        return 0

    def flush(self):
        pass

    def isatty(self):
        return False


if sys.stdout is None:
    sys.stdout = _NullStream()
if sys.stderr is None:
    sys.stderr = _NullStream()

from qt_compat import QtWidgets, Qt, QT_BINDING, run_app, QFont, QTimer   # noqa: E402
import theme                                                       # noqa: E402
import pipeline_api                                                # noqa: E402
from workers import AnalysisWorker                                 # noqa: E402
from pages.setup_page import SetupPage                             # noqa: E402
from pages.monitor_page import MonitorPage                         # noqa: E402
from pages.report_page import ReportPage                           # noqa: E402
from pages.history_page import HistoryPage                         # noqa: E402

NAV_ITEMS = [
    ("1　影片載入與設定", "選擇影片、確認分析參數"),
    ("2　即時監測", "逐幀畫面與 T0 / TB / TH 即時判定"),
    ("3　單次結果報表", "本次分析的完整結果與圖表"),
    ("4　歷史統計比較", "跨受試者橫向比較與匯出"),
]


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.worker = None
        self._last_output_dir = pipeline_api.default_output_dir()
        # ── 批次佇列狀態 ──
        self._queue = []            # 待跑的影片路徑
        self._queue_index = 0       # 目前跑到第幾支（0 起算）
        self._queue_settings = {}
        self._queue_config = {}
        self._queue_done = []       # 成功完成的 (影片, 報告路徑)
        self._queue_failed = []     # 失敗的 (影片, 錯誤訊息)
        self._batch_aborted = False # 使用者按了停止 → 不再往下跑
        self.setWindowTitle("兒童注意力監測與量化分析平台")
        self.resize(1480, 940)
        self.setMinimumSize(1180, 760)
        self._build()

    # ────────────────────────────────────────────
    def _build(self):
        central = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(central)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        lay.addWidget(self._build_sidebar())

        self.stack = QtWidgets.QStackedWidget()
        self.setup_page = SetupPage()
        self.monitor_page = MonitorPage()
        self.report_page = ReportPage()
        self.history_page = HistoryPage()
        for page in (self.setup_page, self.monitor_page, self.report_page, self.history_page):
            self.stack.addWidget(page)
        lay.addWidget(self.stack, 1)
        self.setCentralWidget(central)

        self.status = self.statusBar()
        self.status.showMessage("就緒　・　Qt 綁定：%s　・　專案目錄：%s"
                                % (QT_BINDING, pipeline_api.PROJECT_DIR))

        # 可點擊的元件一律換成手指游標。QSS 沒有 cursor 屬性，只能用程式設定；
        # 少了這個回饋，使用者滑過按鈕時會覺得「這裡好像不能點」。
        for widget in self.findChildren(QtWidgets.QAbstractButton):
            widget.setCursor(Qt.PointingHandCursor)

        # 頁面之間的串接
        self.setup_page.start_requested.connect(self.start_analysis)
        self.setup_page.view_report_requested.connect(self.show_report)
        self.monitor_page.stop_requested.connect(self.stop_analysis)
        self.history_page.open_report_requested.connect(self.show_report)

    def _build_sidebar(self):
        bar = QtWidgets.QWidget()
        bar.setObjectName("Sidebar")
        bar.setFixedWidth(228)
        lay = QtWidgets.QVBoxLayout(bar)
        lay.setContentsMargins(0, 0, 0, 12)
        lay.setSpacing(0)

        title = QtWidgets.QLabel("注意力監測平台")
        title.setObjectName("SidebarTitle")
        subtitle = QtWidgets.QLabel("Joint Attention Assessment")
        subtitle.setObjectName("SidebarSubtitle")
        lay.addWidget(title)
        lay.addWidget(subtitle)

        self.nav_buttons = []
        group = QtWidgets.QButtonGroup(self)
        group.setExclusive(True)
        for index, (label, tip) in enumerate(NAV_ITEMS):
            btn = QtWidgets.QPushButton(label)
            btn.setProperty("role", "nav")
            btn.setCheckable(True)
            btn.setToolTip(tip)
            btn.clicked.connect(lambda _c=False, i=index: self.goto(i))
            group.addButton(btn)
            lay.addWidget(btn)
            self.nav_buttons.append(btn)
        self.nav_buttons[0].setChecked(True)

        lay.addStretch(1)
        footer = QtWidgets.QLabel("ASD / TD 共同注意力\n多模態 AI 自動評估")
        footer.setProperty("role", "muted")
        footer.setStyleSheet("padding: 0 16px; font-size: 11px; line-height: 150%;")
        footer.setWordWrap(True)
        lay.addWidget(footer)
        return bar

    def goto(self, index):
        self.stack.setCurrentIndex(index)
        for i, btn in enumerate(self.nav_buttons):
            btn.setChecked(i == index)

    # ────────────────────────────────────────────
    # 分析流程
    # ────────────────────────────────────────────
    def start_analysis(self, videos, settings, config):
        """接受一份影片清單，依序跑完。單支分析就是長度為 1 的清單。"""
        if self.worker is not None and self.worker.isRunning():
            QtWidgets.QMessageBox.information(
                self, "分析進行中", "目前已有影片正在分析，請先等待完成或按「停止分析」。")
            return
        if not videos:
            return

        self._queue = list(videos)
        self._queue_index = 0
        self._queue_settings = settings
        self._queue_config = config
        self._queue_done = []
        self._queue_failed = []
        self._batch_aborted = False
        self._last_output_dir = settings["output_dir"]

        self.setup_page.set_running(True)
        self.goto(1)
        if len(self._queue) > 1:
            self.monitor_page.append_log(
                "=" * 50 + "\n批次分析：共 %d 支影片\n" % len(self._queue) + "=" * 50)
        self._run_next()

    def _run_next(self):
        """啟動佇列中的下一支影片。跑完或中止時收尾。"""
        if self._batch_aborted or self._queue_index >= len(self._queue):
            self._finish_batch()
            return

        video_path = self._queue[self._queue_index]
        total = len(self._queue)
        name = os.path.basename(video_path)
        # 批次時保留主控台內容，才能回頭看前面幾支的訊息
        self.monitor_page.reset(video_name=name,
                                max_stage=self._queue_settings["max_stage"],
                                queue_index=self._queue_index + 1,
                                queue_total=total,
                                keep_console=(self._queue_index > 0))
        self.monitor_page.set_running(True, "準備中…")

        self.worker = AnalysisWorker(video_path, self._queue_settings["output_dir"],
                                     self._queue_settings, self._queue_config, self)
        self.worker.log.connect(self.monitor_page.append_log)
        self.worker.phase.connect(self._on_phase)
        self.worker.frame_ready.connect(self.monitor_page.update_frame)
        self.worker.stats_ready.connect(self.monitor_page.update_stats)
        self.worker.progress.connect(self.monitor_page.update_progress)
        self.worker.finished_ok.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(self._on_thread_done)
        self.worker.start()
        self.status.showMessage("分析中（%d/%d）：%s" % (self._queue_index + 1, total, name))

    def stop_analysis(self):
        # 停止 = 中止整批，不只是當前這一支。目前這支仍會走完 finally 寫出報告。
        self._batch_aborted = True
        if self.worker is not None and self.worker.isRunning():
            self.worker.request_stop()
            remaining = len(self._queue) - self._queue_index - 1
            if remaining > 0:
                self.status.showMessage(
                    "已送出停止指令，將中止整批（剩餘 %d 支不會執行）" % remaining)
            else:
                self.status.showMessage("已送出停止指令，等待目前這一幀處理完…")

    def _on_phase(self, text):
        self.monitor_page.set_phase(text)
        self.status.showMessage(text)

    def _on_finished(self, txt_path, video_path):
        self._queue_done.append((self._queue[self._queue_index], txt_path, video_path))
        self.monitor_page.append_log("✅ 完成：%s" % os.path.basename(txt_path))

    def _on_failed(self, message):
        # 單支失敗不中斷整批——批次跑一整晚，不該因為第 3 支壞掉就全停。
        video = self._queue[self._queue_index] if self._queue_index < len(self._queue) else "?"
        self._queue_failed.append((os.path.basename(str(video)), message))
        self.monitor_page.append_log("❌ 分析失敗：%s" % message)
        self.status.showMessage("分析失敗：%s" % os.path.basename(str(video)))

    def _on_thread_done(self):
        """worker 執行緒結束（不論成功失敗）→ 換下一支。"""
        self.monitor_page.set_running(False)
        self._queue_index += 1
        # 用 singleShot 讓目前這輪 signal 全部處理完再啟動下一支，
        # 避免在 QThread.finished 的處理過程中就建立新執行緒。
        QTimer.singleShot(0, self._run_next)

    def _finish_batch(self):
        self.setup_page.set_running(False)
        self.setup_page.refresh_video_list()
        max_stage = self._queue_settings.get("max_stage", 10)
        self.history_page.refresh(self._last_output_dir, max_stage=max_stage)

        done, failed = len(self._queue_done), len(self._queue_failed)
        total = len(self._queue)

        # 有成功的就載入最後一份報告，讓使用者馬上看到結果
        if self._queue_done:
            _v, txt_path, out_video = self._queue_done[-1]
            if self.report_page.load_report(txt_path, out_video, max_stage=max_stage):
                self.goto(2)

        if total <= 1:
            if failed:
                QtWidgets.QMessageBox.critical(
                    self, "分析失敗",
                    "%s\n\n完整錯誤訊息請看「即時監測」頁的主控台。"
                    % self._queue_failed[0][1])
            self.status.showMessage("分析完成" if done else "分析失敗")
            return

        summary = "共 %d 支：成功 %d 支" % (total, done)
        if failed:
            summary += "、失敗 %d 支" % failed
        if self._batch_aborted:
            skipped = total - done - failed
            summary += "（使用者中止，%d 支未執行）" % max(skipped, 0)
        self.status.showMessage("批次分析結束 — " + summary)
        self.monitor_page.append_log("=" * 50 + "\n批次結束：" + summary + "\n" + "=" * 50)

        detail = summary
        if self._queue_failed:
            detail += "\n\n失敗清單：\n" + "\n".join(
                "・%s — %s" % (n, m) for n, m in self._queue_failed)
        QtWidgets.QMessageBox.information(self, "批次分析結束", detail)

    def show_report(self, txt_path):
        max_stage = self.setup_page.collect_settings()["max_stage"]
        if self.report_page.load_report(txt_path, max_stage=max_stage):
            self.goto(2)

    # ────────────────────────────────────────────
    def closeEvent(self, event):
        if self.worker is not None and self.worker.isRunning():
            answer = QtWidgets.QMessageBox.question(
                self, "分析尚未結束",
                "目前還在分析影片，關閉視窗會中止分析（已分析的部分仍會寫出報告）。\n確定要關閉嗎？",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No)
            if answer != QtWidgets.QMessageBox.Yes:
                event.ignore()
                return
            self.worker.request_stop()
            # 給 pipeline 時間走完 finally（釋放 VideoWriter、縫合音軌、寫報告）
            self.worker.wait(15000)
        pipeline_api.release_components()
        event.accept()


def main():
    # 高 DPI 螢幕（報告用的筆電多半是）文字才不會糊。
    # PySide6 預設已開啟，這兩個屬性在新版被標為棄用，因此包在 try 裡。
    try:
        QtWidgets.QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        QtWidgets.QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    except Exception:
        pass

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("兒童注意力監測與量化分析平台")
    app.setStyle("Fusion")
    app.setFont(QFont("Microsoft JhengHei UI", 10))
    # build_qss() 會先產生核取方塊的勾勾圖示，再組出完整樣式表
    app.setStyleSheet(theme.build_qss())

    window = MainWindow()
    window.show()
    sys.exit(run_app(app))


if __name__ == "__main__":
    main()
