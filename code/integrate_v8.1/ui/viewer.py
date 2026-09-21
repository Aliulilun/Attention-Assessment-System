# -*- coding: utf-8 -*-
"""
分析結果檢視器（跨平台，不需要 GPU）
=====================================
只有「單次結果報表」與「歷史統計比較」兩個頁面，用來閱讀分析報告
（`event_record` txt），不含任何 AI 分析功能。

為什麼要單獨做這一支？
完整介面（`app.py`）會載入 YOLO、視線估計、Whisper、EasyOCR，
相依 PyTorch + CUDA，只能在 Windows + NVIDIA 顯卡上跑。但「看結果」
這件事本身完全不需要那些東西——這兩頁只用到 PyQt5/PySide6 與
（匯出時才用到的）pandas，所以在 macOS、Linux、沒有顯卡的 Windows
筆電上都能執行。

執行方式：
    python ui/viewer.py

相依套件只有兩個：
    pip install PySide6 pandas      （或 PyQt5 取代 PySide6）
pandas 只有按「匯出」時才需要，沒裝也能正常瀏覽。
"""

import os
import sys

VIEWER_DIR = os.path.dirname(os.path.abspath(__file__))
if VIEWER_DIR not in sys.path:
    sys.path.insert(0, VIEWER_DIR)


# pythonw / .app 啟動時沒有主控台，sys.stdout 會是 None，
# 此時任何 print() 都會丟 AttributeError。與 app.py 相同的防護。
class _NullStream:
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

from qt_compat import QtWidgets, Qt, QT_BINDING, run_app, QFont   # noqa: E402
import theme                                                      # noqa: E402
import pipeline_api                                               # noqa: E402
from pages.report_page import ReportPage                          # noqa: E402
from pages.history_page import HistoryPage                        # noqa: E402

NAV_ITEMS = [
    ("1　單次結果報表", "載入單一 event_record txt，看圖表與各關明細"),
    ("2　歷史統計比較", "掃描整個資料夾，跨受試者比較與匯出"),
]


class ViewerWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("注意力分析結果檢視器")
        self.resize(1380, 900)
        self.setMinimumSize(1080, 700)
        self._build()

    def _build(self):
        central = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(central)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self._build_sidebar())

        self.stack = QtWidgets.QStackedWidget()
        self.report_page = ReportPage()
        self.history_page = HistoryPage()
        self.stack.addWidget(self.report_page)
        self.stack.addWidget(self.history_page)
        lay.addWidget(self.stack, 1)
        self.setCentralWidget(central)

        self.history_page.open_report_requested.connect(self.show_report)

        self.status = self.statusBar()
        self.status.showMessage(
            "檢視器模式　・　Qt 綁定：%s　・　此版本不含分析功能" % QT_BINDING)

        for widget in self.findChildren(QtWidgets.QAbstractButton):
            widget.setCursor(Qt.PointingHandCursor)

    def _build_sidebar(self):
        bar = QtWidgets.QWidget()
        bar.setObjectName("Sidebar")
        bar.setFixedWidth(228)
        lay = QtWidgets.QVBoxLayout(bar)
        lay.setContentsMargins(0, 0, 0, 12)
        lay.setSpacing(0)

        title = QtWidgets.QLabel("結果檢視器")
        title.setObjectName("SidebarTitle")
        subtitle = QtWidgets.QLabel("Joint Attention Report Viewer")
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
        footer = QtWidgets.QLabel(
            "此版本只能閱讀分析報告。\n"
            "要執行 AI 分析請使用\n"
            "Windows + NVIDIA 顯卡的完整版。")
        footer.setProperty("role", "muted")
        footer.setStyleSheet("padding: 0 16px; font-size: 11px; line-height: 150%;")
        footer.setWordWrap(True)
        lay.addWidget(footer)
        return bar

    def goto(self, index):
        self.stack.setCurrentIndex(index)
        for i, btn in enumerate(self.nav_buttons):
            btn.setChecked(i == index)

    def show_report(self, txt_path):
        if self.report_page.load_report(txt_path):
            self.goto(0)


def main():
    try:
        QtWidgets.QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        QtWidgets.QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    except Exception:
        pass

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("注意力分析結果檢視器")
    app.setStyle("Fusion")
    # macOS 沒有微軟正黑體，字型堆疊在 theme.py 的 QSS 裡已經列了
    # PingFang TC / Noto Sans CJK TC 作為後備，這裡不寫死字型名稱。
    if sys.platform.startswith("win"):
        app.setFont(QFont("Microsoft JhengHei UI", 10))
    app.setStyleSheet(theme.build_qss())

    window = ViewerWindow()
    window.show()

    # 啟動時直接把預設輸出資料夾掃一遍，開起來就有東西可看；
    # 命令列也可以直接指定資料夾或單一報告：python ui/viewer.py <path>
    target = sys.argv[1] if len(sys.argv) > 1 else None
    if target and os.path.isfile(target):
        window.show_report(target)
    else:
        folder = target if target and os.path.isdir(target) else pipeline_api.default_output_dir()
        window.history_page.refresh(folder)
        if window.history_page.table.rowCount() > 0:
            window.goto(1)

    sys.exit(run_app(app))


if __name__ == "__main__":
    main()
