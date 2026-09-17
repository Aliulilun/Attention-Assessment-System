# -*- coding: utf-8 -*-
"""
Qt 綁定相容層
==============
本專案跑在 Python 3.9（conda 環境 mediapipe_py39）。PySide6 與 PyQt5
在 3.9 上都可安裝，但兩者的 API 名稱有幾處差異（Signal / pyqtSignal、
exec() / exec_()）。這一層把差異收斂在單一檔案，其他所有 UI 檔案只從
這裡 import，日後換綁定不需要改動任何頁面程式碼。

優先使用 PySide6（官方 Qt for Python、LGPL），找不到時退回 PyQt5。
兩者皆無時給出明確的安裝指示，而不是丟出難以理解的 ImportError。
"""

QT_BINDING = None

try:
    from PySide6 import QtCore, QtGui, QtWidgets           # noqa: F401
    from PySide6.QtCore import Signal, Slot, Qt, QTimer, QThread, QSize  # noqa: F401
    from PySide6.QtGui import (                             # noqa: F401
        QImage, QPixmap, QColor, QPainter, QFont, QPen, QBrush, QIcon,
        QLinearGradient, QPainterPath, QFontMetrics,
    )
    QT_BINDING = "PySide6"
except ImportError:  # pragma: no cover - 取決於使用者環境
    try:
        from PyQt5 import QtCore, QtGui, QtWidgets          # noqa: F401
        from PyQt5.QtCore import pyqtSignal as Signal       # noqa: F401
        from PyQt5.QtCore import pyqtSlot as Slot           # noqa: F401
        from PyQt5.QtCore import Qt, QTimer, QThread, QSize  # noqa: F401
        from PyQt5.QtGui import (                           # noqa: F401
            QImage, QPixmap, QColor, QPainter, QFont, QPen, QBrush, QIcon,
            QLinearGradient, QPainterPath, QFontMetrics,
        )
        QT_BINDING = "PyQt5"
    except ImportError:
        raise ImportError(
            "\n找不到 Qt 綁定，請先在 mediapipe_py39 環境安裝其中一個：\n"
            "    conda activate mediapipe_py39\n"
            "    pip install PySide6\n"
            "  （或）pip install PyQt5\n"
        )


def run_app(app):
    """啟動事件迴圈。PySide6 用 exec()，PyQt5 舊版只有 exec_()。"""
    if hasattr(app, "exec"):
        return app.exec()
    return app.exec_()


def run_dialog(dialog):
    """顯示 modal 對話框，回傳結果碼（同上，兩種綁定命名不同）。"""
    if hasattr(dialog, "exec"):
        return dialog.exec()
    return dialog.exec_()
