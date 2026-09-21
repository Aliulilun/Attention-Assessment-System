# -*- coding: utf-8 -*-
"""
共用元件
=========
圖表刻意用 QPainter 自繪，而不是嵌 matplotlib：
  1. matplotlib 預設字型沒有中文字，Windows 上會整排變成豆腐方塊「□□□」，
     報告投影時特別明顯；Qt 直接沿用系統的中文字型，不需要額外設定。
  2. 嵌入 matplotlib 要指定 backend 並與 Qt 綁定版本對齊（QtAgg + QT_API），
     PySide6/PyQt5 兩種情境要各測一次，是報告前夕不必要的風險。
  3. 這裡的圖表都很單純（長條、圓環），自繪的程式碼量比接 matplotlib 少。
"""

from qt_compat import (
    QtWidgets, Qt, QPainter, QColor, QPen, QBrush, QFont, QSize, QFontMetrics,
)
import theme


# ══════════════════════════════════════════════════════
# 版面小工具
# ══════════════════════════════════════════════════════
class Card(QtWidgets.QFrame):
    """圓角面板，可選標題。"""

    def __init__(self, title=None, parent=None):
        super().__init__(parent)
        self.setProperty("role", "card")
        self._layout = QtWidgets.QVBoxLayout(self)
        self._layout.setContentsMargins(16, 14, 16, 16)
        self._layout.setSpacing(10)
        if title:
            lab = QtWidgets.QLabel(title)
            lab.setProperty("role", "h2")
            self._layout.addWidget(lab)

    def body(self):
        return self._layout

    def add(self, widget, stretch=0):
        self._layout.addWidget(widget, stretch)
        return widget


class StatTile(QtWidgets.QFrame):
    """單一數值卡（大字 + 說明）。"""

    def __init__(self, caption, value="—", color=None, parent=None):
        super().__init__(parent)
        self.setProperty("role", "card")
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(2)
        self.value_label = QtWidgets.QLabel(value)
        f = QFont()
        f.setPointSize(20)
        f.setBold(True)
        self.value_label.setFont(f)
        if color:
            self.value_label.setStyleSheet("color: %s;" % color)
        cap = QtWidgets.QLabel(caption)
        cap.setProperty("role", "muted")
        lay.addWidget(self.value_label)
        lay.addWidget(cap)

    def set_value(self, text, color=None):
        self.value_label.setText(str(text))
        if color:
            self.value_label.setStyleSheet("color: %s;" % color)


class StatusLamp(QtWidgets.QWidget):
    """狀態指示燈：左側圓點 + 名稱 + 右側狀態文字。

    顏色與分析影片上的 OpenCV 標註同步，觀眾看影片與看介面能對得起來。
    """

    def __init__(self, name, parent=None):
        super().__init__(parent)
        self._on = False
        self._on_color = theme.OK
        self._text = "—"
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 3, 0, 3)
        lay.setSpacing(8)
        self._dot = _Dot()
        self._name = QtWidgets.QLabel(name)
        self._value = QtWidgets.QLabel("—")
        self._value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._value.setStyleSheet("color: %s; font-weight: 600;" % theme.TEXT_MUTED)
        lay.addWidget(self._dot)
        lay.addWidget(self._name)
        lay.addStretch(1)
        lay.addWidget(self._value)

    def set_state(self, active, text, on_color=None, off_color=None):
        self._dot.set_color(QColor(on_color or theme.OK) if active
                            else QColor(off_color or theme.IDLE))
        self._value.setText(str(text))
        self._value.setStyleSheet(
            "color: %s; font-weight: 600;" % ((on_color or theme.OK) if active else theme.TEXT_MUTED)
        )


class _Dot(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._color = QColor(theme.IDLE)
        self.setFixedSize(12, 12)

    def set_color(self, color):
        self._color = color
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        # 外圈微光：亮起時比較顯眼，適合投影
        glow = QColor(self._color)
        glow.setAlpha(70)
        p.setBrush(QBrush(glow))
        p.drawEllipse(0, 0, 12, 12)
        p.setBrush(QBrush(self._color))
        p.drawEllipse(3, 3, 6, 6)
        p.end()


def level_badge_text(level):
    """反應等級的中文說明（表格 tooltip / 圖例用）。"""
    return {
        "HI": "HI 高度主動（指向＋看回人）",
        "LI": "LI 視線交替（看物＋看回人）",
        "HR": "HR 遠物基本反應",
        "LR": "LR 低反應",
        "F":  "F 無反應",
    }.get(level, "未偵測")


# ══════════════════════════════════════════════════════
# 圖表
# ══════════════════════════════════════════════════════
class BarChart(QtWidgets.QWidget):
    """單組長條圖：每個 Stage 一根，可各自指定顏色。

    資料格式：[(標籤, 數值, 顏色字串), ...]
    """

    def __init__(self, y_max=4, y_label="", parent=None):
        super().__init__(parent)
        self._data = []
        self._y_max = y_max
        self._y_label = y_label
        self.setMinimumHeight(220)

    def set_data(self, data, y_max=None):
        self._data = list(data)
        if y_max:
            self._y_max = y_max
        self.update()

    def sizeHint(self):
        return QSize(560, 240)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        left, right, top, bottom = 44, 12, 14, 34
        plot_w = max(1, w - left - right)
        plot_h = max(1, h - top - bottom)

        if not self._data:
            p.setPen(QPen(QColor(theme.TEXT_MUTED)))
            p.drawText(self.rect(), Qt.AlignCenter, "尚無資料")
            p.end()
            return

        # 水平格線與 Y 軸刻度
        # int()：y_max 可能被傳入浮點數（例如平均值），range() 只吃整數
        steps = int(min(self._y_max, 5)) or 1
        p.setFont(QFont(self.font().family(), 8))
        for i in range(steps + 1):
            val = self._y_max * i / steps
            y = top + plot_h - plot_h * i / steps
            p.setPen(QPen(QColor(theme.BORDER), 1, Qt.DotLine))
            p.drawLine(left, int(y), left + plot_w, int(y))
            p.setPen(QPen(QColor(theme.TEXT_MUTED)))
            p.drawText(0, int(y) - 8, left - 8, 16, Qt.AlignRight | Qt.AlignVCenter,
                       ("%d" % val) if float(val).is_integer() else ("%.1f" % val))

        n = len(self._data)
        slot = plot_w / float(n)
        bar_w = min(46.0, slot * 0.62)
        for i, (label, value, color) in enumerate(self._data):
            cx = left + slot * (i + 0.5)
            ratio = 0.0 if self._y_max <= 0 else max(0.0, min(1.0, float(value) / self._y_max))
            bar_h = plot_h * ratio
            x = cx - bar_w / 2
            y = top + plot_h - bar_h
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(QColor(color)))
            p.drawRoundedRect(int(x), int(y), int(bar_w), int(max(bar_h, 2)), 4, 4)

            # 數值標籤（0 也標，讓「這關是 0 分」與「沒有這關」看得出差別）
            p.setPen(QPen(QColor(theme.TEXT_MAIN)))
            p.setFont(QFont(self.font().family(), 8, QFont.Bold))
            txt = ("%d" % value) if float(value).is_integer() else ("%.1f" % value)
            p.drawText(int(cx - slot / 2), int(y) - 16, int(slot), 14,
                       Qt.AlignCenter, txt)

            # X 軸標籤（空間不足時自動省略）
            p.setPen(QPen(QColor(theme.TEXT_MUTED)))
            p.setFont(QFont(self.font().family(), 8))
            fm = QFontMetrics(p.font())
            elided = fm.elidedText(str(label), Qt.ElideRight, int(slot) - 2)
            p.drawText(int(cx - slot / 2), h - bottom + 6, int(slot), 20,
                       Qt.AlignHCenter | Qt.AlignTop, elided)
        p.end()


class GroupedBarChart(QtWidgets.QWidget):
    """分組長條圖：同一個 X 位置並排多組（ASD vs TD 比較用）。

    labels: ["S1", "S2", ...]
    series: [(組名, [值...], 顏色), ...]
    """

    def __init__(self, y_max=4, parent=None):
        super().__init__(parent)
        self._labels = []
        self._series = []
        self._y_max = y_max
        self.setMinimumHeight(260)

    def set_data(self, labels, series, y_max=None):
        self._labels = list(labels)
        self._series = list(series)
        if y_max:
            self._y_max = y_max
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        left, right, top, bottom = 44, 12, 30, 34
        plot_w = max(1, w - left - right)
        plot_h = max(1, h - top - bottom)

        if not self._labels or not self._series:
            p.setPen(QPen(QColor(theme.TEXT_MUTED)))
            p.drawText(self.rect(), Qt.AlignCenter, "尚無資料（請先在表格中標記 ASD / TD 分組）")
            p.end()
            return

        # 圖例
        p.setFont(QFont(self.font().family(), 8))
        lx = left
        for name, _vals, color in self._series:
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(QColor(color)))
            p.drawRoundedRect(lx, 8, 10, 10, 2, 2)
            p.setPen(QPen(QColor(theme.TEXT_MUTED)))
            p.drawText(lx + 15, 8, 120, 12, Qt.AlignLeft | Qt.AlignVCenter, str(name))
            lx += 15 + QFontMetrics(p.font()).horizontalAdvance(str(name)) + 22

        steps = int(min(self._y_max, 4)) or 1   # 同上：防止浮點 y_max 讓 range() 爆掉
        for i in range(steps + 1):
            val = self._y_max * i / steps
            y = top + plot_h - plot_h * i / steps
            p.setPen(QPen(QColor(theme.BORDER), 1, Qt.DotLine))
            p.drawLine(left, int(y), left + plot_w, int(y))
            p.setPen(QPen(QColor(theme.TEXT_MUTED)))
            p.setFont(QFont(self.font().family(), 8))
            p.drawText(0, int(y) - 8, left - 8, 16, Qt.AlignRight | Qt.AlignVCenter, "%.1f" % val)

        n = len(self._labels)
        k = max(1, len(self._series))
        slot = plot_w / float(n)
        group_w = slot * 0.7
        bar_w = group_w / k
        for i in range(n):
            base = left + slot * i + (slot - group_w) / 2
            for j, (_name, values, color) in enumerate(self._series):
                value = values[i] if i < len(values) else 0
                ratio = 0.0 if self._y_max <= 0 else max(0.0, min(1.0, float(value) / self._y_max))
                bar_h = plot_h * ratio
                x = base + bar_w * j
                y = top + plot_h - bar_h
                p.setPen(Qt.NoPen)
                p.setBrush(QBrush(QColor(color)))
                p.drawRoundedRect(int(x), int(y), int(max(bar_w - 2, 2)), int(max(bar_h, 2)), 3, 3)
            p.setPen(QPen(QColor(theme.TEXT_MUTED)))
            p.setFont(QFont(self.font().family(), 8))
            fm = QFontMetrics(p.font())
            elided = fm.elidedText(str(self._labels[i]), Qt.ElideRight, int(slot) - 2)
            p.drawText(int(left + slot * i), h - bottom + 6, int(slot), 20,
                       Qt.AlignHCenter | Qt.AlignTop, elided)
        p.end()


class DonutChart(QtWidgets.QWidget):
    """圓環圖：反應等級分布。資料格式 [(標籤, 數值, 顏色), ...]"""

    def __init__(self, center_caption="關卡", parent=None):
        super().__init__(parent)
        self._data = []
        self._caption = center_caption
        self.setMinimumHeight(220)

    def set_data(self, data):
        self._data = [(l, v, c) for (l, v, c) in data if v > 0]
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        total = sum(v for _l, v, _c in self._data)

        size = min(w * 0.5, h) - 16
        size = max(size, 40)
        cx, cy = 16 + size / 2, h / 2

        if total <= 0:
            p.setPen(QPen(QColor(theme.TEXT_MUTED)))
            p.drawText(self.rect(), Qt.AlignCenter, "尚無資料")
            p.end()
            return

        start = 90 * 16  # Qt 角度單位為 1/16 度，從 12 點鐘方向開始
        for _label, value, color in self._data:
            span = int(-360 * 16 * value / total)
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(QColor(color)))
            p.drawPie(int(cx - size / 2), int(cy - size / 2), int(size), int(size), int(start), span)
            start += span

        # 挖空中心 → 圓環
        p.setBrush(QBrush(QColor(theme.BG_PANEL)))
        inner = size * 0.58
        p.drawEllipse(int(cx - inner / 2), int(cy - inner / 2), int(inner), int(inner))
        p.setPen(QPen(QColor(theme.TEXT_MAIN)))
        p.setFont(QFont(self.font().family(), 14, QFont.Bold))
        p.drawText(int(cx - inner / 2), int(cy - inner / 2), int(inner), int(inner * 0.6),
                   Qt.AlignCenter, str(total))
        p.setPen(QPen(QColor(theme.TEXT_MUTED)))
        p.setFont(QFont(self.font().family(), 8))
        p.drawText(int(cx - inner / 2), int(cy), int(inner), int(inner * 0.4),
                   Qt.AlignHCenter | Qt.AlignTop, self._caption)

        # 右側圖例
        p.setFont(QFont(self.font().family(), 9))
        ly = cy - len(self._data) * 11
        lx = 16 + size + 24
        for label, value, color in self._data:
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(QColor(color)))
            p.drawRoundedRect(int(lx), int(ly), 10, 10, 2, 2)
            p.setPen(QPen(QColor(theme.TEXT_MAIN)))
            p.drawText(int(lx + 16), int(ly - 2), 220, 14, Qt.AlignLeft | Qt.AlignVCenter,
                       "%s ・ %d 關" % (label, value))
            ly += 22
        p.end()


class TimelineBar(QtWidgets.QWidget):
    """階段時間軸：把每一關的 T0 位置畫在整支影片的時間長度上。

    讓觀眾一眼看出「哪一關的提示是什麼時候出現的」，以及目前播放到哪。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._marks = []      # [(time_sec, stage, color)]
        self._duration = 0.0
        self._cursor = 0.0
        self.setFixedHeight(46)

    def set_duration(self, seconds):
        self._duration = max(0.0, float(seconds or 0))
        self.update()

    def set_marks(self, marks):
        self._marks = list(marks)
        self.update()

    def set_cursor(self, seconds):
        self._cursor = float(seconds or 0)
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        track_y, track_h = 16, 10
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(QColor(theme.BG_ELEVATED)))
        p.drawRoundedRect(0, track_y, w, track_h, 5, 5)

        if self._duration <= 0:
            p.setPen(QPen(QColor(theme.TEXT_MUTED)))
            p.setFont(QFont(self.font().family(), 8))
            p.drawText(0, 28, w, 16, Qt.AlignCenter, "尚未取得影片長度")
            p.end()
            return

        # 已播放進度
        ratio = max(0.0, min(1.0, self._cursor / self._duration))
        p.setBrush(QBrush(QColor(theme.ACCENT)))
        p.drawRoundedRect(0, track_y, int(w * ratio), track_h, 5, 5)

        p.setFont(QFont(self.font().family(), 7, QFont.Bold))
        for t, stage, color in self._marks:
            x = int(w * max(0.0, min(1.0, t / self._duration)))
            p.setPen(QPen(QColor(color), 2))
            p.drawLine(x, track_y - 5, x, track_y + track_h + 5)
            p.setPen(QPen(QColor(color)))
            p.drawText(x - 14, track_y + track_h + 6, 28, 14, Qt.AlignCenter, str(stage))
        p.end()
