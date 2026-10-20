# -*- coding: utf-8 -*-
"""
即時體驗頁的中文疊圖（QPainter）
================================
cv2.putText 畫不出中文，因此「體驗者」標籤與小遊戲的星星、提示字
在轉成 QImage 之後用 QPainter 疊上。座標一律是原始畫面像素，
這裡依 QImage 實際寬度換算縮放比例。

在 worker 執行緒對 QImage 作畫是 Qt 允許的（不涉及 QWidget）。
"""

import math

from qt_compat import Qt, QColor, QPainter, QFont, QPen, QBrush, QPainterPath
import theme

FONT_FAMILY = "Microsoft JhengHei UI"


def _font(px, bold=True):
    f = QFont(FONT_FAMILY)
    f.setPixelSize(max(10, int(px)))
    f.setBold(bold)
    return f


def _star_path(cx, cy, r_out, r_in, points=5):
    path = QPainterPath()
    for i in range(points * 2):
        r = r_out if i % 2 == 0 else r_in
        a = -math.pi / 2 + i * math.pi / points
        x, y = cx + r * math.cos(a), cy + r * math.sin(a)
        if i == 0:
            path.moveTo(x, y)
        else:
            path.lineTo(x, y)
    path.closeSubpath()
    return path


def _banner(p, text, w, y, px, color=theme.TEXT_MAIN):
    """畫面上方置中、半透明底的大字。"""
    p.setFont(_font(px))
    fm = p.fontMetrics()
    tw = fm.horizontalAdvance(text) if hasattr(fm, "horizontalAdvance") else fm.width(text)
    pad = px * 0.5
    box_w, box_h = tw + pad * 2, px * 1.6
    x = (w - box_w) / 2.0
    p.setPen(Qt.NoPen)
    p.setBrush(QBrush(QColor(0, 0, 0, 160)))
    p.drawRoundedRect(int(x), int(y), int(box_w), int(box_h), 12, 12)
    p.setPen(QPen(QColor(color)))
    p.drawText(int(x), int(y), int(box_w), int(box_h), Qt.AlignCenter, text)


def _fmt(sec):
    return "錯過" if sec is None else "%.2f 秒" % sec


def paint_overlay(qimg, state, frame_w):
    """在 qimg 上疊「體驗者」標籤與遊戲畫面。state 為 LiveAnalyzer 的狀態（含 game）。"""
    s = qimg.width() / float(frame_w)
    W, H = qimg.width(), qimg.height()
    p = QPainter(qimg)
    p.setRenderHint(QPainter.Antialiasing)
    p.setRenderHint(QPainter.TextAntialiasing)

    box = state.get("person_box")
    if box is not None:
        x1, y1 = box[0] * s, box[1] * s
        p.setFont(_font(16))
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(QColor(theme.ACCENT)))
        p.drawRect(int(x1), int(max(0, y1 - 26)), 72, 26)
        p.setPen(QPen(QColor("#0c1016")))
        p.drawText(int(x1), int(max(0, y1 - 26)), 72, 26, Qt.AlignCenter, "體驗者")

    game = state.get("game")
    if game and game["phase"] != "IDLE":
        _paint_game(p, game, s, W, H)
    p.end()


def _paint_game(p, g, s, W, H):
    phase = g["phase"]
    stage_name = "第一關：看" if g["stage"] == "look" else "第二關：指"
    big = max(22, H * 0.065)

    if phase == "TARGET" and g["target_box"] is not None:
        x1, y1, x2, y2 = (v * s for v in g["target_box"])
        p.setPen(QPen(QColor(theme.WARN), 3, Qt.DashLine))
        p.setBrush(QBrush(QColor(255, 200, 87, 40)))
        p.drawRoundedRect(int(x1), int(y1), int(x2 - x1), int(y2 - y1), 16, 16)
        r = min(x2 - x1, y2 - y1) * 0.32
        p.setPen(QPen(QColor("#ffffff"), 3))
        p.setBrush(QBrush(QColor(theme.WARN)))
        p.drawPath(_star_path((x1 + x2) / 2, (y1 + y2) / 2, r, r * 0.45))
        if g["elapsed"] is not None:
            _banner(p, "%.2f 秒" % g["elapsed"], W, H - big * 2.2, big * 0.8, theme.WARN)

    if phase in ("CALIBRATING", "TARGET", "WAIT", "RETURN", "INTRO", "RESULT"):
        _banner(p, g["message"] if phase != "RESULT" else "遊戲結束！成績在右邊",
                W, big * 0.4, big)

    if phase in ("WAIT", "TARGET", "RETURN"):
        p.setFont(_font(big * 0.45, bold=False))
        p.setPen(QPen(QColor(theme.TEXT_MAIN)))
        p.drawText(16, int(H - 16 - big * 0.6), int(W / 2), int(big * 0.6),
                   Qt.AlignLeft | Qt.AlignVCenter,
                   "%s　第 %d / %d 回合" % (stage_name, g["round"], g["rounds"]))

    if phase == "RETURN":
        last = g["results"][g["stage"]][-1] if g["results"][g["stage"]] else None
        color = theme.BAD if last is None else theme.OK
        _banner(p, _fmt(last) if last is None else "%.2f 秒！" % last,
                W, H / 2 - big, big * 1.4, color)

    if phase == "CALIBRATING":
        cx, cy = W / 2, H / 2
        p.setPen(QPen(QColor(theme.ACCENT), 3))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(int(cx - 18), int(cy - 18), 36, 36)
        p.drawLine(int(cx - 30), int(cy), int(cx + 30), int(cy))
        p.drawLine(int(cx), int(cy - 30), int(cx), int(cy + 30))
