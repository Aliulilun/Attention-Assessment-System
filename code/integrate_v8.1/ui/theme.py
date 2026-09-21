# -*- coding: utf-8 -*-
"""
深色主題樣式表與共用色票
========================
顏色刻意與分析影片上的 OpenCV 標註色系一致，讓報告觀眾在「影片畫面」與
「介面數值」之間能直覺對應：
    綠  = 目標物 / 命中
    黃  = GAZING（正在注視目標）
    橘  = 機器人
    紅  = 未達成
"""

# ── 色票 ──────────────────────────────────────────────
BG_BASE      = "#12161d"   # 視窗底色
BG_PANEL     = "#1a1f29"   # 卡片 / 面板
BG_ELEVATED  = "#232a36"   # 輸入元件、表頭
BORDER       = "#2e3745"
TEXT_MAIN    = "#e6ebf2"
TEXT_MUTED   = "#8a97a8"

ACCENT       = "#4da3ff"   # 主色（按鈕、選取）
ACCENT_DARK  = "#2b7fd6"
ACCENT_LIGHT = "#7bbcff"
OK           = "#3ddc84"   # 命中 / 達成
WARN         = "#ffc857"   # GAZING
ROBOT        = "#ff9f43"   # 機器人
BAD          = "#ff5c6c"   # 未達成
BAD_LIGHT    = "#ff8794"
BAD_DARK     = "#e04b5a"
IDLE         = "#4a5568"   # 待機

# 互動回饋用（hover / pressed）。深色主題上只改邊框看不出來，必須連底色一起換。
BG_HOVER     = "#2c3646"
BG_PRESSED   = "#161c25"

# 清單選取列的底色。刻意不用高飽和的 ACCENT 當底：
# 整列被塗成亮藍色時，列首的核取方塊會被蓋掉、看不出有沒有打勾。
SELECT_BG    = "#2b4a6f"

# 反應等級配色（F / LR / HR / LI / HI）
LEVEL_COLORS = {
    "HI": "#3ddc84",
    "LI": "#7ed957",
    "HR": "#ffc857",
    "LR": "#ff9f43",
    "F":  "#ff5c6c",
    "x":  IDLE,
    "未偵測": IDLE,
}

# 兩組對照用色（歷史統計頁）
GROUP_COLORS = {
    "ASD": "#ff9f43",
    "TD":  "#4da3ff",
    "未分組": "#6b7a8f",
}


QSS = f"""
QWidget {{
    background-color: {BG_BASE};
    color: {TEXT_MAIN};
    font-family: "Microsoft JhengHei UI", "Microsoft JhengHei", "PingFang TC",
                 "Noto Sans CJK TC", "Segoe UI", sans-serif;
    font-size: 13px;
}}

QLabel[role="h1"] {{ font-size: 22px; font-weight: 700; }}
QLabel[role="h2"] {{ font-size: 16px; font-weight: 600; }}
QLabel[role="muted"] {{ color: {TEXT_MUTED}; }}

/* ── 側邊導覽 ── */
#Sidebar {{
    background-color: {BG_PANEL};
    border-right: 1px solid {BORDER};
}}
#SidebarTitle {{
    font-size: 15px; font-weight: 700; padding: 18px 16px 4px 16px;
}}
#SidebarSubtitle {{
    color: {TEXT_MUTED}; font-size: 11px; padding: 0 16px 14px 16px;
}}
QPushButton[role="nav"] {{
    background: transparent;
    border: none;
    border-left: 3px solid transparent;
    padding: 12px 16px;
    text-align: left;
    font-size: 13.5px;
    color: {TEXT_MUTED};
}}
QPushButton[role="nav"]:hover {{ background: {BG_ELEVATED}; color: {TEXT_MAIN}; }}
QPushButton[role="nav"]:checked {{
    background: {BG_ELEVATED};
    color: {TEXT_MAIN};
    border-left: 3px solid {ACCENT};
    font-weight: 600;
}}

/* ── 卡片 ── */
QFrame[role="card"] {{
    background-color: {BG_PANEL};
    border: 1px solid {BORDER};
    border-radius: 10px;
}}

/* ── 按鈕 ──
   每個狀態都給明顯的視覺差異：滑鼠移上去要變色、按下去要下沉。
   只改邊框顏色太細微，在深色主題上幾乎看不出來。 */
QPushButton {{
    background-color: {BG_ELEVATED};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 8px 16px;
}}
QPushButton:hover {{
    background-color: {BG_HOVER};
    border-color: {ACCENT};
    color: #ffffff;
}}
QPushButton:pressed {{
    background-color: {BG_PRESSED};
    border-color: {ACCENT_DARK};
    padding-top: 9px;
    padding-bottom: 7px;          /* 讓文字微微下沉，做出被按下的手感 */
}}
QPushButton:disabled {{ color: {IDLE}; border-color: {BORDER}; background-color: {BG_PANEL}; }}

QPushButton[role="primary"] {{
    background-color: {ACCENT}; border: none; color: #08121e; font-weight: 700;
    padding: 10px 22px; border-radius: 6px;
}}
QPushButton[role="primary"]:hover {{ background-color: {ACCENT_LIGHT}; color: #08121e; }}
QPushButton[role="primary"]:pressed {{
    background-color: {ACCENT_DARK}; color: #ffffff;
    padding-top: 11px; padding-bottom: 9px;
}}
QPushButton[role="primary"]:disabled {{ background-color: {IDLE}; color: #99a3b1; }}

QPushButton[role="danger"] {{
    background-color: {BAD}; border: none; color: #2b0508; font-weight: 700;
}}
QPushButton[role="danger"]:hover {{ background-color: {BAD_LIGHT}; color: #2b0508; }}
QPushButton[role="danger"]:pressed {{
    background-color: {BAD_DARK}; color: #ffffff;
    padding-top: 9px; padding-bottom: 7px;
}}
QPushButton[role="danger"]:disabled {{ background-color: {IDLE}; color: #99a3b1; }}

/* ── 輸入元件 ── */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTextEdit {{
    background-color: {BG_ELEVATED};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 6px 8px;
    selection-background-color: {ACCENT};
}}
QComboBox::drop-down {{ border: none; width: 20px; }}
QComboBox QAbstractItemView {{
    background-color: {BG_ELEVATED};
    border: 1px solid {BORDER};
    selection-background-color: {ACCENT};
}}

/* ── 清單 / 表格 ── */
QListWidget, QTableWidget, QTreeWidget {{
    background-color: {BG_PANEL};
    border: 1px solid {BORDER};
    border-radius: 8px;
    gridline-color: {BORDER};
}}
QListWidget::item {{ padding: 8px 10px; border-radius: 4px; }}
QListWidget::item:hover {{ background-color: {BG_HOVER}; }}
QListWidget::item:selected, QTableWidget::item:selected {{
    background-color: {SELECT_BG}; color: #ffffff;
}}
QHeaderView::section {{
    background-color: {BG_ELEVATED};
    color: {TEXT_MUTED};
    border: none;
    border-right: 1px solid {BORDER};
    border-bottom: 1px solid {BORDER};
    padding: 7px 8px;
    font-weight: 600;
}}
QTableCornerButton::section {{ background-color: {BG_ELEVATED}; border: none; }}

/* ── 進度條 ── */
QProgressBar {{
    background-color: {BG_ELEVATED};
    border: 1px solid {BORDER};
    border-radius: 8px;
    height: 16px;
    text-align: center;
    color: {TEXT_MAIN};
}}
QProgressBar::chunk {{ background-color: {ACCENT}; border-radius: 7px; }}

/* ── 分頁 ── */
QTabWidget::pane {{ border: 1px solid {BORDER}; border-radius: 8px; top: -1px; }}
QTabBar::tab {{
    background: transparent; color: {TEXT_MUTED};
    padding: 8px 18px; border-bottom: 2px solid transparent;
}}
QTabBar::tab:selected {{ color: {TEXT_MAIN}; border-bottom: 2px solid {ACCENT}; }}

/* ── 捲軸 ── */
QScrollBar:vertical   {{ background: transparent; width: 10px; margin: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 0; }}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
    background: {BORDER}; border-radius: 5px; min-height: 30px; min-width: 30px;
}}
QScrollBar::handle:hover {{ background: {IDLE}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

/* ── 主控台 ── */
#Console {{
    background-color: #0c1016;
    border: 1px solid {BORDER};
    border-radius: 8px;
    font-family: "Cascadia Mono", "Consolas", "Menlo", monospace;
    font-size: 11.5px;
    color: #b9c6d6;
}}

QSplitter::handle {{ background-color: {BORDER}; }}
QToolTip {{
    background-color: {BG_ELEVATED}; color: {TEXT_MAIN};
    border: 1px solid {BORDER}; padding: 5px;
}}

/* ── 輸入元件的互動回饋 ── */
QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover, QLineEdit:hover {{
    border-color: {ACCENT};
}}
QCheckBox {{ spacing: 8px; padding: 4px 2px; }}
QCheckBox:hover {{ color: #ffffff; }}
"""


# ══════════════════════════════════════════════════════
# 核取方塊圖示
# ══════════════════════════════════════════════════════
def check_icon_path():
    """產生一張白色勾勾 PNG，回傳檔案路徑（失敗時回傳 None）。

    為什麼要自己畫？
    一旦在 QSS 裡設定了 ::indicator 的樣式，Qt 就不再用原生方式繪製，
    連勾勾也一起消失，只剩一個空方塊。而不設樣式的話，原生勾勾在
    被選取（藍底）的那一列上會被蓋掉——使用者會看到「全選了但最上面
    那一列沒打勾」。兩邊都要顧，只能自備圖示。

    QSS 的 image: 只吃檔案路徑，不支援 data URI，所以在執行時畫一張
    到系統暫存目錄。畫不出來就回 None，樣式會退回純色填滿，
    仍然看得出勾選與否，只是沒有勾勾形狀。
    """
    import os
    import tempfile

    try:
        from qt_compat import QPixmap, QPainter, QPen, QColor, Qt

        out_dir = os.path.join(tempfile.gettempdir(), "attention_ui_assets")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "check.png")

        pix = QPixmap(18, 18)
        pix.fill(Qt.transparent)
        painter = QPainter(pix)
        painter.setRenderHint(QPainter.Antialiasing)
        pen = QPen(QColor("#08121e"))
        pen.setWidthF(2.6)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        painter.setPen(pen)
        painter.drawLine(4, 9, 7, 13)      # 勾的短邊
        painter.drawLine(7, 13, 14, 5)     # 勾的長邊
        painter.end()

        if not pix.save(path, "PNG"):
            return None
        return path.replace("\\", "/")     # QSS 的 url() 只吃正斜線
    except Exception:
        return None


def build_qss():
    """回傳完整樣式表（含核取方塊圖示）。app.py 啟動時呼叫這一支。"""
    icon = check_icon_path()
    image_rule = ("image: url(%s);" % icon) if icon else ""
    return QSS + f"""
/* ── 核取方塊（影片清單的勾選框、全選框）── */
QListWidget::indicator, QCheckBox::indicator {{
    width: 18px;
    height: 18px;
    border: 2px solid {IDLE};
    border-radius: 4px;
    background-color: {BG_BASE};
}}
QListWidget::indicator:hover, QCheckBox::indicator:hover {{
    border-color: {ACCENT};
    background-color: {BG_HOVER};
}}
QListWidget::indicator:checked, QCheckBox::indicator:checked {{
    border-color: {ACCENT};
    background-color: {ACCENT};
    {image_rule}
}}
QListWidget::indicator:indeterminate, QCheckBox::indicator:indeterminate {{
    border-color: {ACCENT};
    background-color: {ACCENT_DARK};
}}
"""
