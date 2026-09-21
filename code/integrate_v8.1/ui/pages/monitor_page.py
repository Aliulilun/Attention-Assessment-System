# -*- coding: utf-8 -*-
"""
頁面二：即時監測主畫面
========================
報告時的主秀。左邊播放帶標註的分析畫面，右邊同步顯示 AI 的判斷依據
（階段、語音觸發、指向、視線命中）與逐關的 T0 / TB / TH 累積結果。

畫面上的標註色與右側指示燈色刻意一致，觀眾就能自己把
「影片裡的黃框亮起來」＝「介面上 Gazing 亮起來」＝「TB 表格 +1」串起來，
不需要簡報者一句一句解釋。
"""

import os
import sys

from qt_compat import QtWidgets, Qt, Signal, QPixmap, QFont, QColor
import theme
from widgets import Card, StatTile, StatusLamp, TimelineBar
from report_parser import STAGE_SHORT

TABLE_HEADERS = ["關卡", "T0", "指向", "TB", "TH", "等級"]

# ──────────────────────────────────────────────────────
# 即時預覽的反應等級直接沿用 modules/stage_scoring.py。
# 這支模組是純 Python、沒有任何重量級相依（不會拉進 torch / ultralytics），
# 匯入成本趨近於零，卻能保證「監測頁即時顯示的等級」與
# 「報告檔最終寫出的等級」用的是同一套規則——自己抄一份判定邏輯，
# 日後計分規則一改就會兩邊不一致，是最容易被忽略的資料不一致來源。
# ──────────────────────────────────────────────────────
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

try:
    from modules.stage_scoring import compute_stage_score

    def _preview_level(rec):
        return compute_stage_score(rec)[1]
except Exception:   # pragma: no cover - 只在專案結構被搬動時才會走到
    def _preview_level(_rec):
        return "—"


class MonitorPage(QtWidgets.QWidget):

    stop_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._duration = 0.0
        self._marks = {}
        self._build()
        self.reset()

    # ────────────────────────────────────────────
    def _build(self):
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 18)
        root.setSpacing(12)

        # ── 標頭 ──
        head = QtWidgets.QHBoxLayout()
        titles = QtWidgets.QVBoxLayout()
        titles.setSpacing(2)
        self.title = QtWidgets.QLabel("即時監測")
        self.title.setProperty("role", "h1")
        self.phase_label = QtWidgets.QLabel("待機中")
        self.phase_label.setProperty("role", "muted")
        titles.addWidget(self.title)
        titles.addWidget(self.phase_label)
        head.addLayout(titles)
        head.addStretch(1)
        self.stop_btn = QtWidgets.QPushButton("停止分析")
        self.stop_btn.setProperty("role", "danger")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._on_stop)
        head.addWidget(self.stop_btn)
        root.addLayout(head)

        # ── 進度 ──
        prog_row = QtWidgets.QHBoxLayout()
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress_text = QtWidgets.QLabel("—")
        self.progress_text.setProperty("role", "muted")
        self.progress_text.setMinimumWidth(230)
        self.progress_text.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        prog_row.addWidget(self.progress, 1)
        prog_row.addWidget(self.progress_text)
        root.addLayout(prog_row)

        # ── 主體 ──
        split = QtWidgets.QSplitter(Qt.Horizontal)

        left = QtWidgets.QWidget()
        left_lay = QtWidgets.QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.setSpacing(10)

        video_card = Card()
        self.video_label = QtWidgets.QLabel("尚未開始分析\n\n請先到「影片載入」頁選擇影片")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumHeight(360)
        self.video_label.setStyleSheet(
            "background:#0c1016; border-radius:8px; color:%s;" % theme.TEXT_MUTED)
        video_card.add(self.video_label, 1)
        left_lay.addWidget(video_card, 1)

        timeline_card = Card("階段時間軸（各關 T0 出現位置）")
        self.timeline = TimelineBar()
        timeline_card.add(self.timeline)
        left_lay.addWidget(timeline_card)

        console_card = Card("主控台")
        self.console = QtWidgets.QPlainTextEdit()
        self.console.setObjectName("Console")
        self.console.setReadOnly(True)
        self.console.setMaximumBlockCount(3000)   # 上限保護，長影片 log 上萬行不吃光記憶體
        self.console.setMinimumHeight(130)
        console_card.add(self.console, 1)
        left_lay.addWidget(console_card)
        split.addWidget(left)

        # ── 右側面板 ──
        right = QtWidgets.QWidget()
        right_lay = QtWidgets.QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.setSpacing(10)

        tiles = QtWidgets.QHBoxLayout()
        tiles.setSpacing(10)
        self.stage_tile = StatTile("目前關卡", "—", theme.ACCENT)
        self.score_tile = StatTile("累計分數", "0", theme.OK)
        self.gaze_tile = StatTile("注視次數", "0", theme.WARN)
        for t in (self.stage_tile, self.score_tile, self.gaze_tile):
            tiles.addWidget(t)
        right_lay.addLayout(tiles)

        lamp_card = Card("即時判定")
        self.lamps = {}
        for key, name in [
            ("keyword", "語音觸發窗"),
            ("pointing", "指向偵測 (Pointing)"),
            ("gaze", "視線估計"),
            ("gaze_obj", "注視目標物 (TB)"),
            ("gaze_tester", "看回施測者／機器人 (TH)"),
        ]:
            lamp = StatusLamp(name)
            self.lamps[key] = lamp
            lamp_card.add(lamp)
        self.detail_label = QtWidgets.QLabel("—")
        self.detail_label.setProperty("role", "muted")
        self.detail_label.setWordWrap(True)
        lamp_card.add(self.detail_label)
        right_lay.addWidget(lamp_card)

        table_card = Card("各關 T0 / TB / TH（即時）")
        self.table = QtWidgets.QTableWidget(0, len(TABLE_HEADERS))
        self.table.setHorizontalHeaderLabels(TABLE_HEADERS)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        for c in range(1, len(TABLE_HEADERS)):
            hdr.setSectionResizeMode(c, QtWidgets.QHeaderView.ResizeToContents)
        table_card.add(self.table, 1)
        right_lay.addWidget(table_card, 1)

        split.addWidget(right)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        split.setSizes([760, 460])
        root.addWidget(split, 1)

    # ────────────────────────────────────────────
    # 狀態控制
    # ────────────────────────────────────────────
    def reset(self, video_name=None, max_stage=10,
              queue_index=0, queue_total=0, keep_console=False):
        """切換到新影片時重設畫面。

        queue_index / queue_total：批次分析時顯示「第 N / M 支」。
        keep_console：批次的第二支之後保留主控台內容，
        才能回頭看前面幾支的訊息（每支都清掉會很難追）。
        """
        self._duration = 0.0
        self._marks = {}
        self._max_stage = max_stage
        title = "即時監測"
        if queue_total > 1:
            title += "　[第 %d / %d 支]" % (queue_index, queue_total)
        if video_name:
            title += " ・ %s" % video_name
        self.title.setText(title)
        self.phase_label.setText("待機中")
        self.progress.setValue(0)
        self.progress_text.setText("—")
        if not keep_console:
            self.console.clear()
        else:
            self.console.appendPlainText(
                "\n" + "─" * 46 + "\n▶ 下一支：%s\n" % (video_name or "") + "─" * 46)
        self.timeline.set_duration(0)
        self.timeline.set_marks([])
        self.timeline.set_cursor(0)
        self.video_label.setPixmap(QPixmap())
        self.video_label.setText("尚未開始分析\n\n請先到「影片載入」頁選擇影片")
        self.stage_tile.set_value("—")
        self.score_tile.set_value("0")
        self.gaze_tile.set_value("0")
        for lamp in self.lamps.values():
            lamp.set_state(False, "—")
        self.detail_label.setText("—")
        self._init_table(max_stage)

    def _init_table(self, max_stage):
        self.table.setRowCount(max_stage)
        for i in range(max_stage):
            stage = i + 1
            name = QtWidgets.QTableWidgetItem(STAGE_SHORT.get(stage, "S%d" % stage))
            name.setForeground(QColor(theme.TEXT_MUTED))
            self.table.setItem(i, 0, name)
            for c in range(1, len(TABLE_HEADERS)):
                cell = QtWidgets.QTableWidgetItem("—")
                cell.setTextAlignment(Qt.AlignCenter)
                cell.setForeground(QColor(theme.IDLE))
                self.table.setItem(i, c, cell)

    def set_running(self, running, phase=""):
        self.stop_btn.setEnabled(running)
        if phase:
            self.phase_label.setText(phase)

    def set_phase(self, text):
        self.phase_label.setText(text)

    def _on_stop(self):
        self.stop_btn.setEnabled(False)
        self.phase_label.setText("正在停止…（會保留已分析的部分並寫出報告）")
        self.stop_requested.emit()

    # ────────────────────────────────────────────
    # 資料更新（由 worker signal 呼叫）
    # ────────────────────────────────────────────
    def append_log(self, line):
        self.console.appendPlainText(line)

    def update_frame(self, qimage):
        if qimage is None or qimage.isNull():
            return
        self.video_label.setText("")
        pix = QPixmap.fromImage(qimage)
        self.video_label.setPixmap(
            pix.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def update_progress(self, done, total):
        if total > 0:
            self.progress.setRange(0, total)
            self.progress.setValue(done)
            self.progress_text.setText("%d / %d 幀（%.1f%%）" % (done, total, 100.0 * done / total))
        else:
            # 讀不到總幀數時改用不確定模式，總比顯示錯誤的百分比好
            self.progress.setRange(0, 0)
            self.progress_text.setText("已處理 %d 幀" % done)

    def update_stats(self, s):
        stage = s.get("stage", 0)
        done_flag = s.get("measurement_over")
        self.stage_tile.set_value(
            ("完成" if done_flag else (str(stage) if stage else "—")),
            theme.OK if done_flag else theme.ACCENT)
        self.score_tile.set_value(s.get("total_score", 0))
        self.gaze_tile.set_value(s.get("total_gazing_events", 0))

        self.lamps["keyword"].set_state(
            s.get("in_trigger_window"), "作答中" if s.get("in_trigger_window") else "待機",
            theme.OK)
        self.lamps["pointing"].set_state(
            s.get("pointing_hit"), "偵測到" if s.get("pointing_hit") else "無", theme.OK)

        angles = s.get("gaze_angles")
        if s.get("gaze_ok") and angles:
            self.lamps["gaze"].set_state(True, "P %.1f°  Y %.1f°" % (angles[0], angles[1]), theme.ACCENT)
        else:
            fsm = s.get("fsm_state") or ""
            self.lamps["gaze"].set_state(
                False, "極端轉頭" if fsm == "EXTREME_TURNING" else "未取得")

        self.lamps["gaze_obj"].set_state(
            s.get("gazing_object"), "命中目標" if s.get("gazing_object") else "無", theme.WARN)
        self.lamps["gaze_tester"].set_state(
            s.get("gazing_tester"), "看回人／機器人" if s.get("gazing_tester") else "無",
            theme.ROBOT if stage >= 9 else theme.OK)

        fps = s.get("fps") or 0
        total_frames = s.get("total_frames") or 0
        if self._duration <= 0 and fps > 0 and total_frames > 0:
            self._duration = total_frames / float(fps)
            self.timeline.set_duration(self._duration)
        self.timeline.set_cursor(s.get("time_sec", 0))

        self.detail_label.setText(
            "影片時間 %.1fs　・　牌子讀值 %s　・　%s"
            % (s.get("time_sec", 0), s.get("sign_stage", "—"),
               "已達第 11 關，1-10 量測結束" if done_flag else "量測中"))

        self._update_table(s.get("records") or [], s.get("stage_gazing_counts") or {})

    def _update_table(self, records, _gazing_counts):
        by_stage = {}
        for rec in records:
            stg = rec.get("stage")
            if isinstance(stg, int) and 1 <= stg <= self._max_stage:
                by_stage[stg] = rec          # 同一關若有多筆，以最後一筆為準

        new_marks = False
        for stg, rec in by_stage.items():
            row = stg - 1
            level = _preview_level(rec)
            color = QColor(theme.LEVEL_COLORS.get(level, theme.IDLE))

            def _fmt(value, count):
                if value is None:
                    return "—"
                return "%.1fs" % value + (" ×%d" % count if count > 1 else "")

            cells = [
                "%.1fs" % rec["t0"] if rec.get("t0") is not None else "—",
                _fmt(rec.get("pointing_t"), rec.get("pointing_count", 0)),
                "—" if rec.get("tb_mode") is None else _fmt(rec.get("tb"), rec.get("tb_count", 0)),
                _fmt(rec.get("th"), rec.get("th_count", 0)),
                level,
            ]
            for c, text in enumerate(cells, start=1):
                item = self.table.item(row, c)
                if item is None or item.text() == text:
                    # 內容沒變就不動：update_stats 每幀都會被呼叫，
                    # 無條件 setText 會讓表格每秒重繪 30 次、白白吃掉 UI 執行緒
                    continue
                item.setText(text)
                item.setForeground(color if c == len(cells) else QColor(theme.TEXT_MAIN))
                if c == len(cells):
                    f = QFont()
                    f.setBold(True)
                    item.setFont(f)
            name_item = self.table.item(row, 0)
            if name_item:
                name_item.setForeground(QColor(theme.TEXT_MAIN))

            if rec.get("t0") is not None and stg not in self._marks:
                self._marks[stg] = (rec["t0"], stg, theme.ACCENT)
                new_marks = True

        if new_marks:
            self.timeline.set_marks(sorted(self._marks.values()))
