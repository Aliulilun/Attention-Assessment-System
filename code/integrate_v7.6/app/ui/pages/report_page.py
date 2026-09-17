# -*- coding: utf-8 -*-
"""
頁面三：單次結果報表
======================
讀取 ScoringEngine 寫出的 event_record txt，把純文字報告轉成可以直接投影
的版面：摘要卡、各關明細表、每關得分長條圖、反應等級分布圓環、
Whisper 逐字稿與完整事件流水。

重點是「不重算」——所有數值都直接來自報告檔，介面看到的與交出去的
資料完全一致，避免報告現場出現「投影片跟檔案數字不一樣」的尷尬。
"""

import os
import subprocess
import sys

from qt_compat import QtWidgets, Qt, QColor, QFont
import pipeline_api
import theme
from widgets import Card, StatTile, BarChart, DonutChart, level_badge_text
from report_parser import parse_report, level_counts, LEVEL_ORDER, STAGE_SHORT

DETAIL_HEADERS = [
    "關卡", "關卡名稱", "T0", "計分結束", "指向", "指向次數",
    "TB", "TB 次數", "TH", "TH 次數", "注視次數", "總分", "反應等級",
]


class ReportPage(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._report = None
        self._video_path = ""
        self._build()

    # ────────────────────────────────────────────
    def _build(self):
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(12)

        head = QtWidgets.QHBoxLayout()
        titles = QtWidgets.QVBoxLayout()
        titles.setSpacing(2)
        self.title = QtWidgets.QLabel("單次結果報表")
        self.title.setProperty("role", "h1")
        self.subtitle = QtWidgets.QLabel("尚未載入報告")
        self.subtitle.setProperty("role", "muted")
        titles.addWidget(self.title)
        titles.addWidget(self.subtitle)
        head.addLayout(titles)
        head.addStretch(1)

        self.open_btn = QtWidgets.QPushButton("開啟報告檔…")
        self.open_btn.clicked.connect(self._on_open)
        self.video_btn = QtWidgets.QPushButton("播放分析影片")
        self.video_btn.setEnabled(False)
        self.video_btn.clicked.connect(self._on_play_video)
        self.export_btn = QtWidgets.QPushButton("匯出明細 CSV")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._on_export)
        for b in (self.open_btn, self.video_btn, self.export_btn):
            head.addWidget(b)
        root.addLayout(head)

        # ── 摘要卡 ──
        tiles = QtWidgets.QHBoxLayout()
        tiles.setSpacing(12)
        self.tile_score = StatTile("總分", "—", theme.OK)
        self.tile_gaze = StatTile("注視事件總數", "—", theme.WARN)
        self.tile_hi = StatTile("HI 高度主動", "—", theme.LEVEL_COLORS["HI"])
        self.tile_li = StatTile("LI 視線交替", "—", theme.LEVEL_COLORS["LI"])
        self.tile_f = StatTile("F 無反應", "—", theme.LEVEL_COLORS["F"])
        for t in (self.tile_score, self.tile_gaze, self.tile_hi, self.tile_li, self.tile_f):
            tiles.addWidget(t)
        root.addLayout(tiles)

        # ── 分頁 ──
        tabs = QtWidgets.QTabWidget()
        tabs.addTab(self._build_overview_tab(), "總覽")
        tabs.addTab(self._build_detail_tab(), "各關明細")
        tabs.addTab(self._build_transcript_tab(), "語音逐字稿")
        tabs.addTab(self._build_log_tab(), "事件流水")
        root.addWidget(tabs, 1)

    def _build_overview_tab(self):
        page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(page)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(12)

        bar_card = Card("各關得分（0~4 分：TB 1 分 ＋ TH 2 分 ＋ 指向 1 分）")
        self.bar = BarChart(y_max=4)
        bar_card.add(self.bar, 1)
        lay.addWidget(bar_card, 3)

        bottom = QtWidgets.QHBoxLayout()
        bottom.setSpacing(12)

        donut_card = Card("反應等級分布")
        self.donut = DonutChart("關")
        donut_card.add(self.donut, 1)
        bottom.addWidget(donut_card, 1)

        gaze_card = Card("各關注視次數（GazingCount）")
        self.gaze_bar = BarChart(y_max=10)
        gaze_card.add(self.gaze_bar, 1)
        bottom.addWidget(gaze_card, 1)

        lay.addLayout(bottom, 2)

        legend = QtWidgets.QLabel(
            "　".join(level_badge_text(lv) for lv in LEVEL_ORDER))
        legend.setProperty("role", "muted")
        legend.setWordWrap(True)
        lay.addWidget(legend)
        return page

    def _build_detail_tab(self):
        page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(page)
        lay.setContentsMargins(12, 12, 12, 12)
        self.detail_table = QtWidgets.QTableWidget(0, len(DETAIL_HEADERS))
        self.detail_table.setHorizontalHeaderLabels(DETAIL_HEADERS)
        self.detail_table.verticalHeader().setVisible(False)
        self.detail_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.detail_table.setAlternatingRowColors(False)
        hdr = self.detail_table.horizontalHeader()
        hdr.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        lay.addWidget(self.detail_table)
        return page

    def _build_transcript_tab(self):
        page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(page)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(8)
        note = QtWidgets.QLabel(
            "Whisper large-v3 辨識結果。標示關鍵字的段落即為 T0 觸發依據；"
            "含「請勿忽略／短促的聲音」的段落是 Whisper 幻覺，計分時已自動略過。")
        note.setProperty("role", "muted")
        note.setWordWrap(True)
        lay.addWidget(note)
        self.transcript_table = QtWidgets.QTableWidget(0, 3)
        self.transcript_table.setHorizontalHeaderLabels(["時間", "辨識文字", "命中關鍵字"])
        self.transcript_table.verticalHeader().setVisible(False)
        self.transcript_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.transcript_table.horizontalHeader().setSectionResizeMode(
            1, QtWidgets.QHeaderView.Stretch)
        lay.addWidget(self.transcript_table, 1)
        return page

    def _build_log_tab(self):
        page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(page)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(8)
        filt_row = QtWidgets.QHBoxLayout()
        filt_row.addWidget(QtWidgets.QLabel("篩選"))
        self.log_filter = QtWidgets.QLineEdit()
        self.log_filter.setPlaceholderText("輸入關鍵字過濾，例如 TB、TH、Stage change、T0")
        self.log_filter.textChanged.connect(self._apply_log_filter)
        filt_row.addWidget(self.log_filter, 1)
        lay.addLayout(filt_row)
        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setObjectName("Console")
        self.log_view.setReadOnly(True)
        lay.addWidget(self.log_view, 1)
        return page

    # ────────────────────────────────────────────
    # 載入
    # ────────────────────────────────────────────
    def load_report(self, txt_path, video_path="", max_stage=10):
        if not txt_path or not os.path.exists(txt_path):
            QtWidgets.QMessageBox.warning(self, "找不到報告", "報告檔不存在：\n%s" % txt_path)
            return False
        try:
            report = parse_report(txt_path, max_stage=max_stage)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "解析失敗", "無法解析報告檔：\n%s" % e)
            return False

        self._report = report
        # 沒指定影片路徑時，同目錄下的同名 mp4 就是分析輸出影片
        if not video_path:
            guess = os.path.splitext(txt_path)[0] + ".mp4"
            video_path = guess if os.path.exists(guess) else ""
        self._video_path = video_path

        self.subtitle.setText(
            "%s　・　%s　・　計分版本 %s"
            % (os.path.basename(txt_path), report["video"] or "（未記錄來源影片）",
               report["scoring_version"] or "未知"))
        self.video_btn.setEnabled(bool(video_path))
        self.export_btn.setEnabled(True)

        counts = level_counts(report)
        self.tile_score.set_value(report["total_score"])
        self.tile_gaze.set_value(report["total_gazing_events"])
        self.tile_hi.set_value(counts["HI"])
        self.tile_li.set_value(counts["LI"])
        self.tile_f.set_value(counts["F"])

        self._fill_charts(report, counts)
        self._fill_detail(report)
        self._fill_transcript(report)
        self._fill_log(report)
        return True

    def _fill_charts(self, report, counts):
        bar_data, gaze_data = [], []
        max_gaze = 1
        for stage in range(1, report["max_stage"] + 1):
            rec = report["records"].get(stage)
            if rec is None:
                continue
            label = STAGE_SHORT.get(stage, "S%d" % stage)
            total = rec["total"] if rec["total"] is not None else 0
            color = theme.LEVEL_COLORS.get(rec["level"] or "x", theme.IDLE)
            bar_data.append((label, total, color))
            gaze_data.append((label, rec["gazing_count"], theme.ACCENT))
            max_gaze = max(max_gaze, rec["gazing_count"])
        self.bar.set_data(bar_data, y_max=4)
        self.gaze_bar.set_data(gaze_data, y_max=max_gaze)
        self.donut.set_data([(lv, counts[lv], theme.LEVEL_COLORS[lv]) for lv in LEVEL_ORDER])

    def _fill_detail(self, report):
        stages = [s for s in range(1, report["max_stage"] + 1) if s in report["records"]]
        self.detail_table.setRowCount(len(stages))
        for row, stage in enumerate(stages):
            rec = report["records"][stage]

            def _t(v):
                return "%.2fs" % v if v is not None else "—"

            values = [
                str(stage),
                rec["label"],
                _t(rec["t0"]),
                _t(rec["end"]),
                _t(rec["pointing"]),
                str(rec["pointing_count"] or "—"),
                "無此條件" if rec["tb_na"] else _t(rec["tb"]),
                "—" if rec["tb_na"] else str(rec["tb_count"] or "—"),
                _t(rec["th"]),
                str(rec["th_count"] or "—"),
                str(rec["gazing_count"]),
                str(rec["total"]) if rec["total"] is not None else "—",
                rec["level"] or "未偵測",
            ]
            for col, text in enumerate(values):
                item = QtWidgets.QTableWidgetItem(text)
                if col != 1:
                    item.setTextAlignment(Qt.AlignCenter)
                if col == len(values) - 1:
                    item.setForeground(QColor(theme.LEVEL_COLORS.get(rec["level"] or "x", theme.IDLE)))
                    f = QFont()
                    f.setBold(True)
                    item.setFont(f)
                    item.setToolTip(level_badge_text(rec["level"]))
                elif not rec["detected"]:
                    item.setForeground(QColor(theme.IDLE))
                self.detail_table.setItem(row, col, item)
        self.detail_table.resizeColumnsToContents()
        self.detail_table.horizontalHeader().setSectionResizeMode(
            1, QtWidgets.QHeaderView.Stretch)

    def _fill_transcript(self, report):
        rows = report["transcript"]
        self.transcript_table.setRowCount(len(rows))
        for r, seg in enumerate(rows):
            time_item = QtWidgets.QTableWidgetItem("%.2f ~ %.2fs" % (seg["start"], seg["end"]))
            time_item.setTextAlignment(Qt.AlignCenter)
            text_item = QtWidgets.QTableWidgetItem(seg["text"])
            kw_item = QtWidgets.QTableWidgetItem("、".join(seg["keywords"]) or "—")
            kw_item.setTextAlignment(Qt.AlignCenter)
            if seg["keywords"]:
                kw_item.setForeground(QColor(theme.OK))
            if "請勿忽略" in seg["text"] or "短促的聲音" in seg["text"]:
                for it in (time_item, text_item, kw_item):
                    it.setForeground(QColor(theme.IDLE))
                text_item.setToolTip("Whisper 幻覺段落，計分時已略過")
            self.transcript_table.setItem(r, 0, time_item)
            self.transcript_table.setItem(r, 1, text_item)
            self.transcript_table.setItem(r, 2, kw_item)
        self.transcript_table.resizeColumnToContents(0)
        self.transcript_table.resizeColumnToContents(2)

    def _fill_log(self, report):
        self._log_lines = report["event_log"]
        self._apply_log_filter(self.log_filter.text())

    def _apply_log_filter(self, text):
        lines = getattr(self, "_log_lines", [])
        if text.strip():
            key = text.strip().lower()
            lines = [ln for ln in lines if key in ln.lower()]
        self.log_view.setPlainText("\n".join(lines))

    # ────────────────────────────────────────────
    # 動作
    # ────────────────────────────────────────────
    def _on_open(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "開啟分析報告", pipeline_api.default_output_dir(),
            "分析報告 (*.txt);;所有檔案 (*.*)")
        if path:
            self.load_report(path)

    def _on_play_video(self):
        """用系統預設播放器開啟分析輸出影片。

        故意不在介面內嵌播放器：Qt Multimedia 需要系統解碼器支援，
        在不同 Windows 機器上表現不一致，報告現場風險太高。
        """
        if not self._video_path or not os.path.exists(self._video_path):
            return
        try:
            if sys.platform.startswith("win"):
                os.startfile(self._video_path)     # noqa: S606 - Windows 專用
            elif sys.platform == "darwin":
                subprocess.Popen(["open", self._video_path])
            else:
                subprocess.Popen(["xdg-open", self._video_path])
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "無法播放", str(e))

    def _on_export(self):
        if not self._report:
            return
        default_name = "%s_明細.csv" % self._report["name"]
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "匯出各關明細",
            os.path.join(os.path.dirname(self._report["path"]), default_name),
            "CSV (*.csv);;Excel (*.xlsx)")
        if not path:
            return
        rows = []
        for r in range(self.detail_table.rowCount()):
            rows.append([self.detail_table.item(r, c).text() if self.detail_table.item(r, c) else ""
                         for c in range(self.detail_table.columnCount())])
        try:
            import pandas as pd
            df = pd.DataFrame(rows, columns=DETAIL_HEADERS)
            if path.lower().endswith(".xlsx"):
                df.to_excel(path, index=False)
            else:
                df.to_csv(path, index=False, encoding="utf-8-sig")
            QtWidgets.QMessageBox.information(self, "匯出完成", "已儲存至：\n%s" % path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "匯出失敗", str(e))
