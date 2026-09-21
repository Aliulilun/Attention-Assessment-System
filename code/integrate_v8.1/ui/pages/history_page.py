# -*- coding: utf-8 -*-
"""
頁面四：歷史統計與比較
========================
掃描輸出資料夾內所有分析報告，做跨受試者的橫向比較。

分組（ASD / TD）由使用者在表格中指定，存在 ui/groups.json。
刻意不從檔名猜測分組——影片編號沒有編碼組別資訊，猜錯會直接讓
比較圖表講出錯誤的結論，這在研究情境是不能接受的。
"""

import json
import os

from qt_compat import QtWidgets, Qt, Signal, QColor, QFont
import theme
import pipeline_api
from widgets import Card, StatTile, GroupedBarChart, BarChart
from report_parser import scan_reports, level_counts, LEVEL_ORDER, STAGE_SHORT

GROUPS = ["未分組", "ASD", "TD"]
GROUP_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "groups.json")

BASE_HEADERS = ["影片編號", "分組", "總分", "注視事件", "HI", "LI", "HR", "LR", "F"]


class HistoryPage(QtWidgets.QWidget):

    open_report_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._reports = []
        self._groups = self._load_groups()
        self._max_stage = 10
        self._updating = False
        self._build()
        self.refresh(pipeline_api.default_output_dir())

    # ────────────────────────────────────────────
    def _build(self):
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(12)

        head = QtWidgets.QHBoxLayout()
        titles = QtWidgets.QVBoxLayout()
        titles.setSpacing(2)
        title = QtWidgets.QLabel("歷史統計與比較")
        title.setProperty("role", "h1")
        self.subtitle = QtWidgets.QLabel("尚未載入任何報告")
        self.subtitle.setProperty("role", "muted")
        titles.addWidget(title)
        titles.addWidget(self.subtitle)
        head.addLayout(titles)
        head.addStretch(1)

        self.folder_btn = QtWidgets.QPushButton("更換資料夾…")
        self.folder_btn.clicked.connect(self._on_change_folder)
        self.reload_btn = QtWidgets.QPushButton("重新掃描")
        self.reload_btn.clicked.connect(lambda: self.refresh(self._folder))
        self.export_btn = QtWidgets.QPushButton("匯出總表 Excel")
        self.export_btn.clicked.connect(self._on_export)
        for b in (self.folder_btn, self.reload_btn, self.export_btn):
            head.addWidget(b)
        root.addLayout(head)

        tiles = QtWidgets.QHBoxLayout()
        tiles.setSpacing(12)
        self.tile_n = StatTile("受試者份數", "0", theme.ACCENT)
        self.tile_asd = StatTile("ASD 組", "0", theme.GROUP_COLORS["ASD"])
        self.tile_td = StatTile("TD 組", "0", theme.GROUP_COLORS["TD"])
        self.tile_avg = StatTile("平均總分", "—", theme.OK)
        for t in (self.tile_n, self.tile_asd, self.tile_td, self.tile_avg):
            tiles.addWidget(t)
        tiles.addStretch(1)
        root.addLayout(tiles)

        split = QtWidgets.QSplitter(Qt.Vertical)

        table_card = Card("各受試者結果（在「分組」欄選擇 ASD / TD 後圖表會自動更新）")
        self.table = QtWidgets.QTableWidget(0, len(BASE_HEADERS))
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.itemDoubleClicked.connect(self._on_row_activated)
        table_card.add(self.table, 1)
        split.addWidget(table_card)

        charts = QtWidgets.QWidget()
        charts_lay = QtWidgets.QHBoxLayout(charts)
        charts_lay.setContentsMargins(0, 0, 0, 0)
        charts_lay.setSpacing(12)

        cmp_card = Card("各關平均得分：ASD vs TD")
        self.cmp_chart = GroupedBarChart(y_max=4)
        cmp_card.add(self.cmp_chart, 1)
        charts_lay.addWidget(cmp_card, 3)

        total_card = Card("各受試者總分")
        self.total_chart = BarChart(y_max=40)
        total_card.add(self.total_chart, 1)
        charts_lay.addWidget(total_card, 2)

        split.addWidget(charts)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        root.addWidget(split, 1)

        hint = QtWidgets.QLabel(
            "提示：雙擊任一列可直接開啟該受試者的單次結果報表。"
            "分組設定會存到 ui/groups.json，下次開啟自動帶入。")
        hint.setProperty("role", "muted")
        root.addWidget(hint)

    # ────────────────────────────────────────────
    # 分組儲存
    # ────────────────────────────────────────────
    def _load_groups(self):
        try:
            with open(GROUP_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_groups(self):
        try:
            with open(GROUP_FILE, "w", encoding="utf-8") as f:
                json.dump(self._groups, f, ensure_ascii=False, indent=2)
        except Exception:
            pass   # 分組只是輔助資訊，寫檔失敗不該打斷使用

    # ────────────────────────────────────────────
    # 載入與更新
    # ────────────────────────────────────────────
    def refresh(self, folder, max_stage=None):
        self._folder = folder
        if max_stage:
            self._max_stage = max_stage
        self._reports = scan_reports(folder, max_stage=self._max_stage)
        self.subtitle.setText("%s　・　找到 %d 份報告" % (folder, len(self._reports)))
        self._fill_table()
        self._update_charts()

    def _fill_table(self):
        self._updating = True
        stage_cols = [STAGE_SHORT.get(s, "S%d" % s) for s in range(1, self._max_stage + 1)]
        headers = BASE_HEADERS + stage_cols
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setRowCount(len(self._reports))

        for row, rep in enumerate(self._reports):
            counts = level_counts(rep)
            name_item = QtWidgets.QTableWidgetItem(rep["name"])
            name_item.setData(Qt.UserRole, rep["path"])
            self.table.setItem(row, 0, name_item)

            combo = QtWidgets.QComboBox()
            combo.addItems(GROUPS)
            combo.setCurrentText(self._groups.get(rep["name"], "未分組"))
            combo.currentTextChanged.connect(
                lambda text, key=rep["name"]: self._on_group_changed(key, text))
            self.table.setCellWidget(row, 1, combo)

            plain = [str(rep["total_score"]), str(rep["total_gazing_events"])]
            plain += [str(counts[lv]) for lv in LEVEL_ORDER]
            for offset, text in enumerate(plain):
                item = QtWidgets.QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(row, 2 + offset, item)

            for i, stage in enumerate(range(1, self._max_stage + 1)):
                rec = rep["records"].get(stage)
                level = (rec or {}).get("level") or "—"
                item = QtWidgets.QTableWidgetItem(level)
                item.setTextAlignment(Qt.AlignCenter)
                item.setForeground(QColor(theme.LEVEL_COLORS.get(level, theme.IDLE)))
                f = QFont()
                f.setBold(True)
                item.setFont(f)
                if rec and rec.get("total") is not None:
                    item.setToolTip("第 %d 關　總分 %d" % (stage, rec["total"]))
                self.table.setItem(row, len(BASE_HEADERS) + i, item)

        self.table.resizeColumnsToContents()
        self._updating = False

    def _on_group_changed(self, name, group):
        if self._updating:
            return
        if group == "未分組":
            self._groups.pop(name, None)
        else:
            self._groups[name] = group
        self._save_groups()
        self._update_charts()

    def _update_charts(self):
        n = len(self._reports)
        self.tile_n.set_value(n)
        asd = [r for r in self._reports if self._groups.get(r["name"]) == "ASD"]
        td = [r for r in self._reports if self._groups.get(r["name"]) == "TD"]
        self.tile_asd.set_value(len(asd))
        self.tile_td.set_value(len(td))
        self.tile_avg.set_value(
            "%.1f" % (sum(r["total_score"] for r in self._reports) / n) if n else "—")

        labels = [STAGE_SHORT.get(s, "S%d" % s) for s in range(1, self._max_stage + 1)]

        def _avg_by_stage(reports):
            out = []
            for stage in range(1, self._max_stage + 1):
                vals = [(r["records"].get(stage) or {}).get("total") for r in reports]
                vals = [v for v in vals if v is not None]
                out.append(sum(vals) / float(len(vals)) if vals else 0.0)
            return out

        series = []
        if asd:
            series.append(("ASD (n=%d)" % len(asd), _avg_by_stage(asd), theme.GROUP_COLORS["ASD"]))
        if td:
            series.append(("TD (n=%d)" % len(td), _avg_by_stage(td), theme.GROUP_COLORS["TD"]))
        if not series and self._reports:
            # 還沒分組時仍給一條「全體平均」，圖表不會是空的
            series.append(("全體 (n=%d)" % n, _avg_by_stage(self._reports), theme.ACCENT))
        self.cmp_chart.set_data(labels, series, y_max=4)

        max_total = max([r["total_score"] for r in self._reports] + [1])
        self.total_chart.set_data(
            [(r["name"], r["total_score"],
              theme.GROUP_COLORS.get(self._groups.get(r["name"], "未分組"), theme.ACCENT))
             for r in self._reports],
            y_max=max(max_total, 4))

    # ────────────────────────────────────────────
    # 動作
    # ────────────────────────────────────────────
    def _on_change_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "選擇報告資料夾", self._folder)
        if folder:
            self.refresh(folder)

    def _on_row_activated(self, item):
        row = item.row()
        name_item = self.table.item(row, 0)
        if name_item:
            path = name_item.data(Qt.UserRole)
            if path:
                self.open_report_requested.emit(path)

    def _on_export(self):
        if not self._reports:
            QtWidgets.QMessageBox.information(self, "沒有資料", "目前沒有可匯出的報告。")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "匯出總表", os.path.join(self._folder, "分析總表.xlsx"),
            "Excel (*.xlsx);;CSV (*.csv)")
        if not path:
            return

        headers = ["影片編號", "分組", "總分", "注視事件"] + [
            "%s_%s" % (lv, "關卡數") for lv in LEVEL_ORDER]
        for stage in range(1, self._max_stage + 1):
            headers += ["Stage%d_T0" % stage, "Stage%d_TB" % stage, "Stage%d_TH" % stage,
                        "Stage%d_指向次數" % stage, "Stage%d_總分" % stage,
                        "Stage%d_反應等級" % stage]

        rows = []
        for rep in self._reports:
            counts = level_counts(rep)
            row = [rep["name"], self._groups.get(rep["name"], "未分組"),
                   rep["total_score"], rep["total_gazing_events"]]
            row += [counts[lv] for lv in LEVEL_ORDER]
            for stage in range(1, self._max_stage + 1):
                rec = rep["records"].get(stage) or {}
                row += [
                    rec.get("t0"), rec.get("tb"), rec.get("th"),
                    rec.get("pointing_count", 0),
                    rec.get("total"), rec.get("level") or "未偵測",
                ]
            rows.append(row)

        try:
            import pandas as pd
            df = pd.DataFrame(rows, columns=headers)
            if path.lower().endswith(".csv"):
                df.to_csv(path, index=False, encoding="utf-8-sig")
            else:
                df.to_excel(path, index=False)
            QtWidgets.QMessageBox.information(self, "匯出完成", "已儲存至：\n%s" % path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "匯出失敗", str(e))
