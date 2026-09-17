# -*- coding: utf-8 -*-
"""
頁面一：影片載入與參數設定
============================
報告時的第一站。左邊挑影片、右邊調參數，右下角按「開始分析」。

設計取捨：只暴露「改了真的會生效」的參數。
YOLO 各關的信心值（conf）寫死在 modules/models_manager.py 的 detect_objects
裡，而且各關數值是實測調出來的（例如 Stage 11-12 用 0.15），做成滑桿只會
讓人誤以為可以隨手改，因此不放進介面。這裡放的每一項都確實接到管線上。
"""

import os

from qt_compat import QtWidgets, Qt, Signal
import pipeline_api
import theme
from widgets import Card, StatTile


class SetupPage(QtWidgets.QWidget):

    # 🌟 改為傳清單：支援一次勾選多支影片批次分析
    start_requested = Signal(list, dict, dict)  # ([影片路徑...], settings, config)
    view_report_requested = Signal(str)         # 既有報告 txt 路徑

    def __init__(self, parent=None):
        super().__init__(parent)
        self._config = {}
        self._refreshing = False   # refresh_video_list 期間抑制 itemChanged 回呼
        self._build()
        self.reload_config()
        self.refresh_video_list()

    # ────────────────────────────────────────────
    # 版面
    # ────────────────────────────────────────────
    def _build(self):
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(14)

        title = QtWidgets.QLabel("影片載入與參數設定")
        title.setProperty("role", "h1")
        subtitle = QtWidgets.QLabel(
            "選擇一支受試影片，確認分析參數後開始。系統會先做語音辨識，再載入視覺模型逐幀分析。")
        subtitle.setProperty("role", "muted")
        root.addWidget(title)
        root.addWidget(subtitle)

        # ── 硬體狀態列 ──
        hw_row = QtWidgets.QHBoxLayout()
        hw_row.setSpacing(12)
        has_cuda, desc = pipeline_api.describe_gpu()
        self.gpu_tile = StatTile("推論裝置", "GPU" if has_cuda else "CPU",
                                 theme.OK if has_cuda else theme.WARN)
        self.gpu_tile.setToolTip(desc)
        self.video_count_tile = StatTile("待分析影片", "0")
        self.done_count_tile = StatTile("已有結果", "0", theme.ACCENT)
        for t in (self.gpu_tile, self.video_count_tile, self.done_count_tile):
            hw_row.addWidget(t)
        hw_row.addStretch(1)
        self.gpu_desc = QtWidgets.QLabel(desc)
        self.gpu_desc.setProperty("role", "muted")
        self.gpu_desc.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.gpu_desc.setWordWrap(True)
        self.gpu_desc.setMaximumWidth(360)
        hw_row.addWidget(self.gpu_desc)
        root.addLayout(hw_row)

        # ── 主體：左影片清單 / 右參數 ──
        split = QtWidgets.QHBoxLayout()
        split.setSpacing(14)
        split.addWidget(self._build_video_card(), 3)
        split.addWidget(self._build_param_card(), 2)
        root.addLayout(split, 1)

        # ── 底部動作列 ──
        bottom = QtWidgets.QHBoxLayout()
        self.hint = QtWidgets.QLabel("")
        self.hint.setProperty("role", "muted")
        bottom.addWidget(self.hint, 1)

        self.view_btn = QtWidgets.QPushButton("查看既有結果")
        self.view_btn.setEnabled(False)
        self.view_btn.clicked.connect(self._on_view_existing)
        bottom.addWidget(self.view_btn)

        self.start_btn = QtWidgets.QPushButton("開始分析")
        self.start_btn.setProperty("role", "primary")
        self.start_btn.setEnabled(False)
        self.start_btn.clicked.connect(self._on_start)
        bottom.addWidget(self.start_btn)
        root.addLayout(bottom)

    def _build_video_card(self):
        card = Card("影片來源")
        row = QtWidgets.QHBoxLayout()
        self.dir_edit = QtWidgets.QLineEdit(pipeline_api.default_video_dir())
        self.dir_edit.setReadOnly(True)
        browse = QtWidgets.QPushButton("更換資料夾…")
        browse.clicked.connect(self._on_browse_dir)
        pick = QtWidgets.QPushButton("選單一影片…")
        pick.clicked.connect(self._on_pick_file)
        row.addWidget(self.dir_edit, 1)
        row.addWidget(browse)
        row.addWidget(pick)
        card.body().addLayout(row)

        # 🌟 主控全選框：放在清單「上方」，位置對應到下面每一列的勾選框，
        #    一眼就能看出「這一欄是拿來勾選的」。三態顯示：
        #    全勾 / 全不勾 / 部分勾選（indeterminate）。
        master_row = QtWidgets.QHBoxLayout()
        master_row.setContentsMargins(4, 2, 4, 2)
        master_row.setSpacing(8)
        self.master_check = QtWidgets.QCheckBox("全選")
        self.master_check.setTristate(True)
        self.master_check.setToolTip("勾選 / 取消勾選清單中所有影片")
        self.master_check.clicked.connect(self._on_master_clicked)
        self.pick_hint = QtWidgets.QLabel("")
        self.pick_hint.setProperty("role", "muted")
        btn_todo = QtWidgets.QPushButton("只選未分析")
        btn_todo.setToolTip("勾選所有還沒有分析結果的影片，適合一次補跑整個資料夾")
        btn_todo.clicked.connect(self._check_unfinished)
        master_row.addWidget(self.master_check)
        master_row.addWidget(self.pick_hint, 1)
        master_row.addWidget(btn_todo)
        card.body().addLayout(master_row)

        self.video_list = QtWidgets.QListWidget()
        self.video_list.currentItemChanged.connect(self._on_selection_changed)
        # 🌟 勾選框用來挑批次要跑的影片；itemChanged 只在使用者手動勾選時才更新按鈕
        self.video_list.itemChanged.connect(self._on_item_checked)
        card.add(self.video_list, 1)

        out_row = QtWidgets.QHBoxLayout()
        out_label = QtWidgets.QLabel("輸出目錄")
        out_label.setProperty("role", "muted")
        self.out_edit = QtWidgets.QLineEdit(pipeline_api.default_output_dir())
        self.out_edit.setReadOnly(True)
        out_browse = QtWidgets.QPushButton("更換…")
        out_browse.clicked.connect(self._on_browse_output)
        out_row.addWidget(out_label)
        out_row.addWidget(self.out_edit, 1)
        out_row.addWidget(out_browse)
        card.body().addLayout(out_row)
        return card

    def _build_param_card(self):
        card = Card("分析參數")
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        inner = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(inner)
        form.setLabelAlignment(Qt.AlignLeft)
        form.setSpacing(9)

        self.max_stage_box = QtWidgets.QComboBox()
        self.max_stage_box.addItem("第 1 ~ 10 關（批次版標準流程）", 10)
        self.max_stage_box.addItem("第 1 ~ 7 關（僅真人／道具關卡）", 7)
        self.max_stage_box.addItem("第 1 ~ 14 關（含機器人指物）", 14)
        self.max_stage_box.setToolTip(
            "對應 hurry/main.py 的 ACTIVE_STAGES。\n"
            "第 11 關之後由語音絕對時間軸驅動，批次版預設只量到第 10 關。")
        form.addRow("量測關卡範圍", self.max_stage_box)

        self.yolo_skip = self._spin(1, 10, 1, "每幾幀跑一次 YOLO 物件偵測。\n數值越大越快，但目標框更新會變遲鈍。")
        self.gaze_skip = self._spin(1, 10, 1, "每幾幀跑一次視線估計。\n視線是 TB/TH 判定的核心，建議維持 1。")
        self.ocr_skip = self._spin(1, 30, 5, "每幾幀跑一次階段牌 OCR。\nEasyOCR 是最耗時的一段，調大最能提速；\n過大則換牌反應變慢，可能吃掉語音空窗期。")
        form.addRow("YOLO 跳幀", self.yolo_skip)
        form.addRow("視線跳幀", self.gaze_skip)
        form.addRow("OCR 跳幀", self.ocr_skip)

        sep1 = QtWidgets.QLabel("— 模型設定（寫入 config.yaml）—")
        sep1.setProperty("role", "muted")
        form.addRow(sep1)

        self.device_box = QtWidgets.QComboBox()
        self.device_box.addItem("CUDA（GPU）", "cuda")
        self.device_box.addItem("CPU", "cpu")
        self.device_box.setToolTip("視線估計模型的推論裝置（config.yaml → gaze_estimation.model.device）")
        form.addRow("視線推論裝置", self.device_box)

        self.pitch_offset = QtWidgets.QDoubleSpinBox()
        self.pitch_offset.setRange(-45.0, 45.0)
        self.pitch_offset.setSingleStep(0.5)
        self.pitch_offset.setSuffix(" °")
        self.pitch_offset.setToolTip(
            "視線俯仰角校正量。不同場地的相機高度會造成系統性偏移，\n"
            "此值用來補償（config.yaml → gaze_estimation.calibration.pitch_offset_deg）。")
        form.addRow("視線 Pitch 校正", self.pitch_offset)

        self.balloon_ratio = QtWidgets.QDoubleSpinBox()
        self.balloon_ratio.setRange(0.0, 1.0)
        self.balloon_ratio.setSingleStep(0.01)
        self.balloon_ratio.setDecimals(2)
        self.balloon_ratio.setToolTip(
            "第 5 關氣球的彩度過濾門檻，用來排除白袍衣袖、牆面等誤判。\n"
            "使用白色或透明氣球的場次請設為 0 停用，否則真氣球會被濾掉。")
        form.addRow("氣球彩度門檻", self.balloon_ratio)

        sep2 = QtWidgets.QLabel("— 介面顯示（不影響分析結果）—")
        sep2.setProperty("role", "muted")
        form.addRow(sep2)

        self.preview_fps = self._spin(5, 60, 20, "即時監測畫面的更新上限。\n只影響顯示流暢度，逐幀分析與輸出影片仍是完整的。")
        form.addRow("預覽更新上限 (fps)", self.preview_fps)

        self.preview_width = QtWidgets.QComboBox()
        for wpx in (640, 800, 960, 1280):
            self.preview_width.addItem("%d px" % wpx, wpx)
        self.preview_width.setCurrentIndex(2)
        form.addRow("預覽畫面寬度", self.preview_width)

        scroll.setWidget(inner)
        card.add(scroll, 1)

        btn_row = QtWidgets.QHBoxLayout()
        reset = QtWidgets.QPushButton("還原預設")
        reset.clicked.connect(self._reset_defaults)
        save = QtWidgets.QPushButton("儲存至 config.yaml")
        save.clicked.connect(self._save_config)
        btn_row.addWidget(reset)
        btn_row.addStretch(1)
        btn_row.addWidget(save)
        card.body().addLayout(btn_row)
        return card

    @staticmethod
    def _spin(lo, hi, val, tip=""):
        s = QtWidgets.QSpinBox()
        s.setRange(lo, hi)
        s.setValue(val)
        if tip:
            s.setToolTip(tip)
        return s

    # ────────────────────────────────────────────
    # 資料
    # ────────────────────────────────────────────
    def reload_config(self):
        try:
            self._config = pipeline_api.load_config() or {}
        except Exception:
            self._config = {}
        gz = self._config.get("gaze_estimation", {}) or {}
        device = str((gz.get("model", {}) or {}).get("device", "cuda")).lower()
        self.device_box.setCurrentIndex(1 if device == "cpu" else 0)
        self.pitch_offset.setValue(
            float((gz.get("calibration", {}) or {}).get("pitch_offset_deg", -12.5)))
        self.balloon_ratio.setValue(
            float((self._config.get("object_detection", {}) or {}).get(
                "stage5_min_colorful_ratio", 0.12)))
        try:
            hm = pipeline_api.get_hurry_main()
            self.ocr_skip.setValue(int(hm.OCR_SKIP))
            self.yolo_skip.setValue(int(hm.YOLO_SKIP))
            self.gaze_skip.setValue(int(hm.GAZE_SKIP))
        except Exception:
            pass

    def refresh_video_list(self):
        folder = self.dir_edit.text()
        out_dir = self.out_edit.text()
        # 記住原本勾了哪些，重新整理後盡量還原（例如剛跑完一支後刷新清單）
        previously_checked = set(self.checked_videos())

        self._refreshing = True
        self.video_list.clear()
        videos = pipeline_api.list_videos(folder)
        done = 0
        for path in videos:
            _mp4, txt = pipeline_api.output_paths(path, out_dir)
            finished = os.path.exists(txt)
            done += 1 if finished else 0
            size_mb = os.path.getsize(path) / 1024 ** 2 if os.path.exists(path) else 0
            label = "%s      %.0f MB" % (os.path.basename(path), size_mb)
            if finished:
                label += "      ✔ 已有結果"
            item = QtWidgets.QListWidgetItem(label)
            item.setData(Qt.UserRole, path)
            item.setData(Qt.UserRole + 1, finished)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if path in previously_checked else Qt.Unchecked)
            self.video_list.addItem(item)
        self._refreshing = False

        self.video_count_tile.set_value(len(videos))
        self.done_count_tile.set_value(done)
        if videos:
            self.video_list.setCurrentRow(0)
        else:
            self.hint.setText("此資料夾沒有影片，請更換資料夾或直接選擇單一影片檔。")
            self.start_btn.setEnabled(False)
            self.view_btn.setEnabled(False)
        self._update_start_button()

    def current_video(self):
        item = self.video_list.currentItem()
        return item.data(Qt.UserRole) if item else None

    # ── 批次挑選 ────────────────────────────────
    def checked_videos(self):
        """回傳所有被勾選的影片路徑（依清單順序）。"""
        return [self.video_list.item(i).data(Qt.UserRole)
                for i in range(self.video_list.count())
                if self.video_list.item(i).checkState() == Qt.Checked]

    def videos_to_run(self):
        """實際要送去分析的清單。

        有勾選就跑勾選的；一個都沒勾就跑「目前反白的那一支」——
        單支分析是最常見的操作，不該強迫使用者先去勾一個框。
        """
        checked = self.checked_videos()
        if checked:
            return checked
        current = self.current_video()
        return [current] if current else []

    def _set_all_checked(self, checked):
        self._refreshing = True
        state = Qt.Checked if checked else Qt.Unchecked
        for i in range(self.video_list.count()):
            self.video_list.item(i).setCheckState(state)
        self._refreshing = False
        self._update_start_button()

    def _check_unfinished(self):
        self._refreshing = True
        for i in range(self.video_list.count()):
            item = self.video_list.item(i)
            finished = bool(item.data(Qt.UserRole + 1))
            item.setCheckState(Qt.Unchecked if finished else Qt.Checked)
        self._refreshing = False
        self._update_start_button()

    def _on_item_checked(self, _item):
        if self._refreshing:
            return
        self._update_start_button()

    def _on_master_clicked(self, _checked):
        """點主控框：目前不是全勾就全部勾起來，已經全勾就全部取消。

        用 clicked（不是 stateChanged）才不會被程式碼設定狀態時誤觸發；
        三態核取方塊點擊時會在三個狀態間循環，所以這裡不看它自己的狀態，
        而是看清單目前的實際勾選數，行為才符合直覺。
        """
        total = self.video_list.count()
        checked = len(self.checked_videos())
        self._set_all_checked(checked < total)

    def _sync_master_check(self):
        total = self.video_list.count()
        checked = len(self.checked_videos())
        self.master_check.blockSignals(True)
        if total == 0 or checked == 0:
            self.master_check.setCheckState(Qt.Unchecked)
        elif checked == total:
            self.master_check.setCheckState(Qt.Checked)
        else:
            self.master_check.setCheckState(Qt.PartiallyChecked)
        self.master_check.blockSignals(False)
        self.master_check.setEnabled(total > 0)

    def _update_start_button(self):
        n = len(self.checked_videos())
        total = self.video_list.count()
        if n > 1:
            self.start_btn.setText("開始批次分析（%d 支）" % n)
            self.pick_hint.setText("已勾選 %d / %d 支，會依序跑完" % (n, total))
        elif n == 1:
            self.start_btn.setText("開始分析")
            self.pick_hint.setText("已勾選 1 / %d 支" % total)
        else:
            self.start_btn.setText("開始分析")
            self.pick_hint.setText("未勾選任何影片，將分析目前反白的那一支" if total else "")
        self.start_btn.setEnabled(bool(self.videos_to_run()))
        self._sync_master_check()

    def collect_settings(self):
        return {
            "output_dir": self.out_edit.text(),
            "max_stage": int(self.max_stage_box.currentData()),
            "yolo_skip": self.yolo_skip.value(),
            "gaze_skip": self.gaze_skip.value(),
            "ocr_skip": self.ocr_skip.value(),
            "preview_fps": self.preview_fps.value(),
            "preview_width": int(self.preview_width.currentData()),
        }

    def collect_config(self):
        """把介面上的模型設定疊回 config dict（不寫檔，只在本次執行生效）。"""
        cfg = dict(self._config or {})
        gz = dict(cfg.get("gaze_estimation", {}) or {})
        model = dict(gz.get("model", {}) or {})
        model["device"] = self.device_box.currentData()
        model["use_gpu"] = (self.device_box.currentData() == "cuda")
        gz["model"] = model
        cal = dict(gz.get("calibration", {}) or {})
        cal["pitch_offset_deg"] = float(self.pitch_offset.value())
        gz["calibration"] = cal
        cfg["gaze_estimation"] = gz
        obj = dict(cfg.get("object_detection", {}) or {})
        obj["stage5_min_colorful_ratio"] = float(self.balloon_ratio.value())
        cfg["object_detection"] = obj
        return cfg

    # ────────────────────────────────────────────
    # 事件
    # ────────────────────────────────────────────
    def _on_browse_dir(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "選擇影片資料夾", self.dir_edit.text())
        if folder:
            self.dir_edit.setText(folder)
            self.refresh_video_list()

    def _on_browse_output(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "選擇輸出資料夾", self.out_edit.text())
        if folder:
            self.out_edit.setText(folder)
            self.refresh_video_list()

    def _on_pick_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "選擇影片", self.dir_edit.text(),
            "影片檔 (*.mp4 *.avi *.mov *.mkv);;所有檔案 (*.*)")
        if not path:
            return
        # 直接切換到該檔所在資料夾，並選中它——比只記住單一路徑更好操作
        self.dir_edit.setText(os.path.dirname(path))
        self.refresh_video_list()
        for i in range(self.video_list.count()):
            if self.video_list.item(i).data(Qt.UserRole) == path:
                self.video_list.setCurrentRow(i)
                break

    def _on_selection_changed(self, item, _prev):
        if not item:
            self.view_btn.setEnabled(False)
            self._update_start_button()
            return
        finished = bool(item.data(Qt.UserRole + 1))
        self.view_btn.setEnabled(finished)
        if finished:
            self.hint.setText("這支影片已有分析結果，重新分析會覆蓋原本的輸出。")
        else:
            self.hint.setText("首次分析需先跑 Whisper 語音辨識，可能需要數十秒至數分鐘。")
        self._update_start_button()

    def _on_view_existing(self):
        video = self.current_video()
        if not video:
            return
        _mp4, txt = pipeline_api.output_paths(video, self.out_edit.text())
        if os.path.exists(txt):
            self.view_report_requested.emit(txt)

    def _on_start(self):
        videos = self.videos_to_run()
        if not videos:
            return

        out_dir = self.out_edit.text()
        already = [v for v in videos
                   if os.path.exists(pipeline_api.output_paths(v, out_dir)[1])]
        if already:
            names = "、".join(os.path.basename(v) for v in already[:5])
            if len(already) > 5:
                names += " 等 %d 支" % len(already)
            answer = QtWidgets.QMessageBox.question(
                self, "已有分析結果",
                "%s 已經有分析結果了。\n\n重新分析會覆蓋原本的影片與報告，確定要繼續嗎？\n"
                "（語音辨識會沿用快取，不會重跑 Whisper）" % names,
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No)
            if answer != QtWidgets.QMessageBox.Yes:
                return

        self.start_requested.emit(videos, self.collect_settings(), self.collect_config())

    def _reset_defaults(self):
        self.max_stage_box.setCurrentIndex(0)
        self.yolo_skip.setValue(1)
        self.gaze_skip.setValue(1)
        self.ocr_skip.setValue(5)
        self.preview_fps.setValue(20)
        self.preview_width.setCurrentIndex(2)
        self.reload_config()
        self.hint.setText("已還原為預設參數。")

    def _save_config(self):
        try:
            cfg = self.collect_config()
            pipeline_api.save_config(cfg)
            self._config = cfg
            QtWidgets.QMessageBox.information(
                self, "已儲存",
                "設定已寫入 config.yaml。\n\n"
                "注意：寫入時會重新產生檔案，原本的中文註解不會保留。")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "儲存失敗", str(e))

    def set_running(self, running):
        """分析進行中時鎖住設定，避免中途更動造成前後不一致。"""
        for w in (self.max_stage_box, self.yolo_skip, self.gaze_skip, self.ocr_skip,
                  self.device_box, self.pitch_offset, self.balloon_ratio,
                  self.video_list, self.master_check):
            w.setEnabled(not running)
        if running:
            self.start_btn.setEnabled(False)
        else:
            self._update_start_button()
