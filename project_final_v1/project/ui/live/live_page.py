# -*- coding: utf-8 -*-
"""
頁面五：即時體驗
================
展示現場用：參觀者站到 webcam 前，即時看到專題的兩項核心技術——
視線估計與手勢指向——並玩「先看再指」反應小遊戲。

與專題分析完全隔離：模型只在按「開啟鏡頭」後載入，關閉鏡頭、
切換到別的頁面、或開始正式分析時就釋放（RTX 3050 只有 4GB VRAM）。
"""

import os

from qt_compat import QtWidgets, Qt, QPixmap
import theme
from widgets import Card, StatusLamp
from .live_worker import LiveWorker
from .reaction_game import ReactionGame, summarize, STAGES

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GAZE_INFO = ("AI 先找到你的臉，再從雙眼的影像推算你正在看哪個方向（黃色箭頭）。\n"
             "我們用它判斷孩子有沒有看向施測者指的東西，以及會不會再看回大人。")
POINT_INFO = ("AI 找到你的手部 21 個關鍵點；食指伸直、其他手指彎起來，就從指尖射出一條橘色射線。\n"
              "我們用它判斷孩子會不會主動用手指東西給大人看。\n"
              "（這裡是展示用簡化版，專題正式版另加了骨架交叉驗證與防誤判規則。）")
GAME_INFO = ("第一關「看」：星星出現時，轉頭看向星星那一邊。\n"
             "第二關「指」：星星出現時，用食指指向星星。\n"
             "每關 5 回合，看看你的反應有多快！")
STAGE_LABEL = {"look": "看", "point": "指"}


def _fmt(sec):
    return "—" if sec is None else "%.2f 秒" % sec


class LivePage(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker = None
        self.game = ReactionGame()          # 由頁面持有，「本場最佳」跨開關鏡頭保留
        self._analysis_running = False
        self._build()
        self._set_camera_ui(False)

    # ────────────────────────────────────────────
    def _build(self):
        # 全域 QSS 給每個 QWidget 底色 BG_BASE，卡片內的文字會出現深色色塊；
        # 只在本頁覆寫成透明，不動共用的 theme.py。
        self.setStyleSheet('QFrame[role="card"] QLabel { background: transparent; }')
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 18)
        root.setSpacing(12)

        head = QtWidgets.QHBoxLayout()
        titles = QtWidgets.QVBoxLayout()
        titles.setSpacing(2)
        title = QtWidgets.QLabel("即時體驗")
        title.setProperty("role", "h1")
        self.status_label = QtWidgets.QLabel("站到鏡頭前，體驗我們的 AI 怎麼看懂你的視線與手勢")
        self.status_label.setProperty("role", "muted")
        titles.addWidget(title)
        titles.addWidget(self.status_label)
        head.addLayout(titles)
        head.addStretch(1)
        self.gaze_check = QtWidgets.QCheckBox("視線")
        self.point_check = QtWidgets.QCheckBox("指向")
        for c in (self.gaze_check, self.point_check):
            c.setChecked(True)
            c.toggled.connect(self._push_options)
            head.addWidget(c)
        self.fps_label = QtWidgets.QLabel("FPS —")
        self.fps_label.setProperty("role", "muted")
        self.fps_label.setMinimumWidth(70)
        head.addWidget(self.fps_label)
        self.camera_btn = QtWidgets.QPushButton("開啟鏡頭")
        self.camera_btn.setProperty("role", "primary")
        self.camera_btn.clicked.connect(self._toggle_camera)
        head.addWidget(self.camera_btn)
        root.addLayout(head)

        body = QtWidgets.QHBoxLayout()
        body.setSpacing(12)

        video_card = Card()
        self.video_label = QtWidgets.QLabel("按右上角「開啟鏡頭」開始")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(480, 300)
        self.video_label.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        self.video_label.setStyleSheet(
            "background:#0c1016; border-radius:8px; color:%s; font-size:16px;" % theme.TEXT_MUTED)
        video_card.add(self.video_label, 1)
        body.addWidget(video_card, 3)

        self.side_stack = QtWidgets.QStackedWidget()
        self.side_stack.addWidget(self._build_info_panel())
        self.side_stack.addWidget(self._build_game_panel())
        self.side_stack.setMinimumWidth(330)
        self.side_stack.setMaximumWidth(420)
        body.addWidget(self.side_stack, 1)
        root.addLayout(body, 1)

        foot = QtWidgets.QHBoxLayout()
        self.game_btn = QtWidgets.QPushButton("開始小遊戲")
        self.game_btn.setProperty("role", "primary")
        self.game_btn.clicked.connect(self._toggle_game)
        foot.addWidget(self.game_btn)
        self.best_label = QtWidgets.QLabel()
        self.best_label.setStyleSheet("font-size:14px; padding-left:12px;")
        foot.addWidget(self.best_label)
        foot.addStretch(1)
        root.addLayout(foot)
        self._update_best()

    def _info_card(self, title, text):
        card = Card(title)
        lab = QtWidgets.QLabel(text)
        lab.setWordWrap(True)
        lab.setStyleSheet("font-size:13px; line-height:150%;")
        card.add(lab)
        return card

    def _build_info_panel(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)

        gaze_card = self._info_card("視線估計", GAZE_INFO)
        self.gaze_value = QtWidgets.QLabel("—")
        self.gaze_value.setStyleSheet("font-size:22px; font-weight:700; color:%s;" % theme.WARN)
        self.gaze_angles = QtWidgets.QLabel("")
        self.gaze_angles.setProperty("role", "muted")
        gaze_card.add(self.gaze_value)
        gaze_card.add(self.gaze_angles)
        lay.addWidget(gaze_card)

        point_card = self._info_card("手勢指向", POINT_INFO)
        self.point_lamp = StatusLamp("指向偵測")
        point_card.add(self.point_lamp)
        lay.addWidget(point_card)

        lay.addWidget(self._info_card("反應小遊戲", GAME_INFO))
        lay.addStretch(1)
        return w

    def _build_game_panel(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        card = Card("反應秒數")
        self.game_msg = QtWidgets.QLabel("")
        self.game_msg.setWordWrap(True)
        self.game_msg.setStyleSheet("font-size:15px; font-weight:600;")
        card.add(self.game_msg)

        self.table = QtWidgets.QTableWidget(5, 2)
        self.table.setHorizontalHeaderLabels(["第一關：看", "第二關：指"])
        self.table.setVerticalHeaderLabels(["第 %d 回" % (i + 1) for i in range(5)])
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.table.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.table.setFixedHeight(220)
        card.add(self.table)

        self.summary_label = QtWidgets.QLabel("")
        self.summary_label.setWordWrap(True)
        self.summary_label.setStyleSheet("font-size:14px; line-height:160%;")
        card.add(self.summary_label)
        self.result_note = QtWidgets.QLabel("")
        self.result_note.setWordWrap(True)
        self.result_note.setProperty("role", "muted")
        card.add(self.result_note)
        lay.addWidget(card)
        lay.addStretch(1)
        return w

    # ────────────────────────────────────────────
    # 鏡頭控制
    # ────────────────────────────────────────────
    def _toggle_camera(self):
        if self.worker is not None:
            self.stop_camera()
        else:
            self.start_camera()

    def start_camera(self):
        if self.worker is not None or self._analysis_running:
            return
        self.worker = LiveWorker(PROJECT_DIR, self.game, self)
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.status.connect(self._on_status)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(self._on_worker_done)
        self._push_options()
        self._set_camera_ui(True)
        self.video_label.setText("開啟中…")
        self.worker.start()

    def stop_camera(self, wait=False):
        """關閉鏡頭並釋放模型。wait=True 時等執行緒結束（關閉視窗用）。"""
        if self.worker is None:
            return
        self.worker.request_stop()
        if wait:
            # 模型載入中（約 15 秒）才能檢查停止旗標，等太短執行緒會在執行中被銷毀而閃退
            self.worker.wait(30000)

    def set_analysis_running(self, running):
        """正式分析進行中：關閉鏡頭並停用（4GB VRAM 不夠兩邊同時用）。"""
        self._analysis_running = running
        if running:
            self.stop_camera()
            self.status_label.setText("正式分析進行中，即時體驗暫停使用")
        else:
            self.status_label.setText("站到鏡頭前，體驗我們的 AI 怎麼看懂你的視線與手勢")
        self._set_camera_ui(self.worker is not None)

    def _set_camera_ui(self, on):
        self.camera_btn.setText("關閉鏡頭" if on else "開啟鏡頭")
        self.camera_btn.setEnabled(on or not self._analysis_running)
        self.game_btn.setEnabled(on)
        if not on:
            self.game_btn.setText("開始小遊戲")
            self.fps_label.setText("FPS —")

    def _push_options(self):
        if self.worker is not None:
            self.worker.set_options(self.gaze_check.isChecked(), self.point_check.isChecked())

    # ────────────────────────────────────────────
    # 小遊戲
    # ────────────────────────────────────────────
    def _toggle_game(self):
        if self.worker is None:
            return
        if self.game_btn.text() == "結束遊戲":
            self.worker.cancel_game()
            self.side_stack.setCurrentIndex(0)
            self.game_btn.setText("開始小遊戲")
        else:
            self.worker.start_game()
            self.side_stack.setCurrentIndex(1)
            self.game_btn.setText("結束遊戲")

    def _update_game_panel(self, g):
        if g is None or g["phase"] == "IDLE":
            return
        self.game_msg.setText(g["message"] if g["phase"] != "RESULT" else "遊戲結束！")
        for col, stage in enumerate(STAGES):
            times = g["results"][stage]
            for row in range(5):
                text = ""
                if row < len(times):
                    text = "錯過" if times[row] is None else "%.2f 秒" % times[row]
                item = QtWidgets.QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(row, col, item)
        if g["phase"] == "RESULT":
            lines = []
            for stage in STAGES:
                s = g["summary"][stage]
                lines.append("<b>%s</b>　平均 %s　最快 %s　錯過 %d 次"
                             % (STAGE_LABEL[stage], _fmt(s["avg"]), _fmt(s["fastest"]), s["misses"]))
            self.summary_label.setText("<br>".join(lines))
            self.result_note.setText(g["message"])
            self.game_btn.setText("再玩一次")
            self._update_best()
        else:
            self.summary_label.setText("")
            self.result_note.setText("")
        if g["phase"] == "INTRO":
            self._update_best()

    def _update_best(self):
        best = self.game.best
        self.best_label.setText("本場最佳　看：%s　指：%s"
                                % (_fmt(best["look"]), _fmt(best["point"])))

    # ────────────────────────────────────────────
    # worker 訊號
    # ────────────────────────────────────────────
    def _on_frame(self, qimg, state):
        pix = QPixmap.fromImage(qimg).scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(pix)
        self.fps_label.setText("FPS %.0f" % state.get("fps", 0))

        if state.get("face_found"):
            self.gaze_value.setText(state["gaze_text"])
            self.gaze_angles.setText("左右 %+.0f°　上下 %+.0f°" % (state["h_deg"], state["v_deg"]))
        else:
            self.gaze_value.setText("找不到臉" if state.get("person_found") else "請站到鏡頭前")
            self.gaze_angles.setText("")
        self.point_lamp.set_state(state.get("pointing", False),
                                  "指向中" if state.get("pointing") else "未指向",
                                  on_color=theme.ROBOT)
        self._update_game_panel(state.get("game"))

    def _on_status(self, text):
        if text:
            self.video_label.setText(text)

    def _on_failed(self, message):
        self.video_label.setText(message)
        QtWidgets.QMessageBox.warning(self, "即時體驗", message)

    def _on_worker_done(self):
        self.worker = None
        self.game.cancel()
        self.side_stack.setCurrentIndex(0)
        self._set_camera_ui(False)
        if self.video_label.pixmap() is not None and not self.video_label.pixmap().isNull():
            self.video_label.clear()
            self.video_label.setText("鏡頭已關閉")
        self._update_best()
