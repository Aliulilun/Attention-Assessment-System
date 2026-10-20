# -*- coding: utf-8 -*-
"""P4-1：sign 框過期就不再繪製的回歸測試。

不載入 EasyOCR：用 object.__new__ 造一個只帶繪圖所需屬性的
SignboardTracker，直接驗證 draw_boxes() 的行為，讓測試在沒有 GPU、
沒有模型檔的機器上也能跑，結果完全確定。
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.signboard import SignboardTracker


GREEN = (0, 255, 0)
SIGN_BOX = (300, 200, 60, 50)          # x, y, w, h
CROP = (100, 100, 500, 400)            # 黃色掃描框


def make_tracker(no_detect_frames=0, last_valid_box=SIGN_BOX, stale_frames=25):
    tr = object.__new__(SignboardTracker)
    tr.current_crop_coords = CROP
    tr.last_valid_box = last_valid_box
    tr.no_detect_frames = no_detect_frames
    tr.SIGN_BOX_STALE_FRAMES = stale_frames
    return tr


def blank():
    return np.zeros((600, 800, 3), dtype=np.uint8)


def count_green(frame):
    """純綠像素數（只有 sign 框用 (0,255,0)，黃框是 (0,255,255)）。"""
    b, g, r = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
    return int(np.count_nonzero((g > 200) & (b < 80) & (r < 80)))


class TestSignBoxStaleness:
    def test_fresh_box_is_drawn(self):
        """剛偵測到：綠框照常畫出來。"""
        tr = make_tracker(no_detect_frames=0)
        f = blank()
        tr.draw_boxes(f, 3)
        assert tr.is_sign_box_stale() is False
        assert count_green(f) > 0, "新鮮的 sign 框應該要畫出來"

    def test_stale_box_is_not_drawn(self):
        """久未偵測到：不畫綠框（這是 P4-1 要修的行為）。"""
        tr = make_tracker(no_detect_frames=25)
        f = blank()
        tr.draw_boxes(f, 8)
        assert tr.is_sign_box_stale() is True
        assert count_green(f) == 0, "過期的 sign 框不該再畫，會停在錯誤位置誤導判讀"

    def test_boundary_frame_exactly_at_threshold(self):
        """恰好達到門檻就算過期（>=，不是 >）。"""
        assert make_tracker(no_detect_frames=24).is_sign_box_stale() is False
        assert make_tracker(no_detect_frames=25).is_sign_box_stale() is True

    def test_none_box_is_stale(self):
        """從未偵測到任何牌子時也算過期，且不應崩潰。"""
        tr = make_tracker(no_detect_frames=0, last_valid_box=None)
        f = blank()
        tr.draw_boxes(f, 1)
        assert tr.is_sign_box_stale() is True
        assert count_green(f) == 0

    def test_scan_box_always_drawn(self):
        """黃色掃描框不受影響——過期與否都要畫，否則看不出系統在找哪裡。"""
        for n in (0, 100):
            f = blank()
            make_tracker(no_detect_frames=n).draw_boxes(f, 5)
            b, g, r = f[:, :, 0], f[:, :, 1], f[:, :, 2]
            yellow = int(np.count_nonzero((g > 200) & (r > 200) & (b < 80)))
            assert yellow > 0, f"no_detect_frames={n} 時黃色掃描框消失了"

    def test_stale_state_is_labelled(self):
        """過期時要在畫面上留下標示，不能無聲消失。"""
        tr = make_tracker(no_detect_frames=99)
        f = blank()
        tr.draw_boxes(f, 8)
        # 標示用青色 (0,200,200)，與黃框 (0,255,255)、綠框 (0,255,0) 都不同
        b, g, r = f[:, :, 0], f[:, :, 1], f[:, :, 2]
        cyan = int(np.count_nonzero((g > 150) & (g < 230) & (r > 150) & (r < 230) & (b < 80)))
        assert cyan > 0, "過期時應標示 Sign:N?，讓判讀者知道牌子定位不到"

    def test_draw_does_not_mutate_state(self):
        """繪圖是唯讀的：不得改動 last_valid_box 或計數器。"""
        tr = make_tracker(no_detect_frames=99)
        before = (tr.last_valid_box, tr.no_detect_frames)
        tr.draw_boxes(blank(), 8)
        assert (tr.last_valid_box, tr.no_detect_frames) == before


class TestStaleStateCleared:
    """P4-2：久未偵測到牌子時，主動清掉殘留的追蹤狀態。"""

    @staticmethod
    def _tracker(no_detect_frames):
        """造一個只帶過期清理所需欄位的 tracker（不載入 EasyOCR／KCF）。"""
        tr = object.__new__(SignboardTracker)
        tr.no_detect_frames = no_detect_frames
        tr.SIGN_BOX_STALE_FRAMES = 25
        tr.last_valid_box = SIGN_BOX
        tr.tracker = object()          # 假裝 KCF 還活著
        tr._appear_snapshot = object()
        tr._low_sim_streak = 2
        return tr

    def test_stale_clears_box_and_tracker(self):
        """達門檻：box、KCF tracker、外觀快照、低相似連續計數全部清掉。"""
        tr = self._tracker(25)
        assert tr._expire_stale_sign_state() is True
        assert tr.last_valid_box is None
        assert tr.tracker is None
        assert tr._appear_snapshot is None
        assert tr._low_sim_streak == 0

    def test_fresh_keeps_everything(self):
        """未達門檻：一律不動，避免誤清掉正在正常追蹤的狀態。"""
        tr = self._tracker(24)
        assert tr._expire_stale_sign_state() is False
        assert tr.last_valid_box == SIGN_BOX
        assert tr.tracker is not None
        assert tr._appear_snapshot is not None
        assert tr._low_sim_streak == 2

    def test_boundary_is_inclusive(self):
        """門檻是 >=：24 不清、25 清。"""
        assert self._tracker(24)._expire_stale_sign_state() is False
        assert self._tracker(25)._expire_stale_sign_state() is True

    def test_clearing_is_idempotent(self):
        """已清過就回報沒動作，重複呼叫不崩潰、不重複清。"""
        tr = self._tracker(99)
        assert tr._expire_stale_sign_state() is True
        for _ in range(5):
            assert tr._expire_stale_sign_state() is False
        assert tr.last_valid_box is None

    def test_called_from_process(self):
        """守住這個方法真的被 process() 呼叫——抽成方法後最容易漏接的就是這一步。"""
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "modules", "signboard.py")
        with open(path, encoding="utf-8") as f:
            src = f.read()
        assert "self._expire_stale_sign_state()" in src, "process() 沒有呼叫過期清理"

    def test_recovery_path_does_not_need_old_box(self):
        """復原不依賴舊值：偵測成功時 last_valid_box 由 OCR bbox 重建。

        若哪天有人把成功路徑改成依賴舊 box，清除機制就會造成永久失聯，
        用原始碼檢查守住這個前提。
        """
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "modules", "signboard.py")
        with open(path, encoding="utf-8") as f:
            src = f.read()
        assert "max(0, orig_x + crop_x1 - pad_box)" in src, \
            "主偵測成功路徑不再從 OCR bbox 重建 last_valid_box"


class TestDetectionPathUntouched:
    def test_no_detect_frames_semantics_unchanged(self):
        """守住前提：no_detect_frames 每幀遞增、偵測成功才歸零。

        P4-1 的正確性完全建立在這個語意上。用原始碼檢查守住它，
        避免日後有人改成別的意思而讓過期判定悄悄失效。
        """
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "modules", "signboard.py")
        with open(path, encoding="utf-8") as f:
            src = f.read()
        assert "self.no_detect_frames += 1" in src, "遞增邏輯消失了"
        assert src.count("self.no_detect_frames = 0") >= 2, "歸零點少於預期"

    def test_draw_boxes_is_only_caller_of_stale_check(self):
        """過期判定只能用在繪圖，不得滲進判定路徑。"""
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "modules", "signboard.py")
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
        callers = [i + 1 for i, ln in enumerate(lines)
                   if "is_sign_box_stale()" in ln and "def " not in ln]
        assert len(callers) == 1, (
            f"is_sign_box_stale() 被呼叫了 {len(callers)} 次（行 {callers}）；"
            "它只應該在 draw_boxes 內使用，滲進判定路徑會改變偵測行為"
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
