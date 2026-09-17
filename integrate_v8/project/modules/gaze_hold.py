# -*- coding: utf-8 -*-
"""
視線命中遲滯追蹤器
==================
把「看物品」與「看施測者/機器人」兩條獨立的命中遲滯（hold）邏輯
包成一個類別，取代原本 main.py / hurry/main.py 裡重複的
gaze_obj_hold / gaze_tester_hold 兩個區域變數。

遲滯的目的：射線擦框緣、YOLO 短暫掉框造成的單幀漏判很常見，
命中後「立即亮起、延遲熄滅」（連續 hold_frames 幀沒命中才熄滅）
可以橋接這種短暫漏判，見 GAZE_HIT_HOLD_FRAMES 原本的註解。

A4 修正（2026-09-15）：原本兩個 hold 各自獨立倒數，互不知道對方，
會出現「物品的殘留遲滯還沒退完 + 這一幀施測者被真正命中」同時成立
的情況，導致 TB 和 TH 在同一個時間戳被背靠背計次（見 spec 1-2 節
104-S3 案例：TH 跟 TB 同一時間戳、+0.0s from TB）。

修法（選項 b：新命中打斷對側殘留）：哪一側這一幀有真正的原始命中，
就立刻把「另一側」的殘留遲滯清空——只清對方的殘留倒數，不影響
自己這一側原本橋接短暫漏偵測的能力。這比「先觸發者優先、鎖死到
清零」更寬容：小朋友快速地「看一眼物品→立刻轉頭看人」時，後者
不會被壓住。

對稱性修正（2026-09-17，問題清單 V2 N3）：原本兩段 if 依序執行，
若同一幀 raw_obj 與 raw_tester 同時為 True（Stage 5/6/7 這類施測者
拿著/靠近物品的關卡本來就常見兩者同時成立），會被寫程式碼的順序
決定成「永遠是施測者側勝出」——物品側的新鮮命中反而被自己後面那段
if 的 `_tester_hold = 0` 覆蓋掉。這不是刻意設計，只是實作順序造成的
偏袒。改成先各自判斷「這一幀是否要打斷對側」，兩段各自都真時互不
打斷，兩側同時保留全新遲滯，不再偏向任何一側。
"""


class GazeHoldTracker:
    def __init__(self, hold_frames: int = 10):
        self.hold_frames = hold_frames
        self._obj_hold = 0
        self._tester_hold = 0

    def update(self, raw_obj: bool, raw_tester: bool):
        """餵入本幀的原始命中結果，回傳套用遲滯後的 (看物品, 看施測者)。"""
        # 只有「這一幀只有單側真正命中」時，才打斷對側的殘留遲滯；
        # 兩側同時命中時互不打斷，避免程式碼順序造成偏袒（見上方註解）。
        if raw_obj and not raw_tester:
            self._tester_hold = 0
        if raw_tester and not raw_obj:
            self._obj_hold = 0

        if raw_obj:
            self._obj_hold = self.hold_frames
        elif self._obj_hold > 0:
            self._obj_hold -= 1

        if raw_tester:
            self._tester_hold = self.hold_frames
        elif self._tester_hold > 0:
            self._tester_hold -= 1

        gazing_at_obj = raw_obj or self._obj_hold > 0
        gazing_at_tester = raw_tester or self._tester_hold > 0
        return gazing_at_obj, gazing_at_tester

    def reset(self):
        """階段切換時呼叫，避免上一關最後的遲滯狀態帶進新階段（A5）。"""
        self._obj_hold = 0
        self._tester_hold = 0
