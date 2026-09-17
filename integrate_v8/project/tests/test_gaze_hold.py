# -*- coding: utf-8 -*-
from modules.gaze_hold import GazeHoldTracker


def test_raw_hit_is_immediately_true():
    tracker = GazeHoldTracker(hold_frames=10)
    obj, tester = tracker.update(True, False)
    assert obj is True
    assert tester is False


def test_hold_bridges_short_dropout_on_same_side():
    """單側短暫漏偵測（YOLO 掉框、射線擦框緣）要被遲滯橋接，這是原本
    GAZE_HIT_HOLD_FRAMES 存在的理由，抽成類別後不能失去這個行為。"""
    tracker = GazeHoldTracker(hold_frames=10)
    tracker.update(True, False)  # 第 0 幀：真的命中物品
    for _ in range(9):
        obj, tester = tracker.update(False, False)  # 接下來 9 幀都沒命中
        assert obj is True, "遲滯窗口內應該仍視為看著物品"
    obj, tester = tracker.update(False, False)  # 第 10 幀：遲滯耗盡
    assert obj is False, "超過 hold_frames 幀沒命中應該真的判定移開視線"


def test_fresh_tester_hit_clears_stale_object_hold():
    """A4 的核心迴歸測試，重現 spec 1-2 節 104-S3 案例：
    物品的遲滯還沒退完時，若這一幀施測者被真正命中，
    不該讓 child_is_gazing_at 跟 child_is_gazing_at_tester 同時為 True
    （原本的獨立計數器設計會讓 TB 和 TH 在同一個時間戳被同時計次）。"""
    tracker = GazeHoldTracker(hold_frames=10)
    tracker.update(True, False)  # 幀 0：命中物品，obj_hold 進入倒數
    for _ in range(5):
        tracker.update(False, False)  # 幀 1-5：物品遲滯還在倒數（尚未耗盡）
    obj, tester = tracker.update(False, True)  # 幀 6：施測者被真正命中
    assert obj is False, "新的施測者命中必須立刻打斷物品端的殘留遲滯"
    assert tester is True


def test_fresh_object_hit_clears_stale_tester_hold():
    """跟上一個測試對稱：施測者側的殘留遲滯，也要被物品側的新命中打斷。"""
    tracker = GazeHoldTracker(hold_frames=10)
    tracker.update(False, True)  # 幀 0：命中施測者
    for _ in range(5):
        tracker.update(False, False)
    obj, tester = tracker.update(True, False)  # 幀 6：物品被真正命中
    assert obj is True
    assert tester is False, "新的物品命中必須立刻打斷施測者端的殘留遲滯"


def test_simultaneous_raw_hits_do_not_favor_either_side():
    """V2 N3 迴歸測試：同一幀 raw_obj 與 raw_tester 都為 True 時
    （Stage 5/6/7 施測者拿著物品的常見情境），兩側都該是全新命中，
    不該有任何一側被另一側覆蓋掉殘留遲滯（原本的實作順序永遠讓
    施測者側勝出）。"""
    tracker = GazeHoldTracker(hold_frames=10)
    obj, tester = tracker.update(True, True)
    assert obj is True
    assert tester is True

    # 兩側都應該各自拿到全新的 hold_frames，而不是其中一側被打斷成 0：
    # 下一幀兩側都沒有原始命中時，兩側都應該還在遲滯窗口內。
    obj, tester = tracker.update(False, False)
    assert obj is True, "物品側不該因為同幀有施測者命中就被打斷遲滯"
    assert tester is True, "施測者側不該因為同幀有物品命中就被打斷遲滯"


def test_reset_clears_both_holds():
    tracker = GazeHoldTracker(hold_frames=10)
    tracker.update(True, False)
    tracker.reset()
    obj, tester = tracker.update(False, False)
    assert obj is False
    assert tester is False
