# -*- coding: utf-8 -*-
"""V2 N5 補測：TB/TH 交替鎖、冷卻時間、_apply_frame_to_record 兩道守門。

這些是問題清單 V2 N5 點名「完全裸奔」的計分核心邏輯——過去這類問題
（1-2 節 104-S3 同幀假 TH、1-5 節整輪系統性 TH 消失）都只能靠重跑
3 小時的影片才發現。這裡直接呼叫純邏輯函式，不需要 GPU、不需要影片。
"""
from modules.scoring_engine import (
    ScoringEngine,
    create_trigger_record,
    TB_COOLDOWN_SEC,
    TH_COOLDOWN_SEC,
)


def _bare_engine():
    """不跑 __init__（會需要語音快取檔案），只借用 _apply_frame_to_record
    這個方法；它只讀寫 record 參數和 self.event_logs。"""
    engine = object.__new__(ScoringEngine)
    engine.event_logs = []
    return engine


def _apply(engine, record, time_sec, gazing_at=False, gazing_at_tester=False,
           pointing_hit=False, current_stage=None):
    if current_stage is None:
        current_stage = record["stage"]
    engine._apply_frame_to_record(
        record, time_sec, current_stage, pointing_hit,
        gazing_at, gazing_at_tester,
        gaze_result=None, robot_boxes=[], is_gazing_at_box_func=lambda *_: False,
    )


def test_tb_then_th_then_tb_counts_each_once():
    """看物 -> 看人 -> 看物，是乾淨的交替，三次都該計次。"""
    engine = _bare_engine()
    record = create_trigger_record("Stage1", 1, t0=0.0, end_time=100.0,
                                    tb_mode="object", th_mode="tester")

    _apply(engine, record, 1.0, gazing_at=True)
    assert record["tb_count"] == 1

    _apply(engine, record, 1.0 + TH_COOLDOWN_SEC + 0.1, gazing_at_tester=True)
    assert record["th_count"] == 1

    _apply(engine, record, 1.0 + TH_COOLDOWN_SEC + TB_COOLDOWN_SEC + 0.2, gazing_at=True)
    assert record["tb_count"] == 2, "先看過人（計過 TH）之後，看回物品必須能再算一次 TB"


def test_staying_on_same_target_only_counts_once():
    """持續看著同一個目標，不管幾幀、幾秒，都只能算一次——
    這是交替鎖存在的理由，不是冷卻時間的責任。"""
    engine = _bare_engine()
    record = create_trigger_record("Stage1", 1, t0=0.0, end_time=100.0,
                                    tb_mode="object", th_mode="tester")

    _apply(engine, record, 1.0, gazing_at=True)
    assert record["tb_count"] == 1

    # 冷卻時間早就過了，但因為沒有先計過 TH，交替鎖必須擋住第二次 TB。
    # 用「移開再看回」製造新的上升邊緣，只有交替鎖能擋住它。
    _apply(engine, record, 1.0 + TB_COOLDOWN_SEC + 5.0, gazing_at=False)
    _apply(engine, record, 1.0 + TB_COOLDOWN_SEC + 5.1, gazing_at=True)
    assert record["tb_count"] == 1, "沒有先看回人，交替鎖必須擋下第二次 TB"


def test_th_blocked_before_first_tb_when_stage_requires_tb():
    """有 TB 目標的階段，第一次 TH 必須先有 TB 才算——這是 th_allowed 的門檻。"""
    engine = _bare_engine()
    record = create_trigger_record("Stage1", 1, t0=0.0, end_time=100.0,
                                    tb_mode="object", th_mode="tester")

    _apply(engine, record, 1.0, gazing_at_tester=True)
    assert record["th"] is None, "還沒看過物品，TH 不該成立"


def test_stage8_th_has_no_tb_gate():
    """Stage 8 沒有 TB 目標（tb_mode=None），TH 不需要先達成 TB 即可計次。"""
    engine = _bare_engine()
    record = create_trigger_record("Stage8", 8, t0=0.0, end_time=10.0,
                                    tb_mode=None, th_mode="tester")

    _apply(engine, record, 1.0, gazing_at_tester=True)
    assert record["th_count"] == 1


def test_cooldown_blocks_rapid_repeated_edge_within_window():
    """交替鎖解開後，同一目標的重複上升邊緣如果太密集（< 冷卻時間），
    仍要被冷卻時間擋下——避免 YOLO/視線抖動造成同一次動作被算成多次。"""
    engine = _bare_engine()
    record = create_trigger_record("Stage1", 1, t0=0.0, end_time=100.0,
                                    tb_mode="object", th_mode="tester")
    record["_last_counted_target"] = "head"  # 假裝剛計過一次 TH，交替鎖對 TB 是開的

    _apply(engine, record, 1.0, gazing_at=True)
    assert record["tb_count"] == 1

    # 移開又立刻看回（上升邊緣成立），但間隔遠小於 TB_COOLDOWN_SEC。
    _apply(engine, record, 1.0 + TB_COOLDOWN_SEC / 2, gazing_at=False)
    _apply(engine, record, 1.0 + TB_COOLDOWN_SEC / 2 + 0.01, gazing_at=True)
    assert record["tb_count"] == 1, "冷卻時間內的重複邊緣不該被算成新的一次"


def test_frame_before_t0_is_ignored():
    """_apply_frame_to_record 的第一道守門：time_sec < record['t0'] 直接忽略。"""
    engine = _bare_engine()
    record = create_trigger_record("Stage1", 1, t0=5.0, end_time=100.0,
                                    tb_mode="object", th_mode="tester")

    _apply(engine, record, 1.0, gazing_at=True)  # 早於 t0
    assert record["tb"] is None
    assert record["closed"] is False


def test_frame_after_end_time_closes_record():
    """時間超過 end_time，record 關閉，之後的幀不再處理。"""
    engine = _bare_engine()
    record = create_trigger_record("Stage1", 1, t0=0.0, end_time=5.0,
                                    tb_mode="object", th_mode="tester")

    _apply(engine, record, 6.0, gazing_at=True)
    assert record["closed"] is True
    assert record["tb"] is None


def test_stage_mismatch_closes_record():
    """主迴圈已經切到別的 stage，但 record 還是舊 stage 時，
    第二道守門要直接關閉它，避免把新關卡的幀算進舊關卡。"""
    engine = _bare_engine()
    record = create_trigger_record("Stage1", 1, t0=0.0, end_time=100.0,
                                    tb_mode="object", th_mode="tester")

    _apply(engine, record, 1.0, gazing_at=True, current_stage=2)
    assert record["closed"] is True
    assert record["tb"] is None, "stage 不匹配時這一幀不該被計入任何判定"


def test_c1_empty_robot_boxes_asks_caller_instead_of_silently_false():
    """C1 修正的迴歸測試：robot_boxes 為空時，不再是 any([]) 恆為 False，
    而是呼叫 is_gazing_at_box_func(gaze_result, None) 詢問呼叫端目前的判定。"""
    engine = _bare_engine()
    record = create_trigger_record("Stage9", 9, t0=0.0, end_time=100.0,
                                    tb_mode="object", th_mode="robot_box")
    record["tb"] = -1.0  # 假裝已經達成 TB，讓 th_allowed 成立

    engine._apply_frame_to_record(
        record, 1.0, 9, False,
        True, True,
        gaze_result="dummy_gaze", robot_boxes=[],
        is_gazing_at_box_func=lambda gaze, box: gaze == "dummy_gaze" and box is None,
    )
    assert record["th_count"] == 1, "空框時應詢問呼叫端而非直接判定 False"
