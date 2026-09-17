# -*- coding: utf-8 -*-
import json
from modules.scoring_engine import ScoringEngine


def _make_empty_cache(tmp_path):
    cache_path = str(tmp_path / "empty_speech_cache.json")
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({"segment_records": [], "noise_events": []}, f, ensure_ascii=False)
    return cache_path


def _no_op_gaze_hit(gaze_result, box):
    return False


def test_stage1_fallback_waits_22_seconds_not_15(tmp_path):
    """C4 短解：15 秒門檻太短，會跟「你看」最慢近 20 秒才出現的關鍵字升級
    路徑打架（見 spec C4）。改成 22 秒，17.9 秒前後、22.1 秒前後個別驗證邊界。"""
    cache_path = _make_empty_cache(tmp_path)
    engine = ScoringEngine(cache_path=cache_path, scoring_version="test", video_path="dummy.mp4")

    # current_stage_enter_time 預設 0.0，Stage 1 一路沒有任何語音事件
    engine.update_frame(
        time_sec=21.9, current_stage=1, is_in_trigger_window=False,
        child_is_pointing_hit=False, child_is_gazing_at=False,
        child_is_gazing_at_tester=False, gaze_result=None, robot_rays=None,
        robot_boxes=[], yolo_boxes=[], is_gazing_at_box_func=_no_op_gaze_hit,
    )
    assert not any(r["stage"] == 1 for r in engine.trigger_event_records), (
        "21.9 秒還沒到新門檻（22 秒），不該有 fallback record"
    )

    engine.update_frame(
        time_sec=22.1, current_stage=1, is_in_trigger_window=False,
        child_is_pointing_hit=False, child_is_gazing_at=False,
        child_is_gazing_at_tester=False, gaze_result=None, robot_rays=None,
        robot_boxes=[], yolo_boxes=[], is_gazing_at_box_func=_no_op_gaze_hit,
    )
    assert any(r["stage"] == 1 for r in engine.trigger_event_records), (
        "過了 22 秒門檻應該要建立 fallback record"
    )
