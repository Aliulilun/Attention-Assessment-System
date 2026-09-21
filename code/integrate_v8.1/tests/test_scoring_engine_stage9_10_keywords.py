# -*- coding: utf-8 -*-
import json
from modules.scoring_engine import ScoringEngine


def _make_cache(tmp_path, segment_records):
    cache_path = str(tmp_path / "speech_cache.json")
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({"segment_records": segment_records, "noise_events": []}, f, ensure_ascii=False)
    return cache_path


def _segment(text, start, keywords=None):
    return {
        "text": text,
        "trigger_events": [{
            "id": text + "-" + str(start),
            "event_type": "speech",
            "keywords": keywords or [],
            "start": start,
            "end": start + 1.0,
            "trigger_window": [start, start + 3.0],
        }],
    }


def test_stage9_t0_ignores_bare_hua_but_catches_specific_phrase(tmp_path):
    """C2 方向1：Stage 9 的 T0 不該被「這張圖畫得很漂亮」這種含裸字「畫」
    的無關句子誤觸發，只有「畫一幅／畫畫／畫好了」這類明確片語才算數。"""
    cache_path = _make_cache(tmp_path, [
        _segment("這張圖畫得很漂亮喔", start=5.0),   # 含裸字「畫」，不該被當成 Stage 9 起點
        _segment("我們來畫一幅畫吧", start=50.0),     # 明確片語，應該被當成 Stage 9 起點
    ])
    engine = ScoringEngine(cache_path=cache_path, scoring_version="test", video_path="dummy.mp4")
    assert engine.stage_start_times.get(9) == 50.0, (
        "Stage 9 的 T0 應該取自「畫一幅」這句，不是提早被裸字「畫」誤觸發在 5.0s"
    )


def test_stage10_t0_ignores_bare_san_but_catches_fireworks(tmp_path):
    """C2 方向1：Stage 10 的 T0 不該被「三個蘋果」這種含裸字「三」的無關
    句子誤觸發，移除裸的「三」與「3」，只留「煙火/烟火/321/三二一」。"""
    cache_path = _make_cache(tmp_path, [
        _segment("桌上有三個蘋果", start=10.0),  # 含裸字「三」，不該被當成 Stage 10 起點
        _segment("煙火要來囉準備看", start=60.0),  # 明確詞彙，應該被當成 Stage 10 起點
    ])
    engine = ScoringEngine(cache_path=cache_path, scoring_version="test", video_path="dummy.mp4")
    assert engine.stage_start_times.get(10) == 60.0, (
        "Stage 10 的 T0 應該取自「煙火」這句，不是提早被裸字「三」誤觸發在 10.0s"
    )
