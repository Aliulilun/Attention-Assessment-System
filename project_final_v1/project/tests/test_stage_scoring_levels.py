# -*- coding: utf-8 -*-
"""V2 N5 補測：stage_scoring.compute_stage_score 的等級判定表。

P1-2 把等級判定從「分數查表」改成明確組合判定，理由是 total=2 有兩種
語意完全不同的來源（見 stage_scoring.py 開頭註解）。這裡窮舉所有
(tb, th, pointing) 組合，確保每一種都對應到文件寫的那個等級，
不再只靠人工檢查程式碼。
"""
from modules.stage_scoring import compute_stage_score, is_far_stage


def _record(tb, th, pt, stage=1, label=""):
    return {
        "tb": 1.0 if tb else None,
        "th": 1.0 if th else None,
        "pointing_t": 1.0 if pt else None,
        "stage": stage,
        "label": label,
    }


def test_all_false_is_f():
    total, level = compute_stage_score(_record(False, False, False))
    assert (total, level) == (0, "F")


def test_pointing_and_th_is_hi_highest_initiator():
    """HI = 指向 AND 看回人——臨床上最高階的主動分享。"""
    total, level = compute_stage_score(_record(tb=False, th=True, pt=True))
    assert level == "HI"
    assert total == 3  # th(2) + pointing(1)


def test_tb_and_th_without_pointing_is_li():
    """LI = 看物 AND 看回人、無指向——視線交替但沒有手勢。"""
    total, level = compute_stage_score(_record(tb=True, th=True, pt=False))
    assert level == "LI"
    assert total == 3  # tb(1) + th(2)


def test_all_three_true_is_hi():
    total, level = compute_stage_score(_record(tb=True, th=True, pt=True))
    assert level == "HI"
    assert total == 4


def test_only_tb_or_only_pointing_without_th_is_lr_on_near_stage():
    """只看物或只有指向、沒看回人：近物關卡歸 LR。"""
    _, level_tb = compute_stage_score(_record(tb=True, th=False, pt=False, label="Near"))
    _, level_pt = compute_stage_score(_record(tb=False, th=False, pt=True, label="Near"))
    assert level_tb == "LR"
    assert level_pt == "LR"


def test_only_tb_or_only_pointing_without_th_is_hr_on_far_stage():
    """遠物關卡（label 含 'Far'）total=1 時歸 HR，不是 LR。"""
    _, level_tb = compute_stage_score(_record(tb=True, th=False, pt=False, label="Pointing - Far"))
    _, level_pt = compute_stage_score(_record(tb=False, th=False, pt=True, label="Pointing - Far"))
    assert level_tb == "HR"
    assert level_pt == "HR"


def test_only_th_without_tb_or_pointing_is_lr():
    """只看回人，沒看物也沒指向——罕見組合，文件寫的是 LR。"""
    total, level = compute_stage_score(_record(tb=False, th=True, pt=False))
    assert level == "LR"
    assert total == 2


def test_stage8_special_rule_pointing_beats_th():
    """Stage 8 特規：有指向 -> HI；只有 TH（沒指向）-> LI；皆無 -> F。
    Stage 8 沒有 TB 目標，tb 欄位在這裡應為 None/False，不影響判定。"""
    _, level_both = compute_stage_score(_record(tb=False, th=True, pt=True, stage=8))
    _, level_th_only = compute_stage_score(_record(tb=False, th=True, pt=False, stage=8))
    _, level_none = compute_stage_score(_record(tb=False, th=False, pt=False, stage=8))
    assert level_both == "HI"
    assert level_th_only == "LI"
    assert level_none == "F"


def test_stage8_pointing_without_th_is_still_hi():
    """Stage 8 特規只看 pointing 是否成立，跟 Stage 1-7/9/10 的
    「pt and th 才是 HI」不同——這正是 P1-2 註解特別點名的分支。"""
    _, level = compute_stage_score(_record(tb=False, th=False, pt=True, stage=8))
    assert level == "HI"


def test_is_far_stage_matches_label_keyword():
    assert is_far_stage("真人指遠物(Pointing - Far)") is True
    assert is_far_stage("真人指近物(Pointing - Near)") is False
    assert is_far_stage(None) is False
    assert is_far_stage("") is False
