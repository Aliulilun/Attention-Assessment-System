# -*- coding: utf-8 -*-
"""summarize/overfit_guard.py 的防過度擬合規則。

重點是那幾道「擋下來」的關卡確實會擋：測試組被改、程式碼沒凍結、
結果檔比凍結舊、變化量落在雜訊內、門檻隨影片大幅變動。
"""
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "summarize"))
import overfit_guard as og  # noqa: E402


# ---------------- 切分完整性 ----------------
def test_split_hash_detects_manual_edit():
    cfg = {"dev": [1, 2, 3], "test": [4, 5]}
    cfg["split_hash"] = og.split_hash(cfg["dev"], cfg["test"])
    assert og.check_split_integrity(cfg)
    cfg["test"].append(3)          # 偷偷把開發組影片搬進測試組
    assert not og.check_split_integrity(cfg)


def test_split_hash_ignores_order():
    assert og.split_hash([3, 1], [5, 4]) == og.split_hash([1, 3], [4, 5])


# ---------------- 雜訊 ----------------
def _c(tb, th, pt):
    return {"TB": tb, "TH": th, "Pointing": pt}


def test_noise_band_is_range_across_repeats():
    runs = [_c((5, 2, 1), (3, 1, 0), (1, 3, 0)),
            _c((5, 3, 1), (3, 1, 0), (1, 2, 0)),
            _c((6, 2, 1), (3, 1, 0), (1, 4, 0))]
    band = og.noise_band(runs)
    assert band["TB"] == (1, 1, 0)
    assert band["Pointing"] == (0, 2, 0)


def test_noise_band_needs_two_runs():
    assert og.noise_band([_c((1, 1, 1), (1, 1, 1), (1, 1, 1))]) is None


@pytest.mark.parametrize("delta,noise,expected", [
    (0, 1, "無變化"),
    (-1, 1, "雜訊內"),      # FP 3→2：過去會被當成改善，實際在雜訊內
    (1, 1, "雜訊內"),
    (-2, 1, "超出雜訊"),
    (-2, 2, "雜訊內"),
])
def test_judge_delta(delta, noise, expected):
    assert og.judge_delta(delta, noise) == expected


# ---------------- 集中度 ----------------
def test_concentration_flags_single_video():
    base = {87: {"Pointing": 3}, 42: {"Pointing": 1}, 90: {"Pointing": 2}}
    cur = {87: {"Pointing": 0}, 42: {"Pointing": 1}, 90: {"Pointing": 2}}
    net, top, share = og.concentration(cur, base, "Pointing")
    assert (net, top, share) == (3, 87, 1.0)


def test_concentration_none_when_no_net_gain():
    base = {1: {"TB": 1}, 2: {"TB": 0}}
    cur = {1: {"TB": 0}, 2: {"TB": 1}}
    net, top, share = og.concentration(cur, base, "TB")
    assert net == 0 and share is None


# ---------------- 留一交叉驗證 ----------------
def test_lovo_stable_threshold():
    table = {v: {0.3: (1, 1, 0), 0.4: (1, 0, 0), 0.5: (0, 0, 1)} for v in (1, 2, 3, 4)}
    best, folds = og.lovo(table)
    assert best == 0.4
    assert all(t == 0.4 for _, t, _ in folds)


def test_lovo_held_out_uses_threshold_picked_from_others():
    # 87 的誤報只在寬門檻出現；42/49 的真指向在嚴門檻會漏
    table = {
        87: {0.2: (0, 4, 0), 0.8: (0, 0, 0)},
        42: {0.2: (1, 0, 0), 0.8: (0, 0, 1)},
        49: {0.2: (1, 0, 0), 0.8: (0, 0, 1)},
    }
    _, folds = og.lovo(table)
    by_held = {h: (t, r) for h, t, r in folds}
    assert by_held[87] == (0.2, (0, 4, 0))    # 沒看過 87 時選寬門檻，87 的誤報全部回來
    for h in (42, 49):
        rest = [v for v in table if v != h]
        assert by_held[h][0] == og.best_threshold(table, rest)


def test_objective_tie_breaks_on_fewer_false_positives():
    table = {1: {0.3: (0, 2, 0), 0.6: (0, 0, 0)}}
    assert og.best_threshold(table, [1]) == 0.6


# ---------------- 指向真相 ----------------
def test_pointing_truth_only_overrides_table():
    manual = {42: {4: {"pt": 0}, 7: {"pt": 1}}, 49: {7: {"pt": 1}}}
    cfg = {"pointing_source": "truth_only",
           "pointing_truth": [{"video": 42, "stage": 4, "pointing": 1}]}
    og.apply_pointing_truth(manual, cfg)
    assert manual[42][4]["pt"] == 1
    assert manual[42][7]["pt"] is None      # 對照表的 Pointing 欄不採用
    assert manual[49][7]["pt"] is None


# ---------------- 程式碼雜湊 ----------------
def _make_repo(tmp_path, body=b"x = 1\n"):
    (tmp_path / "modules").mkdir(parents=True)
    (tmp_path / "hurry").mkdir()
    (tmp_path / "modules" / "a.py").write_bytes(body)
    (tmp_path / "hurry" / "main.py").write_bytes(b"pass\n")
    (tmp_path / "main.py").write_bytes(b"pass\n")
    (tmp_path / "config.yaml").write_bytes(b"k: 1\n")
    return str(tmp_path)


def test_code_hash_ignores_crlf(tmp_path):
    a = _make_repo(tmp_path / "a", b"x = 1\n")
    b = _make_repo(tmp_path / "b", b"x = 1\r\n")
    assert og.code_hash(a)[0] == og.code_hash(b)[0]


def test_code_hash_changes_on_threshold_edit(tmp_path):
    root = _make_repo(tmp_path)
    before = og.code_hash(root)[0]
    (tmp_path / "modules" / "a.py").write_bytes(b"x = 2\n")
    assert og.code_hash(root)[0] != before


# ---------------- test 子命令的擋關 ----------------
class _Args(object):
    def __init__(self, **kw):
        self.__dict__.update(kw)


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    root = _make_repo(tmp_path / "repo")
    monkeypatch.setattr(og, "PROJECT_DIR", root)
    monkeypatch.setattr(og, "CONFIG_PATH", str(tmp_path / "guard.json"))
    cfg = {"created": og._now(), "dev": [1, 2], "test": [9],
           "split_hash": og.split_hash([1, 2], [9]),
           "pointing_truth": [], "freezes": [], "test_runs": [], "history": []}
    og.save_config(cfg)
    txt = tmp_path / "out"
    txt.mkdir()
    return tmp_path, root, txt


def test_test_refuses_without_freeze(sandbox):
    _, _, txt = sandbox
    assert og.cmd_test(_Args(txt_dir=str(txt), stages=og.cwm.STAGES)) == 1


def test_test_refuses_after_code_changed(sandbox):
    tmp, root, txt = sandbox
    og.cmd_freeze(_Args(name="v1", note=""))
    (tmp / "repo" / "modules" / "a.py").write_bytes(b"x = 99\n")   # 凍結後又調門檻
    assert og.cmd_test(_Args(txt_dir=str(txt), stages=og.cwm.STAGES)) == 1


def test_test_refuses_results_older_than_freeze(sandbox):
    _, _, txt = sandbox
    p = txt / "9.txt"
    p.write_text("01. Stage 1 -- x\n    TB = 1.0s\n", encoding="utf-8")
    old = time.time() - 3600
    os.utime(str(p), (old, old))
    og.cmd_freeze(_Args(name="v1", note=""))
    assert og.cmd_test(_Args(txt_dir=str(txt), stages=og.cwm.STAGES)) == 1


def test_status_rejects_tampered_split(sandbox):
    cfg = og.load_config()
    cfg["test"] = [1]
    og.save_config(cfg)
    with pytest.raises(SystemExit):
        og.cmd_status(_Args())


def test_init_refuses_overwrite_without_force(sandbox):
    assert og.cmd_init(_Args(test=[2], dev=[1], xlsx=og.cwm.DEFAULT_XLSX, force=False)) == 1


def test_empty_result_files_are_skipped(tmp_path):
    (tmp_path / "49.txt").write_bytes(b"")          # 中斷的重跑留下的 0 byte 檔
    (tmp_path / "42.txt").write_text("01. Stage 1 -- x\n    TB = 1.0s\n", encoding="utf-8")
    data = og.load_results(str(tmp_path))
    assert list(data) == [42]
