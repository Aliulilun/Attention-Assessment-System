# -*- coding: utf-8 -*-
"""P1-3 指向命中方向穩定性 / P1-4 例外快取歸零 的回歸測試。

這裡刻意不載入 YOLO / MediaPipe 模型：用 object.__new__ 造出一個
只帶必要屬性的 InteractionEngine，直接測 resolve_child_pointing_hit()
與 _is_ray_dir_stable() 的判定邏輯，讓測試在沒有 GPU / 沒有模型檔的
機器上也能跑，且結果完全確定（不依賴影片內容）。
"""
import math
import os
import sys
from collections import deque

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.interaction import InteractionEngine


# ----------------------------------------------------------------------
# 輔助：造一個不載入任何模型的 InteractionEngine
# ----------------------------------------------------------------------
def make_engine(child_vectors, child_origins=None, dir_stable_deg=25.0,
                aim_max_deg=25.0, min_gap_scale=0.25, tip=None, scale=0.0):
    """child_vectors: [(dx, dy), ...] 依時間順序的射線向量。

    tip / scale 是 P3-1 接觸判定用的食指尖座標與手掌尺度。
    預設 scale=0 代表「量不到手掌尺度」，此時接觸判定只看框內與否。
    """
    eng = object.__new__(InteractionEngine)
    eng.POINTING_DIR_STABLE_DEG = dir_stable_deg
    eng.S8_DIR_STABLE_DEG = 25.0
    eng.POINTING_AIM_MAX_DEG = aim_max_deg
    eng.POINTING_MIN_GAP_SCALE = min_gap_scale
    eng.last_pointing_tip = {"Child": tip, "Tester": None}
    eng.last_pointing_scale = {"Child": scale, "Tester": 0.0}
    eng.POINTING_HANDS_APART_MIN = 0.8
    eng.last_hands_too_close = False
    if child_origins is None:
        child_origins = [(100, 500)] * len(child_vectors)
    eng.ray_history = {
        "Child": {
            "origin": deque([np.array(o, dtype=float) for o in child_origins], maxlen=5),
            "vector": deque([np.array(v, dtype=float) for v in child_vectors], maxlen=5),
        },
        "Tester": {"origin": deque(maxlen=5), "vector": deque(maxlen=5)},
    }
    return eng


def vec_at(deg, length=100.0):
    """以「螢幕座標」產生指定角度的向量（y 向下，與 atan2(v[1], v[0]) 一致）。"""
    r = math.radians(deg)
    return (length * math.cos(r), length * math.sin(r))


# 一個放在射線正前方（水平往右）的目標框
BOX_AHEAD = (600.0, 450.0, 800.0, 550.0)
MAX_T = 0.6 * math.hypot(1920, 1080)   # 與 analyze_interaction 內的算法一致


# ======================================================================
# _is_ray_dir_stable：新關卡的決策函式本身
# ======================================================================
class TestRayDirStable:

    def test_identical_vectors_are_stable(self):
        eng = make_engine([vec_at(0)] * 5)
        assert eng._is_ray_dir_stable("Child", max_deg=25.0) is True

    def test_single_vector_is_not_stable(self):
        """只有 1 筆歷史 = 剛開始指，資訊不足，不可判定為穩定。"""
        eng = make_engine([vec_at(0)])
        assert eng._is_ray_dir_stable("Child", max_deg=25.0) is False

    def test_empty_history_is_not_stable(self):
        eng = make_engine([])
        assert eng._is_ray_dir_stable("Child", max_deg=25.0) is False

    def test_small_wobble_still_stable(self):
        """真實的指向會有手抖；±8° 必須照樣通過，否則召回率會被這道關卡殺光。"""
        eng = make_engine([vec_at(d) for d in (-8, -4, 0, 4, 8)])
        assert eng._is_ray_dir_stable("Child", max_deg=25.0) is True

    def test_wide_sweep_is_not_stable(self):
        """手臂橫掃 ±40°：這正是 P1-3 要擋掉的動作。"""
        eng = make_engine([vec_at(d) for d in (-40, -20, 0, 20, 40)])
        assert eng._is_ray_dir_stable("Child", max_deg=25.0) is False

    def test_zero_length_vector_rejected(self):
        """含零向量時無法定義方向，必須保守回 False（不可除以 0）。"""
        eng = make_engine([vec_at(0), (0.0, 0.0), vec_at(0)])
        assert eng._is_ray_dir_stable("Child", max_deg=25.0) is False

    def test_threshold_is_respected(self):
        """同一組向量，門檻放寬就該通過——證明門檻真的有接上。"""
        vectors = [vec_at(d) for d in (-30, -15, 0, 15, 30)]
        assert make_engine(vectors)._is_ray_dir_stable("Child", max_deg=25.0) is False
        assert make_engine(vectors)._is_ray_dir_stable("Child", max_deg=45.0) is True


# ======================================================================
# resolve_child_pointing_hit：實際被 analyze_interaction 使用的命中判定
# ======================================================================
class TestResolveChildPointingHit:

    def test_stable_ray_at_box_hits(self):
        """穩定指向目標框 → 命中，且回傳命中的是哪個框。"""
        eng = make_engine([vec_at(d) for d in (-3, -1, 0, 1, 3)])
        hit, box, origin, unit = eng.resolve_child_pointing_hit({"Child"}, [BOX_AHEAD], MAX_T)
        assert hit is True
        assert box == BOX_AHEAD
        assert origin == (100, 500)
        assert unit is not None

    def test_sweeping_ray_over_box_does_not_hit(self):
        """🌟 P1-3 核心回歸：掃動的射線平均後仍會落在框上，但不該算命中。

        ±40° 掃動下，_robust_average_vector 會留下中位數附近那群
        （-20/0/20）再平均，方向仍然對著 BOX_AHEAD——修正前這裡回 True。
        修正後由 _is_ray_dir_stable 擋下。
        """
        vectors = [vec_at(d) for d in (-40, -20, 0, 20, 40)]
        eng = make_engine(vectors)

        # 先確認「平均後的射線確實打中框」，否則這個測試證明不了任何事
        avg = eng._robust_average_vector([np.array(v, dtype=float) for v in vectors])
        unit = avg / np.linalg.norm(avg)
        assert eng.ray_intersects_box((100, 500), unit, BOX_AHEAD, max_t=MAX_T) is True

        # 有了上面那行，下面這行才有意義：擋掉它的是方向穩定性，不是幾何
        hit, box, _, _ = eng.resolve_child_pointing_hit({"Child"}, [BOX_AHEAD], MAX_T)
        assert hit is False
        assert box is None

    def test_dwell_not_reached_no_hit(self):
        """dwell 未達門檻時 Child 不在 pointing_owners，直接不算命中。"""
        eng = make_engine([vec_at(0)] * 5)
        hit, box, origin, unit = eng.resolve_child_pointing_hit(set(), [BOX_AHEAD], MAX_T)
        assert (hit, box, origin, unit) == (False, None, None, None)

    def test_empty_ray_history_no_hit(self):
        eng = make_engine([])
        hit, box, origin, unit = eng.resolve_child_pointing_hit({"Child"}, [BOX_AHEAD], MAX_T)
        assert (hit, box, origin, unit) == (False, None, None, None)

    def test_stable_ray_missing_box(self):
        """穩定但沒指到任何框：不算命中，但射線本身成立（origin/unit 有值）。"""
        eng = make_engine([vec_at(d) for d in (-3, -1, 0, 1, 3)])
        far_box = (600.0, 50.0, 800.0, 150.0)   # 在射線上方，射不到
        hit, box, origin, unit = eng.resolve_child_pointing_hit({"Child"}, [far_box], MAX_T)
        assert hit is False
        assert box is None
        assert origin == (100, 500)
        assert unit is not None

    def test_no_boxes_no_hit(self):
        eng = make_engine([vec_at(d) for d in (-3, -1, 0, 1, 3)])
        hit, box, _, _ = eng.resolve_child_pointing_hit({"Child"}, [], MAX_T)
        assert hit is False
        assert box is None

    def test_max_ray_t_still_enforced(self):
        """P0-3 的射線長度上限不可因為這次重構而失效。"""
        eng = make_engine([vec_at(d) for d in (-3, -1, 0, 1, 3)])
        assert eng.resolve_child_pointing_hit({"Child"}, [BOX_AHEAD], MAX_T)[0] is True
        # 把上限縮到框之前（框左緣距起點 500px）
        assert eng.resolve_child_pointing_hit({"Child"}, [BOX_AHEAD], 100.0)[0] is False

    def test_first_matching_box_is_returned(self):
        """多個框同時在射線上時，回傳最先命中的那個（與原本 break 行為一致）。"""
        eng = make_engine([vec_at(d) for d in (-3, -1, 0, 1, 3)])
        near = (300.0, 450.0, 400.0, 550.0)
        hit, box, _, _ = eng.resolve_child_pointing_hit({"Child"}, [near, BOX_AHEAD], MAX_T)
        assert hit is True
        assert box == near


# ======================================================================
# 召回率把關：新門檻不可把「真的有指」變成永遠測不到
# ======================================================================
class TestRecallNotDestroyed:

    def test_arm_raise_then_hold_eventually_hits(self):
        """真實指向 = 先把手舉過去（方向大幅變動），再停住。

        ray_history 是 maxlen=5 的 deque，舉手那幾幀一開始會留在歷史裡
        讓方向判定為不穩定；重點是它們會被滾出去，指向停住後必須命中。
        若這裡永遠測不到，代表門檻訂得太嚴、召回率會歸零。
        """
        # 前 3 幀：手臂往目標揮過去；之後 7 幀：停在目標上（±2° 手抖）
        seq = [vec_at(60), vec_at(40), vec_at(20)] + \
              [vec_at(d) for d in (2, -2, 1, -1, 0, 1, -1)]

        eng = make_engine([])
        hits = []
        for i, v in enumerate(seq):
            eng.ray_history["Child"]["vector"].append(np.array(v, dtype=float))
            eng.ray_history["Child"]["origin"].append(np.array((100, 500), dtype=float))
            owners = {"Child"} if i + 1 >= 3 else set()      # dwell >= DWELL_FRAMES(3)
            hits.append(eng.resolve_child_pointing_hit(owners, [BOX_AHEAD], MAX_T)[0])

        assert any(hits), "手停住之後仍測不到指向 → 門檻過嚴，召回率會歸零"
        assert hits[-1] is True, "指向已穩定持續，最後一幀必須命中"
        # 舉手過程中（歷史裡還混著大角度）不應該就先報命中
        assert hits[2] is False, "舉手途中方向還在大幅變動，不該算命中"

    def test_allowed_total_spread_is_about_50_deg(self):
        """門檻是「各幀與均值夾角 < 25°」，等於總擺幅約 50° 以內都放行。

        這條把門檻的實際鬆緊度釘在測試裡：日後有人調 POINTING_DIR_STABLE_DEG
        時，能立刻看出對真實手抖的容忍範圍變成多少。
        """
        # 總擺幅 40°（±20）→ 通過
        assert make_engine([vec_at(d) for d in (-20, -10, 0, 10, 20)]) \
            ._is_ray_dir_stable("Child", max_deg=25.0) is True
        # 總擺幅 60°（±30）→ 擋下
        assert make_engine([vec_at(d) for d in (-30, -15, 0, 15, 30)]) \
            ._is_ray_dir_stable("Child", max_deg=25.0) is False


# ======================================================================
# Stage 8 的既有行為不可被 P1-3 動到
# ======================================================================
class TestStage8Untouched:

    def test_s8_threshold_still_separate(self):
        """POINTING_DIR_STABLE_DEG 與 S8_DIR_STABLE_DEG 必須是兩個獨立旋鈕。"""
        eng = make_engine([vec_at(0)] * 3, dir_stable_deg=45.0)
        assert eng.POINTING_DIR_STABLE_DEG == 45.0
        assert eng.S8_DIR_STABLE_DEG == 25.0


# ======================================================================
# P1-4：analyze_interaction 拋例外時，快取不可沿用上一幀的 True
# ======================================================================
class TestPointingCacheOnException:

    def test_batch_entry_resets_cache_on_exception(self):
        """hurry/main.py 例外分支必須把 last_child_is_pointing_hit 歸零。

        用原始碼檢查而非執行整個主迴圈：主迴圈需要影片、模型與 GPU，
        無法在單元測試中重現；而這個修正的重點就是「例外分支裡有沒有
        那一行歸零」，靜態檢查足以守住它不被改回去。
        """
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "hurry", "main.py")
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()

        # 找到 analyze_interaction 的那個 except 區塊
        idx = None
        for i, line in enumerate(lines):
            if "analyze_interaction(display_frame" in line:
                idx = i
                break
        assert idx is not None, "找不到 analyze_interaction 呼叫點"

        block = "".join(lines[idx:idx + 15])
        assert "except" in block, "analyze_interaction 應包在 try/except 內"
        assert "last_child_is_pointing_hit = False" in block, (
            "例外分支未將 last_child_is_pointing_hit 歸零："
            "卡在 True 會讓上升邊緣永遠不再觸發，後續指向全部漏記"
        )


class TestContactVeto:
    """P3-1：有接觸或抓取就不算指向。"""

    def test_origin_inside_box_is_not_a_hit(self):
        """射線起點落在框內 → 不命中。

        這是原本 Ray-AABB 的退化情形：tmin < 0 被 max(0, tmin) 夾成 0，
        tmax >= 0 恆成立，方向完全不參與運算。實測 52 號影片 Stage 7 的
        4 個命中幀全部如此，食指其實指向物品的反方向（夾角 142°）。
        """
        inside = (700.0, 500.0)   # BOX_AHEAD 的正中心
        # 方向朝左（遠離框），舊版仍會回報命中
        eng = make_engine([vec_at(180)] * 5, child_origins=[inside] * 5)
        assert eng.ray_intersects_box(inside, np.array([-1.0, 0.0]), BOX_AHEAD,
                                      max_t=MAX_T) is False
        assert eng.resolve_child_pointing_hit({"Child"}, [BOX_AHEAD], MAX_T)[0] is False

    def test_origin_inside_box_legacy_behaviour_still_available(self):
        """關掉 reject_origin_inside 時維持原本的純 AABB 語意（供其他幾何用途）。"""
        eng = make_engine([vec_at(0)] * 5)
        assert eng.ray_intersects_box((700.0, 500.0), np.array([-1.0, 0.0]), BOX_AHEAD,
                                      max_t=MAX_T, reject_origin_inside=False) is True

    def test_fingertip_inside_box_is_not_a_hit(self):
        """食指尖伸進框內（抓著東西）→ 不命中，即使射線方向完美。"""
        eng = make_engine([vec_at(0)] * 5, tip=(700, 500), scale=40.0)
        assert eng.resolve_child_pointing_hit({"Child"}, [BOX_AHEAD], MAX_T)[0] is False

    def test_fingertip_too_close_is_not_a_hit(self):
        """指尖離框只有 0.1 倍手掌尺度（碰到了）→ 不命中。"""
        eng = make_engine([vec_at(0)] * 5, tip=(596, 500), scale=40.0)  # 間隙 4px = 0.1 scale
        assert eng.resolve_child_pointing_hit({"Child"}, [BOX_AHEAD], MAX_T)[0] is False

    def test_fingertip_far_enough_still_hits(self):
        """指尖離框 1 倍手掌尺度以上 → 正常命中，沒有誤殺。"""
        eng = make_engine([vec_at(0)] * 5, tip=(560, 500), scale=40.0)  # 間隙 40px = 1.0 scale
        assert eng.resolve_child_pointing_hit({"Child"}, [BOX_AHEAD], MAX_T)[0] is True

    def test_missing_scale_does_not_block(self):
        """量不到手掌尺度時只做框內判定，不因為無法量測就否決（避免製造漏偵）。"""
        eng = make_engine([vec_at(0)] * 5, tip=(596, 500), scale=0.0)
        assert eng.resolve_child_pointing_hit({"Child"}, [BOX_AHEAD], MAX_T)[0] is True


class TestAimGate:
    """P3-1：瞄準誤差閘——與手離物品的距離無關。"""

    def test_well_aimed_hits(self):
        eng = make_engine([vec_at(0)] * 5)
        assert eng.resolve_child_pointing_hit({"Child"}, [BOX_AHEAD], MAX_T)[0] is True

    def test_wide_aim_rejected_even_if_ray_grazes_box(self):
        """射線擦過框邊但瞄準誤差超標 → 不算命中。

        近距離的大框角半徑可以到 60° 以上，Ray-AABB 會放行；
        瞄準閘與距離無關，把這種「站很近所以怎麼指都算中」擋掉。
        """
        big_box = (150.0, 400.0, 900.0, 900.0)   # 從 (100,500) 看過去角半徑很大
        origin = (100.0, 500.0)
        eng = make_engine([vec_at(50)] * 5, child_origins=[origin] * 5)
        # Ray-AABB 本身是命中的
        unit = np.array(vec_at(50)) / np.linalg.norm(vec_at(50))
        assert eng.ray_intersects_box(origin, unit, big_box, max_t=MAX_T) is True
        # 但瞄準誤差超過門檻
        assert eng._aim_error_deg(origin, unit, big_box) > eng.POINTING_AIM_MAX_DEG
        assert eng.resolve_child_pointing_hit({"Child"}, [big_box], MAX_T)[0] is False

    def test_aim_error_is_symmetric_and_zero_on_center(self):
        eng = make_engine([vec_at(0)] * 5)
        origin = (100.0, 500.0)
        # 容差 0.01°：get_angle 正規化時加了 1e-9 的 epsilon，
        # 在 arccos(1-1e-9) 附近會被放大成約 0.0026°，屬預期的數值誤差。
        assert eng._aim_error_deg(origin, np.array([1.0, 0.0]), BOX_AHEAD) == pytest.approx(0.0, abs=0.01)
        a_up = eng._aim_error_deg(origin, np.array(vec_at(-10)), BOX_AHEAD)
        a_dn = eng._aim_error_deg(origin, np.array(vec_at(10)), BOX_AHEAD)
        assert a_up == pytest.approx(a_dn, abs=1e-6)


class TestPointingEventMerge:
    """P3-2：一次持續的指向只算一次，訊號抖動不重複計次。"""

    @staticmethod
    def _run(signal_frames, fps=30.0):
        """signal_frames: [bool, ...] 逐幀的 child_is_pointing_hit。回傳計次結果。"""
        from modules.scoring_engine import POINTING_COOLDOWN_SEC, POINTING_GAP_SEC
        rec = {"_prev_pointing": False, "_last_pointing_time": -999.0,
               "_last_pointing_active_time": -999.0, "count": 0}
        for i, hit in enumerate(signal_frames):
            t = i / fps
            resumed = (t - rec["_last_pointing_active_time"]) < POINTING_GAP_SEC
            if hit and not rec["_prev_pointing"] and not resumed:
                if t - rec["_last_pointing_time"] > POINTING_COOLDOWN_SEC:
                    rec["count"] += 1
                    rec["_last_pointing_time"] = t
            if hit:
                rec["_last_pointing_active_time"] = t
            rec["_prev_pointing"] = hit
        return rec["count"]

    def test_continuous_pointing_counts_once(self):
        """連續 16 秒的指向 = 1 次（原本會被 0.8 秒冷卻切成十幾次）。"""
        assert self._run([True] * int(16 * 30)) == 1

    def test_flickering_signal_counts_once(self):
        """每秒閃斷 0.5 秒的 16 秒指向仍是 1 次——中斷沒超過 POINTING_GAP_SEC。"""
        sig = []
        for _ in range(16):
            sig += [True] * 15 + [False] * 15   # 0.5s 有、0.5s 無
        assert self._run(sig) == 1

    def test_genuinely_separate_pointings_count_separately(self):
        """兩次指向中間空了 4 秒 → 正常計 2 次，合併機制沒有吃掉真實事件。"""
        sig = [True] * 60 + [False] * 120 + [True] * 60
        assert self._run(sig) == 2

    def test_gap_just_below_threshold_merges(self):
        """空檔略短於門檻 → 合併為 1 次。"""
        from modules.scoring_engine import POINTING_GAP_SEC
        gap = int((POINTING_GAP_SEC - 0.2) * 30)
        assert self._run([True] * 60 + [False] * gap + [True] * 60) == 1

    def test_gap_just_above_threshold_splits(self):
        """空檔略長於門檻 → 分成 2 次。"""
        from modules.scoring_engine import POINTING_GAP_SEC
        gap = int((POINTING_GAP_SEC + 0.2) * 30)
        assert self._run([True] * 60 + [False] * gap + [True] * 60) == 2


class TestHandsTooCloseVeto:
    """P5-1：雙手靠在一起就不判定指向。"""

    @staticmethod
    def _person(owner, shoulder_w=100.0, wrists=((0.0, 0.0),), conf=0.9):
        """造一個 yolo_people 項目：只填 compute_hands_too_close 會讀的欄位。"""
        kpts = np.zeros((17, 2), dtype=float)
        confs = np.zeros(17, dtype=float)
        kpts[5] = (0.0, 0.0)            # 左肩
        kpts[6] = (shoulder_w, 0.0)     # 右肩
        confs[5] = confs[6] = 0.9
        for idx, w in zip((9, 10), wrists):   # 左腕、右腕
            kpts[idx] = w
            confs[idx] = conf
        return {"owner": owner, "kpts": kpts, "confs": confs}

    def test_child_hands_apart_is_allowed(self):
        """雙手分開（1.5 倍肩寬）→ 不否決。"""
        eng = make_engine([vec_at(0)] * 5)
        ppl = [self._person("Child", 100.0, ((0.0, 200.0), (150.0, 200.0)))]
        assert eng.compute_hands_too_close(ppl) is False

    def test_child_hands_close_is_vetoed(self):
        """雙手靠在一起（0.3 倍肩寬）→ 否決。"""
        eng = make_engine([vec_at(0)] * 5)
        ppl = [self._person("Child", 100.0, ((0.0, 200.0), (30.0, 200.0)))]
        assert eng.compute_hands_too_close(ppl) is True

    def test_tester_hand_near_child_hand_is_vetoed(self):
        """施測者的手靠近兒童的手 → 否決（這是使用者指出的第二種情境）。"""
        eng = make_engine([vec_at(0)] * 5)
        ppl = [self._person("Child", 100.0, ((0.0, 200.0), (500.0, 200.0))),
               self._person("Tester", 300.0, ((20.0, 200.0), (900.0, 900.0)))]
        assert eng.compute_hands_too_close(ppl) is True

    def test_low_confidence_wrist_fails_open(self):
        """關鍵點信心不足 → 放行，不製造漏偵。"""
        eng = make_engine([vec_at(0)] * 5)
        ppl = [self._person("Child", 100.0, ((0.0, 200.0), (30.0, 200.0)), conf=0.2)]
        assert eng.compute_hands_too_close(ppl) is False

    def test_missing_shoulder_fails_open(self):
        """量不到肩寬就無從正規化 → 放行。"""
        eng = make_engine([vec_at(0)] * 5)
        p = self._person("Child", 100.0, ((0.0, 200.0), (30.0, 200.0)))
        p["confs"][5] = 0.1
        assert eng.compute_hands_too_close([p]) is False

    def test_no_people_fails_open(self):
        eng = make_engine([vec_at(0)] * 5)
        assert eng.compute_hands_too_close([]) is False
        assert eng.compute_hands_too_close(None) is False

    def test_veto_blocks_resolve(self):
        """旗標為真時，命中判定直接回傳未命中——即使射線完美瞄準。"""
        eng = make_engine([vec_at(0)] * 5)
        assert eng.resolve_child_pointing_hit({"Child"}, [BOX_AHEAD], MAX_T)[0] is True
        eng.last_hands_too_close = True
        assert eng.resolve_child_pointing_hit({"Child"}, [BOX_AHEAD], MAX_T)[0] is False

    def test_threshold_is_configurable(self):
        """門檻是具名參數，調整後行為跟著變。"""
        eng = make_engine([vec_at(0)] * 5)
        ppl = [self._person("Child", 100.0, ((0.0, 200.0), (60.0, 200.0)))]  # 0.6 倍肩寬
        eng.POINTING_HANDS_APART_MIN = 0.8
        assert eng.compute_hands_too_close(ppl) is True
        eng.POINTING_HANDS_APART_MIN = 0.5
        assert eng.compute_hands_too_close(ppl) is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
