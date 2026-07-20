"""
Stage 反應等級計分模組

依據「ja判定公式.xlsx」的評分邏輯，針對每個 stage 記錄的 Pointing / TB / TH
是否達成進行計分，換算成反應等級（F / LR / HR / LI / HI）。

計分規則（對應 trigger_event_records 中的欄位）：
    TB      (record["tb"])        有值 = 1 分，無值 (None / not achieved) = 0 分
    TH      (record["th"])        有值 = 2 分，無值 (None / not achieved) = 0 分
    Pointing(record["pointing_t"]) 有值 = 1 分，無值 (None / not detected)  = 0 分

    total = TB分 + TH分 + Pointing分   (範圍 0~4)

等級對照（依 ja判定公式.xlsx 的 IF 巢狀公式歸納）：
    total = 0            -> F
    total = 1  (遠物關卡) -> HR
    total = 1  (其他關卡) -> LR
    total = 2             -> HI
    total = 3             -> LI
    total = 4             -> HI

"遠物關卡"（stage label 含 "Far"，例如 真人指遠物 / 機指遠物）在 total=1 時
歸為 HR，其餘關卡在 total=1 時歸為 LR；其餘等級判定與關卡類型無關。
"""

FAR_STAGE_KEYWORD = "Far"


def is_far_stage(label):
    return FAR_STAGE_KEYWORD in (label or "")


def compute_stage_score(record):
    """回傳 (total, level)。record 需含 tb / th / pointing_t / label 欄位。"""
    tb_score = 1 if record.get("tb") is not None else 0
    th_score = 2 if record.get("th") is not None else 0
    pointing_score = 1 if record.get("pointing_t") is not None else 0
    total = tb_score + th_score + pointing_score

    # ============================================================
    # 🌟 新增：Stage 8（手機怪聲）特規等級判定
    # 此題指向不需指中物品（射線存在即計），等級規則：
    #   出現指向（手指）→ HI
    #   僅偵測到 TH（轉頭看人）→ LI
    #   皆無 → F
    # ============================================================
    if record.get("stage") == 8:
        if pointing_score:
            level = "HI"
        elif th_score:
            level = "LI"
        else:
            level = "F"
        return total, level

    if total == 0:
        level = "F"
    elif total == 1:
        level = "HR" if is_far_stage(record.get("label")) else "LR"
    elif total == 2:
        level = "HI"
    elif total == 3:
        level = "LI"
    else:
        level = "HI"

    return total, level


def format_stage_score_suffix(record):
    """回傳可直接附加在 stage 標題行後面的字串，例如 ' -- 反應等級 LI(3分)'"""
    total, level = compute_stage_score(record)
    return " -- 反應等級 " + level + "(" + str(total) + "分)"
