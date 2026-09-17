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
    """回傳 (total, level)。record 需含 tb / th / pointing_t / label 欄位。

    🌟 修改（P1-2）：由分數查表改為語意組合判定。
    原本 total=2 對應 HI，但有兩種完全不同來源：
      - TB=1, TH=0, Pointing=1 → 有看物、有指，但未看回人 → 不應是 HI
      - TB=0, TH=1, Pointing=0 → 只看回人，無看物也無指 → 也不應是 HI
    改用明確組合讓等級語意正確，並與 IJA（共同注意力）臨床定義對齊：
      HI  = 指向（Pointing）AND 看回施測者（TH）—— 最高階主動分享
      LI  = 看向物品（TB）AND 看回施測者（TH）AND 沒有指向 —— 視線交替
      HR  = 遠物關卡且 total=1 —— 有基本反應但未主動
      LR  = 其他關卡且 total=1 —— 低反應
      F   = 無任何達成
    """
    tb  = record.get("tb")       is not None
    th  = record.get("th")       is not None
    pt  = record.get("pointing_t") is not None
    tb_score      = 1 if tb else 0
    th_score      = 2 if th else 0
    pointing_score = 1 if pt else 0
    total = tb_score + th_score + pointing_score

    # ============================================================
    # Stage 8（手機怪聲）特規等級判定（邏輯不變，改用 pt/th bool）
    # 此題指向不需指中物品（射線存在即計），等級規則：
    #   出現指向（手指）→ HI
    #   僅偵測到 TH（轉頭看人）→ LI
    #   皆無 → F
    # ============================================================
    if record.get("stage") == 8:
        if pt:
            level = "HI"
        elif th:
            level = "LI"
        else:
            level = "F"
        return total, level

    # ============================================================
    # 🌟 修改（P1-2）：其他 Stage 改為明確組合判定（取代分數查表）
    # ============================================================
    if pt and th:
        # 指向 + 看回人 = 最高階主動分享（High Initiator）
        level = "HI"
    elif tb and th:
        # 看物 + 看回人、但無指向 = 視線交替主動（Low Initiator）
        level = "LI"
    elif pt or tb:
        # 只有看物或只有指向，但沒看回人
        level = "HR" if is_far_stage(record.get("label")) else "LR"
    elif th:
        # 只看回人，未達主動分享（罕見組合）
        level = "LR"
    else:
        level = "F"

    return total, level


def format_stage_score_suffix(record):
    """回傳可直接附加在 stage 標題行後面的字串，例如 ' -- 反應等級 LI(3分)'"""
    total, level = compute_stage_score(record)
    return " -- 反應等級 " + level + "(" + str(total) + "分)"
