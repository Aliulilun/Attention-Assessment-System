# -*- coding: utf-8 -*-
"""
event_record txt  ×  人工-AI結果差異對照表.xlsx  自動比對工具
================================================================

用途
----
把 AI 跑出來的 event_record（`hurry/output/*.txt` 或 `summarize/files/*.txt`）
與對照表的「人工標記欄」逐格比對，輸出 TB / TH / Pointing 三個構念的
混淆矩陣與關鍵指標，取代原本純手工讀 Excel 的流程。

指向報告（docs/人工-AI指向偵測差異分析.md）第八節要求固定回報四個數字：
    精確率 = TP / (TP + FP)      目標 > 60%
    召回率 = TP / (TP + FN)      守住不低於 45%
    ASD / TD 誤報率比值           目標接近 1.0
    Stage 8 AI 指向人數           人工為 2

用法
----
    conda activate mediapipe_py39
    cd C:\\project

    # 比對單一資料夾
    python summarize\\compare_with_manual.py --txt-dir hurry\\output

    # 前後兩版一起比（會多印一欄變化量）
    python summarize\\compare_with_manual.py --txt-dir hurry\\output ^
                                             --baseline-dir hurry\\output_before2

    # 只看特定關卡
    python summarize\\compare_with_manual.py --txt-dir hurry\\output --stages 3 4

輸出
----
主控台摘要，另可用 --csv 匯出逐格明細（UTF-8-SIG，Excel 直接開）。

對照表的欄位結構（2026-09-13 實地確認）
--------------------------------------
Stage 1~10 每關 6 欄，依序為：
    LR_n 或 HR_n  … 人工：兒童有看向目標物   （對應 AI 的 TB）
    TB_n          … AI  ：TB
    LI_n          … 人工：兒童有看回施測者   （對應 AI 的 TH）
    TH_n          … AI  ：TH
    HI_n（第一個）… 人工：綜合等級 HI
    HI_n（第二個）… AI  ：綜合等級 HI
Stage 11~14 只有人工欄（LR/HR、LI、HI、total），沒有 AI 欄，故本工具只處理 1~10。

指向的反解
----------
對照表沒有 Pointing 欄，但計分公式是決定性的，可由 (TB, TH, HI) 反解：
    total = TB(1) + TH(2) + Pointing(1)
    total 0→F  1→LR/HR  2→HI  3→LI  4→HI
    Stage 8 特規：有 Pointing → HI；只有 TH → LI；皆無 → F
    ┌────┬────┬──────────────────────────────┐
    │ TB │ TH │ 由 HI 反推                    │
    ├────┼────┼──────────────────────────────┤
    │  0 │  0 │ HI 必為 0，無法反解（排除）    │
    │  1 │  0 │ HI=1 ⟺ 有指向                 │
    │  0 │  1 │ HI=1 ⟺ 沒有指向               │
    │  1 │  1 │ HI=1 ⟺ 有指向                 │
    └────┴────┴──────────────────────────────┘
Stage 8 沒有 TB，直接用 HI_8 即等於「有無指向」。
"""

import argparse
import csv
import os
import re
import sys
from collections import OrderedDict

try:
    from openpyxl import load_workbook
except ImportError:
    print("需要 openpyxl：pip install openpyxl")
    sys.exit(1)


# ============================================================
# 預設路徑
# ============================================================
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_XLSX = os.path.join(PROJECT_DIR, "docs", "人工-AI結果差異對照表.xlsx")
STAGES = list(range(1, 11))          # 只有 1~10 有 AI 對照欄
ASD_COL_HEADER = "是否為自閉"        # 對照表的分組欄（1 = ASD）


# ============================================================
# 一、解析 event_record txt
# ============================================================
STAGE_HEAD_RE = re.compile(r"^\s*\d+\.\s*Stage\s+(\d+)\s+--")
TB_RE         = re.compile(r"^\s*TB\s*=\s*(.+)$")
TH_RE         = re.compile(r"^\s*TH\s*=\s*(.+)$")
PT_RE         = re.compile(r"^\s*Pointing\s*=\s*(.+)$")


def _achieved(value_text):
    """TB/TH/Pointing 那一行的右半段是否代表「有達成」。

    未達成的寫法有三種：not achieved / not detected / -- (no TB condition)
    其餘（以秒數開頭）都算達成。
    """
    v = (value_text or "").strip()
    if not v:
        return False
    if v.startswith("--"):
        return False
    low = v.lower()
    if "not achieved" in low or "not detected" in low:
        return False
    return True


def parse_event_record(path):
    """回傳 {stage: {"tb": bool, "th": bool, "pt": bool}}。"""
    result = {}
    current = None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = STAGE_HEAD_RE.match(line)
            if m:
                current = int(m.group(1))
                result.setdefault(current, {"tb": False, "th": False, "pt": False})
                continue
            if current is None:
                continue
            m = TB_RE.match(line)
            if m:
                result[current]["tb"] = _achieved(m.group(1))
                continue
            m = TH_RE.match(line)
            if m:
                result[current]["th"] = _achieved(m.group(1))
                continue
            m = PT_RE.match(line)
            if m:
                result[current]["pt"] = _achieved(m.group(1))
    return result


def load_txt_dir(txt_dir):
    """讀取資料夾內所有 {編號}.txt，回傳 {編號: {stage: {...}}}。"""
    data = {}
    if not os.path.isdir(txt_dir):
        print("找不到資料夾：%s" % txt_dir)
        return data
    for name in sorted(os.listdir(txt_dir)):
        if not name.lower().endswith(".txt"):
            continue
        stem = os.path.splitext(name)[0].strip()
        if not stem.isdigit():
            continue          # 略過 _speech_engine.log 之類的非結果檔
        data[int(stem)] = parse_event_record(os.path.join(txt_dir, name))
    return data


# ============================================================
# 二、解析對照表
# ============================================================
def _norm(v):
    """把儲存格內容正規化成 0 / 1 / None。"""
    if v is None:
        return None
    s = str(v).strip()
    if s == "":
        return None
    try:
        f = float(s)
    except ValueError:
        return None
    return 1 if f >= 0.5 else 0


def build_column_map(ws):
    """掃描標題列，回傳 {stage: {"manual_tb", "ai_tb", "manual_th", "ai_th",
                                "manual_hi", "ai_hi"}}（值為欄索引，1-based）。

    依「同一個 _n 後綴的欄位出現順序」指派角色，不依賴底色，
    因為底色只標在 AI 欄、且不同版本可能調色。
    """
    headers = {}
    for col in range(1, ws.max_column + 1):
        h = ws.cell(row=1, column=col).value
        headers[col] = str(h).strip() if h is not None else ""

    colmap = {}
    for stage in STAGES:
        suffix = "_%d" % stage
        # 依欄位順序收集屬於這一關的欄
        cols = [c for c in sorted(headers) if headers[c].endswith(suffix)]
        roles = OrderedDict()
        hi_seen = 0
        for c in cols:
            name = headers[c]
            base = name[: -len(suffix)].upper()
            if base in ("LR", "HR"):
                roles["manual_tb"] = c
            elif base == "TB":
                roles["ai_tb"] = c
            elif base in ("LI", "LL"):      # 實際檔案裡曾看到 LL_3 這種筆誤
                roles["manual_th"] = c
            elif base == "TH":
                roles["ai_th"] = c
            elif base == "HI":
                hi_seen += 1
                if hi_seen == 1:
                    roles["manual_hi"] = c
                else:
                    roles["ai_hi"] = c
        # Stage 8 沒有 TB 欄位（tb_mode = None），允許缺 ai_tb
        if "manual_th" in roles and "manual_hi" in roles:
            colmap[stage] = roles
    return colmap


def load_manual(xlsx_path):
    """回傳 (manual, groups, colmap)：
    manual = {編號: {stage: {"tb":0/1/None, "th":..., "hi":..., "pt":0/1/None}}}
    groups = {編號: "ASD" / "TD" / None}
    """
    wb = load_workbook(xlsx_path, data_only=True)
    ws = wb[wb.sheetnames[0]]
    colmap = build_column_map(ws)

    # 找「計畫編號」與分組欄
    id_col, asd_col = 1, None
    for col in range(1, ws.max_column + 1):
        h = ws.cell(row=1, column=col).value
        h = str(h).strip() if h is not None else ""
        if "計畫編號" in h:
            id_col = col
        if ASD_COL_HEADER in h:
            asd_col = col

    manual, groups = {}, {}
    for row in range(2, ws.max_row + 1):
        raw_id = ws.cell(row=row, column=id_col).value
        if raw_id is None:
            continue
        try:
            cid = int(float(str(raw_id).strip()))
        except ValueError:
            continue

        per_stage = {}
        any_value = False
        for stage, roles in colmap.items():
            tb = _norm(ws.cell(row=row, column=roles["manual_tb"]).value) if "manual_tb" in roles else None
            th = _norm(ws.cell(row=row, column=roles["manual_th"]).value)
            hi = _norm(ws.cell(row=row, column=roles["manual_hi"]).value)
            if tb is not None or th is not None or hi is not None:
                any_value = True
            per_stage[stage] = {
                "tb": tb, "th": th, "hi": hi,
                "pt": solve_pointing(stage, tb, th, hi),
            }
        if not any_value:
            continue          # 整列空白（被刪除的個案）

        manual[cid] = per_stage
        if asd_col is not None:
            g = _norm(ws.cell(row=row, column=asd_col).value)
            groups[cid] = "ASD" if g == 1 else ("TD" if g == 0 else None)
        else:
            groups[cid] = None
    return manual, groups, colmap


def solve_pointing(stage, tb, th, hi):
    """由 (TB, TH, HI) 反解 Pointing；無法判定時回傳 None。"""
    if hi is None:
        return None
    if stage == 8:
        # Stage 8 沒有 TB，特規：有 Pointing → HI
        return 1 if hi == 1 else 0
    if tb is None or th is None:
        return None
    if tb == 0 and th == 0:
        return None            # HI 必為 0，無法反解
    if tb == 1 and th == 0:
        return 1 if hi == 1 else 0
    if tb == 0 and th == 1:
        return 0 if hi == 1 else 1
    return 1 if hi == 1 else 0   # tb == 1 and th == 1


# ============================================================
# 三、混淆矩陣
# ============================================================
class Confusion(object):
    __slots__ = ("tp", "fp", "fn", "tn", "cases")

    def __init__(self):
        self.tp = self.fp = self.fn = self.tn = 0
        self.cases = []        # [(編號, stage, 人工, AI)]

    def add(self, cid, stage, manual, ai):
        if manual == 1 and ai:
            self.tp += 1
        elif manual == 0 and ai:
            self.fp += 1
            self.cases.append((cid, stage, manual, ai))
        elif manual == 1 and not ai:
            self.fn += 1
            self.cases.append((cid, stage, manual, ai))
        else:
            self.tn += 1

    @property
    def n(self):
        return self.tp + self.fp + self.fn + self.tn

    @property
    def precision(self):
        d = self.tp + self.fp
        return (self.tp / d) if d else None

    @property
    def recall(self):
        d = self.tp + self.fn
        return (self.tp / d) if d else None

    @property
    def fp_rate(self):
        """誤報率 = FP / 實際無此行為的樣本數"""
        d = self.fp + self.tn
        return (self.fp / d) if d else None


def _pct(v):
    return "  n/a " if v is None else ("%5.1f%%" % (v * 100))


# ============================================================
# 四、主流程
# ============================================================
def evaluate(ai_data, manual, groups, stages):
    """回傳 {指標名稱: Confusion} 以及 ASD/TD 分組的 Pointing 混淆矩陣。"""
    overall = {"TB": Confusion(), "TH": Confusion(), "Pointing": Confusion()}
    per_stage = {s: {"TB": Confusion(), "TH": Confusion(), "Pointing": Confusion()}
                 for s in stages}
    by_group = {"ASD": Confusion(), "TD": Confusion()}

    for cid, stage_data in sorted(ai_data.items()):
        if cid not in manual:
            continue
        for stage in stages:
            if stage not in stage_data or stage not in manual[cid]:
                continue
            ai = stage_data[stage]
            mn = manual[cid][stage]

            # Stage 8 沒有 TB 構念，跳過
            if stage != 8 and mn["tb"] is not None:
                overall["TB"].add(cid, stage, mn["tb"], ai["tb"])
                per_stage[stage]["TB"].add(cid, stage, mn["tb"], ai["tb"])
            if mn["th"] is not None:
                overall["TH"].add(cid, stage, mn["th"], ai["th"])
                per_stage[stage]["TH"].add(cid, stage, mn["th"], ai["th"])
            if mn["pt"] is not None:
                overall["Pointing"].add(cid, stage, mn["pt"], ai["pt"])
                per_stage[stage]["Pointing"].add(cid, stage, mn["pt"], ai["pt"])
                g = groups.get(cid)
                if g in by_group:
                    by_group[g].add(cid, stage, mn["pt"], ai["pt"])

    return overall, per_stage, by_group


def print_report(title, overall, per_stage, by_group, ai_data, manual, stages):
    print("")
    print("=" * 72)
    print("  " + title)
    print("=" * 72)
    matched = sorted(set(ai_data) & set(manual))
    print("比對樣本：%d 位（AI 結果 %d 份、對照表 %d 位）"
          % (len(matched), len(ai_data), len(manual)))
    print("涵蓋關卡：%s" % ", ".join(str(s) for s in stages))
    if matched:
        print("編號：%s" % ", ".join(str(c) for c in matched))

    print("")
    print("── 整體混淆矩陣 ──")
    print("%-9s %5s %5s %5s %5s   %7s %7s"
          % ("指標", "TP", "FP", "FN", "TN", "精確率", "召回率"))
    for key in ("TB", "TH", "Pointing"):
        c = overall[key]
        print("%-9s %5d %5d %5d %5d   %7s %7s"
              % (key, c.tp, c.fp, c.fn, c.tn, _pct(c.precision), _pct(c.recall)))

    print("")
    print("── 各關卡明細 ──")
    print("%-6s | %-21s | %-21s | %-21s"
          % ("Stage", "TB (TP/FP/FN)", "TH (TP/FP/FN)", "Pointing (TP/FP/FN)"))
    for s in stages:
        cells = []
        for key in ("TB", "TH", "Pointing"):
            c = per_stage[s][key]
            cells.append("%2d/%2d/%2d  精%s" % (c.tp, c.fp, c.fn, _pct(c.precision)))
        print("%-6d | %-21s | %-21s | %-21s" % (s, cells[0], cells[1], cells[2]))

    print("")
    print("── 指向報告要求的四個數字 ──")
    pt = overall["Pointing"]
    print("  精確率        = %s   （目標 > 60%%）" % _pct(pt.precision))
    print("  召回率        = %s   （守住不低於 45%%）" % _pct(pt.recall))
    asd, td = by_group["ASD"], by_group["TD"]
    if asd.fp_rate is not None and td.fp_rate is not None and td.fp_rate > 0:
        print("  ASD/TD 誤報率 = %s / %s  比值 %.2f   （目標接近 1.0）"
              % (_pct(asd.fp_rate), _pct(td.fp_rate), asd.fp_rate / td.fp_rate))
    else:
        print("  ASD/TD 誤報率 = %s / %s" % (_pct(asd.fp_rate), _pct(td.fp_rate)))

    s8_ai = sum(1 for cid in matched
                if 8 in ai_data[cid] and ai_data[cid][8]["pt"])
    s8_manual = sum(1 for cid in matched
                    if 8 in manual[cid] and manual[cid][8]["pt"] == 1)
    print("  Stage 8 指向  = AI %d 人 / 人工 %d 人" % (s8_ai, s8_manual))

    # 逐筆列出不一致，方便直接去看影片
    print("")
    print("── 不一致明細（人工 vs AI）──")
    any_mismatch = False
    for key in ("TB", "TH", "Pointing"):
        c = overall[key]
        if not c.cases:
            continue
        any_mismatch = True
        fps = [x for x in c.cases if x[2] == 0]
        fns = [x for x in c.cases if x[2] == 1]
        if fps:
            print("  %-8s 假陽性（AI 有、人工無）：%s"
                  % (key, ", ".join("%d-S%d" % (cid, s) for cid, s, _, _ in fps)))
        if fns:
            print("  %-8s 漏偵測（AI 無、人工有）：%s"
                  % (key, ", ".join("%d-S%d" % (cid, s) for cid, s, _, _ in fns)))
    if not any_mismatch:
        print("  （完全一致）")


def write_csv(path, ai_data, manual, stages):
    rows = [("編號", "Stage",
             "人工_TB", "AI_TB", "TB一致",
             "人工_TH", "AI_TH", "TH一致",
             "人工_指向", "AI_指向", "指向一致")]
    for cid in sorted(set(ai_data) & set(manual)):
        for s in stages:
            if s not in ai_data[cid] or s not in manual[cid]:
                continue
            a, m = ai_data[cid][s], manual[cid][s]

            def cmp3(mv, av):
                if mv is None:
                    return ""
                return "OK" if bool(mv) == bool(av) else "DIFF"

            rows.append((
                cid, s,
                m["tb"], int(a["tb"]), cmp3(m["tb"], a["tb"]),
                m["th"], int(a["th"]), cmp3(m["th"], a["th"]),
                m["pt"], int(a["pt"]), cmp3(m["pt"], a["pt"]),
            ))
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        csv.writer(f).writerows(rows)
    print("")
    print(">>> 逐格明細已輸出：%s" % path)


def main():
    ap = argparse.ArgumentParser(
        description="event_record txt 與人工標記對照表的自動比對工具")
    ap.add_argument("--txt-dir", required=True,
                    help="要評估的 event_record 資料夾（例：hurry\\output）")
    ap.add_argument("--baseline-dir", default=None,
                    help="修正前的資料夾，會一併評估以便前後對照")
    ap.add_argument("--xlsx", default=DEFAULT_XLSX, help="對照表路徑")
    ap.add_argument("--stages", type=int, nargs="+", default=STAGES,
                    help="只評估指定關卡，預設 1~10")
    ap.add_argument("--csv", default=None, help="輸出逐格明細 CSV 的路徑")
    args = ap.parse_args()

    if not os.path.exists(args.xlsx):
        print("❌ 找不到對照表：%s" % args.xlsx)
        return 1

    stages = [s for s in args.stages if s in STAGES]
    if not stages:
        print("--stages 只能指定 1~10（11 以後沒有 AI 對照欄）")
        return 1

    print(">>> 讀取對照表：%s" % args.xlsx)
    manual, groups, colmap = load_manual(args.xlsx)
    print("    解析到 %d 位受試者、%d 個關卡的欄位對應" % (len(manual), len(colmap)))
    missing = [s for s in stages if s not in colmap]
    if missing:
        print("這些關卡在對照表中找不到欄位，將略過：%s" % missing)
        stages = [s for s in stages if s in colmap]

    ai_data = load_txt_dir(args.txt_dir)
    print(">>> 讀取 AI 結果：%s（%d 份）" % (args.txt_dir, len(ai_data)))
    if not ai_data:
        print("沒有可用的 event_record txt")
        return 1

    overall, per_stage, by_group = evaluate(ai_data, manual, groups, stages)
    print_report("目前版本：%s" % args.txt_dir,
                 overall, per_stage, by_group, ai_data, manual, stages)

    if args.baseline_dir:
        base_data = load_txt_dir(args.baseline_dir)
        if base_data:
            b_overall, b_per_stage, b_group = evaluate(base_data, manual, groups, stages)
            print_report("對照基準：%s" % args.baseline_dir,
                         b_overall, b_per_stage, b_group, base_data, manual, stages)
            print("")
            print("=" * 72)
            print("  前後變化（正號 = 變好）")
            print("=" * 72)
            print("%-9s %14s %14s" % ("指標", "精確率", "召回率"))
            for key in ("TB", "TH", "Pointing"):
                cur, base = overall[key], b_overall[key]
                dp = ("%+6.1f pt" % ((cur.precision - base.precision) * 100)
                      if cur.precision is not None and base.precision is not None else "   n/a")
                dr = ("%+6.1f pt" % ((cur.recall - base.recall) * 100)
                      if cur.recall is not None and base.recall is not None else "   n/a")
                print("%-9s %14s %14s" % (key, dp, dr))
            print("")
            print("%-9s %6s %6s %6s" % ("指標", "ΔTP", "ΔFP", "ΔFN"))
            for key in ("TB", "TH", "Pointing"):
                cur, base = overall[key], b_overall[key]
                print("%-9s %+6d %+6d %+6d"
                      % (key, cur.tp - base.tp, cur.fp - base.fp, cur.fn - base.fn))

    if args.csv:
        write_csv(args.csv, ai_data, manual, stages)

    return 0


if __name__ == "__main__":
    sys.exit(main())
