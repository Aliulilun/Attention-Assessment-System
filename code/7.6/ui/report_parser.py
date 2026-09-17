# -*- coding: utf-8 -*-
"""
event_record txt 解析器
========================
把 ScoringEngine.write_report() 產出的純文字報告轉成結構化字典，供
「單次結果報表」與「歷史統計」兩個頁面共用。

刻意不依賴 summarize/summarize.py：那支程式的輸出是為了匯出 Excel 而
攤平成一列（Stage1_T0、Stage1_TB…），欄位名寫死且缺少逐字稿與事件流水。
介面需要的是巢狀結構（每個 Stage 一筆紀錄 + 逐字稿 + 事件 log），
兩者用途不同，各自維護比硬套更清楚。

解析目標格式（節錄）：
    === Result Summary ===
    Video: c:\\project\\hurry\\video\\42.mp4
    Total Score: 6
    Total Gazing Events: 25
    01. Stage 1 -- 真人指近物(Pointing - Near) -- 反應等級 LR(1分)
        T0       = 5.83s
        Pointing = 26.56s  (+3.23s from T0) x1     ← 或 "not detected"
        TB       = 6.33s  (+0.5s from T0) x1       ← 或 "not achieved" / "-- (no TB condition)"
        TH       = ...
        Sequence = ...
        計分結束 = 12.67s                           ← 或 "未知"
"""

import os
import re

# 反應等級由高到低（畫圖排序、統計欄位順序都用這個）
LEVEL_ORDER = ["HI", "LI", "HR", "LR", "F"]

STAGE_NAMES = {
    1: "真人指近物(Pointing - Near)",
    2: "真人指近物(Pointing - Near)",
    3: "真人指遠物(Pointing - Far)",
    4: "真人指遠物(Pointing - Far)",
    5: "神奇氣球(Magical Balloon)",
    6: "看偶寫字 (Puppet Writing)",
    7: "開箱驚喜袋(Mystery Bag)",
    8: "手機怪聲(Strange Sound)",
    9: "機器人畫畫(Robot Drawing)",
    10: "機器人煙火秀(Social Referencing)",
    11: "機指近物(Pointing - Near)",
    12: "機指近物(Pointing - Near)",
    13: "機指遠物(Pointing - Far)",
    14: "機指遠物(Pointing - Far)",
}

# 每個 Stage 的簡短中文別名（表頭、圖表軸標籤用，避免過長）
STAGE_SHORT = {
    1: "S1 人指近", 2: "S2 人指近", 3: "S3 人指遠", 4: "S4 人指遠",
    5: "S5 氣球", 6: "S6 操偶", 7: "S7 驚喜袋", 8: "S8 怪聲",
    9: "S9 畫畫", 10: "S10 煙火",
    11: "S11 機指近", 12: "S12 機指近", 13: "S13 機指遠", 14: "S14 機指遠",
}

_HEADER_RE = re.compile(
    r"^\s*\d+\.\s*Stage\s+(\d+)\s*--\s*(.+?)(?:\s*--\s*反應等級\s*([A-Za-z]+)\s*\((\d+)分?\))?\s*$"
)
# "= 26.56s  (+3.23s from T0) x1" → 時間、次數
_VALUE_RE = re.compile(r"=\s*([\d.]+)s")
_COUNT_RE = re.compile(r"x(\d+)\s*$")
_TRANSCRIPT_RE = re.compile(
    r"^\[([\d.]+)s\s*~\s*([\d.]+)s\]\s*(.*?)(?:\s*←\s*關鍵字：(.*))?$"
)


def _blank_record(stage):
    return {
        "stage": stage,
        "label": STAGE_NAMES.get(stage, "Stage%d" % stage),
        "level": None,          # HI / LI / HR / LR / F；None = 此關未出現在報告中
        "total": None,          # 0~4
        "t0": None,
        "end": None,            # 計分結束時間
        "pointing": None, "pointing_count": 0,
        "tb": None, "tb_count": 0, "tb_na": False,   # tb_na = 此關無 TB 條件（Stage 8）
        "th": None, "th_count": 0,
        "gazing_count": 0,
        "detected": False,      # 報告中是否有此 Stage 區塊
    }


def _parse_metric(line):
    """回傳 (時間 or None, 次數, 是否為 '無此條件')。"""
    if "no TB condition" in line:
        return None, 0, True
    if "not detected" in line or "not achieved" in line:
        return None, 0, False
    m = _VALUE_RE.search(line)
    if not m:
        return None, 0, False
    value = float(m.group(1))
    c = _COUNT_RE.search(line.strip())
    return value, (int(c.group(1)) if c else 1), False


def parse_report(path, max_stage=10):
    """解析單一 event_record txt。

    max_stage: 報告要呈現到第幾關（批次版跑 1-10，完整版跑 1-14）。
    回傳 dict，即使檔案缺欄位也保證所有 key 存在，呼叫端不需再做防呆。
    """
    path = str(path)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    result = {
        "path": path,
        "name": os.path.splitext(os.path.basename(path))[0],
        "video": "",
        "total_score": 0,
        "total_gazing_events": 0,
        "scoring_version": "",
        "records": {s: _blank_record(s) for s in range(1, max_stage + 1)},
        "transcript": [],
        "event_log": [],
        "max_stage": max_stage,
    }

    m = re.search(r"^Video:\s*(.+)$", text, re.M)
    if m:
        result["video"] = m.group(1).strip()
        vm = re.search(r"[/\\]([^/\\]+)\.(?:mp4|avi|mov|mkv)$", result["video"], re.I)
        if vm:
            result["name"] = vm.group(1).strip()
    m = re.search(r"^Total Score:\s*(\d+)", text, re.M)
    if m:
        result["total_score"] = int(m.group(1))
    m = re.search(r"^Total Gazing Events:\s*(\d+)", text, re.M)
    if m:
        result["total_gazing_events"] = int(m.group(1))
    m = re.search(r"Scoring Version:\s*(\S+)", text)
    if m:
        result["scoring_version"] = m.group(1)

    # ── 主體：逐行掃描，遇到 "NN. Stage X --" 就切換當前紀錄 ──
    summary_text = text.split("=== Full Event Log ===")[0]
    current = None
    for raw in summary_text.splitlines():
        head = _HEADER_RE.match(raw)
        if head:
            stage = int(head.group(1))
            current = result["records"].setdefault(stage, _blank_record(stage))
            current["detected"] = True
            current["label"] = head.group(2).strip()
            if head.group(3):
                current["level"] = head.group(3).upper()
                current["total"] = int(head.group(4))
            continue
        if current is None:
            continue

        line = raw.strip()
        if line.startswith("T0"):
            v, _, _ = _parse_metric(line)
            current["t0"] = v
        elif line.startswith("Pointing"):
            v, c, _ = _parse_metric(line)
            current["pointing"], current["pointing_count"] = v, c
        elif line.startswith("TB"):
            v, c, na = _parse_metric(line)
            current["tb"], current["tb_count"], current["tb_na"] = v, c, na
        elif line.startswith("TH"):
            v, c, _ = _parse_metric(line)
            current["th"], current["th_count"] = v, c
        elif line.startswith("計分結束"):
            if "未知" not in line:
                m2 = _VALUE_RE.search(line)
                current["end"] = float(m2.group(1)) if m2 else None

    # ── Gazing 統計 ──
    for sm in re.finditer(r"^Stage\s+(\d+)\s*\[.*?\]:\s*(\w+),\s*GazingCount=(\d+)", text, re.M):
        stage = int(sm.group(1))
        rec = result["records"].setdefault(stage, _blank_record(stage))
        rec["gazing_count"] = int(sm.group(3))

    # ── 事件流水 ──
    if "=== Full Event Log ===" in text:
        log_part = text.split("=== Full Event Log ===", 1)[1]
        log_part = log_part.split("=== Whisper 語音辨識逐字稿 ===", 1)[0]
        result["event_log"] = [ln for ln in (l.rstrip() for l in log_part.splitlines())
                               if ln.strip() and not ln.strip().startswith("=")]

    # ── Whisper 逐字稿 ──
    if "=== Whisper 語音辨識逐字稿 ===" in text:
        tr_part = text.split("=== Whisper 語音辨識逐字稿 ===", 1)[1]
        for ln in tr_part.splitlines():
            tm = _TRANSCRIPT_RE.match(ln.strip())
            if tm:
                result["transcript"].append({
                    "start": float(tm.group(1)),
                    "end": float(tm.group(2)),
                    "text": tm.group(3).strip(),
                    "keywords": [k.strip() for k in (tm.group(4) or "").split(",") if k.strip()],
                })

    return result


def level_counts(report):
    """統計各反應等級出現次數，回傳 {HI:n, LI:n, HR:n, LR:n, F:n}。"""
    counts = {lv: 0 for lv in LEVEL_ORDER}
    for rec in report["records"].values():
        if rec["level"] in counts:
            counts[rec["level"]] += 1
    return counts


def scan_reports(folder, max_stage=10):
    """掃描資料夾內所有 event_record txt，回傳解析結果列表（依名稱排序）。

    會略過 Whisper 快取子目錄與明顯不是報告的檔案（用 Result Summary 標頭判斷），
    避免使用者把別的 txt 丟進來就整頁壞掉。
    """
    reports = []
    if not os.path.isdir(folder):
        return reports
    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith(".txt"):
            continue
        fpath = os.path.join(folder, fname)
        try:
            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                head = f.read(400)
            if "Result Summary" not in head:
                continue
            reports.append(parse_report(fpath, max_stage=max_stage))
        except Exception:
            continue

    def _sort_key(r):
        # 影片編號多為數字，數字優先照數值排序，其餘照字串
        return (0, int(r["name"])) if r["name"].isdigit() else (1, 0, r["name"])

    try:
        reports.sort(key=_sort_key)
    except TypeError:
        reports.sort(key=lambda r: r["name"])
    return reports
