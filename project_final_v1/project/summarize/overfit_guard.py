# -*- coding: utf-8 -*-
"""
防過度擬合驗證流程（overfit guard）
====================================

為什麼需要
----------
過去的修改流程是「看幾支影片的失誤 → 寫規則 / 調門檻 → 用同一批影片驗收」，
等於拿考卷讀書再拿同一張考卷打分數，成績一定偏高。本工具把驗證拆成兩組並
強制以下紀律：

1. **開發組 / 測試組分離**：調規則、看失誤只能用開發組（`dev`）。
   測試組在 `dev` 報告中一律隱藏，連檔名都不列。
2. **先凍結、再驗收**：`freeze` 把目前程式碼（modules/、hurry/main.py、
   main.py、config.yaml）算成雜湊存檔；`test` 只接受與某次凍結完全相同的
   程式碼，並把每一次偷看測試組記錄下來，報告時必須揭露次數。
3. **雜訊門檻**：同一份程式重跑本來就有 ±1 次的差異（GPU 非決定性）。
   前後比較時，變化量沒有超出雜訊範圍的一律標為「雜訊內」，不算改善。
4. **單支影片集中度**：淨改善若大部分來自同一支影片，很可能只是替那支影片
   量身訂做，會亮警告。
5. **留一交叉驗證（lovo）**：替門檻型規則檢查「換一組影片，挑出來的門檻
   還一樣嗎」。

指向真相
--------
對照表的 Pointing 欄已確認不可信（見 docs/指向人工標註基準.md），
預設 `pointing_source = "truth_only"`：只用設定檔 `pointing_truth` 裡人工覆核過
的 (影片, 關卡)，其餘格子不計入指向指標。覆核出新的格子時直接加進設定檔。

用法
----
    conda activate mediapipe_py39
    cd C:\\project

    # 0) 第一次：決定測試組（之後不要再改）
    python summarize\\overfit_guard.py init --test 11 64 66

    # 1) 開發期間：只看開發組；附上重跑資料夾可估雜訊
    python summarize\\overfit_guard.py dev --txt-dir hurry\\output_new ^
        --baseline-dir hurry\\output_before2 ^
        --repeat-dirs hurry\\output_new_run2 hurry\\output_new_run3

    # 2) 門檻型規則：留一交叉驗證（sweep CSV 由模擬腳本產生）
    python summarize\\overfit_guard.py lovo --sweep sweep_p5.csv

    # 3) 規則定案：凍結程式碼
    python summarize\\overfit_guard.py freeze --name P5-1 --note "WRIST_CLOSE_RATIO=0.35"

    # 4) 用凍結版本重跑測試組影片後，驗收一次
    python summarize\\overfit_guard.py test --txt-dir hurry\\output_test

    # 隨時查看目前切分、凍結紀錄、測試組被看過幾次
    python summarize\\overfit_guard.py status

設定檔 summarize/overfit_guard.json 是紀錄本身，請一併保存、不要手動改測試組。

sweep CSV 格式（lovo 用）
------------------------
    video,threshold,tp,fp,fn
    42,0.30,1,0,0
    42,0.35,1,0,0
    87,0.30,0,3,0
    ...
每一列 = 某支影片在某個門檻下的計數。模擬腳本逐影片、逐門檻輸出即可。
"""

import argparse
import csv
import datetime
import glob
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import compare_with_manual as cwm  # noqa: E402

PROJECT_DIR = cwm.PROJECT_DIR
CONFIG_PATH = os.path.join(HERE, "overfit_guard.json")

# 會影響判定結果的程式碼；凍結時對這些檔案算雜湊
CODE_GLOBS = [
    "modules/**/*.py",
    "hurry/main.py",
    "main.py",
    "config.yaml",
]

METRICS = ("TB", "TH", "Pointing")
DEFAULT_NOISE = 1          # 沒有重跑資料時的預設雜訊（同機同碼重跑實測 ±1）
CONCENTRATION_WARN = 0.7   # 淨改善有 70% 以上來自同一支影片就警告

# 已人工覆核的指向真相（docs/指向人工標註基準.md，2026-09-22）
DEFAULT_POINTING_TRUTH = [
    {"video": 42, "stage": 4, "pointing": 1, "source": "D-01 舉手指向牆上的球"},
    {"video": 49, "stage": 7, "pointing": 0, "source": "覆核：無指向"},
    {"video": 52, "stage": 7, "pointing": 0, "source": "覆核：無指向"},
    {"video": 87, "stage": 7, "pointing": 0, "source": "覆核：無指向"},
    {"video": 42, "stage": 9, "pointing": 0, "source": "覆核：無指向"},
    {"video": 90, "stage": 7, "pointing": 0, "source": "覆核：無指向"},
]


# ============================================================
# 設定檔
# ============================================================
def _now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def load_config(path=None):
    path = path or CONFIG_PATH
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_config(cfg, path=None):
    path = path or CONFIG_PATH
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def split_hash(dev, test):
    s = "dev:%s|test:%s" % (",".join(map(str, sorted(dev))),
                            ",".join(map(str, sorted(test))))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def check_split_integrity(cfg):
    """測試組被手動改過就回傳 False。"""
    return split_hash(cfg["dev"], cfg["test"]) == cfg.get("split_hash")


# ============================================================
# 程式碼雜湊
# ============================================================
def code_files(root=None):
    root = root or PROJECT_DIR
    files = set()
    for pattern in CODE_GLOBS:
        for p in glob.glob(os.path.join(root, pattern), recursive=True):
            if os.path.isfile(p) and "__pycache__" not in p:
                files.add(os.path.relpath(p, root).replace("\\", "/"))
    return sorted(files)


def code_hash(root=None):
    """回傳 (總雜湊, {相對路徑: 單檔雜湊})。換行統一成 \\n，避免 CRLF 造成假差異。"""
    root = root or PROJECT_DIR
    per_file = {}
    total = hashlib.sha256()
    for rel in code_files(root):
        with open(os.path.join(root, rel), "rb") as f:
            data = f.read().replace(b"\r\n", b"\n")
        h = hashlib.sha256(data).hexdigest()[:16]
        per_file[rel] = h
        total.update(rel.encode("utf-8") + b"\0" + h.encode("ascii") + b"\n")
    return total.hexdigest()[:16], per_file


# ============================================================
# 資料載入（套用切分與指向真相）
# ============================================================
def apply_pointing_truth(manual, cfg):
    source = cfg.get("pointing_source", "truth_only")
    if source == "truth_only":
        for per_stage in manual.values():
            for cell in per_stage.values():
                cell["pt"] = None
    for t in cfg.get("pointing_truth", []):
        cid, stage = int(t["video"]), int(t["stage"])
        if cid in manual and stage in manual[cid]:
            manual[cid][stage]["pt"] = int(t["pointing"])
    return manual


def load_manual(cfg):
    manual, groups, colmap = cwm.load_manual(cfg.get("xlsx", cwm.DEFAULT_XLSX))
    return apply_pointing_truth(manual, cfg), groups, colmap


def filter_ids(ai_data, keep):
    keep = set(keep)
    return {cid: v for cid, v in ai_data.items() if cid in keep}


# ============================================================
# 分析：雜訊、逐影片貢獻、集中度
# ============================================================
def counts(overall):
    return {k: (overall[k].tp, overall[k].fp, overall[k].fn) for k in METRICS}


def noise_band(count_sets):
    """count_sets = 同一份程式碼多次重跑的 counts() 清單。
    回傳 {指標: (TP 全距, FP 全距, FN 全距)}；少於兩次時回傳 None。"""
    if len(count_sets) < 2:
        return None
    band = {}
    for k in METRICS:
        band[k] = tuple(
            max(c[k][i] for c in count_sets) - min(c[k][i] for c in count_sets)
            for i in range(3))
    return band


def judge_delta(delta, noise):
    """delta 為整數變化量；絕對值不超過雜訊範圍就視為雜訊。"""
    if delta == 0:
        return "無變化"
    if abs(delta) <= noise:
        return "雜訊內"
    return "超出雜訊"


def per_video_errors(ai_data, manual, groups, stages):
    """回傳 {影片: {指標: FP + FN}}。"""
    out = {}
    for cid in ai_data:
        if cid not in manual:
            continue
        overall, _, _ = cwm.evaluate({cid: ai_data[cid]}, manual, groups, stages)
        out[cid] = {k: overall[k].fp + overall[k].fn for k in METRICS}
    return out


def concentration(cur_err, base_err, metric):
    """回傳 (淨改善量, 貢獻最大的影片, 佔比)。
    淨改善 = baseline 錯誤數 − 目前錯誤數（正值 = 變好）。沒有淨改善時佔比為 None。"""
    gains = {}
    for cid in set(cur_err) & set(base_err):
        gains[cid] = base_err[cid][metric] - cur_err[cid][metric]
    net = sum(gains.values())
    if net <= 0 or not gains:
        return net, None, None
    top = max(gains, key=lambda c: gains[c])
    return net, top, gains[top] / float(net)


# ============================================================
# 留一交叉驗證
# ============================================================
def _f1(tp, fp, fn):
    d = 2 * tp + fp + fn
    return (2.0 * tp / d) if d else 1.0   # 全部都沒有 = 完全正確


def _objective(tp, fp, fn):
    # F1 優先；同分時 FP 少者優先，再來 FN 少者
    return (_f1(tp, fp, fn), -fp, -fn)


def read_sweep(path):
    """回傳 {影片: {門檻: (tp, fp, fn)}}。"""
    table = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            cid = int(float(row["video"]))
            thr = float(row["threshold"])
            table.setdefault(cid, {})[thr] = (
                int(float(row["tp"])), int(float(row["fp"])), int(float(row["fn"])))
    return table


def best_threshold(table, videos):
    thresholds = sorted(set.intersection(*[set(table[v]) for v in videos]))
    if not thresholds:
        return None
    best, best_key = None, None
    for thr in thresholds:
        tp = sum(table[v][thr][0] for v in videos)
        fp = sum(table[v][thr][1] for v in videos)
        fn = sum(table[v][thr][2] for v in videos)
        key = _objective(tp, fp, fn)
        if best_key is None or key > best_key:
            best, best_key = thr, key
    return best


def lovo(table):
    """回傳 (全體最佳門檻, [(拿掉的影片, 挑到的門檻, 該影片的 (tp,fp,fn))])。"""
    videos = sorted(table)
    overall_best = best_threshold(table, videos)
    folds = []
    for held in videos:
        rest = [v for v in videos if v != held]
        if not rest:
            continue
        thr = best_threshold(table, rest)
        result = table[held].get(thr) if thr is not None else None
        folds.append((held, thr, result))
    return overall_best, folds


# ============================================================
# 子命令
# ============================================================
def require_config():
    cfg = load_config()
    if cfg is None:
        print("❌ 尚未建立切分，請先執行：python summarize\\overfit_guard.py init --test <編號...>")
        sys.exit(1)
    if not check_split_integrity(cfg):
        print("❌ overfit_guard.json 的開發組 / 測試組被手動改過（雜湊不符）。")
        print("   測試組一旦改動，之前所有測試成績都不再有效。")
        print("   若確定要重切，請用 init --force，會留下重切紀錄。")
        sys.exit(1)
    return cfg


def cmd_init(args):
    old = load_config()
    if old is not None and not args.force:
        print("❌ 已經有切分了（測試組：%s）。" % old["test"])
        print("   重切會讓之前的測試成績失效；確定要重切請加 --force。")
        return 1

    manual, _, _ = cwm.load_manual(args.xlsx)
    test = sorted(set(args.test))
    if args.dev:
        dev = sorted(set(args.dev))
    else:
        dev = sorted(set(manual) - set(test))
    overlap = set(dev) & set(test)
    if overlap:
        print("❌ 開發組與測試組重疊：%s" % sorted(overlap))
        return 1
    unknown = [c for c in test if c not in manual]
    if unknown:
        print("⚠️ 這些測試組編號不在對照表裡，將無法評分：%s" % unknown)

    cfg = {
        "created": _now(),
        "xlsx": args.xlsx,
        "dev": dev,
        "test": test,
        "split_hash": split_hash(dev, test),
        "pointing_source": "truth_only",
        "pointing_truth": (old or {}).get("pointing_truth", DEFAULT_POINTING_TRUTH),
        "default_noise": DEFAULT_NOISE,
        "freezes": [],
        "test_runs": [],
        "history": (old or {}).get("history", []),
    }
    if old is not None:
        cfg["history"].append({
            "time": _now(), "event": "resplit",
            "old_test": old["test"], "new_test": test,
            "old_test_runs": len(old.get("test_runs", [])),
        })
    save_config(cfg)
    print(">>> 已建立切分：%s" % CONFIG_PATH)
    print("    測試組（鎖定，只在 test 子命令使用）：%s" % test)
    print("    開發組：%d 位" % len(dev))
    return 0


def cmd_status(args):
    cfg = require_config()
    cur_hash, _ = code_hash()
    print("切分建立：%s" % cfg["created"])
    print("測試組：%s" % cfg["test"])
    print("開發組：%d 位" % len(cfg["dev"]))
    print("指向真相：%s，%d 格人工覆核"
          % (cfg.get("pointing_source"), len(cfg.get("pointing_truth", []))))
    print("")
    print("目前程式碼雜湊：%s" % cur_hash)
    freezes = cfg.get("freezes", [])
    if not freezes:
        print("凍結紀錄：（無）")
    for fz in freezes:
        mark = "  ← 目前程式碼" if fz["code_hash"] == cur_hash else ""
        print("  [%s] %s  %s  %s%s" % (fz["time"], fz["name"], fz["code_hash"],
                                      fz.get("note", ""), mark))
    runs = cfg.get("test_runs", [])
    distinct = len(set(r["code_hash"] for r in runs))
    print("")
    print("測試組已驗收 %d 次（%d 個不同版本）" % (len(runs), distinct))
    if distinct > 1:
        print("⚠️ 測試組已被多個版本看過。第 2 個版本以後的測試成績會偏樂觀，")
        print("   報告中必須揭露這個次數；如果超過 3 次，建議換一批沒看過的影片。")
    for h in cfg.get("history", []):
        print("⚠️ 曾經重切：%s（舊測試組 %s 已驗收 %d 次）"
              % (h["time"], h["old_test"], h["old_test_runs"]))
    return 0


def load_results(txt_dir):
    """讀取 event_record；空檔或解析不到任何關卡的（例如中斷的重跑留下 0 byte 檔）
    直接剔除，否則會被當成「全部沒偵測到」而扭曲前後比較。"""
    raw = cwm.load_txt_dir(txt_dir)
    empty = sorted(c for c, v in raw.items() if not v)
    if empty:
        print("⚠️ %s 裡這些結果檔是空的或無法解析，已略過：%s" % (txt_dir, empty))
    return {c: v for c, v in raw.items() if v}


def _load_dev(txt_dir, cfg):
    raw = load_results(txt_dir)
    hidden = [c for c in raw if c in set(cfg["test"])]
    return filter_ids(raw, cfg["dev"]), len(hidden)


def cmd_dev(args):
    cfg = require_config()
    manual, groups, _ = load_manual(cfg)
    stages = args.stages

    cur, hidden = _load_dev(args.txt_dir, cfg)
    if hidden:
        print("🔒 %s 裡有 %d 份測試組結果，已隱藏不計。" % (args.txt_dir, hidden))
    if not cur:
        print("❌ 沒有開發組的 event_record")
        return 1

    overall, per_stage, by_group = cwm.evaluate(cur, manual, groups, stages)
    cwm.print_report("開發組：%s" % args.txt_dir,
                     overall, per_stage, by_group, cur, manual, stages)
    print("")
    print("指向真相只採人工覆核的 %d 格（對照表 Pointing 欄不可信）"
          % len(cfg.get("pointing_truth", [])))

    # ---- 雜訊 ----
    # 每次重跑都只算大家都有的影片，否則缺檔會被誤當成雜訊
    runs = [cur] + [_load_dev(d, cfg)[0] for d in (args.repeat_dirs or [])]
    shared = set(cur)
    for r in runs[1:]:
        shared &= set(r)
    repeat_counts = []
    for r in runs:
        o, _, _ = cwm.evaluate(filter_ids(r, shared), manual, groups, stages)
        repeat_counts.append(counts(o))
    band = noise_band(repeat_counts)
    default_noise = cfg.get("default_noise", DEFAULT_NOISE)

    print("")
    print("=" * 72)
    print("  雜訊範圍（同一份程式碼重跑的全距）")
    print("=" * 72)
    if band is None:
        print("  未提供 --repeat-dirs，採預設雜訊 ±%d。" % default_noise)
        print("  建議同一版本至少重跑 3 次，雜訊才估得準。")
    else:
        print("  共 %d 次重跑，以共同的 %d 支影片估計" % (len(repeat_counts), len(shared)))
        for k in METRICS:
            print("  %-9s TP ±%d  FP ±%d  FN ±%d" % ((k,) + band[k]))

    if not args.baseline_dir:
        return 0

    base, _ = _load_dev(args.baseline_dir, cfg)
    common = sorted(set(cur) & set(base))
    if not common:
        print("❌ 基準資料夾沒有同編號的開發組結果")
        return 1
    cur_c = filter_ids(cur, common)
    base_c = filter_ids(base, common)
    o_cur, _, _ = cwm.evaluate(cur_c, manual, groups, stages)
    o_base, _, _ = cwm.evaluate(base_c, manual, groups, stages)

    print("")
    print("=" * 72)
    print("  與基準比較：%s（%d 支共同影片）" % (args.baseline_dir, len(common)))
    print("=" * 72)
    print("%-9s %-6s %6s %6s  %s" % ("指標", "", "基準", "目前", "判定"))
    verdicts = []
    for k in METRICS:
        for i, name in enumerate(("TP", "FP", "FN")):
            b = (o_base[k].tp, o_base[k].fp, o_base[k].fn)[i]
            c = (o_cur[k].tp, o_cur[k].fp, o_cur[k].fn)[i]
            noise = band[k][i] if band is not None else default_noise
            v = judge_delta(c - b, noise)
            if v == "超出雜訊":
                good = (c > b) if name == "TP" else (c < b)
                v += "（變好）" if good else "（變差）"
            verdicts.append((k, name, v))
            print("%-9s %-6s %6d %6d  %s" % (k, name, b, c, v))

    # ---- 集中度 ----
    cur_err = per_video_errors(cur_c, manual, groups, stages)
    base_err = per_video_errors(base_c, manual, groups, stages)
    print("")
    print("── 改善集中度（錯誤數 FP+FN 的減少來自哪幾支影片）──")
    for k in METRICS:
        net, top, share = concentration(cur_err, base_err, k)
        if share is None:
            print("  %-9s 淨改善 %+d" % (k, net))
            continue
        warn = "  ⚠️ 過度集中，可能只是替這支影片量身訂做" if share >= CONCENTRATION_WARN and net >= 2 else ""
        print("  %-9s 淨改善 %+d，最大貢獻 %d 號（%.0f%%）%s"
              % (k, net, top, share * 100, warn))

    worse = [cid for cid in common
             if any(cur_err[cid][k] > base_err[cid][k] for k in METRICS)]
    if worse:
        print("  有影片變差：%s（改善別支的同時弄壞這些，要逐一確認）"
              % ", ".join(str(c) for c in worse))

    print("")
    real = [v for v in verdicts if v[2].startswith("超出雜訊")]
    if not real:
        print("📌 結論：所有變化都在雜訊範圍內，不能宣稱這次修改有效。")
    else:
        print("📌 超出雜訊的變化：%s"
              % "、".join("%s %s %s" % v for v in real))
        print("   開發組的改善只代表「在看過的影片上有效」，正式成效以 test 為準。")
    return 0


def cmd_freeze(args):
    cfg = require_config()
    h, per_file = code_hash()
    for fz in cfg["freezes"]:
        if fz["code_hash"] == h:
            print("此版本程式碼已凍結過：%s（%s）" % (fz["name"], fz["time"]))
            return 0
    cfg["freezes"].append({
        "time": _now(), "name": args.name, "note": args.note or "",
        "code_hash": h, "files": per_file,
    })
    save_config(cfg)
    print(">>> 已凍結：%s  雜湊 %s（%d 個檔案）" % (args.name, h, len(per_file)))
    print("    接著用這份程式碼重跑測試組影片，再執行 test。")
    print("    凍結後若再改程式碼（包含調門檻），test 會拒絕執行。")
    return 0


def _diff_files(old, new):
    changed = sorted(p for p in set(old) | set(new) if old.get(p) != new.get(p))
    return changed


def cmd_test(args):
    cfg = require_config()
    h, per_file = code_hash()
    frozen = [fz for fz in cfg["freezes"] if fz["code_hash"] == h]
    if not frozen:
        print("❌ 目前程式碼沒有凍結紀錄，拒絕看測試組。")
        if cfg["freezes"]:
            last = cfg["freezes"][-1]
            changed = _diff_files(last["files"], per_file)
            print("   與最近一次凍結（%s）相比，改動了：" % last["name"])
            for p in changed[:20]:
                print("     - %s" % p)
        print("   請先確定規則與門檻都不再修改，執行 freeze，再重跑測試組。")
        return 1
    fz = frozen[-1]

    raw = load_results(args.txt_dir)
    test_data = filter_ids(raw, cfg["test"])
    if not test_data:
        print("❌ %s 裡沒有測試組的 event_record（測試組：%s）" % (args.txt_dir, cfg["test"]))
        return 1
    missing = [c for c in cfg["test"] if c not in test_data]
    if missing:
        print("⚠️ 測試組缺少：%s（成績只涵蓋部分測試組）" % missing)

    # 結果檔必須在凍結之後產生，否則跑的可能是舊程式碼
    fz_time = datetime.datetime.fromisoformat(fz["time"]).timestamp()
    stale = []
    for cid in test_data:
        p = os.path.join(args.txt_dir, "%d.txt" % cid)
        if os.path.exists(p) and os.path.getmtime(p) < fz_time:
            stale.append(cid)
    if stale:
        print("❌ 這些結果檔比凍結時間還舊，不是用凍結版本跑的：%s" % stale)
        print("   請用目前程式碼重跑後再驗收。")
        return 1

    prior_versions = set(r["code_hash"] for r in cfg["test_runs"])
    manual, groups, _ = load_manual(cfg)
    overall, per_stage, by_group = cwm.evaluate(test_data, manual, groups, args.stages)
    cwm.print_report("測試組（凍結版本 %s）" % fz["name"],
                     overall, per_stage, by_group, test_data, manual, args.stages)

    cfg["test_runs"].append({
        "time": _now(), "freeze": fz["name"], "code_hash": h,
        "txt_dir": args.txt_dir, "videos": sorted(test_data),
        "counts": {k: list(v) for k, v in counts(overall).items()},
    })
    save_config(cfg)

    versions = len(prior_versions | {h})
    print("")
    print("📌 已記錄這次驗收。測試組至今被 %d 個版本看過。" % versions)
    if h in prior_versions:
        print("   （同一凍結版本重看，不會增加偏差。）")
    elif versions > 1:
        print("⚠️ 測試組不是第一次被看。如果這次修改是因為看了上次的測試成績才做的，")
        print("   測試組就已經變成開發組了。報告必須揭露驗收次數。")
    return 0


def cmd_lovo(args):
    cfg = load_config()
    table = read_sweep(args.sweep)
    if cfg is not None:
        leaked = sorted(set(table) & set(cfg["test"]))
        if leaked:
            print("🔒 sweep 含測試組影片 %s，已剔除（測試組不能用來挑門檻）。" % leaked)
            for c in leaked:
                del table[c]
    if len(table) < 3:
        print("❌ 至少需要 3 支影片才做得了留一交叉驗證")
        return 1

    overall_best, folds = lovo(table)
    print("全部 %d 支影片一起挑：最佳門檻 = %s" % (len(table), overall_best))
    print("")
    print("%-8s %10s   %s" % ("拿掉", "挑到門檻", "被拿掉那支的結果 (TP/FP/FN)"))
    tp = fp = fn = 0
    for held, thr, res in folds:
        if res is None:
            print("%-8d %10s   （該門檻沒有資料）" % (held, thr))
            continue
        tp, fp, fn = tp + res[0], fp + res[1], fn + res[2]
        print("%-8d %10s   %d/%d/%d" % (held, thr, res[0], res[1], res[2]))

    in_tp = sum(table[v][overall_best][0] for v in table)
    in_fp = sum(table[v][overall_best][1] for v in table)
    in_fn = sum(table[v][overall_best][2] for v in table)
    print("")
    print("樣本內（用全部影片挑、也用全部影片算）：TP %d  FP %d  FN %d  F1 %.2f"
          % (in_tp, in_fp, in_fn, _f1(in_tp, in_fp, in_fn)))
    print("留一外推（每支都用別支挑的門檻）   ：TP %d  FP %d  FN %d  F1 %.2f"
          % (tp, fp, fn, _f1(tp, fp, fn)))

    chosen = sorted(set(t for _, t, _ in folds if t is not None))
    all_thr = sorted(set.intersection(*[set(v) for v in table.values()]))
    print("")
    if len(chosen) == 1:
        print("📌 門檻穩定：換掉任何一支影片，挑出來的都是 %s。" % chosen[0])
    else:
        idx = [all_thr.index(t) for t in chosen]
        span = max(idx) - min(idx)
        print("📌 挑到的門檻有 %d 種：%s" % (len(chosen), chosen))
        if span > 1:
            print("⚠️ 門檻隨影片大幅變動，這條規則很可能是在配合特定影片。")
            movers = [h for h, t, _ in folds if t != overall_best]
            print("   拿掉這些影片時門檻會變：%s" % movers)
    gap = _f1(in_tp, in_fp, in_fn) - _f1(tp, fp, fn)
    if gap > 0.1:
        print("⚠️ 樣本內 F1 比留一外推高 %.2f，樣本內成績被高估。" % gap)
    return 0


# ============================================================
# 入口
# ============================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description="防過度擬合驗證流程")
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("init", help="建立開發組 / 測試組切分")
    p.add_argument("--test", type=int, nargs="+", required=True, help="測試組影片編號")
    p.add_argument("--dev", type=int, nargs="+", default=None,
                   help="開發組影片編號（省略 = 對照表中其餘全部）")
    p.add_argument("--xlsx", default=cwm.DEFAULT_XLSX)
    p.add_argument("--force", action="store_true", help="覆蓋既有切分（會留下紀錄）")

    sub.add_parser("status", help="顯示切分、凍結紀錄與測試組驗收次數")

    p = sub.add_parser("dev", help="只用開發組評估，附雜訊與集中度檢查")
    p.add_argument("--txt-dir", required=True)
    p.add_argument("--baseline-dir", default=None)
    p.add_argument("--repeat-dirs", nargs="*", default=None,
                   help="同一份程式碼的其他重跑結果，用來估雜訊")
    p.add_argument("--stages", type=int, nargs="+", default=cwm.STAGES)

    p = sub.add_parser("freeze", help="凍結目前程式碼")
    p.add_argument("--name", required=True, help="版本名稱，例如 P5-1")
    p.add_argument("--note", default="", help="門檻值等說明")

    p = sub.add_parser("test", help="用凍結版本驗收測試組（會記錄次數）")
    p.add_argument("--txt-dir", required=True)
    p.add_argument("--stages", type=int, nargs="+", default=cwm.STAGES)

    p = sub.add_parser("lovo", help="門檻型規則的留一交叉驗證")
    p.add_argument("--sweep", required=True, help="video,threshold,tp,fp,fn 的 CSV")

    args = ap.parse_args(argv)
    handlers = {"init": cmd_init, "status": cmd_status, "dev": cmd_dev,
                "freeze": cmd_freeze, "test": cmd_test, "lovo": cmd_lovo}
    if args.cmd not in handlers:
        ap.print_help()
        return 1
    return handlers[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
