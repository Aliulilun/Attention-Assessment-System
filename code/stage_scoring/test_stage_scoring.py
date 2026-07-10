"""
計分模組測試腳本

讀取 output/ 下既有的報告文字檔（例如 49.txt、52.txt），套用
modules/stage_scoring.py 的計分邏輯，在每個 stage 標題行後方補上
反應等級與分數，並輸出成 <原檔名>_scored.txt 供檢視驗證。

用法：
    python test_stage_scoring.py 49 52
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modules.stage_scoring import compute_stage_score

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

BLOCK_PATTERN = re.compile(
    r"^(\d+\. Stage (\d+) -- (.+))\n"
    r"(    T0.*\n)"
    r"(    Pointing\s*=\s*(.+)\n)"
    r"(    TB\s*=\s*(.+)\n)"
    r"(    TH\s*=\s*(.+)\n)",
    re.MULTILINE,
)


def apply_scoring(text):
    def replace(m):
        header, stage, label = m.group(1), m.group(2), m.group(3)
        t0_line = m.group(4)
        pointing_line, pointing_val = m.group(5), m.group(6)
        tb_line, tb_val = m.group(7), m.group(8)
        th_line, th_val = m.group(9), m.group(10)

        record = {
            "label": label,
            "pointing_t": 0.0 if "not detected" not in pointing_val else None,
            "tb": 0.0 if ("not achieved" not in tb_val and "no TB condition" not in tb_val) else None,
            "th": 0.0 if "not achieved" not in th_val else None,
        }
        total, level = compute_stage_score(record)
        new_header = header + " -- 反應等級 " + level + "(" + str(total) + "分)"
        return new_header + "\n" + t0_line + pointing_line + tb_line + th_line

    return BLOCK_PATTERN.sub(replace, text)


def main(names):
    for name in names:
        src_path = os.path.join(OUTPUT_DIR, name + ".txt")
        if not os.path.exists(src_path):
            print(f"[SKIP] not found: {src_path}")
            continue
        with open(src_path, "r", encoding="utf-8") as f:
            text = f.read()

        scored_text = apply_scoring(text)

        dst_path = os.path.join(OUTPUT_DIR, name + "_scored.txt")
        with open(dst_path, "w", encoding="utf-8") as f:
            f.write(scored_text)
        print(f"[OK] {src_path} -> {dst_path}")


if __name__ == "__main__":
    names = sys.argv[1:] or ["49", "52"]
    main(names)
