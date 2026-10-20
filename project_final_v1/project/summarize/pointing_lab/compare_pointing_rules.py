# -*- coding: utf-8 -*-
"""
指向規則對照：專題原版 vs 即時體驗版（用有標籤的手勢資料）
==========================================================
讀 collect_hand_gestures.py 錄的 .npz，對每一種動作計算：
  - 專題原版 modules/interaction.py 的 is_valid_pointing()（只用幾何，
    不含皮膚顏色／前臂檢查——資料只有關鍵點）與它擋掉的原因
  - 即時體驗版 ui/live/live_pointing.py 的 is_pointing_pose()
  - 3D 長度比門檻掃描

只讀取兩邊的判定函式，不修改任何程式。每段動作的前 25 幀（約 1 秒）
是從上一個動作切換過來的過渡期，統計時排除。

執行：
    conda activate mediapipe_py39
    python summarize/pointing_lab/compare_pointing_rules.py [資料.npz ...]
（不給檔名時讀本資料夾所有 hand_gesture_labeled_*.npz）
"""
import collections
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, PROJECT)
sys.path.insert(0, os.path.join(PROJECT, "ui"))

from modules.interaction import InteractionEngine          # noqa: E402
from live.live_pointing import is_pointing_pose, EXT_RATIO, _angle_deg  # noqa: E402

ORDER = ["point_left", "point_right", "point_up", "point_left2", "point_right2",
         "gun", "open_palm", "fist", "v_sign", "relaxed"]
TRANSITION_FRAMES = 25
FRAME_W, FRAME_H = 1280, 720      # collect_hand_gestures.py 的鏡頭解析度


class _LegacyLms:
    """對應 InteractionEngine 期待的 landmarks.landmark[i].x/.y（正規化座標）。"""

    def __init__(self, pts):
        self.landmark = [type("P", (), {"x": x / FRAME_W, "y": y / FRAME_H})() for x, y in pts]


def load(paths):
    labels, p2, p3, fid = [], [], [], []
    for k, p in enumerate(paths):
        d = np.load(p)
        labels.append(d["labels"]); p2.append(d["p2"]); p3.append(d["p3"])
        fid.append(np.full(len(d["labels"]), k))
    return np.concatenate(labels), np.concatenate(p2), np.concatenate(p3), np.concatenate(fid)


def steady_mask(labels, file_ids):
    """每個檔案、每種動作各自從 0 編號，前 TRANSITION_FRAMES 幀視為切換期。"""
    pos = np.zeros(len(labels), int)
    for f in set(file_ids):
        for lab in set(labels[file_ids == f]):
            m = (labels == lab) & (file_ids == f)
            pos[m] = np.arange(m.sum())
    return pos >= TRANSITION_FRAMES


def row(name, ok, labels, steady):
    cells = []
    for lab in ORDER:
        m = (labels == lab) & steady
        cells.append("%s %3.0f%%" % (lab, 100 * ok[m].mean()) if m.any() else "%s   —" % lab)
    print("%-26s %s" % (name, "  ".join(cells)))


def main():
    paths = sys.argv[1:] or sorted(glob.glob(os.path.join(HERE, "hand_gesture_labeled_*.npz")))
    if not paths:
        sys.exit("找不到資料檔，請先執行 collect_hand_gestures.py")
    labels, p2, p3, file_ids = load(paths)
    steady = steady_mask(labels, file_ids)
    print("資料：%s，共 %d 筆（統計排除每段前 %d 幀）\n" % (", ".join(map(os.path.basename, paths)),
                                                       len(labels), TRANSITION_FRAMES))

    eng = object.__new__(InteractionEngine)        # 不載模型，只用純幾何判定
    orig = np.zeros(len(labels), bool)
    reasons = collections.defaultdict(collections.Counter)
    for i in range(len(labels)):
        orig[i] = eng.is_valid_pointing(_LegacyLms(p2[i]), FRAME_W, FRAME_H, "Child",
                                        frame=None, arm_vec=None) is not None
        if steady[i]:
            reasons[labels[i]][eng.last_reject_reason or "通過"] += 1
    live3d = np.array([is_pointing_pose(a, b) for a, b in zip(p2, p3)])
    live2d = np.array([is_pointing_pose(a) for a in p2])

    print("【成立率】指向類（point_*）越高越好；gun 未定；其餘越低越好")
    row("專題原版（幾何部分）", orig, labels, steady)
    row("即時版 2D 彎曲（舊）", live2d, labels, steady)
    row("即時版 3D 長度比 %.2f" % EXT_RATIO, live3d, labels, steady)

    print("\n【專題原版被擋原因】")
    for lab in ORDER:
        print("  %-13s %s" % (lab, dict(reasons[lab].most_common(5))))

    print("\n【3D 長度比門檻掃描】（食指 2D 伸直 + 其他三指 3D 長度 < t × 食指）")
    idx_len = np.linalg.norm(p3[:, 8] - p3[:, 5], axis=-1)
    ext = np.max([np.linalg.norm(p3[:, t] - p3[:, m], axis=-1)
                  for m, t in ((9, 12), (13, 16), (17, 20))], axis=0) / np.maximum(idx_len, 1e-9)
    straight = np.array([_angle_deg(a[5], a[6], a[7]) > 160 and _angle_deg(a[6], a[7], a[8]) > 160
                         for a in p2])
    for t in (0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9):
        row("t = %.2f" % t, straight & (ext < t), labels, steady)


if __name__ == "__main__":
    main()
