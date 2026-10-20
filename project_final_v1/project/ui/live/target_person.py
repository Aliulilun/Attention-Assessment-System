# -*- coding: utf-8 -*-
"""
體驗者選定（多人入鏡時只分析一個人）
====================================
規則：最靠近畫面中心、且人框最大的那個人。

    score = 面積比 × (1 − 中心距離比)
    面積比     = 人框面積 ÷ 畫面面積
    中心距離比 = 人框中心到畫面中心距離 ÷ 畫面半對角線

防跳動：目前目標還在時，別人必須連續 switch_frames 幀分數更高才換人；
目標消失超過 lost_frames 幀才直接改選當下最高分者。

專題的人臉偵測挑「面積最小」的臉（兒童優先），不適合這裡，所以另寫。
"""

import math


def score_box(box, frame_w, frame_h):
    x1, y1, x2, y2 = box
    area_ratio = max(0.0, x2 - x1) * max(0.0, y2 - y1) / float(frame_w * frame_h)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    half_diag = math.hypot(frame_w, frame_h) / 2.0
    dist_ratio = min(1.0, math.hypot(cx - frame_w / 2.0, cy - frame_h / 2.0) / half_diag)
    return area_ratio * (1.0 - dist_ratio)


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / float(area_a + area_b - inter)


class TargetSelector:

    MATCH_IOU = 0.3

    def __init__(self, switch_frames=5, lost_frames=10):
        self.switch_frames = switch_frames
        self.lost_frames = lost_frames
        self.reset()

    def reset(self):
        self._box = None
        self._lost = 0
        self._challenge = 0

    def _adopt(self, boxes, idx):
        self._box = tuple(boxes[idx])
        self._lost = 0
        self._challenge = 0
        return idx

    def update(self, boxes, frame_w, frame_h):
        """回傳本幀體驗者在 boxes 中的索引；看不到體驗者時回傳 None。"""
        boxes = [tuple(b) for b in boxes]
        scores = [score_box(b, frame_w, frame_h) for b in boxes]
        best = max(range(len(boxes)), key=scores.__getitem__) if boxes else None

        if self._box is None:
            return None if best is None else self._adopt(boxes, best)

        matched = None
        if boxes:
            ious = [_iou(self._box, b) for b in boxes]
            cand = max(range(len(boxes)), key=ious.__getitem__)
            if ious[cand] >= self.MATCH_IOU:
                matched = cand

        if matched is None:
            self._lost += 1
            self._challenge = 0
            if self._lost > self.lost_frames:
                self._box = None
                return None if best is None else self._adopt(boxes, best)
            return None

        self._lost = 0
        self._box = boxes[matched]
        if best != matched and scores[best] > scores[matched]:
            self._challenge += 1
            if self._challenge >= self.switch_frames:
                return self._adopt(boxes, best)
        else:
            self._challenge = 0
        return matched
