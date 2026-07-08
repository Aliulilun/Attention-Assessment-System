import json
import os
import numpy as np

# ==========================================
# 冷卻時間常數（防 YOLO 掉偵測導致重複計次）
# ==========================================
TB_COOLDOWN_SEC      = 1.5   # TB 兩次計次之間的最小間隔（秒）
TH_COOLDOWN_SEC      = 1.5   # TH 兩次計次之間的最小間隔（秒）
POINTING_COOLDOWN_SEC = 0.8  # Pointing 兩次計次之間的最小間隔（秒）


def load_keyword_trigger_windows_from_cache(cache_path, target_keywords=("你看", "畫一幅畫")):
    if not os.path.exists(cache_path):
        return []
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    windows = []
    for record in data.get("segment_records", []):
        for event in record.get("trigger_events", []):
            if not any(k in event.get("keywords", []) for k in target_keywords):
                continue
            trigger_window = event.get("trigger_window")
            if trigger_window and len(trigger_window) == 2:
                windows.append((float(trigger_window[0]), float(trigger_window[1])))
    windows.sort(key=lambda item: item[0])
    return windows


def load_speech_events_from_cache(cache_path):
    if not os.path.exists(cache_path):
        return []
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    events = []
    for record in data.get("segment_records", []):
        text = record.get("text", "")
        for event in record.get("trigger_events", []):
            try:
                start_time = float(event.get("start"))
            except (TypeError, ValueError):
                continue
            trigger_window = event.get("trigger_window") or [start_time, start_time + 3.0]
            events.append({
                "id": event.get("id", ""),
                "event_type": event.get("event_type", "speech"),
                "keywords": event.get("keywords", []),
                "text": text,
                "start": start_time,
                "end": float(event.get("end", start_time)),
                "window_start": float(trigger_window[0]),
                "window_end": float(trigger_window[1]),
            })
    events.sort(key=lambda item: item["start"])
    return events


def load_noise_events_from_cache(cache_path):
    if not os.path.exists(cache_path):
        return []
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    events = []
    for event in data.get("noise_events", []):
        try:
            events.append({
                "id": event.get("id", ""),
                "event_type": "noise",
                "keywords": ["怪聲"],
                "text": event.get("label", "noise"),
                "start": float(event.get("start")),
                "end": float(event.get("end")),
                "window_start": float(event.get("start")),
                "window_end": float(event.get("end")),
            })
        except (TypeError, ValueError):
            continue
    events.sort(key=lambda item: item["start"])
    return events


# ==========================================
# create_trigger_record
# ==========================================
def create_trigger_record(label, stage, t0, end_time, tb_mode, th_mode):
    return {
        "label": label,
        "stage": stage,
        "t0": float(t0),
        "end_time": float(end_time),
        # --- 第一次發生時間 ---
        "tb": None,
        "th": None,
        "pointing_t": None,
        # --- 累計次數（含第一次）---
        "tb_count": 0,
        "th_count": 0,
        "pointing_count": 0,
        "tb_mode": tb_mode,
        "th_mode": th_mode,
        # --- 上一幀狀態（上升邊緣偵測）---
        "_prev_tb": False,
        "_prev_th": False,
        "_prev_pointing": False,
        # --- 上次計次時間（防 YOLO 掉偵測重複計數）---
        "_last_tb_time": -999.0,
        "_last_th_time": -999.0,
        "_last_pointing_time": -999.0,
        "closed": False
    }


STAGE_NAMES = {
    1: "真人指近物(Pointing - Near)",
    2: "真人指近物(Pointing - Near)",
    3: "真人指遠物(Pointing - Far)",
    4: "真人指遠物(Pointing - Far)",
    5: "神奇氣球(Magical Balloon)",
    6: "看偶寫字 (Puppet Writing)",
    7: "開筱驚喜袋(Mystery Bag)",
    8: "手機怪聲(Strange Sound)",
    9: "機器人畫畫(Robot Drawing)",
    10: "機器人煙火秀(Social Referencing)",
    11: "機指近物(Pointing - Near)",
    12: "機指近物(Pointing - Near)",
    13: "機指遠物(Pointing - Far)",
    14: "機指遠物(Pointing - Far)",
}


class ScoringEngine:
    def __init__(self, cache_path, scoring_version, video_path):
        self.scoring_version = scoring_version
        self.video_path = video_path
        self.stage_names = STAGE_NAMES
        self.event_logs = ["[SYSTEM] Scoring Version: " + str(scoring_version)]
        self.stage_transition_logs = []
        self.score_event_logs = []
        self.trigger_event_records = []
        self.active_trigger_records = []
        self.you_look_trigger_windows = load_keyword_trigger_windows_from_cache(cache_path)
        self.speech_events = load_speech_events_from_cache(cache_path)
        self.noise_events = load_noise_events_from_cache(cache_path)
        self.stage_start_times = {}
        self._build_absolute_timeline()
        print(">>> [評分系統] 讀取到 " + str(len(self.you_look_trigger_windows)) + " 個計分觸發窗")
        print(">>> [評分系統] 讀取到 " + str(len(self.speech_events)) + " 個語音事件、" + str(len(self.noise_events)) + " 個雜音事件")
        for stg, t in sorted(self.stage_start_times.items()):
            print("    - Stage " + str(stg) + ": " + str(round(t, 2)) + "s")
        self.prev_keyword_state = False
        self.prev_child_hit_state = False
        self.prev_gaze_state = False
        self.prev_tester_gaze_state = False
        self.gazing_cooldown_sec = 0.2
        self.total_score = 0
        self.scored_stages = set()
        self.total_gazing_events = 0
        self.stage_gazing_counts = {}
        self.last_gazing_event_time_by_stage = {}
        self.blocked_gazing_logged_stages = set()
        self.current_stage_enter_time = 0.0
        self.active_you_look_scoring_stage = None
        self.active_you_look_scoring_window = None
        self.processed_speech_event_ids = set()
        self.processed_noise_event_ids = set()
        self.created_object_t0_stages = set()

    def _build_absolute_timeline(self):
        # Stage 8: 使用雜音事件 + 語音中的怪聲/聲音關鍵字
        # 🌟 修正：移除「機器人」避免將 Stage 9 之前的機器人登場誤判為 Stage 8 起點
        t8_cands = [e["start"] for e in self.noise_events]
        for e in self.speech_events:
            if any(k in e["text"] for k in ["怪聲", "聲音", "声音"]):
                t8_cands.append(e["start"])
        t8_cands.append(10**9)
        t8 = min(t8_cands)
        if t8 < 10**9:
            self.stage_start_times[8] = t8

        # Stage 9: 畫畫關鍵字（需在 Stage 8 之後）
        t9_cands = []
        for e in self.speech_events:
            if any(k in e["text"] for k in ["畫", "画"]):
                t9_cands.append(e["start"])
        t9_cands = [t for t in t9_cands if t >= (t8 if t8 < 10**9 else 0)]
        t9_cands.append(10**9)
        t9 = min(t9_cands)
        if t9 < 10**9:
            self.stage_start_times[9] = t9

        # Stage 10: 煙火/321 倒數（需在 Stage 9 之後）
        t10_base = t9 if t9 < 10**9 else (t8 if t8 < 10**9 else 0)
        t10_cands = []
        for e in self.speech_events:
            if any(k in e["text"] for k in ["煙火", "烟火", "321", "三二一", "三", "3"]):
                t10_cands.append(e["start"])
        t10_cands = [t for t in t10_cands if t >= t10_base]
        t10_cands.append(10**9)
        t10 = min(t10_cands)
        if t10 < 10**9:
            self.stage_start_times[10] = t10

        # Stage 11~14: 機器人說「小朋友 你看」才觸發（需在 Stage 10 之後）
        if t10 < 10**9:
            you_look_cands = []
            for e in self.speech_events:
                if "小朋友" in e["text"] and "你看" in e["text"] and e["start"] > t10:
                    you_look_cands.append(e["start"])

            # 將所有可能的時間點排序
            you_look_cands.sort()

            # 🌟 新增：過濾時間相近的重複語音事件（冷卻機制）
            filtered_pointing_times = []
            for t in you_look_cands:
                if not filtered_pointing_times:
                    filtered_pointing_times.append(t)
                else:
                    # 確保兩次「小朋友你看」至少間隔 2 秒
                    if t - filtered_pointing_times[-1] > 2.0:
                        filtered_pointing_times.append(t)

            # 取前 4 次有效的時間點
            pointing_times = filtered_pointing_times[:4]

            for i, t in enumerate(pointing_times):
                self.stage_start_times[11 + i] = t

            if len(pointing_times) >= 2:
                durations = [pointing_times[i+1] - pointing_times[i] for i in range(len(pointing_times)-1)]
                self.avg_pointing_duration = sum(durations) / len(durations)
            else:
                self.avg_pointing_duration = 3.0
        else:
            self.avg_pointing_duration = 3.0

    def handle_stage_change(self, previous_stage, detected_stage, time_sec):
        self.event_logs.append("[" + str(round(time_sec,1)) + "s] Stage change -> " + str(detected_stage))
        self.stage_transition_logs.append("[" + str(round(time_sec,1)) + "s] Stage " + str(detected_stage))
        if self.active_you_look_scoring_stage is not None:
            self.active_you_look_scoring_stage = None
            self.active_you_look_scoring_window = None
        self.current_stage_enter_time = time_sec
        self.prev_gaze_state = False
        self.prev_child_hit_state = False
        self.blocked_gazing_logged_stages.discard(detected_stage)

    def handle_stage_override(self, previous_stage, new_stage, time_sec):
        self.handle_stage_change(previous_stage, new_stage, time_sec)

    def update_frame(self, time_sec, current_stage, is_in_trigger_window, child_is_pointing_hit,
                     child_is_gazing_at, child_is_gazing_at_tester, gaze_result, robot_rays,
                     robot_boxes, yolo_boxes, is_gazing_at_box_func, tester_gaze_angles=None):
        self._update_trigger_records(time_sec, current_stage, child_is_pointing_hit,
                                     child_is_gazing_at, child_is_gazing_at_tester,
                                     gaze_result, robot_rays, robot_boxes, yolo_boxes, is_gazing_at_box_func)
        self._update_gazing_score(time_sec, current_stage, child_is_gazing_at)
        self._update_clinical_logs(time_sec, current_stage, is_in_trigger_window,
                                   child_is_pointing_hit, child_is_gazing_at,
                                   child_is_gazing_at_tester, tester_gaze_angles)
        if is_in_trigger_window and not self.prev_keyword_state:
            self.prev_child_hit_state = False
            self.prev_gaze_state = False
            self.prev_tester_gaze_state = False
        self.prev_keyword_state = is_in_trigger_window
        self.prev_child_hit_state = child_is_pointing_hit
        self.prev_gaze_state = child_is_gazing_at
        self.prev_tester_gaze_state = child_is_gazing_at_tester

    def _update_trigger_records(self, time_sec, current_stage, child_is_pointing_hit,
                                child_is_gazing_at, child_is_gazing_at_tester,
                                gaze_result, robot_rays, robot_boxes, yolo_boxes, is_gazing_at_box_func):
        # --------------------------------------------------
        # 1. T0 建立
        # --------------------------------------------------
        # 🌟 統一檢查：當前階段是否已經有一筆尚未關閉的計分紀錄？（防止每一幀重複建立）
        already_active = any(
            r["stage"] == current_stage and not r.get("closed")
            for r in self.active_trigger_records
        )

        for event in self.speech_events:
            if event["id"] in self.processed_speech_event_ids or event["start"] > time_sec:
                continue
            new_record = None

            if current_stage in [1, 2, 3, 4] and "你看" in event["text"]:
                if not already_active:
                    new_record = create_trigger_record(
                        self.stage_names.get(current_stage, "Stage"+str(current_stage)),
                        current_stage, event["start"], 10**9, "object", "tester"
                    )

            elif current_stage == 8 and abs(event["start"] - self.stage_start_times.get(8, -1)) < 0.05:
                if not already_active:
                    # 🌟 修改為固定加 10 秒（捨棄原本的 event["window_end"]）
                    new_record = create_trigger_record(
                        self.stage_names.get(8, "Stage8"), 8,
                        event["start"], event["start"] + 10.0, None, "tester"
                    )

            elif current_stage == 9 and abs(event["start"] - self.stage_start_times.get(9, -1)) < 0.05:
                if not already_active:
                    s9_end = event["start"] + 15.0
                    for e in self.speech_events:
                        if e["start"] > event["start"] and any(k in e["text"] for k in ["画好了", "畫好了", "你看"]):
                            s9_end = e["start"] + 3.0
                            break
                    new_record = create_trigger_record(
                        self.stage_names.get(9, "Stage9"), 9,
                        event["start"], s9_end, "object", "robot_box"
                    )

            elif current_stage == 10 and abs(event["start"] - self.stage_start_times.get(10, -1)) < 0.05:
                if not already_active:
                    new_record = create_trigger_record(
                        self.stage_names.get(10, "Stage10"), 10,
                        event["start"], event["start"] + 8.0, None, "robot_box"
                    )

            elif 11 <= current_stage <= 14 and abs(event["start"] - self.stage_start_times.get(current_stage, -1)) < 0.05:
                if not already_active:
                    end_time = self.stage_start_times.get(
                        current_stage + 1,
                        event["start"] + getattr(self, "avg_pointing_duration", 3.0)
                    )
                    new_record = create_trigger_record(
                        self.stage_names.get(current_stage, "Stage"+str(current_stage)),
                        current_stage, event["start"], end_time, "object", "robot_box"
                    )

            if new_record:
                self.trigger_event_records.append(new_record)
                self.active_trigger_records.append(new_record)
                self.event_logs.append("[" + str(round(new_record['t0'],1)) + "s] T0: " + new_record['label'])
                # 🌟 建立後馬上將狀態設為 True，防止在同一個 frame 迴圈內被其他 event 重複觸發
                already_active = True

            if time_sec > event["end"]:
                self.processed_speech_event_ids.add(event["id"])

        # 雜音 (Stage 8)
        for noise_event in self.noise_events:
            if noise_event["id"] in self.processed_noise_event_ids or noise_event["start"] > time_sec:
                continue

            if current_stage == 8 and abs(noise_event["start"] - self.stage_start_times.get(8, -1)) < 0.05:
                if not already_active:
                    # 🌟 修改為固定加 10 秒（捨棄原本的 noise_event["window_end"]）
                    new_record = create_trigger_record(
                        self.stage_names.get(8, "Stage8"), 8,
                        noise_event["start"], noise_event["start"] + 10.0, None, "tester"
                    )
                    self.trigger_event_records.append(new_record)
                    self.active_trigger_records.append(new_record)
                    self.event_logs.append("[" + str(round(new_record['t0'],1)) + "s] T0(noise): " + new_record['label'])
                    already_active = True

            if time_sec > noise_event["end"]:
                self.processed_noise_event_ids.add(noise_event["id"])

        # Stage 5~7: YOLO 偵測到目標物時建立 T0
        if current_stage in [5, 6, 7] and current_stage not in self.created_object_t0_stages and len(yolo_boxes) > 0:
            lbl = self.stage_names.get(current_stage, "Stage"+str(current_stage))
            obj_record = create_trigger_record(lbl, current_stage, time_sec, 10**9, "object", "tester")
            self.created_object_t0_stages.add(current_stage)
            self.trigger_event_records.append(obj_record)
            self.active_trigger_records.append(obj_record)
            self.event_logs.append("[" + str(round(time_sec,1)) + "s] T0(obj): " + lbl)
            already_active = True

        # --------------------------------------------------
        # 2. Pointing / TB / TH 計次
        # --------------------------------------------------
        for record in self.active_trigger_records:
            if record.get("closed") or time_sec < record["t0"]:
                continue
            if time_sec > record["end_time"]:
                record["closed"] = True
                continue
            if isinstance(record["stage"], int) and current_stage != record["stage"]:
                record["closed"] = True
                continue

            # --- Pointing ---
            # 🌟 上升邊緣 + 冷卻：防止 YOLO 掉偵測造成同一次指向被重複計算
            if child_is_pointing_hit and not record["_prev_pointing"]:
                if time_sec - record["_last_pointing_time"] > POINTING_COOLDOWN_SEC:
                    if record["pointing_t"] is None:
                        record["pointing_t"] = time_sec
                        record["pointing_count"] = 1
                        self.event_logs.append("[" + str(round(time_sec,1)) + "s] Pointing#1: " + record["label"] + " +" + str(round(time_sec-record["t0"],2)) + "s")
                    else:
                        record["pointing_count"] += 1
                        self.event_logs.append("[" + str(round(time_sec,1)) + "s] Pointing#" + str(record["pointing_count"]) + ": " + record["label"])
                    record["_last_pointing_time"] = time_sec
            record["_prev_pointing"] = child_is_pointing_hit

            # --- TB ---
            # 🌟 上升邊緣 + 冷卻：防止 YOLO 掉偵測造成同一次注視被重複計算
            tb_hit = child_is_gazing_at if record["tb_mode"] == "object" else False
            if record["tb_mode"] is not None and tb_hit and not record["_prev_tb"]:
                if time_sec - record["_last_tb_time"] > TB_COOLDOWN_SEC:
                    if record["tb"] is None:
                        record["tb"] = time_sec
                        record["tb_count"] = 1
                        self.event_logs.append("[" + str(round(time_sec,1)) + "s] TB#1: " + record["label"] + " RT=" + str(round(time_sec-record["t0"],2)) + "s")
                    else:
                        record["tb_count"] += 1
                        self.event_logs.append("[" + str(round(time_sec,1)) + "s] TB#" + str(record["tb_count"]) + ": " + record["label"])
                    record["_last_tb_time"] = time_sec
            record["_prev_tb"] = tb_hit

            # --- TH ---
            # 需先達成 TB（若此 Stage 有 TB 條件）
            # 🌟 上升邊緣 + 冷卻：防止視線偵測閃爍造成重複計數
            th_allowed = record["tb_mode"] is None or record["tb"] is not None
            th_hit = False
            if th_allowed:
                if record["th_mode"] == "tester":
                    th_hit = child_is_gazing_at_tester
                elif record["th_mode"] == "robot_box":
                    th_hit = any(is_gazing_at_box_func(gaze_result, box) for box in robot_boxes)
            if th_allowed and th_hit and not record["_prev_th"]:
                if time_sec - record["_last_th_time"] > TH_COOLDOWN_SEC:
                    if record["th"] is None:
                        record["th"] = time_sec
                        record["th_count"] = 1
                        self.event_logs.append("[" + str(round(time_sec,1)) + "s] TH#1: " + record["label"] + " +" + str(round(time_sec-record["t0"],2)) + "s")
                    else:
                        record["th_count"] += 1
                        self.event_logs.append("[" + str(round(time_sec,1)) + "s] TH#" + str(record["th_count"]) + ": " + record["label"])
                    record["_last_th_time"] = time_sec
            record["_prev_th"] = th_hit

    def _update_gazing_score(self, time_sec, current_stage, child_is_gazing_at):
        matching_you_look_window = None
        if current_stage in [1, 2, 3, 4, 9]:
            for window_start, window_end in self.you_look_trigger_windows:
                if window_start + 1e-6 < self.current_stage_enter_time:
                    continue
                active_end = 10**9
                if current_stage == 9:
                    for e in self.speech_events:
                        if e["start"] > window_start and any(k in e["text"] for k in ["画好了", "畫好了", "你看"]):
                            active_end = e["start"] + 3.0
                            break
                if window_start <= time_sec <= active_end:
                    matching_you_look_window = (window_start, active_end)
                    break
        if matching_you_look_window is not None:
            if (self.active_you_look_scoring_stage != current_stage or
                    self.active_you_look_scoring_window != matching_you_look_window):
                self.active_you_look_scoring_stage = current_stage
                self.active_you_look_scoring_window = matching_you_look_window
        is_in_you_look_scoring_window = (self.active_you_look_scoring_stage == current_stage and
                                         self.active_you_look_scoring_window is not None)
        is_stage_scoring_allowed = (current_stage > 0 and
                                     (current_stage not in [1, 2, 3, 4, 9] or is_in_you_look_scoring_window))
        is_new_gazing_event = is_stage_scoring_allowed and child_is_gazing_at and not self.prev_gaze_state
        if not is_new_gazing_event:
            return
        last_event_time = self.last_gazing_event_time_by_stage.get(current_stage, -10**9)
        if time_sec - last_event_time <= self.gazing_cooldown_sec:
            return
        self.last_gazing_event_time_by_stage[current_stage] = time_sec
        self.total_gazing_events += 1
        self.stage_gazing_counts[current_stage] = self.stage_gazing_counts.get(current_stage, 0) + 1
        score_added = False
        if current_stage not in self.scored_stages:
            self.scored_stages.add(current_stage)
            self.total_score += 1
            score_added = True
            self.score_event_logs.append("[" + str(round(time_sec,1)) + "s] Stage " + str(current_stage) + " scored")
        added_str = "YES" if score_added else "NO"
        self.event_logs.append("[" + str(round(time_sec,1)) + "s] Gazing Stage " + str(current_stage) + " Score=" + str(self.total_score) + " Added=" + added_str)

    def _update_clinical_logs(self, time_sec, current_stage, is_in_trigger_window,
                              child_is_pointing_hit, child_is_gazing_at,
                              child_is_gazing_at_tester, tester_gaze_angles):
        if not is_in_trigger_window:
            return
        if child_is_pointing_hit and not self.prev_child_hit_state:
            self.event_logs.append("[" + str(round(time_sec,1)) + "s] Pointing hit Stage " + str(current_stage))
        if child_is_gazing_at and not self.prev_gaze_state:
            self.event_logs.append("[" + str(round(time_sec,1)) + "s] Gaze hit Stage " + str(current_stage))
        if child_is_gazing_at_tester and not self.prev_tester_gaze_state:
            if tester_gaze_angles is not None:
                self.event_logs.append("[" + str(round(time_sec,2)) + "s] Gaze@Tester P=" + str(round(tester_gaze_angles[0],1)) + " Y=" + str(round(tester_gaze_angles[1],1)))

    def write_report(self, event_log_path):
        with open(event_log_path, "w", encoding="utf-8") as f:
            f.write("=== Result Summary ===\n")
            f.write("Video: " + str(self.video_path) + "\n")
            f.write("-" * 40 + "\n")
            f.write("Total Score: " + str(self.total_score) + "\n")
            f.write("Total Gazing Events: " + str(self.total_gazing_events) + "\n")
            f.write("=" * 40 + "\n")
            f.write("=== T0 / Pointing / TB / TH Detail ===\n")
            f.write("  First-occurrence = timestamp; Count = total (incl. first)\n")
            f.write("  Cooldowns: TB/TH=" + str(TB_COOLDOWN_SEC) + "s, Pointing=" + str(POINTING_COOLDOWN_SEC) + "s\n")
            f.write("=" * 40 + "\n")
            ordered_records = sorted(self.trigger_event_records, key=lambda r: r.get("t0", 0.0))
            for idx, r in enumerate(ordered_records, 1):
                t0 = r["t0"]
                f.write("\n" + str(idx).zfill(2) + ". Stage " + str(r["stage"]) + " -- " + r["label"] + "\n")
                f.write("    T0       = " + str(round(t0, 2)) + "s\n")
                if r.get("pointing_t") is not None:
                    delay = r["pointing_t"] - t0
                    f.write("    Pointing = " + str(round(r["pointing_t"],2)) + "s  (+" + str(round(delay,2)) + "s from T0) x" + str(r["pointing_count"]) + "\n")
                else:
                    f.write("    Pointing = not detected\n")
                if r.get("tb_mode") is None:
                    f.write("    TB       = -- (no TB condition)\n")
                elif r.get("tb") is not None:
                    delay = r["tb"] - t0
                    f.write("    TB       = " + str(round(r["tb"],2)) + "s  (+" + str(round(delay,2)) + "s from T0) x" + str(r["tb_count"]) + "\n")
                else:
                    f.write("    TB       = not achieved\n")
                if r.get("th") is not None:
                    delay = r["th"] - t0
                    extra = ""
                    if r.get("tb") is not None:
                        extra = " / +" + str(round(r["th"] - r["tb"], 2)) + "s from TB"
                    f.write("    TH       = " + str(round(r["th"],2)) + "s  (+" + str(round(delay,2)) + "s from T0" + extra + ") x" + str(r["th_count"]) + "\n")
                else:
                    f.write("    TH       = not achieved\n")
                tb_ok = r.get("tb_mode") is None or r.get("tb") is not None
                th_ok = r.get("th") is not None
                pt_ok = r.get("pointing_t") is not None
                pt_s = "Pointing OK" if pt_ok else "Pointing --"
                tb_s = "TB OK" if tb_ok else "TB --"
                th_s = "TH OK" if th_ok else "TH --"
                f.write("    Sequence = T0 -> " + pt_s + " -> " + tb_s + " -> " + th_s + "\n")
            f.write("\n" + "=" * 40 + "\n")
            f.write("=== Stage Gazing Stats ===\n")
            reported_stages = sorted(set(self.stage_gazing_counts.keys()) | set(self.scored_stages))
            for stage_id in reported_stages:
                score_status = "scored" if stage_id in self.scored_stages else "not scored"
                stage_label = self.stage_names.get(stage_id, "Stage"+str(stage_id))
                count = self.stage_gazing_counts.get(stage_id, 0)
                f.write("Stage " + str(stage_id) + " [" + stage_label + "]: " + score_status + ", GazingCount=" + str(count) + "\n")
            f.write("-" * 40 + "\n")
            f.write("=== Full Event Log ===\n")
            for log in self.event_logs:
                f.write(log + "\n")