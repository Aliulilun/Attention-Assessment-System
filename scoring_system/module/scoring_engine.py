import json
import os

import numpy as np


def load_keyword_trigger_windows_from_cache(cache_path, target_keyword="你看"):
    if not os.path.exists(cache_path):
        return []

    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"⚠️ [評分系統] 無法讀取語音快取檔 {cache_path}: {e}")
        return []

    windows = []
    for record in data.get("segment_records", []):
        for event in record.get("trigger_events", []):
            if target_keyword not in event.get("keywords", []):
                continue
            trigger_window = event.get("trigger_window")
            if not trigger_window or len(trigger_window) != 2:
                continue
            try:
                windows.append((float(trigger_window[0]), float(trigger_window[1])))
            except (TypeError, ValueError):
                continue

    windows.sort(key=lambda item: item[0])
    return windows


def load_speech_events_from_cache(cache_path):
    if not os.path.exists(cache_path):
        return []

    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"⚠️ [評分系統] 無法讀取語音事件快取檔 {cache_path}: {e}")
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
            try:
                window_start = float(trigger_window[0])
                window_end = float(trigger_window[1])
            except (TypeError, ValueError, IndexError):
                window_start = start_time
                window_end = start_time + 3.0

            events.append({
                "id": event.get("id", ""),
                "event_type": event.get("event_type", "speech"),
                "keywords": event.get("keywords", []),
                "text": text,
                "start": start_time,
                "end": float(event.get("end", start_time) or start_time),
                "window_start": window_start,
                "window_end": window_end,
            })

    events.sort(key=lambda item: item["start"])
    return events


def load_noise_events_from_cache(cache_path):
    if not os.path.exists(cache_path):
        return []

    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"⚠️ [評分系統] 無法讀取雜音事件快取檔 {cache_path}: {e}")
        return []

    events = []
    for event in data.get("noise_events", []):
        try:
            start_time = float(event.get("start"))
            end_time = float(event.get("end"))
        except (TypeError, ValueError):
            continue

        events.append({
            "id": event.get("id", ""),
            "event_type": "noise",
            "keywords": ["怪聲"],
            "text": event.get("label", "noise"),
            "start": start_time,
            "end": end_time,
            "window_start": start_time,
            "window_end": end_time,
        })

    events.sort(key=lambda item: item["start"])
    return events


def get_gaze_origin_direction(gaze_result):
    if not gaze_result or not gaze_result.get("success"):
        return None, None

    left_eye = gaze_result.get("left_eye")
    right_eye = gaze_result.get("right_eye")
    gaze_vector = gaze_result.get("gaze_vector")
    if left_eye is None or right_eye is None or gaze_vector is None:
        return None, None

    origin = np.array([
        (left_eye[0] + right_eye[0]) / 2.0,
        (left_eye[1] + right_eye[1]) / 2.0,
    ], dtype=float)
    direction = np.array([gaze_vector[0], gaze_vector[1]], dtype=float)
    norm = np.linalg.norm(direction)
    if norm <= 1e-6:
        return None, None
    return origin, direction / norm


def gaze_intersects_robot_ray(gaze_result, robot_ray, max_distance_px=80.0):
    gaze_origin, gaze_dir = get_gaze_origin_direction(gaze_result)
    if gaze_origin is None:
        return False

    robot_origin = np.array(robot_ray["origin"], dtype=float)
    robot_dir = np.array(robot_ray["direction"], dtype=float)
    robot_norm = np.linalg.norm(robot_dir)
    if robot_norm <= 1e-6:
        return False
    robot_dir = robot_dir / robot_norm

    a = np.dot(gaze_dir, gaze_dir)
    b = np.dot(gaze_dir, robot_dir)
    c = np.dot(robot_dir, robot_dir)
    w0 = gaze_origin - robot_origin
    d = np.dot(gaze_dir, w0)
    e = np.dot(robot_dir, w0)
    denom = a * c - b * b
    if abs(denom) <= 1e-6:
        return False

    t = (b * e - c * d) / denom
    u = (a * e - b * d) / denom
    if t < 0 or u < 0:
        return False

    closest_gaze = gaze_origin + t * gaze_dir
    closest_robot = robot_origin + u * robot_dir
    return np.linalg.norm(closest_gaze - closest_robot) <= max_distance_px


def check_gaze_on_robot_rays(gaze_result, robot_rays):
    return any(gaze_intersects_robot_ray(gaze_result, ray) for ray in robot_rays)


def event_has_keyword(event, keyword):
    return keyword in event.get("keywords", [])


def select_first_event_id(speech_events, keyword):
    for event in speech_events:
        if event_has_keyword(event, keyword):
            return event["id"]
    return None


def build_stage8_pointing_event_map(speech_events, firework_start_time):
    you_look_events = [
        event for event in speech_events
        if event_has_keyword(event, "你看")
        and firework_start_time is not None
        and event["start"] > firework_start_time
    ]
    if len(you_look_events) < 4:
        return {}

    pointing_events = you_look_events[-4:]
    t0s = [event["start"] for event in pointing_events]
    durations = [t0s[index + 1] - t0s[index] for index in range(3)]
    avg_duration = sum(durations) / len(durations) if durations else 3.0
    labels = ["機指近物1", "機指近物2", "機指遠物1", "機指遠物2"]

    return {
        event["id"]: {
            "label": labels[index],
            "end_time": t0s[index + 1] if index < 3 else event["start"] + avg_duration,
        }
        for index, event in enumerate(pointing_events)
    }


def create_trigger_record(label, stage, t0, end_time, tb_mode, th_mode):
    return {
        "label": label,
        "stage": stage,
        "t0": float(t0),
        "end_time": float(end_time),
        "tb": None,
        "th": None,
        "tb_mode": tb_mode,
        "th_mode": th_mode,
        "closed": False,
    }


def create_time_record(label, stage, event_time):
    return {
        "label": label,
        "stage": stage,
        "time": float(event_time),
        "record_type": "time",
    }


class ScoringEngine:
    def __init__(self, cache_path, scoring_version, video_path):
        self.scoring_version = scoring_version
        self.video_path = video_path
        self.event_logs = [f"[SYSTEM] Scoring Version: {scoring_version}"]
        self.stage_transition_logs = []
        self.score_event_logs = []
        self.trigger_event_records = []
        self.active_trigger_records = []

        self.you_look_trigger_windows = load_keyword_trigger_windows_from_cache(cache_path, "你看")
        self.speech_events = load_speech_events_from_cache(cache_path)
        self.noise_events = load_noise_events_from_cache(cache_path)
        self.has_speech_sound_events = any(event_has_keyword(event, "[聲音]") for event in self.speech_events)
        self.stage8_start_event_id = select_first_event_id(self.speech_events, "機器人")
        self.firework_event_id = select_first_event_id(self.speech_events, "321")
        self.firework_start_time = None
        for event in self.speech_events:
            if event["id"] == self.firework_event_id:
                self.firework_start_time = event["start"]
                break
        self.stage8_pointing_event_map = build_stage8_pointing_event_map(
            self.speech_events,
            self.firework_start_time,
        )

        print(f">>> [評分系統] 讀取到 {len(self.you_look_trigger_windows)} 個『你看』計分觸發窗")
        print(f">>> [評分系統] 讀取到 {len(self.speech_events)} 個語音事件、{len(self.noise_events)} 個雜音事件")
        print(f">>> [評分系統] Stage 8 起始事件 ID: {self.stage8_start_event_id or '未找到'}")
        print(f">>> [評分系統] 機器人煙火秀事件 ID: {self.firework_event_id or '未找到'}")
        print(f">>> [評分系統] 機指近/遠物事件數: {len(self.stage8_pointing_event_map)}")

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

    def handle_stage_change(self, previous_stage, detected_stage, time_sec):
        self.event_logs.append(f"[{time_sec:.1f}s] 階段改變：視覺模組確認進入 第 {detected_stage} 階段")
        self.stage_transition_logs.append(f"[{time_sec:.1f}s] 進入 Stage {detected_stage}")

        if self.active_you_look_scoring_stage is not None:
            self.event_logs.append(
                f"[{time_sec:.1f}s] 評分窗關閉：Stage {previous_stage} -> Stage {detected_stage}，"
                f"關閉 Stage {self.active_you_look_scoring_stage} 的『你看』計分窗"
            )
            self.active_you_look_scoring_stage = None
            self.active_you_look_scoring_window = None

        self.current_stage_enter_time = time_sec
        self.prev_gaze_state = False
        self.prev_child_hit_state = False
        self.blocked_gazing_logged_stages.discard(detected_stage)

    def handle_stage_override(self, previous_stage, new_stage, time_sec):
        if self.active_you_look_scoring_stage is not None:
            self.event_logs.append(
                f"[{time_sec:.1f}s] 評分窗關閉：Stage {previous_stage} -> Stage {new_stage}，"
                f"關閉 Stage {self.active_you_look_scoring_stage} 的『你看』計分窗"
            )
            self.active_you_look_scoring_stage = None
            self.active_you_look_scoring_window = None
        self.current_stage_enter_time = time_sec
        self.prev_gaze_state = False
        self.prev_child_hit_state = False

    def update_frame(
        self,
        time_sec,
        current_stage,
        is_in_trigger_window,
        child_is_pointing_hit,
        child_is_gazing_at,
        child_is_gazing_at_tester,
        gaze_result,
        robot_rays,
        robot_boxes,
        yolo_boxes,
        is_gazing_at_box_func,
        tester_gaze_angles=None,
    ):
        self._update_trigger_records(
            time_sec,
            current_stage,
            child_is_gazing_at,
            child_is_gazing_at_tester,
            gaze_result,
            robot_rays,
            robot_boxes,
            yolo_boxes,
            is_gazing_at_box_func,
        )
        self._update_keyword_edge(time_sec, is_in_trigger_window)
        self._update_gazing_score(time_sec, current_stage, child_is_gazing_at)
        self._update_clinical_logs(
            time_sec,
            current_stage,
            is_in_trigger_window,
            child_is_pointing_hit,
            child_is_gazing_at,
            child_is_gazing_at_tester,
            tester_gaze_angles,
        )

        self.prev_child_hit_state = child_is_pointing_hit
        self.prev_gaze_state = child_is_gazing_at
        self.prev_tester_gaze_state = child_is_gazing_at_tester

    def _update_trigger_records(
        self,
        time_sec,
        current_stage,
        child_is_gazing_at,
        child_is_gazing_at_tester,
        gaze_result,
        robot_rays,
        robot_boxes,
        yolo_boxes,
        is_gazing_at_box_func,
    ):
        for event in self.speech_events:
            if event["id"] in self.processed_speech_event_ids or event["start"] > time_sec:
                continue

            new_record = None
            if event["id"] == self.stage8_start_event_id:
                self.trigger_event_records.append(create_time_record("Stage 8 起始", 8, event["start"]))
                self.event_logs.append(f"[{event['start']:.1f}s] Stage 8 起始：偵測到『機器人』語音觸發")
                self.processed_speech_event_ids.add(event["id"])
                continue

            if event_has_keyword(event, "[聲音]") and current_stage in [7, 8]:
                new_record = create_trigger_record("怪聲", "7/8", event["start"], event["window_end"], None, "tester")
            elif event["id"] == self.firework_event_id:
                new_record = create_trigger_record("機器人煙火秀", 8, event["start"], event["window_end"], None, "robot_box")
            elif event["id"] in self.stage8_pointing_event_map:
                stage8_event = self.stage8_pointing_event_map[event["id"]]
                new_record = create_trigger_record(
                    stage8_event["label"],
                    8,
                    event["start"],
                    stage8_event["end_time"],
                    "robot_ray",
                    "robot_box",
                )
            elif event_has_keyword(event, "你看") and current_stage in [1, 2, 3, 4]:
                new_record = create_trigger_record(
                    f"Stage {current_stage}",
                    current_stage,
                    event["start"],
                    event["window_end"],
                    "object",
                    "tester",
                )

            if new_record is not None:
                self.trigger_event_records.append(new_record)
                self.active_trigger_records.append(new_record)
                self.processed_speech_event_ids.add(event["id"])
                self.event_logs.append(f"[{new_record['t0']:.1f}s] 觸發事件建立：{new_record['label']} T0")
            elif time_sec > event["window_end"]:
                self.processed_speech_event_ids.add(event["id"])

        if not self.has_speech_sound_events:
            for noise_event in self.noise_events:
                if noise_event["id"] in self.processed_noise_event_ids or noise_event["start"] > time_sec:
                    continue

                if current_stage in [7, 8]:
                    weird_sound_record = create_trigger_record(
                        "怪聲",
                        "7/8",
                        noise_event["start"],
                        noise_event["window_end"],
                        None,
                        "tester",
                    )
                    self.trigger_event_records.append(weird_sound_record)
                    self.active_trigger_records.append(weird_sound_record)
                    self.processed_noise_event_ids.add(noise_event["id"])
                    self.event_logs.append(f"[{noise_event['start']:.1f}s] 觸發事件建立：怪聲 T0")
                elif time_sec > noise_event["window_end"]:
                    self.processed_noise_event_ids.add(noise_event["id"])

        if current_stage in [5, 6, 7] and current_stage not in self.created_object_t0_stages and len(yolo_boxes) > 0:
            stage_object_labels = {
                5: "Stage 5 balloon",
                6: "Stage 6 doll",
                7: "Stage 7 toy",
            }
            object_record = create_trigger_record(
                stage_object_labels[current_stage],
                current_stage,
                time_sec,
                10**9,
                "object",
                "tester",
            )
            self.created_object_t0_stages.add(current_stage)
            self.trigger_event_records.append(object_record)
            self.active_trigger_records.append(object_record)
            self.event_logs.append(
                f"[{time_sec:.1f}s] 觸發事件建立：{object_record['label']} 偵測到目標物，記錄 T0"
            )

        for record in self.active_trigger_records:
            if record.get("closed") or time_sec < record["t0"]:
                continue
            if time_sec > record["end_time"]:
                record["closed"] = True
                continue
            if isinstance(record["stage"], int) and record["stage"] in [1, 2, 3, 4, 5, 6, 7] and current_stage != record["stage"]:
                record["closed"] = True
                continue

            tb_hit = False
            if record["tb_mode"] == "object":
                tb_hit = child_is_gazing_at
            elif record["tb_mode"] == "robot_ray":
                tb_hit = check_gaze_on_robot_rays(gaze_result, robot_rays)

            if record["tb_mode"] is not None and record["tb"] is None and tb_hit:
                record["tb"] = time_sec
                self.event_logs.append(f"[{time_sec:.1f}s] 觸發事件：{record['label']} 記錄 TB")

            th_allowed = record["tb_mode"] is None or record["tb"] is not None
            th_hit = False
            if th_allowed and record["th_mode"] == "tester":
                th_hit = child_is_gazing_at_tester
            elif th_allowed and record["th_mode"] == "robot_box":
                th_hit = any(is_gazing_at_box_func(gaze_result, box) for box in robot_boxes)

            if th_allowed and record["th"] is None and th_hit:
                record["th"] = time_sec
                self.event_logs.append(f"[{time_sec:.1f}s] 觸發事件：{record['label']} 記錄 TH")

    def _update_keyword_edge(self, time_sec, is_in_trigger_window):
        if is_in_trigger_window and not self.prev_keyword_state:
            self.event_logs.append(f"[{time_sec:.1f}s] 觸發：偵測到引導語音關鍵字，開啟 3 秒高靈敏注意力判定窗")
            self.prev_child_hit_state = False
            self.prev_gaze_state = False
            self.prev_tester_gaze_state = False
        self.prev_keyword_state = is_in_trigger_window

    def _update_gazing_score(self, time_sec, current_stage, child_is_gazing_at):
        matching_you_look_window = None
        if current_stage in [1, 2, 3, 4]:
            for window_start, window_end in self.you_look_trigger_windows:
                if window_start + 1e-6 < self.current_stage_enter_time:
                    continue
                if window_start <= time_sec <= window_end:
                    matching_you_look_window = (window_start, window_end)
                    break

        if matching_you_look_window is not None:
            if (
                self.active_you_look_scoring_stage != current_stage
                or self.active_you_look_scoring_window != matching_you_look_window
            ):
                self.active_you_look_scoring_stage = current_stage
                self.active_you_look_scoring_window = matching_you_look_window
                self.prev_gaze_state = False
                self.event_logs.append(
                    f"[{time_sec:.1f}s] 評分窗開啟：偵測到『你看』語音觸發，"
                    f"Stage {current_stage} 開始允許 Gazing 計分"
                )

        if self.active_you_look_scoring_window is not None:
            _, active_end = self.active_you_look_scoring_window
            if time_sec > active_end or self.active_you_look_scoring_stage != current_stage:
                self.event_logs.append(
                    f"[{time_sec:.1f}s] 評分窗關閉：Stage {self.active_you_look_scoring_stage} 的『你看』計分窗結束"
                )
                self.active_you_look_scoring_stage = None
                self.active_you_look_scoring_window = None

        is_in_you_look_scoring_window = (
            self.active_you_look_scoring_stage == current_stage
            and self.active_you_look_scoring_window is not None
            and self.active_you_look_scoring_window[0] <= time_sec <= self.active_you_look_scoring_window[1]
        )
        is_stage_scoring_allowed = (
            current_stage > 0
            and (current_stage not in [1, 2, 3, 4] or is_in_you_look_scoring_window)
        )

        if (
            current_stage in [1, 2, 3, 4]
            and child_is_gazing_at
            and not is_in_you_look_scoring_window
            and current_stage not in self.blocked_gazing_logged_stages
        ):
            self.blocked_gazing_logged_stages.add(current_stage)
            self.event_logs.append(f"[{time_sec:.1f}s] Gazing 未計分：Stage {current_stage} 尚未進入『你看』語音計分窗")

        is_new_gazing_event = is_stage_scoring_allowed and child_is_gazing_at and not self.prev_gaze_state
        if not is_new_gazing_event:
            return

        last_event_time = self.last_gazing_event_time_by_stage.get(current_stage, -10**9)
        seconds_since_last_event = time_sec - last_event_time
        if seconds_since_last_event <= self.gazing_cooldown_sec:
            return

        self.last_gazing_event_time_by_stage[current_stage] = time_sec
        self.total_gazing_events += 1
        self.stage_gazing_counts[current_stage] = self.stage_gazing_counts.get(current_stage, 0) + 1

        score_added = False
        if current_stage not in self.scored_stages:
            self.scored_stages.add(current_stage)
            self.total_score += 1
            score_added = True
            self.score_event_logs.append(f"[{time_sec:.1f}s] Stage {current_stage} 加分")

        self.event_logs.append(
            f"[{time_sec:.1f}s] Gazing 計數：Stage {current_stage} 視線命中目標物品"
            f"｜Score={self.total_score}"
            f"｜Stage Gazing Count={self.stage_gazing_counts[current_stage]}"
            f"｜Total Gazing Events={self.total_gazing_events}"
            f"｜Seconds Since Last={seconds_since_last_event:.3f}"
            f"｜Score Added={'YES' if score_added else 'NO'}"
        )

    def _update_clinical_logs(
        self,
        time_sec,
        current_stage,
        is_in_trigger_window,
        child_is_pointing_hit,
        child_is_gazing_at,
        child_is_gazing_at_tester,
        tester_gaze_angles,
    ):
        if not is_in_trigger_window:
            return

        if child_is_pointing_hit and not self.prev_child_hit_state:
            self.event_logs.append(f"[{time_sec:.1f}s] 互動成功：小朋友手勢精確指向 Stage {current_stage} 的目標物品")

        if child_is_gazing_at and not self.prev_gaze_state:
            self.event_logs.append(f"[{time_sec:.1f}s] 注視成功：小朋友視線 (Ray-Casting) 命中 Stage {current_stage} 的目標物品")

        if child_is_gazing_at_tester and not self.prev_tester_gaze_state:
            if tester_gaze_angles is not None:
                pitch_deg, yaw_deg = tester_gaze_angles
                self.event_logs.append(f"[{time_sec:.2f}s] 眼神交會：Gazing At Tester (P={pitch_deg:.1f}°, Y={yaw_deg:.1f}°)")
            else:
                self.event_logs.append(f"[{time_sec:.2f}s] 眼神交會：Gazing At Tester")

    def write_report(self, event_log_path):
        with open(event_log_path, "w", encoding="utf-8") as f:
            f.write("=== 互動行為分析結果摘要 ===\n")
            f.write(f"Scoring Version: {self.scoring_version}\n")
            f.write(f"影片來源: {self.video_path}\n")
            f.write("--------------------------------\n")
            f.write(f"總分 Score: {self.total_score}\n")
            f.write(f"有效 Gazing 次數: {self.total_gazing_events}\n")
            f.write("計分規則: Stage 1~4 需在該 Stage 的『你看』觸發窗內；切換 Stage 會關閉舊觸發窗；同一 Stage 只加一次分；Gazing 間隔需大於 0.2 秒。\n")

            f.write("--------------------------------\n")
            f.write("=== 各 Stage 統計 ===\n")
            reported_stages = sorted(set(self.stage_gazing_counts.keys()) | set(self.scored_stages))
            if reported_stages:
                for stage_id in reported_stages:
                    score_status = "已加分" if stage_id in self.scored_stages else "未加分"
                    f.write(f"Stage {stage_id}: {score_status}，Gazing Count = {self.stage_gazing_counts.get(stage_id, 0)}\n")
            else:
                f.write("無有效 Stage 計分資料\n")

            f.write("--------------------------------\n")
            f.write("=== 加分時間點 ===\n")
            if self.score_event_logs:
                for log in self.score_event_logs:
                    f.write(log + "\n")
            else:
                f.write("無加分事件\n")

            f.write("--------------------------------\n")
            f.write("=== Stage 流程 ===\n")
            if self.stage_transition_logs:
                for log in self.stage_transition_logs:
                    f.write(log + "\n")
            else:
                f.write("無 Stage 變化紀錄\n")

            f.write("--------------------------------\n")
            f.write("=== 觸發事件 ===\n")
            self._write_trigger_records(f)

            f.write("--------------------------------\n")
            f.write("=== 完整事件紀錄表 ===\n")
            if self.event_logs:
                for log in self.event_logs:
                    f.write(log + "\n")
            else:
                f.write("無事件紀錄\n")

    def _write_trigger_records(self, f):
        if not self.trigger_event_records:
            f.write("無觸發事件\n")
            return

        ordered_records = sorted(
            self.trigger_event_records,
            key=lambda record: (
                0 if record.get("label") == "怪聲" else 1,
                record.get("t0", record.get("time", 0.0)),
            ),
        )
        normal_index = 1
        for record in ordered_records:
            if record.get("label") == "怪聲":
                prefix = "00"
            else:
                prefix = f"{normal_index:02d}"
                normal_index += 1

            if record.get("record_type") == "time":
                f.write(f"{prefix}. {record['label']} | Stage {record['stage']} | time={record['time']:.2f}s\n")
                continue

            parts = [
                f"{prefix}. {record['label']}",
                f"Stage {record['stage']}",
                f"T0={record['t0']:.2f}s",
            ]
            if record.get("tb") is not None:
                parts.append(f"TB={record['tb']:.2f}s")
            if record.get("th") is not None:
                parts.append(f"TH={record['th']:.2f}s")
            f.write(" | ".join(parts) + "\n")
