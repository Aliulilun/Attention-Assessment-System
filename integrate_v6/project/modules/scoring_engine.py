import json
import os
import numpy as np

from modules.stage_scoring import format_stage_score_suffix

# ==========================================
# 冷卻時間常數（防 YOLO 掉偵測導致重複計次）
# ==========================================
TB_COOLDOWN_SEC      = 1.5   # TB 兩次計次之間的最小間隔（秒）
TH_COOLDOWN_SEC      = 1.5   # TH 兩次計次之間的最小間隔（秒）
POINTING_COOLDOWN_SEC = 0.8  # Pointing 兩次計次之間的最小間隔（秒）

# ==========================================
# 🌟 新增：Whisper 幻覺過濾標記
# Whisper 遇到聽不清的片段時，會把 initial_prompt 逐字複誦回來
# （例：「請勿忽略短促的聲音：看這裡、看這裡…」），
# 這種段落的關鍵字全是假的：會產生假觸發事件與假空窗期，
# 干擾 T0 建立與 Trigger Lock。文字含以下標記的段落一律略過。
# ==========================================
HALLUCINATION_MARKERS = ["請勿忽略", "短促的聲音"]

# 🌟 新增：Stage 1-4 OCR 代償 T0 的等待秒數
# 進入階段後等待「你看/看這裡」關鍵字這麼多秒，等不到就以
# 階段進入時間代償建立 T0（Whisper 幻覺/漏轉錄的影片才需要）。
# Stage 1 前面通常有問名字/寒暄，實測「你看」中位數要等 ~9 秒、
# 最慢近 20 秒才出現，4 秒門檻幾乎每支影片都會提早誤觸發，
# 所以 Stage 1 另外給更寬鬆的門檻；Stage 2~4 銜接快，維持原值。
FALLBACK_T0_DELAY_SEC = 4.0
FALLBACK_T0_DELAY_SEC_STAGE1 = 15.0

# 🌟 新增：Stage 1 的暫存緩衝區延後這麼多秒才開始收集
# 原因：Stage 1 進入前幾秒通常是問名字、寒暄，跟任務無關，
# 不應該把這段時間的張望算成對提示的反應。
STAGE1_BUFFER_START_SEC = 5.5


def is_hallucinated_text(text):
    return any(m in (text or "") for m in HALLUCINATION_MARKERS)


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
        if is_hallucinated_text(record.get("text", "")):
            continue  # 🌟 略過 Whisper 幻覺段落
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
        if is_hallucinated_text(text):
            continue  # 🌟 略過 Whisper 幻覺段落（假關鍵字、假時間窗）
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
        # ============================================================
        # 🌟 新增：交替計次狀態
        # 記錄「上一次成功計次的目標」："object"（物品）或 "head"（人/機器人）。
        # 規則：
        #   - TB 只有在 _last_counted_target != "object" 時才能計
        #     （初始 None 可計第一次；之後必須先看過人才可再計）
        #   - TH 只有在 _last_counted_target == "object" 時才能計
        #     （必須先看過物品才可計；之後必須再看回物品才可再計）
        # 效果：視線停留在同一目標上不管多久、或視線移開又回到
        # 同一目標，都只算一次；只有「物品↔人」真正交替才會累加。
        # ============================================================
        "_last_counted_target": None,
        "closed": False
    }


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
        # 🌟 新增：Stage 1-4 在還沒有正式 T0 紀錄前的暫存緩衝區
        # 存 (time_sec, child_is_pointing_hit, child_is_gazing_at, child_is_gazing_at_tester)，
        # 一旦 T0 確定（關鍵字或 fallback），就把 >= T0 的部分重播進正式紀錄，
        # 避免「T0 確定前的張望」被漏記。
        self.stage_gaze_buffer = []

    def _build_absolute_timeline(self):
        # Stage 8: 使用雜音事件 + 語音中明確命中「怪聲」關鍵字的事件
        # 🌟 修正：移除「機器人」避免將 Stage 9 之前的機器人登場誤判為 Stage 8 起點
        # 🌟 修正：改用精確比對已配對到的關鍵字（event["keywords"]），不再對整句轉錄
        # 文字做「聲音」/「声音」子字串搜尋。子字串比對會連同「[聲音]」這種泛用
        # 佔位關鍵字、甚至 Whisper 幻覺片段（例如把 initial_prompt 逐字複誦回來）
        # 一起誤判為 Stage 8 起點，且因為取 min() 只會讓起點被誤判得更早。
        t8_cands = [e["start"] for e in self.noise_events]
        for e in self.speech_events:
            if "怪聲" in e["keywords"]:
                t8_cands.append(e["start"])
        t8_cands.append(10**9)
        t8 = min(t8_cands)
        if t8 < 10**9:
            self.stage_start_times[8] = t8
            
        # Stage 9: 畫畫關鍵字（需在 Stage 8 之後）
        # 🌟 修改：加入順序防護回退——若「必須在 Stage 8 之後」的過濾
        # 把候選全部濾光（例如怪聲偵測時間偏晚、比畫畫關鍵字還後面），
        # 退回使用未過濾的候選，避免 Stage 9/10 整個消失、不切換也不記錄。
        t9_all = []
        for e in self.speech_events:
            if any(k in e["text"] for k in ["畫", "画"]):
                t9_all.append(e["start"])
        t9_cands = [t for t in t9_all if t >= (t8 if t8 < 10**9 else 0)]
        if not t9_cands and t9_all:
            t9_cands = list(t9_all)  # 🌟 回退：順序過濾撲空時改用全部候選
        t9_cands.append(10**9)
        t9 = min(t9_cands)
        if t9 < 10**9:
            self.stage_start_times[9] = t9

        # Stage 10: 煙火/321 倒數（需在 Stage 9 之後）
        t10_base = t9 if t9 < 10**9 else (t8 if t8 < 10**9 else 0)
        t10_all = []
        for e in self.speech_events:
            if any(k in e["text"] for k in ["煙火", "烟火", "321", "三二一", "三", "3"]):
                t10_all.append(e["start"])
        t10_cands = [t for t in t10_all if t >= t10_base]
        if not t10_cands and t10_all:
            # 🌟 回退：放寬到「在 Stage 8 之後」即可（不強制在 Stage 9 之後）。
            # 不完全解除過濾——「三/3」是常見字，全域取 min 會誤抓影片開頭的語音。
            if t8 < 10**9:
                t10_cands = [t for t in t10_all if t >= t8]
        t10_cands.append(10**9)
        t10 = min(t10_cands)
        if t10 < 10**9:
            self.stage_start_times[10] = t10
            
        # Stage 11~14: 機器人說「小朋友你看」才觸發（需在 Stage 10 之後）
        # 🌟 修改：改用「時間鄰近配對」，不再使用 e["text"]（整段 Whisper 文字）。
        # 問題根因：e["text"] 是整個 Whisper 段落；Stage 10 結尾的「你看，很漂亮吧」
        # 若與前面的「小朋友你看」同屬一個長段落，e["text"] 同時含兩詞 → 誤觸發 Stage 11。
        # 修法：speech engine 對「小朋友」與「你看」各自產生獨立 event，
        # 故改為配對邏輯——找「你看」event 且其前 5 秒內有「小朋友」event 才成立。
        # 實測資料最大間距約 3.86 秒（如 288.980s→292.840s），5 秒窗口足夠覆蓋。
        if t10 < 10**9:
            # 收集 t10 之後所有「小朋友」事件的時間點
            xiaopengou_times = sorted(
                e["start"] for e in self.speech_events
                if "小朋友" in e.get("keywords", []) and e["start"] > t10
            )

            you_look_cands = []
            for e in self.speech_events:
                if e["start"] <= t10:
                    continue
                if "你看" not in e.get("keywords", []):
                    continue
                # 「你看」之前 5 秒內必須有「小朋友」事件，才視為「小朋友你看」組合
                if any(0 < e["start"] - xt <= 5.0 for xt in xiaopengou_times):
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
        # 🌟 新增：保底檢查——切換走之前，前一階段（1~4）若還沒有任何 T0 紀錄，
        # 用目前暫存的資料立刻補建一筆 fallback，不管 FALLBACK_T0_DELAY_SEC
        # 秒數到了沒有。避免階段停留時間短於門檻秒數時，該階段被整個跳過、
        # 完全沒有任何紀錄（比用錯的 fallback 時間還糟）。
        if previous_stage in [1, 2, 3, 4]:
            has_record = any(r["stage"] == previous_stage for r in self.trigger_event_records)
            if not has_record:
                if self.stage_gaze_buffer:
                    fb_t0 = (self.current_stage_enter_time + STAGE1_BUFFER_START_SEC
                              if previous_stage == 1 else self.current_stage_enter_time)
                    self._create_fallback_record(previous_stage, fb_t0, time_sec,
                                                  self.stage_gaze_buffer, "保底")
                # else：連暫存資料都沒有（階段停留時間短於暫存起始秒數），
                # 誠實留白，不編造沒有依據的 T0。
        self.stage_gaze_buffer = []

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

        # 🌟 新增：T0 確定前先把逐幀資料暫存起來，等 T0 確定（關鍵字或
        # fallback）後一次重播進正式紀錄，避免「T0 確定前」的張望被漏記。
        # Stage 1 前面通常有寒暄，暫存延後 STAGE1_BUFFER_START_SEC 秒才開始；
        # Stage 2~4 銜接快，進入當下就開始暫存。
        if current_stage in [1, 2, 3, 4] and not already_active:
            _buffer_start_offset = STAGE1_BUFFER_START_SEC if current_stage == 1 else 0.0
            if time_sec - self.current_stage_enter_time >= _buffer_start_offset:
                self.stage_gaze_buffer.append(
                    (time_sec, child_is_pointing_hit, child_is_gazing_at, child_is_gazing_at_tester)
                )

        for event in self.speech_events:
            if event["id"] in self.processed_speech_event_ids or event["start"] > time_sec:
                continue
            new_record = None

            # 🌟 修改：Stage 1-4 的 T0 觸發詞增加「看這裡」
            # 施測者實際用語不只「你看」；幻覺段落已在載入時過濾，安全
            if current_stage in [1, 2, 3, 4] and ("你看" in event["text"] or "看這裡" in event["text"]):
                # 🌟 修正：新建 T0 紀錄前，檢查這句「你看」的時間點是否真的落在
                # 本階段的時間窗內（>= 本階段進入時間）。
                # 原因：Trigger Lock 期間發生的「你看」事件會被延後處理（保留到
                # processed_speech_event_ids 標記為已處理之前），若延後到下一階段
                # 才被消耗，時間點其實屬於「上一階段」，卻會被當成「現在階段」的
                # T0 收下——導致每個階段都拿到前一階段的你看時間，整條鏈錯位一格。
                if not already_active and event["start"] >= self.current_stage_enter_time:
                    new_record = create_trigger_record(
                        self.stage_names.get(current_stage, "Stage"+str(current_stage)),
                        current_stage, event["start"], 10**9, "object", "tester"
                    )
                elif event["start"] >= self.current_stage_enter_time - 0.5:
                    # 🌟 新增：此階段已有「OCR 代償」紀錄且尚未計次
                    # → 關鍵字補到了，把 t0 升級為關鍵字時間（更精確的 RT 基準）
                    for r in self.active_trigger_records:
                        if (r["stage"] == current_stage and not r.get("closed")
                                and r.get("_t0_source") == "ocr"
                                and r["tb"] is None and r["th"] is None and r["pointing_t"] is None):
                            r["t0"] = float(event["start"])
                            r["_t0_source"] = "keyword"
                            self.event_logs.append("[" + str(round(time_sec,1)) + "s] T0升級(keyword): " + r["label"] + " -> " + str(round(r["t0"],2)) + "s")
                            self.processed_speech_event_ids.add(event["id"])
                            break

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
                        event["start"], event["start"] + 10.0, "object", "robot_box"
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
                # 🌟 新增：此事件已用於建立 T0，立即作廢——
                # 防止同一句「你看」在下一個階段又被拿去建紀錄（t0 失真）
                self.processed_speech_event_ids.add(event["id"])
                if new_record["stage"] in [1, 2, 3, 4]:
                    # 🌟 新增：關鍵字 T0 確定，把 T0 確定前暫存的資料重播進來
                    self._replay_buffer_into_record(new_record, self.stage_gaze_buffer)
                    self.stage_gaze_buffer = []

            # ============================================================
            # 🌟 修改：事件作廢防護（修「階段認出來了卻沒記錄」）
            # 舊邏輯：時間一過 event["end"] 就作廢。但 Trigger Lock 會延後
            # 階段切換，等 current_stage 真的切過去時，該階段的起點關鍵字
            # 早已被作廢 → T0 永遠建立不了、整關漏記。
            # 新邏輯：
            #   1. 事件若是「尚未抵達的階段」的起點（對應 stage_start_times），
            #      無限保留，直到該階段真正開始、record 建立後才作廢。
            #   2. 一般「你看」事件（Stage 1-4 用）給 8 秒寬限期，
            #      涵蓋階段鎖造成的延後切換（t0 仍用事件原始時間，不失真）。
            # ============================================================
            if time_sec > event["end"]:
                keep_alive = False
                for s, t in self.stage_start_times.items():
                    if abs(event["start"] - t) < 0.05 and current_stage < s:
                        keep_alive = True
                        break
                if not keep_alive and ("你看" in event.get("text", "") or "看這裡" in event.get("text", "")) \
                        and time_sec <= event["end"] + 8.0:
                    keep_alive = True
                if not keep_alive:
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
                    
            # 🌟 修改：同語音事件——Stage 8 尚未開始前，其起點雜音事件不作廢，
            # 等切換到 Stage 8、T0 建立後才標記處理，防止延後切換造成漏記
            if time_sec > noise_event["end"]:
                is_pending_s8_start = (abs(noise_event["start"] - self.stage_start_times.get(8, -1)) < 0.05
                                       and current_stage < 8)
                if not is_pending_s8_start:
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

        # ============================================================
        # 🌟 新增：Stage 1-4 OCR 代償 T0
        # Whisper 幻覺/漏轉錄導致該階段完全沒有「你看/看這裡」事件時
        # （例：86 前 60 秒整段是幻覺文字），牌子有翻、階段有進入，
        # 卻永遠建立不了 T0 → 整關漏記。
        # 進入階段超過等待秒數仍無紀錄，改以代償時間建立；之後關鍵字
        # 若補到，上方升級邏輯會把 t0 修正為關鍵字時間。
        # Stage 1 前面通常有寒暄，等待秒數（15s）跟代償時間（進入+5.5s，
        # 對齊暫存起始點）都比 Stage 2~4（4s／進入時間）更寬鬆。
        # ============================================================
        if current_stage in [1, 2, 3, 4] and not already_active:
            _delay = FALLBACK_T0_DELAY_SEC_STAGE1 if current_stage == 1 else FALLBACK_T0_DELAY_SEC
            if time_sec - self.current_stage_enter_time >= _delay:
                _fb_t0 = (self.current_stage_enter_time + STAGE1_BUFFER_START_SEC
                          if current_stage == 1 else self.current_stage_enter_time)
                self._create_fallback_record(current_stage, _fb_t0, time_sec,
                                              self.stage_gaze_buffer, "ocr代償")
                self.stage_gaze_buffer = []
                already_active = True

        # --------------------------------------------------
        # 2. Pointing / TB / TH 計次
        # 🌟 抽成共用函式 _apply_frame_to_record，讓即時逐幀計算跟
        # 暫存緩衝區的事後重播（見 _replay_buffer_into_record）用同一套規則。
        # --------------------------------------------------
        for record in self.active_trigger_records:
            self._apply_frame_to_record(record, time_sec, current_stage, child_is_pointing_hit,
                                        child_is_gazing_at, child_is_gazing_at_tester,
                                        gaze_result, robot_boxes, is_gazing_at_box_func)

    def _apply_frame_to_record(self, record, time_sec, current_stage, child_is_pointing_hit,
                               child_is_gazing_at, child_is_gazing_at_tester,
                               gaze_result, robot_boxes, is_gazing_at_box_func):
        """對單一 record 套用某個時間點的 Pointing/TB/TH 判定。

        供兩處呼叫：即時逐幀迴圈（每幀呼叫一次），以及
        _replay_buffer_into_record（T0 確定後，把暫存的緩衝區資料
        依時間順序一次套用進來，補上 T0 確定前漏接的部分）。
        """
        if record.get("closed") or time_sec < record["t0"]:
            return
        if time_sec > record["end_time"]:
            record["closed"] = True
            return
        if isinstance(record["stage"], int) and current_stage != record["stage"]:
            record["closed"] = True
            return

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
        # 🌟 修改：交替計次 + 上升邊緣 + 冷卻
        # 交替條件：上一次計次的目標不能是物品。
        # → 持續看物品不管多久只算一次；視線移開又看回物品也不再計；
        #   必須先轉頭看人（計了 TH）之後，再看回物品才算 TB 第二次。
        tb_hit = child_is_gazing_at if record["tb_mode"] == "object" else False
        if record["tb_mode"] is not None and tb_hit and not record["_prev_tb"]:
            tb_alternation_ok = (record["_last_counted_target"] != "object")
            if tb_alternation_ok and time_sec - record["_last_tb_time"] > TB_COOLDOWN_SEC:
                if record["tb"] is None:
                    record["tb"] = time_sec
                    record["tb_count"] = 1
                    self.event_logs.append("[" + str(round(time_sec,1)) + "s] TB#1: " + record["label"] + " RT=" + str(round(time_sec-record["t0"],2)) + "s")
                else:
                    record["tb_count"] += 1
                    self.event_logs.append("[" + str(round(time_sec,1)) + "s] TB#" + str(record["tb_count"]) + ": " + record["label"])
                record["_last_tb_time"] = time_sec
                record["_last_counted_target"] = "object"  # 🌟 下一次計次必須是 TH（看人）
        record["_prev_tb"] = tb_hit

        # --- TH ---
        # 需先達成 TB（若此 Stage 有 TB 條件）
        # 🌟 修改：交替計次 + 上升邊緣 + 冷卻
        # 交替條件（有 TB 目標的階段）：上一次計次的目標必須是物品。
        # → 持續看人不管多久只算一次；視線移開又看回人也不再計；
        #   必須先看回物品（計了 TB）之後，再轉頭看人才算 TH 第二次。
        # 無 TB 目標的階段（如 Stage 8 怪聲）沒有物品可交替，
        # 維持原本上升邊緣 + 冷卻的判定。
        th_allowed = record["tb_mode"] is None or record["tb"] is not None
        th_hit = False
        if th_allowed:
            if record["th_mode"] == "tester":
                th_hit = child_is_gazing_at_tester
            elif record["th_mode"] == "robot_box":
                th_hit = any(is_gazing_at_box_func(gaze_result, box) for box in robot_boxes)
        if th_allowed and th_hit and not record["_prev_th"]:
            th_alternation_ok = (record["tb_mode"] is None) or (record["_last_counted_target"] == "object")
            if th_alternation_ok and time_sec - record["_last_th_time"] > TH_COOLDOWN_SEC:
                if record["th"] is None:
                    record["th"] = time_sec
                    record["th_count"] = 1
                    self.event_logs.append("[" + str(round(time_sec,1)) + "s] TH#1: " + record["label"] + " +" + str(round(time_sec-record["t0"],2)) + "s")
                else:
                    record["th_count"] += 1
                    self.event_logs.append("[" + str(round(time_sec,1)) + "s] TH#" + str(record["th_count"]) + ": " + record["label"])
                record["_last_th_time"] = time_sec
                record["_last_counted_target"] = "head"  # 🌟 下一次計次必須是 TB（看回物品）
        record["_prev_th"] = th_hit

    def _replay_buffer_into_record(self, record, buffer):
        """把暫存緩衝區（T0 確定前收集的逐幀資料）依時間順序套用進剛建立的 record。"""
        for (t, p_hit, gaze_obj, gaze_tester) in buffer:
            self._apply_frame_to_record(record, t, record["stage"], p_hit, gaze_obj, gaze_tester,
                                        None, [], lambda _g, _b: False)

    def _create_fallback_record(self, stage, t0, time_sec, buffer, tag):
        """建立 fallback T0 紀錄（進入階段太久沒等到關鍵字，或階段即將切換的保底），
        並把暫存緩衝區重播進去，回傳建立的 record。"""
        lbl = self.stage_names.get(stage, "Stage"+str(stage))
        fb_record = create_trigger_record(lbl, stage, t0, 10**9, "object", "tester")
        fb_record["_t0_source"] = "ocr"
        self.trigger_event_records.append(fb_record)
        self.active_trigger_records.append(fb_record)
        self.event_logs.append("[" + str(round(time_sec,1)) + "s] T0(" + tag + "): " + lbl)
        self._replay_buffer_into_record(fb_record, buffer)
        return fb_record

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
            f.write("  Counting rule: TB/TH alternation (gaze must switch object<->head to count again)\n")
            f.write("=" * 40 + "\n")
            ordered_records = sorted(self.trigger_event_records, key=lambda r: r.get("t0", 0.0))
            for idx, r in enumerate(ordered_records, 1):
                t0 = r["t0"]
                f.write("\n" + str(idx).zfill(2) + ". Stage " + str(r["stage"]) + " -- " + r["label"] + format_stage_score_suffix(r) + "\n")
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