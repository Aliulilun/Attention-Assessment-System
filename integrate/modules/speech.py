import os
import json
import subprocess
import sys

EXPECTED_MATCHING_ALGORITHM_VERSION = 15

class SpeechTrigger:
    def __init__(self, video_path, output_dir, keywords, noise_sample_path=None):
        self.video_path = video_path
        self.output_dir = output_dir
        self.keywords = keywords
        self.cache_path = os.path.join(output_dir, "speech_cache.json")

        # 🌟 怪聲參考音檔路徑：
        #   - 若外部明確傳入 noise_sample_path → 直接使用（適合批次模式，音檔放 model/ 目錄）
        #   - 未傳入 → 回退到 output_dir 下的預設名稱（維持舊版相容）
        if noise_sample_path is not None:
            self.noise_sample_path = noise_sample_path
        else:
            self.noise_sample_path = os.path.join(
                output_dir,
                "noise_reference_2m23_2m33.wav",
            )

        self.transcript_dict = {}
        # 🌟 新增：怪聲偵測時間窗清單（由 noise.wav 模板比對產生）
        self.noise_trigger_windows = []

    def _load_noise_trigger_windows(self, data: dict):
        """
        🌟 新增：從快取 JSON 中讀取 noise_events 的 trigger_window，
        填入 self.noise_trigger_windows，供 is_in_noise_window() 使用。
        只讀取有 trigger_window 且未被 rejected 的事件。
        """
        noise_events = data.get("noise_events", [])
        self.noise_trigger_windows = [
            (float(e["trigger_window"][0]), float(e["trigger_window"][1]))
            for e in noise_events
            if isinstance(e.get("trigger_window"), (list, tuple))
            and len(e["trigger_window"]) >= 2
            and not e.get("rejected_reason")
        ]
        if self.noise_trigger_windows:
            print(f">>> [SpeechTrigger] 載入 {len(self.noise_trigger_windows)} 個怪聲觸發時間窗：{self.noise_trigger_windows}")
        else:
            print(">>> [SpeechTrigger] 無怪聲觸發時間窗（noise.wav 未命中或未提供）")

    def get_trigger_windows(self):
        """
        利用獨立行程 (Subprocess) 啟動語音大腦，徹底避免記憶體崩潰。
        🌟 優化：快取存在時直接讀取，完全跳過子行程冷啟動（省 30~120s）。
        """
        # ── 快取命中：直接讀取，完全不啟動子行程 ────────────────────────────
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                cache_version = (
                    data.get("config", {}).get("matching_algorithm_version")
                )
                if cache_version != EXPECTED_MATCHING_ALGORITHM_VERSION:
                    print(
                        ">>> [SpeechTrigger] 快取版本過舊，重新啟動 Whisper 子行程 "
                        f"({cache_version} -> {EXPECTED_MATCHING_ALGORITHM_VERSION})"
                    )
                    raise ValueError("stale speech cache")
                print(">>> [SpeechTrigger] 快取命中，直接讀取（跳過 Whisper 子行程）")
                records = data.get("segment_records", [])
                self.transcript_dict = {rec['start']: rec['text'] for rec in records}
                # 🌟 新增：讀取怪聲觸發時間窗
                self._load_noise_trigger_windows(data)
                windows = data.get("trigger_windows", [])
                return [(float(w[0]), float(w[1])) for w in windows]
            except ValueError:
                pass
            except Exception as e:
                print(f"⚠️ [SpeechTrigger] 快取讀取失敗 ({e})，重新啟動 Whisper 子行程")

        # ── 無快取：啟動 Whisper 子行程進行語音辨識 ──────────────────────────
        print(">>> [SpeechTrigger] 啟動聽覺大腦 (獨立行程隔離中)...")

        # 取得 speech_engine.py 的絕對路徑
        base_dir = os.path.dirname(os.path.abspath(__file__))
        engine_path = os.path.join(base_dir, "speech_engine.py")

        # 組合關鍵字字串
        kw_str = " ".join(self.keywords)

        # 呼叫獨立的 Python 行程來執行語音辨識
        cmd = [
            sys.executable, engine_path,
            "--video", self.video_path,
            "--output-dir", self.output_dir,
            "--model", "large-v3",
            "--keywords"
        ] + self.keywords

        if os.path.exists(self.noise_sample_path):
            cmd.extend([
                "--noise-sample",
                self.noise_sample_path,
                "--noise-template-threshold",
                "0.85",
            ])
            print(f">>> [SpeechTrigger] 使用怪聲範本：{self.noise_sample_path}")

        try:
            # 啟動獨立行程，並等待它執行完畢
            subprocess.run(cmd, check=True)
            print(">>> [SpeechTrigger] 聽覺大腦分析完畢！讀取結果...")
        except subprocess.CalledProcessError as e:
            print(f"❌ [SpeechTrigger] 語音分析發生錯誤 (Return code: {e.returncode})")
            return []

        # 讀取 speech_engine.py 寫好的 JSON 快取檔
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 建立 Voice Override 字典
            records = data.get("segment_records", [])
            self.transcript_dict = {rec['start']: rec['text'] for rec in records}
            # 🌟 新增：讀取怪聲觸發時間窗
            self._load_noise_trigger_windows(data)

            # 讀取並回傳時間窗
            windows = data.get("trigger_windows", [])
            return [(float(w[0]), float(w[1])) for w in windows]
        else:
            print("⚠️ [SpeechTrigger] 找不到語音快取檔。")
            return []

    def is_in_window(self, current_time_sec, trigger_windows):
        return any(start <= current_time_sec <= end for start, end in trigger_windows)

    def is_in_noise_window(self, current_time_sec: float) -> bool:
        """
        🌟 新增：判斷當前時間是否落在 noise.wav 模板命中的怪聲觸發時間窗內。
        用於 Stage 7→8 的聽覺代償判定，取代舊版的 Whisper 關鍵字偵測。
        需先呼叫 get_trigger_windows() 讓 self.noise_trigger_windows 完成填充。
        """
        return any(start <= current_time_sec <= end for start, end in self.noise_trigger_windows)

    def check_voice_override(self, current_time_sec, keyword="機器人", time_tolerance=0.5):
        for start_time, text in self.transcript_dict.items():
            if abs(current_time_sec - start_time) < time_tolerance and keyword in text:
                print(f"\n🎙️ [Voice Override] 偵測到關鍵字「{keyword}」！強制覆寫系統狀態 ({current_time_sec:.1f}s)")
                return True
        return False