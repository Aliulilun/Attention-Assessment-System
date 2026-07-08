import os
import json
import subprocess
import sys

class SpeechTrigger:
    def __init__(self, video_path, output_dir, keywords, noise_sample_path=None):
        self.video_path = video_path
        self.output_dir = output_dir
        self.keywords = keywords
        self.cache_path = os.path.join(output_dir, "speech_cache.json")
        # 🌟 修改：若外部明確傳入 noise_sample_path，優先使用；
        #          否則退回自動偵測 output_dir/noise_reference_2m23_2m33.wav
        if noise_sample_path is not None:
            self.noise_sample_path = noise_sample_path
        else:
            self.noise_sample_path = os.path.join(
                output_dir,
                "noise_reference_2m23_2m33.wav",
            )
        self.transcript_dict = {}

    def get_trigger_windows(self):
        """
        利用獨立行程 (Subprocess) 啟動語音大腦，徹底避免記憶體崩潰。
        若 speech_cache.json 已存在，直接讀取快取，不重新啟動 Whisper 子行程。
        """
        # 🌟 修改：快取命中時直接讀取，跳過 Whisper 分析（省 30~120s 冷啟動時間）
        if os.path.exists(self.cache_path):
            print(f">>> [SpeechTrigger] 快取已存在，直接讀取（跳過 Whisper）：{self.cache_path}")
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                records = data.get("segment_records", [])
                self.transcript_dict = {rec["start"]: rec["text"] for rec in records}
                windows = data.get("trigger_windows", [])
                return [(float(w[0]), float(w[1])) for w in windows]
            except Exception as e:
                print(f"⚠️ [SpeechTrigger] 快取讀取失敗（{e}），重新執行 Whisper 分析...")

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
                "0.65",
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

            # 讀取並回傳時間窗
            windows = data.get("trigger_windows", [])
            return [(float(w[0]), float(w[1])) for w in windows]
        else:
            print("⚠️ [SpeechTrigger] 找不到語音快取檔。")
            return []

    def is_in_window(self, current_time_sec, trigger_windows):
        return any(start <= current_time_sec <= end for start, end in trigger_windows)

    def check_voice_override(self, current_time_sec, keyword="機器人", time_tolerance=0.5):
        for start_time, text in self.transcript_dict.items():
            if abs(current_time_sec - start_time) < time_tolerance and keyword in text:
                print(f"\n🎙️ [Voice Override] 偵測到關鍵字「{keyword}」！強制覆寫系統狀態 ({current_time_sec:.1f}s)")
                return True
        return False