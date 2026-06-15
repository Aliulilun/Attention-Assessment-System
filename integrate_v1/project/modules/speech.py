import os
import json
import subprocess

class SpeechTrigger:
    def __init__(self, video_path, output_dir, keywords):
        self.video_path = video_path
        self.output_dir = output_dir
        self.keywords = keywords
        self.cache_path = os.path.join(output_dir, "speech_cache.json")
        self.transcript_dict = {}

    def get_trigger_windows(self):
        """
        利用獨立行程 (Subprocess) 啟動語音大腦，徹底避免記憶體崩潰
        """
        print(">>> [SpeechTrigger] 啟動聽覺大腦 (獨立行程隔離中)...")
        
        # 取得 speech_engine.py 的絕對路徑
        base_dir = os.path.dirname(os.path.abspath(__file__))
        engine_path = os.path.join(base_dir, "speech_engine.py")

        # 組合關鍵字字串
        kw_str = " ".join(self.keywords)
        
        # 呼叫獨立的 Python 行程來執行語音辨識
        cmd = [
            "python", engine_path,
            "--video", self.video_path,
            "--output-dir", self.output_dir,
            "--model", "large-v3",  
            "--keywords"
        ] + self.keywords

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