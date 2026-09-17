import os
import json
import subprocess
import sys
from collections import deque

EXPECTED_MATCHING_ALGORITHM_VERSION = 17


def _resolve_python_executable():
    """🌟 新增：回傳可用於啟動子行程的 python 直譯器路徑。

    run_ui.bat 是用 pythonw.exe 啟動 UI 的（刻意不帶主控台），此時
    sys.executable 也會是 pythonw.exe。pythonw 沒有標準輸出，
    子行程一旦 print() 就會直接拋例外並以 exit code 1 結束。
    若偵測到 pythonw.exe，改用同目錄的 python.exe。
    """
    exe = sys.executable or ""
    base = os.path.basename(exe).lower()
    if base.startswith("pythonw"):
        candidate = os.path.join(os.path.dirname(exe), base.replace("pythonw", "python", 1))
        if os.path.exists(candidate):
            return candidate
    return exe

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

    @staticmethod
    def _filter_hallucination_windows(windows, records):
        """
        過濾 Whisper 把 initial_prompt 複誦成辨識文字後產生的假觸發窗。
        這類片段常包含「請勿忽略」或「短促的聲音」，會讓階段切換被假語音卡住。
        """
        markers = ["請勿忽略", "短促的聲音"]
        bad_spans = []
        for rec in records:
            if any(marker in rec.get("text", "") for marker in markers):
                try:
                    bad_spans.append((float(rec["start"]) - 0.1, float(rec["end"]) + 0.1))
                except (KeyError, TypeError, ValueError):
                    continue

        if not bad_spans:
            return windows

        kept = [
            window
            for window in windows
            if not any(start <= float(window[0]) <= end for start, end in bad_spans)
        ]
        dropped = len(windows) - len(kept)
        if dropped:
            print(f">>> [SpeechTrigger] 過濾 {dropped} 個 Whisper 幻覺假時間窗")
        return kept

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
                cached_video_path = os.path.abspath(
                    data.get("video_signature", {}).get("path", "")
                )
                expected_video_path = os.path.abspath(self.video_path)
                if cache_version != EXPECTED_MATCHING_ALGORITHM_VERSION:
                    print(
                        ">>> [SpeechTrigger] 快取版本過舊，重新啟動 Whisper 子行程 "
                        f"({cache_version} -> {EXPECTED_MATCHING_ALGORITHM_VERSION})"
                    )
                    raise ValueError("stale speech cache")
                # 🌟 修正：Windows 路徑比對改用 os.path.normcase() 做大小寫不敏感比較，
                # 避免磁碟機代號大小寫差異（c:\ vs C:\）導致快取被判定為「屬於其他影片」
                # 而重新啟動 Whisper，連帶覆寫手動編輯過的快取（如 86 的怪聲時間點校正）
                if os.path.normcase(cached_video_path) != os.path.normcase(expected_video_path):
                    print(
                        ">>> [SpeechTrigger] 快取屬於其他影片，重新啟動 Whisper 子行程 "
                        f"({cached_video_path or 'unknown'} -> {expected_video_path})"
                    )
                    raise ValueError("speech cache belongs to another video")
                print(">>> [SpeechTrigger] 快取命中，直接讀取（跳過 Whisper 子行程）")
                records = data.get("segment_records", [])
                self.transcript_dict = {rec['start']: rec['text'] for rec in records}
                # 🌟 新增：讀取怪聲觸發時間窗
                self._load_noise_trigger_windows(data)
                windows = self._filter_hallucination_windows(data.get("trigger_windows", []), records)
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

        # 呼叫獨立的 Python 行程來執行語音辨識
        cmd = [
            _resolve_python_executable(), engine_path,
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
                "0.7",
            ])
            print(f">>> [SpeechTrigger] 使用怪聲範本：{self.noise_sample_path}")

        # ============================================================
        # 🌟 修正：子行程的輸出必須導到 pipe，不能繼承父行程的控制代碼
        #
        # 問題根因：run_ui.bat 第 63 行用 pythonw.exe 啟動 UI（刻意不帶
        # 主控台），因此這個行程的 sys.stdout / sys.stderr 是無效的。
        # 原本的 subprocess.run(cmd, check=True) 沒有做任何重導向，
        # 子行程就繼承了那組壞掉的控制代碼 → speech_engine.py 解析完參數
        # 後的第一個 print()（第 2150 行，位置在 try 區塊之前）直接拋例外
        # → 行程以 exit code 1 結束。而且子行程的 stderr 同樣是死的，
        # 所以完全看不到任何訊息，只剩下一句「Return code: 1」。
        # 症狀：UI 跑沒有語音快取的影片必定失敗，指令列卻完全正常。
        #
        # 修法：改用 Popen + PIPE 逐行讀取——
        #   1. 子行程拿到有效的 pipe 控制代碼 → print() 不再炸掉（根治）
        #   2. 逐行轉印到父行程的 sys.stdout → UI 的主控台面板也看得到
        #      Whisper 的即時進度（原本是完全空白的）
        #   3. 子行程真的失敗時，把最後幾行輸出一併印出來當錯誤摘要，
        #      不再只給一個無從查起的 return code
        # ============================================================
        # 🌟 不要讓子行程彈出額外的主控台視窗。
        # 換成 python.exe 之後，Windows 預設會配一個新的 console 給它，
        # 從 UI 啟動時畫面上就會多冒出一個黑視窗。輸出已經全部走 PIPE，
        # 子行程根本不需要 console，用 CREATE_NO_WINDOW 關掉。
        _popen_kwargs = {}
        if os.name == "nt":
            _popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)

        # ============================================================
        # 🌟 關鍵：強制子行程用 UTF-8 輸出
        #
        # Windows 上的 Python 寫到 pipe 時，sys.stdout.encoding 取的是
        # 系統地區編碼（繁中環境是 CP950），而不是 UTF-8——就算主控台
        # 已經 chcp 65001 也沒用，那只管 console、管不到 pipe。
        #
        # 而 speech_engine.py 的訊息含 emoji（❌ 🔊 ⚠️ 🌟）。
        # CP950 編不出 emoji → print() 直接拋 UnicodeEncodeError →
        # 未被接住 → 子行程以 exit code 1 結束。
        #
        # 這正是「指令列跑得起來、UI 一定失敗」的原因：
        #   指令列 → 輸出到 chcp 65001 的 console → UTF-8 → 正常
        #   pipe   → CP950 → 遇到第一個 emoji 就炸
        #
        # PYTHONIOENCODING 指定標準串流編碼；PYTHONUTF8=1 進一步開啟
        # 直譯器的 UTF-8 模式（Python 3.7+），連檔案 I/O 預設也轉 UTF-8，
        # 與 CLAUDE.md「全專案一律 UTF-8」的規定一致。
        # ============================================================
        _env = os.environ.copy()
        _env["PYTHONIOENCODING"] = "utf-8"
        _env["PYTHONUTF8"] = "1"
        _popen_kwargs["env"] = _env

        # 🌟 子行程的完整輸出另外寫成 log 檔。
        # UI 的主控台面板可能被捲掉或被使用者忽略，留一份檔案才查得到原因。
        log_path = os.path.join(self.output_dir, "_speech_engine.log")
        try:
            os.makedirs(self.output_dir, exist_ok=True)
        except Exception:
            pass

        captured = deque(maxlen=2000)   # 全量輸出（上限 2000 行）供寫檔
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,   # stderr 併入 stdout，順序才不會亂
                text=True,
                encoding="utf-8",
                errors="replace",           # 子行程若吐出非 UTF-8 位元組不致整個炸掉
                bufsize=1,                  # 行緩衝，進度才會即時出現
                **_popen_kwargs,
            )
            if proc.stdout is not None:
                for line in proc.stdout:
                    line = line.rstrip()
                    captured.append(line)
                    print(line)
            returncode = proc.wait()
        except Exception as e:
            print(f"❌ [SpeechTrigger] 無法啟動語音分析子行程：{e}")
            print(f"   指令：{cmd[0]}")
            return []

        # 不論成敗都留下 log，方便事後追查
        try:
            with open(log_path, "w", encoding="utf-8") as _lf:
                _lf.write("command: " + " ".join(cmd) + "\n")
                _lf.write("returncode: " + str(returncode) + "\n")
                _lf.write("-" * 60 + "\n")
                _lf.write("\n".join(captured))
        except Exception:
            pass

        if returncode != 0:
            print(f"❌ [SpeechTrigger] 語音分析發生錯誤 (Return code: {returncode})")
            if captured:
                _tail = list(captured)[-30:]
                print("---- 子行程最後 %d 行輸出 ----" % len(_tail))
                for line in _tail:
                    print(line)
                print("--------------------------------")
            else:
                print("   ⚠️ 子行程沒有產生任何輸出——通常代表它在 Python 啟動或")
                print("      模組載入階段就死了（例如 DLL / CUDA 載入失敗）。")
            print(f"   完整記錄：{log_path}")
            return []
        print(">>> [SpeechTrigger] 聽覺大腦分析完畢！讀取結果...")

        # 讀取 speech_engine.py 寫好的 JSON 快取檔
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 建立 Voice Override 字典
            records = data.get("segment_records", [])
            self.transcript_dict = {rec['start']: rec['text'] for rec in records}
            # 🌟 新增：讀取怪聲觸發時間窗
            self._load_noise_trigger_windows(data)

            # 讀取並回傳時間窗（過濾 Whisper 幻覺假窗）
            windows = self._filter_hallucination_windows(data.get("trigger_windows", []), records)
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