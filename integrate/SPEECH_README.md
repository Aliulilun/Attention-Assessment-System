# 語音辨識與怪聲觸發模組

## 功能目標

`speech_engine.py` 負責從影片或音檔中產生可供主程式使用的觸發時間窗，輸出到 `output/speech_cache.json` 與 `output/transcript_with_events.txt`。

主要功能：

- 使用 Whisper 辨識語音片段與字級時間戳。
- 偵測指定關鍵詞，例如「你看」、「看這裡」、「準備」、「開始」、「321」。
- 將每次語音關鍵詞轉成後續判斷視窗，預設為 3 秒。
- 偵測警報型或玩具型怪聲，轉成後續判斷視窗，預設為 10 秒。
- 支援從外部 wav 或目前影片指定時間截取怪聲參考樣本。

## 目前遇到的問題

### 連續「你看」漏判

Whisper 有時會把連續短促的人聲「你看、你看」合併成單一片段，逐字稿只留下一次「你看」，導致主程式少開一次判斷視窗。

目前改善方式：

- 保留原本的直接關鍵詞命中。
- 保留跨片段關鍵詞修復。
- 保留低信心「前向／前像」修復成「你看」。
- 新增連續「你看」補償：當片段文字剛好是「你看」且片段時間偏長時，在同一片段內補出後續觸發點。

### 怪聲誤判

不同影片的背景聲、音樂、玩具聲或高頻環境音可能被當成怪聲，進而產生過長或不必要的 10 秒判斷視窗。

目前改善方式：

- 有提供怪聲參考樣本時，必須通過樣本相似度門檻才輸出怪聲事件。
- 未提供樣本時，限制單一怪聲候選最長時間，避免長段背景聲被當成單一怪聲。
- 仍保留音樂語境過濾，降低歌曲、音樂、重複語音造成的誤判。

## 測試環境

建議使用專案既有虛擬環境：

```bash
cd /Users/huangyuxiang/Documents/integrate
source .venv/bin/activate
```

確認必要工具：

```bash
python3 -m py_compile modules/speech_engine.py
ffmpeg -version
```

若缺少 Whisper：

```bash
pip install openai-whisper
```

## 測試指令

### 執行測試方法

先進入專案並啟動虛擬環境：

```bash
cd /Users/huangyuxiang/Documents/integrate
source .venv/bin/activate
```

確認語法沒問題：

```bash
python -m py_compile modules/speech_engine.py modules/speech.py
```

只測怪聲範本是否自動套用，不跑 Whisper：

```bash
python modules/speech_engine.py --video video/58.mp4 --output-dir output --skip-whisper --force
```

測完看報告：

```bash
cat output/transcript_with_events.txt
```

確認報告前面有以下內容：

```text
怪聲樣本相似度門檻：0.65
怪聲參考來源：/Users/huangyuxiang/Documents/integrate/output/noise_reference_2m23_2m33.wav
```

測完整語音加怪聲流程：

```bash
python modules/speech_engine.py --video video/58.mp4 --output-dir output --force
```

跑整個專案主流程：

```bash
python main.py
```

正式測試時請確認：

- `output/transcript_with_events.txt` 是否有語音觸發「你看」。
- 報告中是否有「連續你看補償觸發」。
- 怪聲事件是否只保留真正相似的聲音。
- 怪聲事件的 `樣本相似度` 是否高於 `0.65`。
- `合併後觸發時間窗` 是否符合物體辨識與視線辨識要啟動的時間段。

注意：如果曾經跑過 `--skip-whisper`，`output/speech_cache.json` 會是怪聲-only 測試結果，沒有語音逐字稿。正式測完整流程時，請重新跑一次不含 `--skip-whisper` 的指令。

### 常用指令

基本測試：

```bash
python3 modules/speech_engine.py --video video/58.mp4 --output-dir output --force
```

只測怪聲，不跑 Whisper：

```bash
python3 modules/speech_engine.py --video video/58.mp4 --output-dir output --skip-whisper --force
```

使用外部怪聲 wav 樣本：

```bash
python3 modules/speech_engine.py --video video/58.mp4 --output-dir output --noise-sample path/to/noise_sample.wav --force
```

目前 `speech_engine.py` 與主流程會自動套用以下範本：

```text
/Users/huangyuxiang/Documents/integrate/output/noise_reference_2m23_2m33.wav
```

只要這個檔案存在於本次 `--output-dir`，即使沒有手動輸入 `--noise-sample`，`speech_engine.py` 也會自動使用它，並使用 `--noise-template-threshold 0.65` 降低怪聲誤判。

從影片內指定時間截取怪聲樣本，例如 02:23 開始截 2.5 秒：

```bash
python3 modules/speech_engine.py --video video/58.mp4 --output-dir output --noise-reference-start 2:23 --noise-reference-duration 2.5 --force
```

調高怪聲樣本門檻，降低誤判：

```bash
python3 modules/speech_engine.py --video video/58.mp4 --output-dir output --noise-sample path/to/noise_sample.wav --noise-template-threshold 0.68 --force
```

縮短未使用樣本時允許的怪聲候選長度：

```bash
python3 modules/speech_engine.py --video video/58.mp4 --output-dir output --noise-max-duration 2.5 --force
```

## 測試重點

測完請看：

- `output/transcript_with_events.txt`
- `output/speech_cache.json`

檢查項目：

- 連續「你看」是否出現多個觸發事件。
- 報告中是否標示「連續你看補償觸發」。
- 怪聲事件是否只保留真正接近樣本的聲音。
- `樣本相似度` 是否高於目前門檻。
- `合併後觸發時間窗` 是否符合主程式要開啟物體辨識與視線辨識的區間。

## 調參建議

- `--noise-template-threshold`：越高越保守。若誤判多，建議從 `0.58` 調到 `0.65` 或 `0.70`。
- `--noise-max-duration`：限制非樣本模式下的怪聲候選長度。若常出現長段背景音誤判，建議調到 `2.0` 到 `3.0`。
- `--window`：語音關鍵詞後的判斷視窗秒數，預設 `3.0`。
- `--noise-window`：怪聲後的判斷視窗秒數，預設 `10.0`。
