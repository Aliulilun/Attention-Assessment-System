# Speech Recognition Module

這個資料夾是早療所專案中的語音辨識模組，負責從影片中擷取音訊、進行中文語音辨識，並整理出可供後續行為分析使用的觸發時間窗。

## 功能介紹

- 使用 ` ffmpeg ` 從影片中抽取音訊。
- 使用 ` OpenAI Whisper ` 進行中文語音轉文字。
- 偵測指定語音關鍵字，例如「開始」、「準備」、「你看」、「看這裡」等。
- 依照關鍵字出現時間建立反應時間窗，預設為觸發後 ` 3 ` 秒。
- 偵測警報型怪聲或高頻聲音事件，並建立對應時間窗。
- 合併重疊的語音與怪聲觸發時間窗。
- 產生文字報告與快取，避免相同影片重複分析。

## 資料夾內容

```text
speech recognition/
├── README.md
├── AUDIO_SETUP.md
├── audio_trigger_pipeline.py
├── requirements-audio.txt
└── setup_audio_env.sh
```

## 環境需求

建議使用 ` macOS `、` Homebrew Python 3 ` 與 ` ffmpeg `。

第一次使用前，請先安裝基本工具：

```bash
brew install python3
brew install ffmpeg
```

接著進入此資料夾並建立語音模組環境：

```bash
cd "/Users/huangyuxiang/Documents/早療所專案/speech recognition"
chmod +x setup_audio_env.sh
./setup_audio_env.sh
```

若執行時遇到 ` pyexpat ` 載入問題，請先設定：

```bash
export DYLD_LIBRARY_PATH=/opt/homebrew/opt/expat/lib
```

更完整的安裝細節請看 ` AUDIO_SETUP.md `。

## 使用方式

啟用虛擬環境：

```bash
cd "/Users/huangyuxiang/Documents/早療所專案/speech recognition"
export DYLD_LIBRARY_PATH=/opt/homebrew/opt/expat/lib
source .venv314/bin/activate
```

分析指定影片：

```bash
python audio_trigger_pipeline.py --video ./video/8.mp4
```

自訂輸出資料夾：

```bash
python audio_trigger_pipeline.py --video ./video/8.mp4 --output-dir ./output
```

強制重新分析，忽略既有快取：

```bash
python audio_trigger_pipeline.py --video ./video/8.mp4 --force
```

自訂關鍵字與反應時間窗：

```bash
python audio_trigger_pipeline.py --video ./video/8.mp4 --window 5 --keywords 開始 準備 看這裡
```

只做怪聲偵測，跳過 ` Whisper `：

```bash
python audio_trigger_pipeline.py --video ./video/8.mp4 --skip-whisper
```

停用怪聲偵測，只做語音關鍵字觸發：

```bash
python audio_trigger_pipeline.py --video ./video/8.mp4 --disable-noise-trigger
```

## 輸出結果

預設輸出到 ` ./output `：

- ` transcript_with_events.txt `：逐字稿、語音關鍵字、怪聲事件與合併後觸發時間窗。
- ` speech_cache.json `：分析快取，下一次相同設定可直接載入。
- ` analysis_audio.wav `：從影片抽出的分析用音訊。

` output/ `、` video/ ` 與虛擬環境屬於本機資料，不建議上傳到 GitHub。

## 與 GitHub 版本控制的關係

本機資料夾：

```text
/Users/huangyuxiang/Documents/早療所專案/speech recognition
```

對應到 GitHub 上：

```text
https://github.com/Aliulilun/Attention-Assessment-System/tree/main/speech%20recognition
```

也就是說，只要在本機修改 ` speech recognition/ ` 裡的程式或文件，並用 Git 提交與推送，GitHub 上這個資料夾就會同步更新。

## 手動更新 GitHub

請在整個 repo 的根目錄執行 Git 指令：

```bash
cd "/Users/huangyuxiang/Documents/早療所專案"
git pull --ff-only
git status
git add "speech recognition"
git commit -m "Update speech recognition module"
git push origin main
```

注意：如果只想更新語音辨識模組，請使用：

```bash
git add "speech recognition"
```

不要直接使用 ` git add . `，避免把 `.venv/`、` output/ `、` video/ ` 或其他本機暫存檔一起加入。
