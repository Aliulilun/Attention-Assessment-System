# 語音模組安裝說明

這份設定以 ` macOS ` 為主，適合目前專題的 ` Whisper + ffmpeg ` 語音流程。

注意：你原本的系統 ` python3 ` 是 ` 3.9.6 ` ，不建議直接拿它安裝 ` PyTorch ` 。目前這份設定改以 ` Homebrew Python 3 ` 和 ` .venv314 ` 為主。

另外，你這台 ` macOS ` 上的 ` Homebrew Python ` 有 ` pyexpat ` 載入問題，所以執行 ` python ` 或 ` pip ` 相關命令前，建議先補上：

```bash
export DYLD_LIBRARY_PATH=/opt/homebrew/opt/expat/lib
```

## 1. 建議版本

- ` Homebrew Python 3 `
- ` ffmpeg `
- ` torch `
- ` torchaudio `
- ` openai-whisper `

## 2. 第一次安裝

先安裝 ` Python ` 與 ` ffmpeg ` ：

```bash
brew install python3
brew install ffmpeg
```

再到專案目錄執行：

```bash
chmod +x setup_audio_env.sh
./setup_audio_env.sh
```

## 3. 手動安裝版本

如果你不想用腳本，也可以手動執行：

```bash
export DYLD_LIBRARY_PATH=/opt/homebrew/opt/expat/lib
python3 -m venv .venv314
source .venv314/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchaudio
python -m pip install -r requirements-audio.txt
```

## 4. 驗證安裝

```bash
export DYLD_LIBRARY_PATH=/opt/homebrew/opt/expat/lib
source .venv314/bin/activate
ffmpeg -version
python -m whisper --help
python audio_trigger_pipeline.py --help
```

## 5. 執行語音模組

```bash
export DYLD_LIBRARY_PATH=/opt/homebrew/opt/expat/lib
source .venv314/bin/activate
python audio_trigger_pipeline.py --video ./video/8.mp4
```

如果你要強制重新跑一次 ` Whisper ` ，可加上：

```bash
export DYLD_LIBRARY_PATH=/opt/homebrew/opt/expat/lib
python audio_trigger_pipeline.py --video ./video/8.mp4 --force
```
