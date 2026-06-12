#!/usr/bin/env bash

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv314}"
EXPAT_LIB_DIR="/opt/homebrew/opt/expat/lib"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "找不到 $PYTHON_BIN 。"
  echo "請先安裝 Homebrew Python ，例如： brew install python3"
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "尚未安裝 ffmpeg 。"
  echo "請先執行： brew install ffmpeg"
  exit 1
fi

env DYLD_LIBRARY_PATH="$EXPAT_LIB_DIR" "$PYTHON_BIN" -m venv "$VENV_DIR"

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

env DYLD_LIBRARY_PATH="$EXPAT_LIB_DIR" python -m pip install --upgrade pip setuptools wheel
env DYLD_LIBRARY_PATH="$EXPAT_LIB_DIR" python -m pip install torch torchaudio
env DYLD_LIBRARY_PATH="$EXPAT_LIB_DIR" python -m pip install -r requirements-audio.txt

echo ""
echo "環境建立完成。"
echo "如果你的 macOS 環境也有 pyexpat 載入問題，執行前請先帶上："
echo "export DYLD_LIBRARY_PATH=$EXPAT_LIB_DIR"
echo "啟用方式： source $VENV_DIR/bin/activate"
echo "測試 Whisper ： python -m whisper --help"
