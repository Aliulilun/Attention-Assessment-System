#!/bin/bash
# ============================================================
#  注意力分析結果檢視器 — macOS / Linux 啟動器
#
#  第一次執行會建立一個獨立的 Python 虛擬環境並安裝 PySide6，
#  大約需要 1~3 分鐘。之後每次啟動只要幾秒。
#
#  這個檢視器只用來閱讀分析報告，不含 AI 分析功能，
#  因此不需要顯示卡，也不會下載任何模型。
# ============================================================
set -u
cd "$(dirname "$0")"
ROOT="$(pwd)"

echo
echo "============================================================"
echo "  注意力分析結果檢視器"
echo "============================================================"
echo "  資料夾：$ROOT"
echo

pause_and_exit() {
    echo
    echo "按 Enter 鍵關閉這個視窗…"
    read -r _ || true
    exit "${1:-1}"
}

# ── 找 Python 3 ──────────────────────────────────────
PY=""
for cand in python3.12 python3.11 python3.10 python3.9 python3; do
    if command -v "$cand" >/dev/null 2>&1; then PY="$cand"; break; fi
done

if [ -z "$PY" ]; then
    echo "[錯誤] 找不到 Python 3。"
    echo
    echo "  macOS 請先安裝 Python："
    echo "    方式一：到 https://www.python.org/downloads/ 下載安裝"
    echo "    方式二：終端機執行  brew install python"
    pause_and_exit 1
fi

echo "  Python：$($PY --version 2>&1)  ($(command -v "$PY"))"

# ── 建立虛擬環境（只做一次）────────────────────────
# 用 venv 而不是直接 pip install，是為了不污染系統 Python；
# macOS 內建的 Python 受系統保護，直接安裝套件會被擋。
if [ ! -x ".venv/bin/python" ]; then
    echo
    echo "  [1/2] 首次執行：建立虛擬環境…"
    rm -rf .venv
    "$PY" -m venv .venv || {
        echo "[錯誤] 建立虛擬環境失敗。"
        echo "       若是 Debian/Ubuntu，請先安裝：sudo apt install python3-venv"
        pause_and_exit 1
    }
    echo "  [2/2] 安裝介面套件（PySide6、pandas）…這一步需要網路，約 1~3 分鐘"
    ./.venv/bin/python -m pip install --upgrade pip --quiet
    ./.venv/bin/python -m pip install --quiet PySide6 pandas openpyxl || {
        echo
        echo "[錯誤] 套件安裝失敗，可能是網路問題。"
        echo "       請確認網路後刪除 .venv 資料夾再執行一次。"
        pause_and_exit 1
    }
    echo "  安裝完成。"
else
    echo "  虛擬環境已就緒。"
fi

echo
echo "  啟動中…"
echo

./.venv/bin/python app/ui/viewer.py "$@"
STATUS=$?

if [ $STATUS -ne 0 ]; then
    echo
    echo "[錯誤] 程式異常結束（代碼 $STATUS），訊息如上。"
    pause_and_exit $STATUS
fi
