
---

```markdown
# 🚀 多模態 AI 互動行為分析系統 (Multi-Modal AI Interactive Behavior Analysis)

這是一個結合**電腦視覺 (Computer Vision)** 與**語音辨識 (Speech Recognition)** 的多模態 AI 系統。本系統專為分析「施測者」與「受測兒童」之間的互動行為而設計，具備動態階段辨識、語音關鍵字觸發、以及精準的 3D 空間指向與碰撞判定能力。

## ✨ 核心技術亮點

* **🎙️ 語音觸發時間窗 (Whisper)**：精準抓取特定引導詞（如「開始」、「機器人」），動態啟用視覺分析，節省算力。
* **👁️ 牌子階段動態追蹤 (EasyOCR)**：具備 ROI 結界搜索、防抖動濾波與容錯重置機制，全自動辨識當前測驗階段。為了消除硬體冷啟動延遲，內建了 OCR 暖機 (Warm-up) 機制。
* **🧠 階段適應性物件偵測 (YOLO Multi-Models)**：依據 OCR 讀取到的「當前階段」，動態切換（Lazy Loading）對應的 YOLO 模型（如前方物件、背景、氣球、泡泡、玩具、機器人），極大化運算資源，避免 GPU 顯存溢出。並統一採用嚴格的信心度門檻 (`conf=0.75`) 與大尺寸推論 (`imgsz=960`) 提升精準度。
* **🎯 高階互動行為判定 (MediaPipe & YOLO Pose)**：
    * **動態身分識別**：利用 Arm Link Score 演算法，精準綁定人體與手部特徵，並透過動態邊界 (`DIVIDER_X`) 區分施測者與兒童。
    * **嚴格指向過濾**：透過幾何計算判斷手指是否確實「伸直」，杜絕握拳或手掌邊緣造成的誤判。
    * **精準碰撞偵測**：實作 **Ray-AABB 射線與矩形精確交集演算法**。同時在畫面上繪製目標物的「白色精準本體框」與「灰色防觸擊外圍虛線框」，射線必須實質切過內層實線框才判定為 "HIT"。
* **📝 互動事件全自動紀錄 (Event Logger)**：系統全時段掃描畫面，並將「階段切換」、「語音觸發」、「兒童成功指向目標物」等關鍵事件，自動彙整並輸出為純文字紀錄檔。

---

## 📂 資料夾架構與說明

本專案採用**模組化設計 (Modular Architecture)**，將各核心功能解耦，以提升系統的穩定性與可維護性。

```text
📁 project_root/
│
├── 📁 model/           # 放置所有訓練好的 YOLO 模型權重檔 (.pt 檔)
│   ├── front_model.pt
│   ├── background_model.pt
│   ├── balloon_model.pt
│   ├── doll_model.pt
│   ├── toy_model.pt
│   ├── robot_point_model.pt
│   └── yolo11n-pose.pt # 人體姿態辨識基礎模型
│
├── 📁 modules/         # 核心演算法模組 (解耦設計)
│   ├── 📁 __pycache__/ # Python 編譯快取檔 (開發時自動生成)
│   ├── __init__.py     
│   ├── interaction.py  # 負責手勢判定、嚴格射線生成與 Ray-AABB 碰撞判定
│   ├── models_manager.py # 負責 YOLO 模型的資源排程、硬體加速 (CUDA/FP16) 與動態載入
│   ├── signboard.py    # 負責 EasyOCR 牌子數字辨識、暖機與 ROI 搜索追蹤
│   └── speech.py       # 負責 Whisper 語音關鍵字擷取、音訊處理與時間窗生成
│
├── 📁 output/          # 程式執行後的輸出結果
│   ├── event_record.txt        # 系統自動生成的互動事件紀錄表 (Event Log)
│   ├── output_result_final.mp4 # 繪製好骨架、射線與 UI 資訊的分析影片
│   └── speech_cache.json       # 語音辨識快取檔
│
├── 📁 video/           # 放置待分析的原始測試影片 (.mp4)
│
└── 📄 main.py          # 系統主程式 (協調各模組的 Orchestrator)

```

---

## 🚀 如何執行 (Getting Started)

### 1. 環境安裝

請確保你的電腦具備支援 CUDA 的 NVIDIA 顯示卡，以獲得最佳效能（內建 GPU 加速支援）。建議安裝對應版本的 PyTorch 與依賴套件（如 `ultralytics`, `mediapipe`, `easyocr`, `opencv-python`）。

### 2. 準備檔案

* 將你的原始測試影片放入 `video/` 資料夾（如 `10.mp4`）。
* 確保所有訓練好的 `.pt` 權重檔已正確放置於 `model/` 資料夾中。

### 3. 執行系統

在終端機中執行主程式：

```bash
python main.py

```

### 4. 執行流程與操作提示

1. **系統啟動與暖機**：系統會啟動聽覺大腦分析語音（初次執行會建立 Cache），同時針對 OCR 引擎進行硬體暖機，消除初次啟動延遲。
2. **框選搜索區域 (ROI)**：視窗彈出後，請用滑鼠框選牌子可能出現的「大範圍區域」（例如整個桌面），選定後按 `Enter` 鍵確認。
3. **自動分析迴圈**：影片開始播放，系統將全時段進行多模態行為分析。畫面左上角會即時顯示：
* `Time`：當下秒數。
* `Stage`：當前辨識到的階段。
* `Keyword Detected`：當下是否處於語音觸發時間窗內。
* `Child Pointing Hit`：兒童的射線是否精準擊中當前階段的目標物。


4. **手動控制**：
* 按 `q` 鍵可隨時提早結束分析。
* 按 `r` 鍵可手動將當前階段強制重置為 `0`。


5. **檢視成果**：程式結束後，請至 `output/` 資料夾查看最終分析影片，以及包含所有關鍵動作時間戳的 `event_record.txt`。

```