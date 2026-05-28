# 🤖 兒童注意力與手勢互動影音整合分析系統

**(Child Attention & Interaction Vision-Audio Analysis System v20)**

## 📝 專案簡介

本系統為一套針對兒童行為評估所開發的**多模態 (Multi-modal) AI 分析引擎**。系統整合了 **Whisper 語音辨識**與**多重電腦視覺模型 (YOLOv11-Pose, MediaPipe)**，能夠自動化化解析臨床測驗影片。

系統會先透過語音辨識精準抓取「關鍵字指令」並設定評估視窗，隨後無縫接軌進入視覺分析。視覺核心能自動追蹤測驗牌卡階段、劃分施測者與兒童的互動區域，並計算 3D 空間指向向量（包含人類與實體機器人），最終生成帶有視覺特效與計分數據的完整分析影片。

---

## ✨ 核心模組與技術亮點

### 🎙️ 第一階段：AI 語音觸發快取機制 (Voice Trigger & Caching)

* **Whisper Large-v3 整合**：利用 OpenAI Whisper 模型掃描影片音軌，精準辨識如「開始」、「321」、「準備囉」等指令關鍵字。
* **動態判定視窗**：偵測到關鍵字後，系統會自動在後續 3 秒內啟動「高靈敏判定視窗」，並將辨識結果快取為 `.txt` 逐字稿，大幅節省重複執行的運算時間。

### 🎴 視覺核心 1：牌卡動態追蹤與階段切換 (Dynamic Tracking)

* **兩段式降級搜索策略 (Two-Stage Fallback)**：系統會即時追蹤桌面上的測驗階段字卡（Template Matching）。若遭遇牌卡切換導致短暫丟失（Lost Patience > 10），系統會自動將追蹤框**重置回使用者最初框選的完美大小**，並將搜索網向外擴展至 250px，確保無縫接軌至下一階段。

### 🧑‍🤝‍🧑 視覺核心 2：楚河漢界身分與手勢意圖分析 (Identity & Intent Analysis)

* **雙重骨架協作**：結合 YOLOv11-Pose (宏觀身體定位) 與 MediaPipe Hands (微觀手指關節)，透過 `calculate_arm_link_score` 演算法，徹底解決手部交錯時的身分誤判問題。
* **物理防呆機制**：嚴格區分「遠距共同注意力 (指向)」與「近距把玩干擾 (觸摸)」。當手部特徵點落入目標物 Bounding Box 時，系統會強制觸發 `TOUCH_WARN` 忽略計分。

### 🤖 視覺核心 3：機器人平滑指向判定 (Robot Pointing Stabilization)

* 針對測驗「第八階段」的機器人互動，系統會自動切換至定製訓練的 YOLO 單點關鍵點模型 (`robot_point_model.pt`)。
* 導入 **SMA 滑動平均演算法 (Simple Moving Average Buffer)**，動態過濾 AI 座標預測的閃爍雜訊，使機器人生成的指向射線如實體雷射筆一般穩定精確。

---

## 📂 資料夾結構與檔案準備 (Project Structure)

請確保您的專案目錄符合以下結構，以便系統順利載入模型與素材：

```text
Project_Root/
│
├── main.py                  # 本系統主程式碼 (v20 整合版)
├── video/
│   └── 9.mp4                # 欲分析的原始測試影片 (需包含音軌)
├── sample/
│   ├── 1.jpg ... 8.jpg      # 階段辨識字卡樣板 (建議清晰無反光)
├── model/
│   ├── front_model.pt       # 前景物件辨識權重
│   ├── background_model.pt  # 背景物件辨識權重
│   ├── balloon_model.pt     # 氣球辨識權重
│   ├── bubble_model.pt      # 泡泡辨識權重
│   ├── toy_model.pt         # 玩具辨識權重
│   └── robot_point_model.pt # 🤖 機器人專屬指向辨識權重
└── output/                  # 系統自動生成的產出資料夾
    ├── output_result_v20.mp4       # 純視覺分析影片 (無聲)
    ├── transcript_with_events_v20.txt  # 語音辨識快取逐字稿
    └── output_with_audio_v20.mp4   # 🎉 最終影音縫合完成版

```

---

## 🚀 執行指南 (How to Run)

### 1. 安裝系統依賴 (Requirements)

請使用 Python 3.8 或以上版本，並安裝必要的運算與影音處理套件：

```bash
pip install numpy opencv-python mediapipe ultralytics pillow moviepy openai-whisper

```

*(註：首次執行時，系統會自動下載 Whisper 大型模型與 YOLOv11-Pose 預訓練模型，需保持網路連線。)*

### 2. 啟動系統與框選目標 (Execution & ROI Selection)

```bash
python main.py

```

程式啟動後，將依序執行：

* **階段一：** 若無快取，系統會先花費數分鐘解析整部影片的語音，並生成 `transcript.txt`。
* **階段二：** 語音解析完成後，會彈出視窗並暫停於影片第一幀。請使用滑鼠**精準框選桌上的「階段 1 字卡」**，完成後按下 `Enter` 或空白鍵。
* **階段三：** 系統進入全幀視覺分析主迴圈。

### 3. 查看最終結果 (Output)

分析完成後，系統會在終端機印出各階段的計分統計表，並透過 MoviePy 將原始聲音與帶有視覺化骨架、射線、判定文字的影片進行最終縫合，存入 `output` 資料夾中。