# 兒童注意力與手勢互動影音整合分析系統
**(Child Attention & Interaction Vision-Audio Analysis System)**

![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)
![YOLO](https://img.shields.io/badge/YOLO-v11-orange)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hands-green)
![Whisper](https://img.shields.io/badge/OpenAI-Whisper-black)

本專案為針對兒童行為評估（共同注意力 Joint Attention）所開發的**多模態 (Multi-modal) AI 影音分析引擎**。系統整合了 OpenAI Whisper 語音辨識與多重電腦視覺模型，能夠自動化解析臨床測驗影片，精準擷取受測者與機器人的空間互動意圖，並產出量化數據與視覺化影片。

---

## 核心技術與功能亮點

### 1. AI 語音觸發與快取機制 (Voice Trigger & Caching)
* 整合 **OpenAI Whisper Large-v3** 模型，精準辨識影片音軌中的關鍵字指令（如「開始」、「321」、「準備囉」）。
* **動態判定視窗**：偵測到關鍵字後，系統自動開啟 3 秒的高靈敏判定視窗。並具備 `.txt` 逐字稿快取功能，大幅降低重複測試時的運算成本。

### 2. 牌卡動態追蹤與防呆切換 (Dynamic Stage Tracking)
* 透過 OpenCV 樣板比對即時追蹤桌面測驗字卡，判斷測驗階段 (Stage 1~8)。
* **兩段式降級搜索策略**：當牌卡因人為切換短暫丟失時，系統會自動重置為初始完美比例 BBox，並將搜索網向外擴展至 250px，徹底解決追蹤卡死陷阱。

### 3. 楚河漢界身分與意圖過濾 (Identity & Intent Analysis)
* 結合 **YOLOv11-Pose** (宏觀骨架定位) 與 **MediaPipe Hands** (微觀手指關節)。
* 獨創「手臂距離分數演算法 (Arm Link Score)」，解決雙手交錯時的身分誤判。
* **物理防呆機制**：嚴格區分「遠距指向」與「近距把玩」。若手部特徵點落入目標物外框內，系統會強制標記為 `TOUCH_WARN` 並忽略計分，萃取最純粹的注意力數據。

### 4. 機器人平滑指向判定 (Robot Pointing Stabilization)
* 針對測驗第八階段，自動切換至定製訓練的 **YOLO 單關鍵點模型** (`robot_point_model.pt`)。
* 導入 **SMA 滑動平均演算法 (Simple Moving Average Buffer)**，動態過濾 AI 單幀預測的座標雜訊，使機器人射線如實體雷射筆般穩定精確。

---

## 資料夾結構 (Project Structure)

請確保在執行前，專案目錄包含以下必要檔案：

```text
Project_Root/
│
├── main.py                  # 系統主程式 (語音+視覺整合版)
├── environment.yml          # Anaconda 虛擬環境配置檔
├── ffmpeg.exe               # 影音處理與聲音縫合核心工具
├── video/
│   └── 9.mp4                # 欲分析的原始測試影片 (需含音軌)
├── sample/
│   ├── 1.jpg ... 8.jpg      # 各階段辨識字卡樣板圖
├── model/
│   ├── front_model.pt       # 前景物件辨識模型
│   ├── background_model.pt  # 背景物件辨識模型
│   ├── balloon_model.pt     # 氣球辨識模型
│   ├── bubble_model.pt      # 泡泡辨識模型
│   ├── toy_model.pt         # 玩具辨識模型
│   └── robot_point_model.pt # 機器人專屬指向模型
└── output/                  # 系統自動生成的產出目錄 (執行後產生)

```

---

## 環境安裝與執行指南 (Installation & Usage)

### 1. 建立虛擬環境與安裝套件

本專案提供完整的 environment.yml，可一鍵還原包含 Python 3.9、YOLO、MediaPipe 及 Whisper 等所有依賴的開發環境：

```bash
# 複製專案到本地
git clone (https://github.com/Aliulilun/Attention-Assessment-System.git)
# 複製專案到本地
cd Attention-Assessment-System

# 透過 environment.yml 一鍵建立並配置虛擬環境
conda env create -f environment.yml

# 啟動虛擬環境 (預設名稱為 mediapipe_py39，若有修改請依照 yml 內的 name 啟動)
conda activate mediapipe_py39

```

*(註：首次執行時，系統會自動自網路下載 Whisper 大型模型與 YOLOv11-Pose 預訓練模型，請保持網路連線。)*

### 2. 啟動系統

```bash
python main.py

```

### 3. 操作流程

1. **語音解析**：若為首次執行該影片，系統會先解析語音並建立快取檔案。
2. **框選基準點**：語音解析完畢後，畫面會暫停於第一幀。請使用滑鼠**精準框選桌上的「階段 1 字卡」**，框選完成後按下 `Enter` 鍵。
3. **自動分析**：系統將進入全幀視覺分析主迴圈。分析完成後，程式會透過 MoviePy 與 FFmpeg 自動將結果影像與原始音軌進行縫合。
4. **取得結果**：最終帶有視覺化骨架、射線與特效的完整影片將輸出於 `output/output_with_audio.mp4`。

---