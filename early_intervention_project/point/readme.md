
---

# 🌟 Ultimate Multi-Modal Behavior Analysis System

**多模態互動行為分析系統 (YOLO + EasyOCR + MediaPipe + Whisper)**

這是一個高度整合的電腦視覺與語音分析系統，專為「階段式互動測驗」或「人類行為分析」所設計。本系統能同時處理影片的**影像**與**音訊**，透過 OCR 動態切換系統階段，並根據當前階段動態載入對應的 YOLO 物件偵測模型。同時，系統結合了 MediaPipe 的手部與姿態追蹤，精準計算出施測者 (Tester) 與受測兒童 (Child) 的「指向 (Pointing)」與「觸碰 (Touching)」互動行為。

## ✨ 核心模組與特色

### 🗣️ 1. 智慧語音觸發系統 (Whisper AI)

* **關鍵字偵測**：使用 OpenAI Whisper (large-v3) 精準辨識影片對話，抓取「開始」、「321」、「你看」等關鍵字。
* **專屬判定視窗**：偵測到關鍵字後，自動開啟黃金 N 秒的「專屬判定視窗」，提升互動計分的精準度。
* **語音快取機制 (Cache)**：首次執行後會自動產生逐字稿與事件紀錄表 (`transcript_with_events.txt`)，下次執行瞬間載入，大幅節省開發與測試時間。

### 🏷️ 2. 動態階段推進機制 (EasyOCR)

* **抗干擾 OCR 追蹤**：運用 EasyOCR 實時追蹤畫面中的數字牌 (1~8)，自動推進測驗階段（如：準備中、第一部分...第八部分）。
* **強效防呆防護**：內建「尺寸異常檢測（防禦 B）」與「耐心值機制 (Lost Patience)」，當綠框咬死背景雜訊時會自動重置並擴大搜索範圍，徹底解決數字誤判與暴衝問題。

### 👁️ 3. 階段適應性物件偵測 (YOLO Multi-Models)

* 系統會根據 OCR 讀取到的「當前階段」，**動態切換**對應的 YOLO 模型進行預測，極大化運算資源：
* 階段 1~2：前方物件 (`front_model.pt`)
* 階段 3~4：背景物件 (`background_model.pt`)
* 階段 5：氣球 (`balloon_model.pt`)
* 階段 6：泡泡 (`bubble_model.pt`) (使用特殊 Beam 射線演算法)
* 階段 7：玩具 (`toy_model.pt`) (內建人臉排除機制，避免將人臉誤判為玩具)
* 階段 8：機器人指向 (`robot_point_model.pt`)



### 👉 4. 高階互動行為判定 (MediaPipe Pose & Hands)

* **動態身分識別**：不依賴死板的左右半邊分割，而是透過「手臂關節連動分數 (Arm Link Score)」與「YOLO 姿態追蹤 ID」，精準區分這隻手是屬於 Tester 還是 Child。
* **指向判定 (Ray-Casting)**：從食指發射虛擬射線，並計算與 YOLO 物件邊界框的交集，完美捕捉「指物」動作。
* **觸碰判定 (Touching)**：嚴謹的碰撞偵測，判定手部是否真實碰觸到目標物。

## 🛠️ 環境設定與依賴套件

請確保您的環境為 Python 3.8+，並安裝以下套件（建議使用 GPU 環境以獲得最佳效能）：

```bash
pip install numpy opencv-python ultralytics mediapipe easyocr openai-whisper Pillow moviepy

```

*(請確保系統已安裝 FFmpeg 以供 Whisper 與 MoviePy 正常處理音訊)*

## 📂 專案檔案架構

執行程式前，請確保目錄結構及模型檔案齊全：

```text
📁 專案根目錄/
├── 📄 main.py                   # 本系統主程式碼
├── 📁 video/                
│   └── 🎞️ 10.mp4                # 輸入影片
├── 📁 model/                    # YOLO 模型放置區
│   ├── front_model.pt
│   ├── background_model.pt
│   ├── balloon_model.pt
│   ├── bubble_model.pt
│   ├── toy_model.pt
│   ├── robot_point_model.pt
│   └── yolo11n-pose.pt          # YOLO 官方姿態模型
└── 📁 output/                   # 自動產生：存放輸出影片與逐字稿

```

## 🚀 使用教學與運行流程

1. **參數微調 (Optional)**：
您可以進入程式碼 `★★★ 第二區：核心演算法參數微調 ★★★` 修改各項閾值，包含 YOLO 信心度 (`CONF_*`)、語音回應視窗秒數 (`RESPONSE_WINDOW_SEC`) 等。
2. **執行程式**：
```bash
python main.py

```


3. **流程步驟**：
* **[階段一] 語音解析**：程式會先讀取快取；若無快取，將啟動 Whisper 進行音訊分析。
* **[階段二] 框選牌子**：畫面彈出後，請框選「數字牌 (OCR)」所在的初始結界範圍（例如：桌面角落），按 `Enter` 或 `Space` 確認。
* **[階段三] 即時渲染**：系統開始全幀運算，並即時顯示骨架、射線、物件框與階段資訊。


4. **結束與匯出**：
* 影片處理完畢或按下 `Esc`/`Q` 鍵後，系統會於終端機印出**各階段的得分統計表**。
* 系統將自動使用 MoviePy 把原始音訊縫合回處理好的視覺影片中，輸出至 `output/output_with_audio.mp4`。



## 📊 數據輸出與分析

程式執行完畢後，除了視覺化影片外，您會獲得：

1. **`transcript_with_events.txt`**：完整的語音逐字稿，並會標記所有觸發特殊判定的精確秒數。
2. **Terminal 得分報表**：自動統計 Child 在各個階段中「有效指向目標物件」的總次數，方便後續量化分析與研究。
