
---

```markdown
# 多模態 AI 互動行為分析與視線追蹤系統
### (Multi-Modal AI Interaction Analysis & 3D Gaze Estimation System)

本系統是一款針對兒童發展與互動行為設計的高階多模態 AI 分析平台。系統採用**「聽覺語音先行、時段場景Segment、多物件動態適應偵測、肢體與視線 3D 向量幾何驗證」**的架構，全時段動態捕捉並紀錄受測者的意圖與專注度指標，最後自動進行音軌無損縫合渲染輸出。

---

## 🚀 核心功能與技術亮點

1. **多模態時序控制 (Cross-Modal Validation)**
   - **聽覺腦 (Speech Trigger)**：利用 OpenAI Whisper 進行高精準度語音辨識與客製化關鍵字過濾，建立動態「觸發時間窗（Trigger Windows）」，同時內建高頻雜訊/怪聲代償機制，提供全時段的聽覺容錯控制。
   - **視覺腦 (Stage Tracker)**：利用 EasyOCR 動態追蹤場景中的實體階段牌數字（1~8），實時推進評估狀態機，當發生人為忘記換牌時，聽覺代償機制會無縫接管自動推至指定階段。

2. **階段適應性物件偵測 (Adaptive Multi-YOLO Managers)**
   - 系統依據 OCR 推進的實驗階段，在幕後動態切換、載入專屬優化的 YOLO 偵測權重（如前方物件、背景物件、氣球、玩偶、玩具及機器人發射點等），在有限的顯存資源下將運算效能壓榨至極致。

3. **雙重空間射線幾何碰撞偵測 (Dual Ray-Casting & AABB Check)**
   - **指向追蹤**：整合 MediaPipe Hands 與 YOLOv11-Pose 骨架模型，動態識別受測兒童與施測人員的手臂，計算食指尖至腕部的 2D/2.5D 指向射線，並導入 **SMA (簡單移動平均) 時序平滑演算法**，徹底消滅 AI 幀間閃爍雜訊。
   - **視線估計 (3D Gaze Network)**：完整實現 **ETH-XGaze 官方學術幾何規範**的五階段視線推理管線（人臉偵測 ➔ 頭部姿態 ➔ 影像正規化 ➔ ResNet-50 視線網絡 ➔ 3D 視線向量轉換），精準計算受測者與物件的 3D 透視相交矩陣。

4. **FFmpeg 自動影音縫合處理**
   - 視覺渲染完畢後，底層自動調用封裝的 FFmpeg 引擎，將處理後的無聲高清畫面與原始影片的音軌進行無損對齊與二進位縫合，免去手動後製流程。

---

## 📂 專案檔案結構 (Repository Architecture)

```text
C:\project\
│
├── main.py                     # 🌟 主系統核心 (整合聽覺、視覺、視線與空間幾何碰撞)
├── ffmpeg.exe                  # 🎵 FFmpeg 執行檔 (用於最終音軌縫合)
│
├── video/                      # 🎥 輸入影片存放區
│   └── 10.mp4                  # 實驗評估測試影片
│
├── output/                     # 📁 分析結果導出區
│   ├── output_result_final7.mp4   # 最終縫合音軌後的完整視覺化行為分析影片
│   ├── event_record7.txt          # 結構化行為事件紀錄表 (包含精確動作時間戳)
│   ├── transcript_with_events.txt # Whisper 語意逐字稿與怪聲事件診斷報告
│   └── speech_cache.json          # 聽覺快取檔 (二次執行時 0.1 秒閃電跳過)
│
├── model/                      # 🧠 AI 核心權重存放區
│   ├── front_model.pt          # YOLO 權重：前方物件
│   ├── background_model.pt     # YOLO 權重：背景物件
│   ├── balloon_model.pt        # YOLO 權重：氣球
│   ├── doll_model.pt           # YOLO 權重：玩偶
│   ├── toy_model.pt            # YOLO 權重：玩具
│   ├── robot_point_model.pt    # YOLO 權重：機器人定位發射點
│   ├── yolo11n-pose.pt         # YOLO 權重：人體姿態骨架模型 (指向起點)
│   │
│   └── gaze/                   # 👁️ 3D 視線追蹤專用模型與空間參數
│       ├── epoch_24_ckpt.pth.tar    # Stage 4: ResNet-50 視線卷積網絡 (約 88MB)
│       ├── nano.pt                  # Stage 1: YOLO 人臉追蹤核心 (約 6MB)
│       ├── hand_landmarker.task     # 互動定位：MediaPipe 官方手部特徵點模型
│       ├── face_landmarker.task     # Stage 1: MediaPipe 官方臉部特徵點網格模型
│       └── face_model_ethxgaze.txt  # Stage 2 & 3: ETH 官方 3D 臉部幾何標準座標檔
│
└── modules/                    # 🧩 自建核心功能組件模組
    ├── __init__.py
    ├── speech.py               # 聽覺組件：Whisper 語音事件窗標定與突發雜訊過濾
    ├── speech_engine.py        # 聽覺組件：底層音訊解碼驅動
    ├── signboard.py            # 視覺狀態：EasyOCR 牌面即時追蹤與邊界防禦機制
    ├── models_manager.py       # 物件偵測：多 YOLO 模型權重動態加載管理器
    ├── interaction.py          # 互動運算：2D Ray-Casting 射線幾何相交與 SMA 防抖
    │
    └── gaze_estimation/        # 👁️ 核心組件：視線估計五階段學術管線
        ├── __init__.py         
        ├── camera_utils.py     # 空間投影：相機虛擬內參矩陣與透視變換
        ├── config_loader.py    # 全域配置：封裝模型參數與硬件線路調配
        ├── gaze_pipeline.py    # 管線核心：整合並依序分發 5 個 Stage 運算流
        ├── stage1_face_detection.py   # Stage 1: Face Bounding Box & Landmarks
        ├── stage2_head_pose.py        # Stage 2: SolvePnP 算頭部歐拉角 (內建 UTF-8 防撞防護)
        ├── stage3_normalization.py    # Stage 3: 圖像透視歸一化 (內建 UTF-8 防撞防護)
        ├── stage4_gaze_network.py     # Stage 4: ResNet 視線深度特徵提取
        ├── stage5_gaze_vector.py      # Stage 5: 3D 向量射線轉換
        └── visualization.py           # 渲染工具：3D 視線軌跡與頭部姿態立體軸線繪製

```

---

## 🛠️ 開發環境與核心依賴套件 (Dependencies)

系統推薦運行於 **Python 3.9** 虛擬環境中。本專案目前實體虛擬環境驗證通過之核心套件版本清單如下（已篩選出關鍵 AI/電腦視覺套件）：

| 核心套件名稱 (Package) | 驗證版本 (Version) | 用途說明 (Description) |
| --- | --- | --- |
| `torch` / `torchvision` | `2.7.1+cu118` / `0.22.1+cu118` | 深度學習核心底座，驅動 ResNet 視線網絡與 YOLO 推理 |
| `ultralytics` | `8.4.19` | 負責動態驅動與多物件偵測、Pose 骨架模型加載 |
| `openai-whisper` | `20250625` | 負責抽取音軌進行大型 ASR 語音逐字稿與觸發窗分析 |
| `mediapipe` | `0.10.35` | 提供底層人臉 468 點特徵定位及雙手關節網格捕捉 |
| `easyocr` | `1.7.2` | 負責裁剪區域內的實驗數字卡片實時 OCR 辨識 |
| `opencv-python` | `4.13.0.92` | 影像矩陣處理、WarpPerspective 正規化變換、UI 面板渲染 |
| `scipy` | `1.13.1` | 負責 Stage 2 頭部姿態分解時的 Rotation 矩陣變換與旋轉運算 |
| `numpy` | `2.0.2` | 提供高維度矩陣幾何空間碰撞與 AABB 邊界運算 |

---

## 📥 安裝與執行部署指南

### 1. 複製專案與環境初始化

打開終端機，執行以下指令建立並啟用專案虛擬環境：

```powershell
# 切換到主目錄
cd C:\project

# 啟用您配置好的虛擬環境 (以 Anaconda 為例)
conda activate mediapipe_py39

```

### 2. 🚨 關鍵大權重檔與文字資產配置 (極重要防當機提示)

由於 GitHub 檔案容量限制，以下核心模型與資料夾元件必須在首次執行前完成手工歸位：

* **ResNet-50 核心視線模型**：請至 GitHub 的 **Releases** 頁面下載 `epoch_24_ckpt.pth.tar` 原檔（大小約 88MB）。**下載後請勿用解壓軟體解開**，直接將原檔移入 `model/gaze/` 底下。
* **3D 人臉幾何座標檔**：請確保 `model/gaze/face_model_ethxgaze.txt` 是透過專用 Raw 按鈕下載的純數字座標（約 1KB）。*（本系統已對 `stage2` 和 `stage3` 的原始碼進行了底層 `encoding='utf-8'` 升級，完美阻絕了 Windows 預設中文編碼引發的無聲閃退崩潰）*。

### 3. 🚀 執行主系統分析

在專案根目錄下直接執行：

```powershell
python main.py

```

* **畫面互動說明**：
* 啟動後，聽覺大腦會先處理音訊（完成後會緩存，下次執行僅需 0.1 秒）。
* 當預覽視窗彈出時，請用滑鼠框選牌子可能出現的大範圍區域（ROI），選定後按 **Enter** 鍵確認，即可看到全時段多模態行為追蹤大面板。
* 執行過程中，按 **`q`** 鍵可隨時提早安全結束並導出影片；按 **`r`** 鍵可手動將當前階段強制重置為 0。



```