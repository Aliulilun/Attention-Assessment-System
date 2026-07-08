# 多模態注意力評估系統
### Multi-modal AI Interactive Behavior Analysis System

> 一套結合電腦視覺、語音辨識與視線追蹤的兒童臨床**共同注意力 (Joint Attention)** 自動化評估系統。  
> 採用嚴謹的 OOP 模組化架構，具備高度容錯率與 GPU 加速支援。

---

## 核心功能亮點

**多模態時序控制**：語音觸發 → 視線追蹤 → 指向射線，三路訊號交叉驗證，自動記錄 T0 / TB / TH 三個關鍵行為時間點。

**階段適應性物件偵測**：系統依 OCR 讀取的階段牌（1~8）動態切換 YOLO 模型，在有限 VRAM 下壓榨最高推理效能（Lazy Loading）。

**雙重射線幾何碰撞**：指向射線（MediaPipe + YOLO-Pose Ray-AABB）搭配 3D 視線向量（ETH-XGaze 五階段管線），雙重確認兒童注意力目標。

**自動音軌縫合輸出**：分析完成後自動呼叫 FFmpeg，將標注影片與原始音軌進行無損對齊與縫合。

---

## 評估時間軸定義

| 符號 | 名稱 | 說明 |
|------|------|------|
| **T0** | 指示時間 | 系統聽到關鍵字，開啟黃金判定時間窗的瞬間 |
| **TB** | 看向物品 | T0 後，兒童首次注視目標物件的瞬間（`child_is_gazing_at` = True）|
| **TH** | 看回施測者 | TB 後，兒童視線移回施測者或機器人的瞬間（`TESTER_ZONE` / `robot_boxes`）|

---

## 專案目錄結構

```text
project_AI/
├── main.py                        # 總指揮官：主迴圈、時間軸控制、影片輸出（Stage 1~14）
├── config.yaml                    # 系統配置檔（路徑、關鍵字、閾值）
├── ffmpeg.exe                     # FFmpeg 執行檔（音軌縫合用）
│
├── hurry/                         # 快速批次模式（Stage 6~10，多影片一次跑完）
│   ├── main.py                    # 批次總控：先跑所有影片 Whisper，再批次跑視覺分析
│   └── output/                    # 批次輸出（含各影片的 _speech_xxx/ 快取子目錄）
│
├── modules/
│   ├── speech.py                  # 語音觸發器（Subprocess 隔離執行）
│   ├── speech_engine.py           # Whisper 語音辨識核心（抗幻覺過濾、繁簡容錯、音訊模板怪聲比對）
│   ├── signboard.py               # EasyOCR 牌子追蹤器（1~8 階段切換狀態機、換牌位置凍結）
│   ├── models_manager.py          # 動態 YOLO 模型管理員（Lazy Loading）
│   ├── interaction.py             # 互動引擎（YOLO-Pose + MediaPipe Ray-Casting）
│   ├── scoring_engine.py          # 計分引擎（時間窗事件解析）
│   └── gaze_estimation/
│       ├── gaze_pipeline.py       # 視線估計 5 階段流程整合
│       ├── stage1_face_detection.py   # Stage 1: 人臉偵測與特徵點
│       ├── stage2_head_pose.py        # Stage 2: SolvePnP 頭部歐拉角
│       ├── stage3_normalization.py    # Stage 3: 影像透視正規化
│       ├── stage4_gaze_network.py     # Stage 4: ResNet-50 視線卷積網路
│       ├── stage5_gaze_vector.py      # Stage 5: 3D 視線向量轉換
│       ├── camera_utils.py            # 相機內參矩陣與透視變換
│       ├── config_loader.py           # 全域配置封裝
│       ├── state_manager.py           # 視線有限狀態機 (FSM)
│       └── visualization.py           # 視線軌跡與頭部姿態立體渲染
│
├── model/                         # AI 模型權重（不上傳 Git，見下方說明）
│   ├── noise.wav                  # ⭐ 手機怪聲參考樣本（Stage 8 音訊模板，1~3 秒）
│   ├── front_model.pt             # YOLO：前方物件
│   ├── background_model.pt        # YOLO：背景物件
│   ├── balloon_model.pt           # YOLO：氣球
│   ├── doll_model.pt              # YOLO：玩偶
│   ├── toy_model.pt               # YOLO：玩具
│   ├── tablet_model.pt            # YOLO：平板
│   ├── robot_model.pt             # YOLO：機器人
│   ├── yolo11n-pose.pt            # YOLO：人體姿態骨架
│   └── gaze/
│       ├── epoch_24_ckpt.pth.tar  # ResNet-50 視線網路權重（~88MB）
│       ├── face_landmarker.task   # MediaPipe 臉部特徵點模型
│       ├── hand_landmarker.task   # MediaPipe 手部特徵點模型
│       ├── nano.pt                # YOLO 人臉追蹤核心
│       └── face_model_ethxgaze.txt # ETH-XGaze 3D 人臉幾何座標
│
├── video/                         # 輸入影片（不上傳 Git）
└── output/                        # 系統輸出（不上傳 Git）
    ├── output_result_final_*.mp4  # 標注影片（含原始音軌）
    ├── event_record_*.txt         # T0/TB/TH 事件時間戳記錄
    ├── transcript_with_events.txt # 語音逐字稿與觸發事件對照
    └── speech_cache.json          # Whisper 辨識結果快取
```

---

## 核心模組說明

### 語音辨識大腦 (`speech.py` / `speech_engine.py`)
- 使用 **OpenAI Whisper large-v3** 進行語音辨識
- 以 **Subprocess 獨立行程隔離**執行，防止 VRAM 崩潰
- 快取機制（`speech_cache.json`），第二次執行跳過辨識階段（< 0.1s）
- 支援繁簡體字容錯（「畫」/「画」、「這裡」/「这里」等）
- **Stage 8 怪聲偵測採雙軌架構**：優先使用 `model/noise.wav` 音訊模板（頻譜相似度 ≥ 0.65）；找不到音檔時退回 RMS / 頻譜特徵 fallback 模式
- Stage 8 時間軸起點由 **音訊 noise_events 全權決定**，已移除 Whisper 文字 fallback（防止 initial_prompt 幻覺中的「短促的聲音」誤觸發）
- `scoring_engine` 內建防早跳保護：Stage 8 只能從 Stage 7 推進，杜絕 Stage 1 直跳

### 牌子追蹤器 (`signboard.py`)
- **EasyOCR** 辨識 1~8 數字牌，驅動評估階段狀態機；每 2 幀執行一次 OCR 降低運算負擔
- **7 軌 OCR 策略**：Normal / CLAHE / Adaptive / Bold Erosion / Sharpen / Otsu / Inverted，多軌投票提升容錯率
- **Seven-Hunt Mode**（Stage 6 專屬）：當 `current_stage == 6` 且 7 軌仍未找到「7」時，以 `allowlist='7'` 強制 EasyOCR 只輸出「7」，並以 3x 放大圖（CLAHE / Otsu / Inverted 三變體）補強辨識——根治帶橫槓歐式「7」被誤讀為「1」的問題
- 動態追蹤結界 (TRACKING_PAD = 180px)，防止跟丟；Backup 全 ROI 掃描只做位置再錨，不影響升階邏輯
- 防抖投票機制（history deque + 升階保護），有效隔絕背景誤觸發
- **換牌位置凍結 (`_occlusion_freeze`)**：偵測到手部遮擋牌子（膚色像素比 > 閾值）時，KCF 追蹤器繼續內部運行但結果不寫入，掃描框釘在原位；手退走後 OCR 在原位找到新牌即自動解凍——根治換牌期間追蹤器跟手漂移導致後續辨識全錯的問題

### 動態模型管理員 (`models_manager.py`)
- 依評估階段自動切換 7 種 YOLO 模型
- **惰性載入 (Lazy Loading)**，避免 VRAM 一次爆炸
- 自動偵測 CUDA，啟用 `cudnn.benchmark` 加速
- 後期階段（11~14）支援雙模型並行偵測（`yolo_boxes` + `robot_boxes`）

### 互動判定引擎 (`interaction.py`)
- 結合 **YOLO11-Pose** 與 **MediaPipe Hands（Tasks API 影片追蹤模式，`num_hands=2`）**
- **身分識別防禦**：手臂關節連動分數 (Arm Link Score) 區分施測者與兒童；跨 Zone 手臂可視性閾值 (`CROSS_ZONE_LINK_TH = 1.0`) 阻絕跨人污染
- **手部幾何驗證 (`_validate_hand_geometry`)**：進入任何判定邏輯前先驗（1）食指尖 / 中指尖到手腕距離不超過幀寬 55%（`MAX_HAND_SPAN_RATIO = 0.55`，防跨人污染）；（2）食指尖到手腕最小延伸距離 ≥ 幀寬 3%（`MIN_FINGER_EXTEND_RATIO = 0.030`，防純手腕遮擋誤觸發）
- **射線投射 Ray-AABB**：從兒童食指發射長度 1500px 的虛擬射線，判定是否擊中目標框
- **SMA 平滑化**：Simple Moving Average 過濾幀間高頻抖動；臉部誤觸拒絕半徑 90px

### 視線估計模組 (`gaze_estimation/`)
- 完整實作 **ETH-XGaze 官方幾何規範**的五階段推理管線
- 輸出 Pitch / Yaw 角度與 2D / 3D 視線向量
- 有限狀態機 (FSM) 管理視線狀態轉換（`GazeFSMManager`）
- 視線-物體交集使用 Ray-Casting 幾何碰撞（`ray_intersects_box`）

---

## 環境需求

- **Python** 3.9+
- **CUDA** 11.8+（強烈建議；CPU 模式速度較慢）
- **FFmpeg**（放置於專案根目錄或加入系統 PATH）

### 核心依賴套件版本（驗證通過）

| 套件 | 版本 | 用途 |
|------|------|------|
| `torch` / `torchvision` | `2.7.1+cu118` | 深度學習底座、ResNet 視線網路、YOLO 推理 |
| `ultralytics` | `8.4.19` | YOLO 物件偵測 / Pose 骨架 |
| `openai-whisper` | `20250625` | 語音辨識 (ASR) |
| `mediapipe` | `0.10.35` | 手部 / 臉部特徵點偵測 |
| `easyocr` | `1.7.2` | 數字牌 OCR 辨識 |
| `opencv-python` | `4.13.0.92` | 影像處理、渲染 |
| `scipy` | `1.13.1` | 頭部姿態旋轉矩陣運算 |
| `numpy` | `2.0.2` | 矩陣幾何運算 |
| `pyyaml` | — | 配置檔讀取 |
| `moviepy` | — | 影片後製輔助 |

---

## 安裝與部署

### 1. 建立虛擬環境

```powershell
conda create -n multimodal_py39 python=3.9
conda activate multimodal_py39
```

### 2. 安裝 PyTorch（CUDA 版）

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. 安裝其他套件

```bash
pip install ultralytics mediapipe openai-whisper easyocr opencv-python scipy numpy pyyaml moviepy
```

### 4. 配置模型權重

> ⚠️ 模型檔案因體積較大，不包含於 Git 倉庫，請另行取得並放置於 `model/` 目錄。

- `epoch_24_ckpt.pth.tar`（~88MB）請從 **Releases** 頁面下載，**勿解壓**，直接放入 `model/gaze/`
- `face_model_ethxgaze.txt` 請以 **Raw 下載**取得純文字座標檔（約 1KB）

### 5. 放置怪聲參考音檔

> Stage 8（手機怪聲）採音訊模板比對，需提供一段 1~3 秒的手機警報聲樣本。

將任意格式（`.wav` / `.mp3` 等）的怪聲音檔命名為 `noise.wav`，放入：

```
model/noise.wav
```

缺少此檔案時，系統會自動退回純頻譜特徵偵測模式（準確率較低，且 Whisper 初始提示幻覺可能導致 Stage 8 誤觸發）。

> ⚠️ 若先前曾在沒有 `noise.wav` 的狀態下執行過，請刪除舊快取再重跑：
> - hurry 模式：`hurry/output/_speech_{影片名}/speech_cache.json`
> - main 模式：`output/speech_cache.json`

### 6. 執行系統

**單影片全階段分析（Stage 1~14）：**

```bash
python main.py
```

- 啟動後顯示 `video/` 目錄影片選單，輸入編號選擇影片
- 語音大腦首次執行會辨識音訊（結果快取後，下次 < 0.1s 跳過）
- 預覽視窗彈出後，請框選牌子可能出現的大範圍 ROI，按 **Enter** 確認
- 執行中按 **`q`** 安全結束並導出影片；按 **`r`** 手動重置階段為 0

**批次快速模式（Stage 6~10，多影片）：**

```bash
python hurry/main.py
```

- 自動掃描 `video/` 下所有影片
- **Step 0**：所有影片 Whisper 語音辨識先跑完（此時 VRAM 空著，GPU 全力辨識）
- **Step 1**：載入 YOLO / Gaze / EasyOCR 視覺模型，逐支批次分析
- 重複執行時快取命中，Whisper 步驟幾乎零耗時

---

## 輸出說明

| 檔案 | 說明 |
|------|------|
| `output/output_result_final_*.mp4` | 帶有 YOLO / 視線 / 射線標注的輸出影片（含原始音軌） |
| `output/event_record_*.txt` | T0 / TB / TH 事件時間戳記錄 |
| `output/transcript_with_events.txt` | 語音逐字稿與觸發事件對照表 |
| `output/speech_cache.json` | Whisper 辨識結果快取（避免重複辨識） |

---

## 編碼與安全守則

- 所有檔案讀寫均使用 `encoding='utf-8'`，確保 Windows 中文環境不閃退
- 影片、模型、輸出等大型檔案已加入 `.gitignore`
- `stage2_head_pose.py` 與 `stage3_normalization.py` 已完成底層 UTF-8 升級，阻絕 Windows 預設編碼崩潰

---

## 技術棧

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%2011.8-orange)
![YOLO](https://img.shields.io/badge/Ultralytics-YOLO11-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hands%20%2B%20Pose-red)
![Whisper](https://img.shields.io/badge/Whisper-large--v3-lightgrey)
![EasyOCR](https://img.shields.io/badge/EasyOCR-Stage%201~8-yellow)
![ETH-XGaze](https://img.shields.io/badge/ETH--XGaze-5%20Stage%20Pipeline-blueviolet)
