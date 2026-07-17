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
integrate_v5/
├── main.py                        # 總指揮官：主迴圈、時間軸控制、影片輸出（Stage 1~14）
├── config.yaml                    # 系統配置檔（路徑、關鍵字、閾值）
├── ffmpeg.exe                     # FFmpeg 執行檔（音軌縫合用）
│
├── hurry/                         # 快速批次模式（Stage 1~10，多影片一次跑完）
│   ├── main.py                    # 批次總控：先跑所有影片 Whisper，再批次跑視覺分析
│   └── output/                    # 批次輸出（含各影片的 _speech_xxx/ 快取子目錄）
│
├── modules/
│   ├── speech.py                  # 語音觸發器（Subprocess 隔離 + noise 時間窗管理）
│   ├── speech_engine.py           # Whisper 語音辨識核心（抗幻覺過濾、繁簡容錯、音訊模板怪聲比對）
│   ├── signboard.py               # EasyOCR 牌子追蹤器（1~8 階段切換狀態機、換牌位置凍結）
│   ├── models_manager.py          # 動態 YOLO 模型管理員（Lazy Loading）
│   ├── interaction.py             # 互動引擎（YOLO-Pose + MediaPipe Ray-Casting）
│   ├── scoring_engine.py          # 計分引擎（絕對時間軸 + 三類事件快取載入）
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
│   ├── noise.wav                  # ⭐ 手機怪聲參考樣本（Stage 7→8 音訊模板，1~3 秒）
│   ├── hand_landmarker.task       # MediaPipe 手部特徵點模型（主程式用）
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
│       ├── hand_landmarker.task   # MediaPipe 手部特徵點模型（視線模組用）
│       ├── nano.pt                # YOLO 人臉追蹤核心
│       └── face_model_ethxgaze.txt # ETH-XGaze 3D 人臉幾何座標
│
├── video/                         # 輸入影片（不上傳 Git）
└── output/                        # 系統輸出（不上傳 Git）
    ├── output_result_final_*.mp4  # 標注影片（含原始音軌）
    ├── event_record_*.txt         # T0/TB/TH 事件時間戳記錄
    └── speech_cache.json          # Whisper 辨識結果快取
```

---

## 核心模組說明

### 語音辨識大腦 (`speech.py` / `speech_engine.py`)

- 使用 **OpenAI Whisper large-v3** 進行語音辨識
- 以 **Subprocess 獨立行程隔離**執行，防止 VRAM 崩潰
- 快取機制（`speech_cache.json`），第二次執行跳過辨識階段（< 0.1s）
- 支援繁簡體字容錯（「畫」/「画」、「這裡」/「这里」等）

**Stage 7→8 怪聲偵測邏輯（重要）：**

- 提供 `model/noise.wav` 時：`speech_engine` 以頻譜模板相似度比對全片，找出最佳命中時段並寫入快取的 `noise_events`，同時**完全停用 RMS / 頻譜特徵 fallback**，防止假陽性
- `speech.py` 從快取的 `noise_events[].trigger_window` 讀入怪聲時間窗，填充 `self.noise_trigger_windows`
- 主迴圈透過 `speech.is_in_noise_window(current_time_sec)` 判斷是否切換 Stage 7 → 8
- 關鍵字清單中**不再包含**「怪聲」/「嗶」/「逼」/「[聲音]」（Whisper 音效符號非口說詞語，留著會造成誤觸發）
- 缺少 `noise.wav` 時，系統自動退回純頻譜特徵偵測模式（準確率較低）

> ⚠️ 若先前曾在沒有 `noise.wav` 的狀態下執行過，請刪除舊快取再重跑：
> - hurry 模式：`hurry/output/_speech_{影片名}/speech_cache.json`
> - main 模式：`output/speech_cache.json`

### 牌子追蹤器 (`signboard.py`)

- **EasyOCR** 辨識 1~8 數字牌，驅動評估階段狀態機；每 2 幀執行一次 OCR 降低運算負擔（Stage 7 降到每 5 幀一次）
- **3 軌 OCR 策略**：Normal（原始灰階）/ CLAHE（對比增強）/ Adaptive（自適應二值化），跨軌 NMS 去重取信心最高者；若軌道 1 已達高信心閾值（≥0.75）則直接採用，跳過軌道 2、3 節省運算
- **Stage 6→7 模板比對補強**：數字 6、7 外形相似易混淆，改用預先載入的旋轉/縮放模板（`model/signboardphoto/7*.png`）做 Template Matching，需連續 3 幀「模板信心 > OCR 讀出 6 的信心」才允許升至 Stage 7
- 動態追蹤結界 (TRACKING_PAD = 180px)，防止跟丟；Backup 全 ROI 掃描只做位置再錨，不影響升階邏輯
- 防抖投票機制（history deque + 升階保護），有效隔絕背景誤觸發
- **遮擋凍結（外觀直方圖比對）**：牌子首次被 OCR 確認時記錄外觀直方圖快照，之後每幀比對 KCF 追蹤框內容的直方圖相關係數，連續低於閾值（0.55）達 3 幀視為手部遮擋，凍結掃描框位置；手退走後 OCR 在原位找到新牌即自動解凍
- **應急框機制**：連續 25 幀完全偵測不到數字時，切換到應急框模式（掃描範圍重設回 ROI 中心 1.5 倍大小強制全域掃描），連續 3 次 OCR 成功才退出
- 批次模式使用 `initialize_roi_auto(first_frame)`（自動框選右下 1/4，⚠️ 若牌子實際不在此範圍或該區域另有印刷數字會誤鎖）；單片模式使用 `initialize_roi(first_frame)`（互動框選）

### 動態模型管理員 (`models_manager.py`)

- 依評估階段自動切換 7 種 YOLO 模型
- **惰性載入 (Lazy Loading)**，避免 VRAM 一次爆炸
- 自動偵測 CUDA，啟用 `cudnn.benchmark` 加速
- Stage 9~14（機器人登場後）支援雙模型並行偵測（`target_boxes` + `robot_boxes`），供 TH 判定「看回機器人」使用
- **Stage 5（氣球）三道防誤判過濾**：長寬比 >2.2、框內膚色比例 >35%、彩度不足（三者皆為排除手臂/衣物被誤判成氣球的防線，最後一項門檻可在 `config.yaml` 的 `stage5_min_colorful_ratio` 調整或停用）

### 互動判定引擎 (`interaction.py`)

- 結合 **YOLO11-Pose**（`conf=0.3`，適應施測者背對或兒童被桌子遮擋）與 **MediaPipe Hands（Tasks API 影片追蹤模式，`num_hands=2`）**
- **`hand_model_path` 參數**：由主程式明確傳入 `model/hand_landmarker.task` 的絕對路徑，避免 Windows junction/symlink 導致 `__file__` 解析到錯誤真實路徑，引發 `FileNotFoundError`
- **跨影片單調時間戳**：批次模式下，`_ts_base` 機制確保 MediaPipe VIDEO 模式的時間戳在跨影片時單調遞增，防止違反 API 要求導致崩潰
- **骨架與射線始終執行**：`analyze_interaction()` 不設 `len(yolo_boxes) > 0` 前置條件，讓骨架標注與手部偵測在任何幀都可見；`yolo_boxes` 為空時僅跳過射線碰撞判定
- **身分識別防禦**：手臂關節連動分數 (Arm Link Score) 區分施測者與兒童，左右手分數取最小值，對 MediaPipe handedness 翻轉免疫
- **食指實體驗證**：沿食指 PIP→TIP 連線做皮膚色比對，防止 MediaPipe 在手部鬆握/遮擋時「幻想」出伸直食指造成假指向
- **射線投射 Ray-AABB**：從兒童食指發射長度 1500px 的虛擬射線，判定是否擊中目標框
- **SMA 平滑化**：Simple Moving Average 過濾幀間高頻抖動；臉部誤觸拒絕半徑 90px

### 計分引擎 (`scoring_engine.py`)

- 從 `speech_cache.json` 讀取三類事件，重建完整評估時間軸：
  - `load_keyword_trigger_windows_from_cache()`：讀取關鍵字觸發窗（T0）
  - `load_speech_events_from_cache()`：讀取所有語音段落事件
  - `load_noise_events_from_cache()`：讀取怪聲事件（Stage 8）
- `_build_absolute_timeline()` 依快取事件自動建立各階段起始時間，無需手動設定時間軸
- **Stage 1-4 T0 代償與暫存重播**：Whisper 幻覺或漏轉錄可能導致整關聽不到關鍵字，系統會在等待逾時（Stage 1 給 15 秒、Stage 2-4 給 4 秒）後以代償時間建立 T0，並把 T0 確定前暫存的逐幀資料重播進正式紀錄；關鍵字若之後補到會自動把 T0 升級為更精確的關鍵字時間。詳細規則見 `系統架構說明.md` 第 8 節。
- `update_frame()` 介面保持不變，與 main.py / hurry/main.py 完全相容
- 輸出屬性：`trigger_event_records`、`stage_gazing_counts`、`total_score`、`total_gazing_events`、`event_logs`

### 視線估計模組 (`gaze_estimation/`)

- 完整實作 **ETH-XGaze 官方幾何規範**的五階段推理管線
- 輸出 Pitch / Yaw 角度與 2D / 3D 視線向量
- 有限狀態機 (FSM) 管理視線狀態轉換（`GazeFSMManager`）
- 視線-物體交集使用 Ray-Casting 幾何碰撞（`ray_intersects_box`）

---

## 環境需求

- **Python** 3.14（2026-07-16 已從 3.11.15 驗證升級，完整過程與升級前套件版本見專案根目錄 `環境建置與版本更新說明.md`；沒有 CUDA 的電腦可改裝 CPU 版 PyTorch，程式碼不需改動，見該文件說明）
- **CUDA** 13.x（強烈建議；CPU 模式速度較慢）
- **FFmpeg**（放置於專案根目錄或加入系統 PATH）

### 核心依賴套件版本（2026-07-16 驗證通過，Python 3.14.6 環境）

| 套件 | 版本 | 用途 |
|------|------|------|
| `torch` / `torchvision` | `2.13.0+cu130` | 深度學習底座、ResNet 視線網路、YOLO 推理 |
| `ultralytics` | `8.4.96` | YOLO 物件偵測 / Pose 骨架 |
| `openai-whisper` | `20250625` | 語音辨識 (ASR) |
| `mediapipe` | `0.10.35` | 手部 / 臉部特徵點偵測 |
| `easyocr` | `1.7.2` | 數字牌 OCR 辨識 |
| `opencv-python` / `opencv-contrib-python` / `opencv-python-headless` | `4.13.0.92`（三者版本須完全一致，且須在其他套件裝完後**最後手動釘選**，否則會被自動升級到 5.0.x 導致 `signboard.py` 的 `TrackerKCF` 相容性風險）| 影像處理、渲染、KCF 追蹤 |
| `scipy` | `1.18.0` | 頭部姿態旋轉矩陣運算 |
| `numpy` | `2.4.6` | 矩陣幾何運算（⚠️ 不可升到 2.5+，`numba`／Whisper 依賴要求 `numpy<2.5`）|
| `pyyaml` | `6.0.3` | 配置檔讀取 |

> 完整套件清單（含所有間接依賴）與逐步安裝順序，請見專案根目錄的 `環境建置與版本更新說明.md`。

---

## 安裝與部署

### 1. 建立虛擬環境

```powershell
conda create -n attention_integrate python=3.14
conda activate attention_integrate
```

### 2. 安裝 PyTorch（CUDA 版，務必第一個裝；沒有 GPU 就把 index-url 換成 `.../cpu`）

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

> 注意：這裡不需要裝 `torchaudio`——專案的音訊處理走 FFmpeg 子行程與 Whisper 自帶的音訊載入，沒有任何模組匯入 `torchaudio`。

### 3. 安裝其他套件（`opencv-*` 務必留到最後一步單獨釘選版本，見下方警告）

```bash
pip install ultralytics
pip install openai-whisper easyocr mediapipe
pip install pyyaml pandas openpyxl pytest pip-tools py-spy
pip install opencv-python==4.13.0.92 opencv-contrib-python==4.13.0.92 opencv-python-headless==4.13.0.92
```

⚠️ `ultralytics`／`easyocr`／`mediapipe` 安裝時都會自動附帶裝上 OpenCV 的最新版（5.0.x），必須在所有套件裝完後，最後單獨執行上面第 4 行把三個 OpenCV 套件重新釘選回 `4.13.0.92`，否則 `signboard.py` 用到的 `cv2.TrackerKCF_create` 有相容性風險。

### 4. 配置模型權重

> ⚠️ 模型檔案因體積較大，不包含於 Git 倉庫，請另行取得並放置於 `model/` 目錄。

- `epoch_24_ckpt.pth.tar`（~88MB）請從 **Releases** 頁面下載，**勿解壓**，直接放入 `model/gaze/`
- `face_model_ethxgaze.txt` 請以 **Raw 下載**取得純文字座標檔（約 1KB）
- `hand_landmarker.task` 需放在 `model/` 根目錄（主程式用）與 `model/gaze/`（視線模組用）各一份

### 5. 放置怪聲參考音檔

Stage 8（手機怪聲觸發 Stage 7→8）採音訊模板比對，需提供怪聲樣本：

```
model/noise.wav        ← 必須放在此確切路徑
```

- 格式：任意（`.wav` / `.mp3` 均可，系統內部轉換）
- 長度：建議 1~3 秒的手機警報聲片段
- 提供後會**完全停用** RMS / 頻譜 fallback，確保不會有假陽性

缺少此檔案時，系統退回純頻譜特徵偵測（準確率較低，建議一定要提供）。

### 6. 執行系統

**單影片全階段分析（Stage 1~14）：**

```bash
python main.py
```

- 語音大腦首次執行會辨識音訊（結果快取後，下次 < 0.1s 跳過）
- 預覽視窗彈出後，請框選牌子可能出現的大範圍 ROI，按 **Enter** 確認
- 執行中按 **`q`** 安全結束並導出影片；按 **`r`** 手動重置階段為 0

**批次快速模式（Stage 1~10，多影片）：**

```bash
python hurry/main.py
```

- 自動掃描 `hurry/video/` 下所有影片
- **Step 0**：所有影片 Whisper 語音辨識先跑完（VRAM 空著，GPU 全力辨識）
- **Step 1**：載入 YOLO / Gaze / EasyOCR 視覺模型，逐支批次分析
- 每支影片使用獨立快取子目錄 `output/_speech_{影片名}/`，避免多影片共用快取互相污染
- 重複執行時快取命中，Whisper 步驟幾乎零耗時

---

## 輸出說明

| 檔案 | 說明 |
|------|------|
| `output/output_result_final_*.mp4` | 帶有 YOLO / 視線 / 射線標注的輸出影片（含原始音軌） |
| `output/event_record_*.txt` | T0 / TB / TH 事件時間戳記錄 |
| `output/speech_cache.json` | Whisper 辨識結果快取（含 `noise_events`、`trigger_windows`、`segment_records`） |

---

## 常見問題排查

**骨架或指向射線不出現：** 確認 `interaction.analyze_interaction()` 的呼叫沒有被 `if len(yolo_boxes) > 0:` 包住（應無條件呼叫）；YOLO-Pose `conf` 若設太高（如 0.5）在施測者背對或兒童被桌遮擋時會整人消失，應使用 `conf=0.3`。

**MediaPipe FileNotFoundError：** 確認 `InteractionEngine` 建立時有傳入 `hand_model_path=os.path.join(MODEL_DIR, 'hand_landmarker.task')`。若路徑為 Windows junction（例如 `C:\project` → `C:\Users\wayne\Desktop\project\project_v3`），`__file__` 會解析到真實路徑，導致相對路徑計算錯誤。

**Stage 7→8 不切換，或 `[聲音]` 出現在觸發視窗：** 確認關鍵字清單中已移除 `"怪聲"`、`"嗶"`、`"逼"`、`"[聲音]"`；確認 `model/noise.wav` 存在；如有舊快取請刪除後重跑。

**批次模式多影片時間軸亂掉：** 確認 `interaction.reset_tracking()` 在每支影片開頭有被呼叫（重設 `_ts_base` 與 MediaPipe 時間戳偏移）。

**EasyOCR CUDA OOM：** hurry/main.py 已加入 OOM 捕捉，自動清 VRAM 後繼續。若頻繁出現，可嘗試調高 `OCR_SKIP`（減少 OCR 頻率）或降低影片解析度。

---

## 編碼與安全守則

- 所有檔案讀寫均使用 `encoding='utf-8'`，確保 Windows 中文環境不閃退
- 影片、模型、輸出等大型檔案已加入 `.gitignore`
- `stage2_head_pose.py` 與 `stage3_normalization.py` 已完成底層 UTF-8 升級

---

## 技術棧

![Python](https://img.shields.io/badge/Python-3.14-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%2013.0-orange)
![YOLO](https://img.shields.io/badge/Ultralytics-YOLO11-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hands%20%2B%20Pose-red)
![Whisper](https://img.shields.io/badge/Whisper-large--v3-lightgrey)
![EasyOCR](https://img.shields.io/badge/EasyOCR-Stage%201~8-yellow)
![ETH-XGaze](https://img.shields.io/badge/ETH--XGaze-5%20Stage%20Pipeline-blueviolet)
