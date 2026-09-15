# 兒童注意力監測與量化分析平台

### Multi-modal AI Joint Attention Assessment System

> 結合電腦視覺、語音辨識與視線追蹤，針對自閉症類群障礙（ASD）與典型發展（TD）兒童，
> 自動評估**共同注意力（Joint Attention）**能力的分析平台。
> 提供命令列批次分析與圖形化操作介面兩種使用方式。

---

## 目錄

- [這套系統在做什麼](#這套系統在做什麼)
- [三種使用方式](#三種使用方式)
- [免安裝版下載](#免安裝版下載)
- [專案目錄結構](#專案目錄結構)
- [環境安裝](#環境安裝)
- [執行方式](#執行方式)
- [輸出說明](#輸出說明)
- [核心模組](#核心模組)
- [常見問題排查](#常見問題排查)
- [文件索引](#文件索引)

---

## 這套系統在做什麼

施測流程共 14 個關卡，兒童在每一關被引導去注意某個目標（近物、遠物、氣球、操偶、
驚喜袋、怪聲、機器人畫畫與煙火秀……）。系統從錄影中自動抓出三個關鍵時間點：

| 符號 | 名稱 | 定義 |
|------|------|------|
| **T0** | 指示時間 | 系統聽到關鍵字（或偵測到目標物出現、怪聲響起），開啟判定時間窗的瞬間 |
| **TB** | 看向物品 | T0 之後，兒童首次注視目標物的瞬間 |
| **TH** | 看回施測者 | TB 之後，兒童把視線移回施測者或機器人的瞬間 |

三者加上「是否有指向動作（Pointing）」換算成每一關的**反應等級**：

```
TB 1 分 ＋ TH 2 分 ＋ Pointing 1 分  →  0~4 分
```

| 等級 | 條件 | 臨床意義 |
|------|------|----------|
| **HI** | 指向 ＋ 看回人 | 高度主動分享 |
| **LI** | 看向物 ＋ 看回人（無指向） | 視線交替主動 |
| **HR** | 遠物關卡，僅達成一項 | 有基本反應但未主動 |
| **LR** | 其他關卡，僅達成一項 | 低反應 |
| **F** | 皆未達成 | 無反應 |

判定規則的完整定義在 `modules/stage_scoring.py`；各關的 T0 / TB / TH 觸發條件與
結束時機，詳見 `CLAUDE.md` 的「各關卡偵測內容」章節。

### 技術構成

**多模態時序控制** — 語音觸發、視線射線、指向射線三路訊號交叉驗證。

**階段適應性物件偵測** — 依 OCR 讀到的階段牌動態切換 YOLO 模型（Lazy Loading），
在有限 VRAM 下維持推論效能。

**雙重射線幾何碰撞** — 指向射線（MediaPipe Hands + YOLO-Pose，Ray-AABB）
搭配 3D 視線向量（ETH-XGaze 五階段管線），雙重確認注意力目標。

**自動音軌縫合** — 分析完成後呼叫 FFmpeg，把標註影片與原始音軌無損對齊縫合。

---

## 三種使用方式

| 方式 | 進入點 | 適用情境 |
|------|--------|----------|
| **圖形化介面** | `run_ui.bat` / `ui/app.py` | 展示、單支影片操作、看報表與跨受試者比較 |
| **批次分析（Stage 1–10）** | `python hurry/main.py` | 一次跑完整個資料夾的影片，全自動不需互動 |
| **完整版分析（Stage 1–14）** | `python main.py` | 單支影片、需手動框選牌子 ROI、含機器人指物關卡 |

三者共用 `modules/` 的同一套計分邏輯，輸出格式一致。

---

## 免安裝版下載

> **給只想跑分析、不需要改程式的使用者。**
> 不需要安裝 Python、不需要 Anaconda、不需要另裝 CUDA Toolkit，解壓縮後雙擊即可執行。

### 下載連結

| 版本 | 大小 | 連結 |
|------|------|------|
| 免安裝完整版（含 Python 環境與全部模型權重）| 約 7~9 GB | **【https://drive.google.com/file/d/1qOrchtuMGub_8_92vS-ABmlk6YVw5xH6/view?usp=sharing】** |


### 安裝與執行

1. 解壓縮到**路徑不含中文**的資料夾，建議 `D:\attention` 或 `C:\attention`
2. 雙擊 `START_UI.bat`
3. 第一次執行會自動完成三件事，約 5~15 分鐘（畫面有進度訊息，請勿關閉視窗）：
   解壓縮內含的 Python 環境 → `conda-unpack` 修正環境內路徑 → 安裝 Whisper / EasyOCR 模型快取
4. 之後每次啟動只要 10~20 秒

把要分析的影片放進 `app\hurry\video\`，結果會輸出到 `app\hurry\output\`。
完整操作說明見壓縮檔內的 `README_FIRST.txt`。

### 執行需求

| 項目 | 需求 |
|------|------|
| 作業系統 | Windows 10 / 11（64 位元）|
| 顯示卡 | NVIDIA 顯卡 + 最新驅動程式 |
| 硬碟空間 | 至少 20 GB 可用空間 |
| 網路 | 不需要（模型都已內含）|

> ⚠️ **沒有 NVIDIA 顯卡**仍可開啟介面與檢視報表，但按「開始分析」會慢到不切實際
> （一支 6 分鐘的影片可能要跑數小時）。
>
> ⚠️ **顯示記憶體不足**時語音辨識會自動改用 CPU，結果一樣正確但該段慢很多
> （6 分鐘影片約 10~20 分鐘），主控台會明確提示。影像分析仍走 GPU 不受影響。

### 這個發佈版不含什麼

不含任何受試者影片或分析結果，`video` 與 `output` 資料夾都是空的。
另請注意分析報告（`.txt`）會包含語音辨識的**完整逐字稿**，若處理臨床錄影，
散布這些檔案前請確認符合所屬單位的研究倫理規範。

> 💡 **開發者請往下看。** 若你要修改程式碼，請照下方「環境安裝」自行建立
> `mediapipe_py39` 環境。免安裝版的 `release/` 是 `build_release.bat` 的打包產物，
> 體積 7~9 GB 且已列入 `.gitignore`，不在 Git 倉庫中。

---

## 專案目錄結構

```text
C:\project\
├── README.md                      # 本檔
├── CLAUDE.md                      # 開發規範、各關卡判定規格、環境注意事項
├── config.yaml                    # 統一設定（視線模型路徑、裝置、校正、閾值）
├── run_ui.bat                     # 圖形化介面啟動器（雙擊即可）
├── ffmpeg.exe                     # FFmpeg 本體（音軌縫合用）
│
├── main.py                        # 完整版：互動式單影片分析（Stage 1~14）
├── model_test.py                  # 本機 YOLO 模型快速測試
├── gpu_test.py                    # GPU / CUDA 環境確認腳本
│
├── ui/                            #   圖形化介面（詳見 ui/檔案清單.md）
│   ├── app.py                     #   主視窗與四個頁籤的串接
│   ├── run_ui.bat                 #   啟動器（ui\ 版）
│   ├── qt_compat.py               #   PySide6 / PyQt5 相容層
│   ├── theme.py                   #   深色主題與色票
│   ├── pipeline_api.py            #   與 hurry/main.py 的橋接、模型快取
│   ├── workers.py                 #   背景執行緒（避免視窗卡住）
│   ├── report_parser.py           #   event_record txt 解析
│   ├── widgets.py                 #   共用元件與自繪圖表
│   ├── pages/                     #   四個功能頁面
│   ├── README_UI.md               #   介面操作說明
│   └── 檔案清單.md                 #   介面各檔案職責與改動紀錄
│
├── hurry/                         # 批次版（Stage 1~10，多影片一次跑完）
│   ├── main.py                    #   批次總控：先跑完所有 Whisper，再批次視覺分析
│   ├── video/                     #   批次輸入影片（不上傳 Git）
│   └── output/                    #   批次輸出：{編號}.mp4 + {編號}.txt
│                                  #   以及各影片獨立的 _speech_{編號}/ 語音快取
│
├── modules/                       # 核心分析模組（完整版與批次版共用）
│   ├── speech.py                  #   語音觸發器（關鍵字窗 + noise.wav 怪聲窗）
│   ├── speech_engine.py           #   Whisper 辨識核心（獨立子行程，抗幻覺過濾）
│   ├── signboard.py               #   EasyOCR 階段牌追蹤狀態機
│   ├── models_manager.py          #   YOLO 模型 Lazy Loading（依 Stage 切換）
│   ├── interaction.py             #   指向手勢判定（MediaPipe + YOLO-Pose）
│   ├── scoring_engine.py          #   T0 / TB / TH 計分核心、報告輸出
│   ├── stage_scoring.py           #   反應等級計算（F / LR / HR / LI / HI）
│   └── gaze_estimation/           #   視線估計 5 階段管線
│       ├── gaze_pipeline.py       #     整合入口
│       ├── stage1_face_detection.py   # 人臉偵測與特徵點
│       ├── stage2_head_pose.py        # SolvePnP 頭部歐拉角
│       ├── stage3_normalization.py    # 影像透視正規化
│       ├── stage4_gaze_network.py     # ResNet-50 視線網路
│       ├── stage5_gaze_vector.py      # 3D 視線向量轉換
│       ├── state_manager.py           # 視線有限狀態機（防閃爍）
│       └── visualization.py           # 視線與頭部姿態渲染
│
├── summarize/                     # 跨受試者統計
│   ├── summarize.py               #   批次解析 txt → CSV / Excel 統計表
│   ├── files/                     #   放入要解析的 event_record txt
│   └── output/                    #   輸出：txt_資料統整.csv / .xlsx
│
├── docs/                          #   文件與分析報告
│   ├── 環境安裝說明.md
│   ├── 人工-AI視線差異分析.md
│   ├── 人工-AI指向偵測差異分析.md
│   ├── 人工-AI結果差異對照表.xlsx
│   └── modify.md                  #   1-10 版的修改指南（已套用，保留供追溯）
│
├── model/                         # AI 模型權重（不上傳 Git）
│   ├── front_model.pt             #   Stage 1、2、11、12：近物
│   ├── background_model.pt        #   Stage 3、4、13、14：遠物
│   ├── balloon_model.pt           #   Stage 5：氣球
│   ├── doll_model.pt              #   Stage 6：操偶
│   ├── toy_model.pt               #   Stage 7：玩具
│   ├── tablet_model.pt            #   Stage 9、10：平板
│   ├── robot_model.pt             #   Stage 9+：機器人本體（TH 判定用）
│   ├── yolo11n-pose.pt            #   人體姿態骨架
│   ├── noisesample/noise.wav      #   Stage 8 怪聲比對樣板
│   ├── signboardphoto/            #   Stage 7 牌子模板圖（7_1~7_4.png）
│   └── gaze/
│       ├── epoch_24_ckpt.pth.tar  #   ResNet-50 視線網路權重（~88MB）
│       ├── face_landmarker.task   #   MediaPipe 臉部特徵點
│       ├── hand_landmarker.task   #   MediaPipe 手部特徵點
│       ├── nano.pt                #   YOLO 人臉偵測
│       └── face_model_ethxgaze.txt  # ETH-XGaze 3D 人臉幾何座標
│
├── video/                         # 完整版輸入影片（不上傳 Git）
└── output/                        # 完整版輸出（不上傳 Git）
```

---

## 環境安裝

完整逐步說明見 [`docs/環境安裝說明.md`](docs/環境安裝說明.md)，以下是摘要。

### 前置需求

| 項目 | 版本 |
|------|------|
| 作業系統 | Windows 10 / 11 |
| Python | 3.9（conda 環境 `mediapipe_py39`）|
| CUDA | 11.8+（強烈建議；CPU 模式慢很多）|
| FFmpeg | 已內附 `ffmpeg.exe`，不需另裝 |

### 建立環境

```powershell
conda create -n mediapipe_py39 python=3.9
conda activate mediapipe_py39

# PyTorch（CUDA 11.8 版）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 其餘套件
pip install -r requirements.txt

# 圖形化介面（要用 UI 才需要）
pip install PyQt5
```

> 💡 介面支援 PySide6 與 PyQt5 兩種 Qt 綁定，會自動偵測。
> **但這台機器只能用 PyQt5**：`mediapipe_py39` 的 `Library\bin` 已有另一套 Qt6 DLL，
> 會讓 PySide6 匯入時噴 `DLL load failed ... 找不到指定的程序`。詳見 `ui/README_UI.md`。

> ⚠️ **不要用 `.venv`（Python 3.12）跑本專案**——缺 `easyocr`、`opencv` 版本不相容
> （沒有 `TrackerKCF`），會直接噴錯。conda 的 `base` 與 `web` 環境同樣不適用。
>
> 註：文件早期版本曾把環境名稱寫成 `attention_integrate`，那是**別台機器**的名稱。
> 目前這台機器實際使用的是 `mediapipe_py39`，可用 `conda env list` 確認。

### 核心依賴版本（驗證通過）

| 套件 | 版本 | 用途 |
|------|------|------|
| `torch` / `torchvision` | `2.7.1+cu118` | 深度學習底座、視線網路、YOLO 推論 |
| `ultralytics` | `8.4.19` | YOLO 物件偵測 / Pose 骨架 |
| `openai-whisper` | `20250625` | 語音辨識 |
| `mediapipe` | `0.10.35` | 手部 / 臉部特徵點 |
| `easyocr` | `1.7.2` | 階段牌數字 OCR |
| `opencv-python` | `4.13.0.92` | 影像處理與渲染 |
| `scipy` | `1.13.1` | 頭部姿態旋轉矩陣 |
| `numpy` | `2.0.2` | 矩陣幾何運算 |
| `PySide6` | 任意 | 圖形化介面（選用）|

### 模型權重

模型檔體積大，不含於 Git 倉庫，需另行取得後放入 `model/`：

- `epoch_24_ckpt.pth.tar`（~88MB）放入 `model/gaze/`，**勿解壓縮**
- `face_model_ethxgaze.txt` 需以 **Raw 下載**取得純文字座標檔（約 1KB）
- `noise.wav` 放在 `model/noisesample/`，1~3 秒的手機警報聲片段。
  提供後系統會**完全停用** RMS / 頻譜 fallback，避免假陽性；
  缺此檔會退回純頻譜特徵偵測，準確率明顯較低

---

## 執行方式

### 圖形化介面

雙擊 `run_ui.bat`，或：

```powershell
conda activate mediapipe_py39
cd C:\project
python ui\app.py
```

四個頁籤：**① 影片載入與設定 → ② 即時監測 → ③ 單次結果報表 → ④ 歷史統計比較**。
操作細節與展示流程見 [`ui/README_UI.md`](ui/README_UI.md)。

介面本身不做任何分析判斷，所有計分仍由 `modules/` 負責，
顯示的數字與命令列跑出來的完全一致。

### 批次分析（Stage 1–10）

```powershell
conda activate mediapipe_py39
cd C:\project\hurry
python main.py
```

- 自動掃描 `hurry/video/` 下所有影片，跑完為止
- **Step 0**：先跑完所有影片的 Whisper 語音辨識（此時 VRAM 空著，GPU 全力辨識，快 5–10 倍）
- **Step 1**：載入 YOLO / 視線 / EasyOCR 視覺模型，逐支批次分析
- 每支影片使用獨立快取子目錄 `output/_speech_{影片名}/`，避免互相污染
- 輸出影片與報告都已存在的影片會自動跳過，中斷後可直接重跑續做

### 完整版分析（Stage 1–14）

```powershell
conda activate mediapipe_py39
cd C:\project
python main.py
```

- 終端機輸入影片編號
- 預覽視窗彈出後，框選牌子可能出現的範圍，按 **Enter** 確認
- 執行中按 **`q`** 安全結束並匯出影片，按 **`r`** 手動重置階段

### 跨受試者統計

把要統計的 event_record txt 放進 `summarize/files/`，然後：

```powershell
cd C:\project\summarize
python summarize.py
```

輸出 `summarize/output/txt_資料統整.csv` 與 `.xlsx`。

> 圖形化介面的「歷史統計比較」頁也能做同樣的事，而且可以直接標 ASD / TD 分組
> 並產生組間比較圖。`summarize/files/` 目前有 69 份報告，在介面中把資料夾
> 切到那裡就能一次比較全部樣本。

---

## 輸出說明

| 檔案 | 說明 |
|------|------|
| `{編號}.mp4` | 帶 YOLO 框、視線箭頭、指向射線標註的影片（含原始音軌）|
| `{編號}.txt` | 分析報告：總分、各關 T0/TB/TH/反應等級、Gazing 統計、完整事件流水、Whisper 逐字稿 |
| `_speech_{編號}/speech_cache.json` | Whisper 辨識快取（含 `trigger_windows`、`noise_events`、`segment_records`）|

報告 txt 的格式範例：

```
=== T0 / Pointing / TB / TH Detail ===

04. Stage 4 -- 真人指遠物(Pointing - Far) -- 反應等級 HI(4分)
    T0       = 23.33s
    Pointing = 26.56s  (+3.23s from T0) x1
    TB       = 24.03s  (+0.7s from T0) x1
    TH       = 27.1s   (+3.77s from T0 / +3.07s from TB) x1
    Sequence = T0 -> Pointing OK -> TB OK -> TH OK
    計分結束 = 39.99s
```

---

## 核心模組

### 語音辨識（`speech.py` / `speech_engine.py`）

- **Whisper large-v3**，以獨立子行程隔離執行，防止與視覺模型搶 VRAM 而崩潰
- 快取機制：第二次執行直接讀 `speech_cache.json`，跳過辨識（< 0.1s）
- **顯示記憶體不足時自動退回 CPU**：large-v3 以 FP32 執行約需 8~10 GB VRAM。
  只判斷「有沒有 CUDA」是不夠的——顯卡存在但記憶體不足時會丟 CUDA OOM，
  例外被上層接住後變成「語音辨識靜默失效」：沒有觸發時間窗、Stage 8/9/10
  全部失準，畫面上卻看不出異常。現在偵測到 OOM 會清快取並改用 CPU 重跑，
  慢很多但結果正確，且主控台會明確說明發生了什麼事
- 繁簡容錯（「畫」/「画」、「這裡」/「这里」）
- **抗幻覺過濾**：Whisper 聽不清時會把 prompt 逐字複誦回來
  （「請勿忽略短促的聲音：看這裡、看這裡…」），這種段落的關鍵字全是假的，計分時一律略過
- **Stage 7→8 怪聲偵測**：以 `model/noisesample/noise.wav` 做頻譜模板相似度比對
  （Cosine Similarity ≥ 0.7），找出最佳命中時段寫入 `noise_events`

> ⚠️ 若曾在沒有 `noise.wav` 的狀態下執行過，請刪除舊快取再重跑：
> `hurry/output/_speech_{影片名}/speech_cache.json` 或 `output/speech_cache.json`

### 階段牌追蹤（`signboard.py`）

- **EasyOCR** 辨識數字牌驅動階段狀態機
- **7 軌 OCR 策略**：Normal / CLAHE / Adaptive / Bold Erosion / Sharpen / Otsu / Inverted，多軌投票提升容錯
- **Seven-Hunt Mode**（Stage 6 專屬）：7 軌仍找不到「7」時，強制 `allowlist='7'` 並用
  3 倍放大圖補強——根治帶橫槓歐式「7」被誤讀成「1」的問題
- **換牌位置凍結**：偵測到手遮擋牌子時，KCF 追蹤器繼續跑但結果不寫入，掃描框釘在原位
- 批次模式自動框選右下 1/4（`initialize_roi_auto`）；完整版為互動框選（`initialize_roi`）

### 動態模型管理（`models_manager.py`）

- 依關卡自動切換 7 種 YOLO 模型，**Lazy Loading** 避免 VRAM 一次爆掉
- 自動偵測 CUDA，啟用 `cudnn.benchmark` 與 TF32
- Stage 9 之後支援雙模型並行（目標物 `yolo_boxes` ＋ 機器人 `robot_boxes`）
- Stage 5 額外做**彩度與膚色過濾**，排除白袍衣袖、牆面被誤判成氣球

### 互動判定（`interaction.py`）

- **YOLO11-Pose**（`conf=0.3`，適應施測者背對或兒童被桌子遮擋）
  結合 **MediaPipe Hands**（Tasks API VIDEO 模式，`num_hands=2`）
- **Ray-AABB 射線投射**：從兒童食指發射虛擬射線，判定是否擊中目標框
- **身分識別防禦**：以手臂關節連動分數區分施測者與兒童
- **跨影片單調時間戳**：`_ts_base` 確保批次模式下 MediaPipe VIDEO 模式時間戳單調遞增
- **SMA 平滑化**過濾幀間抖動；臉部誤觸拒絕半徑 90px

### 計分引擎（`scoring_engine.py` / `stage_scoring.py`）

- 從 `speech_cache.json` 讀三類事件重建絕對時間軸：關鍵字觸發窗、語音段落、怪聲事件
- **TB / TH 交替計次**：視線停在同一目標不重複計，必須「物品 ↔ 人」真正交替才累加
- **冷卻時間**：TB / TH 各 1.5 秒、Pointing 0.8 秒，防 YOLO 掉偵測造成重複計次
- `stage_scoring.compute_stage_score()` 以語意組合（而非分數查表）判定反應等級，
  與 IJA 臨床定義對齊

### 視線估計（`gaze_estimation/`）

- 完整實作 **ETH-XGaze 官方幾何規範**的五階段推論管線，輸出 Pitch / Yaw 與 3D 視線向量
- **防閃爍三件套**：5 幀滑動平均射線、命中框外擴 10% 容差、命中遲滯
  （立即亮起、連續 10 幀未命中才熄滅）
- `GazeFSMManager` 管理視線狀態轉換，含極端轉頭（EXTREME_TURNING）代償

---

## 常見問題排查

**介面開不起來，說找不到 Qt 綁定**
`conda activate mediapipe_py39` 後 `pip install PyQt5`。

**PySide6 匯入時噴 `DLL load failed while importing QtCore: 找不到指定的程序`**
conda 環境的 `Library\bin` 裡有另一套 Qt6 DLL 被優先載入，缺少 PySide6 需要的函式。
重裝 PySide6 無效，改用 PyQt5（Qt5，DLL 檔名不同不會衝突）：
`pip uninstall -y PySide6 PySide6-Essentials PySide6-Addons shiboken6` 後 `pip install PyQt5`。

**骨架或指向射線不出現**
確認 `interaction.analyze_interaction()` 沒有被 `if len(yolo_boxes) > 0:` 包住（應無條件呼叫）。
YOLO-Pose 的 `conf` 設太高（如 0.5）時，施測者背對或兒童被桌子遮擋會整個人消失，應維持 `conf=0.3`。

**MediaPipe FileNotFoundError**
確認 `InteractionEngine` 建立時有明確傳入 `hand_model_path`。若專案路徑是 Windows junction
（例如 `C:\project` → 實際在別處），`__file__` 會解析到真實路徑導致相對路徑算錯。

**Stage 7→8 不切換**
確認 `model/noisesample/noise.wav` 存在；確認關鍵字清單沒有混入音效符號
（`"嗶"`、`"[聲音]"` 這類非口說詞語會造成誤觸發）；有舊快取請刪掉重跑。

**批次模式多影片時間軸亂掉**
確認每支影片開頭有呼叫 `interaction.reset_tracking()`（重設 `_ts_base` 與 MediaPipe 時間戳偏移）。

**EasyOCR CUDA OOM**
`hurry/main.py` 已有 OOM 捕捉，會自動清 VRAM 後繼續。頻繁出現可調高 OCR 跳幀
（介面上的「OCR 跳幀」，或 `hurry/main.py` 的 `OCR_SKIP`）。

**Stage 1 的 T0 太早被代償建立**
Stage 1 前面通常有問名字、寒暄，實測「你看」中位數要等約 9 秒。
`scoring_engine.py` 已為 Stage 1 單獨放寬到 15 秒（`FALLBACK_T0_DELAY_SEC_STAGE1`），
Stage 2~4 維持 4 秒。

---

## 編碼與安全守則

- 所有檔案讀寫一律 `encoding='utf-8'`，避免 Windows 預設編碼（CP950）造成閃退或亂碼
- **推論與繪圖分離**：所有 AI 推論吃乾淨的 `frame`，所有視覺化畫在 `display_frame = frame.copy()`。
  違反此原則會造成模型互相污染（OCR 讀到 YOLO 框線、MediaPipe 特徵點偏移）
- 影片、模型、輸出等大型檔案已列入 `.gitignore`

---

## 技術棧

![Python](https://img.shields.io/badge/Python-3.9-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%2011.8-orange)
![YOLO](https://img.shields.io/badge/Ultralytics-YOLO11-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hands%20%2B%20Pose-red)
![Whisper](https://img.shields.io/badge/Whisper-large--v3-lightgrey)
![EasyOCR](https://img.shields.io/badge/EasyOCR-Stage%20Board-yellow)
![ETH-XGaze](https://img.shields.io/badge/ETH--XGaze-5%20Stage%20Pipeline-blueviolet)
![Qt](https://img.shields.io/badge/PySide6%20%2F%20PyQt5-Desktop%20UI-41cd52)
