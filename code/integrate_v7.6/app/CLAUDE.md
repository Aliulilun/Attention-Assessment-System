<!-- rtk-instructions v2 -->
# RTK (Rust Token Killer) - Token-Optimized Commands
## Golden Rule
**Always prefix commands with `rtk`**. If RTK has a dedicated filter, it uses it. If not, it passes through unchanged. This means RTK is always safe to use.
**Important**: Even in command chains with `&&`, use `rtk`:
```bash
# ❌ Wrong
git add . && git commit -m "msg" && git push
# ✅ Correct
rtk git add . && rtk git commit -m "msg" && rtk git push
```
## RTK Commands by Workflow
### Test (60-99% savings)
```bash
rtk pytest              # Python test failures only (90%)
rtk test <cmd>          # Generic test wrapper - failures only
```
### Git (59-80% savings)
```bash
rtk git status          # Compact status
rtk git log             # Compact log (works with all git flags)
rtk git diff            # Compact diff (80%)
rtk git add             # Ultra-compact confirmations (59%)
rtk git commit          # Ultra-compact confirmations (59%)
rtk git push / pull     # Ultra-compact confirmations
```
### Files & Search (60-75% savings)
```bash
rtk ls <path>           # Tree format, compact (65%)
rtk read <file>         # Code reading with filtering (60%)
rtk grep <pattern>      # Search grouped by file (75%)
rtk find <pattern>      # Find grouped by directory (70%)
```
### Analysis & Debug (70-90% savings)
```bash
rtk err <cmd>           # Filter errors only from any command
rtk log <file>          # Deduplicated logs with counts
rtk summary <cmd>       # Smart summary of command output
```
### Meta Commands
```bash
rtk gain                # View token savings statistics
rtk init                # Add RTK instructions to CLAUDE.md
rtk init --global       # Add RTK to ~/.claude/CLAUDE.md
```
<!-- /rtk-instructions -->

---

## 語言規定
**所有回覆一律使用繁體中文。**

---

## 編碼規定
**所有檔案皆使用 UTF-8 編碼，Claude 讀寫檔案時也一律使用 UTF-8。**
檔案寫入（如 `event_record.txt`、逐字稿、快取 JSON）必須明確指定 `encoding='utf-8'`，避免 Windows 預設編碼（CP950）導致閃退或亂碼。

---

## 專案簡介
**兒童注意力監測與量化分析平台（Attention Assessment System）**
針對自閉症類群障礙（ASD）與典型發展（TD）兒童，透過多模態 AI 技術（視覺 + 語音 + 幾何推論）自動評估共同注意力（Joint Attention）能力。

---

## 技術棧
- **語言**：Python 3.9（conda 環境 `mediapipe_py39`）
- **目標偵測**：YOLO v11（ultralytics）
- **視線 / 姿態估計**：MediaPipe FaceMesh、自訓練 gaze 模型（ETH-XGaze，ResNet-50，GPU/CUDA 推論）
- **語音辨識**：OpenAI Whisper large-v3（獨立子行程，避免 VRAM 衝突）
- **手勢 / 骨架**：YOLO v11 Pose + MediaPipe HandLandmarker（VIDEO 模式）
- **OCR**：EasyOCR（階段牌辨識，allowlist='12345678'）
- **影像處理**：OpenCV、NumPy、PyTorch（CUDA）
- **設定管理**：PyYAML（`config.yaml`）

---

## 專案結構
```
C:\project\
├── main.py                    # 完整版入口：互動式（終端機輸入影片編號）
│                               #   手動框選牌子 ROI、Stage 1~14 全跑
├── hurry/
│   ├── main.py                # 🌟 批次版入口：自動掃描 hurry/video/ 資料夾
│   │                           #   全部影片跑完為止、Stage 1~10
│   ├── video/                 # 批次輸入影片（不上傳 Git）
│   └── output/                # 批次輸出：{編號}.mp4 + {編號}.txt
├── modules/                   # 完整版與批次版共用的核心模組
│   ├── speech.py              # 語音觸發器前端（關鍵字窗口 + noise.wav 怪聲窗口）
│   ├── speech_engine.py       # Whisper 分析核心（獨立子行程）
│   ├── signboard.py           # EasyOCR 牌子追蹤狀態機（Stage 1~7）
│   ├── models_manager.py      # YOLO 模型 Lazy Loading（依 Stage 切換）
│   ├── interaction.py         # 指向手勢判定（MediaPipe + YOLO-Pose）
│   ├── scoring_engine.py      # T0 / TB / TH 計分核心
│   ├── stage_scoring.py       # 反應等級計算（F / LR / HR / LI / HI）
│   └── gaze_estimation/       # 視線估計 5 階段管線
│       ├── gaze_pipeline.py
│       ├── stage1_face_detection.py
│       ├── stage2_head_pose.py
│       ├── stage3_normalization.py
│       ├── stage4_gaze_network.py
│       ├── stage5_gaze_vector.py
│       ├── state_manager.py   # GazeFSM（防閃爍有限狀態機）
│       └── visualization.py
├── ui/                        # 🌟 圖形化介面（PySide6／PyQt5 皆可）
│   ├── app.py                 #   主視窗：側邊四頁籤導覽與頁面串接
│   ├── run_ui.bat             #   啟動器（ui\ 版；根目錄另有一支同功能的）
│   ├── qt_compat.py           #   Qt 綁定相容層（唯一處理 PySide6/PyQt5 差異之處）
│   ├── theme.py               #   深色主題 QSS 與色票
│   ├── pipeline_api.py        #   以 importlib 把 hurry/main.py 當函式庫載入 + 模型快取
│   ├── workers.py             #   AnalysisWorker（QThread）：語音→載模型→逐幀分析
│   ├── report_parser.py       #   event_record txt → 結構化資料
│   ├── widgets.py             #   共用元件與 QPainter 自繪圖表（不用 matplotlib，中文才不變方塊）
│   ├── pages/                 #   ①影片載入設定 ②即時監測 ③單次報表 ④歷史統計
│   ├── README_UI.md           #   介面操作說明與報告展示流程
│   └── 檔案清單.md             #   介面各檔職責、對既有程式的改動紀錄
├── summarize/
│   ├── summarize.py           # 批次解析 txt → CSV / Excel 統計表
│   ├── files/                 # 放入要解析的 event_record txt 檔案
│   └── output/                # 輸出：txt_資料統整.csv / .xlsx
├── docs/                      # 🌟 文件與分析報告（純文件，無程式引用）
│   ├── 環境安裝說明.md
│   ├── 人工-AI視線差異分析.md
│   ├── 人工-AI指向偵測差異分析.md
│   ├── 人工-AI結果差異對照表.xlsx
│   └── modify.md              #   1-10 版修改指南（已套用，保留供追溯）
├── model/                     # 模型權重（不上傳 Git）
├── video/                     # 完整版輸入影片（不上傳 Git）
├── output/                    # 完整版輸出（event_record_1to14_5.txt 等）
├── README.md                  # 專案總覽（安裝、三種執行方式、模組說明、排查）
├── config.yaml                # 統一設定（gaze 模型路徑、閾值等）
├── run_ui.bat                 # 圖形化介面啟動器（雙擊即可）
├── ffmpeg.exe                 # FFmpeg 本體（音軌縫合用）
├── model_test.py              # 本機 YOLO 模型快速測試
├── gpu_test.py                # GPU / CUDA 環境確認腳本
└── CLAUDE.md
```

---

## 環境啟動
```powershell
# 執行 main.py / hurry/main.py / ui/app.py 一律用這個 conda 環境
conda activate mediapipe_py39
python --version   # 應為 3.9.x
```
⚠️ 本機實際可用的環境是 **`mediapipe_py39`**（Python 3.9，已驗證 easyocr / ultralytics /
whisper / mediapipe / cv2 / torch 齊全，`torch.cuda.is_available()` 為 True）。
文件早期版本曾寫成 `attention_integrate`，那是**別台機器**的環境名稱，本機並不存在。

⚠️ 系統上另有 `.venv`（Python 3.12）與 conda 環境 `web`、`base`，**不要用它們跑本專案**——
`.venv` 缺 `easyocr`、`opencv` 版本不相容（沒有 `TrackerKCF`），會直接噴錯。

⚠️ 若 `conda env list` 出現名為 `attention_integrate` 的環境，那是一次失敗的 clone
留下的半套殘骸（沒有 `python.exe`，activate 後 `python` 會落回 base 的 3.12），
請執行 `conda remove --name attention_integrate --all` 清掉，以免日後又選錯。

---

## 常用指令
```powershell
conda activate mediapipe_py39

# 🌟 圖形化介面（需先 pip install PySide6；或雙擊 C:\project\run_ui.bat）
cd C:\project
python ui\app.py

# 批次版：自動掃描 hurry/video/ 底下所有影片，跑完為止
cd C:\project\hurry
python main.py

# 完整版：互動式輸入單一影片編號，手動框選牌子 ROI，跑 Stage 1~14
cd C:\project
python main.py

# 統計分析：將 event_record txt 放到 summarize/files/，跑完輸出 xlsx
cd C:\project\summarize
python summarize.py
```

---

## 模型說明（`model/`）
| 模型檔案 | 對應 Stage（見 `modules/models_manager.py`）|
|----------|------|
| `front_model.pt` | Stage 1、2（真人指近物）；Stage 11、12（機器人指近物，conf=0.15）|
| `background_model.pt` | Stage 3、4（真人指遠物）；Stage 13、14（機器人指遠物）|
| `balloon_model.pt` | Stage 5（神奇氣球）|
| `doll_model.pt` | Stage 6（看偶寫字）|
| `toy_model.pt` | Stage 7（開箱驚喜袋）|
| `tablet_model.pt` | Stage 9、10（機器人畫畫 / 煙火秀，看向平板）|
| `robot_model.pt` | 機器人本體偵測（Stage 9+ TH 判定用，conf=0.6）|
| `yolo11n-pose.pt` | YOLO-Pose：骨架 / 施測者兒童分區 |
| `model/gaze/epoch_24_ckpt.pth.tar` | 視線估計主模型（ResNet-50，ETH-XGaze）|

Stage 8（手機怪聲）沒有對應 YOLO 模型，T0 由 `model/noisesample/noise.wav` 樣板比對（Cosine Similarity ≥ 0.7）觸發。

---

## 開發注意事項
- **config.yaml** 是所有模組的統一設定入口，修改前先確認影響範圍
- 視線估計模型走 **GPU/CUDA**（`config.yaml` 的 `gaze_estimation.model.device: "cuda"`），不是 CPU
- 語音觸發關鍵字定義在各入口的 `SPEECH_KEYWORDS`（`hurry/main.py` 與 `main.py` 清單不完全相同，以各自檔案為準）
- `hurry/main.py`（批次版）與 `main.py`（完整版）**Stage 判定邏輯不同**：批次版有 Trigger Lock + 升階佇列防止跳號漏記；完整版偵測到新階段就立即同步
- 怪聲觸發（Stage 7→8）改由 `noise.wav` 模板比對負責；`WEIRD_SOUND_TRIGGERS` 只保留口說詞語（如「機器人」），不再含聲音符號
- **推論與繪圖分離**：所有 AI 推論吃乾淨的 `frame`，所有視覺化畫在 `display_frame = frame.copy()`，最後 `out.write(display_frame)`。違反此原則會造成模型互相污染（OCR 讀到 YOLO 框線、MediaPipe 臉部特徵點偏移）
- **視線防閃爍**：使用 `last_valid_gaze` + `gaze_fallback_counter`（容許 5 幀失敗再清空），邏輯判定與繪圖都用 `active_gaze = last_valid_gaze`
- TB / TH **必須交替計次**（`_last_counted_target`）：視線停在同一目標不重複計，必須先看人（計 TH）才能再計下一次 TB

---

## 各關卡偵測內容與 T0 / TB / TH / 結束條件

> `tb_mode` = 什麼算看向目標；`th_mode` = 什麼算看回來；`end_time` = 計分窗口關閉時機

### Stage 1-2：真人指近物（Pointing - Near）
- **YOLO 模型**：`front_model.pt`，conf=0.70，面積 > 40% 畫面者濾除（防把人體誤判成物品）
- **T0**：語音出現「你看」或「看這裡」時建立；若 Whisper 漏轉錄超過 **4 秒**仍無關鍵字，OCR 代償自動補建 T0（時間點取進入該 Stage 的瞬間）；關鍵字晚到時會自動升級 T0 時間戳
- **TB**：視線射線命中 YOLO 近物框（`child_is_gazing_at = True`）
- **TH**：視線命中施測者區域 TESTER_ZONE（含 Pitch > −5°、Yaw > 10° 角度過濾）
- **計分結束**：Stage 切換時立即關閉（`end_time = ∞`）
- **Stage 升階**：OCR 牌子偵測到 1→2

---

### Stage 3-4：真人指遠物（Pointing - Far）
- **YOLO 模型**：`background_model.pt`，conf=0.75
- **T0**：同 Stage 1-2（「你看」/「看這裡」關鍵字，OCR 代償延遲同為 4 秒）
- **TB**：視線命中遠物框
- **TH**：視線命中施測者區域
- **計分結束**：Stage 切換
- **Stage 升階**：OCR 牌子偵測到 3→4

---

### Stage 5：神奇氣球
- **YOLO 模型**：`balloon_model.pt`，conf=0.75
- **T0**：**YOLO 首次偵測到氣球**的那一幀（不依賴語音）
- **TB**：視線命中氣球框
- **TH**：視線命中施測者區域
- **計分結束**：Stage 切換
- **Stage 升階**：OCR 牌子偵測到 5

---

### Stage 6：看偶寫字（Puppet Writing）
- **YOLO 模型**：`doll_model.pt`，conf=0.70，面積 > 40% 者濾除
- **T0**：**YOLO 首次偵測到操偶**的那一幀
- **TB**：視線命中操偶框
- **TH**：視線命中施測者區域
- **計分結束**：Stage 切換
- **Stage 升階**：OCR 牌子偵測到 6

---

### Stage 7：開箱驚喜袋（Mystery Bag）
- **YOLO 模型**：`toy_model.pt`，conf=0.70，面積 > 40% 者濾除
- **T0**：**YOLO 首次偵測到玩具**的那一幀
- **TB**：視線命中玩具框
- **TH**：視線命中施測者區域
- **計分結束**：Stage 切換
- **Stage 升階**：OCR + 模板比對（7.png，需連續 3 幀 TM 分數 > OCR 讀出 6 的信心值）；亦可由 noise.wav 命中後代償跳至 Stage 8

---

### Stage 8：手機怪聲（Strange Sound）
- **YOLO 模型**：無（Stage 8 無目標物可看）
- **T0**：`noise.wav` 模板比對（Cosine Similarity ≥ 0.7）命中時刻，或語音事件中關鍵字「怪聲」對應的 `stage_start_times[8]`（誤差 < 0.05s）；兩者取最早出現
- **TB**：**無**（`tb_mode = None`，不要求先看物品）
- **TH**：視線命中施測者區域；**不需先達成 TB 即可計次**
- **Pointing**：只要 MediaPipe 指向射線存在即計（不需命中特定物品，由 `last_child_pointing_active` 判斷）
- **計分結束**：**T0 + 10 秒**（固定判定窗口，硬截止）
- **Stage 進入**：絕對時間軸（noise.wav 命中 or 語音事件比對）

---

### Stage 9：機器人畫畫（Robot Drawing）
- **YOLO 模型**：`tablet_model.pt`（conf=0.75）+ `robot_model.pt`（conf=0.6，TH 用）
- **T0**：語音文字含「畫」或「画」的事件，時間點需與 `stage_start_times[9]` 誤差 < 0.05s
- **TB**：視線命中平板框
- **TH**：視線命中 `robot_boxes`（`th_mode = "robot_box"`）
- **計分結束**：語音出現「畫好了 / 你看」後 **+3 秒**；若找不到則 **T0 + 15 秒**
- **Stage 進入**：絕對時間軸

---

### Stage 10：機器人煙火秀（Social Referencing）
- **YOLO 模型**：`tablet_model.pt`（conf=0.75）+ `robot_model.pt`（conf=0.6，TH 用）
- **T0**：語音文字含「煙火 / 321 / 三二一 / 三 / 3」的事件，與 `stage_start_times[10]` 誤差 < 0.05s；若這些字在 Stage 9 之後才出現（含 Stage 8 之後的回退機制）才採用
- **TB**：視線命中平板框
- **TH**：視線命中 `robot_boxes`
- **計分結束**：**T0 + 10 秒**
- **Stage 進入**：絕對時間軸

---

### Stage 11-14：機器人指物（Robot Pointing）
- **YOLO 模型**：
  - Stage 11、12（指近物）→ `front_model.pt`，conf=**0.15**（實測目標信心度偏低）+ `robot_model.pt`
  - Stage 13、14（指遠物）→ `background_model.pt`，conf=0.75 + `robot_model.pt`
- **T0**：語音同時含「小朋友」+「你看」的事件，與 `stage_start_times[11-14]` 誤差 < 0.05s；相鄰兩次語音事件需間隔 ≥ 2 秒（去除重複觸發），取前 4 筆依序對應 Stage 11→14
- **TB**：視線命中 YOLO 目標框
- **TH**：視線命中 `robot_boxes`
- **計分結束**：`stage_start_times[n+1]`（下一關的起始時間）；最後一關用各關平均間隔（`avg_pointing_duration`，預設 3.0 秒）
- **Stage 進入**：絕對時間軸（同上語音事件）

---

### 通用計次規則（所有 Stage 共用）

| 事件 | 觸發條件 | 冷卻時間 | 交替鎖 |
|------|---------|---------|--------|
| **Pointing** | 射線命中目標框，上升邊緣 | 0.8 秒 | 無 |
| **TB** | 視線命中目標框，上升邊緣 | 1.5 秒 | `_last_counted_target ≠ "object"`（必須先計過 TH）|
| **TH** | 視線命中施測者/機器人，上升邊緣 | 1.5 秒 | `_last_counted_target == "object"`（必須先計過 TB）；Stage 8 無此限制 |

---

## 核心詞彙表
| 術語 | 定義 |
|------|------|
| **T0** | 系統聽到關鍵字、開啟語音觸發時間窗的瞬間 |
| **TB** | T0 後，兒童首次看向目標物件（`child_is_gazing_at = True`）的瞬間 |
| **TH** | TB 後，兒童將視線看回施測者或機器人的瞬間 |
| **Time Window** | 語音觸發後開啟的黃金判定時間（例如 3 秒內） |
| **ROI** | Region of Interest，限定 AI 只在此區域內辨識 |
| **Ray-Casting** | 從兒童食指發射虛擬射線，計算是否擊中 YOLO 目標框（Ray-AABB 演算法）|

---

## 開發輔助工具（`mediapipe_py39` 環境已裝）
- **`pytest`**：單元測試框架，可直接對 `scoring_engine.py` 的 T0/TB/TH 邏輯寫回歸測試
- **`pip-tools`**（`pip-compile` / `pip-sync`）：從 `requirements.in` 產生鎖定版本的 `requirements.txt`，避免重演 opencv 版本打架問題
- **`py-spy`**：零修改對執行中行程做效能取樣（`py-spy top --pid <PID>`），用來定位 `main.py` / `hurry/main.py` 實際瓶頸
