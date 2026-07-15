# B1229011 模型訓練分支

此分支（`B1229011`）由 B1229011 維護，內容為 YOLO 模型訓練資料與相關輔助程式的整理版本。

## 資料夾結構

```
.
├── model/          # 各類別 YOLO 訓練專案（僅上傳 best.pt / last.pt 權重）
│   ├── background_train/
│   ├── balloon_train/
│   ├── bubble_train/
│   ├── doll_train/
│   ├── foreground_train/
│   ├── pad_train/
│   ├── robothead_train/
│   └── toy_train/
├── code/           # 優化 / 重構的程式（全部上傳）(上到下為舊到新)
│   ├── scoring_system/     # 評分系統
│   ├── signboard_system/   # stage牌辨識 (初版)
│   ├── 0708debug           # 聲音辨識與小bug修正
│   ├── stage_scoring       # JA判斷系統，內附有測試腳本(把文字檔放入output後執行)
│   ├── 0711stage10         # stage10修改TB判定
│   ├── hurry               # 修改路徑問題(絕對路徑改為鄉對路徑)、調用優化
│   └── 0714                # 修改stage1~4的重大bug，並優化硬體調用(包含關閉預覽視窗)
├── model_test/     # 本機模型測試用（影片、輸出結果等），不納入版控
├── .gitignore
└── README.md
```

## 版控規則（.gitignore）

- 預設忽略所有檔案，僅白名單放行以下內容：
  - `model/` 底下各訓練資料夾中的 `best.pt`、`last.pt`（訓練用資料集、預訓練權重不上傳；版本命名規則見下方「說明」）
  - `code/` 底下所有檔案（程式碼、模組、必要的示意圖片）
  - `model_test/model_test.py`（其餘測試影片、輸出結果等不上傳）
  - `.gitignore`、`README.md`

## 說明

### model/

- 各 `*_train/` 對應不同偵測類別（背景、氣球、泡泡、玩偶、前景、階段牌、機器人頭部、玩具）的訓練專案，僅保留每次訓練成果的 `best.pt`（最佳權重）與 `last.pt`（最後一輪權重），訓練資料集與預訓練權重（`yolo11s.pt`/`yolo26n.pt` 等）不上傳。
- 版本依資料夾名稱後綴日期排序，數字越大版本越新；無日期標示或非日期編號者為舊版測試用 model。

### code/（依上傳時間由舊到新）

- `scoring_system`：評分系統整合版，含 `main_scoring.py` 主程式與 `module/scoring_engine.py` 評分核心邏輯。
- `signboard_system`：階段牌辨識模組（初版），使用 EasyOCR 追蹤畫面中的階段牌數字（`modules/signboard.py`），`model/signboardphoto/` 為辨識用參考圖片、`test_signboard.py` 為測試腳本。
- `0708debug`：修正語音誤判問題——過濾 Whisper 對 `initial_prompt` 的幻覺複誦（`is_prompt_echo()`）、將怪聲比對相似度門檻由 0.58/0.65 提高至 0.85、Stage 8 起點判定改用精確關鍵字比對而非整句子字串搜尋、更新 `MATCHING_ALGORITHM_VERSION` 使舊快取失效；詳細問題根因與驗證方式見資料夾內 `README.md`。
- `stage_scoring`：JA（共同注意力）判斷系統，內附 `test_stage_scoring.py` 測試腳本，將文字檔放入 `output/` 後即可執行驗證。
- `0711stage10`：修改 Stage 10 的 TB 判定邏輯（`modules/scoring_engine.py`）。
- `hurry`：修正 `main.py` 路徑問題（絕對路徑改為相對路徑）、已有輸出檔案則自動跳過以避免重複處理、加入硬體加速（偵測到 CUDA 核心則調用 GPU，否則自動退回 CPU）。

### model_test/

- 僅供本機驗證模型效果使用，內容（測試影片、輸出結果等）不納入版控；遠端僅保留 `model_test.py` 供測試參考。
