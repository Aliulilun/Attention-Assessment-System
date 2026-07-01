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
├── code/           # 優化 / 重構的程式（全部上傳）
│   ├── scoring_system/     # 評分系統
│   └── signboard_system/   # stage牌辨識 (初版)
├── model_test/     # 本機模型測試用（影片、輸出結果等），不納入版控
├── .gitignore
└── README.md
```

## 版控規則（.gitignore）

- 預設忽略所有檔案，僅白名單放行以下內容：
  - `model/` 底下各訓練資料夾中的 `best.pt`、`last.pt`（訓練用資料集、預訓練權重 `yolo11s.pt`/`yolo26n.pt` 等不上傳）
  - `code/` 底下所有檔案（程式碼、模組、必要的示意圖片）
  - `.gitignore`、`README.md`
- `model_test/` 整個資料夾完全排除，不會有任何檔案被上傳（測試影片、暫存快取、模型測試腳本皆為本機使用）

## 說明

- `model/` 內各 `*_train/` 對應不同偵測類別（背景、氣球、泡泡、玩偶、前景、階段牌、機器人頭部、玩具）的訓練專案，僅保留每次訓練成果的 `best.pt`（最佳權重）與 `last.pt`（最後一輪權重）。
- `code/scoring_system` 為評分系統的整合版程式（`main_scoring.py` 及其模組）。
- `code/signboard_system` 為階段牌辨識模組，使用 EasyOCR 追蹤畫面中的階段牌數字。
- `model_test/` 僅供本機驗證模型效果使用，內容不會同步到遠端版本庫。
