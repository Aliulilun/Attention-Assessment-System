
# 多模態 AI 兒童聯合注意力行為分析系統

> **Multi-Modal AI System for Joint Attention Assessment in Children**
>
> 整合電腦視覺、語音辨識與深度學習，自動化分析受測兒童在結構化測驗情境中的聯合注意力行為

[![Branch](https://img.shields.io/badge/branch-integrate__ver1-blue)](https://github.com/Aliulilun/Attention-Assessment-System/tree/integrate_ver1)
[![Python](https://img.shields.io/badge/python-3.8--3.12-green)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-orange)](LICENSE)

---

## 目錄

1. [專案簡介](#1-專案簡介)
2. [研究背景與動機](#2-研究背景與動機)
3. [系統架構總覽](#3-系統架構總覽)
4. [實驗設計與評分規則](#4-實驗設計與評分規則)
5. [核心技術模組](#5-核心技術模組)
6. [資料夾結構](#6-資料夾結構)
7. [環境需求與安裝](#7-環境需求與安裝)
8. [模型檔案準備](#8-模型檔案準備)
9. [執行方式](#9-執行方式)
10. [輸出格式說明](#10-輸出格式說明)
11. [系統限制與注意事項](#11-系統限制與注意事項)
12. [貢獻者](#12-貢獻者)
13. [引用](#13-引用)

---

## 1. 專案簡介

本系統為一套全自動化的多模態 AI 分析平台，專為**兒童聯合注意力（Joint Attention, JA）行為**的臨床評估設計。系統透過分析施測過程錄影，自動偵測兒童在不同測驗題目（Stage）中的三項核心行為指標：

| 指標 | 定義 |
|------|------|
| **Pointing** | 兒童手指確實指向當前題目的目標物 |
| **TB** (Turning Behavior) | 兒童視線轉向並注視目標物 |
| **TH** (Turn-back to Human) | 兒童在 TB 發生後，視線轉回施測者或機器人（社會性共享） |

依據上述三項指標的達成與否，系統自動計算各題目的反應等級，最終推算受測兒童的聯合注意力類型，協助臨床工作者進行初步評估。

---

## 2. 研究背景與動機

聯合注意力（Joint Attention）是嬰幼兒社會認知發展的核心里程碑，也是自閉症類群障礙（ASD）早期篩查的關鍵指標之一。傳統臨床評估依賴人工觀察與紙本記錄，耗時費力，且受觀察者主觀因素影響。

本研究目標為**透過自動化多模態分析**，提供客觀、可重現的評量依據，減輕臨床人員負擔，並支援大規模施測資料的標準化收集。

### 測驗情境

系統基於結構化施測情境設計，包含：

- **真人指物**（施測者指向近端/遠端物品）
- **社會情境事件**（神奇氣球、看偶寫字、開箱驚喜袋、手機怪聲）
- **機器人互動**（機器人畫畫、機器人煙火秀、機器人指向近端/遠端物品）

---

## 3. 系統架構總覽

```
                    ┌──────────────────────────────────────────┐
                    │          待分析錄影 (MP4)                 │
                    └──────────────┬───────────────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                    ▼
    ┌──────────────────┐  ┌─────────────────┐  ┌────────────────────┐
    │   語音辨識模組   │  │  牌子辨識模組   │  │  視線估計模組      │
    │  (Whisper STT)   │  │  (EasyOCR)      │  │  (ETH-XGaze)       │
    │                  │  │                 │  │                    │
    │ · 關鍵字觸發窗   │  │ · Stage 1~7     │  │ · 5 階段視線管線   │
    │ · 怪聲模板比對   │  │   數字牌辨識    │  │ · Ray-AABB 射線    │
    │ · 語音快取 (JSON)│  │ · ROI 搜索追蹤  │  │ · 時序 FSM 盲追蹤  │
    └────────┬─────────┘  └───────┬─────────┘  └──────────┬─────────┘
             │                    │                        │
             └────────────────────┼────────────────────────┘
                                  ▼
                    ┌──────────────────────────┐
                    │      主控程式 (main.py)   │
                    │   Stage 推進大腦          │
                    │   · OCR 視覺判斷 (1~7)   │
                    │   · 絕對時間軸推進 (8~14) │
                    │   · 聽覺代償 (7→8)        │
                    └──────────────┬────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                    ▼
    ┌──────────────────┐  ┌─────────────────┐  ┌────────────────────┐
    │   指向判斷模組   │  │  YOLO 模型管理  │  │   計分引擎         │
    │  (interaction.py)│  │ (models_manager) │  │ (scoring_engine.py)│
    │                  │  │                 │  │                    │
    │ · YOLO Pose      │  │ · 多模型動態    │  │ · T0/TB/TH 偵測    │
    │   骨架偵測       │  │   載入 (Lazy)   │  │ · 事件紀錄         │
    │ · MediaPipe Hands│  │ · CUDA/FP16     │  │ · 計分 & 等級輸出  │
    │ · Ray-AABB 碰撞  │  │   硬體加速      │  │ · 結果報告         │
    └──────────────────┘  └─────────────────┘  └──────────┬─────────┘
                                                           │
                                                           ▼
                                            ┌────────────────────────┐
                                            │   聯合注意力等級判定   │
                                            │   (stage_scoring.py)   │
                                            │                        │
                                            │  F / LR / HR / LI / HI │
                                            └────────────────────────┘
```

---

## 4. 實驗設計與評分規則

### 4.1 測驗題目（Stage）對照表

| Stage | 題目名稱 | Stage 判定方式 | 備註 |
|-------|---------|--------------|------|
| 1 | 真人指近物（第 1 次） | OCR 牌子辨識 | T0 = 偵測到「你看」 |
| 2 | 真人指近物（第 2 次） | OCR 牌子辨識 | T0 = 偵測到「你看」 |
| 3 | 真人指遠物（第 1 次） | OCR 牌子辨識 | T0 = 偵測到「你看」 |
| 4 | 真人指遠物（第 2 次） | OCR 牌子辨識 | T0 = 偵測到「你看」 |
| 5 | 神奇氣球 | OCR 牌子辨識 | T0 = YOLO 偵測到目標物 |
| 6 | 看偶寫字 | OCR 牌子辨識 | T0 = YOLO 偵測到目標物 |
| 7 | 開箱驚喜袋 | OCR 牌子辨識 | T0 = YOLO 偵測到目標物 |
| 8 | 手機怪聲 | 聽覺代償（noise.wav 模板比對）| T0 = 怪聲起始時間 |
| 9 | 機器人畫畫 | 絕對時間軸（「畫」關鍵字）| T0 = 「畫」詞出現時間 |
| 10 | 機器人煙火秀 | 絕對時間軸（「321/煙火」關鍵字）| T0 = 煙火倒數起始 |
| 11 | 機指近物（第 1 次） | 絕對時間軸（「小朋友你看」）| T0 = 第 1 次「小朋友你看」 |
| 12 | 機指近物（第 2 次） | 絕對時間軸（「小朋友你看」）| T0 = 第 2 次「小朋友你看」 |
| 13 | 機指遠物（第 1 次） | 絕對時間軸（「小朋友你看」）| T0 = 第 3 次「小朋友你看」 |
| 14 | 機指遠物（第 2 次） | 絕對時間軸（推算）| T0 = 第 4 次「小朋友你看」 |

---

### 4.2 機器人指物題（Stage 11~14）時間軸設計

機器人指物題（Stage 11~14）共設 4 輪，依序為兩次近端指物、兩次遠端指物，以「你看（小朋友你看）」關鍵字作為各階段的時間界標。

```
時間 ──────────────────────────────────────────────────────►

  [S11 T0]          [S12 T0]          [S13 T0]          [S14 T0]
     │                  │                  │                  │
  第1次「小朋友你看」  第2次「小朋友你看」  第3次「小朋友你看」  第4次「小朋友你看」
     │<── S11 判斷窗 ──>│<── S12 判斷窗 ──>│<── S13 判斷窗 ──>│<──── S14 判斷窗 ────>
     │                  │                  │                  │                      │
 近端指物 #1         近端指物 #2         遠端指物 #1         遠端指物 #2              │
                                                                                      │
 S14 結束時間點 = S14_T0 + avg(S11 窗長, S12 窗長, S13 窗長)   ──────────────────────►
```

**各階段判斷窗結束邏輯**：

| 階段 | 判斷窗結束點 |
|------|------------|
| S11 | 偵測到第 2 次「小朋友你看」（即 S12_T0） |
| S12 | 偵測到第 3 次「小朋友你看」（即 S13_T0） |
| S13 | 偵測到第 4 次「小朋友你看」（即 S14_T0） |
| S14 | S14_T0 + 前三階段判斷窗時間長度之平均值 |

---

### 4.3 三項行為指標定義

#### T0（觸發時間點）
各題目的起始計時點，依 Stage 類型不同而有不同觸發來源：
- **Stage 1~4**：「你看」語音關鍵字被偵測的瞬間
- **Stage 5~7**：YOLO 首次偵測到該題目目標物的瞬間
- **Stage 8**：手機怪聲（noise.wav 模板比對命中）的起始時間
- **Stage 9**：「畫」相關語音關鍵字出現的瞬間
- **Stage 10**：「321 / 煙火」倒數語音關鍵字出現的瞬間
- **Stage 11~14**：「小朋友你看」語音關鍵字出現的瞬間（依序取 4 次）

#### TB（Turning Behavior，目標物注視）
在 T0 後，系統偵測到兒童視線射線與當前題目目標物包圍框（AABB）發生交叉的首次時間點。

- **偵測方式**：視線向量 Ray-AABB 射線交叉判定
- **冷卻機制**：同一次注視事件最小間隔 1.5 秒（防 YOLO 掉幀重複計數）

#### TH（Turn-back to Human/Robot，社會性回視）
在 TB 達成後，系統偵測到兒童視線轉回「社會參照對象」的首次時間點。

- **Stage 1~8**：TH 對象為畫面左側施測者區域（Pitch > -5°, Yaw > 10°）
- **Stage 9~14**：TH 對象為機器人所在位置（機器人雙手間之空間包圍框）
- **冷卻機制**：同一次回視事件最小間隔 1.5 秒

---

### 4.4 計分與聯合注意力等級

各 Stage 依據 TB、TH、Pointing 是否達成進行計分：

| 指標 | 達成得分 | 未達成得分 |
|------|---------|----------|
| TB | 1 分 | 0 分 |
| TH | 2 分 | 0 分 |
| Pointing | 1 分 | 0 分 |
| **合計** | **0 ~ 4 分** | |

依據合計分數與題目類型，對應聯合注意力反應等級：

| 分數 | 近端題（Non-Far Stage） | 遠端題（Far Stage） | 定義 |
|------|----------------------|-------------------|------|
| 0 | **F** | **F** | Failed，未有任何反應 |
| 1 | **LR** | **HR** | Low / High Responding |
| 2 | **HI** | **HI** | High Initiating |
| 3 | **LI** | **LI** | Low Initiating |
| 4 | **HI** | **HI** | High Initiating |

> **遠端題**（Stage 3, 4, 13, 14）在僅得 1 分時直接歸類為 HR，反映遠端社會參照的相對難度較高。

---

## 5. 核心技術模組

### 5.1 語音辨識（`modules/speech.py`, `modules/speech_engine.py`）

| 技術要素 | 說明 |
|---------|------|
| **模型** | OpenAI Whisper large-v3（本地離線推理） |
| **關鍵字觸發** | 以 `trigger_window` 機制標記關鍵詞前後 3 秒的語音影響窗 |
| **怪聲偵測** | `noise.wav` 模板比對（而非 Whisper 關鍵字），防止 Whisper 將音效符號誤判為口說詞語 |
| **快取機制** | 首次執行後將 Whisper 結果寫入 `speech_cache.json`，後續重新執行直接讀取，大幅縮短啟動時間 |
| **關鍵字集合** | 開始、321、三二一、準備、你看、小朋友、看這裡、準備囉、機器人、放煙火、煙火、三、畫一幅、畫好了、特別的畫 |

---

### 5.2 牌子辨識（`modules/signboard.py`）

| 技術要素 | 說明 |
|---------|------|
| **模型** | EasyOCR（字元白名單：`1234567`） |
| **適用範圍** | 僅負責辨識 Stage 1~7，Stage 8 以上改用絕對時間軸推進 |
| **ROI 搜索** | 手動框選牌子可能出現的畫面區域（支援 `BATCH_AUTO_ROI=1` 環境變數自動化批次執行） |
| **防抖動機制** | 連續多幀確認 + 防止 Stage 倒退的單向遞進邏輯 |
| **暖機機制** | 系統啟動時對 GPU 進行 dummy 圖片推理，消除硬體冷啟動延遲 |

---

### 5.3 視線估計（`modules/gaze_estimation/`）

本模組採用**五階段視線估計管線**（基於 ETH-XGaze 架構），並擴展加入時序有限狀態機（FSM）以處理極端轉頭時特徵點丟失的情況。

```
輸入幀
  │
  ▼
Stage 1: 人臉偵測
  · YOLO nano：快速定位人臉候選框
  · MediaPipe Face Landmarker (Tasks API)：提取 468 個 3D 面部特徵點
  │
  ▼
Stage 2: 頭部姿態估計
  · OpenCV solvePnP (ITERATIVE)：求解 3D 旋轉矩陣與平移向量
  · 使用 ETH-XGaze 3D 頭部模型 (face_model_ethxgaze.txt)
  │
  ▼
Stage 3: 影像正規化
  · 依據頭部姿態將眼部區域正規化至標準視角
  · 輸出：224×224 正規化眼部影像
  │
  ▼
Stage 4: 視線推理網路
  · ETH-XGaze ResNet-50 (epoch_24_ckpt.pth.tar)
  · 輸入：正規化影像；輸出：(pitch, yaw) 視線角度
  · Pitch 偏移校正：-12.5°（可於 config.yaml 調整）
  │
  ▼
Stage 5: 視線向量轉換
  · 將 (pitch, yaw) 角度轉換為 3D 視線方向向量
  · 投影至 2D 畫面坐標系用於後續 Ray-AABB 碰撞判定
```

**時序有限狀態機（GazeFSMManager）**：

當面部特徵點因極端轉頭而消失時，FSM 進入 `EXTREME_TURNING` 狀態，利用前幀頭部位置與 YOLO 頭部框進行盲追蹤，保持視線方向的連續估計，避免因短暫丟幀造成事件遺漏。

---

### 5.4 指向判斷（`modules/interaction.py`）

| 技術要素 | 說明 |
|---------|------|
| **骨架偵測** | YOLOv11-Pose (`yolo11n-pose.pt`)：偵測人體關鍵點（17 個骨架節點） |
| **手部偵測** | MediaPipe Hand Landmarker (Tasks API)：提取手指關鍵點 |
| **身分識別** | Arm Link Score 演算法：以手臂連結分數動態區分施測者與兒童（以畫面左側 35% 為施測者區邊界） |
| **指向過濾** | 幾何計算手指伸直程度，拒絕握拳或手掌邊緣誤觸發 |
| **碰撞判定** | Ray-AABB 精確射線與矩形交集算法：視線射線需確實通過物品「精確本體框」才判定 HIT |
| **SMA 平滑** | 簡單移動平均（window=5）平滑骨架關鍵點坐標，降低 YOLO 掉幀抖動 |

---

### 5.5 YOLO 多模型管理（`modules/models_manager.py`）

| 技術要素 | 說明 |
|---------|------|
| **動態載入** | Lazy Loading：僅在對應 Stage 被啟用時才載入該 Stage 的模型，避免 GPU 顯存溢出 |
| **硬體加速** | 自動偵測 CUDA 可用性，支援 FP16 半精度推理 |
| **多模型對照** | 依 Stage 自動派發對應客製化 YOLO 模型（近物、背景、氣球、看偶、玩具、機器人） |
| **推理參數** | 統一採用信心度門檻 `conf=0.75`，推理解析度 `imgsz=960` |

各 Stage 對應模型一覽：

| Stage | 模型檔案 | 偵測目標 |
|-------|---------|---------|
| 1~2 | `front_model.pt` | 近端目標物 |
| 3~4 | `background_model.pt` | 遠端目標物 |
| 5 | `balloon_model.pt` | 氣球 |
| 6 | `doll_model.pt` | 看偶 |
| 7 | `toy_model.pt` | 玩具袋 |
| 8 | — | 無 YOLO 模型（怪聲觸發） |
| 9~10 | `robot_point_model.pt` | 平板/機器人 |
| 11~14 | `robot_point_model.pt` + 固定 ROI | 近/遠物品 + 機器人 |

> **Stage 11~12 特別設計**：近端物品位置固定，系統採用固定比例 ROI 框（預設：畫面下半部 45%~100%），在 ROI 內執行 YOLO 後**永久鎖定**首次偵測到的物品框位置，後續幀直接使用鎖定座標，不再執行 YOLO，節省運算資源。

---

### 5.6 計分引擎（`modules/scoring_engine.py`）

`ScoringEngine` 類別負責整合所有模組的即時輸出，維護各 Stage 的計分狀態，並於影片分析結束後輸出完整報告。

**主要流程**：

1. `__init__`：從語音快取建立 Stage 8~14 的**絕對時間軸**（`_build_absolute_timeline`）
2. `update_frame`：每幀呼叫，更新：
   - Trigger Record（T0 建立 / Pointing / TB / TH 計次）
   - 注視計分（與「你看」觸發窗交叉比對）
   - 臨床事件紀錄
3. `write_report`：影片結束後輸出 `event_record.txt`

**防重複計次機制**：

- 上升邊緣偵測（Rising Edge Detection）：僅在狀態從 False → True 的第一幀觸發，防止連續幀重複計數
- 冷卻時間（Cooldown）：TB/TH 間隔 ≥ 1.5 秒，Pointing 間隔 ≥ 0.8 秒

---

### 5.7 聯合注意力等級（`modules/stage_scoring.py`）

依據各 Stage 的 `trigger_event_record` 計算總分與等級：

```python
total = TB_score (0|1) + TH_score (0|2) + Pointing_score (0|1)
# 0 分 → F；1 分近端 → LR；1 分遠端 → HR；2 分 → HI；3 分 → LI；4 分 → HI
```

---

## 6. 資料夾結構

```
integrate/                         # 系統根目錄
│
├── main.py                        # 主程式（協調器），執行此檔案啟動分析
├── config.yaml                    # 視線估計與其他參數設定
├── requirements.txt               # Python 依賴套件清單
├── ffmpeg.exe                     # Windows 用 FFmpeg（音軌縫合）
├── yolo11n-pose.pt                # YOLOv11-Pose 人體骨架模型
│
├── model/                         # 所有模型權重（需自行放置）
│   ├── front_model.pt             # Stage 1~2：近端物品偵測
│   ├── background_model.pt        # Stage 3~4：遠端物品偵測
│   ├── balloon_model.pt           # Stage 5：氣球偵測
│   ├── doll_model.pt              # Stage 6：看偶偵測
│   ├── toy_model.pt               # Stage 7：玩具袋偵測
│   ├── robot_point_model.pt       # Stage 9~14：機器人/平板偵測
│   │
│   ├── gaze/                      # 視線估計相關模型
│   │   ├── nano.pt                # YOLO nano（人臉初步定位）
│   │   ├── face_landmarker.task   # MediaPipe Face Landmarker
│   │   ├── epoch_24_ckpt.pth.tar  # ETH-XGaze 視線推理網路
│   │   ├── face_model_ethxgaze.txt# 3D 頭部模型（用於 solvePnP）
│   │   └── hand_landmarker.task   # MediaPipe Hand Landmarker
│   │
│   └── noisesample/
│       └── noise.wav              # 手機怪聲模板（Stage 7→8 切換觸發）
│
├── modules/                       # 核心功能模組（解耦設計）
│   ├── __init__.py
│   ├── speech.py                  # 語音觸發窗管理（關鍵字提取 + 噪聲比對）
│   ├── speech_engine.py           # Whisper 推理引擎（完整 STT 後端）
│   ├── signboard.py               # EasyOCR 牌子辨識（Stage 1~7）
│   ├── models_manager.py          # YOLO 多模型動態載入與派發
│   ├── interaction.py             # 手勢指向分析（骨架 + 手部 + Ray-AABB）
│   ├── scoring_engine.py          # 計分核心（T0/TB/TH 狀態機 + 報告輸出）
│   ├── stage_scoring.py           # JA 等級判定（F/LR/HR/LI/HI）
│   │
│   └── gaze_estimation/           # 視線估計子模組
│       ├── __init__.py
│       ├── gaze_pipeline.py       # 五階段視線估計管線整合
│       ├── stage1_face_detection.py
│       ├── stage2_head_pose.py
│       ├── stage3_normalization.py
│       ├── stage4_gaze_network.py
│       ├── stage5_gaze_vector.py
│       ├── camera_utils.py        # 相機內參工具
│       ├── config_loader.py       # 視線模組設定載入
│       ├── state_manager.py       # 時序 FSM（極端轉頭盲追蹤）
│       └── visualization.py      # 視線箭頭與面框視覺化
│
├── video/                         # 放置待分析的 MP4 影片
│
└── output/                        # 分析結果輸出
    ├── output_result_final_1to14_5.mp4  # 標注完成的分析影片（含音軌）
    ├── temp_no_audio_1to14_5.mp4        # 中間暫存無聲影片（分析完成後自動刪除）
    ├── event_record_1to14_5.txt         # 詳細事件紀錄與計分報告
    └── speech_cache.json                # Whisper 語音辨識快取
```

---

## 7. 環境需求與安裝

### 7.1 基本需求

| 項目 | 需求 |
|------|------|
| **作業系統** | Windows 10/11（主要開發環境）；macOS / Linux 亦可執行 |
| **Python** | 3.8 ~ 3.12 |
| **GPU（建議）** | NVIDIA GPU（支援 CUDA 11.8 或以上），無 GPU 可使用 CPU 但速度較慢 |
| **記憶體** | 建議 16 GB RAM 以上 |
| **磁碟空間** | 模型檔案約 3~5 GB，建議預留 10 GB 以上 |

### 7.2 安裝步驟

**方案 A：使用 `venv`（推薦）**

```bash
# 1. 進入 integrate 目錄
cd integrate

# 2. 建立虛擬環境
python -m venv .venv

# 3. 啟動虛擬環境
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 4. 升級 pip
python -m pip install --upgrade pip

# 5. 安裝所有依賴
pip install -r requirements.txt
```

**方案 B：使用 Conda**

```bash
conda create -n attention_system python=3.9 -y
conda activate attention_system
pip install -r requirements.txt
```

**GPU 加速（CUDA，選配）**

若您有 NVIDIA GPU，建議改用 CUDA 版 PyTorch 以獲得最佳效能：

```bash
# 以 CUDA 11.8 為例（請依您的 CUDA 版本調整）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**macOS 特別說明**

macOS 使用者需確保 Python 為 ARM64 原生版本（M1/M2/M3 晶片），PyTorch 將自動啟用 MPS（Metal Performance Shaders）加速。此外，語音處理需安裝 PortAudio：

```bash
brew install portaudio
```

**Linux 特別說明**

```bash
sudo apt-get install portaudio19-dev python3-pyaudio
```

### 7.3 主要依賴套件

```
ultralytics>=8.0.0       # YOLO 系列模型推理
torch>=2.0.0             # PyTorch 深度學習框架
torchvision>=0.15.0      # PyTorch 視覺工具
opencv-contrib-python>=4.13.0  # OpenCV（含 contrib 模組）
mediapipe>=0.10.32       # 人臉 & 手部關鍵點（Tasks API）
openai-whisper           # Whisper 語音辨識
easyocr                  # 牌子數字辨識
pandas>=2.0.0            # 數據處理
pyyaml>=6.0              # 設定檔解析
numpy>=1.24.0            # 科學計算
```

---

## 8. 模型檔案準備

> 由於模型檔案體積龐大，**不包含在 Git 儲存庫中**，需另行下載或自行訓練後放置於指定位置。

### 8.1 視線估計模型（必要）

下載 ETH-XGaze 預訓練模型，放置於 `integrate/model/gaze/`：

| 檔案名稱 | 說明 | 來源 |
|---------|------|------|
| `nano.pt` | YOLO nano 人臉定位 | 專案 GitHub Releases |
| `face_landmarker.task` | MediaPipe Face Landmarker | [MediaPipe Model Cards](https://developers.google.com/mediapipe/solutions/vision/face_landmarker) |
| `epoch_24_ckpt.pth.tar` | ETH-XGaze ResNet-50 | 專案 GitHub Releases |
| `face_model_ethxgaze.txt` | 3D 頭部點雲模型 | 專案 GitHub Releases |
| `hand_landmarker.task` | MediaPipe Hand Landmarker | [MediaPipe Model Cards](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker) |

### 8.2 YOLO 客製化模型（必要）

放置於 `integrate/model/`：

| 檔案名稱 | 用途 |
|---------|------|
| `front_model.pt` | 近端目標物偵測 (Stage 1~2) |
| `background_model.pt` | 遠端目標物偵測 (Stage 3~4) |
| `balloon_model.pt` | 氣球偵測 (Stage 5) |
| `doll_model.pt` | 看偶偵測 (Stage 6) |
| `toy_model.pt` | 玩具袋偵測 (Stage 7) |
| `robot_point_model.pt` | 機器人/平板偵測 (Stage 9~14) |

### 8.3 其他必要檔案

| 路徑 | 說明 |
|------|------|
| `model/noisesample/noise.wav` | 手機怪聲模板音頻（用於 Stage 7→8 切換） |
| `yolo11n-pose.pt` | YOLOv11-Pose 骨架模型（根目錄） |

---

## 9. 執行方式

### 9.1 基本執行

```bash
# 啟動虛擬環境後，進入 integrate 目錄
cd integrate

# 執行主程式
python main.py
```

系統啟動後會出現互動提示：

```
>>> 請輸入要分析的影片編號（直接按 Enter 使用預設 video/61.mp4）：
```

- 輸入影片編號（如 `61` 或 `61.mp4`），系統自動在 `video/` 目錄下尋找對應檔案
- 直接按 Enter 使用 `config.yaml` 設定的預設影片
- 兩次輸入均失敗時，自動選取 `video/` 目錄下第一支 MP4

### 9.2 啟動流程

1. **語音分析**（首次執行需數分鐘）：Whisper 對整段影片進行語音辨識，建立 `speech_cache.json`
2. **絕對時間軸建立**：ScoringEngine 依語音事件自動推算 Stage 8~14 的起始時間
3. **OCR 暖機**：對 EasyOCR 進行 dummy 圖片推理，消除第一幀延遲
4. **ROI 框選**：彈出視窗，請用滑鼠框選牌子可能出現的畫面區域，按 `Enter` 確認
5. **主分析迴圈**：影片逐幀分析，即時顯示預覽視窗

### 9.3 分析過程中的操作

| 按鍵 | 功能 |
|------|------|
| `q` | 提早結束分析（已處理的部分仍會輸出報告） |
| `r` | 手動重置 Stage 為 0（調試用途） |

### 9.4 批次自動化執行（選配）

設定環境變數 `BATCH_AUTO_ROI=1` 可跳過手動框選，適合批次處理大量影片：

```bash
# Windows
set BATCH_AUTO_ROI=1 && python main.py

# macOS / Linux
BATCH_AUTO_ROI=1 python main.py
```

### 9.5 設定 `config.yaml`

若需調整視線估計參數，編輯 `config.yaml`：

```yaml
gaze_estimation:
  calibration:
    pitch_offset_deg: -12.5  # 視線仰角偏移校正（依實驗室環境調整）
  model:
    use_gpu: true             # 是否使用 GPU 推理
```

---

## 10. 輸出格式說明

### 10.1 分析影片（`output_result_final_1to14_5.mp4`）

輸出帶有標注的分析影片（含原始音軌），畫面左上角顯示即時分析狀態：

```
Time: 12.5 s                    ← 當前時間
Stage: 3 (1-7 OCR / 8-14 Auto) ← 當前 Stage 及判定方式
Keyword Detected: YES (Active)  ← 語音關鍵字觸發窗狀態
Child Pointing Hit: YES!        ← 兒童指向是否命中目標物
Gaze: P=-8.3 Y=15.2             ← 視線 Pitch/Yaw 角度（度）
Child Gazing At Object: YES!    ← 視線是否注視目標物（TB）
Score 2 | S3 Hit 1 | Total 4    ← 計分狀態
Task 真人指遠物(Pointing-Far) T0 11.20 ← 當前 Task 計時資訊
```

畫面上同時標注：

- **視線箭頭**（紅色）：兒童當前視線方向
- **目標物框**（綠色/青色鎖定框）：YOLO 偵測到的目標物
- **機器人框**（橘色）：YOLO 偵測到的機器人
- **GAZING! 黃色框**：視線命中目標物時的高亮標記
- **GAZING AT ROBOT (TH)! 紅色框**：視線命中機器人（TH 達成）
- **紫色框**：FSM 極端轉頭盲追蹤狀態

---

### 10.2 事件紀錄報告（`event_record_1to14_5.txt`）

純文字格式的完整評估報告，結構如下：

```
=== Result Summary ===
Video: video/61.mp4
----------------------------------------
Total Score: 8
Total Gazing Events: 12
========================================
=== T0 / Pointing / TB / TH Detail ===
  First-occurrence = timestamp; Count = total (incl. first)
  Cooldowns: TB/TH=1.5s, Pointing=0.8s
========================================

01. Stage 1 -- 真人指近物(Pointing - Near) -- 反應等級 HI(4分)
  T0 = 8.23s
  Pointing = 9.15s (+0.92s from T0) x2
  TB = 9.01s (+0.78s from T0) x1
  TH = 11.30s (+3.07s from T0 / +2.29s from TB) x1
  Sequence = T0 -> Pointing OK -> TB OK -> TH OK

02. Stage 2 -- 真人指近物(Pointing - Near) -- 反應等級 LI(3分)
  T0 = 45.60s
  Pointing = not detected
  TB = 46.20s (+0.60s from T0) x1
  TH = 48.10s (+2.50s from T0 / +1.90s from TB) x1
  Sequence = T0 -> Pointing -- -> TB OK -> TH OK

...（其餘 Stage 依序列出）

========================================
=== Stage Gazing Stats ===
Stage 1 [真人指近物(Pointing - Near)]: scored, GazingCount=3
Stage 2 [真人指近物(Pointing - Near)]: scored, GazingCount=2
...
----------------------------------------
=== Full Event Log ===
[0.1s] [SYSTEM] Scoring Version: 1-14_ABSOLUTE_TEXT_V12_EXPANDED
[8.2s] Stage change -> 1
[8.23s] T0: 真人指近物(Pointing - Near)
[9.01s] TB#1: 真人指近物(Pointing - Near) RT=0.78s
[9.15s] Pointing#1: 真人指近物(Pointing - Near) +0.92s from T0
...
```

---

### 10.3 語音快取（`speech_cache.json`）

Whisper 辨識結果與觸發事件的快取檔，供後續重新執行時直接讀取，避免重複推理。包含：
- `segment_records`：逐段語音辨識結果
- `trigger_events`：各關鍵字觸發事件（含 `trigger_window` 起止時間）
- `noise_events`：怪聲模板比對偵測結果

---

## 11. 系統限制與注意事項

### 硬體相關

- **GPU 顯存**：同時載入 Whisper large-v3 + ETH-XGaze + 多個 YOLO 模型約需 6~8 GB GPU 顯存，建議使用顯存 8 GB 以上的 GPU
- **CPU 執行**：可正常執行但分析速度將低於即時（約 2~5 FPS），不建議用於長時間影片
- **ffmpeg**：音軌縫合需要系統安裝 FFmpeg 並加入環境變數 PATH；Windows 使用者可使用根目錄的 `ffmpeg.exe`

### 視線估計

- 視線估計依賴正面或輕微側臉（Yaw ≤ ±30°），極端側臉時自動切換至 FSM 盲追蹤模式
- Pitch 偏移校正值（`pitch_offset_deg`）需依實際拍攝角度與鏡頭位置調整
- 如環境光線不足或人臉遮擋，視線估計信心度可能下降

### 語音辨識

- Whisper large-v3 對閩南語、客語等非普通話語音支援有限
- 環境噪音（如冷氣聲、桌椅摩擦聲）可能造成幻覺文字（Hallucination），建議錄音時使用收音麥克風
- 怪聲（Stage 8）的 `noise.wav` 模板需對應實際施測影片中的音效特徵

### 牌子辨識

- EasyOCR 對模糊、傾斜或部分遮擋的數字牌辨識率可能下降
- ROI 框選建議涵蓋牌子整個可能出現的區域，但避免含入複雜背景
- 強烈反光或背景對比不足時建議調整照明

### 計分規則

- 各 Stage 的 T0~結束窗口邏輯嚴格依賴「你看 / 小朋友你看」語音事件的正確辨識，若關鍵字被 Whisper 遺漏，對應 Stage 的計分紀錄可能無法建立
- Stage 14 的結束時間為前三階段均值推算，若 Stage 11~13 任一關鍵字缺失，推算精度可能下降

---

## 12. 貢獻者

| 角色 | 貢獻內容 |
|------|---------|
| 研究設計 | 實驗設計、評分規則制定、JA 等級對照表 |
| 系統開發 | 多模態整合、模組化架構、計分引擎 |
| 模型訓練 | 客製化 YOLO 模型、ETH-XGaze 微調 |
| 臨床驗證 | 施測流程標準化、人工標記資料集建立 |

---
## 13. 引用與參考 (Citation & References)

### 核心模型引用

```bibtex
@inproceedings{zhang2020ethxgaze,
  title     = {ETH-XGaze: A Large Scale Dataset for Gaze Estimation
               under Extreme Head Poses and Gaze Directions},
  author    = {Xucong Zhang and Seonwook Park and Thabo Beeler and
               Derek Bradley and Siyu Tang and Otmar Hilliges},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2020}
}
@misc{abcfsa2023yolov8,
  author       = {Abcfsa},
  title        = {YOLOv8\_head\_detector: Head Detection Model based on YOLOv8},
  howpublished = {\url{https://github.com/Abcfsa/YOLOv8_head_detector}},
  year         = {2023},
  note         = {GitHub Repository}
}
```

### 參考技術文件

- **ETH-XGaze**: [GitHub](https://github.com/xucong-zhang/ETH-XGaze) | [arXiv](https://arxiv.org/abs/2007.15837)
- **Gaze Normalization (CVPR 2015)**: Zhang et al., *Appearance-Based Gaze Estimation in the Wild*
- **MediaPipe Face Mesh**: [官方文件](https://developers.google.com/mediapipe/solutions/vision/face_landmarker)
- **OpenCV solvePnP**: [官方文件](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)
- **OpenAI Whisper**: [GitHub](https://github.com/openai/whisper)
- **Ultralytics YOLO**: [官方文件](https://docs.ultralytics.com/)
- **YOLOv8 Head Detector**: [GitHub](https://github.com/Abcfsa/YOLOv8_head_detector) (by Abcfsa)

---

## 14. 授權聲明 (License)

本專案程式碼供**學術研究與教育用途**。

### 開源模型致謝

本專案之視覺偵測模組深度受益於開源社群的貢獻，在此特別感謝下列的開源專案與模型：
- **YOLOv8_head_detector** (`https://github.com/Abcfsa/YOLOv8_head_detector`): 本系統之頭部偵測核心技術與預訓練權重源自開發者 **Abcfsa** 的開源成果。該模型的優異性能顯著提升了本平台在人臉/頭部偵測階段（Stage 1）的強健性與辨識效率，在此致以誠摯的感謝。

- **ETH-XGaze 預訓練模型**：遵循 ETH-XGaze 原始授權條款（Xucong Zhang et al., ECCV 2020）
- **MediaPipe**：Apache License 2.0
- **Ultralytics YOLO**：AGPL-3.0 License
- **OpenAI Whisper**：MIT License

> 本系統**不取代醫療診斷**，僅供輔助研究分析使用。臨床應用請諮詢專業醫療人員。
---

*本文件最後更新：2026 年 7 月 11 日*
