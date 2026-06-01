# CSIE_project
專題
成員名單：B1109240劉立綸、B1229011李瑋恆、B1229056黃昱翔、B1229059黃璿軒
---
# 兒童注意力監測與量化分析平台
# Attention Assessment System for ASD and TD Children

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Research%20Only-lightgrey.svg)]()
[![Status](https://img.shields.io/badge/Status-In%20Development-yellow.svg)]()
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Hands%20%7C%20FaceMesh-green.svg)](https://developers.google.com/mediapipe)
[![YOLO](https://img.shields.io/badge/YOLO-v11-orange.svg)](https://github.com/ultralytics/ultralytics)
[![Whisper](https://img.shields.io/badge/OpenAI-Whisper%20large--v3-black.svg)](https://github.com/openai/whisper)

> ⚠️ **開發狀態說明**：本專案目前包含三個獨立子模組（`eye_tracking`、`point`、`speech recognition`），各模組功能已獨立完成，**尚未完整整合為單一系統**。若要測試各模組功能，請依照各子模組的說明**分別獨立測試**，詳見 [模組測試指引](#7-模組測試指引-module-testing-guide)。

---

## 目錄 (Table of Contents)

- [系統簡介](#1-系統簡介-system-overview)
- [研究背景](#2-研究背景-research-background)
- [系統架構](#3-系統架構-system-architecture)
- [專案結構](#4-專案結構-project-structure)
- [環境需求](#5-環境需求-requirements)
- [安裝指南](#6-安裝指南-installation)
- [模組測試指引](#7-模組測試指引-module-testing-guide)
- [輸出格式](#8-輸出格式-output-format)
- [系統限制](#9-系統限制-limitations)
- [貢獻者](#10-貢獻者-contributors)
- [引用與參考](#11-引用與參考-citation--references)
- [授權聲明](#12-授權聲明-license)

---

## 1. 系統簡介 (System Overview)

本系統為一套**兒童注意力監測與量化分析平台**，專注於自閉症類群障礙（ASD）與典型發展（TD）兒童之**共同注意力（Joint Attention）能力評估**，特別針對：

- **反應性共同注意力（RJA, Responding to Joint Attention）**

系統透過多模態 AI 技術（視覺 + 語音 + 幾何推論），自動判定：

> 兒童是否在指定時間內將注意力（視線或手勢）導向目標物，並產出可供臨床與研究使用的**時間序列事件與統計報表**。

### 核心目標

| 目標 | 說明 |
|------|------|
| 量化評估 | 將「注意力」轉換為可計算指標，提供統計分析基礎 |
| 完整紀錄 | 視線軌跡、手勢向量、語音觸發時間點 |
| 客觀再現 | 降低人工觀察造成的主觀誤差，建立可重複的評估機制 |
| 臨床輔助 | 提供臨床決策支援（CDSS），**不取代醫療診斷** |

---

## 2. 研究背景 (Research Background)

### 2.1 共同注意力與 ASD

共同注意力（Joint Attention）是幼兒社交認知發展的核心能力，指個體能與他人共同關注同一物體或事件的能力。ASD 兒童在此能力上常有顯著缺損，是早期篩查的重要指標之一。

### 2.2 現有評估方法的局限性

- 傳統評估高度依賴專業人員的主觀觀察
- 人工編碼費時費力，難以大規模推廣
- 缺乏客觀、可重複的量化指標

### 2.3 本系統的創新點

1. **多模態融合**：結合視覺（視線/手勢）與語音訊號，實現條件式觸發分析
2. **時序分離架構**：語音先處理、視覺後推論，降低記憶體峰值
3. **向量幾何判定**：以射線投射（Ray Casting）取代主觀判斷，實現客觀的注意力判定
4. **動態時間窗機制**：語音關鍵字觸發後自動開啟 3 秒高靈敏判定視窗

---

## 3. 系統架構 (System Architecture)

系統採用**三層式架構（Three-Layer Architecture）**，並由三個獨立子模組協同運作：

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Layer 1：資料層                               │
│  影片輸入 (.mp4) ──► 影像讀取  │  音訊抽取  │  影像前處理            │
└─────────────────────────────────────────────────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       Layer 2：AI 推論層                              │
│                                                                       │
│  ┌─────────────────┐  ┌──────────────────┐  ┌────────────────────┐  │
│  │  eye_tracking   │  │      point       │  │ speech recognition │  │
│  │                 │  │                  │  │                    │  │
│  │ YOLO nano       │  │ YOLOv11-Pose     │  │ Whisper large-v3   │  │
│  │ MediaPipe       │  │ MediaPipe Hands  │  │ 關鍵字偵測          │  │
│  │ ETH-XGaze       │  │ OpenCV 樣板比對  │  │ 時間窗建立          │  │
│  │ ResNet-50       │  │ Whisper          │  │                    │  │
│  └─────────────────┘  └──────────────────┘  └────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       Layer 3：分析層                                 │
│  Greedy Bipartite Matching  │  Ray Casting  │  注意力事件量化        │
└─────────────────────────────────────────────────────────────────────┘
```

### AI 模型技術總覽

| 子模組 | 功能 | 核心模型 |
|--------|------|----------|
| `eye_tracking` | 視線方向估計 | ETH-XGaze (ResNet-50) |
| `eye_tracking` | 人臉偵測 | YOLO nano + MediaPipe Face Landmarker |
| `eye_tracking` | 頭部姿態估計 | OpenCV solvePnP |
| `point` | 骨架/手勢偵測 | YOLOv11-Pose + MediaPipe Hands |
| `point` | 物件偵測 | 客製化 YOLO 模型群 |
| `speech recognition` | 語音轉文字 | OpenAI Whisper large-v3 |

---

## 4. 專案結構 (Project Structure)

```
Attention-Assessment-System/
│
├── README.md                          # 本文件
│
├── eye_tracking/                      # 子模組一：視線估計系統
│   ├── README.md                      # 子模組詳細說明
│   ├── config.yaml                    # 系統配置文件
│   ├── requirements.txt               # Python 依賴套件
│   ├── process_video.py               # 影片處理主程式（推薦）
│   ├── test_gaze_arrow.py             # 單張圖片 / Webcam 測試
│   ├── stages/                        # 五階段處理流程
│   │   ├── stage1_face_detection.py   # Stage 1: 人臉偵測
│   │   ├── stage2_head_pose.py        # Stage 2: 頭部姿態估計
│   │   ├── stage3_normalization.py    # Stage 3: 影像正規化
│   │   ├── stage4_gaze_network.py     # Stage 4: ResNet-50 推理
│   │   └── stage5_gaze_vector.py      # Stage 5: 視線向量轉換
│   ├── utils/                         # 工具函數
│   │   ├── visualization.py
│   │   └── camera_utils.py
│   └── models/                        # ⚠️ 需從 Releases 下載
│       ├── nano.pt
│       ├── face_landmarker.task
│       ├── epoch_24_ckpt.pth.tar
│       └── face_model_ethxgaze.txt
│
├── point/                             # 子模組二：手勢與互動分析
│   ├── readme.md                      # 子模組詳細說明
│   ├── environment.yml                # Conda 虛擬環境配置
│   ├── project.py                     # 系統主程式
│   ├── ffmpeg.exe                     # 影音縫合工具（Windows）
│   ├── model/                         # ⚠️ YOLO 客製化模型
│   │   ├── front_model.pt
│   │   ├── background_model.pt
│   │   ├── balloon_model.pt
│   │   ├── bubble_model.pt
│   │   ├── toy_model.pt
│   │   └── robot_point_model.pt
│   ├── sample/                        # 測驗字卡樣板圖 (1.jpg ~ 8.jpg)
│   ├── video/                         # 輸入影片目錄
│   └── output/                        # 輸出結果目錄
│
├── speech recognition/                # 子模組三：語音辨識與觸發
│   ├── AUDIO_SETUP.md                 # 安裝說明（macOS）
│   ├── audio_trigger_pipeline.py      # 語音分析主程式
│   ├── requirements-audio.txt         # Python 依賴套件
│   └── setup_audio_env.sh             # 自動安裝腳本
│
└── doc/                               # 設計文件與報告
    └── B1109240_設計文件書.pdf
```

---

## 5. 環境需求 (Requirements)

> ⚠️ 因三個子模組開發環境不同，**建議為每個子模組建立獨立的虛擬環境**，避免套件版本衝突。

### eye_tracking 模組

| 項目 | 需求 |
|------|------|
| Python | 3.8 – 3.12 |
| 作業系統 | macOS / Linux / Windows |
| RAM | ≥ 2 GB |
| GPU | 可選（CPU 亦可，約 10–15 FPS） |

**核心套件**：`opencv-contrib-python >= 4.13.0`、`ultralytics >= 8.0.0`、`mediapipe >= 0.10.32`、`torch >= 2.0.0`、`torchvision >= 0.15.0`

### point 模組

| 項目 | 需求 |
|------|------|
| Python | 3.9 |
| 作業系統 | Windows（主要開發環境） |
| 套件管理 | Anaconda / Miniconda |

**核心套件**：`ultralytics 8.4.19`、`mediapipe 0.10.21`、`openai-whisper`、`moviepy`、`torch 2.8.0`、`easyocr`

### speech recognition 模組

| 項目 | 需求 |
|------|------|
| Python | Homebrew Python 3（macOS）|
| 作業系統 | macOS（主要開發環境）|
| 系統工具 | `ffmpeg`（需透過 Homebrew 安裝）|

**核心套件**：`openai-whisper`、`torch`、`torchaudio`

---

## 6. 安裝指南 (Installation)

### 6.1 取得專案

```bash
git clone https://github.com/Aliulilun/Attention-Assessment-System.git
cd Attention-Assessment-System
```

### 6.2 下載預訓練模型

前往 [GitHub Releases](https://github.com/Aliulilun/Attention-Assessment-System/releases) 下載 **model-release**，取得以下模型並放至對應目錄：

**eye_tracking/models/**（約 133 MB）：
- `nano.pt`（約 6 MB）— YOLO nano 頭部偵測模型
- `face_landmarker.task`（約 25 MB）— MediaPipe 特徵點模型
- `epoch_24_ckpt.pth.tar`（約 102 MB）— ETH-XGaze 視線估計模型

**point/model/**：
- 各 YOLO 客製化模型（`front_model.pt`、`robot_point_model.pt` 等）

### 6.3 建立各模組環境

**eye_tracking 模組：**
```bash
cd eye_tracking
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

**point 模組（Windows + Conda）：**
```bash
cd point
conda env create -f environment.yml
conda activate mediapipe_py39
```

**speech recognition 模組（macOS）：**
```bash
brew install python3 ffmpeg
cd "speech recognition"
chmod +x setup_audio_env.sh
./setup_audio_env.sh
```
> 若遇到 `pyexpat` 載入問題，執行前請先補上：
> ```bash
> export DYLD_LIBRARY_PATH=/opt/homebrew/opt/expat/lib
> ```

---

## 7. 模組測試指引 (Module Testing Guide)

> ⚠️ **重要提示**：目前三個子模組（`eye_tracking`、`point`、`speech recognition`）**尚未完整整合為單一系統**，各模組目前為獨立開發的 Side Project。若要測試，請依照以下說明**個別進入各子模組目錄進行測試**。 近期會盡快將各個Side Project整合再一起。且由於目前研究影片有個資問題，因此尚未上傳可測試影片檔，若有測試影片需求，目前應自行上傳測試影片，敬請見諒，本團隊日後將快速補上可測試影片。

---

###  eye_tracking — 視線估計模組

```bash
cd eye_tracking
source .venv/bin/activate

# 處理影片並輸出 CSV 數據（推薦）
python process_video.py --input your_video.mp4 --output output.mp4 --csv gaze_data.csv

# 僅輸出 CSV（速度較快）
python process_video.py --input your_video.mp4 --csv gaze_data.csv

# 使用 Webcam 即時測試
python test_gaze_arrow.py --mode webcam

# 測試單張圖片
python test_gaze_arrow.py --mode image --image test_images/your_image.jpg

# 各階段獨立測試
python test_stage1.py   # 人臉偵測
python test_stage2.py   # 頭部姿態
python test_stage3.py   # 影像正規化
python test_stage4.py   # 視線推理
```

---

###  point — 手勢互動分析模組

```bash
cd point
conda activate mediapipe_py39

# 將待分析影片放至 video/ 目錄後執行
python project.py
```

執行流程：
1. 系統自動解析語音，建立逐字稿快取（首次執行需下載 Whisper 模型）
2. 畫面暫停於第一幀，請用滑鼠**框選「階段 1 字卡」**後按 `Enter`
3. 系統進入全幀視覺分析主迴圈
4. 分析完成後輸出於 `output/output_with_audio.mp4`

---

###  speech recognition — 語音辨識模組

```bash
cd "speech recognition"
export DYLD_LIBRARY_PATH=/opt/homebrew/opt/expat/lib   # macOS 需要
source .venv314/bin/activate

# 分析指定影片的音軌
python audio_trigger_pipeline.py --video ./video/8.mp4

# 強制重新執行 Whisper（忽略快取）
python audio_trigger_pipeline.py --video ./video/8.mp4 --force

# 查看說明
python audio_trigger_pipeline.py --help
```

---

## 8. 輸出格式 (Output Format)

### eye_tracking 模組 — CSV 輸出欄位

| 欄位名稱 | 單位 | 說明 |
|----------|------|------|
| `frame_idx` | — | 幀索引（從 0 開始）|
| `timestamp_sec` | 秒 | 時間戳 |
| `head_pitch_deg` | 度 | 頭部俯仰角 |
| `head_yaw_deg` | 度 | 頭部偏航角 |
| `head_roll_deg` | 度 | 頭部翻滾角 |
| `gaze_pitch_deg` | 度 | 視線俯仰角 |
| `gaze_yaw_deg` | 度 | 視線偏航角 |
| `gaze_vector_x/y/z` | — | 3D 視線單位向量 |
| `face_bbox_x/y/w/h` | 像素 | 人臉框位置與大小 |

### point 模組 — 輸出影片

- **帶標注影片**：`output/output_with_audio.mp4`
  - 骨架框線、手勢向量、射線軌跡、碰撞特效、注意力事件標記

### speech recognition 模組 — 輸出

- **逐字稿快取**（`.txt`）：含關鍵字時間戳
- **時間窗事件清單**：觸發關鍵字對應的 3 秒判定視窗列表

---

## 9. 系統限制 (Limitations)

### 視覺模組（eye_tracking）
- 光線需穩定，避免逆光環境
- 受試者建議在相機 1.5 公尺內
- 頭部姿態超過 ±60° 時準確度下降
- 理論視線估計誤差（ETH-XGaze）：跨資料集約 5°–7°

### 視覺模組（point）
- 目前開發環境以 Windows 為主，macOS / Linux 需自行調整路徑設定
- 多人場景下手部身分識別依賴「手臂距離分數演算法」，極端遮擋情況下可能誤判

### 語音模組（speech recognition）
- 噪音干擾、疊音會降低 Whisper 辨識準確度
- macOS 環境下需額外設定 `DYLD_LIBRARY_PATH`

### 整合限制
- ⚠️ 三個子模組目前**尚未整合為統一執行流程**，需分別安裝環境與測試

---

## 10. 貢獻者 (Contributors)

| 學號 | 姓名 | 主要貢獻 |
|------|------|----------|
| B1109240 | 劉立綸 | 視線估計模組（eye_tracking）|
| B1229059 | 黃璿軒 | 手勢互動分析模組（point）|
| B1229011 | 李瑋恆 | 相關yolo模型|
| B1229056 | 黃昱翔 | 協助語音辨識模組（speech recognition）|


---

## 11. 引用與參考 (Citation & References)

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
```

### 參考技術文件

- **ETH-XGaze**: [GitHub](https://github.com/xucong-zhang/ETH-XGaze) | [arXiv](https://arxiv.org/abs/2007.15837)
- **Gaze Normalization (CVPR 2015)**: Zhang et al., *Appearance-Based Gaze Estimation in the Wild*
- **MediaPipe Face Mesh**: [官方文件](https://developers.google.com/mediapipe/solutions/vision/face_landmarker)
- **OpenCV solvePnP**: [官方文件](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)
- **OpenAI Whisper**: [GitHub](https://github.com/openai/whisper)
- **Ultralytics YOLO**: [官方文件](https://docs.ultralytics.com/)

---

## 12. 授權聲明 (License)

本專案程式碼供**學術研究與教育用途**。

- **ETH-XGaze 預訓練模型**：遵循 ETH-XGaze 原始授權條款（Xucong Zhang et al., ECCV 2020）
- **MediaPipe**：Apache License 2.0
- **Ultralytics YOLO**：AGPL-3.0 License
- **OpenAI Whisper**：MIT License

> 本系統**不取代醫療診斷**，僅供輔助研究分析使用。臨床應用請諮詢專業醫療人員。

---

*最後更新：2026 年 6 月*
