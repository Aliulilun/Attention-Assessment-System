# 視線估計整合更新日誌
# Gaze Estimation Integration Changelog

**更新日期**：2026 年 6 月 14 日  
**專案**：Attention Assessment System - Integrate Module  
**版本**：v1.0

---

## 📋 目錄

1. [更新概述](#更新概述)
2. [新增功能](#新增功能)
3. [檔案修改清單](#檔案修改清單)
4. [技術細節](#技術細節)
5. [使用說明](#使用說明)
6. [測試方法](#測試方法)

---

## 🎯 更新概述

本次更新將 `eye_tracking` 模組成功整合至 `integrate` 專案，實現了**多模態注意力監測系統**，結合以下三種模態：

| 模態 | 技術 | 功能 |
|------|------|------|
| 🗣️ **語音** | Whisper | 語音觸發與關鍵字偵測 |
| 👉 **手勢** | MediaPipe Hands + YOLO Pose | 手勢指向判定 (Ray Casting) |
| 👀 **視線** | ETH-XGaze + MediaPipe Face | 視線注視判定 (Ray Casting) |

### 核心目標

✅ 實現視線向量在影片畫面上的即時可視化  
✅ 整合 Ray Casting 演算法判定視線是否落在目標物體上  
✅ 統一「手勢指向」與「視線注視」兩種注意力模式的事件記錄  
✅ 修復 MediaPipe API 版本不相容問題

---

## 🆕 新增功能

### 1. 視線向量可視化

在影片處理過程中，系統現在會在每一幀上繪製：

- ✅ **綠色人臉框**：標示偵測到的人臉位置
- ✅ **紅色視線箭頭**：從雙眼發出，指向注視方向
- ✅ **視線角度數值**：顯示 Pitch（俯仰角）與 Yaw（偏航角）
- ✅ **3D 視線向量**：顯示 `[x, y, z]` 分量
- ✅ **人臉檢測置信度**：顯示 YOLO 頭部偵測的信心值

**範例輸出畫面**：
```
┌─────────────────────────────┐
│  Conf: 0.85                 │ ← 置信度
│  P: -5.3° Y: 12.7°          │ ← 角度
│  Vec: [0.22, -0.09, 0.97]   │ ← 3D 向量
├─────────────────────────────┤
│     ┌──────┐                │
│     │ 臉部 │ ← 綠色框       │
│     └──────┘                │
│      ↗️                     │ ← 紅色箭頭（視線方向）
└─────────────────────────────┘
```

---

### 2. 視線注視判定 (Gaze Ray Casting)

系統現在能夠判定**視線是否落在目標物體上**（氣球、玩偶、玩具等），使用與手勢指向相同的 **Ray-AABB 交集演算法**。

#### 判定邏輯

```
1. 視線起點 = (左眼中心 + 右眼中心) / 2
2. 視線方向 = gaze_vector 的 2D 投影 (x, y)
3. 目標物體 = YOLO 偵測到的邊界框 (x1, y1, x2, y2)
4. 判定結果 = ray_intersects_box(起點, 方向, 邊界框)
```

#### 可視化標記

當視線落在物體上時：

- 🟡 **黃色粗框** (5px)：高亮顯示正在注視的物體
- 🟡 **"GAZING!" 標記**：在物體上方顯示注視狀態
- 📊 **UI 面板更新**：`Child Gazing At Object: YES!`

---

### 3. 事件記錄擴充

事件記錄檔 (`event_record.txt`) 現在會記錄**視線注視事件**：

```
=== 互動行為分析事件紀錄表 ===
影片來源: video/test.mp4
--------------------------------
[5.2s] 觸發：偵測到引導語音關鍵字
[6.8s] 階段改變：進入 第 5 階段
[7.3s] 視線：正在注視階段 5 物體          ← 🆕 新增
[8.1s] 互動：小朋友成功指向當下場景的物品 (Stage 5)
[12.5s] 視線：正在注視階段 5 物體         ← 🆕 新增
```

---

### 4. UI 資訊面板擴充

左上角即時預覽面板新增一行：

```
┌────────────────────────────────────┐
│ Time:  12.5 s                      │
│ Stage: 5                           │
│ Keyword Detected: YES (Active)     │
│ Child Pointing Hit: NO             │
│ Gaze: P=-5.3 Y=12.7               │
│ Child Gazing At Object: YES!       │ ← 🆕 新增
└────────────────────────────────────┘
```

---

## 📁 檔案修改清單

### 1. `/integrate/main.py` ⭐

**新增**：
- 第 21-95 行：Ray Casting 相關函數
  - `ray_intersects_box()` - 射線-邊界框交集演算法
  - `is_gazing_at_box()` - 判斷視線是否落在單個物體上
  - `check_gaze_on_objects()` - 檢查視線是否落在任何物體上

**修改**：
- 第 15 行：導入視線可視化模組 `draw_gaze_with_face_box`
- 第 209 行：新增 `prev_gaze_state` 狀態追蹤變數
- 第 233 行：儲存 `yolo_boxes` 供視線判定使用
- 第 247-271 行：視線注視判定與物體高亮標記
- 第 287-289 行：視線注視事件記錄
- 第 310 行：UI 面板新增視線注視狀態顯示

**移除**：
- 註解掉 `import pandas as pd`（暫時不輸出 CSV）
- 註解掉視線數據儲存與統計相關代碼

---

### 2. `/integrate/modules/interaction.py` 🔧

**重大更新：MediaPipe API 遷移**

**問題**：
舊版 `mp.solutions.hands.Hands()` 在新版 MediaPipe (≥0.10.32) 中已被移除，導致 `AttributeError: module 'mediapipe' has no attribute 'solutions'`。

**解決方案**：
將整個 `InteractionEngine` 類別從舊 API 遷移到新的 **MediaPipe Tasks API**。

**修改細節**：

| 項目 | 舊版 API | 新版 API |
|------|---------|----------|
| Import | `mp.solutions.hands` | `mediapipe.tasks.python.vision.HandLandmarker` |
| 初始化 | `mp.solutions.hands.Hands()` | `HandLandmarker.create_from_options()` |
| 模型檔案 | 無需手動下載 | `hand_landmarker.task` (自動下載) |
| 偵測方法 | `.process(rgb_frame)` | `.detect(mp_image)` |
| 結果結構 | `.multi_hand_landmarks` | `.hand_landmarks` |
| Landmark 存取 | `.landmark[i]` | `[i]` (直接索引) |

**新增**：
- 第 8 行：導入新 API 相關模組
- 第 38-57 行：`_get_hand_model_path()` 方法 - 自動下載 `hand_landmarker.task`
- 第 23-37 行：使用新 API 初始化手部偵測器

**修改**：
- 第 66-72 行：`is_valid_pointing()` - 更新 landmark 存取方式
- 第 224-229 行：`analyze_interaction()` - 更新影像處理與偵測邏輯

---

### 3. `/integrate/modules/gaze_estimation/gaze_pipeline.py` 👁️

**新增**：
- 第 129-132 行：計算左右眼中心位置
  ```python
  # ETH-XGaze 6點: [33, 133, 362, 263, 61, 291]
  # 索引: [右眼外, 右眼內, 左眼外, 左眼內, 左嘴角, 右嘴角]
  right_eye_center = ((landmarks[0] + landmarks[1]) / 2).astype(int)
  left_eye_center = ((landmarks[2] + landmarks[3]) / 2).astype(int)
  ```

**修改**：
- 第 138-139 行：返回結果新增 `left_eye` 和 `right_eye` 欄位
  ```python
  'left_eye': tuple(left_eye_center),
  'right_eye': tuple(right_eye_center)
  ```

---

### 4. `/integrate/modules/gaze_estimation/visualization.py` 🎨

**無修改**

此檔案的 `draw_gaze_with_face_box()` 函數已支援所有需要的功能：
- ✅ 視線箭頭繪製
- ✅ 角度與向量顯示
- ✅ 人臉框繪製
- ✅ 雙眼視線箭頭

---

## 🔧 技術細節

### Ray Casting 演算法

本系統採用 **Ray-AABB (Axis-Aligned Bounding Box) 交集演算法**，這是一種高效的 2D 射線-矩形碰撞檢測方法。

#### 數學原理

給定：
- 射線起點 `O = (ox, oy)`
- 射線方向 `D = (dx, dy)`
- 矩形邊界框 `B = (x1, y1, x2, y2)`

計算：
```
tx1 = (x1 - ox) / dx
tx2 = (x2 - ox) / dx
ty1 = (y1 - oy) / dy
ty2 = (y2 - oy) / dy

tmin = max(min(tx1, tx2), min(ty1, ty2))
tmax = min(max(tx1, tx2), max(ty1, ty2))

有交集 ⇔ tmax ≥ max(0, tmin)
```

#### 特殊處理

1. **避免除以零**：當 `dx` 或 `dy` 為 0 時，設為極小值 `1e-5`
2. **方向性檢查**：`tmax ≥ 0` 確保射線是往前延伸，而非往後
3. **數值穩定性**：使用浮點數運算避免整數截斷誤差

---

### MediaPipe Tasks API 架構

```
新版 API 層級結構：
mediapipe
  └── tasks
       └── python
            ├── BaseOptions
            └── vision
                 ├── HandLandmarker
                 ├── HandLandmarkerOptions
                 └── RunningMode
```

**關鍵配置**：
```python
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=RunningMode.IMAGE,  # 影片逐幀處理模式
    num_hands=4,                      # 最多偵測 4 隻手
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5
)
```

---

## 📖 使用說明

### 執行指令

```bash
cd /Users/liulilun/Documents/Attention-Assessment-System/integrate
python3 main.py
```

### 輸入要求

1. **影片檔案**：放置於 `config.yaml` 指定的路徑
   ```yaml
   video:
     input_path: 'video/10.mp4'
   ```

2. **模型檔案**：確保以下模型已放置於 `model/gaze/` 目錄
   - `nano.pt` (YOLO 頭部偵測)
   - `face_landmarker.task` (MediaPipe 臉部特徵點)
   - `epoch_24_ckpt.pth.tar` (ETH-XGaze 視線估計)
   - `face_model_ethxgaze.txt` (3D 人臉模型)
   - `hand_landmarker.task` (MediaPipe 手部特徵點，**自動下載**)

### 輸出檔案

執行完成後，會在 `output/` 目錄生成：

1. **標注影片**：`output_result_final.mp4`
   - 包含所有可視化標記（人臉框、視線箭頭、物體高亮等）

2. **事件記錄**：`event_record.txt`
   - 記錄所有關鍵事件（語音觸發、階段變更、指向命中、視線注視）

---

## 🧪 測試方法

### 預期行為

當系統正常運作時，您應該觀察到：

#### 1. 視窗顯示

- ✅ 預覽視窗名稱：`Multi-Modal AI System Preview`
- ✅ 視窗尺寸：1280 x 720 (可調整)
- ✅ 左上角 UI 面板顯示 6 行資訊

#### 2. 視線可視化

- ✅ 綠色人臉框清晰可見
- ✅ 紅色視線箭頭從雙眼發出
- ✅ 角度數值合理 (Pitch: -90° ~ 90°, Yaw: -90° ~ 90°)

#### 3. 注視判定

- ✅ 當視線對準物體時，物體出現黃色粗框
- ✅ UI 面板顯示 `Child Gazing At Object: YES!`
- ✅ 終端輸出事件記錄

#### 4. 效能指標

| 環境 | 預期 FPS | 處理時間 (1 分鐘影片) |
|------|----------|---------------------|
| 有 GPU (CUDA) | 5-15 FPS | 約 2-6 分鐘 |
| 無 GPU (CPU) | 0.5-2 FPS | 約 15-60 分鐘 |

### 常見問題排查

#### 問題 1：`AttributeError: module 'mediapipe' has no attribute 'solutions'`

**原因**：MediaPipe 版本過舊或 `interaction.py` 未更新  
**解決**：確認 `interaction.py` 已更新為新版 API（本次更新已完成）

#### 問題 2：視線箭頭方向錯誤

**原因**：相機內參矩陣不準確  
**解決**：調整 `config.yaml` 中的 `focal_length` 參數

#### 問題 3：視線注視判定過於敏感/遲鈍

**原因**：Ray Casting 閾值需調整  
**解決**：可在 `is_gazing_at_box()` 中加入角度閾值過濾

---

## 📊 系統架構圖

```
┌─────────────────────────────────────────────────────────────────┐
│                         integrate/main.py                       │
│                       (主控制流程)                                │
└────────────┬────────────────────────────────────────────────────┘
             │
    ┌────────┼────────┬─────────────┬──────────────┐
    │        │        │             │              │
    ▼        ▼        ▼             ▼              ▼
┌──────┐ ┌──────┐ ┌────────┐ ┌───────────┐ ┌─────────────┐
│Speech│ │Sign- │ │Models  │ │Interaction│ │GazeEstimation│
│Trigger│ │board │ │Manager │ │Engine     │ │Pipeline     │
└──────┘ └──────┘ └────────┘ └───────────┘ └─────────────┘
   │        │         │            │              │
   ▼        ▼         ▼            ▼              ▼
Whisper  EasyOCR   YOLO      MediaPipe       ETH-XGaze
                   Models     Hands         + MediaPipe
                                             Face Mesh
```

---

## 🎯 下一步規劃

### 短期目標

- [ ] 優化 Ray Casting 演算法（加入角度閾值過濾）
- [ ] 支援 CSV 格式的視線數據輸出（可選）
- [ ] 加入多人場景下的視線追蹤

### 中期目標

- [ ] 整合注意力持續時間統計
- [ ] 實現視線熱力圖 (Heatmap) 生成
- [ ] 加入物體注視時長分析

### 長期目標

- [ ] 建立注意力評估量化指標體系
- [ ] 整合臨床決策支援系統 (CDSS)
- [ ] 開發 Web 介面進行遠端分析


---

## 📚 參考資料

### 學術論文

1. **ETH-XGaze**: Zhang et al., "ETH-XGaze: A Large Scale Dataset for Gaze Estimation under Extreme Head Poses and Gaze Directions", ECCV 2020
2. **Gaze Normalization**: Zhang et al., "Appearance-Based Gaze Estimation in the Wild", CVPR 2015

### 技術文件

- [MediaPipe Tasks Python API](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker/python)
- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
- [OpenCV solvePnP](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga549c2075fac14829ff4a58bc931c033d)

---

**最後更新**：2026 年 6 月 14 日  
**版本**：v1.0  
**專案**：Attention Assessment System - Integrate Module
