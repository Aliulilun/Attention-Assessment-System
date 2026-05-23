# 視線估計系統 (Gaze Estimation System)

基於 ETH-XGaze 預訓練模型的完整視線估計流程實現。

## 專案結構

```
eye_tracking/
├── README.md                           # 專案說明文件
├── config.yaml                         # 系統配置文件
├── requirements.txt                    # Python 依賴套件
│
├── process_video.py                   # 影片處理主程式（推薦）
├── test_gaze_arrow.py                 # 單張圖片/Webcam 測試程式
├── my_gaze_estimation.py              # 整合類別（可作為庫使用）
│
├── stages/                            # 五個階段的實現
│   ├── __init__.py
│   ├── stage1_face_detection.py      # Stage 1: 人臉檢測（YOLO + MediaPipe Tasks）
│   ├── stage2_head_pose.py           # Stage 2: 頭部姿態估計（solvePnP）
│   ├── stage3_normalization.py       # Stage 3: 圖像正規化（ETH-XGaze 標準）
│   ├── stage4_gaze_network.py        # Stage 4: 神經網絡推理（ResNet-50）
│   └── stage5_gaze_vector.py         # Stage 5: 視線向量轉換
│
├── utils/                             # 工具函數
│   ├── __init__.py
│   ├── visualization.py              # 可視化工具（視線箭頭、人臉框等）
│   └── camera_utils.py               # 相機參數處理
│
├── models/                            # 模型文件
│   ├── epoch_24_ckpt.pth.tar         # ETH-XGaze 預訓練模型（ResNet-50）
│   └── face_model_ethxgaze.txt       # 3D 人臉模型（6 點，單位：cm）
│
├── test_images/                       # 測試圖片
│   └── (放置你的測試圖片)
│
├── test_stage*.py                     # 各階段的獨立測試腳本
│
└── 文檔/                              # 專案文檔
```

## 五個階段說明

### Stage 1: 人臉檢測與特徵點定位 (Face Detection)

**實現文件**: `stages/stage1_face_detection.py`

- **架構**: YOLO + MediaPipe Tasks API 兩階段級聯
- **使用工具**: 
  - **YOLO (Ultralytics nano.pt)**: 初步頭部定位
  - **MediaPipe Face Landmarker (v0.10.32)**: 精確特徵點提取
- **三階段檢測流程**:
  1. **中央空間先驗裁切**（可選，針對高解析度大圖）
     - 自動裁切畫面中央區域（默認 1600×960）
     - 提高小目標檢測率，降低運算開銷
  2. **YOLO 頭部檢測**
     - 快速定位頭部邊界框
     - 提取檢測置信度
  3. **MediaPipe 特徵點回歸**
     - 在 YOLO ROI 內進行高精度特徵點提取
     - 提取 **468 個 3D 面部特徵點**
     - 仿射平移變換：將局部座標映射回全域座標系
- **關鍵點選擇** (基於 Zhang et al., 2018 ETRA):
  ```
  - 左眼外角 (index 33)
  - 左眼內角 (index 133)
  - 右眼內角 (index 362)
  - 右眼外角 (index 263)
  - 左嘴角 (index 61)
  - 右嘴角 (index 291)
  ```
- **配置參數**:
  - `yolo_model_path`: YOLO 模型路徑（默認 `models/nano.pt`）
  - `yolo_conf`: YOLO 檢測閾值（默認 0.4）
  - `face_landmarker_task`: MediaPipe 模型路徑（默認 `models/face_landmarker.task`）
  - `min_confidence`: MediaPipe 最小檢測置信度（默認 0.5）
  - `crop_padding_ratio`: ROI 動態填充比例（默認 0.2）
  - `target_center_w` / `target_center_h`: 中央裁切區域尺寸（默認 1600×960）
- **輸出數據**:
  - `bbox`: 人臉邊界框 `[x_min, y_min, x_max, y_max]`（xyxy 格式，基於特徵點計算）
  - `yolo_head_bbox`: YOLO 原始頭部框（供視覺化參考）
  - `landmarks_468`: 完整的 468 個特徵點 `(468, 2)`（全域座標）
  - `landmarks_2d_selected`: 選定的 6 個關鍵點 `(6, 2)`（全域座標）
  - `confidence`: YOLO 檢測置信度 (0.0-1.0)
  - `num_landmarks`: 特徵點總數（468）

---

### Stage 2: 頭部姿態估計 (Head Pose Estimation)

**實現文件**: `stages/stage2_head_pose.py`

- **使用工具**: OpenCV `solvePnP` (SOLVEPNP_ITERATIVE)
- **核心功能**:
  - 使用 **PnP (Perspective-n-Point)** 算法求解 6DoF 頭部姿態
  - 從 2D 圖像點和 3D 模型點計算頭部的旋轉和平移
- **3D 人臉模型**: `models/face_model_ethxgaze.txt`
  - 6 個關鍵點的 3D 座標（單位：**cm**）
  - 來源：MediaPipe Canonical Face Model
  - 座標系：X 向右，Y 向上，Z 向前
- **輸出數據**:
  - `rvec`: 旋轉向量 (3,) - Rodrigues 表示
  - `tvec`: 平移向量 (3,) - 人臉中心在相機座標系中的位置（單位：cm）
  - `rotation_matrix`: 旋轉矩陣 `R_head` (3, 3)
  - `euler_angles`: 歐拉角（度）
    - `pitch`: 俯仰角（向上為正）
    - `yaw`: 偏航角（向右為正）
    - `roll`: 翻滾角（順時針為正）
  - `distance`: 人臉到相機的距離（單位：cm）
- **注意事項**:
  - Stage 2 專注於「頭部姿態」估計，不涉及視線方向
  - 輸出的 `R_head` 和 `tvec` 會被 Stage 3 用於正規化

---

### Stage 3: 圖像正規化 (Image Normalization)

**實現文件**: `stages/stage3_normalization.py`

- **使用工具**: OpenCV `warpPerspective` + 自定義旋轉矩陣構建
- **核心功能**:
  - 將人臉圖像轉換到「虛擬正面相機」視角
  - 消除頭部旋轉和距離變化的影響
  - **對齊 ETH-XGaze 官方標準**（參考 `demo.py`）
- **正規化座標系構建** (關鍵實現):
  ```python
  # Z 軸：從相機指向人臉中心（tvec 方向）
  forward = tvec / ||tvec||
  
  # Y 軸：Z 軸 × 頭部 X 軸（確保眼睛水平線保持水平）
  down = cross(head_X_axis, forward)
  
  # X 軸：Y 軸 × Z 軸（右手座標系）
  right = cross(down, forward)
  
  # 正規化旋轉矩陣 R_norm
  R_norm = [right, down, forward]  # 行向量形式
  ```
- **正規化參數** (ETH-XGaze 標準):
  - `focal_norm`: 960.0（虛擬相機焦距）
  - `distance_norm`: 60.0 cm（虛擬相機到人臉的距離）
  - `output_size`: (224, 224)（神經網絡輸入尺寸）
- **變換流程**:
  1. 計算縮放矩陣 `S`（將人臉移動到固定距離）
  2. 計算單應性矩陣 `W = K_norm @ S @ R_norm @ R_head^T @ K_orig^(-1)`
  3. 使用 `warpPerspective` 得到正規化圖像
- **輸出數據**:
  - `normalized_image`: 正規化後的圖像 (224, 224, 3) RGB 格式
  - `warp_matrix`: 單應性矩陣 `W` (3, 3)
  - `head_rot_norm`: 正規化旋轉矩陣 `R_norm` (3, 3)
  - `face_center_distance`: 人臉到相機的實際距離（cm）
  - `scale_factor`: 縮放係數

---

### Stage 4: 神經網絡推理 (Gaze Network)

**實現文件**: `stages/stage4_gaze_network.py`

- **使用模型**: ETH-XGaze 預訓練的 ResNet-50
  - 模型文件：`models/epoch_24_ckpt.pth.tar`
  - 架構：ResNet-50 backbone + 全連接層 (2048 → 2)
- **輸入要求**:
  - 圖像尺寸：224×224×3
  - 顏色格式：**RGB**（注意：OpenCV 默認是 BGR，需要轉換）
  - 正規化：ImageNet 標準
    - Mean: [0.485, 0.456, 0.406]
    - Std: [0.229, 0.224, 0.225]
- **推理流程**:
  1. 將正規化圖像從 BGR 轉為 RGB
  2. 轉換為 PyTorch tensor 並正規化
  3. 通過 ResNet-50 提取特徵 (2048 維)
  4. 全連接層輸出 2D 視線角度
- **輸出數據**:
  - `gaze_angles`: 視線角度 `[pitch, yaw]`（弧度）
  - `gaze_angles_deg`: 視線角度 `[pitch, yaw]`（度）
  - `success`: 推理是否成功
- **角度定義** (第一人稱視角):
  - **Pitch > 0°**: 向上看 (looking up)
  - **Pitch < 0°**: 向下看 (looking down)
  - **Yaw > 0°**: 向右看 (looking right)
  - **Yaw < 0°**: 向左看 (looking left)
- **注意事項**:
  - 模型輸出的是「正規化相機座標系」中的視線角度
  - 這是相對於「虛擬正面相機」的角度，不是相對於原始相機

---

### Stage 5: 視線向量轉換 (Gaze Vector Conversion)

**實現文件**: `stages/stage5_gaze_vector.py`

- **核心功能**: 將 2D 視線角度 (pitch, yaw) 轉換為 3D 單位向量
- **轉換公式** (對齊 ETH-XGaze 官方):
  ```python
  x = cos(pitch) × sin(yaw)    # 水平分量
  y = sin(pitch)                # 垂直分量
  z = cos(pitch) × cos(yaw)     # 深度分量
  
  # 歸一化
  gaze_vector = [x, y, z] / ||[x, y, z]||
  ```
- **座標系定義** (正規化後的相機座標系):
  - **X 軸**: 向右 (right)
  - **Y 軸**: 向下 (down)
  - **Z 軸**: 從相機指向人臉 (forward)
- **向量分量意義**:
  - `x > 0`: 視線偏右；`x < 0`: 視線偏左
  - `y > 0`: 視線偏下；`y < 0`: 視線偏上
  - `z > 0`: 視線朝前（遠離相機，正常情況）
- **輸出數據**:
  - `gaze_vector`: 3D 單位向量 (3,) - 長度為 1.0
  - 支援雙向轉換：
    - `angles_to_vector(pitch, yaw)` → 向量
    - `vector_to_angles(gaze_vector)` → (pitch, yaw)
- **應用場景**:
  - 計算視線與物體的交點
  - 判斷用戶在看螢幕的哪個區域
  - 視線追蹤和注意力分析

---

### 完整流程示意圖

```
原始圖像 (1920×1080)
    ↓
[Stage 1] YOLO 頭部檢測 + MediaPipe 特徵點提取
    ├─ 中央空間先驗裁切 (可選)
    ├─ YOLO 定位頭部 ROI + 置信度
    └─ MediaPipe 提取 468 特徵點 + 仿射變換回全域
    ↓
人臉框 + 6 個關鍵點 (2D) + 置信度
    ↓
[Stage 2] solvePnP 頭部姿態估計
    ↓
R_head, tvec（頭部在相機座標系）
    ↓
[Stage 3] 圖像正規化
    ↓
正規化圖像 (224×224, RGB)
    ↓
[Stage 4] ResNet-50 推理
    ↓
視線角度 (pitch, yaw)
    ↓
[Stage 5] 向量轉換
    ↓
3D 視線向量 (x, y, z)
```

---

### 關鍵參數速查

| 階段 | 關鍵參數 | 預設值 | 說明 |
|------|---------|--------|------|
| Stage 1 | `yolo_conf` | 0.4 | YOLO 檢測閾值 |
| Stage 1 | `min_confidence` | 0.5 | MediaPipe 檢測置信度（降至 0.3 可提高遠距離檢測）|
| Stage 1 | `target_center_w/h` | 1600×960 | 中央空間先驗裁切區域尺寸 |
| Stage 2 | `face_model_path` | `face_model_ethxgaze.txt` | ETH-XGaze 6 點模型 |
| Stage 3 | `focal_norm` | 960.0 | ETH-XGaze 標準焦距 |
| Stage 3 | `distance_norm` | 60.0 cm | 虛擬相機距離 |
| Stage 3 | `output_size` | (224, 224) | 神經網絡輸入尺寸 |
| Stage 4 | `model_path` | `epoch_24_ckpt.pth.tar` | 預訓練模型 |
| Stage 4 | `use_gpu` | False | 使用 GPU 加速 |

## 安裝

### 1. 安裝依賴套件

```bash
pip install -r requirements.txt
```

### 2. 下載預訓練模型

ETH-XGaze 預訓練模型 `epoch_24_ckpt.pth.tar` 應該已經放在 `models/` 目錄中。

如果沒有，請從 [ETH-XGaze GitHub](https://github.com/xucong-zhang/ETH-XGaze) 下載。

### 3. 準備模型文件

確保以下模型文件已放置在 `models/` 目錄：

- `nano.pt` - YOLO nano 模型（用於頭部檢測）
- `face_landmarker.task` - MediaPipe Face Landmarker 模型（自動下載）
- `face_model_ethxgaze.txt` - ETH-XGaze 6 點 3D 人臉模型
- `epoch_24_ckpt.pth.tar` - ETH-XGaze 預訓練視線估計模型

MediaPipe 模型會在首次運行時自動下載。

## 使用方法

### 🎬 方法 1: 影片處理（推薦用於研究和分析）

**使用 `process_video.py`** - 完整的視線追蹤管線，支援輸出影片和 CSV 數據

#### 基本使用 - 處理影片並導出數據

```bash
python process_video.py --input your_video.mp4 --output output.mp4 --csv gaze_data.csv
```

#### 僅導出 CSV 數據（速度更快，不輸出影片）

```bash
python process_video.py --input your_video.mp4 --csv gaze_data.csv
```

#### 顯示即時預覽（查看處理進度）

```bash
python process_video.py --input your_video.mp4 --output output.mp4 --show-preview
```

#### 跳幀處理（提高處理速度）

```bash
# 每 3 幀處理 1 幀（速度提升 3 倍）
python process_video.py --input your_video.mp4 --skip-frames 2 --csv gaze_data.csv
```

#### 測試模式（只處理前 100 幀）

```bash
python process_video.py --input your_video.mp4 --max-frames 100 --csv test.csv
```

**支援格式**: `.mp4`, `.avi`, `.MOV`, `.mkv` 等常見影片格式

---

###  方法 2: 單張圖片 / Webcam 測試

**使用 `test_gaze_arrow.py`** - 快速測試和視覺化

#### 處理單張圖片

```bash
python test_gaze_arrow.py --mode image --image test_images/your_image.jpg
```

#### 使用 Webcam 即時測試

```bash
python test_gaze_arrow.py --mode webcam
```

**視覺化效果**:
-  綠色人臉框
-  紅色視線箭頭（從雙眼射出）
-  角度顯示（Pitch/Yaw）
-  3D 向量顯示
-  方向標籤（Looking Up/Down/Left/Right）
-  人臉置信度

---

###  方法 3: 作為 Python 庫使用

**使用 `my_gaze_estimation.py`** - 整合到你的專案中

```python
from my_gaze_estimation import GazeEstimationPipeline
import cv2

# 初始化流程
pipeline = GazeEstimationPipeline(config_path='config.yaml')

# 讀取圖像
image = cv2.imread('test_images/your_image.jpg')

# 執行完整流程
result = pipeline.estimate(image)

if result:
    print(f"視線角度: Pitch={result['gaze_pitch_deg']:.1f}°, Yaw={result['gaze_yaw_deg']:.1f}°")
    print(f"視線向量: {result['gaze_vector']}")
    
    # 繪製結果
    vis_image = pipeline.visualize(image, result)
    cv2.imwrite('output/result.jpg', vis_image)
```

---

### 📊 CSV 輸出數據格式

使用 `process_video.py` 處理影片後，CSV 文件包含以下欄位：

| 欄位名稱 | 單位 | 說明 |
|---------|------|------|
| `frame_idx` | - | 幀索引（從 0 開始） |
| `timestamp_sec` | 秒 | 時間戳 |
| **頭部姿態** | | |
| `head_pitch_deg` | 度 | 頭部俯仰角 |
| `head_yaw_deg` | 度 | 頭部偏航角 |
| `head_roll_deg` | 度 | 頭部翻滾角 |
| **視線角度** | | |
| `gaze_pitch_rad` | 弧度 | 視線俯仰角（弧度） |
| `gaze_yaw_rad` | 弧度 | 視線偏航角（弧度） |
| `gaze_pitch_deg` | 度 | 視線俯仰角（度）|
| `gaze_yaw_deg` | 度 | 視線偏航角（度）|
| **視線向量** | | |
| `gaze_vector_x` | - | 3D 視線向量 X 分量 |
| `gaze_vector_y` | - | 3D 視線向量 Y 分量 |
| `gaze_vector_z` | - | 3D 視線向量 Z 分量 |
| **人臉位置** | | |
| `face_bbox_x` | 像素 | 人臉框左上角 X |
| `face_bbox_y` | 像素 | 人臉框左上角 Y |
| `face_bbox_w` | 像素 | 人臉框寬度 |
| `face_bbox_h` | 像素 | 人臉框高度 |

**範例數據分析**:

```python
import pandas as pd
import matplotlib.pyplot as plt

# 讀取 CSV
df = pd.read_csv('gaze_data.csv')

# 繪製視線角度隨時間變化
plt.figure(figsize=(12, 4))
plt.plot(df['timestamp_sec'], df['gaze_pitch_deg'], label='Pitch')
plt.plot(df['timestamp_sec'], df['gaze_yaw_deg'], label='Yaw')
plt.xlabel('Time (s)')
plt.ylabel('Angle (deg)')
plt.legend()
plt.title('Gaze Direction Over Time')
plt.show()

# 統計摘要
print(f"平均 Pitch: {df['gaze_pitch_deg'].mean():.2f}°")
print(f"平均 Yaw: {df['gaze_yaw_deg'].mean():.2f}°")
```

## 配置說明

編輯 `config.yaml` 來自定義系統行為：

- **face_detection**: 調整 MediaPipe 檢測參數
- **normalization**: 調整圖像正規化參數
- **model**: 設置使用 GPU 或 CPU
- **output**: 控制結果保存和可視化

## 系統需求

- **Python**: 3.8 - 3.12
- **作業系統**: macOS (已測試), Linux, Windows
- **記憶體**: 至少 2GB RAM
- **GPU**: 可選（使用 CPU 也可運行，處理速度約 10-15 FPS）

### 核心依賴套件

- `opencv-contrib-python >= 4.13.0` - 圖像處理和 solvePnP
- `ultralytics >= 8.0.0` - YOLO 頭部檢測
- `mediapipe >= 0.10.32` - 人臉特徵點檢測（Tasks API）
- `torch >= 2.0.0` - 神經網絡推理
- `torchvision >= 0.15.0` - ResNet-50 模型
- `numpy >= 1.24.0` - 數值計算
- `pandas >= 2.0.0` - CSV 數據處理

完整依賴列表請參見 `requirements.txt`

---

## 常見問題

### Q1: 未檢測到人臉？

**可能原因**:
- 人臉太小（距離相機超過 1 公尺）
- 光線不足
- 人臉被遮擋
- 置信度閾值太高

**解決方法**:
```python
# 方法 1: 降低置信度閾值（在程式碼中）
detector = FaceDetector(config={'min_confidence': 0.3})  # 默認 0.5

# 方法 2: 使用更高解析度的相機
# 方法 3: 改善光線條件
```

---

### Q2: 視線估計結果不準確？

**可能原因**:
- 模型的系統性偏差（需要個人校準）
- 圖像質量不佳
- 頭部姿態過大（超過 ±60°）

**理論誤差**:
根據 ETH-XGaze 論文，模型在測試集上的誤差約為：
- **Within-dataset**: 3.5° - 4.5°
- **Cross-dataset**: 5.0° - 7.0°

---

### Q3: 為什麼視線箭頭顯示「向下」，但我覺得在看 center？

**這是正常現象！原因**:

1. **座標系差異**: 模型輸出的是「正規化相機座標系」中的角度
   - 虛擬相機原點在「臉部中心」（6 個 landmark 的平均）
   - 你的眼睛在臉部中心**上方** 約 2-3cm
   - 當你看向相機時，視線需要略微「向下」指向虛擬原點

2. **模型偏差**: ETH-XGaze 模型可能有 5-10° 的系統性偏差
   - 這在視線估計領域是常見的
   - 可以通過個人校準來修正

3. **第一人稱 vs 觀察者視角**:
   - 你的感受（第一人稱）：「我在看相機」
   - 模型的視角（正規化座標系）：「視線相對於臉部中心向下 13°」

**驗證方法**: 試著看相機的正上方和正下方，觀察 Pitch 角度的變化

---

### Q4: 使用 GPU 時出錯？

**錯誤訊息**: `RuntimeError: CUDA out of memory` 或 `No CUDA device found`

**解決方法**:
```python
# 方法 1: 使用 CPU（在 stage4 初始化時）
gaze_estimator = GazeEstimator(config={'use_gpu': False})

# 方法 2: 檢查 CUDA 是否可用
import torch
print(f"CUDA available: {torch.cuda.is_available()}")

# 方法 3: 安裝對應版本的 PyTorch
# 訪問 https://pytorch.org 獲取適合你的 CUDA 版本的安裝指令
```

---

### Q5: 處理影片速度太慢？

**優化建議**:

1. **跳幀處理**:
   ```bash
   python process_video.py --input video.mp4 --skip-frames 2  # 速度提升 3x
   ```

2. **不輸出影片**（只導出 CSV）:
   ```bash
   python process_video.py --input video.mp4 --csv data.csv  # 速度提升 2x
   ```

3. **使用 GPU**:
   ```python
   # 在 process_video.py 第 58-61 行
   self.gaze_estimator = GazeEstimator(config={
       'model_path': 'models/epoch_24_ckpt.pth.tar',
       'use_gpu': True  # 改為 True（需要 CUDA）
   })
   ```

4. **降低解析度**: 如果影片解析度過高（如 4K），可以先用 ffmpeg 降低解析度

**參考處理速度** (MacBook M1, CPU mode):
- 1080p 影片: ~12 FPS
- 720p 影片: ~15 FPS
- 使用 `skip-frames 2`: ~36 FPS (有效處理速度)

---

### Q6: `ModuleNotFoundError` 或套件版本衝突？

**解決方法**:

```bash
# 建議使用虛擬環境
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# 或
.venv\Scripts\activate  # Windows

# 安裝所有依賴
pip install -r requirements.txt

# 如果仍有問題，嘗試升級 pip
pip install --upgrade pip
```

## 參考資料

### 📚 論文和資料集

- **ETH-XGaze Dataset & Paper**:
  - [GitHub Repository](https://github.com/xucong-zhang/ETH-XGaze)
  - [論文](https://arxiv.org/abs/2007.15837): *Xucong Zhang et al., "ETH-XGaze: A Large Scale Dataset for Gaze Estimation under Extreme Head Poses and Gaze Directions", ECCV 2020*
  
- **Normalization Method**:
  - [論文](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Zhang_Appearance-Based_Gaze_Estimation_2015_CVPR_paper.pdf): *Xucong Zhang et al., "Appearance-Based Gaze Estimation in the Wild", CVPR 2015*
  - [ETRA 2018 Paper](https://dl.acm.org/doi/10.1145/3204493.3204548): *Xucong Zhang et al., "MPIIGaze: Real-World Dataset and Deep Appearance-Based Gaze Estimation", ETRA 2018*

### 🛠️ 工具和技術文檔

- **MediaPipe Face Mesh**: [官方文檔](https://google.github.io/mediapipe/solutions/face_mesh.html)
  - 468 個 3D 面部特徵點
  - Canonical Face Model 定義
  
- **OpenCV solvePnP**: [官方文檔](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)
  - PnP 問題求解
  - 頭部姿態估計

- **PyTorch ResNet-50**: [TorchVision Models](https://pytorch.org/vision/stable/models.html)

---

## 授權

本專案基於 ETH-XGaze 的預訓練模型，遵循其原始授權條款。

**預訓練模型**: ETH-XGaze (Xucong Zhang et al., ECCV 2020)  
**實現代碼**: 本專案為教育和研究目的而實現

---

## 致謝

- **ETH-XGaze Team**: 提供高質量的預訓練模型和數據集
- **Ultralytics**: 提供高效的 YOLO 目標檢測框架
- **MediaPipe Team**: 提供快速準確的人臉特徵點檢測工具
- **OpenCV Community**: 提供強大的計算機視覺庫

---
