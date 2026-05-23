# process_video.py 重寫說明

## 更新日期
2026-03-16

## 更新原因

舊的 `process_video.py` 基於過時的接口，有多個嚴重錯誤：

### 舊版本的問題

1. **Stage 3 調用錯誤**：
   - 傳入了不存在的 `landmarks` 參數
   - 使用錯誤的返回值鍵名 `['image']` 而非 `['normalized_image']`

2. **Stage 4 調用錯誤**：
   - 調用不存在的 `predict()` 方法，應該是 `estimate()`
   - 使用錯誤的返回值鍵名 `['pitch']`/`['yaw']` 而非 `['gaze_angles'][0/1]`

3. **配置錯誤**：
   - 引用不存在的 `face_model_mediapipe.txt`
   - 使用錯誤的 `distance_norm: 600.0`（應該是 60.0）

## 新版本特點

###  完全符合最新接口

1. **Stage 1-5 接口正確**：
   -  Stage 3: `normalize(image, rotation_vector, translation_vector, camera_matrix)`
   -  Stage 4: `estimate(normalized_image)` 返回 `{'gaze_angles': [pitch, yaw], ...}`
   -  Stage 5: `angles_to_vector(pitch, yaw)`

2. **配置正確**：
   -  使用 `face_model_ethxgaze.txt`（6 點模型）
   -  使用 `distance_norm: 60.0`（正確的距離）

3. **可視化效果**：
   -  使用 `draw_gaze_with_face_box()`（與 test_gaze_arrow.py 相同）
   -  顯示人臉框、視線箭頭、角度信息、方向標籤

###  保留所有原有功能

1. **影片處理**：
   -  逐幀處理影片
   -  輸出帶標註的影片

2. **數據導出**：
   -  導出 CSV 格式的視線數據
   -  包含時間戳、頭部姿態、視線角度、3D 向量

3. **性能優化**：
   -  跳幀處理（`--skip-frames`）
   -  最大幀數限制（`--max-frames`）
   -  進度條顯示（tqdm）

4. **預覽功能**：
   -  即時預覽窗口（`--show-preview`）

## 使用方法

### 基本用法

```bash
# 激活虛擬環境
source .venv/bin/activate

# 基本使用（處理影片並導出數據）
python process_video.py --input video.mp4 --output output.mp4 --csv data.csv

# 僅導出數據（不輸出影片，速度更快）
python process_video.py --input video.mp4 --csv data.csv

# 跳幀處理（速度提升 3 倍）
python process_video.py --input video.mp4 --skip-frames 2 --csv data.csv

# 測試模式（只處理前 100 幀）
python process_video.py --input video.mp4 --max-frames 100 --csv test.csv

# 顯示即時預覽
python process_video.py --input video.mp4 --output output.mp4 --show-preview
```

### 命令行參數

| 參數 | 必需 | 說明 |
|------|------|------|
| `--input` | ✅ | 輸入影片路徑 |
| `--output` | ❌ | 輸出影片路徑（不指定則不輸出影片）|
| `--csv` | ❌ | 輸出 CSV 數據路徑 |
| `--show-preview` | ❌ | 顯示處理預覽窗口 |
| `--skip-frames` | ❌ | 跳幀處理（0=所有幀，1=每2幀處理1幀，2=每3幀處理1幀）|
| `--max-frames` | ❌ | 最大處理幀數（用於測試）|

### CSV 輸出格式

導出的 CSV 包含以下欄位：

```
frame_idx           # 幀索引
timestamp_sec       # 時間戳（秒）
head_pitch_deg      # 頭部俯仰角（度）
head_yaw_deg        # 頭部偏航角（度）
head_roll_deg       # 頭部翻滾角（度）
gaze_pitch_rad      # 視線俯仰角（弧度）
gaze_yaw_rad        # 視線偏航角（弧度）
gaze_pitch_deg      # 視線俯仰角（度）
gaze_yaw_deg        # 視線偏航角（度）
gaze_vector_x       # 3D 視線向量 X 分量
gaze_vector_y       # 3D 視線向量 Y 分量
gaze_vector_z       # 3D 視線向量 Z 分量
face_bbox_x         # 人臉框 X 座標
face_bbox_y         # 人臉框 Y 座標
face_bbox_w         # 人臉框寬度
face_bbox_h         # 人臉框高度
```

## 技術細節

### 處理流程

每一幀的處理流程：

1. **Stage 1**：人臉檢測 → 獲取 468 個特徵點 → 選擇 6 個關鍵點
2. **Stage 2**：頭部姿態估計 → 計算 rvec, tvec, 歐拉角
3. **Stage 3**：圖像正規化 → 生成 224×224 正規化圖像
4. **Stage 4**：神經網絡推理 → 預測視線角度（pitch, yaw）
5. **Stage 5**：視線向量轉換 → 轉換為 3D 單位向量

### 可視化效果

使用 `draw_gaze_with_face_box()` 函數，顯示：
- 人臉邊界框（藍色）
- 視線箭頭（紅色，從鼻尖或眼睛中心出發）
- 視線角度信息（Pitch, Yaw）
- 視線方向標籤（例如：「向右上看」）

### 性能優化

- **跳幀處理**：`--skip-frames 2` 可以提升 3 倍速度
- **僅導出數據**：不指定 `--output` 可以節省編碼時間
- **進度條**：使用 tqdm 顯示處理進度

## 與其他測試文件的關係

| 文件 | 用途 | 模式 |
|------|------|------|
| `test_stage1.py` | 測試 Stage 1 | 圖像/Webcam |
| `test_stage2.py` | 測試 Stage 1-2 | 圖像/Webcam |
| `test_stage3.py` | 測試 Stage 1-3 | 圖像/Webcam |
| `test_stage4.py` | 測試 Stage 1-4 | 圖像/Webcam |
| `test_gaze_arrow.py` | 測試 Stage 1-5（進階可視化）| 圖像/Webcam |
| **`process_video.py`** | **批量處理影片** | **影片文件** |

**關鍵差異**：
- `test_*.py`：即時測試（webcam 或單張圖像）
- `process_video.py`：批量處理影片文件，導出數據

## 測試建議

### 1. 快速測試（處理前 10 幀）

```bash
python process_video.py --input your_video.mp4 --max-frames 10 --csv test.csv --show-preview
```

### 2. 完整處理

```bash
# 僅導出數據（最快）
python process_video.py --input video.mp4 --csv data.csv

# 或同時輸出影片
python process_video.py --input video.mp4 --output output.mp4 --csv data.csv
```

### 3. 數據分析

處理完成後，可以用 Python 分析 CSV：

```python
import pandas as pd
import matplotlib.pyplot as plt

# 讀取數據
df = pd.read_csv('data.csv')

# 繪製視線角度隨時間變化
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(df['timestamp_sec'], df['gaze_pitch_deg'])
plt.ylabel('Pitch (度)')
plt.title('視線俯仰角隨時間變化')

plt.subplot(2, 1, 2)
plt.plot(df['timestamp_sec'], df['gaze_yaw_deg'])
plt.ylabel('Yaw (度)')
plt.xlabel('時間 (秒)')
plt.title('視線偏航角隨時間變化')

plt.tight_layout()
plt.savefig('gaze_analysis.png')
plt.show()
```

## 總結

###  新版本優勢

1. **接口正確**：與最新的 Stage 1-5 完全一致
2. **配置正確**：使用正確的 face model 和參數
3. **可視化美觀**：使用 test_gaze_arrow.py 的進階可視化
4. **功能完整**：保留所有原有功能（跳幀、預覽、CSV 導出）
5. **可以運行**：已驗證 `--help` 可以正常執行

###  下一步

建議用一個短視頻測試：

```bash
python process_video.py --input test_video.mp4 --max-frames 100 --csv test.csv --show-preview
```

確認一切正常後，再處理完整的影片。
