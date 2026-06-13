# 兒童注意力監測系統 - 設計規格書

# Attention Assessment System - Design Specification

**專案名稱**：Attention Assessment System for ASD and TD Children  
**版本**：2.0  
**最後更新**：2026 年 6 月  
**文件作者**：專題團隊（B1109240 劉立綸、B1229011 李瑋恆、B1229056 黃昱翔、B1229059 黃璿軒）

github:[https://github.com/Aliulilun/Attention-Assessment-System.git](https://github.com/Aliulilun/Attention-Assessment-System.git)

---

## 目錄 (Table of Contents)

1. [系統架構圖](#1-系統架構圖)
2. [使用案例圖](#2-使用案例圖)
3. [循序圖](#3-循序圖)
  - 3.1 [整合系統完整流程](#31-整合系統完整流程)
  - 3.2 [視線估計五階段流程](#32-視線估計五階段流程)
4. [活動圖](#4-活動圖)
  - 4.1 [以 Stage 為單位的分析流程](#41-以-stage-為單位的分析流程)
  - 4.2 [Ray Casting 注意力判定演算法](#42-ray-casting-注意力判定演算法)
  - 4.3 [語音觸發時間窗機制](#43-語音觸發時間窗機制)
5. [類別圖](#5-類別圖)
6. [狀態圖](#6-狀態圖)
  - 6.1 [題目偵測狀態機（EasyOCR）](#61-題目偵測狀態機easyocr)
  - 6.2 [時間窗狀態轉換](#62-時間窗狀態轉換)
7. [部署圖](#7-部署圖)

---

## 1. 系統架構圖

### 說明

本圖展示整合後系統的三層架構，以及四個核心功能模組（`stage`、`speech`、`point`、`eye_tracking`）的資料流向。系統輸入單一實驗影片，連續偵測題目進度（Stage 1~8），對每個題目分別進行視線與手勢分析，最終輸出各題目的注意力判定結果。

### 架構圖 (Component Diagram)

```mermaid
graph TB
    subgraph INPUT["資料輸入層"]
        VID[實驗影片 MP4<br/>含音軌]
    end

    subgraph PREPROCESS["前處理"]
        AUDIO[音訊抽取]
        FRAMES[影像逐幀讀取]
    end

    subgraph STAGE_MODULE["題目偵測模組 (stage)"]
        EASYOCR[EasyOCR<br/>數字牌辨識 1~8]
        STAGE_LOGIC[單向階段推進邏輯<br/>防呆/防閃爍機制]
        CURRENT_STAGE[當前題目 Stage N]
    end

    subgraph SPEECH_MODULE["語音觸發模組 (point 內建)"]
        WHISPER[Whisper large-v3<br/>語音轉文字]
        KEYWORD[關鍵字偵測]
        TIME_WIN[建立 3 秒時間窗]
    end

    subgraph POINT_MODULE["手勢分析模組 (point)"]
        YOLO_POSE[YOLOv11-Pose<br/>人體骨架偵測]
        MP_HANDS[MediaPipe Hands<br/>手部關鍵點]
        ID_MATCH[手臂距離分數<br/>身分匹配演算法]
        YOLO_OBJ[客製化 YOLO 模型群<br/>目標物件偵測]
        POINT_VEC[指向向量計算]
    end

    subgraph GAZE_MODULE["視線估計模組 (eye_tracking)"]
        S1[Stage1<br/>YOLO nano 人臉偵測]
        S2[Stage2<br/>solvePnP 頭部姿態]
        S3[Stage3<br/>影像正規化 224x224]
        S4[Stage4<br/>ETH-XGaze ResNet-50]
        S5[Stage5<br/>視線向量轉換]
    end

    subgraph ENGINE["注意力判定引擎"]
        RAY[Ray Casting<br/>射線碰撞檢測]
        EVENT[注意力事件記錄]
    end

    subgraph OUTPUT["輸出層（以題目為單位）"]
        VID_OUT[標注影片<br/>含骨架/視線/射線]
        REPORT[各題目分析報表<br/>Stage 1~8 結果]
        CSV[時間序列 CSV]
    end

    VID --> AUDIO
    VID --> FRAMES
    AUDIO --> WHISPER
    WHISPER --> KEYWORD
    KEYWORD --> TIME_WIN
    FRAMES --> EASYOCR
    EASYOCR --> STAGE_LOGIC
    STAGE_LOGIC --> CURRENT_STAGE
    FRAMES --> YOLO_POSE
    YOLO_POSE --> MP_HANDS
    MP_HANDS --> ID_MATCH
    ID_MATCH --> POINT_VEC
    FRAMES --> YOLO_OBJ
    FRAMES --> S1
    S1 --> S2 --> S3 --> S4 --> S5
    CURRENT_STAGE -->|決定當前題目目標物| RAY
    TIME_WIN -->|觸發判定視窗| RAY
    S5 -->|視線向量| RAY
    POINT_VEC -->|指向向量| RAY
    YOLO_OBJ -->|目標物 BBox| RAY
    RAY --> EVENT
    EVENT --> REPORT
    EVENT --> VID_OUT
    EVENT --> CSV

    style STAGE_MODULE fill:#fff0cc
    style GAZE_MODULE fill:#e1f5ff
    style POINT_MODULE fill:#e8ffe1
    style SPEECH_MODULE fill:#ffe1f5
    style ENGINE fill:#ffe8e1
```



### 關鍵技術說明


| 模組                  | 核心技術                           | 功能                  |
| ------------------- | ------------------------------ | ------------------- |
| 題目偵測 (stage)        | EasyOCR + 單向推進邏輯               | 辨識數字牌 1~8，追蹤當前題目    |
| 語音觸發 (point)        | Whisper large-v3               | 關鍵字偵測，建立 3 秒判定時間窗   |
| 手勢分析 (point)        | YOLOv11-Pose + MediaPipe Hands | 骨架偵測、手部關鍵點、指向向量     |
| 視線估計 (eye_tracking) | ETH-XGaze ResNet-50            | 5 階段視線方向估計，輸出 3D 向量 |
| 注意力判定               | Ray Casting + Greedy Matching  | 空間碰撞判定，計算各題目命中率     |


---

## 2. 使用案例圖

### 說明

展示系統的主要使用者（研究人員、臨床醫師、系統管理員）以及核心功能，強調系統以「各題目（Stage 1~8）分析結果」為最終輸出。

### 使用案例圖 (Use Case Diagram)

```mermaid
graph LR
    subgraph SYS["系統邊界：注意力分析系統"]
        UC1[上傳實驗影片]
        UC2[自動偵測題目進度<br/>Stage 1~8]
        UC3[視線方向分析]
        UC4[手勢指向分析]
        UC5[語音關鍵字觸發]
        UC6[各題目注意力判定]
        UC7[匯出各題目分析報表]
        UC8[查看標注影片]
        UC9[設定模型參數]
    end

    RA[研究人員]
    CL[臨床醫師]
    AD[系統管理員]

    RA --> UC1
    RA --> UC7
    RA --> UC8
    CL --> UC7
    CL --> UC8
    AD --> UC9

    UC2 -.include.-> UC1
    UC3 -.include.-> UC1
    UC4 -.include.-> UC1
    UC5 -.include.-> UC1
    UC6 -.include.-> UC2
    UC6 -.include.-> UC3
    UC6 -.include.-> UC4
    UC6 -.include.-> UC5
    UC7 -.include.-> UC6
    UC8 -.include.-> UC6
```



### 使用案例說明


| 使用案例           | 描述                            | 主要角色      |
| -------------- | ----------------------------- | --------- |
| UC1: 上傳實驗影片    | 輸入含音軌的實驗影片 (MP4)              | 研究人員      |
| UC2: 自動偵測題目進度  | EasyOCR 辨識數字牌，自動推進 Stage 1~8  | 系統自動      |
| UC3: 視線方向分析    | eye_tracking 5 階段流程，輸出視線向量    | 系統自動      |
| UC4: 手勢指向分析    | point 模組偵測骨架、手部、指向向量          | 系統自動      |
| UC5: 語音關鍵字觸發   | Whisper 偵測關鍵字，建立 3 秒時間窗       | 系統自動      |
| UC6: 各題目注意力判定  | Ray Casting 判定每個 Stage 的注意力命中 | 系統自動      |
| UC7: 匯出各題目分析報表 | 輸出 Stage 1~8 命中率統計與 CSV       | 研究人員、臨床醫師 |
| UC8: 查看標注影片    | 觀看含視線箭頭、手勢向量、Stage 標籤的影片      | 研究人員、臨床醫師 |
| UC9: 設定模型參數    | 調整 YOLO 閾值、時間窗長度等             | 系統管理員     |


---

## 3. 循序圖

### 3.1 整合系統完整流程

#### 說明

展示從影片輸入到各題目結果輸出的完整互動時序，包含題目偵測、語音觸發、視線估計、手勢偵測與注意力判定的協作關係。

```mermaid
sequenceDiagram
    participant User as 使用者
    participant Main as 整合主系統
    participant Stage as 題目偵測<br/>(EasyOCR)
    participant Speech as 語音模組<br/>(Whisper)
    participant Gaze as 視線模組<br/>(eye_tracking)
    participant Point as 手勢模組<br/>(point)
    participant Engine as 注意力<br/>判定引擎
    participant Report as 結果報表

    User->>Main: 上傳影片 (MP4)

    Main->>Speech: 抽取音訊
    activate Speech
    Speech->>Speech: Whisper 轉錄（或讀取快取）
    Speech->>Speech: 掃描關鍵字，建立時間窗清單
    Speech-->>Main: 時間窗清單 [(T_start, T_end), ...]
    deactivate Speech

    Note over Main: 逐幀處理迴圈開始

    loop 每一幀
        Main->>Stage: 傳入當前幀
        activate Stage
        Stage->>Stage: 每 15 幀執行 EasyOCR 掃描
        Stage->>Stage: 白名單過濾 + 單向推進邏輯
        Stage-->>Main: 當前題目 Stage N
        deactivate Stage

        Main->>Gaze: 傳入當前幀
        activate Gaze
        Gaze->>Gaze: Stage1 YOLO nano 人臉偵測
        Gaze->>Gaze: Stage2 solvePnP 頭部姿態
        Gaze->>Gaze: Stage3 影像正規化 224x224
        Gaze->>Gaze: Stage4 ETH-XGaze ResNet-50 推理
        Gaze->>Gaze: Stage5 視線向量轉換
        Gaze-->>Main: 視線向量 (x,y,z)
        deactivate Gaze

        Main->>Point: 傳入當前幀
        activate Point
        Point->>Point: YOLOv11-Pose 骨架偵測
        Point->>Point: MediaPipe 手部關鍵點
        Point->>Point: 手臂距離分數 身分匹配
        Point->>Point: YOLO 目標物件偵測
        Point->>Point: 指向向量計算
        Point-->>Main: 指向向量 + 目標物 BBox
        deactivate Point

        alt 當前幀在語音時間窗內
            Main->>Engine: 視線向量+指向向量+目標物+Stage N
            activate Engine
            Engine->>Engine: Ray Casting 視線碰撞檢測
            Engine->>Engine: Ray Casting 指向碰撞檢測
            Engine->>Engine: TOUCH_WARN 防呆過濾
            Engine->>Engine: 記錄 Stage N 注意力事件
            Engine-->>Main: 本幀判定結果
            deactivate Engine
        end
    end

    Note over Main: 影片處理完畢

    Main->>Report: 彙整 Stage 1~8 全部事件
    activate Report
    Report->>Report: 計算各題目命中率
    Report->>Report: 生成標注影片
    Report->>Report: 匯出 CSV 時間序列
    Report-->>User: 各題目分析報表 + 標注影片
    deactivate Report
```



---

### 3.2 視線估計五階段流程

#### 說明

詳細展示 `eye_tracking` 模組五個處理階段的順序互動與資料轉換過程。

```mermaid
sequenceDiagram
    participant Frame as 輸入幀
    participant S1 as Stage1<br/>人臉偵測
    participant S2 as Stage2<br/>頭部姿態
    participant S3 as Stage3<br/>正規化
    participant S4 as Stage4<br/>神經網路
    participant S5 as Stage5<br/>向量轉換

    Frame->>S1: 原始影像 (1920x1080)
    activate S1
    S1->>S1: YOLO nano 頭部框
    S1->>S1: MediaPipe 468 特徵點
    S1->>S1: 選取 6 個關鍵點
    S1-->>S2: 2D 關鍵點座標
    deactivate S1

    activate S2
    S2->>S2: 載入 3D 人臉模型
    S2->>S2: solvePnP 求解 6DoF
    S2-->>S3: R_head, tvec
    deactivate S2

    activate S3
    S3->>S3: 構建正規化旋轉矩陣 R_norm
    S3->>S3: 構建單應性矩陣 W
    S3->>S3: warpPerspective 透視變換
    S3-->>S4: 正規化影像 (224x224 RGB)
    deactivate S3

    activate S4
    S4->>S4: BGR to RGB + ImageNet 標準化
    S4->>S4: ResNet-50 特徵提取 (2048 維)
    S4->>S4: FC 層輸出視線角度
    S4-->>S5: 視線角度 (pitch, yaw)
    deactivate S4

    activate S5
    S5->>S5: angles_to_vector(pitch, yaw)
    S5->>S5: 向量正規化
    S5-->>Frame: 3D 視線向量 (x, y, z)
    deactivate S5
```



#### 階段資料轉換說明


| 階段      | 輸入               | 核心演算法                 | 輸出                    |
| ------- | ---------------- | --------------------- | --------------------- |
| Stage 1 | 原始影像 (1920×1080) | YOLO nano + MediaPipe | 6 個 2D 關鍵點            |
| Stage 2 | 2D 關鍵點 + 3D 模型   | OpenCV solvePnP       | 旋轉矩陣 R_head、平移向量 tvec |
| Stage 3 | R_head, tvec     | warpPerspective       | 正規化影像 (224×224)       |
| Stage 4 | 正規化影像            | ResNet-50             | 視線角度 (pitch, yaw)     |
| Stage 5 | 視線角度             | 三角函數轉換                | 3D 單位向量 (x, y, z)     |


---

## 4. 活動圖

### 4.1 以 Stage 為單位的分析流程

#### 說明

展示系統以題目（Stage 1~8）為分析單位的完整處理流程，包含 EasyOCR 題目偵測、語音時間窗觸發、視線與手勢並行分析、以及各題目結果彙整輸出。

```mermaid
flowchart TD
    Start([影片開始]) --> SpeechOffline[語音離線分析<br/>建立時間窗清單]
    SpeechOffline --> InitStage[初始化 Stage = 1]
    InitStage --> FrameLoop{讀取下一幀}

    FrameLoop -->|幀存在| OCRCheck{每 15 幀?}
    FrameLoop -->|影片結束| GenReport[生成各題目報表]

    OCRCheck -->|是| RunOCR[EasyOCR 掃描數字牌]
    OCRCheck -->|否| SkipOCR[沿用上一幀 Stage]

    RunOCR --> StageAdvance{偵測到新數字<br/>且符合推進條件?}
    StageAdvance -->|是| UpdateStage[更新 Stage N<br/>更換對應目標物件]
    StageAdvance -->|否| KeepStage[保持當前 Stage]

    UpdateStage --> RunAnalysis
    KeepStage --> RunAnalysis
    SkipOCR --> RunAnalysis

    RunAnalysis[並行分析當前幀]
    RunAnalysis --> GazeRun[視線估計<br/>eye_tracking 5 階段]
    RunAnalysis --> GestureRun[手勢偵測<br/>point 模組]

    GazeRun --> CheckWindow{在時間窗內?}
    GestureRun --> CheckWindow

    CheckWindow -->|否| Visualize
    CheckWindow -->|是| RayCasting[Ray Casting 判定<br/>視線 + 指向]

    RayCasting --> TouchCheck{手部在 BBox 內?}
    TouchCheck -->|是| WARN[TOUCH_WARN 不計分]
    TouchCheck -->|否| HitCheck{碰撞命中?}

    HitCheck -->|是| RecordHit[記錄 Stage N 命中事件]
    HitCheck -->|否| RecordMiss[記錄 Stage N 未命中]

    WARN --> Visualize
    RecordHit --> Visualize
    RecordMiss --> Visualize

    Visualize[繪製標注畫面<br/>視線箭頭 + 手勢向量 + Stage 標籤]
    Visualize --> FrameLoop

    GenReport --> StageReport[各題目命中率統計<br/>Stage 1~8 結果彙整]
    StageReport --> VideoOut[輸出標注影片]
    StageReport --> CSVOut[匯出 CSV 時間序列]
    StageReport --> End([結束])

    style UpdateStage fill:#FFD700
    style RecordHit fill:#90EE90
    style WARN fill:#FFB6C1
    style RecordMiss fill:#f0f0f0
```



---

### 4.2 Ray Casting 注意力判定演算法

#### 說明

展示系統如何以射線投射（Ray Casting）判定受試者是否將注意力（視線或手勢）導向當前題目（Stage N）的目標物，以及防呆機制（TOUCH_WARN）的運作。

```mermaid
flowchart TD
    Start([本幀觸發判定]) --> GetStage[取得當前 Stage N<br/>對應目標物 BBox]
    GetStage --> GetGaze[取得視線向量]
    GetGaze --> GetHand[取得手勢向量]

    GetHand --> LoopTarget{遍歷 Stage N<br/>所有目標物}

    LoopTarget --> GazeRay[計算視線射線]
    GazeRay --> GazeHit{視線射線與<br/>目標物碰撞?}

    GazeHit -->|是| LogGaze[記錄視線命中]
    GazeHit -->|否| HandExist

    LogGaze --> HandExist{手部存在?}
    HandExist -->|否| NextObj1
    HandExist -->|是| HandRay[計算指向射線]

    HandRay --> TouchWarnCheck{手部座標在<br/>BBox 內部?}
    TouchWarnCheck -->|是| TouchWarn[TOUCH_WARN<br/>標記並忽略計分]
    TouchWarnCheck -->|否| HandHit{指向射線與<br/>目標物碰撞?}

    HandHit -->|是| LogHand[記錄手勢命中]
    HandHit -->|否| NextObj2

    TouchWarn --> NextObj2
    LogHand --> NextObj2
    NextObj1 --> MoreObjs
    NextObj2 --> MoreObjs

    MoreObjs{還有目標物?}
    MoreObjs -->|是| LoopTarget
    MoreObjs -->|否| CalcStageScore[更新 Stage N 本幀分數]

    CalcStageScore --> End([結束])

    style LogGaze fill:#90EE90
    style LogHand fill:#87CEEB
    style TouchWarn fill:#FFB6C1
```



---

### 4.3 語音觸發時間窗機制

#### 說明

展示系統如何使用 Whisper 進行語音辨識、偵測關鍵字並建立動態判定時間窗，以及快取機制降低重複運算成本的完整流程。

```mermaid
flowchart TD
    Start([開始: 影片處理]) --> ExtractAudio[抽取音訊]

    ExtractAudio --> CheckCache{存在逐字稿快取?}
    CheckCache -->|是| LoadCache[載入快取 TXT]
    CheckCache -->|否| RunWhisper[執行 Whisper 語音辨識]

    RunWhisper --> SaveCache[儲存逐字稿快取]
    SaveCache --> ParseTranscript
    LoadCache --> ParseTranscript[解析逐字稿]

    ParseTranscript --> Found{偵測到關鍵字?}

    Found -->|否| ReturnEmpty[返回空清單]
    Found -->|是| CreateWindow[建立時間窗]

    CreateWindow --> SetStart[起始時間 = 關鍵字時間戳 T]
    SetStart --> SetEnd[結束時間 = T + 3 秒]
    SetEnd --> AddToList[加入時間窗清單]

    AddToList --> MoreKeywords{還有其他關鍵字?}
    MoreKeywords -->|是| Found
    MoreKeywords -->|否| ReturnList[返回時間窗清單]

    ReturnList --> VideoLoop[進入影片逐幀處理]
    ReturnEmpty --> VideoLoop

    VideoLoop --> End([結束])

    style Found fill:#FFD700
    style CreateWindow fill:#90EE90
```



---

## 5. 類別圖

### 說明

展示整合系統的核心類別結構，包含新增的 `StageDetector`（EasyOCR 題目偵測）與 `ReportRecorder`（各題目結果紀錄）。

```mermaid
classDiagram
    class IntegratedAttentionSystem {
        -stage_detector: StageDetector
        -gaze_pipeline: GazeEstimationPipeline
        -gesture_analyzer: GestureAnalyzer
        -speech_analyzer: SpeechAnalyzer
        -attention_engine: AttentionEngine
        -report_recorder: ReportRecorder
        +process_video(video_path: str)
        -_process_frame(frame, timestamp, stage)
    }

    class StageDetector {
        -reader: easyocr.Reader
        -current_stage: int
        -last_best_detection: dict
        -frame_cnt: int
        -zone_roi: tuple
        +__init__()
        +detect(frame: ndarray): int
        -_run_ocr(frame: ndarray): list
        -_advance_stage(detected_num: int): bool
        -_draw_overlay(frame: ndarray): ndarray
    }

    class SpeechAnalyzer {
        -model: whisper.Whisper
        -keywords: list
        +analyze(video_path: str): list
        -_load_or_transcribe(video_path: str): str
        -_build_time_windows(transcript: str): list
    }

    class GazeEstimationPipeline {
        -face_detector: FaceDetector
        -head_pose_estimator: HeadPoseEstimator
        -normalizer: ImageNormalizer
        -gaze_estimator: GazeEstimator
        -gaze_converter: GazeVectorConverter
        +estimate(frame: ndarray): dict
    }

    class FaceDetector {
        -yolo_model: YOLO
        -face_landmarker: FaceLandmarker
        +detect(image: ndarray): dict
    }

    class GestureAnalyzer {
        -yolo_pose: YOLO
        -mp_hands: MediaPipe
        -yolo_objects: dict
        +analyze(frame: ndarray, stage: int): dict
        -_match_identity(skeletons, hands): list
        -_calc_point_vector(hand_kp: ndarray): ndarray
    }

    class AttentionEngine {
        +judge(gaze_vec, hand_vec, targets, stage): dict
        -_ray_cast_gaze(gaze_vec, bbox): bool
        -_ray_cast_hand(hand_vec, bbox): bool
        -_touch_warn_check(hand_kp, bbox): bool
    }

    class ReportRecorder {
        -events: dict
        +log_event(stage: int, timestamp: float, hit_type: str)
        +export_csv(path: str)
        +export_video(frames: list, path: str)
        +get_stage_summary(): dict
    }

    IntegratedAttentionSystem --> StageDetector
    IntegratedAttentionSystem --> SpeechAnalyzer
    IntegratedAttentionSystem --> GazeEstimationPipeline
    IntegratedAttentionSystem --> GestureAnalyzer
    IntegratedAttentionSystem --> AttentionEngine
    IntegratedAttentionSystem --> ReportRecorder

    GazeEstimationPipeline --> FaceDetector
    GestureAnalyzer ..> YOLOv11Pose : 骨架偵測
    StageDetector ..> EasyOCR : 數字牌辨識
    SpeechAnalyzer ..> Whisper : 語音轉文字
```



---

## 6. 狀態圖

### 6.1 題目偵測狀態機（EasyOCR）

#### 說明

展示 `stage` 模組如何透過 EasyOCR 辨識數字牌，以單向防呆邏輯推進題目（Stage 1~8），同時每個 Stage 皆並行進行視線與手勢分析。

```mermaid
stateDiagram-v2
    [*] --> Idle: 系統啟動

    Idle --> Stage1: 使用者框選 ROI 後<br/>EasyOCR 偵測到數字 1

    state "各 Stage 共用行為" as Analyzing {
        [*] --> Scanning
        Scanning --> OCRRun: 每 15 幀執行一次
        OCRRun --> Scanning: 無新數字 / 雜訊忽略
        OCRRun --> Advancing: 偵測到合法新數字<br/>(current <= num <= current+2)
        Advancing --> Scanning: 更新 Stage 完成
    }

    Stage1 --> Analyzing
    Stage1 --> Stage2: EasyOCR 偵測到 2 或 3
    Stage2 --> Stage3: EasyOCR 偵測到 3 或 4
    Stage3 --> Stage4: EasyOCR 偵測到 4 或 5
    Stage4 --> Stage5: EasyOCR 偵測到 5 或 6
    Stage5 --> Stage6: EasyOCR 偵測到 6 或 7
    Stage6 --> Stage7: EasyOCR 偵測到 7 或 8
    Stage7 --> Stage8: EasyOCR 偵測到 8

    Stage8 --> Completed: 影片結束

    Completed --> [*]

    note right of Stage1
        每個 Stage 並行進行：
        視線估計 (eye_tracking)
        手勢偵測 (point)
        語音時間窗判定
    end note

    note right of Stage8
        Stage 8 特殊處理：
        切換至 robot_point_model.pt
        啟用 SMA 滑動平均濾波
    end note
```



### 狀態轉換規則


| 狀態                  | 觸發條件                      | 說明           |
| ------------------- | ------------------------- | ------------ |
| Idle → Stage1       | 使用者框選 ROI，EasyOCR 偵測到 1   | 初始化          |
| Stage N → Stage N+1 | EasyOCR 偵測到 N+1 或 N+2 的數字 | 單向推進，最多跳 2 級 |
| 任何 Stage            | 偵測到超出範圍或回退數字              | 視為雜訊，忽略      |
| Stage 8 → Completed | 影片結束                      | 彙整輸出         |


---

### 6.2 時間窗狀態轉換

#### 說明

展示語音關鍵字觸發後，3 秒高靈敏判定時間窗的狀態轉換過程。

```mermaid
stateDiagram-v2
    [*] --> Idle: 影片開始

    Idle --> Listening: 逐幀監聽時間戳

    Listening --> Triggered: 當前時間戳落入時間窗<br/>(T_start, T_start+3s)

    state Triggered {
        [*] --> ActiveJudging
        ActiveJudging --> EventLogged: 每幀執行 Ray Casting
        EventLogged --> ActiveJudging: 繼續判定
    }

    Triggered --> Listening: 超過 T_start + 3s

    Listening --> [*]: 影片結束

    note right of Triggered
        記錄當前 Stage N 的
        命中事件與時間戳
    end note
```



---

## 7. 部署圖

### 說明

展示整合後系統在統一環境下的部署架構，目標是將 `point` 與 `eye_tracking` 合併至單一 Conda 環境。

```mermaid
graph TB
    subgraph TARGET["目標：統一整合環境"]
        subgraph ENV["Conda Python 3.9<br/>attention_system_integrated"]
            subgraph GAZE["eye_tracking 功能"]
                E1[YOLO nano]
                E2[MediaPipe >= 0.10.32<br/>Face Landmarker Tasks API]
                E3[ETH-XGaze ResNet-50]
                E4[OpenCV solvePnP]
            end
            subgraph GESTURE["point 功能"]
                P1[YOLOv11-Pose]
                P2[MediaPipe 0.10.32<br/>Hands solutions API]
                P3[Whisper large-v3]
                P4[YOLO 客製化模型群]
            end
            subgraph STAGE_ENV["stage 功能"]
                ST1[EasyOCR<br/>數字牌辨識 1~8]
            end
        end
        FFMPEG[ffmpeg<br/>音訊抽取/影片縫合]
    end

    subgraph MODELS["模型檔案"]
        M1[eye_tracking/models/<br/>nano.pt / face_landmarker.task<br/>epoch_24_ckpt.pth.tar]
        M2[point/model/<br/>front/background/balloon<br/>bubble/toy/robot_point]
    end

    subgraph PLATFORM["執行平台"]
        WIN[Windows<br/>主要開發環境]
        MAC[macOS<br/>可支援]
    end

    subgraph GITHUB_RES["GitHub"]
        REPO[原始碼倉庫]
        RELEASE[Releases 模型下載]
    end

    REPO -.git clone.-> TARGET
    RELEASE -.下載模型.-> MODELS
    MODELS --> ENV
    TARGET --> WIN
    TARGET --> MAC

    style GAZE fill:#e1f5ff
    style GESTURE fill:#e8ffe1
    style STAGE_ENV fill:#fff0cc
```



### 部署環境需求（整合後）


| 項目          | 規格                           |
| ----------- | ---------------------------- |
| Python      | 3.9                          |
| 套件管理        | Conda (Anaconda / Miniconda) |
| 作業系統        | Windows（主要）/ macOS（可支援）      |
| MediaPipe   | 0.10.32（統一版本）                |
| PyTorch     | 2.8.0                        |
| Ultralytics | 8.4.19                       |
| 系統工具        | ffmpeg                       |


---

## 附錄 A：技術堆疊總覽


| 類別   | 套件               | 版本       | 用途            |
| ---- | ---------------- | -------- | ------------- |
| 深度學習 | PyTorch          | 2.8.0    | 神經網路推理        |
| 深度學習 | TorchVision      | 0.23.0   | ResNet-50 載入  |
| 電腦視覺 | OpenCV           | 4.13.0   | 影像處理、solvePnP |
| 電腦視覺 | MediaPipe        | 0.10.32  | 人臉 + 手部偵測     |
| 目標偵測 | Ultralytics YOLO | 8.4.19   | 人體骨架 + 物件偵測   |
| 字元辨識 | EasyOCR          | 1.7.2    | 數字牌辨識         |
| 語音處理 | OpenAI Whisper   | large-v3 | 語音轉文字         |
| 影片處理 | FFmpeg           | 系統版      | 音訊抽取 + 影片縫合   |
| 影片處理 | MoviePy          | 2.2.1    | 影片編輯          |
| 資料處理 | NumPy            | 2.0.2    | 矩陣運算          |
| 資料處理 | Pandas           | 2.3.3    | CSV 與時間序列     |


---

## 附錄 B：關鍵演算法參考

### Ray Casting（射線投射）

- **應用場景**：3D 視線向量與 2D 邊界框的碰撞檢測

### Greedy Bipartite Matching（貪婪二分圖匹配）

- **應用場景**：解決雙手交錯時的身分識別問題

### PnP (Perspective-n-Point)

- **實作**：`cv2.solvePnP(SOLVEPNP_ITERATIVE)`
- **應用場景**：從 6 個 2D 特徵點與 3D 模型求解 6DoF 頭部姿態

### SMA (Simple Moving Average)

- **公式**：`MA(t) = (x[t] + ... + x[t-n+1]) / n`
- **應用場景**：Stage 8 機器人指向座標的雜訊過濾

### EasyOCR 單向推進邏輯

- **白名單**：只辨識 `1~8` 的數字
- **推進條件**：`current_stage <= detected_num <= current_stage + 2`
- **防呆**：偵測到回退或超範圍數字時視為雜訊忽略

---

## 附錄 C：文件版本歷史


| 版本  | 日期         | 作者   | 變更摘要                                    |
| --- | ---------- | ---- | --------------------------------------- |
| 1.0 | 2026/06/02 | 專題團隊 | 初版發布                                    |
| 2.0 | 2026/06/07 | 專題團隊 | 修正 UML 圖，反映 EasyOCR stage 偵測、整合架構與各題目輸出 |


---

*本文件使用 Mermaid 語法生成 UML 圖表，可直接在 GitHub 或支援 Mermaid 的 Markdown 編輯器中渲染。*