# CSIE_project
專題
成員名單：B1109240劉立綸、B1229011李瑋恆、B1229056黃昱翔、B1229059黃璿軒

---

# 兒童注意力監測與量化分析平台 Attention Assessment System for ASD and TD Children

## 1. 系統簡介 (System Overview)

本系統為一套**兒童注意力監測與量化分析平台**，專注於自閉症類群障礙（ASD）與典型發展（TD）兒童之**共同注意力（Joint Attention）能力評估**，特別針對：

* **反應性共同注意力（RJA, Responding to Joint Attention）**

系統透過多模態 AI 技術（視覺 + 語音），自動判定：

> 兒童是否在指定時間內將注意力（視線或手勢）導向目標物
並產出可供臨床與研究使用的**時間序列事件與統計報表**。

---

## 1.1 規格目的 

本專案的設計目標為：

* 建立**客觀、可重複的注意力評估機制**
* 降低人工觀察造成的主觀誤差
* 提供臨床決策支援（Clinical Decision Support System, CDSS）

> 本系統**不取代醫療診斷**，僅作為輔助分析工具
---

## 1.2 規格範圍 (Scope)

本系統涵蓋以下核心技術模組：

###  視覺分析模組

* 人臉關鍵點偵測（MediaPipe Face Mesh）
* 頭部姿態估計（PnP 解法）
* 視線方向估計（Gaze Vector Estimation）
* 手勢與指向向量建模

###  語音觸發模組

* 語音辨識（Whisper large-v3）
* 關鍵字偵測
* 動態時間窗（3秒 reaction window）

###  邏輯判定模組

* 貪婪排他性匹配（Greedy Bipartite Matching）
* 射線碰撞檢測（Ray Casting）
* 注意力事件量化

---

# 2. 系統概述 (System Description)

## 2.1 系統目標 (Objectives)

###  核心目標

1. **量化評估**

   * 將「注意力」轉換為可計算指標
   * 提供統計分析基礎

2. **完整紀錄**

   * 視線軌跡
   * 手勢向量
   * 語音觸發時間點

---

## 2.2 系統範圍 (System Scope)

###  輸入 (Input)

* 固定視角實驗影片（MP4）
* 測驗提示字卡（影像）

###  處理 (Processing)

多模態 AI 推論：

* 物件偵測（YOLO）
* 視線估計（Gaze Estimation）
* 手勢辨識（MediaPipe）
* 語音分析（Whisper）

###  輸出 (Output)

* 分析影片（含標註）
* 注意力事件報表（時間戳記）

---

## 2.3 系統架構 (System Architecture)

系統採用**三層式架構（Three-Layer Architecture）**：

---

###  Layer 1：資料層 (Data Layer)

負責資料擷取與前處理：

* Video Capture（影像讀取）
* Audio Extraction（音訊抽取）
* Frame Preprocessing（影像標準化）

---

###  Layer 2：AI 推論層 (AI Inference Layer)

核心 AI 模型：

| 模組   | 技術                    |
| ---- | --------------------- |
| 物件偵測 | YOLO                  |
| 手勢辨識 | MediaPipe Hands       |
| 視線估計 | ETH-XGaze (ResNet-50) |
| 語音辨識 | Whisper               |

---

###  Layer 3：分析層 (Analysis Layer)

關鍵演算法：

#### 1. 身分匹配

* Greedy Bipartite Matching
* 解決手部重疊問題

#### 2. 空間判定

* Ray Casting（射線投射）
* Bounding Box Collision

#### 3. 注意力判斷

* 視線向量 ∩ 目標物
* 指向向量 ∩ 目標物

---

# 系統運作流程 (Execution Pipeline)

## 核心設計：**條件式觸發（Event-driven Processing）**

### 流程：

1. 語音分析（離線）
2. 偵測關鍵字 → 建立時間窗
3. 影像逐幀分析
4. 僅在「有效時間窗」內：

   * 啟動手勢 + 視線判定
5. 輸出事件結果

---

#  輸入 / 輸出設計

## 輸入

* `.mp4` 實驗影片
* YOLO 權重
* OCR 字卡

## 輸出

###  視覺輸出

包含：

* 物件框
* 視線射線
* 手勢向量

###  數據輸出

包含：

* 語音逐字稿
* 關鍵字標記
* 事件時間戳

---

#  系統特點 (Key Features)

##  1.多模態融合（Multimodal Fusion）

融合：

* 視覺（Vision）
* 語音（Audio）
* 空間幾何（Geometry）

---

##  2.時序分離架構（Temporal Decoupling）

* 語音先處理
* 視覺後推論
* 降低記憶體峰值

---

##  3.高精度注意力判定

透過：

* 視線向量
* 手指向量
* 空間碰撞檢測

實現客觀判斷

---

## 4. 動態載入（Dynamic Loading）

優勢：

* 降低 RAM 使用
* 提升穩定性
* 模組解耦

---

# 系統限制 (Limitations)

##  語音限制

* 噪音干擾會影響 Whisper
* 疊音會降低準確度

##  視覺限制

* 光線需穩定
* 避免逆光

## 空間限制

* 受試者需在 1.5m 內
* 固定攝影機視角
* 
---

#  結論

本系統為一個：

> **結合電腦視覺 + 語音辨識 + 幾何推論的行為量化系統**
其核心創新在於：

* 「時間窗觸發機制」
* 「向量幾何 + AI融合判定」
* 「自動化 Joint Attention 評估」
