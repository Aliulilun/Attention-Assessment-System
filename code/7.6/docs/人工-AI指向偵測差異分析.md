# 指向（Pointing）偵測差異分析報告（修訂版）

**分析日期**：2026-08-14
**資料來源**：`人工-AI結果差異對照表.xlsx`（黃色 = 人工與 AI 不一致）
**程式依據**：`modules/interaction.py`、`modules/stage_scoring.py`、`modules/scoring_engine.py`
**樣本**：70 位幼兒（ASD 33、TD 37）× Stage 1~10，雙方資料齊全者 **633 筆**

> **本版修訂重點**：前一版報告把 Excel 的欄位對應搞錯了（誤把人工的 `LI` 欄當成「有指向」），
> 導致 A/B/C 三類分類與「元兇是 TH 不是指向」的結論站不住腳。
> 本版重新從欄位定義與程式碼推導，結論**相反**：指向才是主要誤差來源。詳見第一、二節。

---

## 一、先修正欄位對應（前版最大的錯誤）

Excel 每個 Stage 有 6 欄，是**人工／AI 交錯配對**，底色淺藍（`DDEBF7`）者為 AI 欄，黃色 = 該對不一致：

| 欄位對 | 人工欄名 | AI 欄名 | 實際代表的構念 |
|--------|---------|--------|--------------|
| 第 1 對 | `LR_n` / `HR_n` | `TB_n` | 兒童**有看向目標物** |
| 第 2 對 | `LI_n` | `TH_n` | 兒童**有看回施測者／機器人** |
| 第 3 對 | `HI_n` | `HI_n` | 綜合等級 HI 是否達成 |

**前版的錯誤**：把人工 `LI_n` 讀成「人工說有指向」。實際上 `LI_n` 配對的是 AI 的 `TH_n`，
代表的是**視線看回人**，跟指向無關。建立在這個誤讀上的「A類純假陽 16 / B類等級誤判 45 / C類漏偵 16」
整套分類因此無效，其核心結論「B 類 45 例真正元兇是 TH，不是指向本身」方向是反的。

**另一個發現**：Stage 11~14 只有人工欄（`LR/LI/HI/total`），**沒有 AI 對照欄**，
所以本次比對只涵蓋 Stage 1~10。

---

## 二、指向沒有獨立欄位，但可以被「解出來」

Excel 沒有 Pointing 欄，但 `modules/stage_scoring.py` 的計分是決定性的：

```
total = TB(1分) + TH(2分) + Pointing(1分)
total = 0 → F   1 → LR/HR   2 → HI   3 → LI   4 → HI
Stage 8 特規：有 Pointing → HI；只有 TH → LI；皆無 → F
```

已知 `TB`、`TH`、`HI` 三欄，就能反解 `Pointing`：

| TB | TH | Pointing=0 | Pointing=1 | ⇒ 由 HI 反推 |
|----|----|-----------|-----------|-------------|
| 0 | 0 | total 0 → F | total 1 → LR | HI 必為 0，**無法反解** |
| 1 | 0 | total 1 → LR | total 2 → **HI** | HI=1 ⟺ 有指向 |
| 0 | 1 | total 2 → **HI** | total 3 → LI | HI=1 ⟺ **沒有**指向 |
| 1 | 1 | total 3 → LI | total 4 → **HI** | HI=1 ⟺ 有指向 |

**驗證**：AI 側 681 筆全數符合此公式，0 筆矛盾 → 反解模型正確。
唯一有歧義的組合 `(TB=0, TH=1)` 在人工與 AI 兩側**都是 0 筆**，
`(TB=1, TH=0, HI=1)` 人工 2 筆、AI 1 筆，方向唯一。
扣掉 `(0,0,0)` 這種無法反解的 26 筆，得到 **633 筆可直接比對指向的資料**。

---

## 三、指向偵測的真實表現

### 混淆矩陣（N = 633）

| | AI 說沒指向 | AI 說有指向 |
|---|---|---|
| **人工說沒指向** | 536 (TN) | **61 (假陽性 FP)** |
| **人工說有指向** | **19 (漏偵測 FN)** | 17 (TP) |

**準確率 87.4%｜精確率 21.8%｜召回率 47.2%**

準確率 87.4% 是假象——因為真實指向只佔 5.7%（36/633），全猜「沒指向」也有 94.3%。
真正的指標是**精確率 21.8%：AI 每報 5 次指向，只有 1 次是真的。**

人工認定指向 36 次，AI 報了 78 次，**過度偵測 2.2 倍**。

### 各 Stage 明細

| Stage | 關卡 | N | 人工指向 | AI指向 | FP | FN | TP | 精確率 |
|-------|------|---|---------|--------|----|----|----|-------|
| 1 | 真人指近物 | 60 | 2 | 4 | 4 | 2 | 0 | 0% |
| 2 | 真人指近物 | 64 | 0 | 0 | 0 | 0 | 0 | — |
| 3 | 真人指遠物 | 63 | 2 | 2 | 2 | 2 | 0 | 0% |
| 4 | 真人指遠物 | 56 | 3 | 4 | 2 | 1 | 2 | 50% |
| 5 | 神奇氣球 | 67 | 5 | 6 | 4 | 3 | 2 | 33% |
| 6 | 看偶寫字 | 64 | 2 | **8** | **8** | 2 | 0 | **0%** |
| 7 | **開箱驚喜袋** | 60 | 16 | 27 | **16** | 5 | 11 | 41% |
| 8 | **手機怪聲** | 70 | 2 | **20** | **18** | 0 | 2 | **10%** |
| 9 | 機器人畫畫 | 68 | 3 | 5 | 5 | 3 | 0 | 0% |
| 10 | 機器人煙火秀 | 61 | 1 | 2 | 2 | 1 | 0 | 0% |
| **合計** | | **633** | **36** | **78** | **61** | **19** | **17** | **21.8%** |

**Stage 8（18 例）+ Stage 7（16 例）= 56% 的假陽性。**
Stage 8 尤其誇張：人工只認定 2 位幼兒有指向，AI 報了 20 位。

### ASD / TD 差異

| 組別 | N | 人工指向率 | AI指向率 | 誤報率 (FP/實際無指向) | 漏偵 |
|------|---|-----------|---------|---------------------|------|
| ASD | 271 | 6.6% | **18.5%** | **15.4%** | 7 |
| TD | 362 | 5.0% | 7.7% | 6.4% | 12 |

兩組真實指向率其實相近（6.6% vs 5.0%），但 **AI 在 ASD 組的誤報率是 TD 組的 2.4 倍**。
這是臨床上最危險的偏誤方向——系統會系統性高估 ASD 幼兒的主動溝通能力。

---

## 四、HI 欄位不一致的成因拆解（推翻前版結論）

HI 欄位黃色（人工/AI 不一致）共 **81 例**，逐例比對 TB、TH、Pointing 三者：

| 成因 | 例數 | 佔比 |
|------|------|------|
| **只有指向判錯**（TB、TH 都對） | **64** | **79%** |
| 指向 + TH 都錯 | 14 | 17% |
| 只有視線（TB/TH）錯，指向沒問題 | 3 | 4% |

**78/81（96%）的 HI 誤差都牽涉指向錯誤。**
前版說「B 類 45 例真正元兇是 TH」——實際上純視線造成的 HI 誤差只有 **3 例**。

---

## 五、根本原因（對照程式碼，非猜測）

### 原因 1（最關鍵）：`dwell` 計數器不是「連續」，是「累積」

`modules/interaction.py:886-903`：

```python
if current_frame_pointing[owner]:
    self.dwell_counters[owner] += 1
    self.hold_counters[owner] = self.HOLD_FRAMES        # 12
else:
    if self.hold_counters[owner] > 0:
        self.hold_counters[owner] -= 1                  # ← dwell 沒有歸零
    else:
        self.dwell_counters[owner] = 0
        self.ray_history[owner]["origin"].clear()
        self.ray_history[owner]["vector"].clear()

if self.dwell_counters[owner] >= self.DWELL_FRAMES and self.hold_counters[owner] > 0:
    pointing_owners.add(owner)
```

註解寫的是「**連續**指向 >= DWELL_FRAMES 幀」，但實作不是。
只要中斷幀數不超過 `HOLD_FRAMES=12`，`dwell` 就**不會歸零，只會一路累加**。

**後果：**
- 在 11 幀（約 0.37 秒）內出現 3 次**不連續**的單幀誤判 → 判定為有效指向。
- 更糟的是**一旦成立就下不來**：只要每 12 幀內有任意 1 幀誤判（約 8% 的偵測率），
  `hold` 就被重設回 12，射線可以**無限期存活**。MediaPipe 的隨機閃爍完全足以維持這個狀態。

這是 FP=61 的頭號來源，也解釋了為什麼 Stage 7、8 這種「手一直在動」的關卡最嚴重。

### 原因 2：`hold` 期間 `ray_history` 不清空 → 幽靈射線掃過目標框

`hold` 緩衝期內舊向量留在 `ray_history`（`maxlen=5`），射線用這些**過期向量**的平均繼續繪製與判定。
手已經放下、甚至完全離開畫面後，射線還能存在 12 幀，並在這段期間掃過 YOLO 目標框產生 `Child HIT`。

### 原因 3：射線無限長、無深度、無角度容差

`ray_intersects_box`（`interaction.py:304`）是標準 Ray-AABB，**射線沒有長度上限**：

```python
tmin = max(min(tx1,tx2), min(ty1,ty2)); tmax = min(max(tx1,tx2), max(ty1,ty2))
return bool(tmax >= max(0, tmin))
```

- 目標在畫面另一端、距離 1500px 以外，只要方向大致對就算命中。
- 這是純 2D 判定，沒有深度資訊。幼兒指向近處，射線在影像上仍可能穿過遠處物件的框。
- 射線起點是**手腕**、方向是**手腕→食指尖**（`interaction.py:869`）。這條基線只有約 1 倍掌長，
  手腕定位誤差幾像素，投射到 1500px 遠端就是數十至上百像素的橫向偏移。

### 原因 4：Stage 8 幾乎沒有任何約束

`main.py:402` / `hurry/main.py:479`：

```python
if current_stage == 8 and interaction.last_child_pointing_active:
    child_is_pointing_hit = True
```

Stage 8 的判定條件是「**射線存在**」——不需命中目標、不限方向、不限持續時間。
再加上 `stage_scoring.py` 的 Stage 8 特規「有 Pointing 直接 → HI」，
等於**在 T0+10 秒的窗口內，任何一瞬間出現指向形狀的手，就給 HI**。
結果：人工 2 例 vs AI 20 例，精確率 10%。

幼兒聽到怪聲時的驚訝抬手、遮耳、探索動作，只要手形短暫接近「食指伸、其餘收」，就會中標。

### 原因 5：`stage_scoring.py` 用 total 分數查表，把不同組合壓成同一級

```
total = 2 → HI
```
但 `total=2` 有兩種完全不同的來源：
- `TB=1, TH=0, Pointing=1`：有看物、有指，**但沒看回人** → 給 HI
- `TB=0, TH=1, Pointing=0`：只看回人，**沒看物也沒指** → 也給 HI

臨床 IJA 定義中，HI（高階主動）需要**手勢 + 眼神接觸**兩者兼具，第二種情況不該是 HI。
本資料中第二種組合實際出現 0 次、第一種 AI 側 1 次（人工側 2 次），**當前影響很小**，
但這是等著爆炸的設計缺陷，建議改成明確的組合判定而非分數查表。

### 原因 6（漏偵測 FN=19）

- 幼兒指向動作快（< 3 幀）時，`DWELL_FRAMES=3` 直接吃掉。
- 手指指向角度偏斜，2D 射線未穿過精確的 BBox。
- 手被身體/物品遮擋 → `skin-finger` / `knuckle` 皮膚驗證擋掉。
- TD 組漏偵較多（12 vs 7），推測是動作較自然連貫、不符合「靜態標準指向手勢」的嚴格幾何檢查。

> **注意**：FN=19 相對 FP=61 小很多。**在精確率只有 21.8% 的情況下，任何放寬判定的改動都會讓情況更糟。**
> 前版建議的「BBox 加 10~15% padding」不建議現在做。

---

## 六、對前版建議的檢討

| 前版建議 | 評估 |
|---------|------|
| 在 `scoring_engine` 加 `POINTING_MIN_DURATION_FRAMES=2` | **改錯地方**。`interaction.py` 已有 `DWELL_FRAMES=3`，問題不是門檻太低，是**累積邏輯有 bug**（原因 1）。 |
| 調高 MediaPipe `min_hand_detection_confidence` | **會造成回退**。`interaction.py:74` 註解已記錄：0.50 實測會漏掉清楚指物的真手，才刻意調回 0.40。 |
| BBox 加 10~15% padding 改善漏偵 | **方向錯誤**。FP 是 FN 的 3.2 倍，放寬只會惡化精確率。 |
| Stage 8 加方向一致性檢查 | **方向正確**，但前版低估了嚴重性（說「5 例」，實際 18 例）。 |
| 「MediaPipe 只偵測食指與拇指的幾何角度」 | **與程式碼不符**。`is_valid_pointing` 已檢查五指彎曲、指尖遠近排序、PIP 彎角、拇指收攏、皮膚色實體驗證共 10 餘道關卡。幾何檢查不是弱點。 |

---

## 七、建議的修改（依投報率排序）

### P0-1：修正 dwell 為真正的「連續幀」計數

```python
# modules/interaction.py  analyze_interaction()
if current_frame_pointing[owner]:
    self.dwell_counters[owner] += 1
    self.hold_counters[owner] = self.HOLD_FRAMES
else:
    if self.hold_counters[owner] > 0:
        self.hold_counters[owner] -= 1
        self.dwell_counters[owner] = 0          # ★ 新增：中斷即歸零，hold 只保留射線不保留 dwell
    else:
        self.dwell_counters[owner] = 0
        self.ray_history[owner]["origin"].clear()
        self.ray_history[owner]["vector"].clear()
```

若要保留「短暫掉偵測不算中斷」的彈性，改用滑動窗口更精準：

```python
# 近 N 幀中至少 M 幀有指向才成立（取代單純累加）
self.POINT_WIN, self.POINT_MIN_HITS = 8, 6      # 8 幀內至少 6 幀（75% duty）
self.point_window = {"Child": deque(maxlen=8), "Tester": deque(maxlen=8)}
...
self.point_window[owner].append(current_frame_pointing[owner])
active = sum(self.point_window[owner]) >= self.POINT_MIN_HITS
```

**預估**：這一項單獨就能吃掉大部分 FP，尤其 Stage 7/8 的閃爍型誤判。

### P0-2：Stage 8 加入方向穩定性 + 最短持續時間

Stage 8 沒有目標物可比對，唯一能用的約束是**動作本身的一致性**：

```python
# modules/interaction.py
POINTING_DIR_STABLE_DEG   = 25      # 連續幀射線方向偏差需 < 25°
POINTING_MIN_HOLD_FRAMES  = 10      # 約 0.33 秒（30fps），指向需持續這麼久才算數

# 在 last_child_pointing_active 成立前，額外要求：
#   ray_history["Child"]["vector"] 內各向量與中位角度的偏差都 < 25°
#   且 dwell（修正後的連續幀數） >= POINTING_MIN_HOLD_FRAMES
```

驚訝抬手、遮耳、探索是**快速掃動**，方向不會穩定；刻意指向會停住。這是最能區分兩者的訊號。

### P0-3：射線加上最大有效距離

```python
# ray_intersects_box 加入 t 上限；或呼叫端過濾
MAX_RAY_LEN_RATIO = 0.6     # 射線有效長度 <= 0.6 × 畫面對角線
```
避免「指向近處的手，射線在影像上打到畫面另一端的物件框」。

### P1-1：改用手臂向量校正射線方向

目前方向 = 手腕→食指尖（基線短、雜訊放大）。
`calculate_arm_link_score` 已經取得 YOLO-Pose 的肩／肘／腕關鍵點，可以直接複用：

```python
# 融合：手肘→手腕（穩定、長基線）與 手腕→食指尖（精確、短基線）
arm_vec    = wrist_kpt - elbow_kpt
finger_vec = idx_tip - wri
raw_vec    = 0.4 * unit(arm_vec) + 0.6 * unit(finger_vec)
```
手臂向量提供穩定的粗方向，食指提供修正，可明顯降低長距離的橫向漂移。

### P1-2：`stage_scoring.py` 改為組合判定

```python
def compute_stage_score(record):
    tb = record.get("tb") is not None
    th = record.get("th") is not None
    pt = record.get("pointing_t") is not None
    total = (1 if tb else 0) + (2 if th else 0) + (1 if pt else 0)

    if record.get("stage") == 8:
        return total, ("HI" if pt else "LI" if th else "F")

    if pt and th:   level = "HI"          # 手勢 + 眼神接觸 = 高階主動
    elif tb and th: level = "LI"          # 視線交替、無手勢 = 低階主動
    elif tb or pt:  level = "HR" if is_far_stage(record.get("label")) else "LR"
    elif th:        level = "LR"          # 只看人，未達主動分享
    else:           level = "F"
    return total, level
```
語意明確，不再靠分數碰撞查表。

### P2：先不要碰漏偵測（FN）

BBox padding、放寬皮膚驗證、降低 `DWELL_FRAMES` 這類改動，等 **精確率拉到 60% 以上**再考慮。
現階段每放進 1 個真陽性，會同時放進約 3 個假陽性。

---

## 八、建議的驗證流程

改完後不要只看整體準確率（會被 94% 的負樣本稀釋），固定回報這四個數字：

```
精確率 = TP / (TP + FP)      ← 現況 21.8%，目標 > 60%
召回率 = TP / (TP + FN)      ← 現況 47.2%，守住不低於 45%
ASD 誤報率 / TD 誤報率        ← 現況 15.4% / 6.4%，目標比值接近 1.0
Stage 8 AI 指向人數           ← 現況 20，人工 2
```

可用本報告的反解方法直接從新產出的 `event_record` txt 算出 Pointing 欄，
與 Excel 人工欄逐格比對（`summarize/summarize.py` 已有解析 txt 的基礎，可擴充）。

---

## 九、結論

1. **前版報告的欄位對應錯誤**，其「元兇是 TH 不是指向」的結論方向相反。
   實際上 **96%（78/81）的 HI 誤差都牽涉指向判錯**，純視線問題只有 3 例。

2. **指向偵測精確率只有 21.8%**——每 5 次報告有 4 次是誤報。這是目前系統最弱的一環。

3. **首要 bug 是 `dwell` 累積邏輯**（`interaction.py:886-903`）：註解寫「連續」，實作是「累積」，
   導致零星閃爍就能維持指向狀態且永不歸零。幾何檢查那 10 幾道關卡本身沒問題。

4. **Stage 8 是最大破口**（FP 18 例，精確率 10%）：無目標、無方向、無時長約束。

5. **ASD 組誤報率是 TD 組的 2.4 倍**，系統性高估 ASD 幼兒的主動溝通能力，是最需要優先處理的臨床偏誤。

---

## 附註：分析方法的可重現性

本報告所有數字由 `人工-AI結果差異對照表.xlsx` 直接計算，方法如下：

1. 讀取每個 Stage 的 6 欄，依底色區分人工／AI 欄。
2. 以 `stage_scoring.compute_stage_score` 的公式，從 `(TB, TH, HI)` 反解 `Pointing`。
3. 排除 `(TB=0, TH=0, HI=0)` 這種無法反解的 26 筆與雙方缺值者，得 633 筆。
4. 反解的唯一性驗證：AI 側 681 筆全數符合公式（0 矛盾）；歧義組合 `(0,1,·)` 兩側皆 0 筆。
