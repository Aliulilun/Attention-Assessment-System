# 計分引擎修正（A4 + A5 + A6 + C4 + C2 + A7 文件澄清）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修正 `程式碼問題清單_對照人工AI差異.md` 中已決議的一批項目——A4（TB/TH 遲滯互斥，採選項 b：新命中打斷對側殘留）、A5（階段切換遲滯歸零）、A6（`MAX_GAZE_FALLBACK` 縮短）、C4（Stage 1 fallback 門檻延後）、C2 方向1（Stage 9/10 T0 關鍵字具體化）、A7 附帶的 pitch 閾值不一致（僅文件澄清，不改行為）——並以 10 支 benchmark 影片重跑 `compare_with_manual.py` 驗證整批改動。

**Architecture:** 把 `main.py` 與 `hurry/main.py` 裡重複的視線命中遲滯邏輯（`gaze_obj_hold` / `gaze_tester_hold`）抽成 `modules/gaze_hold.py` 的 `GazeHoldTracker` 類別，用 pytest 對純邏輯做回歸測試；`scoring_engine.py` 內的常數與關鍵字修改直接對 `ScoringEngine` 用合成的 speech_cache fixture 做單元測試；`main.py`/`hurry/main.py` 的接線改動因為是大型逐幀迴圈、無法單獨單元測試，改用「語法編譯 + grep 自洽性檢查」把關，真正的行為驗證留給最後一個任務的 GPU benchmark 重跑。

**Tech Stack:** Python 3.9（conda 環境 `mediapipe_py39`）、pytest、既有的 `summarize/compare_with_manual.py`。

**Spec:** `C:\Users\wayne\Desktop\project\integrate_v7.1\程式碼問題清單_對照人工AI差異.md`（章節 A4、A5、A6、C4、C2、A7 附帶發現；建議處理順序表）

## Global Constraints

- 所有 Python 檔案的 `open()` 一律指定 `encoding='utf-8'`（專案既有規範，見 CLAUDE.md E4）
- 測試與程式碼修改一律在 `C:\Users\wayne\Desktop\project\integrate_v7.1\release\app\` 這份程式碼上進行（`release/app` 是這個環境唯一存在的 v7.1 原始碼；E1 提到的另一份主目錄副本這次不處理，依使用者指示）
- 所有指令在 `mediapipe_py39` 環境下執行：`C:\Users\wayne\miniconda3\condabin\conda.bat run -n mediapipe_py39 <command>`（或先 `conda activate mediapipe_py39`）
- 方法論（沿用 spec 文件「方法論備忘」）：一次只改一組、影響關卡互不重疊的改動可以合併驗證；收緊判定前先確認真陽性清單不能被誤殺；改動沒效果要先確認有沒有被執行到，不能只看數字
- **不要改動** `interaction.py` 的 `ARM_FINGER_MAX_ANGLE`、`TESTER_HEAD_EXPAND`、`HEAD_KPT_CONF_TH` 等 A1/B1 v2 已定案的參數——這些已經改完，本計畫只負責重跑驗證，不重新調整
- **不要重試** A7 的 `EXTREME_TURNING` 補償機制本身（`max_hold_frames`、無條件給 TB 等）——已實驗證實會傷真陽性，spec 明文禁止重試

---

## 現況記錄：E5 項目無法在這份程式碼上執行

Spec 的 E5 提到要修正 `三檔結果差異分析.md` 裡「GPU 效能差異 → 幀取樣率不同」的錯誤歸因。搜過整個 `C:\Users\wayne\Desktop\project` 都找不到這個檔案——它應該只存在於原作者另一台機器或另一份目錄上，這次環境裡沒有。**本計畫略過 E5**，之後在有該檔案的環境上再處理。

---

### Task 1: `GazeHoldTracker`——把命中遲滯邏輯抽成可測試的共用類別（A4 選項 b）

**Files:**
- Create: `modules/gaze_hold.py`
- Create: `tests/test_gaze_hold.py`
- Create: `pytest.ini`

**Interfaces:**
- Produces: `class GazeHoldTracker` in `modules/gaze_hold.py`，建構子 `GazeHoldTracker(hold_frames: int = 10)`；方法 `update(self, raw_obj: bool, raw_tester: bool) -> tuple[bool, bool]`（回傳 `(child_is_gazing_at, child_is_gazing_at_tester)`）；方法 `reset(self) -> None`（兩個內部倒數計時器歸零）。Task 2、Task 3 會 import 並使用這個類別與這三個名字，不要改名。

- [ ] **Step 1: 安裝 pytest 到 `mediapipe_py39` 環境**

Run:
```
C:\Users\wayne\miniconda3\condabin\conda.bat run -n mediapipe_py39 pip install pytest
```
Expected: `Successfully installed pytest-...`

- [ ] **Step 2: 建立 `pytest.ini`，讓 `tests/` 底下的檔案能 `import modules`**

`pytest.ini`（放在 `app/` 目錄，跟 `main.py` 同一層）：

```ini
[pytest]
testpaths = tests
pythonpath = .
```

- [ ] **Step 3: 寫失敗測試 `tests/test_gaze_hold.py`**

```python
# -*- coding: utf-8 -*-
from modules.gaze_hold import GazeHoldTracker


def test_raw_hit_is_immediately_true():
    tracker = GazeHoldTracker(hold_frames=10)
    obj, tester = tracker.update(True, False)
    assert obj is True
    assert tester is False


def test_hold_bridges_short_dropout_on_same_side():
    """單側短暫漏偵測（YOLO 掉框、射線擦框緣）要被遲滯橋接，這是原本
    GAZE_HIT_HOLD_FRAMES 存在的理由，抽成類別後不能失去這個行為。"""
    tracker = GazeHoldTracker(hold_frames=10)
    tracker.update(True, False)  # 第 0 幀：真的命中物品
    for _ in range(9):
        obj, tester = tracker.update(False, False)  # 接下來 9 幀都沒命中
        assert obj is True, "遲滯窗口內應該仍視為看著物品"
    obj, tester = tracker.update(False, False)  # 第 10 幀：遲滯耗盡
    assert obj is False, "超過 hold_frames 幀沒命中應該真的判定移開視線"


def test_fresh_tester_hit_clears_stale_object_hold():
    """A4 的核心迴歸測試，重現 spec 1-2 節 104-S3 案例：
    物品的遲滯還沒退完時，若這一幀施測者被真正命中，
    不該讓 child_is_gazing_at 跟 child_is_gazing_at_tester 同時為 True
    （原本的獨立計數器設計會讓 TB 和 TH 在同一個時間戳被同時計次）。"""
    tracker = GazeHoldTracker(hold_frames=10)
    tracker.update(True, False)  # 幀 0：命中物品，obj_hold 進入倒數
    for _ in range(5):
        tracker.update(False, False)  # 幀 1-5：物品遲滯還在倒數（尚未耗盡）
    obj, tester = tracker.update(False, True)  # 幀 6：施測者被真正命中
    assert obj is False, "新的施測者命中必須立刻打斷物品端的殘留遲滯"
    assert tester is True


def test_fresh_object_hit_clears_stale_tester_hold():
    """跟上一個測試對稱：施測者側的殘留遲滯，也要被物品側的新命中打斷。"""
    tracker = GazeHoldTracker(hold_frames=10)
    tracker.update(False, True)  # 幀 0：命中施測者
    for _ in range(5):
        tracker.update(False, False)
    obj, tester = tracker.update(True, False)  # 幀 6：物品被真正命中
    assert obj is True
    assert tester is False, "新的物品命中必須立刻打斷施測者端的殘留遲滯"


def test_reset_clears_both_holds():
    tracker = GazeHoldTracker(hold_frames=10)
    tracker.update(True, False)
    tracker.reset()
    obj, tester = tracker.update(False, False)
    assert obj is False
    assert tester is False
```

- [ ] **Step 4: 執行測試，確認全部失敗（`modules.gaze_hold` 還不存在）**

Run:
```
C:\Users\wayne\miniconda3\condabin\conda.bat run -n mediapipe_py39 pytest tests/test_gaze_hold.py -v
```
Expected: `ModuleNotFoundError: No module named 'modules.gaze_hold'`（5 個測試全部 ERROR）

- [ ] **Step 5: 實作 `modules/gaze_hold.py`**

```python
# -*- coding: utf-8 -*-
"""
視線命中遲滯追蹤器
==================
把「看物品」與「看施測者/機器人」兩條獨立的命中遲滯（hold）邏輯
包成一個類別，取代原本 main.py / hurry/main.py 裡重複的
gaze_obj_hold / gaze_tester_hold 兩個區域變數。

遲滯的目的：射線擦框緣、YOLO 短暫掉框造成的單幀漏判很常見，
命中後「立即亮起、延遲熄滅」（連續 hold_frames 幀沒命中才熄滅）
可以橋接這種短暫漏判，見 GAZE_HIT_HOLD_FRAMES 原本的註解。

A4 修正（2026-09-15）：原本兩個 hold 各自獨立倒數，互不知道對方，
會出現「物品的殘留遲滯還沒退完 + 這一幀施測者被真正命中」同時成立
的情況，導致 TB 和 TH 在同一個時間戳被背靠背計次（見 spec 1-2 節
104-S3 案例：TH 跟 TB 同一時間戳、+0.0s from TB）。

修法（選項 b：新命中打斷對側殘留）：哪一側這一幀有真正的原始命中，
就立刻把「另一側」的殘留遲滯清空——只清對方的殘留倒數，不影響
自己這一側原本橋接短暫漏偵測的能力。這比「先觸發者優先、鎖死到
清零」更寬容：小朋友快速地「看一眼物品→立刻轉頭看人」時，後者
不會被壓住。
"""


class GazeHoldTracker:
    def __init__(self, hold_frames: int = 10):
        self.hold_frames = hold_frames
        self._obj_hold = 0
        self._tester_hold = 0

    def update(self, raw_obj: bool, raw_tester: bool):
        """餵入本幀的原始命中結果，回傳套用遲滯後的 (看物品, 看施測者)。"""
        if raw_obj:
            self._obj_hold = self.hold_frames
            self._tester_hold = 0  # 打斷施測者側的殘留遲滯
        elif self._obj_hold > 0:
            self._obj_hold -= 1

        if raw_tester:
            self._tester_hold = self.hold_frames
            self._obj_hold = 0  # 打斷物品側的殘留遲滯
        elif self._tester_hold > 0:
            self._tester_hold -= 1

        gazing_at_obj = raw_obj or self._obj_hold > 0
        gazing_at_tester = raw_tester or self._tester_hold > 0
        return gazing_at_obj, gazing_at_tester

    def reset(self):
        """階段切換時呼叫，避免上一關最後的遲滯狀態帶進新階段（A5）。"""
        self._obj_hold = 0
        self._tester_hold = 0
```

- [ ] **Step 6: 重新執行測試，確認全部通過**

Run:
```
C:\Users\wayne\miniconda3\condabin\conda.bat run -n mediapipe_py39 pytest tests/test_gaze_hold.py -v
```
Expected: `5 passed`

- [ ] **Step 7: Commit**

```bash
git add pytest.ini tests/test_gaze_hold.py modules/gaze_hold.py
git commit -m "feat: add GazeHoldTracker with mutual-interrupt hold logic (A4)"
```

(若此專案尚未初始化 git，改用一般檔案儲存即可，不必勉強建 repo。)

---

### Task 2: 把 `hurry/main.py` 接上 `GazeHoldTracker`，並在階段切換時重置（A5）

**Files:**
- Modify: `hurry/main.py:36-43`（import 區）、`hurry/main.py:296-313`（宣告區）、`hurry/main.py:687-699`（每幀套用區）、`hurry/main.py:399,424,445`（三處 `scoring.handle_stage_change` 呼叫點）

**Interfaces:**
- Consumes: Task 1 的 `GazeHoldTracker(hold_frames=10)`、`.update(raw_obj, raw_tester) -> (bool, bool)`、`.reset()`

- [ ] **Step 1: 加上 import**

在 `hurry/main.py:43`（`from modules.gaze_estimation.state_manager import GazeFSMManager` 那行）之後加一行：

```python
from modules.gaze_hold import GazeHoldTracker
```

- [ ] **Step 2: 把獨立變數換成 tracker 實例**

找到 `hurry/main.py:311-313`：

```python
    GAZE_HIT_HOLD_FRAMES = 10
    gaze_obj_hold = 0     # 看物品的遲滯倒數
    gaze_tester_hold = 0  # 看人/機器人的遲滯倒數
```

改成：

```python
    GAZE_HIT_HOLD_FRAMES = 10
    # 🌟 A4 修正（2026-09-15）：改用 GazeHoldTracker，讓「看物品」與「看施測者」
    # 兩側的遲滯互相打斷，避免同一時間戳被同時判定成立（見 modules/gaze_hold.py）
    gaze_hold_tracker = GazeHoldTracker(hold_frames=GAZE_HIT_HOLD_FRAMES)
```

- [ ] **Step 3: 把每幀套用邏輯換成呼叫 tracker**

找到 `hurry/main.py:687-699`：

```python
            if raw_gaze_obj:
                gaze_obj_hold = GAZE_HIT_HOLD_FRAMES
                child_is_gazing_at = True
            elif gaze_obj_hold > 0:
                gaze_obj_hold -= 1
                child_is_gazing_at = True

            if raw_gaze_tester:
                gaze_tester_hold = GAZE_HIT_HOLD_FRAMES
                child_is_gazing_at_tester = True
            elif gaze_tester_hold > 0:
                gaze_tester_hold -= 1
                child_is_gazing_at_tester = True
```

改成：

```python
            child_is_gazing_at, child_is_gazing_at_tester = gaze_hold_tracker.update(
                raw_gaze_obj, raw_gaze_tester
            )
```

- [ ] **Step 4: 在三個 `scoring.handle_stage_change` 呼叫點之後重置 tracker（A5）**

`hurry/main.py` 裡有三處呼叫（第 399、424、445 行附近），每一處呼叫之後都補一行 `gaze_hold_tracker.reset()`。例如第 399 行：

```python
                    scoring.handle_stage_change(current_stage, next_stage, next_detected_time)
```

改成：

```python
                    scoring.handle_stage_change(current_stage, next_stage, next_detected_time)
                    gaze_hold_tracker.reset()  # 🌟 A5 修正：避免上一關最後的遲滯狀態帶進新階段
```

對第 424 行、第 445 行附近的另外兩處 `scoring.handle_stage_change(...)` 呼叫也做同樣的事（各自照原本的縮排補一行 `gaze_hold_tracker.reset()`）。

- [ ] **Step 5: 語法編譯檢查**

Run:
```
C:\Users\wayne\miniconda3\condabin\conda.bat run -n mediapipe_py39 python -m py_compile hurry/main.py
```
Expected: 無輸出、離開碼 0（沒有語法錯誤）

- [ ] **Step 6: grep 自洽性檢查——確認舊變數完全清乾淨、新變數確實接上**

Run:
```
grep -n "gaze_obj_hold\|gaze_tester_hold\|gaze_hold_tracker" hurry/main.py
```
Expected：只看得到 `gaze_hold_tracker` 的宣告（1 處）、`.update(...)` 呼叫（1 處）、三處 `.reset()`——**完全看不到** `gaze_obj_hold` 或 `gaze_tester_hold` 這兩個舊名字。如果還有殘留，代表某處沒改乾淨，要修到乾淨為止。

- [ ] **Step 7: Commit**

```bash
git add hurry/main.py
git commit -m "refactor: wire GazeHoldTracker into hurry/main.py, reset on stage change"
```

---

### Task 3: 把 `main.py` 接上 `GazeHoldTracker`（跟 Task 2 對稱的改動）

**Files:**
- Modify: `main.py`（import 區、`main.py:293-295` 宣告區、`main.py:554-566` 每幀套用區、`main.py:318,331,345` 三處 `scoring.handle_stage_change` 呼叫點）

**Interfaces:**
- Consumes: 同 Task 2，`modules.gaze_hold.GazeHoldTracker`

- [ ] **Step 1: 確認 `main.py` 的 import 區塊位置，加上同樣的 import**

在 `main.py` 裡找 `from modules.gaze_estimation` 開頭的 import 那一段，加：

```python
from modules.gaze_hold import GazeHoldTracker
```

- [ ] **Step 2: 把 `main.py:293-295` 的獨立變數換成 tracker**

原本：

```python
    GAZE_HIT_HOLD_FRAMES = 10
    gaze_obj_hold = 0     # 看物品的遲滯倒數
    gaze_tester_hold = 0  # 看人/機器人的遲滯倒數
```

改成（跟 Task 2 Step 2 同樣的寫法）：

```python
    GAZE_HIT_HOLD_FRAMES = 10
    # 🌟 A4 修正（2026-09-15）：改用 GazeHoldTracker，讓「看物品」與「看施測者」
    # 兩側的遲滯互相打斷，避免同一時間戳被同時判定成立（見 modules/gaze_hold.py）
    gaze_hold_tracker = GazeHoldTracker(hold_frames=GAZE_HIT_HOLD_FRAMES)
```

- [ ] **Step 3: 把 `main.py:554-566` 的每幀套用邏輯換成呼叫 tracker**

原本：

```python
            if raw_gaze_obj:
                gaze_obj_hold = GAZE_HIT_HOLD_FRAMES
                child_is_gazing_at = True
            elif gaze_obj_hold > 0:
                gaze_obj_hold -= 1
                child_is_gazing_at = True

            if raw_gaze_tester:
                gaze_tester_hold = GAZE_HIT_HOLD_FRAMES
                child_is_gazing_at_tester = True
            elif gaze_tester_hold > 0:
                gaze_tester_hold -= 1
                child_is_gazing_at_tester = True
```

改成：

```python
            child_is_gazing_at, child_is_gazing_at_tester = gaze_hold_tracker.update(
                raw_gaze_obj, raw_gaze_tester
            )
```

- [ ] **Step 4: 在 `main.py:318、331、345` 三處 `scoring.handle_stage_change` 呼叫之後補 `gaze_hold_tracker.reset()`**

跟 Task 2 Step 4 做法相同，三處都要補。

- [ ] **Step 5: 語法編譯檢查**

Run:
```
C:\Users\wayne\miniconda3\condabin\conda.bat run -n mediapipe_py39 python -m py_compile main.py
```
Expected: 無輸出、離開碼 0

- [ ] **Step 6: grep 自洽性檢查**

Run:
```
grep -n "gaze_obj_hold\|gaze_tester_hold\|gaze_hold_tracker" main.py
```
Expected：同 Task 2 Step 6，完全看不到舊變數名字殘留。

- [ ] **Step 7: Commit**

```bash
git add main.py
git commit -m "refactor: wire GazeHoldTracker into main.py, reset on stage change"
```

---

### Task 4: A6——`MAX_GAZE_FALLBACK` 5 → 3

**Files:**
- Modify: `main.py:280`
- Modify: `hurry/main.py:296`

**Interfaces:** 無（純常數調整，不影響其他任務的介面）

- [ ] **Step 1: 修改 `hurry/main.py:296`**

原本：
```python
    MAX_GAZE_FALLBACK = 5         # 超過此值才清空 last_valid_gaze（防閃爍）
```
改成：
```python
    MAX_GAZE_FALLBACK = 3         # 🌟 A6 修正（2026-09-15）：原值 5 高於視線報告建議的 2~3，
                                  # 縮短容忍幀數以減少閃爍容忍期間吃到錯誤視線的機會
```

- [ ] **Step 2: 對 `main.py:280` 做同樣的修改**

- [ ] **Step 3: 語法編譯檢查**

Run:
```
C:\Users\wayne\miniconda3\condabin\conda.bat run -n mediapipe_py39 python -m py_compile main.py hurry/main.py
```
Expected: 無輸出、離開碼 0

- [ ] **Step 4: Commit**

```bash
git add main.py hurry/main.py
git commit -m "tune: shorten MAX_GAZE_FALLBACK from 5 to 3 (A6)"
```

---

### Task 5: C4 短解——Stage 1 fallback 門檻 15s → 22s

**Files:**
- Modify: `modules/scoring_engine.py:30`
- Test: `tests/test_scoring_engine_fallback.py`

**Interfaces:**
- Consumes: `modules.scoring_engine.ScoringEngine(cache_path, scoring_version, video_path)`、`.handle_stage_change(previous_stage, detected_stage, time_sec)`、`.update_frame(time_sec, current_stage, is_in_trigger_window, child_is_pointing_hit, child_is_gazing_at, child_is_gazing_at_tester, gaze_result, robot_rays, robot_boxes, yolo_boxes, is_gazing_at_box_func, tester_gaze_angles=None)`、`.trigger_event_records`（list of dict，每個 dict 有 `"stage"` 欄位）

- [ ] **Step 1: 寫失敗測試 `tests/test_scoring_engine_fallback.py`**

```python
# -*- coding: utf-8 -*-
import json
from modules.scoring_engine import ScoringEngine


def _make_empty_cache(tmp_path):
    cache_path = str(tmp_path / "empty_speech_cache.json")
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({"segment_records": [], "noise_events": []}, f, ensure_ascii=False)
    return cache_path


def _no_op_gaze_hit(gaze_result, box):
    return False


def test_stage1_fallback_waits_22_seconds_not_15(tmp_path):
    """C4 短解：15 秒門檻太短，會跟「你看」最慢近 20 秒才出現的關鍵字升級
    路徑打架（見 spec C4）。改成 22 秒，17.9 秒前後、22.1 秒前後個別驗證邊界。"""
    cache_path = _make_empty_cache(tmp_path)
    engine = ScoringEngine(cache_path=cache_path, scoring_version="test", video_path="dummy.mp4")

    # current_stage_enter_time 預設 0.0，Stage 1 一路沒有任何語音事件
    engine.update_frame(
        time_sec=17.9, current_stage=1, is_in_trigger_window=False,
        child_is_pointing_hit=False, child_is_gazing_at=False,
        child_is_gazing_at_tester=False, gaze_result=None, robot_rays=None,
        robot_boxes=[], yolo_boxes=[], is_gazing_at_box_func=_no_op_gaze_hit,
    )
    assert not any(r["stage"] == 1 for r in engine.trigger_event_records), (
        "17.9 秒還沒到新門檻（22 秒），不該有 fallback record"
    )

    engine.update_frame(
        time_sec=22.1, current_stage=1, is_in_trigger_window=False,
        child_is_pointing_hit=False, child_is_gazing_at=False,
        child_is_gazing_at_tester=False, gaze_result=None, robot_rays=None,
        robot_boxes=[], yolo_boxes=[], is_gazing_at_box_func=_no_op_gaze_hit,
    )
    assert any(r["stage"] == 1 for r in engine.trigger_event_records), (
        "過了 22 秒門檻應該要建立 fallback record"
    )
```

- [ ] **Step 2: 執行測試，確認第一個斷言就失敗**

Run:
```
C:\Users\wayne\miniconda3\condabin\conda.bat run -n mediapipe_py39 pytest tests/test_scoring_engine_fallback.py -v
```
Expected: `AssertionError: 17.9 秒還沒到新門檻...`（因為現在還是 15 秒門檻，17.9 秒時 fallback 早就建立了）

- [ ] **Step 3: 修改 `modules/scoring_engine.py:30`**

原本：
```python
FALLBACK_T0_DELAY_SEC_STAGE1 = 15.0
```
改成：
```python
FALLBACK_T0_DELAY_SEC_STAGE1 = 22.0
```

同時更新第 26-28 行的註解（原本寫「最慢近 20 秒才出現，4 秒門檻...」），在後面補一句：

```python
# 🌟 C4 修正（2026-09-15）：15 秒門檻與「你看」最慢近 20 秒才出現的
# 關鍵字升級路徑互相打架——fallback 在關鍵字之前就建立，若期間已經
# 計過 TB/TH，升級路徑的「三者皆為 None」條件就再也無法滿足，T0
# 永久卡在 fallback 時間，該關 RT 全部失真。改成 22 秒，讓關鍵字有
# 機會先到。
```

- [ ] **Step 4: 重新執行測試，確認通過**

Run:
```
C:\Users\wayne\miniconda3\condabin\conda.bat run -n mediapipe_py39 pytest tests/test_scoring_engine_fallback.py -v
```
Expected: `1 passed`

- [ ] **Step 5: Commit**

```bash
git add modules/scoring_engine.py tests/test_scoring_engine_fallback.py
git commit -m "fix: delay Stage 1 fallback T0 threshold from 15s to 22s (C4 short fix)"
```

---

### Task 6: C2 方向1——Stage 9/10 T0 關鍵字具體化

**Files:**
- Modify: `modules/scoring_engine.py:254`（Stage 9 關鍵字）、`modules/scoring_engine.py:295`（Stage 10 關鍵字）
- Test: `tests/test_scoring_engine_stage9_10_keywords.py`

**Interfaces:**
- Consumes: 同 Task 5 的 `ScoringEngine` 建構子；新增用到 `engine.stage_start_times`（dict，key 是 stage 編號）

- [ ] **Step 1: 寫失敗測試 `tests/test_scoring_engine_stage9_10_keywords.py`**

```python
# -*- coding: utf-8 -*-
import json
from modules.scoring_engine import ScoringEngine


def _make_cache(tmp_path, segment_records):
    cache_path = str(tmp_path / "speech_cache.json")
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({"segment_records": segment_records, "noise_events": []}, f, ensure_ascii=False)
    return cache_path


def _segment(text, start, keywords=None):
    return {
        "text": text,
        "trigger_events": [{
            "id": text + "-" + str(start),
            "event_type": "speech",
            "keywords": keywords or [],
            "start": start,
            "end": start + 1.0,
            "trigger_window": [start, start + 3.0],
        }],
    }


def test_stage9_t0_ignores_bare_hua_but_catches_specific_phrase(tmp_path):
    """C2 方向1：Stage 9 的 T0 不該被「這張圖畫得很漂亮」這種含裸字「畫」
    的無關句子誤觸發，只有「畫一幅／畫畫／畫好了」這類明確片語才算數。"""
    cache_path = _make_cache(tmp_path, [
        _segment("這張圖畫得很漂亮喔", start=5.0),   # 含裸字「畫」，不該被當成 Stage 9 起點
        _segment("我們來畫一幅畫吧", start=50.0),     # 明確片語，應該被當成 Stage 9 起點
    ])
    engine = ScoringEngine(cache_path=cache_path, scoring_version="test", video_path="dummy.mp4")
    assert engine.stage_start_times.get(9) == 50.0, (
        "Stage 9 的 T0 應該取自「畫一幅」這句，不是提早被裸字「畫」誤觸發在 5.0s"
    )


def test_stage10_t0_ignores_bare_san_but_catches_fireworks(tmp_path):
    """C2 方向1：Stage 10 的 T0 不該被「三個蘋果」這種含裸字「三」的無關
    句子誤觸發，移除裸的「三」與「3」，只留「煙火/烟火/321/三二一」。"""
    cache_path = _make_cache(tmp_path, [
        _segment("桌上有三個蘋果", start=10.0),  # 含裸字「三」，不該被當成 Stage 10 起點
        _segment("煙火要來囉準備看", start=60.0),  # 明確詞彙，應該被當成 Stage 10 起點
    ])
    engine = ScoringEngine(cache_path=cache_path, scoring_version="test", video_path="dummy.mp4")
    assert engine.stage_start_times.get(10) == 60.0, (
        "Stage 10 的 T0 應該取自「煙火」這句，不是提早被裸字「三」誤觸發在 10.0s"
    )
```

- [ ] **Step 2: 執行測試，確認兩個測試都失敗**

Run:
```
C:\Users\wayne\miniconda3\condabin\conda.bat run -n mediapipe_py39 pytest tests/test_scoring_engine_stage9_10_keywords.py -v
```
Expected: 兩個測試都 `AssertionError`（現在會分別讀到 5.0 跟 10.0，不是 50.0 跟 60.0）

- [ ] **Step 3: 修改 `modules/scoring_engine.py:254`（Stage 9 關鍵字）**

原本：
```python
        t9_all = []
        for e in self.speech_events:
            if any(k in e["text"] for k in ["畫", "画"]):
                t9_all.append(e["start"])
        t9_earliest = min(t9_all) if t9_all else float('inf')
```

改成：
```python
        t9_all = []
        # 🌟 C2 方向1 修正（2026-09-15）：裸字「畫/画」太容易被無關句子
        # 誤觸發（例："這張圖畫得很漂亮"），改成語意明確的完整片語。
        # 不能改用 e["keywords"] 精確比對——speech_engine 目前配對不到
        # 單字關鍵字，keywords 欄位只會出現「你看」「看這裡」，改了
        # Stage 9 會整關消失（見上方 C2 已撤回的紀錄）。
        for e in self.speech_events:
            if any(k in e["text"] for k in ["畫一幅", "畫畫", "畫好了", "画一幅", "画画", "画好了"]):
                t9_all.append(e["start"])
        t9_earliest = min(t9_all) if t9_all else float('inf')
```

- [ ] **Step 4: 修改 `modules/scoring_engine.py:295`（Stage 10 關鍵字）**

原本：
```python
        t10_all = []
        for e in self.speech_events:
            if any(k in e["text"] for k in ["煙火", "烟火", "321", "三二一", "三", "3"]):
                t10_all.append(e["start"])
```

改成：
```python
        t10_all = []
        # 🌟 C2 方向1 修正（2026-09-15）：移除裸的「三」與「3」——這是 spec
        # 點名最危險的一處，任何句子裡出現「三」或數字 3 都會誤觸發整個
        # Stage 10 判定窗口（T0+10s）落在錯誤時間點。保留語意明確的詞彙。
        for e in self.speech_events:
            if any(k in e["text"] for k in ["煙火", "烟火", "321", "三二一"]):
                t10_all.append(e["start"])
```

- [ ] **Step 5: 重新執行測試，確認通過**

Run:
```
C:\Users\wayne\miniconda3\condabin\conda.bat run -n mediapipe_py39 pytest tests/test_scoring_engine_stage9_10_keywords.py -v
```
Expected: `2 passed`

- [ ] **Step 6: 執行前面幾個任務累積的完整測試套件，確認沒有互相打架**

Run:
```
C:\Users\wayne\miniconda3\condabin\conda.bat run -n mediapipe_py39 pytest tests/ -v
```
Expected: 全部通過（Task 1 的 5 個 + Task 5 的 1 個 + Task 6 的 2 個 = 8 個）

- [ ] **Step 7: Commit**

```bash
git add modules/scoring_engine.py tests/test_scoring_engine_stage9_10_keywords.py
git commit -m "fix: make Stage 9/10 T0 keywords specific instead of bare characters (C2 direction 1)"
```

---

### Task 7: A7 附帶發現——澄清 pitch 閾值不一致（僅文件註解，不改行為）

**Files:**
- Modify: `modules/gaze_estimation/state_manager.py:56`（註解）、`state_manager.py:71,77`（註解）

**Interfaces:** 無（純註解，零行為改動）

**背景**：`state_manager.py:56` 把「向上看」定義成 `pitch < 0`，但 `EXTREME_TURNING` 盲追蹤的進入條件（第 71、77 行）用的是更寬鬆的 `last_pitch < 10`。這兩個門檻服務不同目的——第 56 行是「正常追蹤期，用來分類方向」，第 71/77 行是「已經追丟人臉，判斷要不要進入盲追蹤安全閥」——**不建議改動任何數值**：A7 的教訓證明收緊這個機制的任何嘗試都會先傷到真陽性（代償觸發的時機與小朋友真的轉頭看人高度重疊），而目前沒有任何實驗數據支持該往哪個方向調。這個任務只把「為什麼兩處不同」寫清楚，避免下一個維護者誤以為是筆誤而動手「修一致」。

- [ ] **Step 1: 在 `state_manager.py:56` 補註解**

原本：
```python
            if pitch < 0:  # 向上看（在你的系統坐標系中，頭部向上為負值）
```
改成：
```python
            # 🌟 閾值澄清（2026-09-15，對應 spec A7 附帶發現）：這裡用 pitch < 0，
            # 比下方 EXTREME_TURNING 進入條件的 last_pitch < 10 更嚴格。
            # 兩者刻意不同：這裡是「正常追蹤期」的方向分類，只在人臉還能
            # 穩定量測時使用；下方是「已經追丟人臉」時的最後一筆姿態回顧，
            # 用來判斷要不要進入盲追蹤安全閥，故意放寬以涵蓋更多可能情境。
            # 不要為了「一致性」把兩者改成同一個值——A7 的實驗教訓是收緊
            # 這一整條補償機制的任何嘗試都會先傷到真陽性（代償觸發的時機
            # 與小朋友真的轉頭看人高度重疊），目前沒有實驗數據支持該往哪個
            # 方向調，貿然「修一致」風險自負。
            if pitch < 0:  # 向上看（在你的系統坐標系中，頭部向上為負值）
```

- [ ] **Step 2: 在 `state_manager.py:71` 附近（EXTREME_TURNING 進入條件）補一句對應註解**

在第 69-70 行「依據盲區前一影格的運動向量外推意圖 / 臨界條件：Yaw 偏航超過 35° 且 Pitch 呈現上揚趨勢」後面加一行：

```python
                # （此處 last_pitch < 10 刻意比上方 FRONTAL 期間的 pitch < 0 寬鬆，
                #  見上方 update() 開頭 pitch < 0 那行的註解說明）
```

- [ ] **Step 3: 確認沒有任何行為改動——跑一次語法編譯即可**

Run:
```
C:\Users\wayne\miniconda3\condabin\conda.bat run -n mediapipe_py39 python -m py_compile modules/gaze_estimation/state_manager.py
```
Expected: 無輸出、離開碼 0

- [ ] **Step 4: Commit**

```bash
git add modules/gaze_estimation/state_manager.py
git commit -m "docs: clarify why FRONTAL and EXTREME_TURNING pitch thresholds differ (A7 follow-up, no behavior change)"
```

---

### Task 8: 10 支 benchmark 重跑，驗證 A1+A2+B1 v2 與本計畫全部改動

這個任務沒有程式碼可寫——它是把前面 7 個任務的改動，加上先前已經完成但還沒驗證的 A1+A2+B1 v2，一次送進真正的影片分析管線，用 `summarize/compare_with_manual.py` 對照人工標記表。這是 spec 文件裡「建議處理順序」第一順位的事，也是唯一能證明以上改動有沒有引入新迴歸的方法。

**Files:**
- 不修改程式碼，只執行既有的 `hurry/main.py` 與 `summarize/compare_with_manual.py`

- [ ] **Step 1: 確認語音快取還在，不要重跑 Whisper**

Run:
```
ls hurry/output/ | grep _speech_
```
Expected: 看到 10 支 benchmark 影片（42、49、52、64、66、74、82、86、90、104）各自的 `_speech_*` 資料夾。**這次的改動都沒有動到語音快取的格式或內容**（C2 方向1 只改文字比對用的關鍵字清單，不改快取檔案本身），所以應該完全沿用，不會觸發 Whisper 重跑。

- [ ] **Step 2: 備份目前的 `output`（如果裡面还有上一輪 v1 的結果）**

Run:
```
mv hurry/output hurry/output_A1_v1_backup   # 如果 output 底下已經是實驗三 v1 的結果
mkdir hurry/output
```
（若 `hurry/output` 目前是空的或本來就還沒放過任何一輪實驗結果，跳過這步。）

⚠️ 沿用 spec 開頭的快取保護提醒：**只搬 `*.txt`/`*.mp4`，`_speech_*` 資料夾一定要留在 `hurry/output/` 裡**，不要整個資料夾改名搬移。

- [ ] **Step 3: 執行批次分析**

Run:
```
C:\Users\wayne\miniconda3\condabin\conda.bat run -n mediapipe_py39 python hurry/main.py
```
Expected: 10 支 benchmark 影片依序跑完，各自產出 `{編號}.mp4`、`{編號}.txt` 到 `hurry/output/`。這一步會用到 GPU，時間視影片長度而定。

- [ ] **Step 4: 執行比對腳本，對照 `output_before2`（僅 P0 修正的固定基準）**

Run:
```
C:\Users\wayne\miniconda3\condabin\conda.bat run -n mediapipe_py39 python summarize\compare_with_manual.py --txt-dir hurry\output --baseline-dir hurry\output_before2
```

- [ ] **Step 5: 對照 spec 1-4 節「驗收重點」表逐項檢查**

| 看哪裡 | 預期 |
|--------|------|
| TH 的 `ΔFP` | 要是負的，特別看 Stage 1（原本 6 FP）、Stage 2（原本 5 FP）有沒有掉 |
| Stage 4 的 49-S4、66-S4、104-S4 | 必須保住（A1 目前唯一確定的貢獻）|
| Stage 8 的 FN | 別比 v1 的 49-S8、82-S8 更嚴重 |
| 影片上的 `x arm-align` | 有出現代表 B1 生效 |
| 影片上的 `(zone fallback)` | 大量出現代表 A1 的頭部框取不到 |
| **新增：Stage 3 的 TH** | A4 修正後，104-S3 這類「TH 跟 TB 同一時間戳」的案例應該消失或時間戳被拉開；不該讓 42-S3、52-S3 這兩個已知 FP 變得更糟 |
| **新增：Stage 1 的 RT（反應時間）** | C4 把門檻從 15s 延後到 22s，理論上不該讓任何一支影片的 Stage 1 T0 變成用 fallback（因為關鍵字通常在 22 秒內就到了）；若某支影片的 Stage 1 T0 來源從 keyword 退化成 ocr代償，代表那支影片的「你看」真的很晚才出現，需要另外檢查逐字稿 |
| **新增：Stage 9/10 的 T0 時間** | 對照 `hurry/output/{編號}.txt` 裡 Stage 9/10 的 T0 時間戳，跟改動前（`output_A1_v1_backup` 或更早的基準）比較，確認沒有被提早誤觸發的裸字「畫」「三」命中 |

- [ ] **Step 6: 記錄結果，更新 spec 文件的「零、現況總覽」表**

把這次驗證的結果（哪些項目變成 ✅ 已驗證有效、哪些需要進一步調整）寫回
`C:\Users\wayne\Desktop\project\integrate_v7.1\程式碼問題清單_對照人工AI差異.md`
的「零、現況總覽」表與「一、驗證紀錄」章節，格式比照現有的「實驗一/二/三」記法（日期、假設、結果表格、判定）。這一步不是選配——spec 文件本身的價值就在於持續記錄實驗結果，讓下一輪不用重新摸索。

---

## Self-Review 摘要（寫完計畫後的檢查）

- **Spec 覆蓋**：A4（Task 1-3）、A5（Task 2-3 的 reset）、A6（Task 4）、C4 短解（Task 5）、C2 方向1（Task 6）、A7 附帶的閾值不一致（Task 7，文件澄清）、A1+A2+B1 v2 驗證（Task 8）全部有對應任務。E1（使用者已決定這次不處理）、E5（檔案在這個環境不存在，已於文件開頭記錄）、A3/C6/D1/D3（spec 本身標記為需要 A1 基準穩定後才動、或非程式問題）刻意不放進這份計畫，留給下一輪。
- **Placeholder 掃描**：所有 Step 都有實際指令或程式碼，沒有「之後補」「參考 Task N」這類佔位敘述。
- **型別/介面一致性**：`GazeHoldTracker.update()` 的回傳順序 `(obj, tester)` 在 Task 1 定義、Task 2/3 使用時保持一致；`ScoringEngine` 的建構子與 `update_frame` 簽名在 Task 5、Task 6 兩處測試中完全一致（皆取自 `modules/scoring_engine.py` 目前的真實簽名，非憑空杜撰）。

---

Plan complete and saved to `docs/superpowers/plans/2026-09-15-scoring-engine-fixes.md`. Two execution options:

1. **Subagent-Driven (recommended)** - dispatch a fresh subagent per task, review between tasks, fast iteration
2. **Inline Execution** - execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
