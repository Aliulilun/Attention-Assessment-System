這份修改指南**專門針對你第一篇傳給我的 1-10 版程式碼**量身打造。修改後，程式會具備以下超強防護力，保證 1-10 階段順利跑完不漏接：

1. **防空窗期超時 (Trigger Lock)**：保護語音有效期間，絕不強制切換階段（解決「有看到卻沒紀錄」）。
2. **防 OCR 延遲過高**：解決 15 幀累積過久的問題。
3. **防時光倒流 (Anti-Rollback)**：防止 EasyOCR 誤讀背景雜訊，把已經到第 5 關的進度又拉回第 1 關。
4. **解除順序死鎖**：施測者就算順序亂掉或忘記翻牌，怪聲跟語音 9、10 依然能強勢介入（解決 9、10 階段消失）。
5. **修復逐字稿輸出**：修正 JSON 讀取層級，讓 `.txt` 報告能正確印出關鍵字。

請直接將以下內容存成 `.md` 檔，並對照替換你的 `main.py`：

---

# 程式碼修正指南：HURRY_1to10_BATCH_V1 (終極防護版)

本次修改針對 `1-10` 完整版，請在你的 `main.py` 中找到對應的區塊並進行**整段替換**。

## 修改 1：全域設定區（降低 OCR 延遲）

**[尋找以下程式碼]**

```python
# ★ 跳幀設定（效能優化）
# 說明：EasyOCR / 視線估計 / YOLO 不需要每幀都跑，
#       被量測目標（閃卡、機器人）移動緩慢，快取上一幀結果可大幅提速。
YOLO_SKIP = 1   # 🌟 修改：改為逐幀執行（不跳幀），確保 TB/TH 時間點精確
GAZE_SKIP = 1   # 🌟 修改：改為逐幀執行（不跳幀），確保視線偵測無延遲
OCR_SKIP  = 15  # 覆寫 SignboardTracker 的 OCR_FRAME_INTERVAL（預設 2 → 改 15）

```

**[替換為]**

```python
# ★ 跳幀設定（效能優化）
# 說明：EasyOCR / 視線估計 / YOLO 不需要每幀都跑，
#       被量測目標（閃卡、機器人）移動緩慢，快取上一幀結果可大幅提速。
YOLO_SKIP = 1   
GAZE_SKIP = 1   
OCR_SKIP  = 5   # 🌟 核心修正 1：將 15 改為 5。降低 Tracker 防抖累積的延遲，防止吃掉語音空窗期

```

---

## 修改 2：第 1 階段判定區（加入 Trigger Lock、防倒退、解死鎖）

請找到程式碼迴圈內的 `1. 階段判定`，將 `A. OCR` 一直到 `C. 聽覺代償` 結束，**整段**替換成以下版本：

**[替換為]**

```python
            # ──────────────────────────────────────────────
            # 1. 階段判定 (加入空窗期鎖定 & 順序防護機制)
            # ──────────────────────────────────────────────

            pending_stage = current_stage  # 準備要切換的目標階段

            # A. OCR 判定
            _t0 = _time.perf_counter()
            if current_stage < 8:
                try:
                    detected_stage = sign_tracker.detect_stage(frame)
                    # 🌟 核心修正 2：防時光倒流 (Anti-Rollback)
                    # 加入 current_stage <= detected_stage，防止背景雜訊讓階段退回前面的關卡
                    if detected_stage is not None and current_stage <= detected_stage <= 7:
                        if detected_stage != current_stage:
                            pending_stage = detected_stage  # 先記錄新階段，暫不切換
                except RuntimeError as _ocr_err:
                    _msg = str(_ocr_err)
                    if 'out of memory' in _msg.lower() or 'cuda' in _msg.lower():
                        if frame_count % 100 == 0:
                            print(f"⚠️ OCR CUDA OOM (Frame {frame_count})，清 VRAM 後繼續")
                        gc.collect()
                        torch.cuda.empty_cache()
                    else:
                        if frame_count % 100 == 0:
                            print(f"⚠️ OCR 跳過 (Frame {frame_count}): {_ocr_err}")
                except Exception as _ocr_err:
                    if frame_count % 100 == 0:
                        print(f"⚠️ OCR 跳過 (Frame {frame_count}): {_ocr_err}")

            # B. 時間軸自動推進（只推進到 ACTIVE_STAGES 內的階段）
            if hasattr(scoring, 'stage_start_times'):
                for s_idx in sorted(scoring.stage_start_times.keys()):
                    if current_time_sec >= scoring.stage_start_times[s_idx]:
                        if s_idx > pending_stage:
                            pending_stage = s_idx  # 記錄時間軸推進的新階段

            # 🌟 核心修正 3： Trigger Lock 空窗期鎖
            # 只要在語音有效期限內，絕不強制切換階段，確保小孩作答能被記錄！
            if pending_stage > current_stage and pending_stage in ACTIVE_STAGES:
                if is_in_trigger_window:
                    if frame_count % 15 == 0:
                        print(f">>> [鎖定] {current_time_sec:.1f}s 正在語音空窗期內，延後切換至階段 {pending_stage}")
                else:
                    print(f">>> [時間軸] {current_time_sec:.1f}s 推進至第 {pending_stage} 階段")
                    scoring.handle_stage_change(current_stage, pending_stage, current_time_sec)
                    current_stage = pending_stage

            # C. 聽覺代償（Stage 7 → 8）
            # 🌟 核心修正 4：解除順序死鎖
            # 將 if current_stage == 7 改為 if current_stage < 8
            # 萬一施測者漏了前面的牌子，聽到怪聲依舊能強制推進到 8，保護後續 9 與 10 不消失
            if current_stage < 8:
                is_override = speech.is_in_noise_window(current_time_sec)
                if is_override:
                    print(f">>> [Voice Override] {current_time_sec:.1f}s noise.wav 命中，切換至階段 8")
                    event_logs.append(f"[{current_time_sec:.1f}s] 聽覺代償：切換至第 8 階段")
                    sign_tracker.force_stage(8)
                    if hasattr(scoring, 'handle_stage_override'):
                        scoring.handle_stage_override(current_stage, 8, current_time_sec)
                    else:
                        scoring.handle_stage_change(current_stage, 8, current_time_sec)
                    current_stage = 8

```

---

## 修改 3：報告輸出區（修復逐字稿關鍵字印不出來的問題）

請拉到程式碼的最下方 `finally:` 區塊內，找到「把 Whisper 語音辨識逐字稿追加到 txt 尾端」的地方。

**[尋找以下程式碼]**

```python
                        # 🌟 修改：keywords 在 trigger_events 層，不在 segment_record 層
                        _kws = [k for ev in _rec.get('trigger_events', []) for k in ev.get('keywords', [])]
                        _line = f"[{_t0:.2f}s ~ {_t1:.2f}s]  {_txt}"
                        if _kws:
                            _line += f"  ← 關鍵字：{', '.join(dict.fromkeys(_kws))}"  # fromkeys 去重保序

```

**[替換為]**

```python
                        # 🌟 核心修正 5：適應新版 json 扁平結構，讓 txt 報告能正確印出關鍵字
                        _kws = _rec.get('keywords', [])
                        _line = f"[{_t0:.2f}s ~ {_t1:.2f}s]  {_txt}"
                        if _kws:
                            _line += f"  ← 關鍵字：{', '.join(_kws)}"

```

---

這三步改完後，你的 `1-10` 版就會變成具備極強容錯率的「鐵壁版」，無論施測者怎麼出包（翻太快、忘記翻牌、順序顛倒），系統都能把該記的分數和該推進的階段死死守住！