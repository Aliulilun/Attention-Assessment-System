# 0708 怪聲（Stage 8）誤判修正

## 問題

`transcript_with_events.txt` 中，怪聲事件實際判定視窗為 `263.750s ~ 273.750s`，
但 Stage 8（手機怪聲）卻在 `4.200s ~ 7.200s` 就被觸發。

## 根因

1. **Whisper 幻覺**：影片開頭 [00:00-00:04] 沒有清楚語音，Whisper 把
   `transcribe()` 的 `initial_prompt`（內含「請勿忽略短促的聲音，不要與雜音合併。」）
   逐字複誦回來當作辨識結果。
2. 這段幻覺文字剛好含有「聲音」，命中關鍵字清單裡的 `"[聲音]"`（經正規化後等於「聲音」）。
3. `scoring_engine.py` 判定 Stage 8 起點時，對整句轉錄文字做「怪聲/聲音/声音」子字串搜尋，
   再取所有候選時間的 `min()`——幻覺片段的 4.2s 因此蓋過真正的雜音事件（263.75s）。
4. 另外，`speech_engine.py` 裡的 `detect_template_noise_event()`（全片滑動視窗比對參考音檔）
   即使找到的「最相似」片段相似度其實很低，也一定會回傳一個候選，等於每支影片都可能生出假的怪聲事件。

## 修正內容

1. `speech_engine.py`
   - 新增 `is_prompt_echo()`，過濾與 `initial_prompt` 高度重疊的幻覺片段
   - `detect_template_noise_event()` 補上相似度門檻檢查
   - 移除 `select_alarm_noise_events()` 中多餘的早期 return
   - 相似度門檻 0.58 / 0.65 → **0.85**
   - `MATCHING_ALGORITHM_VERSION` 13 → **15**（強制舊快取失效）
   - Whisper 改為自動偵測 CUDA
   - `.wav` 參考樣本直接讀取，免轉檔

2. `scoring_engine.py`
   - Stage 8 起點候選改用精確關鍵字比對 `"怪聲" in e["keywords"]`，不再對整句轉錄文字做子字串搜尋

3. `speech.py`
   - 新增 `EXPECTED_MATCHING_ALGORITHM_VERSION` 快取版本檢查（版本不符自動忽略舊快取重跑）
   - 相似度門檻同步改 **0.85**

4. `main.py`
   - 怪聲參考音檔路徑改指向新的 `model/noisesample/noise.wav`

## 驗證

- 4 個檔案皆 `py_compile` 通過。
- 用實際 bug 情境（幻覺文字「請勿忽略短促的聲音：」+ 一批真實對話句子）測試 `is_prompt_echo()`：
  幻覺片段正確判定為 `True`（會被過濾），所有真實語句皆為 `False`（不誤殺正常語音）。
- 模擬 Stage 8 候選邏輯：即使語音中混入舊的誤判事件與真正提到「怪聲」的合理語句，
  取到的起點仍正確落在雜音事件時間，不會被拉早。

## 附註

同一批修正也合併了組員（TEAM_UPDATE_2026-07-08）對 `speech_engine.py` / `speech.py` 的更新，
详見上方表格；`hand_landmarker.task` 路徑問題（`interaction.py`）為另一個獨立 bug，不屬於本次怪聲修正範圍。
