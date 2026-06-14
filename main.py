import os
import sys
import cv2
import gc
import traceback
import numpy as np # 🌟 新增 numpy 以產生暖機圖片

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from modules.speech import SpeechTrigger
from modules.signboard import SignboardTracker
from modules.models_manager import ModelManager
from modules.interaction import InteractionEngine

# ==========================================
# ★ 1. 基本設定與路徑
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(BASE_DIR, 'video', '10.mp4')
MODEL_DIR = os.path.join(BASE_DIR, 'model')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    
OUTPUT_VIDEO_PATH = os.path.join(OUTPUT_DIR, 'output_result_final5.mp4')
EVENT_LOG_PATH = os.path.join(OUTPUT_DIR, 'event_record5.txt') # 📝 事件紀錄檔輸出路徑

def main():
    print("==================================================")
    print("🚀 多模態 AI 互動行為分析系統 v5.0 (全時段視覺+事件紀錄)...")
    print("==================================================")

    if not os.path.exists(VIDEO_PATH):
        sys.exit(f"❌ 錯誤：找不到影片：{VIDEO_PATH}")

    # --- 系統初始化 ---
    print("\n>>> [系統] 正在啟動聽覺與視覺模組...")
    speech = SpeechTrigger(
        video_path=VIDEO_PATH, 
        output_dir=OUTPUT_DIR, 
        keywords=["開始", "321", "準備", "你看", "看這裡", "準備囉", "機器人", "怪聲", "嗶", "逼", "[聲音]", "放煙火"]
    )
    trigger_windows = speech.get_trigger_windows()

    sign_tracker = SignboardTracker(allowlist='12345678')
    
    # 🌟 新增：EasyOCR 暖機代碼 (消除初次框選後的卡頓延遲)
    print(">>> [系統] 正在為 OCR 引擎暖機，消除硬體啟動延遲...")
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    sign_tracker.reader.readtext(dummy_img) # 讓 GPU 提前分配記憶體並載入模型

    model_manager = ModelManager(model_dir=MODEL_DIR)
    pose_path = os.path.join(MODEL_DIR, 'yolo11n-pose.pt')
    interaction = InteractionEngine(pose_model_path=pose_path, sma_window=5)

    # --- 影片串流設定 ---
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened(): sys.exit("❌ 無法開啟影片")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_w, frame_h))

    success, first_frame = cap.read()
    if success: sign_tracker.initialize_roi(first_frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    cv2.namedWindow("Multi-Modal AI System Preview", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Multi-Modal AI System Preview", 1280, 720)

    # --- 變數與狀態追蹤 (用於防重複紀錄) ---
    frame_count = 0
    current_stage = 0
    
    event_logs = []  # 儲存所有發生的事件
    prev_keyword_state = False  # 紀錄上一幀是否在語音窗內
    prev_child_hit_state = False # 紀錄上一幀小孩是否有指到

    print("\n>>> [系統] 開始進入主分析迴圈...")
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
                
            frame_count += 1
            current_time_sec = frame_count / fps

            # 1. 狀態判定
            is_in_trigger_window = speech.is_in_window(current_time_sec, trigger_windows)
            
            # A. 視覺判斷 (OCR)
            detected_stage = sign_tracker.detect_stage(frame)
            if detected_stage is not None:
                if detected_stage != current_stage:
                    event_logs.append(f"[{current_time_sec:.1f}s] 階段改變：視覺模組確認進入 第 {detected_stage} 階段")
                current_stage = detected_stage

            # ==========================================
            # 🌟 B. 聽覺代償機制 (防呆：沒放 8 號牌子時用語音/怪聲推進)
            # ==========================================
            # 只有在第 7 階段時，才允許怪聲或特定語音觸發進入第 8 階段
            if current_stage == 7:
                WEIRD_SOUND_TRIGGERS = ["怪聲", "嗶", "逼", "[聲音]", "機器人"]
                
                # 檢查當下時間點，是否有觸發上述任何一個怪聲或口令
                is_override_triggered = any(
                    speech.check_voice_override(current_time_sec, keyword=kw) 
                    for kw in WEIRD_SOUND_TRIGGERS
                )
                
                if is_override_triggered:
                    print(f"\n>>> [Voice Override] {current_time_sec:.1f}s 聽覺模組偵測到怪聲/口令，代償啟動！切換至階段 8")
                    event_logs.append(f"[{current_time_sec:.1f}s] 聽覺代償：偵測到怪聲/口令，強制切換至第 8 階段")
                    
                    # 呼叫 tracker 強制切換，清空視覺模組的舊記憶與追蹤框
                    sign_tracker.force_stage(8) 
                    current_stage = 8

            sign_tracker.draw_boxes(frame, current_stage)

            # 2. 全時段視覺偵測 (不管有沒有關鍵字都執行)
            child_is_pointing_hit = False
            try:
                if current_stage > 0:
                    yolo_boxes = model_manager.detect_objects(frame, stage=current_stage)
                    # 取得回傳狀態：小孩是否有命中關鍵物
                    child_is_pointing_hit = interaction.analyze_interaction(frame, yolo_boxes)
            except Exception as e:
                print(f"⚠️ 偵測跳過 (Frame {frame_count}): {e}")

            # 3. 📝 事件追蹤與紀錄檔寫入 (避免每幀重複紀錄)
            if is_in_trigger_window and not prev_keyword_state:
                event_logs.append(f"[{current_time_sec:.1f}s] 觸發：偵測到引導語音關鍵字")
            prev_keyword_state = is_in_trigger_window

            if child_is_pointing_hit and not prev_child_hit_state:
                event_logs.append(f"[{current_time_sec:.1f}s] 互動：小朋友成功指向當下場景的物品 (Stage {current_stage})")
            prev_child_hit_state = child_is_pointing_hit

            # 4. 🖥️ 左上角 UI 資訊面板繪製
            c_text = (0, 255, 255) # 黃色一般字體
            c_key = (0, 255, 0) if is_in_trigger_window else (150, 150, 150)
            c_hit = (0, 255, 0) if child_is_pointing_hit else (0, 0, 255)

            cv2.putText(frame, f"Time:  {current_time_sec:.1f} s", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, c_text, 2)
            cv2.putText(frame, f"Stage: {current_stage}", (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, c_text, 2)
            
            keyword_text = "YES (Active)" if is_in_trigger_window else "NO (Idle)"
            cv2.putText(frame, f"Keyword Detected: {keyword_text}", (15, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, c_key, 2)
            
            hit_text = "YES!" if child_is_pointing_hit else "NO"
            cv2.putText(frame, f"Child Pointing Hit: {hit_text}", (15, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.8, c_hit, 2)

            # 5. 輸出與記憶體管理
            out.write(frame)
            cv2.imshow("Multi-Modal AI System Preview", frame)
            
            if frame_count % 100 == 0: gc.collect()
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('r'):  
                current_stage = 0
                sign_tracker.current_stage = 0
                event_logs.append(f"[{current_time_sec:.1f}s] 手動重置階段為 0")

    except Exception as e:
        print(f"\n❌ [系統致命錯誤] 崩潰: {e}")
        traceback.print_exc()
    finally:
        # 6. 📝 儲存事件紀錄檔
        print(f"\n>>> [系統] 正在匯出事件紀錄檔至 {EVENT_LOG_PATH} ...")
        with open(EVENT_LOG_PATH, 'w', encoding='utf-8') as f:
            f.write("=== 互動行為分析事件紀錄表 ===\n")
            f.write(f"影片來源: {VIDEO_PATH}\n")
            f.write("--------------------------------\n")
            for log in event_logs:
                f.write(log + "\n")

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print("✅ 測試結束，影片與紀錄檔皆已安全存檔！")

if __name__ == "__main__":
    main()