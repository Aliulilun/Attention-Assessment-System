import ssl
ssl._create_default_https_context = ssl._create_unverified_context
#我的電腦不知道為什麼語音辨識模型，會有驗證失敗的問題 所以打了前面這行，但你們刪掉應該沒差

import os
import sys
import cv2
import gc
import traceback
import numpy as np
# import pandas as pd  # 暫時不需要儲存視線數據
import yaml  # 🌟 新增用於讀取配置文件

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from modules.speech import SpeechTrigger
from modules.signboard import SignboardTracker
from modules.models_manager import ModelManager
from modules.interaction import InteractionEngine
from modules.gaze_estimation import GazeEstimationPipeline  # 🌟 新增視線估計模組
from modules.gaze_estimation.visualization import draw_gaze_with_face_box  # 🌟 新增視線可視化


# ==========================================
# 🌟 視線-物體交集判定函數 (Ray Casting)
# ==========================================
def ray_intersects_box(origin, dir_vec, box):
    """
    精準碰撞：Ray-AABB 射線與矩形邊界框交集演算法
    只有當射線「實體穿過」指定的 box 時才會回傳 True
    
    Args:
        origin: 射線起點 (x, y)
        dir_vec: 射線方向向量 (dx, dy)
        box: 邊界框 (x1, y1, x2, y2)
    
    Returns:
        bool: True 表示射線穿過邊界框
    """
    ox, oy = origin
    dx, dy = dir_vec
    x1, y1, x2, y2 = box
    
    # 避免除以零的錯誤
    dx = dx if dx != 0 else 1e-5
    dy = dy if dy != 0 else 1e-5
    
    tx1 = (x1 - ox) / dx
    tx2 = (x2 - ox) / dx
    ty1 = (y1 - oy) / dy
    ty2 = (y2 - oy) / dy
    
    tmin = max(min(tx1, tx2), min(ty1, ty2))
    tmax = min(max(tx1, tx2), max(ty1, ty2))
    
    # tmax >= 0 確保是往前看，且 tmax >= tmin 代表有交集
    return tmax >= max(0, tmin)


def is_gazing_at_box(gaze_result, object_bbox):
    """
    判斷視線是否落在物體上
    
    Args:
        gaze_result: 從 gaze_pipeline.estimate() 返回的結果
        object_bbox: 物體邊界框 (x1, y1, x2, y2)
    
    Returns:
        bool: True 表示正在注視該物體
    """
    if not gaze_result or not gaze_result.get('success'):
        return False
    
    # 獲取視線起點（雙眼中心）
    left_eye = gaze_result.get('left_eye')
    right_eye = gaze_result.get('right_eye')
    
    if left_eye is None or right_eye is None:
        return False
    
    eye_center = (
        (left_eye[0] + right_eye[0]) / 2,
        (left_eye[1] + right_eye[1]) / 2
    )
    
    # 獲取視線方向向量（2D投影）
    gaze_vector = gaze_result['gaze_vector']  # [x, y, z]
    direction = (gaze_vector[0], gaze_vector[1])
    
    # 使用 Ray Casting 判定
    return ray_intersects_box(eye_center, direction, object_bbox)


def check_gaze_on_objects(gaze_result, yolo_boxes):
    """
    檢查視線是否落在任何物體上
    
    Args:
        gaze_result: 視線估計結果
        yolo_boxes: YOLO 偵測到的物體邊界框列表
    
    Returns:
        bool: True 表示正在注視至少一個物體
    """
    for box in yolo_boxes:
        if is_gazing_at_box(gaze_result, box):
            return True
    return False

# ==========================================
# ★ 1. 基本設定與路徑
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 🌟 新增：讀取配置文件
CONFIG_PATH = os.path.join(BASE_DIR, 'config.yaml')
try:
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        CONFIG = yaml.safe_load(f)
    print(f">>> [系統] 成功載入配置文件: {CONFIG_PATH}")
except FileNotFoundError:
    print(f"⚠️ [系統] 警告：找不到配置文件 {CONFIG_PATH}，使用默認配置")
    CONFIG = {}

VIDEO_PATH = os.path.join(BASE_DIR, CONFIG.get('video', {}).get('input_path', 'video/10.mp4'))
MODEL_DIR = os.path.join(BASE_DIR, 'model')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    
OUTPUT_VIDEO_PATH = os.path.join(OUTPUT_DIR, CONFIG.get('output', {}).get('video_path', 'output/output_result_final.mp4').split('/')[-1])
EVENT_LOG_PATH = os.path.join(OUTPUT_DIR, CONFIG.get('output', {}).get('event_log_path', 'output/event_record.txt').split('/')[-1])
# GAZE_CSV_PATH = os.path.join(OUTPUT_DIR, CONFIG.get('output', {}).get('gaze_csv_path', 'output/gaze_data.csv').split('/')[-1])  # 暫時不輸出視線數據

def main():
    print("==================================================")
    print("🚀 多模態 AI 互動行為分析系統 v5.0 (全時段視覺+事件紀錄)...")
    print("==================================================")

    if not os.path.exists(VIDEO_PATH):
        sys.exit(f"❌ 錯誤：找不到影片：{VIDEO_PATH}")

    # --- 系統初始化 ---
    print("\n>>> [系統] 正在啟動聽覺與視覺模組...")
    
    # 語音觸發
    keywords = CONFIG.get('speech', {}).get('keywords', ["開始", "321", "準備", "你看", "看這裡", "準備囉", "機器人"])
    speech = SpeechTrigger(
        video_path=VIDEO_PATH, 
        output_dir=OUTPUT_DIR, 
        keywords=keywords
    )
    trigger_windows = speech.get_trigger_windows()

    # 階段偵測
    allowlist = CONFIG.get('signboard', {}).get('allowlist', '12345678')
    sign_tracker = SignboardTracker(allowlist=allowlist)
    
    # 🌟 EasyOCR 暖機代碼
    print(">>> [系統] 正在為 OCR 引擎暖機，消除硬體啟動延遲...")
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    sign_tracker.reader.readtext(dummy_img)

    # 物件偵測
    model_manager = ModelManager(model_dir=MODEL_DIR)
    pose_path = os.path.join(MODEL_DIR, CONFIG.get('yolo_models', {}).get('pose_model', 'yolo11n-pose.pt'))
    sma_window = CONFIG.get('interaction', {}).get('sma_window', 5)
    interaction = InteractionEngine(pose_model_path=pose_path, sma_window=sma_window)
    
    # 🌟 新增：視線估計模組初始化
    print("\n>>> [系統] 正在啟動視線估計模組...")
    gaze_config = CONFIG.get('gaze_estimation', {})
    gaze_pipeline = GazeEstimationPipeline(config=gaze_config)
    # gaze_data_list = []  # 暫時不儲存視線數據

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
    prev_gaze_state = False      # 🌟 新增：紀錄上一幀視線是否有注視物體

    print("\n>>> [系統] 開始進入主分析迴圈...")
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
                
            frame_count += 1
            current_time_sec = frame_count / fps

            # 1. 狀態判定
            is_in_trigger_window = speech.is_in_window(current_time_sec, trigger_windows)
            
            detected_stage = sign_tracker.detect_stage(frame)
            if detected_stage is not None:
                if detected_stage != current_stage:
                    event_logs.append(f"[{current_time_sec:.1f}s] 階段改變：進入 第 {detected_stage} 階段")
                current_stage = detected_stage

            if speech.check_voice_override(current_time_sec, keyword="機器人"):
                if current_stage != 8:
                    event_logs.append(f"[{current_time_sec:.1f}s] 語音強制覆寫：進入 第 8 階段 (機器人)")
                current_stage = 8

            sign_tracker.draw_boxes(frame, current_stage)

            # 2. 全時段視覺偵測 (不管有沒有關鍵字都執行)
            child_is_pointing_hit = False
            yolo_boxes = []  # 🌟 新增：儲存 YOLO 偵測結果供後續視線判定使用
            try:
                if current_stage > 0:
                    yolo_boxes = model_manager.detect_objects(frame, stage=current_stage)
                    # 取得回傳狀態：小孩是否有命中關鍵物
                    child_is_pointing_hit = interaction.analyze_interaction(frame, yolo_boxes)
            except Exception as e:
                print(f"⚠️ 偵測跳過 (Frame {frame_count}): {e}")
            
            # 🌟 新增：視線估計（在觸發窗口內或階段 > 0 時執行）
            gaze_result = None
            child_is_gazing_at = False  # 🌟 新增：視線注視狀態
            if is_in_trigger_window or current_stage > 0:
                try:
                    gaze_result = gaze_pipeline.estimate(frame)
                    
                    if gaze_result and gaze_result.get('success'):
                        # 🌟 繪製視線向量到畫面上
                        face_bbox = gaze_result.get('face_bbox')
                        pitch_rad = gaze_result['gaze_angles'][0]  # 弧度
                        yaw_rad = gaze_result['gaze_angles'][1]    # 弧度
                        gaze_vector = gaze_result['gaze_vector']
                        confidence = gaze_result.get('confidence', 0.0)
                        left_eye = gaze_result.get('left_eye')
                        right_eye = gaze_result.get('right_eye')
                        
                        if face_bbox is not None:
                            frame = draw_gaze_with_face_box(
                                frame,
                                face_bbox,
                                pitch_rad,
                                yaw_rad,
                                gaze_vector=gaze_vector,
                                left_eye=left_eye,
                                right_eye=right_eye,
                                confidence=confidence,
                                show_angles=True,
                                show_direction_label=False,  # 不顯示方向標籤（Center/Up/Down等）
                                show_gaze_vector=True,
                                bbox_format='xyxy'  # ETH-XGaze 使用 xyxy 格式
                            )
                        
                        # 🌟 新增：視線注視判定（使用 Ray Casting）
                        if current_stage > 0 and len(yolo_boxes) > 0:
                            child_is_gazing_at = check_gaze_on_objects(gaze_result, yolo_boxes)
                            
                            # 如果正在注視物體，高亮顯示該物體
                            if child_is_gazing_at:
                                for box in yolo_boxes:
                                    if is_gazing_at_box(gaze_result, box):
                                        # 繪製黃色粗框標記正在注視的物體
                                        cv2.rectangle(frame, 
                                                    (int(box[0]), int(box[1])),
                                                    (int(box[2]), int(box[3])),
                                                    (0, 255, 255), 5)  # 黃色粗框
                                        
                                        # 在物體上方顯示「GAZING!」標記
                                        cv2.putText(frame, "GAZING!", 
                                                  (int(box[0]), int(box[1])-35),
                                                  cv2.FONT_HERSHEY_SIMPLEX, 1.2, 
                                                  (0, 255, 255), 3)
                except Exception as e:
                    if frame_count % 100 == 0:  # 每100幀才報告一次錯誤
                        print(f"⚠️ 視線估計跳過 (Frame {frame_count}): {e}")

            # 3. 📝 事件追蹤與紀錄檔寫入 (避免每幀重複紀錄)
            if is_in_trigger_window and not prev_keyword_state:
                event_logs.append(f"[{current_time_sec:.1f}s] 觸發：偵測到引導語音關鍵字")
            prev_keyword_state = is_in_trigger_window

            if child_is_pointing_hit and not prev_child_hit_state:
                event_logs.append(f"[{current_time_sec:.1f}s] 互動：小朋友成功指向當下場景的物品 (Stage {current_stage})")
            prev_child_hit_state = child_is_pointing_hit
            
            # 🌟 新增：視線注視事件記錄
            if child_is_gazing_at and not prev_gaze_state:
                event_logs.append(f"[{current_time_sec:.1f}s] 視線：正在注視階段 {current_stage} 物體")
            prev_gaze_state = child_is_gazing_at


            # 4. 🖥️ 左上角 UI 資訊面板繪製
            c_text = (0, 255, 255) # 黃色一般字體
            c_key = (0, 255, 0) if is_in_trigger_window else (150, 150, 150)
            c_hit = (0, 255, 0) if child_is_pointing_hit else (0, 0, 255)
            c_gaze = (0, 255, 0) if gaze_result and gaze_result.get('success') else (150, 150, 150)
            c_gazing = (0, 255, 255) if child_is_gazing_at else (0, 0, 255)  # 🌟 新增：視線注視狀態顏色

            cv2.putText(frame, f"Time:  {current_time_sec:.1f} s", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, c_text, 2)
            cv2.putText(frame, f"Stage: {current_stage}", (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, c_text, 2)
            
            keyword_text = "YES (Active)" if is_in_trigger_window else "NO (Idle)"
            cv2.putText(frame, f"Keyword Detected: {keyword_text}", (15, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, c_key, 2)
            
            hit_text = "YES!" if child_is_pointing_hit else "NO"
            cv2.putText(frame, f"Child Pointing Hit: {hit_text}", (15, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.8, c_hit, 2)
            
            # 🌟 新增：視線估計資訊顯示
            if gaze_result and gaze_result.get('success'):
                pitch = gaze_result['gaze_angles_deg'][0]
                yaw = gaze_result['gaze_angles_deg'][1]
                cv2.putText(frame, f"Gaze: P={pitch:.1f} Y={yaw:.1f}", (15, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, c_gaze, 2)
            else:
                cv2.putText(frame, "Gaze: N/A", (15, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, c_gaze, 2)
            
            # 🌟 新增：視線注視狀態顯示
            gazing_text = "YES!" if child_is_gazing_at else "NO"
            cv2.putText(frame, f"Child Gazing At Object: {gazing_text}", (15, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.8, c_gazing, 2)

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
        
        # 🌟 暫時不儲存視線數據到 CSV（已在畫面上顯示）
        # if gaze_data_list:
        #     print(f"\n>>> [系統] 正在匯出視線數據至 {GAZE_CSV_PATH} ...")
        #     gaze_df = pd.DataFrame(gaze_data_list)
        #     gaze_df.to_csv(GAZE_CSV_PATH, index=False, encoding='utf-8')
        #     print(f"✅ 視線數據已成功儲存！共 {len(gaze_data_list)} 筆記錄")

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print("\n✅ 測試結束，影片與事件紀錄檔已安全存檔！")

if __name__ == "__main__":
    main()