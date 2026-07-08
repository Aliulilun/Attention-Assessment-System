import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os
import sys
import cv2
import gc
import traceback
import numpy as np 
import subprocess # 保留功能：用於呼叫 FFmpeg 進行音軌縫合
import yaml       # 保留功能：用於讀取配置文件

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from modules.speech import SpeechTrigger
from modules.signboard import SignboardTracker
from modules.models_manager import ModelManager
from modules.interaction import InteractionEngine
from modules.scoring_engine import ScoringEngine

# 視線估計模組與視覺化工具
from modules.gaze_estimation import GazeEstimationPipeline  
from modules.gaze_estimation.visualization import draw_gaze_with_face_box  

# 導入時序有限狀態機管理員
from modules.gaze_estimation.state_manager import GazeFSMManager 

# ==========================================
# 🌟 視線-物體交集判定函數 (Ray Casting)
# ==========================================
def ray_intersects_box(origin, dir_vec, box):
    ox, oy = origin
    dx, dy = dir_vec
    x1, y1, x2, y2 = box
    
    dx = dx if dx != 0 else 1e-5
    dy = dy if dy != 0 else 1e-5
    
    tx1 = (x1 - ox) / dx
    tx2 = (x2 - ox) / dx
    ty1 = (y1 - oy) / dy
    ty2 = (y2 - oy) / dy
    
    tmin = max(min(tx1, tx2), min(ty1, ty2))
    tmax = min(max(tx1, tx2), max(ty1, ty2))
    
    return tmax >= max(0, tmin)

def is_gazing_at_box(gaze_result, object_bbox):
    if not gaze_result or not gaze_result.get('success'):
        return False
    
    left_eye = gaze_result.get('left_eye')
    right_eye = gaze_result.get('right_eye')
    if left_eye is None or right_eye is None: return False
    
    eye_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
    gaze_vector = gaze_result['gaze_vector']  
    direction = (gaze_vector[0], gaze_vector[1])
    
    return ray_intersects_box(eye_center, direction, object_bbox)

def check_gaze_on_objects(gaze_result, yolo_boxes):
    for box in yolo_boxes:
        if is_gazing_at_box(gaze_result, box): return True
    return False


# ==========================================
# ★ 1. 基本設定與路徑
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 載入 YAML 設定檔 (若沒有檔案也不會報錯)
CONFIG_PATH = os.path.join(BASE_DIR, 'config.yaml')
try:
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        CONFIG = yaml.safe_load(f)
    print(f">>> [系統] 成功載入配置文件: {CONFIG_PATH}")
except FileNotFoundError:
    CONFIG = {}

VIDEO_PATH = os.path.join(BASE_DIR, CONFIG.get('video', {}).get('input_path', 'video/51.mp4'))
MODEL_DIR = os.path.join(BASE_DIR, 'model')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    
TEMP_VIDEO_PATH = os.path.join(OUTPUT_DIR, 'temp_no_audio_1to14_5.mp4')
OUTPUT_VIDEO_PATH = os.path.join(OUTPUT_DIR, 'output_result_final_1to14_5.mp4')
EVENT_LOG_PATH = os.path.join(OUTPUT_DIR, 'event_record_1to14_5.txt')

# 採用 main.py 升級後的版本參數
SCORING_VERSION = '1-14_ABSOLUTE_TEXT_V12_EXPANDED'


def main():
    print("==================================================")
    print("🚀 多模態 AI 互動行為分析系統 (狀態機盲追蹤 + 1-14無縫全通版)...")
    print("==================================================")
    print(f">>> [評分系統] Scoring Version: {SCORING_VERSION}")

    if not os.path.exists(VIDEO_PATH):
        sys.exit(f"❌ 錯誤：找不到影片：{VIDEO_PATH}")

    # --- 系統初始化 ---
    print("\n>>> [系統] 正在啟動聽覺模組與讀取時間軸...")
    
    # 🌟 採用 main.py 全字典字庫
    # 🌟 修改：移除 "怪聲", "嗶", "逼", "[聲音]"
    #          這些是「聲音符號」而非口說詞語，留著會讓 Whisper 音效記號被當關鍵字觸發。
    #          Stage 7→8 怪聲觸發改由 noise.wav 模板比對（is_in_noise_window）負責。
    speech = SpeechTrigger(
        video_path=VIDEO_PATH,
        output_dir=OUTPUT_DIR,
        keywords=["開始", "321", "三二一", "準備", "你看", "小朋友", "看這裡", "準備囉", "機器人", "放煙火", "煙火", "放烟火", "烟火", "三", "畫一幅", "画一幅", "画1幅", "畫", "画", "畫好了", "画好了", "特別的畫"],
        # 🌟 修改：路徑改為 model/noise.wav（原 model/noisesample/noise.wav 不存在）
        noise_sample_path=os.path.join(MODEL_DIR, 'noise.wav')
    )
    trigger_windows = speech.get_trigger_windows()
    
    scoring = ScoringEngine(
        cache_path=speech.cache_path,
        scoring_version=SCORING_VERSION,
        video_path=VIDEO_PATH,
    )

    # 💡 只有 1~7 階段允許透過牌子偵測 (main.py 設定)
    sign_tracker = SignboardTracker(allowlist='1234567')
    
    print(">>> [系統] 正在為 OCR 引擎暖機，消除硬體啟動延遲...")
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    sign_tracker.reader.readtext(dummy_img) 

    # 模型大腦：負責自動切換各階段的各式模型
    model_manager = ModelManager(model_dir=MODEL_DIR)
    pose_path = os.path.join(MODEL_DIR, 'yolo11n-pose.pt')
    # 🌟 修改：明確傳入 hand_model_path，避免 Windows junction/symlink 下 __file__
    #          解析到真實路徑（C:\Users\wayne\Desktop\project\project_v3）找不到模型
    hand_path = os.path.join(MODEL_DIR, 'hand_landmarker.task')
    interaction = InteractionEngine(pose_model_path=pose_path, hand_model_path=hand_path, sma_window=5)

    print(">>> [系統] 正在啟動視線估計模組與時序狀態機...")
    gaze_config = CONFIG.get('gaze_estimation', {})
    gaze_pipeline = GazeEstimationPipeline(config=gaze_config)
    fsm = GazeFSMManager(history_len=5, max_hold_frames=30)

    # --- 影片串流設定 ---
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened(): sys.exit("❌ 無法開啟影片")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out = cv2.VideoWriter(TEMP_VIDEO_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_w, frame_h))

    success, first_frame = cap.read()
    if success: sign_tracker.initialize_roi(first_frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    cv2.namedWindow("Multi-Modal AI System Preview", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Multi-Modal AI System Preview", 1280, 720)

    frame_count = 0
    current_stage = 0
    event_logs = scoring.event_logs

    # 🌟 新增：Stage 11/12 近端物品「固定 ROI 框」
    # 原理：桌面近端物品位置固定，不依賴 YOLO 偵測，改用硬寫比例框直接覆蓋桌面範圍。
    # 座標以幀寬高的比例定義，適配任意解析度。
    # 如需調整框位：修改下方四個比例值（0.0~1.0），重新執行即可。
    #   LEFT_RATIO  = 左邊界 / 幀寬   RIGHT_RATIO = 右邊界 / 幀寬
    #   TOP_RATIO   = 上邊界 / 幀高   BOT_RATIO   = 下邊界 / 幀高
    FIXED_ROI_11_12_LEFT  = 0.0    # ← 桌面左界 (畫面最左)
    FIXED_ROI_11_12_RIGHT = 1.0    # ← 桌面右界 (畫面最右)
    FIXED_ROI_11_12_TOP   = 0.45   # ← 桌面上界 (涵蓋畫面下半部，確保右側較高處的物件進入搜尋範圍)
    FIXED_ROI_11_12_BOT   = 1.0    # ← 桌面下界 (畫面底部)
    fixed_roi_11_12 = (
        int(frame_w * FIXED_ROI_11_12_LEFT),
        int(frame_h * FIXED_ROI_11_12_TOP),
        int(frame_w * FIXED_ROI_11_12_RIGHT),
        int(frame_h * FIXED_ROI_11_12_BOT),
    )
    locked_boxes_11_12 = {}  # {stage_num: [box_list]} — 永久鎖定：一旦偵測到就不再更新

    # 場地界線：左側 35% 區域視為施測者區
    divider_x = int(frame_w * 0.35)
    TESTER_ZONE_BBOX = [0, 0, divider_x, frame_h]

    print("\n>>> [系統] 開始進入主分析迴圈...")
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
                
            frame_count += 1
            current_time_sec = frame_count / fps
            is_in_trigger_window = speech.is_in_window(current_time_sec, trigger_windows)

            # ==================================================
            # 1. 🌟 階段判定大腦 (OCR vs 絕對時間軸 vs 聽覺代償)
            # ==================================================
            # A. 視覺判斷 (OCR) - 限定 1~7
            if current_stage < 8:
                detected_stage = sign_tracker.detect_stage(frame)
                if detected_stage is not None and detected_stage <= 7:
                    if detected_stage != current_stage:
                        scoring.handle_stage_change(current_stage, detected_stage, current_time_sec)
                        current_stage = detected_stage

            # B. 絕對時間軸推進 (自動對齊 ScoringEngine 設定)
            expected_stage = current_stage
            if hasattr(scoring, 'stage_start_times'):
                for s_idx in sorted(scoring.stage_start_times.keys()):
                    if current_time_sec >= scoring.stage_start_times[s_idx]:
                        if s_idx > expected_stage: 
                            expected_stage = s_idx
                            
            if expected_stage > current_stage:
                print(f"\n>>> [時間軸推進] {current_time_sec:.1f}s 系統強制推進至 第 {expected_stage} 階段")
                scoring.handle_stage_change(current_stage, expected_stage, current_time_sec)
                current_stage = expected_stage

            # C. 聽覺代償機制 (階段 7 -> 8)
            # 🌟 修改：怪聲觸發改為 noise.wav 模板比對命中（is_in_noise_window），
            #          不再依賴 Whisper 關鍵字（[聲音]/嗶/逼），避免音效符號誤觸發
            if current_stage == 7:
                if speech.is_in_noise_window(current_time_sec):
                    print(f"\n>>> [Voice Override] {current_time_sec:.1f}s noise.wav 命中，代償啟動！切換至階段 8")
                    event_logs.append(f"[{current_time_sec:.1f}s] 聽覺代償：noise.wav 命中，強制切換至第 8 階段")
                    sign_tracker.force_stage(8)
                    if hasattr(scoring, 'handle_stage_override'):
                        scoring.handle_stage_override(current_stage, 8, current_time_sec)
                    else:
                        scoring.handle_stage_change(current_stage, 8, current_time_sec)
                    current_stage = 8

            sign_tracker.draw_boxes(frame, current_stage)

            # ==================================================
            # 2. 視覺偵測 (YOLO) 與模型派發
            # ==================================================
            child_is_pointing_hit = False
            yolo_boxes = []  
            robot_boxes = []
            
            try:
                if current_stage > 0:

                    # 🌟 Stage 11/12：在固定 ROI 範圍內裁切畫面後再跑 YOLO
                    # 固定框只是「搜尋範圍」，YOLO 在裁切區域內偵測到的才是真正的物品框
                    if current_stage in [11, 12]:
                        rx1, ry1, rx2, ry2 = fixed_roi_11_12

                        # 視覺化：黃色框顯示搜尋範圍
                        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 200, 0), 2)
                        cv2.putText(frame, f"Search ROI (Stage {current_stage})",
                                    (rx1 + 5, ry1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

                        if current_stage in locked_boxes_11_12:
                            # ✅ 已鎖定：直接用鎖定的框，完全跳過 YOLO
                            yolo_boxes = locked_boxes_11_12[current_stage]
                        else:
                            # 尚未鎖定：裁切 ROI 後跑 YOLO
                            roi_crop = frame[ry1:ry2, rx1:rx2]
                            roi_result = model_manager.detect_objects(roi_crop, stage=current_stage)
                            if isinstance(roi_result, tuple) and len(roi_result) == 2:
                                roi_boxes, robot_boxes = roi_result
                            else:
                                roi_boxes = roi_result

                            # 座標轉回原始幀空間
                            yolo_boxes = [(x1 + rx1, y1 + ry1, x2 + rx1, y2 + ry1)
                                          for (x1, y1, x2, y2) in roi_boxes]

                            # 🔒 一旦偵測到就永久鎖定，後續幀不再執行 YOLO
                            if len(yolo_boxes) > 0:
                                locked_boxes_11_12[current_stage] = list(yolo_boxes)
                                print(f">>> [Stage {current_stage}] 🔒 目標物已鎖定：{yolo_boxes}")

                        # 機器人框對完整幀另外取（不受 ROI 限制）
                        full_result = model_manager.detect_objects(frame, stage=current_stage)
                        if isinstance(full_result, tuple) and len(full_result) == 2:
                            _, robot_boxes = full_result

                    else:
                        # 其他階段：正常對完整幀偵測
                        detect_result = model_manager.detect_objects(frame, stage=current_stage)
                        if current_stage >= 9 and isinstance(detect_result, tuple) and len(detect_result) == 2:
                            yolo_boxes, robot_boxes = detect_result
                        else:
                            yolo_boxes = detect_result

                    # 🌟 視覺化：YOLO 偵測到的目標物品（鎖定=青色粗框 / 一般=綠色細框）
                    is_locked_box = current_stage in locked_boxes_11_12
                    box_color = (0, 255, 255) if is_locked_box else (0, 255, 0)
                    box_thick = 3 if is_locked_box else 2
                    for box in yolo_boxes:
                        bx1, by1, bx2, by2 = map(int, box)
                        cv2.rectangle(frame, (bx1, by1), (bx2, by2), box_color, box_thick)
                        if current_stage == 9:
                            label = "Tablet (Stage 9)"
                        elif is_locked_box:
                            label = f"LOCKED (Stage {current_stage})"
                        else:
                            label = f"Target (Stage {current_stage})"
                        cv2.putText(frame, label, (bx1, by1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
                        
                    # 🌟 視覺化回饋：將「機器人」畫上橘色專屬追蹤框
                    for box in robot_boxes:
                        bx1, by1, bx2, by2 = map(int, box)
                        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 165, 255), 2) # 橘色框
                        cv2.putText(frame, "Robot", (bx1, by1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

                    # 🌟 修改：移除 len(yolo_boxes) > 0 的前置條件
                    #    骨架偵測（YOLO-Pose）與手部偵測（MediaPipe）應始終執行，
                    #    yolo_boxes 為空時 analyze_interaction 仍正常運行，只是無射線碰撞判定。
                    child_is_pointing_hit = interaction.analyze_interaction(frame, yolo_boxes)
            except Exception as e:
                if frame_count % 100 == 0: 
                    print(f"⚠️ 偵測跳過 (Frame {frame_count}): {e}")

            # ==================================================
            # 3. 👁️ 核心整合重構：視線估計與時序狀態機幾何級聯
            # ==================================================
            gaze_result = None
            child_is_gazing_at = False 
            child_is_gazing_at_tester = False
            face_result_for_fsm = None
            pose_result_for_fsm = None

            if is_in_trigger_window or current_stage > 0:
                try:
                    gaze_result = gaze_pipeline.estimate(frame)
                    
                    if gaze_result:
                        if gaze_result.get('success'):
                            # ─── 情況 A：正常追蹤期（面部特徵點完備） ───
                            face_bbox = gaze_result.get('face_bbox')
                            pitch_rad = gaze_result['gaze_angles'][0]
                            yaw_rad = gaze_result['gaze_angles'][1]
                            gaze_vector = gaze_result['gaze_vector']
                            pitch_deg = gaze_result['gaze_angles_deg'][0]
                            yaw_deg = gaze_result['gaze_angles_deg'][1]
                            confidence = gaze_result.get('confidence', 0.0)
                            left_eye = gaze_result.get('left_eye')
                            right_eye = gaze_result.get('right_eye')
                            
                            if face_bbox is not None:
                                frame = draw_gaze_with_face_box(
                                    frame, face_bbox, pitch_rad, yaw_rad,
                                    gaze_vector=gaze_vector, left_eye=left_eye, right_eye=right_eye,
                                    confidence=confidence, show_angles=True,
                                    show_direction_label=False, show_gaze_vector=True, bbox_format='xyxy'
                                )
                            
                            # 標準幾何射線要求（Ray Casting 碰撞判定 - 目標物）
                            if current_stage > 0 and len(yolo_boxes) > 0:
                                child_is_gazing_at = check_gaze_on_objects(gaze_result, yolo_boxes)
                                
                                # 命中物體繪製黃色外框
                                if child_is_gazing_at:
                                    for box in yolo_boxes:
                                        if is_gazing_at_box(gaze_result, box):
                                            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 255), 5)
                                            cv2.putText(frame, "GAZING!", (int(box[0]), int(box[1])-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                            
                            # ==================================================
                            # 🌟 動態 TH 判定邏輯 (看回施測者 vs 看回機器人)
                            # ==================================================
                            if current_stage >= 9:
                                # 第 9~14 階段：TH 目標強制改為「看回機器人」
                                if len(robot_boxes) > 0:
                                    child_is_gazing_at_tester = check_gaze_on_objects(gaze_result, robot_boxes)
                                    if child_is_gazing_at_tester:
                                        # 視覺化回饋：看中機器人
                                        for box in robot_boxes:
                                            if is_gazing_at_box(gaze_result, box):
                                                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 4)
                                                cv2.putText(frame, "GAZING AT ROBOT (TH)!", (int(box[0]), int(box[1])-35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            else:
                                # 第 1~8 階段：維持原版，TH 目標為「看回左方施測者區」
                                if is_gazing_at_box(gaze_result, TESTER_ZONE_BBOX):
                                    if pitch_deg > -5 and yaw_deg > 10:
                                        child_is_gazing_at_tester = True
                                        cv2.rectangle(
                                            frame,
                                            (TESTER_ZONE_BBOX[0], TESTER_ZONE_BBOX[1]),
                                            (TESTER_ZONE_BBOX[2], TESTER_ZONE_BBOX[3]),
                                            (0, 0, 0),
                                            3
                                        )
                                        cv2.putText(
                                            frame,
                                            "GAZING AT TESTER!",
                                            (TESTER_ZONE_BBOX[0] + 250, TESTER_ZONE_BBOX[1] + 30),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.8,
                                            (0, 0, 0),
                                            2
                                        )

                            # 完整封裝傳遞給狀態機的資料
                            face_result_for_fsm = {
                                'yolo_head_bbox': gaze_result.get('yolo_head_bbox'),
                                'num_landmarks': gaze_result.get('num_landmarks', 468) 
                            }
                            pose_result_for_fsm = {'success': True, 'euler_angles': gaze_result.get('head_pose')}
                        
                        else:
                            # ─── 情況 B：特徵點丟失，但前端 YOLO 依舊鎖定頭部 ───
                            if 'face_result' in gaze_result:
                                face_result_for_fsm = gaze_result['face_result']
                            pose_result_for_fsm = None
                            
                except Exception as e:
                    if frame_count % 100 == 0: 
                        print(f"⚠️ 視線估計跳過 (Frame {frame_count}): {e}")

            # 呼叫時序狀態機進行記憶更新
            fsm_target = fsm.update(face_result_for_fsm, pose_result_for_fsm)

            # 處理極端轉頭維持期的代償邏輯與紫色 UI 渲染
            if fsm.current_state == "EXTREME_TURNING":
                if current_stage in [3, 4] and fsm_target in ["LEFT_BACK_UPPER", "RIGHT_BACK_UPPER"]:
                    child_is_gazing_at = True
                
                if face_result_for_fsm is not None and face_result_for_fsm.get('yolo_head_bbox') is not None:
                    x1, y1, x2, y2 = map(int, face_result_for_fsm['yolo_head_bbox'])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)  
                    cv2.putText(frame, "YOLO Head Only", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)


            # ==================================================
            # 4. 📝 事件追蹤與紀錄檔寫入
            # ==================================================
            tester_gaze_angles = None
            if gaze_result and gaze_result.get('success'):
                tester_gaze_angles = (
                    gaze_result['gaze_angles_deg'][0],
                    gaze_result['gaze_angles_deg'][1],
                )

            scoring.update_frame(
                time_sec=current_time_sec,
                current_stage=current_stage,
                is_in_trigger_window=is_in_trigger_window,
                child_is_pointing_hit=child_is_pointing_hit,
                child_is_gazing_at=child_is_gazing_at,
                child_is_gazing_at_tester=child_is_gazing_at_tester,
                gaze_result=gaze_result,
                robot_rays=[], # 舊機器人射線已徹底拔除
                robot_boxes=robot_boxes, 
                yolo_boxes=yolo_boxes,
                is_gazing_at_box_func=is_gazing_at_box,
                tester_gaze_angles=tester_gaze_angles,
            )


            # ==================================================
            # 5. 🖥️ 左上角 UI 資訊面板繪製
            # ==================================================
            c_text = (0, 255, 255) 
            c_key = (0, 255, 0) if is_in_trigger_window else (150, 150, 150)
            c_hit = (0, 255, 0) if child_is_pointing_hit else (0, 0, 255)
            c_gaze = (0, 255, 0) if gaze_result and gaze_result.get('success') else (150, 150, 150)
            c_gazing = (0, 255, 255) if child_is_gazing_at else (0, 0, 255) 

            cv2.putText(frame, f"Time:  {current_time_sec:.1f} s", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, c_text, 2)
            cv2.putText(frame, f"Stage: {current_stage} (1-7 OCR / 8-14 Auto)", (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, c_text, 2)
            
            keyword_text = "YES (Active)" if is_in_trigger_window else "NO (Idle)"
            cv2.putText(frame, f"Keyword Detected: {keyword_text}", (15, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, c_key, 2)
            
            hit_text = "YES!" if child_is_pointing_hit else "NO"
            cv2.putText(frame, f"Child Pointing Hit: {hit_text}", (15, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.8, c_hit, 2)

            if gaze_result and gaze_result.get('success'):
                pitch = gaze_result['gaze_angles_deg'][0]
                yaw = gaze_result['gaze_angles_deg'][1]
                cv2.putText(frame, f"Gaze: P={pitch:.1f} Y={yaw:.1f}", (15, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, c_gaze, 2)
            else:
                status_str = f"Gaze: Blind Tracking ({fsm.current_state})" if fsm.current_state == "EXTREME_TURNING" else "Gaze: N/A"
                cv2.putText(frame, status_str, (15, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            
            gazing_text = "YES!" if child_is_gazing_at else "NO"
            cv2.putText(frame, f"Child Gazing At Object: {gazing_text}", (15, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.8, c_gazing, 2)

            current_stage_gazing_count = scoring.stage_gazing_counts.get(current_stage, 0)
            score_text = f"Score {scoring.total_score} | S{current_stage} Hit {current_stage_gazing_count} | Total {scoring.total_gazing_events}"
            cv2.putText(frame, score_text, (15, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, c_text, 2)

            # 保留 main_scoring 版進階的 Task 標記對照字典
            preview_label_map = {
                "怪聲": "Noise",
                "Stage 8 起始": "Stage8 Start",
                "機器人煙火秀": "Firework",
                "機指近物1": "Near1",
                "機指近物2": "Near2",
                "機指遠物1": "Far1",
                "機指遠物2": "Far2",
            }
            latest_record = scoring.trigger_event_records[-1] if scoring.trigger_event_records else None
            if latest_record:
                raw_label = latest_record.get("label", "Task")
                label = preview_label_map.get(raw_label, raw_label if raw_label.startswith("Stage") else raw_label)
                if latest_record.get("record_type") == "time":
                    task_text = f"Task {label} @ {latest_record['time']:.2f}s"
                else:
                    task_parts = [f"Task {label}", f"T0 {latest_record['t0']:.2f}"]
                    if latest_record.get("tb") is not None:
                        task_parts.append(f"TB {latest_record['tb']:.2f}")
                    if latest_record.get("th") is not None:
                        task_parts.append(f"TH {latest_record['th']:.2f}")
                    task_text = " | ".join(task_parts)
            else:
                task_text = "Task -"
            cv2.putText(frame, task_text, (15, 285), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 6. 輸出與記憶體管理
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
        print(f"\n>>> [系統] 正在匯出事件紀錄檔至 {EVENT_LOG_PATH} ...")
        scoring.write_report(EVENT_LOG_PATH)

        cap.release()
        if 'out' in locals() and out is not None: out.release()
        cv2.destroyAllWindows()
        
        # ==========================================
        # 🎵 7. 影音縫合階段 (Audio Muxing via FFmpeg)
        # ==========================================
        print("\n>>> [系統] 視覺畫面渲染完成，開始處理音軌縫合...")
        if os.path.exists(TEMP_VIDEO_PATH):
            ffmpeg_cmd = [
                'ffmpeg', '-y', 
                '-i', TEMP_VIDEO_PATH,
                '-i', VIDEO_PATH,
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-map', '0:v:0',
                '-map', '1:a:0',
                '-shortest', 
                OUTPUT_VIDEO_PATH
            ]
            
            try:
                print(f">>> [系統] 正在使用 FFmpeg 將聲音縫合回影片，這只要幾秒鐘...")
                subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                if os.path.exists(TEMP_VIDEO_PATH): os.remove(TEMP_VIDEO_PATH)
                print(f"\n✅ [系統大功告成] 最終有聲影片已存至：{OUTPUT_VIDEO_PATH}")
            except FileNotFoundError:
                print("\n❌ [錯誤] 找不到 FFmpeg！系統無法合併音軌。")
                print("👉 請確保您已下載 FFmpeg 並設定環境變數 (PATH)。")
                print(f"👉 您的無聲影片已保留在：{TEMP_VIDEO_PATH}")
            except subprocess.CalledProcessError as e:
                print(f"\n❌ [錯誤] FFmpeg 合併失敗。")
                print(f"👉 您的無聲影片已保留在：{TEMP_VIDEO_PATH}")
        else:
            print("❌ [錯誤] 找不到暫存的無聲影片，無法合併音軌.")
            
if __name__ == "__main__":
    main()