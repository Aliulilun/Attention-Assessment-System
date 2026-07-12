import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os
import sys
import cv2
import gc
import traceback
import numpy as np
import subprocess
import yaml
import torch  # 🌟 新增：用於 CUDA 狀態偵測與 VRAM 清理

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# 🌟 壓制 MediaPipe/glog 遙測上傳失敗的噪音 log（Clearcut uploader，與程式邏輯無關）
os.environ["GLOG_minloglevel"]      = "3"   # 0=INFO 1=WARNING 2=ERROR 3=FATAL；設 3 只顯示致命錯誤
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"   # 同時壓制 TensorFlow 底層 log

# ============================================================
# ★ 批次版：量測第 1-10 階段
#    - ACTIVE_STAGES = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
#    - 批次掃描 video/ 資料夾，一次跑完所有影片
#    - 輸出 output/{原始檔名}.mp4 + output/{原始檔名}.txt
#    - MODEL_DIR 指向 C:\project\model（不需複製模型檔）
# ============================================================

# 讓 Python 找到原始專案的 modules
PROJECT_DIR = r'C:\project'
sys.path.insert(0, PROJECT_DIR)

from modules.speech import SpeechTrigger
from modules.signboard import SignboardTracker
from modules.models_manager import ModelManager
from modules.interaction import InteractionEngine
from modules.scoring_engine import ScoringEngine
from modules.gaze_estimation import GazeEstimationPipeline
from modules.gaze_estimation.visualization import draw_gaze_with_face_box
from modules.gaze_estimation.state_manager import GazeFSMManager

# ============================================================
# ★ 全域設定
# ============================================================
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))  # C:\project\hurry
MODEL_DIR   = os.path.join(PROJECT_DIR, 'model')          # 指向原始專案模型目錄
VIDEO_DIR   = os.path.join(BASE_DIR, 'video')
OUTPUT_DIR  = os.path.join(BASE_DIR, 'output')
CONFIG_PATH = os.path.join(PROJECT_DIR, 'config.yaml')    # 使用原始專案的 config

ACTIVE_STAGES   = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}   # ★ 量測第 1-10 階段
SHOW_PREVIEW    = True                 # ★ False = 關閉預覽視窗（批次加速，省 I/O）
SCORING_VERSION = 'HURRY_1to10_BATCH_V1'

# ★ 跳幀設定（效能優化）
# 說明：EasyOCR / 視線估計 / YOLO 不需要每幀都跑，
#       被量測目標（閃卡、機器人）移動緩慢，快取上一幀結果可大幅提速。
YOLO_SKIP = 3   # 每 N 幀才執行一次 YOLO 物件偵測（物件幾乎不動，快取複用安全）
GAZE_SKIP = 2   # 每 N 幀才執行一次視線估計（TB/TH 判定容許 2 幀延遲，不影響精度）
OCR_SKIP  = 15  # 覆寫 SignboardTracker 的 OCR_FRAME_INTERVAL（預設 2 → 改 15）
                # 牌子在畫面靜止數秒，每 0.5s 偵測一次即可

# ★ 怪聲參考音檔（放在 model/ 目錄，跨所有影片共用）
# 🌟 修改：speech.py 新增 noise_sample_path 參數，此處明確傳入固定路徑；
#          若檔案不存在則傳 None → speech_engine 退回純頻譜特徵偵測
NOISE_SAMPLE_PATH = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model', 'noisesample', 'noise.wav')
)
# 等同於 C:\project\model\noisesample\noise.wav

SPEECH_KEYWORDS = [
    "開始", "321", "三二一", "準備", "你看", "小朋友", "看這裡", "準備囉",
    # 🌟 修改：移除 "怪聲", "嗶", "逼", "[聲音]"
    #          這些是「聲音符號」而非口說詞語，Whisper 遇到怪聲時會自行寫入 [聲音]，
    #          留在 SPEECH_KEYWORDS 裡會讓 Whisper 的音效記錄被當關鍵字觸發時間窗。
    #          怪聲觸發改由 noise.wav 模板比對（is_in_noise_window）負責。
    "機器人", "放煙火", "煙火", "放烟火", "烟火",
    "三", "畫一幅", "画一幅", "画1幅", "畫", "画", "畫好了", "画好了", "特別的畫"
]

# 🌟 修改：怪聲觸發（Stage 7→8）改為純 noise.wav 模板比對，
#          此清單保留「機器人」（口說詞語，仍需 Whisper 偵測）供其他覆寫邏輯使用
WEIRD_SOUND_TRIGGERS = ["機器人"]


# ============================================================
# ★ 視線-物體交集判定函數 (Ray Casting)
# ============================================================
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
    left_eye  = gaze_result.get('left_eye')
    right_eye = gaze_result.get('right_eye')
    if left_eye is None or right_eye is None:
        return False
    eye_center   = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
    gaze_vector  = gaze_result['gaze_vector']
    direction    = (gaze_vector[0], gaze_vector[1])
    return ray_intersects_box(eye_center, direction, object_bbox)

def check_gaze_on_objects(gaze_result, yolo_boxes):
    for box in yolo_boxes:
        if is_gazing_at_box(gaze_result, box):
            return True
    return False


# ============================================================
# ★ 單一影片處理函數
# ============================================================
def process_single_video(video_path, output_dir, model_manager, interaction,
                          gaze_pipeline, gaze_config, sign_tracker):
    """
    處理單一影片，輸出:
      output/{basename}.mp4  — 有聲分析影片
      output/{basename}.txt  — 事件紀錄
    """
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    temp_path      = os.path.join(output_dir, f'_temp_{video_basename}.mp4')
    out_video_path = os.path.join(output_dir, f'{video_basename}.mp4')
    out_txt_path   = os.path.join(output_dir, f'{video_basename}.txt')

    print(f"\n{'='*60}")
    print(f"📹  [{video_basename}]  開始處理...")
    print(f"{'='*60}")

    # --- 語音（每支影片用獨立子目錄存 speech_cache.json） ---
    # 🌟 修改：改用 per-video 子目錄，避免批次模式下多影片共用同一份快取
    #          （若共用，Video2 會讀到 Video1 的 Whisper 逐字稿 → 觸發時間窗全錯）
    #          搭配 speech.py 的快取先讀取優化，重複執行同一批次時 Whisper 子行程
    #          完全不會被啟動（省每支影片 30~120s 的模型冷啟動開銷）
    video_speech_dir = os.path.join(output_dir, f'_speech_{video_basename}')
    os.makedirs(video_speech_dir, exist_ok=True)

    speech = SpeechTrigger(
        video_path=video_path,
        output_dir=video_speech_dir,        # ← per-video 子目錄，隔離各影片快取
        keywords=SPEECH_KEYWORDS,
        noise_sample_path=NOISE_SAMPLE_PATH if os.path.exists(NOISE_SAMPLE_PATH) else None,
    )
    trigger_windows = speech.get_trigger_windows()

    scoring = ScoringEngine(
        cache_path=speech.cache_path,
        scoring_version=SCORING_VERSION,
        video_path=video_path,
    )

    # --- 牌子追蹤：直接重置傳入的共用 tracker，不重新載入 EasyOCR ---
    # 🌟 sign_tracker 由 main() 建立一次並傳入，這裡只重置影片狀態
    sign_tracker.reset()

    # --- 互動引擎：重置 ByteTracker / ray_history / MediaPipe timestamp ---
    # 🌟 新增：防止前一支影片的 track ID、射線歷史等殘留資料干擾當前影片判定
    interaction.reset_tracking()

    # --- 視線狀態機（每支影片重設） ---
    fsm = GazeFSMManager(history_len=5, max_hold_frames=30)

    # --- 影片串流 ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 無法開啟：{video_path}")
        return

    fps     = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
        print(f"⚠️ [{video_basename}] FPS 讀取失敗，使用預設 30fps")
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 🌟 新增：影片尺寸為 0 代表 VideoCapture 雖成功開啟但無有效幀，提前離開
    if frame_w <= 0 or frame_h <= 0:
        print(f"❌ [{video_basename}] 影片尺寸無效 ({frame_w}×{frame_h})，跳過")
        cap.release()
        return

    out = cv2.VideoWriter(temp_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_w, frame_h))
    # 🌟 新增：VideoWriter 初始化失敗偵測（codec 不支援、路徑錯誤等情況下靜默失敗）
    if not out.isOpened():
        print(f"⚠️ [{video_basename}] VideoWriter 開啟失敗，輸出影片將空白（繼續評分）")

    success, first_frame = cap.read()
    if success:
        # 🌟 批次模式：自動框選右下 1/4 作為牌子搜尋範圍，不需使用者互動
        sign_tracker.initialize_roi_auto(first_frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    win_name = f"Preview: {video_basename} (Stages 1-10)"
    if SHOW_PREVIEW:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, 1280, 720)

    frame_count   = 0
    current_stage = 0
    event_logs    = scoring.event_logs

    # 🌟 跳幀快取（搭配 YOLO_SKIP / GAZE_SKIP 使用）
    last_yolo_boxes  = []    # 上一次 YOLO 偵測到的目標物框
    last_robot_boxes = []    # 上一次 YOLO 偵測到的機器人框
    last_gaze_result = None  # 上一次視線估計結果（含 gaze_vector, face_bbox 等）

    # 🌟 計時診斷（每 TIMING_REPORT_INTERVAL 幀印一次各段耗時，協助找瓶頸）
    import time as _time
    TIMING_REPORT_INTERVAL = 50
    _t_ocr = _t_yolo = _t_interaction = _t_gaze = _t_other = 0.0
    _t_frame_start = None

    # 場地界線：左側 35% 區域視為施測者區（Stage 6-8 的 TH 判定目標）
    divider_x        = int(frame_w * 0.35)
    TESTER_ZONE_BBOX = [0, 0, divider_x, frame_h]

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count      += 1
            current_time_sec  = frame_count / fps
            is_in_trigger_window = speech.is_in_window(current_time_sec, trigger_windows)
            _t_frame_start = _time.perf_counter()
            # 🌟 修正 _t_other 計算：記錄本幀開始前各段累積量，幀末再取差值
            _prev_ocr = _t_ocr; _prev_yolo = _t_yolo
            _prev_interact = _t_interaction; _prev_gaze = _t_gaze

            # ──────────────────────────────────────────────
            # 1. 階段判定
            # ──────────────────────────────────────────────

            # A. OCR（只識別牌子 6 和 7）
            _t0 = _time.perf_counter()
            if current_stage < 8:
                try:
                    detected_stage = sign_tracker.detect_stage(frame)
                    if detected_stage is not None and detected_stage <= 7:
                        if detected_stage != current_stage:
                            scoring.handle_stage_change(current_stage, detected_stage, current_time_sec)
                            current_stage = detected_stage
                except RuntimeError as _ocr_err:
                    # 🌟 EasyOCR CUDA OOM 安全捕捉：不中斷幀迴圈，跳過本幀 OCR
                    #    若為 OOM，主動清 VRAM cache 後繼續（下幀 OCR 可能恢復）
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
            expected_stage = current_stage
            if hasattr(scoring, 'stage_start_times'):
                for s_idx in sorted(scoring.stage_start_times.keys()):
                    if current_time_sec >= scoring.stage_start_times[s_idx]:
                        if s_idx > expected_stage:
                            expected_stage = s_idx

            if expected_stage > current_stage and expected_stage in ACTIVE_STAGES:
                print(f">>> [時間軸] {current_time_sec:.1f}s 推進至第 {expected_stage} 階段")
                scoring.handle_stage_change(current_stage, expected_stage, current_time_sec)
                current_stage = expected_stage

            # C. 聽覺代償（Stage 7 → 8）
            # 🌟 修改：怪聲觸發改為 noise.wav 模板比對命中（is_in_noise_window），
            #          不再依賴 Whisper 關鍵字（[聲音]/嗶/逼），避免 Whisper 音效符號誤觸發
            if current_stage == 7:
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

            # 🌟 修改：傳 sign_tracker.current_stage（tracker 實際讀到的數字）
            # 而非 main.py 的 current_stage（0），避免顯示「Sign:0」誤導
            # 例：牌子顯示「1」→ OCR 正確讀成 1 → Sign:1（而非 Sign:0）
            sign_tracker.draw_boxes(frame, sign_tracker.current_stage)
            _t_ocr += _time.perf_counter() - _t0  # 計時：OCR 段結束

            # ──────────────────────────────────────────────
            # 2. 視覺偵測（只在 ACTIVE_STAGES 內執行）
            # ──────────────────────────────────────────────
            child_is_pointing_hit = False
            yolo_boxes  = []
            robot_boxes = []

            _t0 = _time.perf_counter()
            try:
                if current_stage in ACTIVE_STAGES:
                    # 🌟 YOLO 跳幀：每 YOLO_SKIP 幀才執行一次偵測，其餘幀複用上一結果
                    #    量測目標（閃卡、機器人）幾乎靜止，快取 3 幀完全安全
                    if frame_count % YOLO_SKIP == 0:
                        detect_result = model_manager.detect_objects(frame, stage=current_stage)
                        if current_stage >= 9 and isinstance(detect_result, tuple) and len(detect_result) == 2:
                            yolo_boxes, robot_boxes = detect_result
                        else:
                            yolo_boxes  = detect_result if detect_result else []
                            robot_boxes = []
                        last_yolo_boxes  = yolo_boxes
                        last_robot_boxes = robot_boxes
                    else:
                        yolo_boxes  = last_yolo_boxes
                        robot_boxes = last_robot_boxes

                    # 視覺化：目標物（綠色框）
                    for box in yolo_boxes:
                        bx1, by1, bx2, by2 = map(int, box)
                        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                        
                        # 🌟 依照 main 的設定，Stage 9 與 10 標示為 Tablet
                        if current_stage in [9, 10]:
                            label = f"Tablet (Stage {current_stage})"
                        else:
                            label = f"Target (S{current_stage})"
                            
                        cv2.putText(frame, label,
                                    (bx1, by1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # 視覺化：機器人（橘色框）
                    for box in robot_boxes:
                        bx1, by1, bx2, by2 = map(int, box)
                        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 165, 255), 2)
                        cv2.putText(frame, "Robot",
                                    (bx1, by1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

            except Exception as e:
                if frame_count % 100 == 0:
                    print(f"⚠️ 偵測跳過 (Frame {frame_count}): {e}")
            _t_yolo += _time.perf_counter() - _t0  # 計時：YOLO 段結束

            # 🌟 修改：移除 len(yolo_boxes) > 0 的前置條件
            #    原因：骨架偵測（YOLO-Pose）與手部偵測（MediaPipe）應始終執行，
            #          讓使用者在預覽視窗中隨時看到 Body/Hand 標記與指向射線。
            #          yolo_boxes 為空時 analyze_interaction 仍正常運行，只是無射線碰撞判定。
            _t0 = _time.perf_counter()
            if frame_count % YOLO_SKIP == 0:
                try:
                    # 🌟 傳入影片內部精確時間（ms），搭配 _ts_base 確保跨影片 timestamp 單調遞增
                    _elapsed_ms = int(current_time_sec * 1000)
                    child_is_pointing_hit = interaction.analyze_interaction(frame, yolo_boxes, elapsed_ms=_elapsed_ms)
                except Exception as _ia_err:
                    # MediaPipe 或 YOLO-Pose 臨時故障不中斷整影片，只印警告
                    if frame_count % 100 == 0:
                        print(f"⚠️ analyze_interaction 跳過 (Frame {frame_count}): {_ia_err}")
            _t_interaction += _time.perf_counter() - _t0  # 計時：Interaction 段結束

            # ──────────────────────────────────────────────
            # 3. 視線估計
            # ──────────────────────────────────────────────
            gaze_result               = None
            child_is_gazing_at        = False
            child_is_gazing_at_tester = False
            face_result_for_fsm       = None
            pose_result_for_fsm       = None

            _t0 = _time.perf_counter()
            if is_in_trigger_window or current_stage in ACTIVE_STAGES:
                try:
                    # 🌟 視線跳幀：每 GAZE_SKIP 幀才執行一次 5-stage 視線估計 pipeline
                    #    TB/TH 判定時間解析度 ≥ 0.1s，2 幀快取延遲（≤66ms）完全在容許範圍
                    if frame_count % GAZE_SKIP == 0:
                        gaze_result = gaze_pipeline.estimate(frame)
                        last_gaze_result = gaze_result
                    else:
                        gaze_result = last_gaze_result

                    if gaze_result and gaze_result.get('success'):
                        face_bbox    = gaze_result.get('face_bbox')
                        pitch_rad    = gaze_result['gaze_angles'][0]
                        yaw_rad      = gaze_result['gaze_angles'][1]
                        gaze_vector  = gaze_result['gaze_vector']
                        pitch_deg    = gaze_result['gaze_angles_deg'][0]
                        yaw_deg      = gaze_result['gaze_angles_deg'][1]
                        confidence   = gaze_result.get('confidence', 0.0)
                        left_eye     = gaze_result.get('left_eye')
                        right_eye    = gaze_result.get('right_eye')

                        if face_bbox is not None:
                            frame = draw_gaze_with_face_box(
                                frame, face_bbox, pitch_rad, yaw_rad,
                                gaze_vector=gaze_vector, left_eye=left_eye, right_eye=right_eye,
                                confidence=confidence, show_angles=True,
                                show_direction_label=False, show_gaze_vector=True, bbox_format='xyxy'
                            )

                        # Ray casting：視線是否命中目標物
                        if current_stage in ACTIVE_STAGES and len(yolo_boxes) > 0:
                            child_is_gazing_at = check_gaze_on_objects(gaze_result, yolo_boxes)
                            if child_is_gazing_at:
                                for box in yolo_boxes:
                                    if is_gazing_at_box(gaze_result, box):
                                        cv2.rectangle(frame,
                                                       (int(box[0]), int(box[1])),
                                                       (int(box[2]), int(box[3])),
                                                       (0, 255, 255), 5)
                                        cv2.putText(frame, "GAZING!",
                                                    (int(box[0]), int(box[1]) - 15),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                        # TH 判定：Stage 9-10 看回機器人；Stage 6-8 看回施測者
                        if current_stage >= 9:
                            if len(robot_boxes) > 0:
                                child_is_gazing_at_tester = check_gaze_on_objects(gaze_result, robot_boxes)
                                if child_is_gazing_at_tester:
                                    for box in robot_boxes:
                                        if is_gazing_at_box(gaze_result, box):
                                            cv2.rectangle(frame,
                                                           (int(box[0]), int(box[1])),
                                                           (int(box[2]), int(box[3])),
                                                           (0, 0, 255), 4)
                                            cv2.putText(frame, "GAZING AT ROBOT (TH)!",
                                                        (int(box[0]), int(box[1]) - 35),
                                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        else:
                            # Stage 6, 7, 8：TH 目標為左方施測者區
                            if is_gazing_at_box(gaze_result, TESTER_ZONE_BBOX):
                                if pitch_deg > -5 and yaw_deg > 10:
                                    child_is_gazing_at_tester = True
                                    cv2.rectangle(frame,
                                                   (TESTER_ZONE_BBOX[0], TESTER_ZONE_BBOX[1]),
                                                   (TESTER_ZONE_BBOX[2], TESTER_ZONE_BBOX[3]),
                                                   (0, 0, 0), 3)
                                    cv2.putText(frame, "GAZING AT TESTER!",
                                                (TESTER_ZONE_BBOX[0] + 250, TESTER_ZONE_BBOX[1] + 30),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

                        face_result_for_fsm = {
                            'yolo_head_bbox': gaze_result.get('yolo_head_bbox'),
                            'num_landmarks':  gaze_result.get('num_landmarks', 468)
                        }
                        pose_result_for_fsm = {
                            'success': True,
                            'euler_angles': gaze_result.get('head_pose')
                        }

                    elif gaze_result and 'face_result' in gaze_result:
                        face_result_for_fsm = gaze_result['face_result']
                        pose_result_for_fsm = None

                except Exception as e:
                    if frame_count % 100 == 0:
                        print(f"⚠️ 視線估計跳過 (Frame {frame_count}): {e}")
            _t_gaze += _time.perf_counter() - _t0  # 計時：Gaze 段結束

            # 時序狀態機更新
            try:
                fsm_target = fsm.update(face_result_for_fsm, pose_result_for_fsm)
            except Exception as _fsm_err:
                if frame_count % 100 == 0:
                    print(f"⚠️ fsm.update 跳過 (Frame {frame_count}): {_fsm_err}")
                fsm_target = None

            # 極端轉頭代償（EXTREME_TURNING）
            if fsm.current_state == "EXTREME_TURNING":
                if current_stage in [3, 4] and fsm_target in ["LEFT_BACK_UPPER", "RIGHT_BACK_UPPER"]:
                    child_is_gazing_at = True
                if face_result_for_fsm is not None and face_result_for_fsm.get('yolo_head_bbox') is not None:
                    x1, y1, x2, y2 = map(int, face_result_for_fsm['yolo_head_bbox'])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                    cv2.putText(frame, "YOLO Head Only",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

            # ──────────────────────────────────────────────
            # 4. 評分更新
            # ──────────────────────────────────────────────
            tester_gaze_angles = None
            if gaze_result and gaze_result.get('success'):
                tester_gaze_angles = (
                    gaze_result['gaze_angles_deg'][0],
                    gaze_result['gaze_angles_deg'][1],
                )

            try:
                scoring.update_frame(
                    time_sec=current_time_sec,
                    current_stage=current_stage,
                    is_in_trigger_window=is_in_trigger_window,
                    child_is_pointing_hit=child_is_pointing_hit,
                    child_is_gazing_at=child_is_gazing_at,
                    child_is_gazing_at_tester=child_is_gazing_at_tester,
                    gaze_result=gaze_result,
                    robot_rays=[],
                    robot_boxes=robot_boxes,
                    yolo_boxes=yolo_boxes,
                    is_gazing_at_box_func=is_gazing_at_box,
                    tester_gaze_angles=tester_gaze_angles,
                )
            except Exception as _sc_err:
                if frame_count % 100 == 0:
                    print(f"⚠️ scoring.update_frame 跳過 (Frame {frame_count}): {_sc_err}")

            # ──────────────────────────────────────────────
            # 5. UI 資訊面板
            # ──────────────────────────────────────────────
            c_text   = (0, 255, 255)
            c_key    = (0, 255, 0)    if is_in_trigger_window               else (150, 150, 150)
            c_hit    = (0, 255, 0)    if child_is_pointing_hit               else (0, 0, 255)
            c_gaze   = (0, 255, 0)    if gaze_result and gaze_result.get('success') else (150, 150, 150)
            c_gazing = (0, 255, 255)  if child_is_gazing_at                  else (0, 0, 255)

            cv2.putText(frame,
                        f"Time: {current_time_sec:.1f}s  |  {video_basename}",
                        (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, c_text, 2)
            cv2.putText(frame,
                        f"Stage: {current_stage}  [Measuring: 1-10]",
                        (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, c_text, 2)
            cv2.putText(frame,
                        f"Keyword: {'YES (Active)' if is_in_trigger_window else 'NO (Idle)'}",
                        (15, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, c_key, 2)
            cv2.putText(frame,
                        f"Pointing Hit: {'YES!' if child_is_pointing_hit else 'NO'}",
                        (15, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.8, c_hit, 2)

            if gaze_result and gaze_result.get('success'):
                p = gaze_result['gaze_angles_deg'][0]
                y = gaze_result['gaze_angles_deg'][1]
                cv2.putText(frame, f"Gaze: P={p:.1f} Y={y:.1f}",
                            (15, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, c_gaze, 2)
            else:
                status_str = (f"Gaze: Blind ({fsm.current_state})"
                              if fsm.current_state == "EXTREME_TURNING" else "Gaze: N/A")
                cv2.putText(frame, status_str,
                            (15, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

            cv2.putText(frame,
                        f"Gazing At Object: {'YES!' if child_is_gazing_at else 'NO'}",
                        (15, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.8, c_gazing, 2)

            score_text = (f"Score {scoring.total_score} | "
                          f"S{current_stage} Hit {scoring.stage_gazing_counts.get(current_stage, 0)} | "
                          f"Total {scoring.total_gazing_events}")
            cv2.putText(frame, score_text,
                        (15, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, c_text, 2)

            # 🌟 修正：用本幀差值（而非累積量）計算 Other 耗時，避免 _t_other 變負數
            _elapsed_frame = _time.perf_counter() - _t_frame_start
            _t_other += _elapsed_frame - (_t_ocr - _prev_ocr) - (_t_yolo - _prev_yolo) \
                        - (_t_interaction - _prev_interact) - (_t_gaze - _prev_gaze)

            try:
                out.write(frame)
            except Exception as _wr_err:
                if frame_count % 100 == 0:
                    print(f"⚠️ out.write 失敗 (Frame {frame_count}): {_wr_err}")

            # 🌟 計時報表：每 TIMING_REPORT_INTERVAL 幀印一次各段平均耗時
            if frame_count % TIMING_REPORT_INTERVAL == 0 and frame_count > 0:
                n = TIMING_REPORT_INTERVAL
                total = _t_ocr + _t_yolo + _t_interaction + _t_gaze + _t_other
                print(
                    f"⏱️  [Frame {frame_count}] 各段平均耗時（ms/幀）"
                    f"  OCR={_t_ocr/n*1000:.1f}"
                    f"  YOLO={_t_yolo/n*1000:.1f}"
                    f"  Interact={_t_interaction/n*1000:.1f}"
                    f"  Gaze={_t_gaze/n*1000:.1f}"
                    f"  Other={_t_other/n*1000:.1f}"
                    f"  Total={total/n*1000:.1f}ms"
                )
                _t_ocr = _t_yolo = _t_interaction = _t_gaze = _t_other = 0.0

            # 🌟 修正預覽卡頓：
            #   1. 先 resize 到視窗尺寸再 imshow（減少顯示管線的資料量，從 1080p 降至 720p）
            #   2. 每 2 幀才刷新一次預覽（output 影片仍逐幀寫入，評分不受影響）
            #   3. waitKey 移入 SHOW_PREVIEW 區塊（關閉預覽時不做阻塞呼叫）
            if SHOW_PREVIEW and frame_count % 2 == 0:
                preview_frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_NEAREST)
                cv2.imshow(win_name, preview_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print(">>> [使用者] 按下 Q，中止當前影片處理")
                    break

            if frame_count % 100 == 0:
                gc.collect()

    except Exception as e:
        print(f"\n❌ [{video_basename}] 崩潰: {e}")
        traceback.print_exc()

    finally:
        print(f"\n>>> 匯出事件紀錄 → {out_txt_path}")
        # 🌟 新增：write_report 若拋出例外，會從 finally 傳出並覆蓋原本的 crash 訊息
        try:
            scoring.write_report(out_txt_path)
        except Exception as _wr_err:
            print(f"⚠️ write_report 失敗：{_wr_err}")
            traceback.print_exc()

        # 🌟 新增：把 Whisper 語音辨識逐字稿追加到 txt 尾端
        try:
            import json as _json
            if os.path.exists(speech.cache_path):
                with open(speech.cache_path, 'r', encoding='utf-8') as _f:
                    _cache = _json.load(_f)
                _records = _cache.get('segment_records', [])
                with open(out_txt_path, 'a', encoding='utf-8') as _f:
                    _f.write("\n" + "=" * 40 + "\n")
                    _f.write("=== Whisper 語音辨識逐字稿 ===\n")
                    _f.write(f"  共 {len(_records)} 段，快取檔：{speech.cache_path}\n")
                    _f.write("=" * 40 + "\n")
                    for _rec in _records:
                        _t0  = _rec.get('start', 0.0)
                        _t1  = _rec.get('end',   0.0)
                        _txt = _rec.get('text', '').strip()
                        # 🌟 修改：keywords 在 trigger_events 層，不在 segment_record 層
                        _kws = [k for ev in _rec.get('trigger_events', []) for k in ev.get('keywords', [])]
                        _line = f"[{_t0:.2f}s ~ {_t1:.2f}s]  {_txt}"
                        if _kws:
                            _line += f"  ← 關鍵字：{', '.join(dict.fromkeys(_kws))}"  # fromkeys 去重保序
                        _f.write(_line + "\n")
                print(f">>> 逐字稿已附加至 {out_txt_path}（{len(_records)} 段）")
            else:
                print(f"⚠️  speech_cache.json 不存在，跳過逐字稿輸出")
        except Exception as _e:
            print(f"⚠️  逐字稿追加失敗：{_e}")

        # 🌟 修正：用 try/except 包裹每個釋放操作，避免崩潰後 handle 失效導致 WinError 6
        #    拋出的例外若不攔截會從 finally 傳出 → 被 main() 誤判為頂層錯誤
        try:
            cap.release()
        except Exception:
            pass
        try:
            if 'out' in locals() and out is not None:
                out.release()
        except Exception:
            pass
        if SHOW_PREVIEW:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        # 🌟 釋放本影片幀迴圈中累積的臨時 GPU 張量（YOLO result / Gaze 中間結果）
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        # ── 音軌縫合 ──────────────────────────────────────
        print(">>> 縫合音軌 (FFmpeg)...")
        if os.path.exists(temp_path):
            # 優先使用專案內附的 ffmpeg.exe（Windows 免安裝版）
            ffmpeg_exe = os.path.join(PROJECT_DIR, 'ffmpeg.exe')
            if not os.path.exists(ffmpeg_exe):
                ffmpeg_exe = 'ffmpeg'  # 退回使用系統 PATH 中的 ffmpeg

            ffmpeg_cmd = [
                ffmpeg_exe, '-y',
                '-i', temp_path,
                '-i', video_path,
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-map', '0:v:0',
                '-map', '1:a:0',
                '-shortest',
                out_video_path
            ]
            try:
                subprocess.run(ffmpeg_cmd,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                os.remove(temp_path)
                print(f"✅  有聲影片 → {out_video_path}")
            except FileNotFoundError:
                print(f"❌ 找不到 FFmpeg，無聲影片保留於：{temp_path}")
            except subprocess.CalledProcessError:
                print(f"❌ FFmpeg 縫合失敗，無聲影片保留於：{temp_path}")
        else:
            print("❌ 暫存影片不存在，無法縫合音軌")

    print(f">>> [{video_basename}] 處理完畢\n")


# ============================================================
# ★ 批次主程式
# ============================================================
def main():
    print("=" * 60)
    print("🚀  批次分析系統 — 量測第 1-10 階段")
    print(f"    ACTIVE_STAGES = {sorted(ACTIVE_STAGES)}")
    print(f"    SCORING_VERSION = {SCORING_VERSION}")
    print("=" * 60)

    # 🌟 CUDA / GPU 狀態總覽（啟動時顯示，方便確認是否啟用顯卡加速）
    if torch.cuda.is_available():
        _gpu_name  = torch.cuda.get_device_name(0)
        _vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f">>> GPU：{_gpu_name}  |  VRAM：{_vram_total:.1f} GB")
        print(">>> CUDA 加速：✅ 已啟用（YOLO / Gaze / EasyOCR 均使用 GPU）")
    else:
        print(">>> ⚠️  CUDA 不可用，所有推論將使用 CPU（速度顯著較慢）")
        print("         請確認 NVIDIA 驅動與 torch+cuda 版本是否匹配")

    # 🌟 修改：切換工作目錄至原始專案根目錄
    os.chdir(PROJECT_DIR)
    print(f">>> 工作目錄已設定為：{PROJECT_DIR}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 載入 config ---
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            CONFIG = yaml.safe_load(f)
        print(f">>> 成功載入配置：{CONFIG_PATH}")
    except FileNotFoundError:
        CONFIG = {}
        print(f"⚠️ 找不到 config.yaml，使用預設值")

    gaze_config = CONFIG.get('gaze_estimation', {})

    # --- 收集影片清單（提前到 GPU 模型載入之前）---
    if not os.path.isdir(VIDEO_DIR):
        sys.exit(f"❌ 找不到影片資料夾：{VIDEO_DIR}\n   請將影片放入 {VIDEO_DIR}")

    EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV'}
    video_files = sorted([
        os.path.join(VIDEO_DIR, f)
        for f in os.listdir(VIDEO_DIR)
        if os.path.splitext(f)[1] in EXTS
    ])

    if not video_files:
        sys.exit(f"❌ {VIDEO_DIR} 中沒有找到任何影片（支援 {EXTS}）")

    print(f">>> 找到 {len(video_files)} 支影片：")
    for f in video_files:
        print(f"    - {os.path.basename(f)}")
    print()

    # ══════════════════════════════════════════════════════
    # 🌟 Step 0：批次語音預處理（GPU 模型尚未載入，Whisper 可全速使用 GPU）
    #
    # 根本原因修正：原本每支影片進入 frame loop 前才跑 Whisper（CPU），
    # 5 支影片 × large-v3 CPU 辨識 = 可能長達數小時等待。
    # 改為：全部影片的語音辨識集中在此一次完成，
    #   ① 此時 YOLO / Gaze / EasyOCR 尚未佔用 VRAM → Whisper 可用 GPU → 快 5-10x
    #   ② 快取建立後，process_single_video 的 get_trigger_windows() 直接讀取 → 不再啟動子行程
    #   ③ 重複執行時快取已存在，此段幾乎是零耗時
    # ══════════════════════════════════════════════════════
    print("=" * 60)
    print("📢  Step 0：批次語音辨識（GPU Whisper，GPU 模型載入前執行）")
    print("=" * 60)
    for _vp in video_files:
        _bn   = os.path.splitext(os.path.basename(_vp))[0]
        _sdir = os.path.join(OUTPUT_DIR, f'_speech_{_bn}')
        _cpath = os.path.join(_sdir, 'speech_cache.json')
        if os.path.exists(_cpath):
            print(f"    [{_bn}] ✅ 快取已存在，跳過 Whisper")
        else:
            print(f"    [{_bn}] 🔊 執行 Whisper 語音辨識...")
            os.makedirs(_sdir, exist_ok=True)
            try:
                _sp = SpeechTrigger(
                    video_path=_vp,
                    output_dir=_sdir,
                    keywords=SPEECH_KEYWORDS,
                    noise_sample_path=NOISE_SAMPLE_PATH if os.path.exists(NOISE_SAMPLE_PATH) else None,
                )
                _sp.get_trigger_windows()
                del _sp
            except Exception as _sp_err:
                # 🌟 語音辨識失敗時不中斷批次：frame loop 仍可執行，只是沒有觸發時間窗
                print(f"    [{_bn}] ⚠️ 語音辨識失敗（{_sp_err}），frame loop 繼續但無語音觸發")
            gc.collect()
    print(">>> Step 0 完成：全部影片語音快取就緒\n")

    # --- 一次性初始化重型 GPU 元件（語音預處理完成後才載入，避免 VRAM 衝突）---
    print("=" * 60)
    print("📢  Step 1：載入 GPU 視覺模型")
    print("=" * 60)
    model_manager = ModelManager(model_dir=MODEL_DIR)
    pose_path = os.path.join(MODEL_DIR, 'yolo11n-pose.pt')
    hand_path = os.path.join(MODEL_DIR, 'gaze', 'hand_landmarker.task')  # 🌟 修改：實際路徑在 model/gaze/ 下
    # 🌟 修改：明確傳入 hand_model_path，避免 Windows junction/symlink 下
    #          __file__ 解析到真實路徑（project_v3/）而非 C:\project\ 導致 FileNotFoundError
    interaction = InteractionEngine(pose_model_path=pose_path, hand_model_path=hand_path, sma_window=5)
    gaze_pipeline = GazeEstimationPipeline(config=gaze_config)

    # 🌟 EasyOCR 只載入一次
    print(">>> 載入 EasyOCR（僅一次）...")
    sign_tracker = SignboardTracker(allowlist='1234567')
    # 🌟 新增：暖機失敗不應中斷批次（第一幀 OCR 可能慢但不崩潰）
    try:
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        sign_tracker.reader.readtext(dummy_img)
        print(">>> EasyOCR 暖機完成")
    except Exception as _ocr_warm_err:
        print(f"⚠️ EasyOCR 暖機失敗（{_ocr_warm_err}），繼續執行（第一幀 OCR 可能較慢）")
    # 🌟 修改：覆寫 OCR 跳幀間隔（預設 2 幀 → 15 幀）
    #          牌子靜止展示數秒，每 0.5s 偵測一次已足夠；
    #          降低 EasyOCR 呼叫頻率是最有效的效能改善手段。
    sign_tracker.OCR_FRAME_INTERVAL        = OCR_SKIP
    sign_tracker.OCR_FRAME_INTERVAL_STAGE7 = OCR_SKIP
    print(f">>> OCR 跳幀間隔：{OCR_SKIP} 幀（預設 2）")
    print(">>> 共用模型初始化完成\n")

    # --- 批次處理 ---
    for i, video_path in enumerate(video_files, 1):
        print(f"\n[{i}/{len(video_files)}]  {os.path.basename(video_path)}")
        try:
            process_single_video(
                video_path, OUTPUT_DIR,
                model_manager, interaction,
                gaze_pipeline, gaze_config,
                sign_tracker   # 🌟 共用 tracker，內部 reset() 重置狀態不重載 EasyOCR
            )
        except Exception as _e:
            print(f"\n❌ [{os.path.basename(video_path)}] 頂層捕捉到例外，跳過並繼續批次：{_e}")
            traceback.print_exc()

        # 🌟 每支影片結束後強制清理：
        #    - gc.collect()：讓 Python 提前回收 YOLO result 物件等臨時張量
        #    - torch.cuda.empty_cache()：釋放 PyTorch 保留但暫不使用的 VRAM 分配器快取
        #    批次處理後期 YOLO/Gaze/EasyOCR 模型都在 VRAM，若不清理快取，
        #    累積的臨時張量會導致 CUDA OOM → OCR 先崩 → 整程式掛掉
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            used  = torch.cuda.memory_allocated() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f">>> VRAM 清理完畢：{used:.2f} GB 使用中 / {total:.2f} GB 總量")

    print("\n" + "=" * 60)
    print("✅  全部影片處理完成！")
    print(f"📁  輸出目錄：{OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()