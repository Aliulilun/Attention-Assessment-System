import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os
import sys
import cv2
import gc
import math  # 🌟 新增（A2）：射線長度上限需要計算畫面對角線與向量正規化
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
# 🌟 修改：PROJECT_DIR 改為動態推算（hurry/ 的上一層），不再寫死 C:\project
#          （hurry/main.py 實際位於 integrate_v5/hurry/，modules・model・config.yaml
#          都在 integrate_v5/ 下，寫死路徑在其他機器 / 資料夾改名後會直接找不到檔案）
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))  # .../integrate_v5/hurry
PROJECT_DIR = os.path.dirname(BASE_DIR)                   # .../integrate_v5
sys.path.insert(0, PROJECT_DIR)

from modules.speech import SpeechTrigger
from modules.signboard import SignboardTracker
from modules.models_manager import ModelManager
from modules.interaction import InteractionEngine
from modules.scoring_engine import ScoringEngine
from modules.gaze_estimation import GazeEstimationPipeline
from modules.gaze_estimation.visualization import draw_gaze_with_face_box
from modules.gaze_estimation.state_manager import GazeFSMManager
from modules.gaze_hold import GazeHoldTracker

# ============================================================
# ★ 全域設定
# ============================================================
MODEL_DIR   = os.path.join(PROJECT_DIR, 'model')          # 指向原始專案模型目錄
# VIDEO_DIR = os.path.join(BASE_DIR, '..', '..', 'github', 'model_test', 'input')
VIDEO_DIR   = os.path.join(BASE_DIR, 'video')
OUTPUT_DIR  = os.path.join(BASE_DIR, 'output')
CONFIG_PATH = os.path.join(PROJECT_DIR, 'config.yaml')    # 使用原始專案的 config

ACTIVE_STAGES   = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}   # ★ 量測第 1-10 階段
SHOW_PREVIEW    = False                # ★ False = 關閉預覽視窗（批次加速，省 I/O）
SCORING_VERSION = 'HURRY_1to10_BATCH_V1'

# ★ 跳幀設定（效能優化）
# 說明：EasyOCR / 視線估計 / YOLO 不需要每幀都跑，
#       被量測目標（閃卡、機器人）移動緩慢，快取上一幀結果可大幅提速。
YOLO_SKIP = 1
GAZE_SKIP = 1
OCR_SKIP  = 5   # 🌟 核心修正 1：將 15 改為 5。降低 Tracker 防抖累積的延遲，防止吃掉語音空窗期

# ============================================================
# 🌟 新增（A1）：TH 判定用的施測者頭部框外擴比例
# 語意同 expand_box() 的 ratio——各邊各外擴「框尺寸 × 此比例」。
#   0.0 = 只用頭部框本身（最嚴格，最貼合「看臉」的臨床定義）
#   0.3 = 各邊外擴 30%（目前值）
#   0.8 = ⚠️ 第一版用這個，太大了
#
# ⚠️ 2026-09-14 實測教訓：把 0.8 說成「保守」是說反了。
# 對【判定框】而言，框越大命中越多、假陽性越多——保守的方向是「小」。
# 施測者靠近鏡頭時肩寬在畫面上很大，頭部半徑 = 肩寬 × 0.40 本來就不小，
# 再乘 0.8 外擴後整個框可以吃掉畫面左半邊（90 號 13.9s 實例），
# 這是 Stage 1、2 假陽性暴增的主因。
# ============================================================
# 🌟 A1 再調校（2026-09-16）：0.3 太嚴格，經隔離診斷確認是 104-S7、
# 82-S1、86-S2 三處 TH 消失的根本原因——加了除錯記錄後發現這三處
# 頭部框每一幀都有算出來（從未回傳 None），但框的範圍沒有涵蓋到
# 視線射線實際指向的位置，box_hit 持續為 False（104 甚至連續
# 111.99s~117.6s 整整 5 秒都沒中）。屬於「框太小」而非「偵測失敗」。
# 改成 0.6（面積約 4.84 倍，仍遠低於 v1 的 0.8=6.76 倍），先在
# 104/82/86（待修）+ 42/52（Stage 3 已知假陽性，須確認沒有惡化）
# 五支影片上驗證後才視為定案。
#
# 🌟 A1 三度調校（2026-09-16，診斷 8+9）：0.6 修復 82-S1/82-S7 後，
# 仍有 49-S8、82-S8、86-S2、90-S4、104-S7 五處 TH 遺漏。診斷 8 逐幀
# 記錄頭部框的 binding 來源，發現 scale_based（受
# HEAD_SIZE_FROM_SHOULDER 控制）在絕大多數幀是決定框大小的因素；
# 診斷 9 進一步對「角度門檻已通過（方向確認正確）但 box_hit=False」
# 的幀做事後重算，測試把框放大到不同倍率時最小能命中的倍率
# （minfix_ratio）。結果分成兩類、必須分開看待：
#   - 49-S8、82-S8：角度門檻通過的幀數夠多（各 304、339 幀），
#     其中未命中但可被「放大到 1.0」救回的比例最高（69/105、
#     83/152，均 >50%），是唯一有實證支持「框太小」這個機制的案例。
#   - 86-S2、90-S4、104-S7：問題根本不在框大小——這三處絕大多數
#     幀（86: 119/120、104: 537/546）連角度門檻本身都過不了（視線
#     估計的 pitch/yaw 方向本身就沒有指向施測者），框再怎麼放大
#     都無關；少數幾幀通過門檻但需要 3~5 倍以上才能救回，屬於視線
#     估計模型準確度的問題，不是本次參數調校能解決的範圍。
# 因此把 0.6 調到 1.0（面積約 9 倍，已超過 v1 那次造成假陽性暴增的
# 0.8=6.76 倍——但 v1 當時「角度門檻」根本還沒加進去，這次調整是在
# 角度門檻已存在、且已被證實能擋住方向錯誤的假命中之後才做的，風險
# 性質不同）。
#
# 🌟 診斷 10 實測結果（2026-09-16，已定案）：8 支影片（42/52/64 已知
# 假陽性風險案例 + 49/82/86/90/104 待修案例）完整驗證通過——
# **49-S8、82-S8、86-S2 三處修復成功**（86-S2 是原本預期外的額外
# 修復），**無新增假陽性**（64-S1 正確維持被抑制）。90-S4、104-S7
# 如預期仍未修復，證實是視線角度估計模型本身的準確度問題，超出框
# 大小這個機制的能力範圍，不屬於本次參數調校可處理的範圍。
# 完整驗證數據見 .superpowers/sdd/2026-09-15-scoring-engine-fixes/
# diagnostics/diag10_expand10_validation_result.md。
TESTER_HEAD_EXPAND = 1.0

# ★ 怪聲參考音檔（放在 model/ 目錄，跨所有影片共用）
# 🌟 修改：speech.py 新增 noise_sample_path 參數，此處明確傳入固定路徑；
#          若檔案不存在則傳 None → speech_engine 退回純頻譜特徵偵測
NOISE_SAMPLE_PATH = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model', 'noisesample', 'noise.wav')
)
# 等同於 C:\project\model\noisesample\noise.wav

# 🌟 修改：關鍵字清單精簡——只保留 1-10 階段偵測/計分實際會用到的詞。
# 對照程式消費點（scoring_engine._update_trigger_records / _build_absolute_timeline /
# _update_gazing_score）：
#   你看、看這裡  → Stage 1-4 的 T0 觸發 + 計分時間窗 + Stage 9 結束偵測
#   畫、画        → Stage 9 起點（子字串同時涵蓋「畫一幅」「畫好了」等）
#   煙火、烟火、321、三二一、三 → Stage 10 起點（煙火倒數）
# 已移除：開始/準備/準備囉/小朋友/機器人/放煙火 等——1-10 版沒有任何
# 邏輯讀取這些詞，只會多產生無用的觸發時間窗干擾階段鎖。
# 怪聲（Stage 8）由 noise.wav 模板比對負責，不走關鍵字。
SPEECH_KEYWORDS = [
    "你看", "看這裡",
    "畫", "画",
    "煙火", "烟火", "321", "三二一", "三",
]


# ============================================================
# ★ 視線-物體交集判定函數 (Ray Casting)
# ============================================================
def ray_intersects_box(origin, dir_vec, box, max_t=None):
    """Ray-AABB 碰撞檢測。

    🌟 新增（A2）：max_t 限制射線有效長度（像素；呼叫端須傳入單位方向向量）。
    原本射線是無限長的——視線只要方向大致對，就算目標在畫面另一端、
    距離 1500px 以外也算命中。這在判定框是「畫面左 35% 滿版長條」時
    無所謂（框就在旁邊），但 A1 把 TH 的判定框換成小小的頭部框之後，
    無限長射線會讓遠處的小框更容易被亂掃到，兩者必須一起修。
    （modules/interaction.py 的指向射線早在 P0-3 就補了同樣的上限。）
    """
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
    if max_t is not None:
        tmax = min(tmax, max_t)
    return tmax >= max(0, tmin)

def is_gazing_at_box(gaze_result, object_bbox, max_t=None):
    if not gaze_result or not gaze_result.get('success'):
        return False
    left_eye  = gaze_result.get('left_eye')
    right_eye = gaze_result.get('right_eye')
    if left_eye is None or right_eye is None:
        return False
    eye_center   = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
    gaze_vector  = gaze_result['gaze_vector']
    direction    = (gaze_vector[0], gaze_vector[1])
    # 🌟 A2：max_t 以像素為單位，方向向量必須先正規化才有意義
    if max_t is not None:
        _n = math.hypot(direction[0], direction[1])
        if _n < 1e-9:
            return False
        direction = (direction[0] / _n, direction[1] / _n)
    return ray_intersects_box(eye_center, direction, object_bbox, max_t=max_t)

def check_gaze_on_objects(gaze_result, yolo_boxes):
    for box in yolo_boxes:
        if is_gazing_at_box(gaze_result, box):
            return True
    return False

def expand_box(box, ratio=0.10):
    """🌟 新增：命中測試用的框外擴（與根目錄 main.py 同步的防閃爍修正）。
    YOLO 框每幀會抖動、視線射線擦框緣時會一幀中一幀不中，
    外擴 10% 給判定留容差（只影響命中測試，不影響畫面上的框）。"""
    x1, y1, x2, y2 = box
    mx = (x2 - x1) * ratio
    my = (y2 - y1) * ratio
    return (x1 - mx, y1 - my, x2 + mx, y2 + my)


# ============================================================
# ★ 單一影片處理函數
# ============================================================
def process_single_video(video_path, output_dir, model_manager, interaction,
                          gaze_pipeline, gaze_config, sign_tracker,
                          frame_callback=None, should_stop=None):
    """
    處理單一影片，輸出:
      output/{basename}.mp4  — 有聲分析影片
      output/{basename}.txt  — 事件紀錄

    🌟 新增（UI 整合用，CLI 行為完全不變）：
      frame_callback(payload:dict)  每幀呼叫一次，把已標註的畫面與即時統計
                                    交給外部（例如 PySide6 介面）顯示。
                                    傳 None（預設）＝完全不影響原本流程。
      should_stop() -> bool         每幀開頭檢查一次，回傳 True 即中止本影片
                                    （介面上的「停止」按鈕）。
    """
    # 🌟 修正：strip() 去掉尾部空格，防止「74 .mp4」→ basename「74 」帶空格進入路徑中間元件
    # Windows 只對最後路徑元件去尾部空格，中間元件不正規化，導致 _speech_74 /speech_cache.json
    # 找不到實際的 _speech_74\speech_cache.json → noise_events=[] → Stage 8/9/10 全失效
    video_basename = os.path.splitext(os.path.basename(video_path))[0].strip()
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

    # 🌟 新增（B2）：告知 InteractionEngine 目前的跳幀設定。
    # 它的 dwell / hold 以「被呼叫次數」累加，而主迴圈只在
    # frame_count % YOLO_SKIP == 0 時呼叫，不換算的話門檻會隨跳幀漂移。
    if hasattr(interaction, "set_frame_stride"):
        interaction.set_frame_stride(YOLO_SKIP)

    # --- 視線狀態機（每支影片重設） ---
    # ⚠️ A7 實驗紀錄：曾改為 max_hold_frames=10，實測對 TB/TH 無任何可觀察
    #    影響（TB 時間戳一秒不差），已還原為 30。詳見 scoring_engine.py 內
    #    TB 計次處的實驗紀錄。
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
    # 🌟 新增：總幀數（UI 進度條用；讀取失敗時給 0，介面會退回顯示已處理幀數）
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

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

    # 🌟 修正：Stage 1-7 的升階事件佇列（取代單一 pending_stage 變數）
    # 原因：pending_stage 每幀都會被重置成 current_stage，若該幀升階時剛好
    # 處於 Trigger Lock 期間，這個值下一幀就消失。當「你看」判定視窗幾乎
    # 首尾相接時（例如連續多張牌子各自觸發一次「你看」），中間發生的每一次
    # 真實升階（如 2、3）都會被鎖掉、遺失，鎖解除時只會同步到「當下」sign_tracker
    # 的最新階段，直接跳號、漏記中繼階段。改用佇列依序保留每次升階的
    # (階段, 原始偵測時間)，鎖解除後每幀只套用最前面一筆，確保依序不跳號。
    stage_transition_queue = []

    # 🌟 跳幀快取（搭配 YOLO_SKIP / GAZE_SKIP 使用）
    last_yolo_boxes  = []    # 上一次 YOLO 偵測到的目標物框
    last_robot_boxes = []    # 上一次 YOLO 偵測到的機器人框
    last_child_is_pointing_hit = False  # 上一次指向判定結果（YOLO_SKIP 跳幀時沿用）

    # 🌟 視線容錯快取（推論與繪圖分離用）
    last_valid_gaze = None        # 只存 success=True 的最近一筆視線結果
    gaze_fallback_counter = 0     # 連續失敗幀數計數器
    MAX_GAZE_FALLBACK = 3         # 🌟 A6 修正（2026-09-15）：原值 5 高於視線報告建議的 2~3，
                                  # 縮短容忍幀數以減少閃爍容忍期間吃到錯誤視線的機會

    # ==================================================
    # 🌟 新增：視線「命中判定」防閃爍三件套（與根目錄 main.py 同步）
    # 根因：判定是逐幀原始射線相交，視線向量每幀抖動幾度 +
    #       YOLO 框抖動/掉偵測，射線擦框緣時就一幀中一幀不中。
    # 1. gaze_ray_history：對眼睛中心與視線向量做 5 幀滑動平均
    # 2. GAZE_BOX_MARGIN：命中測試時物品框外擴 10% 容差
    # 3. GAZE_HIT_HOLD_FRAMES：命中遲滯——立即亮起、
    #    連續 10 幀（約 0.33s）沒命中才熄滅，橋接短暫漏判
    # （宣告在 process_single_video 內 → 每支影片自動重置，批次安全）
    # ==================================================
    from collections import deque as _deque
    gaze_ray_history = _deque(maxlen=5)   # (left_eye, right_eye, gaze_vector)
    GAZE_BOX_MARGIN = 0.10
    GAZE_HIT_HOLD_FRAMES = 10
    # 🌟 A4 修正（2026-09-15）：改用 GazeHoldTracker，讓「看物品」與「看施測者」
    # 兩側的遲滯互相打斷，避免同一時間戳被同時判定成立（見 modules/gaze_hold.py）
    gaze_hold_tracker = GazeHoldTracker(hold_frames=GAZE_HIT_HOLD_FRAMES)

    # 🌟 計時診斷（每 TIMING_REPORT_INTERVAL 幀印一次各段耗時，協助找瓶頸）
    import time as _time
    TIMING_REPORT_INTERVAL = 50
    _t_ocr = _t_yolo = _t_interaction = _t_gaze = _t_other = 0.0
    _t_frame_start = None

    # 場地界線：左側 35% 區域視為施測者區
    # ⚠️ A1 之後，TESTER_ZONE_BBOX 只在「拿不到施測者頭部框」時當退路使用
    divider_x        = int(frame_w * 0.35)
    TESTER_ZONE_BBOX = [0, 0, divider_x, frame_h]

    # 🌟 新增（A2）：視線射線的最大有效長度 = 0.6 × 畫面對角線
    # 與 modules/interaction.py 的 P0-3（指向射線）採同一個係數
    MAX_GAZE_RAY_LEN = 0.6 * math.hypot(frame_w, frame_h)

    try:
        while cap.isOpened():
            # 🌟 新增：外部中止鉤子（UI 的停止按鈕）。放在讀幀前，
            # 中止後仍會走 finally → 已處理的部分照樣寫出報告與影片。
            if should_stop is not None and should_stop():
                print(">>> [外部] 收到中止指令，停止當前影片處理")
                break

            ret, frame = cap.read()
            if not ret:
                break

            frame_count      += 1
            current_time_sec  = frame_count / fps
            is_in_trigger_window = speech.is_in_window(current_time_sec, trigger_windows)

            # 🌟 推論與繪圖分離：推論全用 frame（乾淨原始幀），繪圖全用 display_frame
            display_frame = frame.copy()

            _t_frame_start = _time.perf_counter()
            # 🌟 修正 _t_other 計算：記錄本幀開始前各段累積量，幀末再取差值
            _prev_ocr = _t_ocr; _prev_yolo = _t_yolo
            _prev_interact = _t_interaction; _prev_gaze = _t_gaze

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
                            # 🌟 修正：改存進佇列（含真正偵測到的時間），不再直接覆蓋
                            # pending_stage——避免 Trigger Lock 期間發生的中繼階段被蓋掉
                            stage_transition_queue.append((detected_stage, current_time_sec))
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

            # A2. 依序套用佇列中的 Stage 1-7 升階事件（每幀最多套用一筆）
            # 鎖定規則與原本相同：語音「你看」判定視窗內延後套用；
            # 差別在於佇列會保留每一次真正發生的升階，鎖解除後依序補上，
            # 不會因為鎖定期間又發生了更後面的升階而把中繼階段整個蓋掉。
            if stage_transition_queue:
                next_stage, next_detected_time = stage_transition_queue[0]
                if is_in_trigger_window:
                    if frame_count % 15 == 0:
                        print(f">>> [鎖定] {current_time_sec:.1f}s 正在語音空窗期內，延後切換至階段 "
                              f"{next_stage}（佇列剩餘 {len(stage_transition_queue)} 筆）")
                else:
                    stage_transition_queue.pop(0)
                    print(f">>> [時間軸] {next_detected_time:.1f}s 推進至第 {next_stage} 階段")
                    scoring.handle_stage_change(current_stage, next_stage, next_detected_time)
                    gaze_hold_tracker.reset()  # 🌟 A5 修正：避免上一關最後的遲滯狀態帶進新階段
                    current_stage = next_stage
                    pending_stage = current_stage  # 🌟 同步：A2 可能已推進 current_stage，避免 B 用到舊值

            # B. 時間軸自動推進（只推進到 ACTIVE_STAGES 內的階段）
            if hasattr(scoring, 'stage_start_times'):
                for s_idx in sorted(scoring.stage_start_times.keys()):
                    if current_time_sec >= scoring.stage_start_times[s_idx]:
                        if s_idx > pending_stage:
                            pending_stage = s_idx  # 記錄時間軸推進的新階段

            # 🌟 核心修正 3： Trigger Lock 空窗期鎖
            # 只要在語音有效期限內，絕不強制切換階段，確保小孩作答能被記錄！
            # 🌟 修正：語音/怪聲驅動的階段（8、9、10）不受鎖延後。
            # 這些階段的 T0 就是關鍵字/怪聲本身，而關鍵字一出現必然開啟
            # 語音空窗期 → 鎖住切換 → 9/10 階段關鍵字密集、空窗期一個接
            # 一個 → 切換被無限延後，看起來就是「有關鍵字卻不切換也不記錄」。
            # 鎖只保留給 OCR 牌子驅動的 1-7 階段（保護小孩作答期間不被硬切）。
            if pending_stage > current_stage and pending_stage in ACTIVE_STAGES:
                voice_driven = pending_stage >= 8
                if is_in_trigger_window and not voice_driven:
                    if frame_count % 15 == 0:
                        print(f">>> [鎖定] {current_time_sec:.1f}s 正在語音空窗期內，延後切換至階段 {pending_stage}")
                else:
                    print(f">>> [時間軸] {current_time_sec:.1f}s 推進至第 {pending_stage} 階段")
                    scoring.handle_stage_change(current_stage, pending_stage, current_time_sec)
                    gaze_hold_tracker.reset()  # 🌟 A5 修正：避免上一關最後的遲滯狀態帶進新階段
                    current_stage = pending_stage
                    # 🌟 新增：語音驅動階段（8/9/10）同步 tracker 顯示，
                    # 讓畫面上的「Sign:N」與左上角 Stage 一致（純顯示同步，
                    # tracker 在 stage>=8 本來就不再跑 OCR，不影響判定）
                    if current_stage >= 8:
                        sign_tracker.force_stage(current_stage)

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
                        gaze_hold_tracker.reset()  # 🌟 A5 修正：避免上一關最後的遲滯狀態帶進新階段
                    else:
                        scoring.handle_stage_change(current_stage, 8, current_time_sec)
                        gaze_hold_tracker.reset()  # 🌟 A5 修正：避免上一關最後的遲滯狀態帶進新階段
                    current_stage = 8

            # 🌟 修改：傳 sign_tracker.current_stage（tracker 實際讀到的數字）
            # 而非 main.py 的 current_stage（0），避免顯示「Sign:0」誤導
            # 例：牌子顯示「1」→ OCR 正確讀成 1 → Sign:1（而非 Sign:0）
            sign_tracker.draw_boxes(display_frame, sign_tracker.current_stage)
            _t_ocr += _time.perf_counter() - _t0  # 計時：OCR 段結束

            # ──────────────────────────────────────────────
            # 🌟 新增：1-10 量測結束邊界
            # 影片進入第 11 階段（機指近物，「小朋友你看」）後，
            # 1-10 的量測即告結束。current_stage 會停在 10
            # （ACTIVE_STAGES 擋住 11+ 切換），但 Gazing 統計只看
            # 當前階段——不設邊界的話，11-14 階段期間小朋友的注視
            # 會全部灌進 Stage 10 的 Score / Hit 統計，數據失真。
            # ──────────────────────────────────────────────
            # 🌟 修正（C7）：原本只用 stage_start_times.get(11) 當邊界，但這在
            # 批次版永遠是 None——本檔第 78-85 行的 SPEECH_KEYWORDS 刻意不含
            # 「小朋友」，而 scoring_engine.py 建立 stage_start_times[11..14] 的
            # 條件正是「keywords 含小朋友」。結果 _t11 永遠是 None、
            # measurement_over 永遠是 False，上面註解宣稱的保護從未生效，
            # 11-14 階段的注視仍會一路灌進 Stage 10 的 Gazing 統計。
            # 改為優先採用 Stage 10 計分紀錄的 end_time（T0+10s），不依賴關鍵字；
            # 若 stage_start_times[11] 真的存在（完整版的關鍵字設定）仍一併採用。
            measurement_over = False
            for _r in scoring.trigger_event_records:
                if _r.get("stage") == 10:
                    _e10 = _r.get("end_time")
                    if _e10 is not None and _e10 < 10**8 and current_time_sec > _e10:
                        measurement_over = True
                    break
            if not measurement_over and hasattr(scoring, 'stage_start_times'):
                _t11 = scoring.stage_start_times.get(11)
                if _t11 is not None and current_time_sec >= _t11:
                    measurement_over = True

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
                        cv2.rectangle(display_frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)

                        # 🌟 依照 main 的設定，Stage 9 與 10 標示為 Tablet
                        if current_stage in [9, 10]:
                            label = f"Tablet (Stage {current_stage})"
                        else:
                            label = f"Target (S{current_stage})"

                        cv2.putText(display_frame, label,
                                    (bx1, by1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # 視覺化：機器人（橘色框）
                    for box in robot_boxes:
                        bx1, by1, bx2, by2 = map(int, box)
                        cv2.rectangle(display_frame, (bx1, by1), (bx2, by2), (0, 165, 255), 2)
                        cv2.putText(display_frame, "Robot",
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
                    # 🌟 傳入 display_frame（與 frame 同內容的乾淨複本），
                    #    骨架/射線畫在 display_frame，不污染 frame 供 Gaze 推論使用。
                    _elapsed_ms = int(current_time_sec * 1000)
                    last_child_is_pointing_hit = interaction.analyze_interaction(display_frame, yolo_boxes, elapsed_ms=_elapsed_ms)
                except Exception as _ia_err:
                    # MediaPipe 或 YOLO-Pose 臨時故障不中斷整影片，只印警告
                    if frame_count % 100 == 0:
                        print(f"⚠️ analyze_interaction 跳過 (Frame {frame_count}): {_ia_err}")
            child_is_pointing_hit = last_child_is_pointing_hit  # 🌟 沿用快取結果（YOLO_SKIP 跳幀時保持穩定）

            # ============================================================
            # 🌟 新增：Stage 8 指向特規
            # 第 8 題（怪聲）沒有目標物，指向計分「只要小朋友的指向射線
            # 存在」就算一次，不需要射線指中任何物品。
            # 其他階段維持原邏輯（射線必須命中目標物才算 Pointing Hit）。
            # ============================================================
            # 🌟 修改（P0-2）：改用嚴格版旗標 last_child_pointing_stable
            # 需連續 >= S8_MIN_DWELL_FRAMES 幀（≈0.33s）且方向偏差 < 25°，
            # 排除幼兒驚訝抬手 / 遮耳 / 隨機掃動等非指向動作。
            if current_stage == 8 and interaction.last_child_pointing_stable:
                child_is_pointing_hit = True
            _t_interaction += _time.perf_counter() - _t0  # 計時：Interaction 段結束

            # ──────────────────────────────────────────────
            # 3. 視線估計（推論用乾淨 frame，結果快取防閃爍）
            # ──────────────────────────────────────────────
            child_is_gazing_at        = False
            child_is_gazing_at_tester = False
            face_result_for_fsm       = None
            pose_result_for_fsm       = None

            _t0 = _time.perf_counter()
            if is_in_trigger_window or current_stage in ACTIVE_STAGES:
                try:
                    # 🌟 視線跳幀：每 GAZE_SKIP 幀才執行一次推論；其餘幀沿用 last_valid_gaze
                    if frame_count % GAZE_SKIP == 0:
                        _cur_gaze = gaze_pipeline.estimate(frame)  # 推論用乾淨 frame
                        if _cur_gaze and _cur_gaze.get('success'):
                            # 🌟 成功：更新快取，歸零計數器
                            last_valid_gaze = _cur_gaze
                            gaze_fallback_counter = 0
                        elif _cur_gaze:
                            # 🌟 失敗（YOLO 有頭但 MediaPipe 失效）：計數超限才清空快取
                            gaze_fallback_counter += 1
                            if gaze_fallback_counter > MAX_GAZE_FALLBACK:
                                last_valid_gaze = None
                                gaze_ray_history.clear()  # 🌟 視線真正丟失：清空平滑歷史
                            if 'face_result' in _cur_gaze:
                                face_result_for_fsm = _cur_gaze['face_result']
                        else:
                            gaze_fallback_counter += 1
                            if gaze_fallback_counter > MAX_GAZE_FALLBACK:
                                last_valid_gaze = None
                                gaze_ray_history.clear()  # 🌟 同上
                except Exception as e:
                    if frame_count % 100 == 0:
                        print(f"⚠️ 視線估計跳過 (Frame {frame_count}): {e}")
            _t_gaze += _time.perf_counter() - _t0  # 計時：Gaze 段結束

            # 🌟 使用快取結果（防閃爍）：gaze 短暫失敗時保持上一幀的箭頭與判定
            active_gaze = last_valid_gaze

            raw_gaze_obj = False       # 🌟 本幀原始命中（平滑射線 + 容差框）
            raw_gaze_tester = False
            gazed_obj_boxes = []       # 🌟 本幀被注視的物品框（供繪圖用）
            gazed_robot_boxes = []
            gazed_tester_box = None    # 🌟 新增（A1）：本幀命中的施測者頭部框（供繪圖用）
            smoothed_gaze = active_gaze

            if active_gaze and active_gaze.get('success'):
                pitch_deg = active_gaze['gaze_angles_deg'][0]
                yaw_deg   = active_gaze['gaze_angles_deg'][1]

                # ─── 🌟 視線射線 5 幀滑動平均（消除逐幀角度抖動）───
                _le = active_gaze.get('left_eye')
                _re = active_gaze.get('right_eye')
                _gv = active_gaze.get('gaze_vector')
                if _le is not None and _re is not None and _gv is not None:
                    gaze_ray_history.append((np.array(_le, dtype=float),
                                             np.array(_re, dtype=float),
                                             np.array(_gv, dtype=float)))
                if len(gaze_ray_history) > 0:
                    smoothed_gaze = dict(active_gaze)
                    smoothed_gaze['left_eye']    = tuple(np.mean([h[0] for h in gaze_ray_history], axis=0))
                    smoothed_gaze['right_eye']   = tuple(np.mean([h[1] for h in gaze_ray_history], axis=0))
                    smoothed_gaze['gaze_vector'] = tuple(np.mean([h[2] for h in gaze_ray_history], axis=0))

                # ─── 射線碰撞判定（平滑射線 + 外擴容差框）───
                if current_stage in ACTIVE_STAGES and len(yolo_boxes) > 0:
                    for box in yolo_boxes:
                        # 🌟 A2：TB 也套用同一條射線長度上限，保持判定一致
                        if is_gazing_at_box(smoothed_gaze, expand_box(box, GAZE_BOX_MARGIN),
                                            max_t=MAX_GAZE_RAY_LEN):
                            gazed_obj_boxes.append(box)
                    raw_gaze_obj = len(gazed_obj_boxes) > 0

                # ─── TH 判定（純計算）───
                if current_stage >= 9:
                    for box in robot_boxes:
                        if is_gazing_at_box(smoothed_gaze, expand_box(box, GAZE_BOX_MARGIN),
                                            max_t=MAX_GAZE_RAY_LEN):
                            gazed_robot_boxes.append(box)
                    raw_gaze_tester = len(gazed_robot_boxes) > 0
                else:
                    # ============================================================
                    # 🌟 修改（A1）：改用施測者「頭部框」判定 TH
                    # 舊做法拿 TESTER_ZONE_BBOX（畫面左 35% 滿版長條）做射線
                    # 相交測試——兒童眼睛在右側，視線只要有負的 x 分量就必定
                    # 相交，框測試恆為真，真正生效的只有 pitch/yaw 角度門檻。
                    # 這是對照表中 TH 假陽性 122 筆的結構性來源。
                    # 新做法改用 YOLO-Pose 推出的施測者頭部框 + 外擴容差 +
                    # A2 的射線長度上限；取不到頭部框時退回舊行為。
                    # ============================================================
                    _tester_head = getattr(interaction, "last_tester_head_box", None)
                    if _tester_head is not None:
                        gazed_tester_box = expand_box(_tester_head, TESTER_HEAD_EXPAND)
                        # ⚠️ 角度門檻必須保留（2026-09-14 實測教訓）
                        # 第一版 A1 把 pitch/yaw 門檻拿掉，結果 TH 精確率
                        # 84.5% → 77.9%、假陽性 13 → 21。實例：90 號 13.9s
                        # Gaze P=-14.1 Y=-16.6，yaw 是負的、小朋友根本沒看向
                        # 施測者，舊門檻本來擋得掉。
                        # 頭部框負責「位置」、角度門檻負責「方向」，兩者互補。
                        if is_gazing_at_box(smoothed_gaze, gazed_tester_box,
                                            max_t=MAX_GAZE_RAY_LEN):
                            if pitch_deg > -5 and yaw_deg > 10:
                                raw_gaze_tester = True
                    else:
                        if is_gazing_at_box(smoothed_gaze, TESTER_ZONE_BBOX):
                            if pitch_deg > -5 and yaw_deg > 10:
                                raw_gaze_tester = True

                # ─── FSM 資料封裝 ───
                face_result_for_fsm = {
                    'yolo_head_bbox': active_gaze.get('yolo_head_bbox'),
                    'num_landmarks':  active_gaze.get('num_landmarks', 468)
                }
                pose_result_for_fsm = {
                    'success': True,
                    'euler_angles': active_gaze.get('head_pose')
                }

            # ==================================================
            # 🌟 命中遲滯：立即亮起、延遲熄滅
            # 射線擦框緣或 YOLO 短暫掉框造成的單幀漏判會被橋接，
            # 只有連續 GAZE_HIT_HOLD_FRAMES 幀真的沒命中才判定移開視線。
            # 開始時間不延遲 → TB/TH 反應時間（RT）不受影響。
            # ==================================================
            child_is_gazing_at, child_is_gazing_at_tester = gaze_hold_tracker.update(
                raw_gaze_obj, raw_gaze_tester
            )

            # ==================================================
            # 🎨 繪圖區：視線箭頭、命中框（全畫在 display_frame）
            # ==================================================
            if active_gaze and active_gaze.get('success'):
                face_bbox   = active_gaze.get('face_bbox')
                pitch_rad   = active_gaze['gaze_angles'][0]
                yaw_rad     = active_gaze['gaze_angles'][1]
                gaze_vector = active_gaze['gaze_vector']
                left_eye    = active_gaze.get('left_eye')
                right_eye   = active_gaze.get('right_eye')
                confidence  = active_gaze.get('confidence', 0.0)

                if face_bbox is not None:
                    display_frame = draw_gaze_with_face_box(
                        display_frame, face_bbox, pitch_rad, yaw_rad,
                        gaze_vector=gaze_vector, left_eye=left_eye, right_eye=right_eye,
                        confidence=confidence, show_angles=True,
                        show_direction_label=False, show_gaze_vector=True, bbox_format='xyxy'
                    )

                # 命中物體標記（GAZING!）
                # 🌟 修改：改用判定區算好的 gazed_obj_boxes（平滑射線 + 容差框），
                # 與指示燈完全同步，不再重複做原始射線測試造成畫面閃爍
                if child_is_gazing_at:
                    for box in gazed_obj_boxes:
                        cv2.rectangle(display_frame,
                                       (int(box[0]), int(box[1])),
                                       (int(box[2]), int(box[3])),
                                       (0, 255, 255), 5)
                        cv2.putText(display_frame, "GAZING!",
                                    (int(box[0]), int(box[1]) - 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # TH 視覺化
                if current_stage >= 9 and child_is_gazing_at_tester:
                    for box in gazed_robot_boxes:
                        cv2.rectangle(display_frame,
                                       (int(box[0]), int(box[1])),
                                       (int(box[2]), int(box[3])),
                                       (0, 0, 255), 4)
                        cv2.putText(display_frame, "GAZING AT ROBOT (TH)!",
                                    (int(box[0]), int(box[1]) - 35),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                elif current_stage < 9 and child_is_gazing_at_tester:
                    # 🌟 修改（A1）：命中的是頭部框就畫頭部框，看得出判定依據；
                    # 走退路（頭部框取不到）時才畫回原本的整條 TESTER ZONE
                    if gazed_tester_box is not None:
                        _tx1, _ty1, _tx2, _ty2 = [int(v) for v in gazed_tester_box]
                        cv2.rectangle(display_frame, (_tx1, _ty1), (_tx2, _ty2), (0, 0, 0), 3)
                        cv2.putText(display_frame, "GAZING AT TESTER!",
                                    (_tx1, max(_ty1 - 10, 20)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                    else:
                        cv2.rectangle(display_frame,
                                       (TESTER_ZONE_BBOX[0], TESTER_ZONE_BBOX[1]),
                                       (TESTER_ZONE_BBOX[2], TESTER_ZONE_BBOX[3]),
                                       (0, 0, 0), 3)
                        cv2.putText(display_frame, "GAZING AT TESTER! (zone fallback)",
                                    (TESTER_ZONE_BBOX[0] + 250, TESTER_ZONE_BBOX[1] + 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

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
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                    cv2.putText(display_frame, "YOLO Head Only",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

            # ──────────────────────────────────────────────
            # 4. 評分更新
            # ──────────────────────────────────────────────
            tester_gaze_angles = None
            if active_gaze and active_gaze.get('success'):
                tester_gaze_angles = (
                    active_gaze['gaze_angles_deg'][0],
                    active_gaze['gaze_angles_deg'][1],
                )

            # 🌟 修改：量測結束後不再餵入評分引擎，
            # 避免 11-14 階段的注視/指向累積進 Stage 10 統計
            try:
                if not measurement_over:
                    scoring.update_frame(
                        time_sec=current_time_sec,
                        current_stage=current_stage,
                        is_in_trigger_window=is_in_trigger_window,
                        child_is_pointing_hit=child_is_pointing_hit,
                        child_is_gazing_at=child_is_gazing_at,
                        child_is_gazing_at_tester=child_is_gazing_at_tester,
                        gaze_result=active_gaze,
                        robot_rays=[],
                        robot_boxes=robot_boxes,
                        yolo_boxes=yolo_boxes,
                        # 🌟 修改：計分引擎的 TH(robot_box) 原本用原始 is_gazing_at_box 重算，
                        # 會繞過平滑與遲滯、再度閃爍。改為直接回傳主迴圈已穩定的判定結果。
                        is_gazing_at_box_func=lambda _gaze, _box: child_is_gazing_at_tester,
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
            c_gaze   = (0, 255, 0)    if active_gaze and active_gaze.get('success') else (150, 150, 150)
            c_gazing = (0, 255, 255)  if child_is_gazing_at                  else (0, 0, 255)

            cv2.putText(display_frame,
                        f"Time: {current_time_sec:.1f}s  |  {video_basename}",
                        (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, c_text, 2)
            # 🌟 修改：量測結束後 UI 顯示 DONE，明確告知後續不再計分
            _stage_label = (f"Stage: {current_stage}  [Measuring: 1-10]"
                            if not measurement_over else
                            f"Stage: {current_stage}  [DONE - Stage 11+ reached]")
            cv2.putText(display_frame, _stage_label,
                        (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, c_text, 2)
            cv2.putText(display_frame,
                        f"Keyword: {'YES (Active)' if is_in_trigger_window else 'NO (Idle)'}",
                        (15, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, c_key, 2)
            cv2.putText(display_frame,
                        f"Pointing Hit: {'YES!' if child_is_pointing_hit else 'NO'}",
                        (15, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.8, c_hit, 2)

            if active_gaze and active_gaze.get('success'):
                p = active_gaze['gaze_angles_deg'][0]
                y = active_gaze['gaze_angles_deg'][1]
                cv2.putText(display_frame, f"Gaze: P={p:.1f} Y={y:.1f}",
                            (15, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, c_gaze, 2)
            else:
                status_str = (f"Gaze: Blind ({fsm.current_state})"
                              if fsm.current_state == "EXTREME_TURNING" else "Gaze: N/A")
                cv2.putText(display_frame, status_str,
                            (15, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

            cv2.putText(display_frame,
                        f"Gazing At Object: {'YES!' if child_is_gazing_at else 'NO'}",
                        (15, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.8, c_gazing, 2)

            score_text = (f"Score {scoring.total_score} | "
                          f"S{current_stage} Hit {scoring.stage_gazing_counts.get(current_stage, 0)} | "
                          f"Total {scoring.total_gazing_events}")
            cv2.putText(display_frame, score_text,
                        (15, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, c_text, 2)

            # 🌟 修正：用本幀差值（而非累積量）計算 Other 耗時，避免 _t_other 變負數
            _elapsed_frame = _time.perf_counter() - _t_frame_start
            _t_other += _elapsed_frame - (_t_ocr - _prev_ocr) - (_t_yolo - _prev_yolo) \
                        - (_t_interaction - _prev_interact) - (_t_gaze - _prev_gaze)

            try:
                out.write(display_frame)
            except Exception as _wr_err:
                if frame_count % 100 == 0:
                    print(f"⚠️ out.write 失敗 (Frame {frame_count}): {_wr_err}")

            # ──────────────────────────────────────────────
            # 🌟 新增：把本幀狀態推給外部 UI（frame_callback 為 None 時整段跳過）
            #
            # ⚠️ 契約：callback 是同步呼叫，'frame' 直接傳 display_frame 本身
            #    （不複製）。呼叫端必須在 callback 內就把畫面用完或自行複製，
            #    不可以留著參考——下一幀會覆寫同一塊記憶體。
            #    1080p 一幀約 6 MB，30fps 無條件複製等於每秒白搬 180 MB。
            #    ui/workers.py 的 bgr_to_qimage() 內部的 cvtColor / resize
            #    本來就會產生新陣列並 .copy() 給 Qt，所以這裡不需要再複製。
            #
            # 整段包 try/except：介面出錯絕不影響分析與輸出。
            # ──────────────────────────────────────────────
            if frame_callback is not None:
                try:
                    frame_callback({
                        'frame':              display_frame,
                        'frame_count':        frame_count,
                        'total_frames':       total_frames,
                        'fps':                fps,
                        'time_sec':           current_time_sec,
                        'stage':              current_stage,
                        'sign_stage':         sign_tracker.current_stage,
                        'measurement_over':   measurement_over,
                        'in_trigger_window':  is_in_trigger_window,
                        'pointing_hit':       bool(child_is_pointing_hit),
                        'gazing_object':      bool(child_is_gazing_at),
                        'gazing_tester':      bool(child_is_gazing_at_tester),
                        'gaze_ok':            bool(active_gaze and active_gaze.get('success')),
                        'gaze_angles':        tester_gaze_angles,
                        'fsm_state':          fsm.current_state,
                        'total_score':        scoring.total_score,
                        'total_gazing_events': scoring.total_gazing_events,
                        'stage_gazing_counts': dict(scoring.stage_gazing_counts),
                        'records':            [dict(r) for r in scoring.trigger_event_records],
                        'video_basename':     video_basename,
                    })
                except Exception as _cb_err:
                    if frame_count % 100 == 0:
                        print(f"⚠️ frame_callback 失敗 (Frame {frame_count}): {_cb_err}")

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
                preview_frame = cv2.resize(display_frame, (1280, 720), interpolation=cv2.INTER_NEAREST)
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
                        # 🌟 核心修正 5：適應新版 json 扁平結構，讓 txt 報告能正確印出關鍵字
                        # （相容處理：扁平層讀不到時退回 trigger_events 層，新舊快取都能印）
                        _kws = _rec.get('keywords', [])
                        if not _kws:
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
# 🌟 新增：可重用的初始化函式（CLI 與 UI 共用同一份邏輯）
# 原本這三段程式碼寫死在 main() 裡面，UI 要用只能整段複製，
# 一旦兩邊漂移就會出現「介面跑出來的分數跟命令列不一樣」。
# 抽成函式後 main() 只是呼叫者之一，單一事實來源。
# ============================================================
def load_project_config(config_path=None):
    """讀取 config.yaml，找不到時回傳空 dict（與原本行為一致）。"""
    path = config_path or CONFIG_PATH
    try:
        with open(path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f) or {}
        print(f">>> 成功載入配置：{path}")
        return cfg
    except FileNotFoundError:
        print("⚠️ 找不到 config.yaml，使用預設值")
        return {}


def enable_cuda_optimizations():
    """印出 GPU 狀態並開啟 cudnn.benchmark / TF32（無 CUDA 時只印警告）。"""
    if torch.cuda.is_available():
        _gpu_name   = torch.cuda.get_device_name(0)
        _vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f">>> GPU：{_gpu_name}  |  VRAM：{_vram_total:.1f} GB")
        print(">>> CUDA 加速：✅ 已啟用（YOLO / Gaze / EasyOCR 均使用 GPU）")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return True
    print(">>> ⚠️  CUDA 不可用，所有推論將使用 CPU（速度顯著較慢）")
    print("         請確認 NVIDIA 驅動與 torch+cuda 版本是否匹配")
    return False


def prepare_speech_cache(video_path, output_dir):
    """對單一影片跑 Whisper 語音辨識並建立快取（快取已存在則直接跳過）。

    ⚠️ 必須在 build_shared_components() 之前呼叫：此時 YOLO / Gaze / EasyOCR
    尚未佔用 VRAM，Whisper 才能用 GPU 全速跑（否則會慢 5-10 倍）。
    回傳語音快取所在的子目錄。
    """
    basename    = os.path.splitext(os.path.basename(video_path))[0].strip()
    speech_dir  = os.path.join(output_dir, f'_speech_{basename}')
    cache_path  = os.path.join(speech_dir, 'speech_cache.json')
    if os.path.exists(cache_path):
        print(f"    [{basename}] ✅ 快取已存在，跳過 Whisper")
        return speech_dir
    print(f"    [{basename}] 🔊 執行 Whisper 語音辨識...")
    os.makedirs(speech_dir, exist_ok=True)
    try:
        _sp = SpeechTrigger(
            video_path=video_path,
            output_dir=speech_dir,
            keywords=SPEECH_KEYWORDS,
            noise_sample_path=NOISE_SAMPLE_PATH if os.path.exists(NOISE_SAMPLE_PATH) else None,
        )
        _sp.get_trigger_windows()
        del _sp
    except Exception as _sp_err:
        # 語音辨識失敗不中斷：frame loop 仍可執行，只是沒有觸發時間窗
        print(f"    [{basename}] ⚠️ 語音辨識失敗（{_sp_err}），frame loop 繼續但無語音觸發")
    gc.collect()
    return speech_dir


def build_shared_components(config=None, model_dir=None, ocr_skip=None):
    """一次性載入所有重型 GPU 元件，回傳可跨影片重複使用的元件字典。"""
    CONFIG   = config if config is not None else load_project_config()
    _mdir    = model_dir or MODEL_DIR
    _obj_cfg = CONFIG.get('object_detection', {}) if isinstance(CONFIG, dict) else {}

    model_manager = ModelManager(
        model_dir=_mdir,
        stage5_min_colorful_ratio=_obj_cfg.get('stage5_min_colorful_ratio', 0.12),
    )
    pose_path = os.path.join(_mdir, 'yolo11n-pose.pt')
    hand_path = os.path.join(_mdir, 'gaze', 'hand_landmarker.task')
    interaction   = InteractionEngine(pose_model_path=pose_path,
                                      hand_model_path=hand_path, sma_window=5)
    gaze_pipeline = GazeEstimationPipeline(config=CONFIG.get('gaze_estimation', {}))

    print(">>> 載入 EasyOCR（僅一次）...")
    sign_tracker = SignboardTracker(allowlist='1234567')
    try:
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        sign_tracker.reader.readtext(dummy_img)
        print(">>> EasyOCR 暖機完成")
    except Exception as _ocr_warm_err:
        print(f"⚠️ EasyOCR 暖機失敗（{_ocr_warm_err}），繼續執行（第一幀 OCR 可能較慢）")

    _skip = OCR_SKIP if ocr_skip is None else int(ocr_skip)
    sign_tracker.OCR_FRAME_INTERVAL        = _skip
    sign_tracker.OCR_FRAME_INTERVAL_STAGE7 = _skip
    print(f">>> OCR 跳幀間隔：{_skip} 幀（預設 2）")
    print(">>> 共用模型初始化完成\n")

    return {
        'model_manager': model_manager,
        'interaction':   interaction,
        'gaze_pipeline': gaze_pipeline,
        'sign_tracker':  sign_tracker,
        'gaze_config':   CONFIG.get('gaze_estimation', {}),
        'config':        CONFIG,
    }


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
    #   cudnn.benchmark：固定輸入尺寸重複推論時 cuDNN 會挑最快演算法並快取
    #   allow_tf32：Ampere 以上 GPU 用 TF32 取代 FP32，精度幾乎無損但吞吐提升
    enable_cuda_optimizations()

    # 🌟 修改：切換工作目錄至原始專案根目錄
    os.chdir(PROJECT_DIR)
    print(f">>> 工作目錄已設定為：{PROJECT_DIR}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 載入 config ---
    CONFIG = load_project_config()
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
        prepare_speech_cache(_vp, OUTPUT_DIR)
    print(">>> Step 0 完成：全部影片語音快取就緒\n")

    # --- 一次性初始化重型 GPU 元件（語音預處理完成後才載入，避免 VRAM 衝突）---
    print("=" * 60)
    print("📢  Step 1：載入 GPU 視覺模型")
    print("=" * 60)
    # 🌟 修改：改用抽出的 build_shared_components()，與 UI 共用同一份初始化邏輯
    #    （內含 ModelManager / InteractionEngine / GazePipeline / EasyOCR 暖機
    #      與 OCR 跳幀間隔覆寫，行為與原本逐行寫在這裡時完全相同）
    _comp         = build_shared_components(CONFIG)
    model_manager = _comp['model_manager']
    interaction   = _comp['interaction']
    gaze_pipeline = _comp['gaze_pipeline']
    sign_tracker  = _comp['sign_tracker']

    # --- 批次處理 ---
    for i, video_path in enumerate(video_files, 1):
        video_basename = os.path.splitext(os.path.basename(video_path))[0].strip()  # 🌟 修正：strip()
        print(f"\n[{i}/{len(video_files)}]  {os.path.basename(video_path)}")

        # 🌟 新增：輸出影片與報告皆已存在 → 視為已完成，跳過重跑
        #    （批次中途中斷重新執行時，避免重複處理已完成的影片）
        out_video_path = os.path.join(OUTPUT_DIR, f'{video_basename}.mp4')
        out_txt_path   = os.path.join(OUTPUT_DIR, f'{video_basename}.txt')
        if os.path.exists(out_video_path) and os.path.exists(out_txt_path):
            print(f"    ✅ 已完成（{video_basename}.mp4 / .txt 皆存在），跳過")
            continue

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