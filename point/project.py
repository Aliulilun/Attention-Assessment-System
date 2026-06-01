import os
import sys
import numpy as np
import re
from collections import deque 

# ==========================================
# ★★★ 第一區：系統與模型路徑設定 ★★★
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_FILE_PATH = os.path.join(BASE_DIR, 'video', '9.mp4') 
SAMPLE_DIR = os.path.join(BASE_DIR, 'sample') 

OUTPUT_DIR = os.path.join(BASE_DIR, 'output') 
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR) 
OUTPUT_FILE_PATH = os.path.join(OUTPUT_DIR, 'output_result_v20.mp4') 
TRANSCRIPT_PATH = os.path.join(OUTPUT_DIR, 'transcript_with_events_v20.txt') 

MODEL_FRONT_PATH = os.path.join(BASE_DIR, 'model', 'front_model.pt')
MODEL_BACKGROUND_PATH = os.path.join(BASE_DIR, 'model', 'background_model.pt')
MODEL_BALLOON_PATH = os.path.join(BASE_DIR, 'model', 'balloon_model.pt')
MODEL_BUBBLE_PATH = os.path.join(BASE_DIR, 'model', 'bubble_model.pt')
MODEL_TOY_PATH = os.path.join(BASE_DIR, 'model', 'toy_model.pt') 
MODEL_POSE_PATH = 'yolo11n-pose.pt'

MODEL_ROBOT_POINT_PATH = os.path.join(BASE_DIR, 'model', 'robot_point_model.pt')

REF_IMGS = {str(i): os.path.join(SAMPLE_DIR, f'{i}.jpg') for i in range(1, 9)}

STAGE_LEVELS = {
    "準備中": 0, "第一部分": 1, "第二部分": 2, "第三部分": 3,
    "第四部分": 4, "第五部分": 5, "第六部分": 6, "第七部分": 7, "第八部分": 8
}

# ==========================================
# ★★★ 第二區：核心演算法參數微調 ★★★
# ==========================================
TARGET_KEYWORDS = ["開始", "321", "三二一", "3 2 1", "準備", "你看", "看這裡", "準備囉"]
RESPONSE_WINDOW_SEC = 3.0   

SEARCH_PAD = 60             
ALPHA_SMOOTH = 0.25         
MATCH_INTERVAL = 3          
MATCH_THRESHOLD = 0.50      

YOLO_INTERVAL = 2       

AUTO_ROI_START_RATIO = 0.30  
CONF_FRONT = 0.35        
CONF_BACKGROUND = 0.35   
CONF_BALLOON = 0.50      
CONF_BUBBLE = 0.35           
CONF_TOY = 0.40          
CONF_POSE = 0.50             

CONF_ROBOT = 0.60
IOU_ROBOT = 0.50

TOY_TOP_LIMIT_RATIO = 0.10     
BOTTOM_LIMIT_RATIO = 0.75      
TOY_MAX_HEIGHT_RATIO = 0.25    
FACE_EXCLUSION_RATIO = 0.10    
BUBBLE_IGNORE_RIGHT_RATIO = 0.80 
BUBBLE_BEAM_WIDTH = 100          
MAX_BUBBLE_SIZE_RATIO = 0.15    

TOUCH_TOLERANCE = 15          
TOUCH_DISTANCE_LIMIT = 40    

C_CHILD = (255, 0, 0); C_TESTER = (128, 128, 128); C_TEXT = (0, 0, 255)
C_FRONT = (255, 128, 0); C_BACKGROUND = (0, 128, 255) 
C_BALLOON = (255, 0, 255); C_BUBBLE = (0, 255, 255); C_TOY = (0, 255, 0)
C_BEAM_ZONE = (0, 255, 255)
C_RAY_CHILD = (0, 255, 0)      
C_RAY_TESTER = (0, 165, 255)   
C_TOUCH_WARN = (0, 0, 255)    

# ==========================================
#  機器人平滑穩定機制 (SMA Buffer) 
# ==========================================
SMOOTHING_FRAMES = 5
robot_base_buffer = deque(maxlen=SMOOTHING_FRAMES)
robot_tip_buffer = deque(maxlen=SMOOTHING_FRAMES)

def get_smoothed_point(buffer):
    valid_points = [pt for pt in buffer if pt is not None]
    if not valid_points: return None
    avg_x = int(sum([pt[0] for pt in valid_points]) / len(valid_points))
    avg_y = int(sum([pt[1] for pt in valid_points]) / len(valid_points))
    return (avg_x, avg_y)

# ==========================================
#  [第一階段] Whisper 語音快取機制 
# ==========================================
print(f"--- 系統啟動中 ---")
if not os.path.exists(VIDEO_FILE_PATH): sys.exit(f" 找不到影片檔案：{VIDEO_FILE_PATH}")

trigger_windows = [] 
if os.path.exists(TRANSCRIPT_PATH):
    print(f"\n 發現快取語音紀錄檔 ({TRANSCRIPT_PATH})，瞬間載入！")
    with open(TRANSCRIPT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if "已啟動" in line and "的專屬判定視窗" in line:
                nums = re.findall(r"\d+\.\d+", line)
                if len(nums) >= 2: trigger_windows.append((float(nums[0]), float(nums[1])))
else:
    print("\n [階段一] 優先執行 AI 語音觸發解析 (需時數分鐘，請稍候)...")
    try:
        import whisper
        whisper_model = whisper.load_model("large-v3") 
        result = whisper_model.transcribe(
            VIDEO_FILE_PATH, language="zh", condition_on_previous_text=False, 
            no_speech_threshold=0.4, compression_ratio_threshold=2.4, fp16=False, verbose=False
        ) 
        with open(TRANSCRIPT_PATH, "w", encoding="utf-8") as f:
            f.write("=== 影片語音逐字稿 (含觸發事件紀錄表) ===\n")
            f.write(f"系統設定：大型模型啟動，偵測到關鍵字後開啟 {RESPONSE_WINDOW_SEC} 秒專屬判定\n\n")
            last_text = ""
            for segment in result["segments"]:
                start_time = segment["start"]; end_time = segment["end"]; text = segment["text"].strip()
                if text == last_text: continue
                last_text = text
                time_str = f"[{int(start_time//60):02d}:{int(start_time%60):02d} - {int(end_time//60):02d}:{int(end_time%60):02d}]"
                found_keywords = [kw for kw in TARGET_KEYWORDS if kw in text]
                if found_keywords:
                    end_window = start_time + RESPONSE_WINDOW_SEC
                    trigger_windows.append((start_time, end_window))
                    f.write(f" {time_str}  [觸發: {'/'.join(found_keywords)}] {text}\n       (已啟動專屬判定)\n")
                else: f.write(f"   {time_str} {text}\n")
    except Exception as e:
        print(f"\n 語音解析失敗: {e}"); sys.exit()

# ==========================================
#  [第二階段] 載入視覺套件與初始化 
# ==========================================
print("\n>>> 正在喚醒視覺核心套件...")
try:
    import cv2
    import mediapipe as mp
    from PIL import Image, ImageDraw, ImageFont
    from ultralytics import YOLO
    import traceback
except ImportError as e: sys.exit(f" 視覺套件載入失敗: {e}")

def imread_chinese(path):
    if os.path.exists(path): return cv2.imdecode(np.fromfile(path, dtype=np.uint8), 0) 
    return None

templates = {key: imread_chinese(path) for key, path in REF_IMGS.items() if imread_chinese(path) is not None}
if not templates: sys.exit(" 錯誤：找不到任何字卡樣板！")

mp_hands = mp.solutions.hands

try:
    model_front = YOLO(MODEL_FRONT_PATH)
    model_background = YOLO(MODEL_BACKGROUND_PATH)
    model_balloon = YOLO(MODEL_BALLOON_PATH)
    model_bubble = YOLO(MODEL_BUBBLE_PATH)
    model_toy = YOLO(MODEL_TOY_PATH)
    model_pose = YOLO(MODEL_POSE_PATH) 
    if not os.path.exists(MODEL_ROBOT_POINT_PATH): sys.exit(f" 找不到機器人模型：{MODEL_ROBOT_POINT_PATH}")
    model_robot_point = YOLO(MODEL_ROBOT_POINT_PATH)
except Exception as e: sys.exit(f"YOLO 載入失敗: {e}")

cap = cv2.VideoCapture(VIDEO_FILE_PATH)
success, first_frame = cap.read()
if not success: sys.exit("無法讀取影片")

FRAME_H, FRAME_W = first_frame.shape[:2]
FRAME_FPS = cap.get(cv2.CAP_PROP_FPS) 
BOTTOM_LIMIT_Y = int(FRAME_H * BOTTOM_LIMIT_RATIO); TOY_TOP_LIMIT_Y = int(FRAME_H * TOY_TOP_LIMIT_RATIO) 
FACE_EXCLUSION_RADIUS = int(FRAME_H * FACE_EXCLUSION_RATIO); DIVIDER_X = int(FRAME_W * AUTO_ROI_START_RATIO)

def smart_match_gray(roi, tmpl_gray):
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    found = None; (tH, tW) = tmpl_gray.shape[:2]
    for scale in np.linspace(0.6, 1.2, 8): 
        resized_w, resized_h = int(tW * scale), int(tH * scale)
        if resized_h > roi_gray.shape[0] or resized_w > roi_gray.shape[1]: continue
        result = cv2.matchTemplate(roi_gray, cv2.resize(tmpl_gray, (resized_w, resized_h)), cv2.TM_CCOEFF_NORMED)
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
        if found is None or maxVal > found[0]: found = (maxVal, maxLoc, scale, resized_w, resized_h)
    return found if found else (0.0, (0,0), 1.0, 0, 0)

def draw_chinese_text(img, text, position, text_color, text_size=30):
    try:
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        font = ImageFont.truetype("C:/Windows/Fonts/msjh.ttc", text_size)
        draw.text(position, text, fill=text_color, font=font)
        return cv2.cvtColor(np.asarray(img_pil), cv2.COLOR_RGB2BGR)
    except: return img

def check_pointing_ray_only(img, wrist, index, target_data, mode="box"):
    if wrist is None or index is None: return False, wrist, wrist
    vec = np.array(index) - np.array(wrist)
    if np.linalg.norm(vec) == 0: return False, wrist, wrist
    unit_vec = vec / np.linalg.norm(vec)
    start_point = index; end_point = np.array(start_point) + unit_vec * 2000 
    pt1, pt2 = (int(start_point[0]), int(start_point[1])), (int(end_point[0]), int(end_point[1]))
    is_hit = False
    
    if mode == "robot_pointing": return True, pt1, pt2
        
    if mode == "box":
        for box in (target_data if isinstance(target_data, list) else [target_data]):
            if box is None: continue
            bx1, by1, bx2, by2 = map(int, box)
            hit, _, _ = cv2.clipLine((bx1, by1, bx2-bx1, by2-by1), pt1, pt2)
            if hit: is_hit = True; break
    elif mode == "beam":
        for box in target_data:
            bx1, by1, bx2, by2 = map(int, box)
            cx, cy = (bx1 + bx2) // 2, (by1 + by2) // 2
            ap = np.array((cx, cy)) - np.array(start_point)
            dist = np.linalg.norm(np.cross(vec, ap)) / np.linalg.norm(vec) if np.linalg.norm(vec)!=0 else 0
            if dist < BUBBLE_BEAM_WIDTH and np.dot(vec, ap) > 0:
                cv2.line(img, pt1, (cx, cy), C_BEAM_ZONE, 1); is_hit = True; break
    return is_hit, pt1, pt2

def is_hand_touching_object_strict(img, wrist, index, pinky, box, margin=TOUCH_TOLERANCE):
    if box is None: return False
    bx1, by1, bx2, by2 = map(int, box)
    touch_x1, touch_y1 = bx1 - margin, by1 - margin
    touch_x2, touch_y2 = bx2 + margin, by2 + margin
    def is_in(pt): return False if pt is None else (touch_x1 <= pt[0] <= touch_x2) and (touch_y1 <= pt[1] <= touch_y2)
    if is_in(wrist) or is_in(index) or is_in(pinky): return True
    if index is not None and np.linalg.norm(np.array(index) - np.array([(bx1+bx2)//2, (by1+by2)//2])) < TOUCH_DISTANCE_LIMIT: return True
    return False

def select_roi_native(window_name, frame, target_width=1000):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    h, w = frame.shape[:2]
    cv2.resizeWindow(window_name, target_width, int(target_width * h / w))
    roi = cv2.selectROI(window_name, frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(window_name)
    return roi if roi[2] > 0 else (0,0,0,0)

def calculate_arm_link_score(mp_wrist, kpts, confs, side="right"):
    s_idx, e_idx, w_idx = (6, 8, 10) if side == "right" else (5, 7, 9)
    if confs[w_idx] > 0.4: return np.linalg.norm(np.array(mp_wrist) - np.array(kpts[w_idx]))
    if confs[e_idx] > 0.4 and confs[s_idx] > 0.4:
        S, E, W_mp = np.array(kpts[s_idx]), np.array(kpts[e_idx]), np.array(mp_wrist)
        arm_vec = E - S 
        arm_len = np.linalg.norm(arm_vec)
        if arm_len == 0: return float('inf')
        unit_arm = arm_vec / arm_len
        elbow_to_mp = W_mp - E
        proj_len = np.dot(elbow_to_mp, unit_arm) 
        if proj_len < -30: return float('inf') 
        dist_to_line = np.abs(np.cross(unit_arm, elbow_to_mp))
        return dist_to_line + (proj_len * 0.1)
    if confs[e_idx] > 0.4: return np.linalg.norm(np.array(mp_wrist) - np.array(kpts[e_idx])) + 50
    if confs[s_idx] > 0.4: return np.linalg.norm(np.array(mp_wrist) - np.array(kpts[s_idx])) + 100
    return float('inf')

# ==========================================
# ★ 執行主迴圈 ★
# ==========================================
tx, ty, tw, th = select_roi_native("STEP: Select Card 1", first_frame)
if tw == 0: sys.exit()

#  紀錄「最初的完美框大小」，防止迷失時框變形
init_w, init_h = tw, th

card_cx, card_cy, card_w, card_h = tx + tw // 2, ty + th // 2, tw, th
smoothed_box = (tx, ty, tx+tw, ty+th)
current_stage_num, current_stage, lost_patience = 1, "第一部分", 0

stage_data = { k: {"frames": 0, "point": 0} for k in STAGE_LEVELS.keys() if k != "準備中" }
is_pointing_now = { k: False for k in STAGE_LEVELS.keys() }; pointing_log = []; last_logged_second = -1 
frame_cnt = 0

fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out_writer = cv2.VideoWriter(OUTPUT_FILE_PATH, fourcc, FRAME_FPS, (FRAME_W, FRAME_H))

main_win = 'Analysis: Ultimate Integrated'
cv2.namedWindow(main_win, cv2.WINDOW_NORMAL)
cv2.resizeWindow(main_win, 1000, int(1000 * FRAME_H / FRAME_W))

target_list = []; current_mode = "none"; target_name = ""
tester_id = None; child_id = None

print("\n 開始全幀視覺分析...")

try:
    with mp_hands.Hands(min_detection_confidence=0.2, min_tracking_confidence=0.2, max_num_hands=4) as hands_detector:
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break
            frame_cnt += 1; out_img = frame.copy()
            current_time_sec = frame_cnt / FRAME_FPS
            time_str = f"{int(current_time_sec//60):02d}:{int(current_time_sec%60):02d}"
            is_in_trigger_window = any(start <= current_time_sec <= end for start, end in trigger_windows)

            # --- 1. 結界追蹤 ---
            if frame_cnt % MATCH_INTERVAL == 0:
                if lost_patience > 10: 
                    # 找不到牌子，可能換牌子了。強制將方框重置回最初完美大小，並在周圍給予額外的搜索範圍
                    card_w, card_h = init_w, init_h
                    current_pad = 250 # 擴大至 250px 的額外範圍
                else:
                    # 平時正常追蹤，給予 60px 的範圍
                    current_pad = SEARCH_PAD 

                sx1, sy1 = max(0, int(card_cx - card_w//2 - current_pad)), max(0, int(card_cy - card_h//2 - current_pad))
                sx2, sy2 = min(FRAME_W, int(card_cx + card_w//2 + current_pad)), min(FRAME_H, int(card_cy + card_h//2 + current_pad))
                crop = frame[sy1:sy2, sx1:sx2]

                best_score = 0; best_key = None; best_info = None
                for key, tmpl_gray in templates.items():
                    if int(key) < current_stage_num: continue 
                    if int(key) > current_stage_num + 2: continue 
                    score, loc, scale, w_res, h_res = smart_match_gray(crop, tmpl_gray)
                    if score > best_score: best_score = score; best_key = key; best_info = (loc, w_res, h_res)
                
                if best_score > MATCH_THRESHOLD:
                    lost_patience = 0 
                    new_cx, new_cy = sx1 + best_info[0][0] + best_info[1]//2, sy1 + best_info[0][1] + best_info[2]//2
                    card_cx = card_cx * (1 - ALPHA_SMOOTH) + new_cx * ALPHA_SMOOTH
                    card_cy = card_cy * (1 - ALPHA_SMOOTH) + new_cy * ALPHA_SMOOTH
                    card_w, card_h = best_info[1], best_info[2]
                    smoothed_box = (card_cx - card_w//2, card_cy - card_h//2, card_cx + card_w//2, card_cy + card_h//2)
                    
                    if int(best_key) > current_stage_num:
                        current_stage_num = int(best_key)
                        stage_map = {"1":"第一部分", "2":"第二部分", "3":"第三部分", "4":"第四部分", "5":"第五部分", "6":"第六部分", "7":"第七部分", "8":"第八部分"}
                        current_stage = stage_map[best_key]
                        print(f"[{time_str}] 階段推進: {current_stage} (分數:{best_score:.2f})")
                else: 
                    lost_patience += 1

            bx1, by1, bx2, by2 = map(int, smoothed_box)
            if lost_patience <= 15: 
                cv2.rectangle(out_img, (bx1, by1), (bx2, by2), (0, 255, 0), 4, cv2.LINE_AA)
                cv2.putText(out_img, "Tracking", (bx1, by1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)
            elif lost_patience <= 30: 
                cv2.rectangle(out_img, (bx1, by1), (bx2, by2), (0, 0, 255), 4, cv2.LINE_AA)
                cv2.putText(out_img, "Waiting...", (bx1, by1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
            else: 
                cv2.rectangle(out_img, (bx1, by1), (bx2, by2), (0, 255, 255), 4, cv2.LINE_AA)
                cv2.putText(out_img, "Searching...", (bx1, by1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 3)

            # --- 2. YOLO 物件與持續追蹤骨架 ---
            yolo_people = []    
            pose_results = model_pose.track(frame, persist=True, imgsz=640, conf=CONF_POSE, verbose=False)
            
            if pose_results and len(pose_results)>0 and pose_results[0].keypoints is not None:
                kpts_all = pose_results[0].keypoints.xy.cpu().numpy()
                conf_all = pose_results[0].keypoints.conf.cpu().numpy()
                boxes_all = pose_results[0].boxes.xyxy.cpu().numpy()
                track_ids = pose_results[0].boxes.id.int().cpu().tolist() if pose_results[0].boxes.id is not None else [-1]*len(boxes_all)
                
                for i in range(len(boxes_all)):
                    tid = track_ids[i]
                    cx = (boxes_all[i][0] + boxes_all[i][2]) / 2
                    
                    if tid != -1:
                        if tester_id is None and cx < DIVIDER_X: tester_id = tid
                        elif child_id is None and cx > DIVIDER_X: child_id = tid

                        if tid == tester_id: owner = "Tester"
                        elif tid == child_id: owner = "Child"
                        else: owner = "Child" if cx > DIVIDER_X else "Tester"
                    else: owner = "Child" if cx > DIVIDER_X else "Tester"
                        
                    yolo_people.append({"owner": owner, "box": boxes_all[i], "kpts": kpts_all[i], "confs": conf_all[i]})

            if frame_cnt % YOLO_INTERVAL == 0:
                target_list = []; current_mode = "none"; target_name = ""
                
                if current_stage in ["準備中", "第一部分", "第二部分"]: 
                    current_mode = "box"; target_name = "Front Object"
                    for r in model_front(frame, imgsz=640, conf=CONF_FRONT, verbose=False):
                        for box in r.boxes: target_list.append(box.xyxy[0].cpu().numpy())
                elif current_stage in ["第三部分", "第四部分"]: 
                    current_mode = "box"; target_name = "Background Object"
                    for r in model_background(frame, imgsz=640, conf=CONF_BACKGROUND, verbose=False):
                        for box in r.boxes: target_list.append(box.xyxy[0].cpu().numpy())
                elif current_stage == "第五部分": 
                    current_mode = "box"; target_name = "Balloon"
                    for r in model_balloon(frame, imgsz=640, conf=CONF_BALLOON, verbose=False):
                        for box in r.boxes:
                            b = box.xyxy[0].cpu().numpy()
                            if (b[1]+b[3])/2 < BOTTOM_LIMIT_Y: target_list.append(b)
                elif current_stage == "第六部分": 
                    current_mode = "beam"; target_name = "Bubble"
                    for r in model_bubble(frame, imgsz=640, conf=CONF_BUBBLE, verbose=False):
                        for box in r.boxes:
                            b = box.xyxy[0].cpu().numpy()
                            if b[2]-b[0] <= (FRAME_W * MAX_BUBBLE_SIZE_RATIO) and (b[0]+b[2])/2 <= (FRAME_W * BUBBLE_IGNORE_RIGHT_RATIO): target_list.append(b)
                elif current_stage == "第七部分":
                    current_mode = "box"; target_name = "Toy"
                    for r in model_toy(frame, imgsz=640, conf=CONF_TOY, verbose=False):
                        for box in r.boxes:
                            b = box.xyxy[0].cpu().numpy(); cx, cy = (b[0]+b[2])/2, (b[1]+b[3])/2
                            if b[3]-b[1] <= (FRAME_H * TOY_MAX_HEIGHT_RATIO) and (TOY_TOP_LIMIT_Y < cy < BOTTOM_LIMIT_Y):
                                face_pts = [ (int(p["kpts"][j][0]), int(p["kpts"][j][1])) for p in yolo_people for j in range(5) if p["confs"][j]>0.5 ]
                                if not any(np.linalg.norm(np.array([cx, cy]) - np.array(fpt)) < FACE_EXCLUSION_RADIUS for fpt in face_pts): target_list.append(b)
                elif current_stage == "第八部分":
                    current_mode = "robot_pointing"; target_name = "Robot Pointing"
            
            for b in target_list:
                color = C_BALLOON
                if target_name == "Front Object": color = C_FRONT
                elif target_name == "Background Object": color = C_BACKGROUND
                elif target_name == "Bubble": color = C_BUBBLE
                elif target_name == "Toy": color = C_TOY
                
                bx1, by1, bx2, by2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                cv2.rectangle(out_img, (bx1, by1), (bx2, by2), color, 4) 
                cv2.putText(out_img, target_name, (bx1, by1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # ==========================================================
            # --- 3. 動作判定機制 ---
            # ==========================================================
            child_scored = False; child_is_touching = False
            tester_pointing_hits = []; child_pointing_hits = []

            if current_stage != "第八部分":
                mp_results = hands_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if mp_results.multi_hand_landmarks:
                    for i, hand_lms in enumerate(mp_results.multi_hand_landmarks):
                        p_wri, p_idx, p_pnk = (int(hand_lms.landmark[0].x * FRAME_W), int(hand_lms.landmark[0].y * FRAME_H)), (int(hand_lms.landmark[8].x * FRAME_W), int(hand_lms.landmark[8].y * FRAME_H)), (int(hand_lms.landmark[20].x * FRAME_W), int(hand_lms.landmark[20].y * FRAME_H))
                        
                        best_owner = None
                        best_score = float('inf') 
                        
                        for person in yolo_people:
                            score_left = calculate_arm_link_score(p_wri, person["kpts"], person["confs"], "left")
                            score_right = calculate_arm_link_score(p_wri, person["kpts"], person["confs"], "right")
                            score = min(score_left, score_right)
                            if score < best_score:
                                best_score = score
                                best_owner = person["owner"]
                                
                        if best_score > 400 or best_owner is None:
                            vec_x = hand_lms.landmark[0].x - hand_lms.landmark[9].x
                            if vec_x < -0.015: best_owner = "Tester" 
                            elif vec_x > 0.015: best_owner = "Child" 
                            else: best_owner = "Child" if p_wri[0] > DIVIDER_X else "Tester"

                        vec_x = hand_lms.landmark[0].x - hand_lms.landmark[9].x
                        if best_owner == "Child" and vec_x < -0.02: best_owner = "Tester" 
                        elif best_owner == "Tester" and vec_x > 0.02: best_owner = "Child"  

                        hand_color = C_RAY_CHILD if best_owner == "Child" else C_RAY_TESTER
                        for pt in [4, 8, 12, 16, 20]: 
                            cv2.circle(out_img, (int(hand_lms.landmark[pt].x * FRAME_W), int(hand_lms.landmark[pt].y * FRAME_H)), 4, hand_color, -1)
                        cv2.line(out_img, p_wri, p_idx, hand_color, 2)
                        
                        if best_owner == "Child":
                            for box in target_list:
                                if is_hand_touching_object_strict(out_img, p_wri, p_idx, p_pnk, box): child_is_touching = True
                            hit, p1, p2 = check_pointing_ray_only(out_img, p_wri, p_idx, target_list, mode=current_mode)
                            if hit: child_pointing_hits.append((p1, p2))
                                
                        elif best_owner == "Tester":
                            hit, p1, p2 = check_pointing_ray_only(out_img, p_wri, p_idx, target_list, mode=current_mode)
                            if hit: tester_pointing_hits.append((p1, p2))

            elif current_stage == "第八部分":
                robot_results = model_robot_point(frame, conf=CONF_ROBOT, iou=IOU_ROBOT, verbose=False)
                current_base = None
                current_tip = None
                
                if robot_results and len(robot_results) > 0:
                    r = robot_results[0]
                    if r.keypoints is not None and r.boxes is not None:
                        boxes_r = r.boxes.xyxy.cpu().numpy()
                        kpts_r = r.keypoints.xy.cpu().numpy() 
                        
                        if len(boxes_r) > 0:
                            bx1, by1, bx2, by2 = boxes_r[0] 
                            cx, cy = int((bx1 + bx2) / 2), int((by1 + by2) / 2)
                            
                            if len(kpts_r[0]) > 0:
                                tx, ty = int(kpts_r[0][0][0]), int(kpts_r[0][0][1])
                                if tx > 0 and ty > 0:
                                    current_base, current_tip = (cx, cy), (tx, ty)

                robot_base_buffer.append(current_base)
                robot_tip_buffer.append(current_tip)

                smooth_base = get_smoothed_point(robot_base_buffer)
                smooth_tip = get_smoothed_point(robot_tip_buffer)

                if smooth_base and smooth_tip:
                    cv2.circle(out_img, smooth_base, 8, (0, 255, 255), -1) 
                    cv2.circle(out_img, smooth_tip, 8, (0, 0, 255), -1)    
                    
                    hit, p1, p2 = check_pointing_ray_only(out_img, smooth_base, smooth_tip, [], mode="robot_pointing")
                    if hit:
                        tester_pointing_hits.append((p1, p2)) 

            # ==========================================================
            # --- 4. 畫面渲染與計分 ---
            # ==========================================================
            for (p1, p2) in tester_pointing_hits: 
                ray_color = (0, 255, 255) if current_stage == "第八部分" else C_RAY_TESTER
                cv2.line(out_img, p1, p2, ray_color, 4 if current_stage == "第八部分" else 2)
            
            if current_stage == "第八部分":
                cv2.putText(out_img, "[STAGE 8] DETECTING ROBOT POINTING", (DIVIDER_X + 20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                if len(tester_pointing_hits) > 0:
                    if not is_pointing_now[current_stage]:
                        stage_data[current_stage]["point"] += 1; is_pointing_now[current_stage] = True 
                        current_sec_int = int(current_time_sec)
                        if current_sec_int != last_logged_second:
                            print(f" [得分紀錄] {time_str} (第八部分) - 偵測到機器人平滑指向")  
                            last_logged_second = current_sec_int
                else: is_pointing_now[current_stage] = False
            else:
                if child_is_touching:
                    cv2.putText(out_img, "CHILD TOUCHING (IGNORED)", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, C_TOUCH_WARN, 2)
                else:
                    if len(child_pointing_hits) > 0:
                        child_scored = True
                        for (p1, p2) in child_pointing_hits: cv2.line(out_img, p1, p2, C_RAY_CHILD, 3) 

                if current_stage in stage_data:
                    stage_data[current_stage]["frames"] += 1
                    if child_scored:
                        cv2.putText(out_img, "CHILD POINTING HIT!", (DIVIDER_X + 50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, C_RAY_CHILD, 3)
                        if not is_pointing_now[current_stage]:
                            stage_data[current_stage]["point"] += 1; is_pointing_now[current_stage] = True 
                            current_sec_int = int(current_time_sec)
                            if current_sec_int != last_logged_second:
                                log_msg = f"{time_str} ({current_stage}) - 命中目標: {target_name}"
                                pointing_log.append(log_msg); print(f" [得分紀錄] {log_msg}")  
                                last_logged_second = current_sec_int
                    else: is_pointing_now[current_stage] = False

            if is_in_trigger_window:
                cv2.rectangle(out_img, (0, 0), (FRAME_W, FRAME_H), (0, 165, 255), 8)
                out_img = draw_chinese_text(out_img, "⚡ 語音觸發判定中...", (FRAME_W//2 - 120, 30), (0, 165, 255), 40)

            cv2.line(out_img, (DIVIDER_X, 0), (DIVIDER_X, FRAME_H), (255, 255, 255), 2) 
            cv2.putText(out_img, "TESTER ZONE", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_TESTER, 1)
            cv2.putText(out_img, "CHILD ZONE", (DIVIDER_X + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_CHILD, 1)
            
            cv2.rectangle(out_img, (0, 30), (400, 100), (0, 0, 0), -1)
            out_img = cv2.addWeighted(out_img, 0.8, out_img, 0.2, 0)
            out_img = draw_chinese_text(out_img, f"階段: {current_stage}", (10, 50), (255,0,0), 30)
            
            out_writer.write(out_img)
            cv2.imshow(main_win, out_img)
            if cv2.waitKey(1) & 0xFF in [ord('q'), 27]: break

except Exception as e: print(f"\n 程式發生錯誤: {e}"); traceback.print_exc()
finally:
    cap.release(); out_writer.release(); cv2.destroyAllWindows()
    print("\n" + "="*80)
    print(f"{'階段名稱':^10} | {'總幀數':^8} | {'小孩有效指向次數 (第八階段為機器人)':^12}")
    print("="*80)
    for k, v in stage_data.items(): print(f"{k:^10} | {v['frames']:^8} | {v['point']:^12}")
    print("="*80)

    try:
        from moviepy import VideoFileClip, AudioFileClip
        FINAL_VIDEO_PATH = os.path.join(OUTPUT_DIR, 'output_with_audio_v20.mp4')
        video_clip = VideoFileClip(OUTPUT_FILE_PATH); audio_clip = AudioFileClip(VIDEO_FILE_PATH)
        video_clip.with_audio(audio_clip).write_videofile(FINAL_VIDEO_PATH, codec="libx264", audio_codec="aac", logger=None)
        video_clip.close(); audio_clip.close()
        print(f"🎉 聲音縫合成功！")
    except Exception as e: print(f"\n 聲音合併失敗: {e}")