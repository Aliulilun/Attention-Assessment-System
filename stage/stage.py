import os
import sys
import cv2
import easyocr

# ==========================================
# 1. 初始化與路徑設定
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(BASE_DIR, 'video', '10.mp4')

# 防呆機制：確保 output 資料夾存在，避免存檔時崩潰
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'output_ocr_window1.mp4')

print(">>> 正在載入 EasyOCR 引擎 (首次執行可能稍慢)...")
reader = easyocr.Reader(['en']) 

cap = cv2.VideoCapture(VIDEO_PATH)
success, first_frame = cap.read()
if not success:
    sys.exit(f"❌ 找不到影片：{VIDEO_PATH}")

FRAME_H, FRAME_W = first_frame.shape[:2]
FPS = cap.get(cv2.CAP_PROP_FPS)

# ==========================================
# 2. 框選大範圍「搜索結界」
# ==========================================
print(">>> 請框選牌子可能出現的「大範圍區域」(例如右下角整個桌子)")
print(">>> 提示：按住滑鼠左鍵拖曳，選好後按 Enter 或 Space 鍵確認。")

# 強制允許調整視窗大小，並預設縮小到好操作的解析度
cv2.namedWindow("Select OCR Zone", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Select OCR Zone", 1280, 720)

roi = cv2.selectROI("Select OCR Zone", first_frame, showCrosshair=True, fromCenter=False)
cv2.destroyWindow("Select OCR Zone")

if roi[2] == 0:
    sys.exit("⚠️ 未框選，結束程式。")

zone_x, zone_y, zone_w, zone_h = roi 
current_stage = 1
frame_cnt = 0

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, FPS, (FRAME_W, FRAME_H))

print("\n✅ 開始即時視覺辨識！")
print("⚠️ 提示：隨時可以按視窗上的 'Esc' 鍵，或在終端機按 'Ctrl + C' 安全退出！\n")

# ==========================================
# 3. 主迴圈
# ==========================================
cv2.namedWindow('OCR Live Test', cv2.WINDOW_NORMAL)
cv2.resizeWindow('OCR Live Test', 1280, 720)

# 用來記憶上一秒抓到的牌子位置，讓綠框在沒掃描的幀數也能保持顯示 (防閃爍)
last_best_detection = None 

try:
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        frame_cnt += 1
        out_img = frame.copy()

        # 畫出搜索結界 (黃框)
        cv2.rectangle(out_img, (zone_x, zone_y), (zone_x+zone_w, zone_y+zone_h), (0, 255, 255), 2)
        cv2.putText(out_img, "OCR Search Zone", (zone_x, zone_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # 每 15 幀 (約 0.5 秒) 掃描一次
        if frame_cnt % 15 == 0:
            crop_img = frame[zone_y : zone_y+zone_h, zone_x : zone_x+zone_w]
            gray_crop = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            
            # 🛠️ 強化：放大 2.5 倍進行辨識，專治像「7」這種極細字體
            results = reader.readtext(gray_crop, allowlist='12345678', mag_ratio=2.5)

            valid_detections = []
            
            # 收集所有符合條件的數字 (包含被放倒在桌上的舊牌子)
            for (bbox, text, prob) in results:
                clean_text = text.strip()
                
                # 🛠️ 修正 Bug：確保 "76" 這種連體嬰能被同時拆解出 7 和 6 (不使用 break)
                for target_num in ['1', '2', '3', '4', '5', '6', '7', '8']:
                    if target_num in clean_text and prob > 0.20:
                        valid_detections.append({
                            "num": int(target_num),
                            "bbox": bbox,
                            "prob": prob
                        })
            
            if valid_detections:
                all_nums = [d["num"] for d in valid_detections]
                max_num = max(all_nums)
                
                # 階段推進邏輯：只允許前進，最多允許一次跳 2 級
                if current_stage <= max_num <= (current_stage + 2):
                    if current_stage < max_num:
                        print(f"[{frame_cnt//FPS:.1f}s] 🎯 成功推進至階段: {max_num}")
                    current_stage = max_num
                
                # 單一鎖定過濾：只挑選跟「當前階段」一樣的數字，其他的當成垃圾忽略
                best_detection = None
                for d in valid_detections:
                    if d["num"] == current_stage:
                        if best_detection is None or d["prob"] > best_detection["prob"]:
                            best_detection = d
                
                # 更新記憶
                last_best_detection = best_detection
            else:
                last_best_detection = None # 沒抓到東西就清空

        # 每一幀都負責畫出「記憶中最新」的綠框
        if last_best_detection:
            bbox = last_best_detection["bbox"]
            prob = last_best_detection["prob"]
            
            bx1 = int(bbox[0][0]) + zone_x
            by1 = int(bbox[0][1]) + zone_y
            bx2 = int(bbox[2][0]) + zone_x
            by2 = int(bbox[2][1]) + zone_y
            
            cv2.rectangle(out_img, (bx1, by1), (bx2, by2), (0, 255, 0), 3)
            cv2.putText(out_img, f"Found: {current_stage} ({prob:.2f})", (bx1, by1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 顯示系統當前認定的階段
        cv2.putText(out_img, f"Current Stage: {current_stage}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

        out_writer.write(out_img)
        
        # 即時顯示有視窗的畫面
        cv2.imshow('OCR Live Test', out_img)
        if cv2.waitKey(1) & 0xFF == 27: # 按 Esc 退出
            print("\n🛑 收到 Esc 退出指令！")
            break

except KeyboardInterrupt:
    print("\n🛑 收到終端機 Ctrl+C 中斷指令！")

finally:
    cap.release()
    out_writer.release()
    cv2.destroyAllWindows()
    print(f"🎉 影片已安全儲存！快去資料夾點開看看：{OUTPUT_PATH}")