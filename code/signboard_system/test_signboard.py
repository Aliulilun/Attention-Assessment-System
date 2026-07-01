"""
signboard 獨立測試腳本
- 不需要啟動 main.py
- 模擬從 Stage 6 開始，專門測試 Stage 7 模板比對
- 空白鍵暫停/繼續，Q 鍵離開，數字鍵 1-6 手動切換目前階段
"""
import sys
import os
import cv2
from tkinter import Tk, filedialog

# 讓 Python 找到 modules/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modules'))
from signboard import SignboardTracker

BASE_DIR = os.path.dirname(__file__)
VIDEO_DIR = os.path.join(BASE_DIR, 'video')
START_STAGE = 1

def select_video_file():
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    video_path = filedialog.askopenfilename(
        title="選擇要測試的影片",
        initialdir=VIDEO_DIR,
        filetypes=[
            ("Video files", "*.mp4 *.avi *.mov *.mkv"),
            ("MP4 files", "*.mp4"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    return video_path

def main():
    video_path = select_video_file()
    if not video_path:
        print("[INFO] 未選擇影片，程式結束")
        return

    tracker = SignboardTracker(allowlist='1234567')

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] 無法開啟影片：{video_path}")
        return

    print(">> 讀取第一幀以框選 ROI...")
    ret, first_frame = cap.read()
    if not ret:
        print("[ERROR] 影片無法讀取第一幀")
        return

    tracker.initialize_roi(first_frame)
    tracker.current_stage = START_STAGE
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    print(f">> 開始播放，從 Stage {START_STAGE} 開始")
    print("   空白鍵: 暫停/繼續 | Q: 離開 | 1-6: 手動設定目前 Stage")

    paused = False
    frame_idx = 0
    last_detected = None

    cv2.namedWindow("Signboard Test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Signboard Test", 1280, 720)

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print(">> 影片播放完畢")
                break
            frame_idx += 1

            detected = tracker.detect_stage(frame)
            if detected is not None:
                last_detected = detected
                print(f"[Frame {frame_idx:04d}] *** Stage 升級 → {detected} ***")

            display = frame.copy()
            tracker.draw_boxes(display, tracker.current_stage)

            # 畫面 HUD（黑色描邊 + 亮色前景）
            def hud_text(img, text, pos, scale, color):
                cv2.putText(img, text, (pos[0]+2, pos[1]+2),
                            cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
                cv2.putText(img, text, pos,
                            cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)

            hud_text(display, f"Stage: {tracker.current_stage}", (15, 45),  1.3, (0, 200, 255))
            hud_text(display, f"Frame: {frame_idx}",             (15, 90),  1.0, (0, 255, 255))
            hud_text(display, f"Last detected: {last_detected if last_detected is not None else '---'}", (15, 135), 1.1, (0, 255, 100))

            cv2.imshow("Signboard Test", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord(' '):
            paused = not paused
            print(f">> {'暫停' if paused else '繼續'}")
        elif ord('1') <= key <= ord('6'):
            tracker.current_stage = key - ord('0')
            print(f">> 手動切換 Stage → {tracker.current_stage}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
