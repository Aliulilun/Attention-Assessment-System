"""
有標籤的手勢資料收集（webcam）
==============================
開一個視窗依序用中文提示動作（每個 6 秒），只存 MediaPipe 手部關鍵點
（2D 像素座標 + 3D world landmarks），不存任何影像。

用途：評估指向判定規則（見 compare_pointing_rules.py 與
docs/指向改進待辦_即時體驗對照.md）。換人錄可以擴充樣本、檢查規則是否只適用單一受試者。

執行（先關掉 UI，鏡頭同時只能給一個程式用）：
    conda activate mediapipe_py39
    python summarize/pointing_lab/collect_hand_gestures.py [輸出檔名.npz]
"""
import os, sys, time
import cv2, numpy as np, mediapipe as mp
from PIL import Image, ImageDraw, ImageFont
from mediapipe.tasks.python import BaseOptions, vision
S = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(os.path.dirname(S))
OUT = sys.argv[1] if len(sys.argv) > 1 else time.strftime("hand_gesture_labeled_%Y%m%d_%H%M.npz")
STEPS = [  # (標籤, 是否應判定為指向, 提示)
    ("ready", None, "準備：把手放下，3 秒後開始"),
    ("point_left", True, "用食指指向『左邊』（其他手指握起來）"),
    ("point_right", True, "用食指指向『右邊』"),
    ("point_up", True, "用食指指向『上方』"),
    ("point_left2", True, "再一次：指向左邊，手可以前後移動"),
    ("point_right2", True, "再一次：指向右邊，手可以前後移動"),
    ("open_palm", False, "手掌張開（五指打開）"),
    ("fist", False, "握拳"),
    ("v_sign", False, "比 YA（食指＋中指）"),
    ("relaxed", False, "手自然放鬆、隨意動"),
    ("gun", False, "比槍（拇指朝上＋食指）——先記錄，之後再決定算不算"),
]
DUR = {"ready": 3}
font = ImageFont.truetype(r"C:\Windows\Fonts\msjh.ttc", 34)
opt = vision.HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=os.path.join(PROJECT, "model", "gaze", "hand_landmarker.task")),
                                   running_mode=vision.RunningMode.VIDEO, num_hands=2)
lmk = vision.HandLandmarker.create_from_options(opt)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
if not cap.read()[0]: sys.exit("camera busy? close the UI first")
rows = []; t0 = time.time(); n = 0
for label, positive, text in STEPS:
    dur = DUR.get(label, 6); ts = time.time()
    while time.time() - ts < dur:
        ok, f = cap.read(); f = cv2.flip(f, 1); h, w = f.shape[:2]; n += 1
        r = lmk.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(f, cv2.COLOR_BGR2RGB)), int((time.time() - t0) * 1000) + n)
        for img_lm, world_lm in zip(r.hand_landmarks, r.hand_world_landmarks):
            P2 = np.array([[p.x * w, p.y * h] for p in img_lm]); P3 = np.array([[p.x, p.y, p.z] for p in world_lm])
            if label != "ready":
                rows.append((label, -1 if positive is None else int(positive), P2, P3))
            for a in P2: cv2.circle(f, tuple(a.astype(int)), 3, (0, 255, 0), -1)
        img = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)); d = ImageDraw.Draw(img)
        d.rectangle((0, 0, w, 60), fill=(0, 0, 0)); d.text((16, 10), "%s　剩 %.0f 秒" % (text, dur - (time.time() - ts)), font=font, fill=(255, 220, 90))
        cv2.imshow("hand data", cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)); cv2.waitKey(1)
cv2.destroyAllWindows(); cap.release(); lmk.close()
np.savez(os.path.join(S, OUT), labels=np.array([r[0] for r in rows]), pos=np.array([r[1] for r in rows]),
         p2=np.array([r[2] for r in rows]), p3=np.array([r[3] for r in rows]))
from collections import Counter
print("saved", len(rows), dict(Counter(r[0] for r in rows)))
