from ultralytics import YOLO
from pathlib import Path
from tkinter import Tk, filedialog
import cv2
import torch
import time
import numpy as np
import gc

BASE_DIR = Path(r"C:\Users\wayne\Desktop\project\model_test")
INPUT_DIR = BASE_DIR / "input"
MODEL_DIR = BASE_DIR / "model"   # 模型請放在 model_test/model

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

# 模式 2 的模型切換順序
ORDERED_MODEL_STEMS = [
    "front_model",
    "background_model",
    "balloon_model",
    "doll_model",
    "toy_model",
]

# 推論參數
CONF = 0.70
IMGSZ = 960
PREVIEW_SCALE = 0.75

KEY_ESC = 27
KEY_SPACE = 32


def ensure_dirs():
    """
    只建立 input 和 model 資料夾。
    這版不建立 output，也不會輸出任何檔案。
    """
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def select_run_mode():
    """
    終端機選單：
    1. 選擇單一模型，再選擇圖片/影片
    2. 依指定順序使用 model_test/model 裡的模型，空白鍵切換下一個模型
    """
    print("請選擇執行模式：")
    print("1. 選擇單一模型，再選擇圖片或影片")
    print("2. 依序使用 front/background/balloon/doll/toy 模型，空白鍵切換下一個")
    print()

    while True:
        choice = input("請輸入 1 或 2：").strip()
        if choice in {"1", "2"}:
            return choice
        print("輸入錯誤，請重新輸入 1 或 2。")


def select_model():
    root = Tk()
    root.withdraw()

    model_path = filedialog.askopenfilename(
        title="選擇要使用的 YOLO 模型",
        initialdir=str(MODEL_DIR),
        filetypes=[("YOLO model", "*.pt")]
    )

    root.destroy()
    return Path(model_path) if model_path else None


def select_input_file():
    root = Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        title="選擇要偵測的圖片或影片",
        initialdir=str(INPUT_DIR),
        filetypes=[
            ("Image or Video", "*.jpg *.jpeg *.png *.bmp *.webp *.mp4 *.avi *.mov *.mkv"),
            ("Images", "*.jpg *.jpeg *.png *.bmp *.webp"),
            ("Videos", "*.mp4 *.avi *.mov *.mkv"),
        ]
    )

    root.destroy()
    return Path(file_path) if file_path else None


def find_model_by_stem(target_stem: str):
    """
    依模型名稱尋找 .pt 檔。

    建議檔名：
    - front_model.pt
    - background_model.pt
    - balloon_model.pt
    - doll_model.pt
    - toy_model.pt

    如果找不到完全同名，也會接受 front_model_best.pt 這類以指定名稱開頭的檔名。
    """
    target = target_stem.lower()
    model_files = sorted([p for p in MODEL_DIR.glob("*.pt") if p.is_file()])

    exact_matches = [p for p in model_files if p.stem.lower() == target]
    if exact_matches:
        return exact_matches[0]

    prefix_matches = [p for p in model_files if p.stem.lower().startswith(target)]
    if prefix_matches:
        return prefix_matches[0]

    return None


def get_ordered_model_paths():
    """
    依指定順序取得模型：
    front_model -> background_model -> balloon_model -> doll_model -> toy_model
    """
    ordered_paths = []
    missing_models = []

    for stem in ORDERED_MODEL_STEMS:
        model_path = find_model_by_stem(stem)
        if model_path is None:
            missing_models.append(stem)
        else:
            ordered_paths.append(model_path)

    return ordered_paths, missing_models


def print_ordered_model_list(model_paths, missing_models):
    print("模式 2 模型順序：")
    for idx, stem in enumerate(ORDERED_MODEL_STEMS, start=1):
        matched = next((p for p in model_paths if p.stem.lower() == stem.lower() or p.stem.lower().startswith(stem.lower())), None)
        if matched:
            print(f"{idx}. {stem} -> {matched.name}")
        else:
            print(f"{idx}. {stem} -> 找不到")

    if missing_models:
        print()
        print("找不到以下模型，會略過：")
        for stem in missing_models:
            print(f"- {stem}.pt")
    print()


def read_image_unicode(path: Path):
    """
    支援中文路徑的圖片讀取。
    """
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


def resize_preview(frame, scale=PREVIEW_SCALE):
    if scale == 1:
        return frame

    h, w = frame.shape[:2]
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(frame, (new_w, new_h))


def draw_fps(frame, fps):
    text = f"FPS: {fps:.1f}"
    cv2.putText(
        frame,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )
    return frame


def draw_model_name(frame, model_name: str):
    text = f"Model: {model_name}"
    cv2.putText(
        frame,
        text,
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )
    return frame


def draw_control_hint(frame, hint: str):
    h, _ = frame.shape[:2]
    cv2.putText(
        frame,
        hint,
        (20, h - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )
    return frame


def load_model(model_path: Path):
    print(f"載入模型：{model_path}")

    model = YOLO(str(model_path))

    if torch.cuda.is_available():
        print(f"使用 GPU：{torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
    else:
        print("沒有偵測到 CUDA，改用 CPU")

    print("模型類別：", model.names)
    return model


def release_model(model):
    """
    批次跑多個模型時，釋放上一個模型，避免 GPU 記憶體累積。
    """
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def predict_image_preview(model, input_path: Path, model_name: str):
    print(f"開始偵測圖片：{input_path}")
    print("這版只顯示預覽，不會輸出圖片。")

    image = read_image_unicode(input_path)

    if image is None:
        print("圖片讀取失敗")
        return

    use_cuda = torch.cuda.is_available()

    results = model.predict(
        source=image,
        conf=CONF,
        imgsz=IMGSZ,
        device=0 if use_cuda else "cpu",
        half=use_cuda,
        verbose=False
    )

    annotated = results[0].plot()
    annotated = draw_model_name(annotated, model_name)

    preview = resize_preview(annotated)
    window_name = f"YOLO Preview - Image - {model_name}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, preview)

    print("圖片偵測完成")
    print("按任意鍵關閉目前預覽視窗")
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)


def predict_video_preview(model, input_path: Path, model_name: str):
    print(f"開始偵測影片：{input_path}")
    print("這版只顯示預覽，不會輸出影片。")

    cap = cv2.VideoCapture(str(input_path))

    if not cap.isOpened():
        print("影片開啟失敗")
        return

    use_cuda = torch.cuda.is_available()

    window_name = f"YOLO Preview - Video - {model_name}, press Q to stop"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_count = 0
    start_time = time.time()
    prev_time = time.time()

    print("處理中，預覽視窗按 Q 可停止目前這個模型")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        results = model.predict(
            source=frame,
            conf=CONF,
            imgsz=IMGSZ,
            device=0 if use_cuda else "cpu",
            half=use_cuda,
            verbose=False
        )

        annotated = results[0].plot()

        now = time.time()
        instant_fps = 1 / (now - prev_time) if now > prev_time else 0
        prev_time = now

        annotated = draw_fps(annotated, instant_fps)
        annotated = draw_model_name(annotated, model_name)

        preview = resize_preview(annotated)
        cv2.imshow(window_name, preview)

        frame_count += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == ord("Q") or key == KEY_ESC:
            print("已手動停止目前這個模型")
            break

    cap.release()
    cv2.destroyWindow(window_name)

    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0

    print("影片預覽完成")
    print(f"處理幀數：{frame_count}")
    print(f"平均 FPS：{avg_fps:.2f}")


def run_prediction_preview(model, input_path: Path, model_name: str):
    suffix = input_path.suffix.lower()

    if suffix in IMAGE_EXTS:
        predict_image_preview(model, input_path, model_name)

    elif suffix in VIDEO_EXTS:
        predict_video_preview(model, input_path, model_name)

    else:
        print(f"不支援的檔案格式：{suffix}")


def run_single_model_mode():
    model_path = select_model()

    if model_path is None:
        print("未選擇模型，程式結束")
        return

    input_path = select_input_file()

    if input_path is None:
        print("未選擇輸入檔案，程式結束")
        return

    model = load_model(model_path)

    try:
        run_prediction_preview(model, input_path, model_path.name)
    finally:
        release_model(model)


def switch_to_next_model(current_model, model_paths, next_idx):
    """
    模式 2 用：釋放目前模型並載入下一個模型。
    next_idx 是下一個模型在 model_paths 裡的 index。
    """
    release_model(current_model)

    if next_idx >= len(model_paths):
        return None

    next_model_path = model_paths[next_idx]
    print("=" * 60)
    print(f"切換到下一個模型：{next_model_path.name}")
    next_model = load_model(next_model_path)
    return next_model


def run_ordered_models_image_switch_mode(input_path: Path, model_paths):
    image = read_image_unicode(input_path)

    if image is None:
        print("圖片讀取失敗")
        return

    use_cuda = torch.cuda.is_available()
    current_idx = 0
    current_model = load_model(model_paths[current_idx])
    window_name = "YOLO Preview - Ordered Models - Image"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("圖片模式：按空白鍵切換下一個模型，按 Q 或 ESC 結束。")

    try:
        while True:
            model_path = model_paths[current_idx]
            model_name = model_path.name

            results = current_model.predict(
                source=image,
                conf=CONF,
                imgsz=IMGSZ,
                device=0 if use_cuda else "cpu",
                half=use_cuda,
                verbose=False
            )

            annotated = results[0].plot()
            annotated = draw_model_name(annotated, model_name)
            annotated = draw_control_hint(annotated, "SPACE: next model | Q/ESC: quit")

            preview = resize_preview(annotated)
            cv2.imshow(window_name, preview)

            key = cv2.waitKey(0) & 0xFF

            if key == KEY_SPACE:
                if current_idx + 1 >= len(model_paths):
                    print("已經是最後一個模型，結束模式 2。")
                    break

                current_idx += 1
                next_model = switch_to_next_model(current_model, model_paths, current_idx)
                if next_model is None:
                    break
                current_model = next_model

            elif key == ord("q") or key == ord("Q") or key == KEY_ESC:
                print("已手動結束模式 2")
                break

            else:
                print("請按空白鍵切換下一個模型，或按 Q / ESC 結束。")

    finally:
        release_model(current_model)
        cv2.destroyWindow(window_name)


def run_ordered_models_video_switch_mode(input_path: Path, model_paths):
    cap = cv2.VideoCapture(str(input_path))

    if not cap.isOpened():
        print("影片開啟失敗")
        return

    use_cuda = torch.cuda.is_available()
    current_idx = 0
    current_model = load_model(model_paths[current_idx])
    window_name = "YOLO Preview - Ordered Models - Video"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_count = 0
    start_time = time.time()
    prev_time = time.time()

    print("影片模式：按空白鍵切換下一個模型，按 Q 或 ESC 結束。")
    print("模型順序：front_model -> background_model -> balloon_model -> doll_model -> toy_model")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("影片播放結束")
                break

            model_path = model_paths[current_idx]
            model_name = model_path.name

            results = current_model.predict(
                source=frame,
                conf=CONF,
                imgsz=IMGSZ,
                device=0 if use_cuda else "cpu",
                half=use_cuda,
                verbose=False
            )

            annotated = results[0].plot()

            now = time.time()
            instant_fps = 1 / (now - prev_time) if now > prev_time else 0
            prev_time = now

            annotated = draw_fps(annotated, instant_fps)
            annotated = draw_model_name(annotated, model_name)
            annotated = draw_control_hint(annotated, "SPACE: next model | Q/ESC: quit")

            preview = resize_preview(annotated)
            cv2.imshow(window_name, preview)
            frame_count += 1

            key = cv2.waitKey(1) & 0xFF

            if key == KEY_SPACE:
                if current_idx + 1 >= len(model_paths):
                    print("已經是最後一個模型，結束模式 2。")
                    break

                current_idx += 1
                next_model = switch_to_next_model(current_model, model_paths, current_idx)
                if next_model is None:
                    break
                current_model = next_model
                prev_time = time.time()

            elif key == ord("q") or key == ord("Q") or key == KEY_ESC:
                print("已手動結束模式 2")
                break

    finally:
        cap.release()
        release_model(current_model)
        cv2.destroyWindow(window_name)

    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0

    print("模式 2 預覽完成")
    print(f"處理幀數：{frame_count}")
    print(f"平均 FPS：{avg_fps:.2f}")


def run_all_models_mode():
    model_paths, missing_models = get_ordered_model_paths()

    print_ordered_model_list(model_paths, missing_models)

    if not model_paths:
        print(f"在模型資料夾找不到指定模型：{MODEL_DIR}")
        print("請先把以下模型放進 model_test/model：")
        for stem in ORDERED_MODEL_STEMS:
            print(f"- {stem}.pt")
        return

    input_path = select_input_file()

    if input_path is None:
        print("未選擇輸入檔案，程式結束")
        return

    suffix = input_path.suffix.lower()

    if suffix in IMAGE_EXTS:
        run_ordered_models_image_switch_mode(input_path, model_paths)
    elif suffix in VIDEO_EXTS:
        run_ordered_models_video_switch_mode(input_path, model_paths)
    else:
        print(f"不支援的檔案格式：{suffix}")


def main():
    ensure_dirs()

    print("=== YOLO Model Test - Preview Only ===")
    print(f"Input folder : {INPUT_DIR}")
    print(f"Model folder : {MODEL_DIR}")
    print("Output folder: 不使用，這版不會輸出任何檔案")
    print()

    mode = select_run_mode()

    if mode == "1":
        run_single_model_mode()
    elif mode == "2":
        run_all_models_mode()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
