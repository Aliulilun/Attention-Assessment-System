# -*- coding: utf-8 -*-
"""
分析管線橋接層
===============
把 hurry/main.py 當成函式庫載入，供介面呼叫。

為什麼用 importlib 動態載入而不是 `from hurry import main`？
  1. hurry/ 沒有 __init__.py，不是套件；
  2. 專案根目錄也有一支 main.py，直接 `import main` 兩者名稱會撞在一起，
     先載到哪一支取決於 sys.path 順序，非常難除錯。
用 spec_from_file_location 指定完整路徑並命名為 "hurry_main"，兩支 main.py
可以同時存在於同一個行程中而互不干擾。

介面「不重新實作」任何分析邏輯——所有計分、階段判定、輸出格式都仍然由
hurry/main.py 與 modules/ 負責，確保介面跑出來的數字與命令列完全一致。
"""

import importlib.util
import os
import sys
import threading

UI_DIR      = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(UI_DIR)              # C:\project
HURRY_DIR   = os.path.join(PROJECT_DIR, "hurry")
HURRY_MAIN  = os.path.join(HURRY_DIR, "main.py")
CONFIG_PATH = os.path.join(PROJECT_DIR, "config.yaml")

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")

_module = None
_module_lock = threading.Lock()

# 已載入的重型 GPU 元件（跨影片重複使用，避免每按一次「開始分析」
# 就重載 YOLO / Gaze / EasyOCR，模型冷啟動要 30 秒以上，報告現場等不起）
_components = None
_components_signature = None


def get_hurry_main():
    """載入（或取回已載入的）hurry/main.py 模組。"""
    global _module
    with _module_lock:
        if _module is not None:
            return _module
        if not os.path.exists(HURRY_MAIN):
            raise FileNotFoundError("找不到分析主程式：%s" % HURRY_MAIN)
        if PROJECT_DIR not in sys.path:
            sys.path.insert(0, PROJECT_DIR)
        spec = importlib.util.spec_from_file_location("hurry_main", HURRY_MAIN)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["hurry_main"] = mod
        spec.loader.exec_module(mod)
        _module = mod
        return mod


def default_video_dir():
    return os.path.join(HURRY_DIR, "video")


def default_output_dir():
    return os.path.join(HURRY_DIR, "output")


def list_videos(folder):
    """列出資料夾內的影片（不遞迴），回傳完整路徑列表。"""
    if not os.path.isdir(folder):
        return []
    return sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(VIDEO_EXTS)
    )


def output_paths(video_path, output_dir):
    """回傳該影片對應的 (輸出影片, 輸出報告) 路徑。

    basename 的 .strip() 與 hurry/main.py 一致——影片檔名結尾有空格時
    （"74 .mp4"）Windows 只會正規化最後一段路徑，中間目錄不會，
    不去掉會導致語音快取目錄對不上。
    """
    base = os.path.splitext(os.path.basename(video_path))[0].strip()
    return (os.path.join(output_dir, base + ".mp4"),
            os.path.join(output_dir, base + ".txt"))


def load_config():
    return get_hurry_main().load_project_config(CONFIG_PATH)


def save_config(cfg):
    """把設定寫回 config.yaml（保留 UTF-8 與中文註解外的內容）。

    註：yaml.safe_dump 會丟掉原檔註解，因此只在使用者按下「儲存設定」時
    才寫入，平常的執行只在記憶體中套用覆寫值。
    """
    import yaml
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)


def describe_gpu():
    """回傳 (是否有 CUDA, 描述字串)，給設定頁顯示硬體狀態。"""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            return True, "%s ・ VRAM %.1f GB ・ CUDA 已啟用" % (name, vram)
        return False, "未偵測到 CUDA，將以 CPU 推論（速度顯著較慢）"
    except Exception as e:
        return False, "無法查詢 GPU 狀態：%s" % e


def apply_runtime_settings(settings):
    """把介面上的跳幀 / 階段範圍設定套進 hurry_main 的模組層變數。

    這些值在 process_single_video 內是以 global 讀取的，所以在模組物件上
    直接指派即可生效，不需要改動函式簽名。
    """
    hm = get_hurry_main()
    hm.YOLO_SKIP = max(1, int(settings.get("yolo_skip", hm.YOLO_SKIP)))
    hm.GAZE_SKIP = max(1, int(settings.get("gaze_skip", hm.GAZE_SKIP)))
    hm.OCR_SKIP = max(1, int(settings.get("ocr_skip", hm.OCR_SKIP)))
    hm.SHOW_PREVIEW = False          # 預覽一律走介面，不再另開 OpenCV 視窗
    max_stage = int(settings.get("max_stage", 10))
    hm.ACTIVE_STAGES = set(range(1, max_stage + 1))
    return hm


def _signature(settings, cfg):
    """判斷是否需要重建 GPU 元件的指紋。

    只有會影響模型建構的項目才列入：OCR 跳幀寫在 sign_tracker 上、
    氣球彩度門檻寫在 ModelManager 上、視線裝置與校正寫在 GazePipeline 上。
    YOLO/GAZE 跳幀是每幀讀 global，改了不需要重建。
    """
    gz = (cfg or {}).get("gaze_estimation", {})
    return (
        int(settings.get("ocr_skip", 5)),
        float((cfg or {}).get("object_detection", {}).get("stage5_min_colorful_ratio", 0.12)),
        str(gz.get("model", {}).get("device", "cuda")),
        float(gz.get("calibration", {}).get("pitch_offset_deg", -12.5)),
    )


def get_components(settings, cfg, force_reload=False):
    """取得（必要時才建立）共用的 GPU 元件。"""
    global _components, _components_signature
    hm = get_hurry_main()
    sig = _signature(settings, cfg)
    if _components is not None and sig == _components_signature and not force_reload:
        print(">>> 沿用已載入的模型（設定未變動，略過重新載入）")
        return _components
    if _components is not None:
        print(">>> 設定已變更，重新載入模型…")
        release_components()
    _components = hm.build_shared_components(cfg, ocr_skip=settings.get("ocr_skip"))
    _components_signature = sig
    return _components


def release_components():
    """釋放已載入的模型與 VRAM（關閉程式或切換設定時呼叫）。"""
    global _components, _components_signature
    _components = None
    _components_signature = None
    try:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
