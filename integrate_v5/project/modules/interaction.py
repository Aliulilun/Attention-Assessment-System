import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
from collections import deque
from ultralytics import YOLO
from typing import Optional, List, Dict, Tuple, Any

class LegacyHandLms:
    def __init__(self, lms: Any):
        self.landmark = lms

class InteractionEngine:
    WRIST = 0
    THB_TIP = 4   # 🌟 新增：大拇指指尖（收攏檢查用）
    IDX_MCP = 5
    IDX_PIP = 6
    IDX_TIP = 8
    MID_MCP = 9
    MID_PIP = 10
    MID_TIP = 12
    RNG_MCP = 13
    RNG_PIP = 14
    RNG_TIP = 16
    PNK_MCP = 17
    PNK_PIP = 18
    PNK_TIP = 20

    def __init__(
        self,
        pose_model_path: Optional[str] = None,
        hand_model_path: Optional[str] = None,  # 🌟 新增：明確傳入 hand_landmarker.task 路徑，避免 __file__ 解析錯誤
        sma_window: int = 5,
        dwell_frames: int = 3,
        divider_ratio: float = 0.35,
        hold_frames: int = 12
    ):
        self.DWELL_FRAMES: int = dwell_frames
        self.divider_ratio: float = divider_ratio
        self.HOLD_FRAMES: int = hold_frames

        print(">>> [InteractionEngine] 載入人類骨架模型 (YOLO-Pose)...")
        if pose_model_path and os.path.exists(pose_model_path):
            self.model_human = YOLO(pose_model_path)
        else:
            self.model_human = YOLO('yolo11n-pose.pt')

        print(">>> [InteractionEngine] 載入 MediaPipe Hand Landmarker (Tasks API VIDEO MODE)...")
        # 🌟 修改：優先使用明確傳入的 hand_model_path；
        #          退回 __file__ 推算（Windows junction/symlink 下可能解析錯誤）
        if hand_model_path and os.path.exists(hand_model_path):
            model_path = hand_model_path
        else:
            # 🌟 修改：hand_landmarker.task 實際放在 model/gaze/ 子目錄下
            model_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'model', 'gaze', 'hand_landmarker.task'
            )
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ 找不到 MediaPipe 模型檔案: {model_path}")

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            # 🌟 修改 1：num_hands 4 -> 2
            # 場景中最多 2 人，每人最多 1 隻手即可。
            # 原本 4 隻讓 MediaPipe 拼命去找第 3、4 隻手，
            # 導致把臉、衣服紋路都誤判為手。
            num_hands=2,
            running_mode=vision.RunningMode.VIDEO,
            # 🌟 修改 2：三個信心閾值從 0.1 提高至 0.35
            # 0.1 太寬鬆，人臉輪廓（鼻樑 + 臉頰）的置信度常介於 0.1~0.3，
            # 提高至 0.35 後可有效過濾臉部誤判，同時不影響真實手部偵測。
            min_hand_detection_confidence=0.35,
            min_hand_presence_confidence=0.35,
            min_tracking_confidence=0.30
        )
        self.mp_hands = vision.HandLandmarker.create_from_options(options)
        self.timestamp_ms: int = 0
        # 🌟 新增：本幀「小朋友指向射線是否存在」（不看有沒有指中物品）
        # Stage 8（怪聲）的指向計分只需要射線存在即可，由 main 讀取此屬性
        self.last_child_pointing_active: bool = False
        self._ts_base: int = 0  # 🌟 新增：批次模式跨影片 elapsed_ms 基準偏移

        self.C_CHILD: Tuple[int, int, int] = (0, 255, 0)
        self.C_TESTER: Tuple[int, int, int] = (0, 165, 255)

        self.dwell_counters: Dict[str, int] = {"Child": 0, "Tester": 0}
        self.hold_counters: Dict[str, int] = {"Child": 0, "Tester": 0}

        self.ray_history: Dict[str, Dict[str, deque]] = {
            "Child": {"origin": deque(maxlen=sma_window), "vector": deque(maxlen=sma_window)},
            "Tester": {"origin": deque(maxlen=sma_window), "vector": deque(maxlen=sma_window)}
        }
        self.track_owner_votes: Dict[int, deque] = {}
        self.owner_vote_window: int = max(5, sma_window * 2)
        self.ray_angle_history: Dict[str, deque] = {
            "Child": deque(maxlen=sma_window),
            "Tester": deque(maxlen=sma_window)
        }

        # ============================================================
        # 🌟 新增（問題1：邊界處歸屬閃爍）：手部歸屬追蹤器
        # 以「手腕位置」跨幀匹配同一隻手，對逐幀的 raw owner 做多數投票，
        # 並加入遲滯：必須連續多幀壓倒性的相反證據才允許換主人，
        # 避免手在 Tester/Child 分界線附近時綠橘來回跳。
        # 每個 track: {"pos": (x,y), "owner": str, "votes": deque, "age": int}
        # ============================================================
        self.hand_owner_tracks: List[Dict[str, Any]] = []
        self.HAND_TRACK_DIST_RATIO: float = 0.10   # 匹配半徑（佔幀寬比例）
        self.HAND_OWNER_VOTE_WIN: int = 11         # 投票視窗（幀）
        self.HAND_OWNER_SWITCH_RATIO: float = 0.7  # 相反票數佔比 ≥ 0.7 才換主人
        self.HAND_TRACK_MAX_AGE: int = 15          # 連續幾幀沒匹配到就刪除 track

    @staticmethod
    def get_angle(v1: np.ndarray, v2: np.ndarray) -> float:
        unit_v1 = v1 / (np.linalg.norm(v1) + 1e-9)
        unit_v2 = v2 / (np.linalg.norm(v2) + 1e-9)
        dot_product = np.clip(np.dot(unit_v1, unit_v2), -1.0, 1.0)
        return float(np.degrees(np.arccos(dot_product)))

    @staticmethod
    def _hand_scale(wri: np.ndarray, mid_mcp: np.ndarray) -> float:
        scale = float(np.linalg.norm(mid_mcp - wri))
        return max(scale, 1e-6)

    @staticmethod
    def draw_25d_laser(frame: np.ndarray, start_pt: Tuple[int, int], end_pt: Tuple[int, int], color: Tuple[int, int, int]) -> None:
        overlay = frame.copy()
        cv2.line(overlay, start_pt, end_pt, color, 16, lineType=cv2.LINE_AA)
        cv2.line(overlay, start_pt, end_pt, (255, 255, 255), 6, lineType=cv2.LINE_AA)
        alpha = 0.5
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.line(frame, start_pt, end_pt, (255, 255, 255), 2, lineType=cv2.LINE_AA)
        cv2.circle(frame, start_pt, 6, (255, 255, 255), -1)
        cv2.circle(frame, start_pt, 9, color, 2)

    def is_valid_pointing(self, landmarks: LegacyHandLms, frame_w: int, frame_h: int, owner: str) -> Optional[float]:
        def get_pt(idx: int) -> np.ndarray:
            return np.array([landmarks.landmark[idx].x * frame_w, landmarks.landmark[idx].y * frame_h])

        wri = get_pt(self.WRIST)
        idx_mcp = get_pt(self.IDX_MCP)
        idx_pip = get_pt(self.IDX_PIP)
        idx_tip = get_pt(self.IDX_TIP)
        mid_mcp = get_pt(self.MID_MCP)
        mid_pip = get_pt(self.MID_PIP)
        mid_tip = get_pt(self.MID_TIP)
        rng_mcp = get_pt(self.RNG_MCP)
        rng_pip = get_pt(self.RNG_PIP)
        rng_tip = get_pt(self.RNG_TIP)
        pnk_mcp = get_pt(self.PNK_MCP)
        pnk_pip = get_pt(self.PNK_PIP)
        pnk_tip = get_pt(self.PNK_TIP)

        margin = 0.01
        for node in [self.WRIST, self.IDX_MCP, self.IDX_PIP, self.IDX_TIP, self.MID_PIP, self.MID_TIP, self.RNG_PIP, self.RNG_TIP]:
            lm = landmarks.landmark[node]
            if not (margin <= lm.x <= 1-margin and margin <= lm.y <= 1-margin):
                return None

        dist = lambda p1, p2: float(np.linalg.norm(p1 - p2))
        scale = self._hand_scale(wri, mid_mcp)

        if dist(idx_tip, idx_mcp) <= dist(idx_pip, idx_mcp) + 0.04 * scale: return None
        if dist(idx_tip, wri) < 0.17 * scale: return None

        # ============================================================
        # 🌟 新增（問題4：射線連錯手指）：食指尖必須是「離手腕最遠」的指尖
        # 若中指/無名指/小指尖比食指尖更遠，代表：
        #   (a) 這不是標準指向手勢（伸出的是別根手指），或
        #   (b) MediaPipe landmark 標錯位（"食指" landmark 實際落在中指上）
        # 兩種情況都不該形成射線，直接拒絕，保證射線永遠是手腕→食指。
        # ============================================================
        d_idx_wri = dist(idx_tip, wri)
        if d_idx_wri <= dist(mid_tip, wri): return None
        if d_idx_wri <= dist(rng_tip, wri): return None
        if d_idx_wri <= dist(pnk_tip, wri): return None

        # ============================================================
        # 🌟 新增（問題3：手放桌上誤生成射線）：兩種角色一律要求
        # 中指與無名指「彎曲」。手平放桌面時五指全部伸直，
        # 中指/無名指尖會遠超過其 PIP，此檢查會直接拒絕。
        # 真正的指向手勢（食指伸、其餘捲起）不受影響。
        # ============================================================
        is_mid_folded = dist(mid_tip, wri) < dist(mid_pip, wri) + 0.10 * scale
        is_rng_folded = dist(rng_tip, wri) < dist(rng_pip, wri) + 0.10 * scale
        if not (is_mid_folded and is_rng_folded): return None

        # ============================================================
        # 🌟 新增：標準指向手勢的完整四指檢查
        # 有效指向 = 只有食指伸出、「其他四指全部縮起」：
        # 1. 小指也必須彎曲（原本只檢查中指與無名指）
        # 2. 大拇指必須收攏貼著拳頭——拇指尖需靠近食指或中指的
        #    掌指關節（距離 <= 0.6 倍手掌長）。比出「讚」或 L 形
        #    手勢時拇指外張，距離會遠超此值 → 不形成射線。
        # ============================================================
        is_pnk_folded = dist(pnk_tip, wri) < dist(pnk_pip, wri) + 0.10 * scale
        if not is_pnk_folded: return None

        thb_tip = get_pt(self.THB_TIP)
        if min(dist(thb_tip, idx_mcp), dist(thb_tip, mid_mcp)) > 0.60 * scale:
            return None  # 拇指外張：非標準指向手勢

        if owner == "Tester":
            dist_idx = dist(idx_tip, idx_mcp)
            if dist_idx <= dist(mid_tip, mid_mcp) or dist_idx <= dist(rng_tip, rng_mcp) or dist_idx <= dist(pnk_tip, pnk_mcp):
                return None
            if dist(idx_tip, wri) < dist(mid_tip, wri) + 0.13 * scale: return None
            if (idx_tip[0] - wri[0]) < 0.07 * scale: return None
            # 🌟 新增（問題3）：食指方向須與手腕→食指根方向大致一致（同兒童標準、稍寬鬆）
            if self.get_angle(idx_mcp - wri, idx_tip - idx_mcp) > 50: return None
        else:
            if dist(idx_tip, wri) <= dist(mid_tip, wri) + 0.07 * scale: return None
            if self.get_angle(idx_mcp - wri, idx_tip - idx_mcp) > 40: return None

        idx_angle = self.get_angle(idx_pip - idx_mcp, idx_tip - idx_pip)
        # 🌟 新增（問題3）：食指本身必須打直（PIP 彎曲角 ≤ 45°）。
        # 手鬆放桌上、食指微彎垂下時角度會超標，不形成射線。
        if idx_angle > 45: return None
        return idx_angle

    @staticmethod
    def calculate_arm_link_score(mp_wrist: Tuple[int, int], kpts: np.ndarray, confs: np.ndarray, side: str = "right", scale: float = 1.0) -> float:
        s_idx, e_idx, w_idx = (6, 8, 10) if side == "right" else (5, 7, 9)
        scale = max(float(scale), 1e-6)
        if confs[w_idx] > 0.4: return float(np.linalg.norm(np.array(mp_wrist) - np.array(kpts[w_idx]))) / scale
        if confs[e_idx] > 0.4 and confs[s_idx] > 0.4:
            S, E, W_mp = np.array(kpts[s_idx]), np.array(kpts[e_idx]), np.array(mp_wrist)
            arm_vec = E - S
            arm_len = np.linalg.norm(arm_vec)
            if arm_len == 0: return float('inf')
            unit_arm = arm_vec / arm_len
            elbow_to_mp = W_mp - E
            proj_len = np.dot(elbow_to_mp, unit_arm)
            if proj_len < -0.25 * scale: return float('inf')
            dist_to_line = np.abs(np.cross(unit_arm, elbow_to_mp))
            return float((dist_to_line + (proj_len * 0.1)) / scale)
        if confs[e_idx] > 0.4: return float(np.linalg.norm(np.array(mp_wrist) - np.array(kpts[e_idx])) / scale + 0.4)
        if confs[s_idx] > 0.4: return float(np.linalg.norm(np.array(mp_wrist) - np.array(kpts[s_idx])) / scale + 0.8)
        return float('inf')

    @staticmethod
    def estimate_body_scale(kpts: np.ndarray, confs: np.ndarray, box: np.ndarray) -> float:
        if confs[5] > 0.4 and confs[6] > 0.4:
            shoulder_width = float(np.linalg.norm(kpts[5] - kpts[6]))
            if shoulder_width > 1e-3:
                return shoulder_width
        box_w = float(box[2] - box[0])
        box_h = float(box[3] - box[1])
        return max(float(np.hypot(box_w, box_h)) * 0.3, 1.0)

    @staticmethod
    def ray_intersects_box(origin: Tuple[float, float], dir_vec: np.ndarray, box: Tuple[float, float, float, float]) -> bool:
        ox, oy = origin; dx, dy = dir_vec; x1, y1, x2, y2 = box
        dx = dx if dx != 0 else 1e-9; dy = dy if dy != 0 else 1e-9
        tx1, tx2 = (x1 - ox) / dx, (x2 - ox) / dx
        ty1, ty2 = (y1 - oy) / dy, (y2 - oy) / dy
        tmin = max(min(tx1, tx2), min(ty1, ty2)); tmax = min(max(tx1, tx2), max(ty1, ty2))
        return bool(tmax >= max(0, tmin))

    def draw_dashed_rectangle(self, img: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int], color: Tuple[int, int, int], thickness: int = 1) -> None:
        x1, y1 = pt1; x2, y2 = pt2; length = 15
        lines = [((x1, y1), (x1+length, y1)), ((x1, y1), (x1, y1+length)),
                 ((x2, y1), (x2-length, y1)), ((x2, y1), (x2, y1+length)),
                 ((x1, y2), (x1+length, y2)), ((x1, y2), (x1, y2-length)),
                 ((x2, y2), (x2-length, y2)), ((x2, y2), (x2, y2-length))]
        for pt_a, pt_b in lines: cv2.line(img, pt_a, pt_b, color, thickness)

    @staticmethod
    def _robust_average_vector(vectors: List[np.ndarray], max_deviation_deg: float = 25.0) -> np.ndarray:
        if len(vectors) == 1:
            return vectors[0]
        angles_x = []
        for v in vectors:
            norm = np.linalg.norm(v)
            if norm > 1e-6:
                angles_x.append(math.degrees(math.atan2(v[1], v[0])))
        if not angles_x:
            return np.mean(vectors, axis=0)
        median_angle = float(np.median(angles_x))
        kept = []
        for v in vectors:
            norm = np.linalg.norm(v)
            if norm <= 1e-6:
                continue
            ang = math.degrees(math.atan2(v[1], v[0]))
            diff = abs(((ang - median_angle) + 180) % 360 - 180)
            if diff <= max_deviation_deg:
                kept.append(v)
        if not kept:
            kept = vectors
        return np.mean(kept, axis=0)

    # ============================================================
    # 🌟 新增：臉部關鍵點排斥半徑（像素）
    # COCO 關鍵點 0=鼻子, 1=左眼, 2=右眼, 3=左耳, 4=右耳
    # MediaPipe 偵測到的「手腕」若落在此半徑內，視為臉部誤判，跳過。
    # ============================================================
    FACE_KPT_INDICES = [0, 1, 2, 3, 4]
    FACE_REJECT_RADIUS = 90  # 像素（地板值；近距離/高解析度時改用 body_scale 放大）
    # 🌟 新增：臉部排斥半徑的 body_scale 係數
    # body_scale ≈ 肩寬；臉部關鍵點大致落在 ~0.6 倍肩寬內，故以此正規化排斥半徑，
    # 避免固定 90px 在近距離/高解析度下太小、或誤殺靠近自己臉部的真實指向手。
    FACE_REJECT_SCALE_K = 0.6

    # 🌟 修改：手部幾何驗證常數
    # 手腕到任意手指關鍵點最大距離（佔幀寬比例）
    # 原本 0.25（480px@1920）太嚴：近距離手腕到中指尖距離可達 500-600px → 合法手被拒
    # 改為 0.55（1056px@1920）：跨人汙染 span > 65% 仍被 YOLO 跨區可見性檢查攔截
    MAX_HAND_SPAN_RATIO = 0.55
    # 食指尖到手腕最小距離（佔幀寬比例）
    # 低於此距離代表手指被遮擋、只偵測到手腕，不顯示也不射線
    MIN_FINGER_EXTEND_RATIO = 0.030

    # 🌟 新增：單一施測者場景的「跨區連動門檻」
    # 只偵測到施測者、手腕落在兒童區時用來區分：
    #  min_score <= 門檻 → 手與施測者手臂強連動 → 施測者跨區伸手指物（維持 Tester）
    #  min_score >  門檻 → 連動弱 → 可能是身體漏偵測的兒童手（才歸 Child）
    # arm link score 已用 body_scale 正規化（同一手腕約 0.1~0.5），1.0 留有安全邊際。可調。
    CROSS_ZONE_LINK_TH = 1.0

    # 🌟 新增：arm link 歸屬的可信門檻
    # min_score 超過此值代表這隻手跟「贏家」的手臂連動其實很弱，
    # 歸屬結果不可信，改走 fallback（身體框包含判定）。
    # 原本的防呆門檻 1.5 太寬：施測者跨區、手臂被遮擋時，
    # 兒童以 1.2~1.4 的爛分數「贏了」也照樣被採信 → 假 Child HIT。
    ARM_LINK_ACCEPT_TH = 1.0
    # 🌟 新增：fallback 身體框外擴比例（佔框長邊）
    FALLBACK_BOX_EXPAND = 0.15
    # 🌟 新增：曖昧情況偏向施測者的分差門檻
    # 兩人的手臂連動分數接近時（Child 沒有贏過 Tester 這個差距以上），
    # 一律判給施測者。理由：施測者伸手進兒童區指物時，小朋友常緊鄰
    # 手旁，YOLO 對兒童手腕的估計會飄到施測者手附近、以些微分差搶贏
    # → 產生假的 Child 射線。臨床上「假的小朋友指向」比「漏掉一幀」
    # 更糟，故平手或險勝時偏向 Tester。
    TESTER_PRIORITY_MARGIN = 0.25
    # 🌟 新增：Child 認領的正向證據門檻
    # 兩人都在場時，一隻手要被判給 Child，Child 自己的手臂連動分數
    # 必須 <= 此值（有真實手臂證據）。防止施測者高舉指物的手落在
    # 兒童身體框內時，被 fallback 的「框包含」邏輯誤判給 Child。
    CHILD_CLAIM_TH = 1.0

    def _is_near_face(self, wrist_px: Tuple[int, int], yolo_people: List[dict]) -> bool:
        """
        檢查 MediaPipe 偵測到的「手腕」是否落在任何人的臉部關鍵點附近。
        若是，代表這是臉部誤判（人臉被誤認為手掌），應跳過。
        """
        wx, wy = float(wrist_px[0]), float(wrist_px[1])
        for person in yolo_people:
            # 🌟 修改：排斥半徑改用該人 body_scale 正規化（取地板值與比例放大的較大者）
            reject_radius = max(self.FACE_REJECT_RADIUS,
                                float(person.get("scale", 0.0)) * self.FACE_REJECT_SCALE_K)
            for kpt_idx in self.FACE_KPT_INDICES:
                if person["confs"][kpt_idx] > 0.35:
                    kx, ky = float(person["kpts"][kpt_idx][0]), float(person["kpts"][kpt_idx][1])
                    if math.hypot(wx - kx, wy - ky) < reject_radius:
                        return True
        return False

    def _smooth_hand_owner(self, wrist_px: Tuple[int, int], raw_owner: str, frame_w: int) -> str:
        """
        🌟 新增（問題1）：手部歸屬時間平滑 + 遲滯。
        以手腕位置找最近的既有 track（半徑 = 幀寬 * HAND_TRACK_DIST_RATIO），
        把本幀的 raw_owner 投進該 track 的投票視窗；
        只有當「相反 owner」票數佔比 >= HAND_OWNER_SWITCH_RATIO 時才切換，
        否則維持原本 owner，徹底消除邊界處綠/橘閃爍。
        """
        wx, wy = float(wrist_px[0]), float(wrist_px[1])
        match_radius = frame_w * self.HAND_TRACK_DIST_RATIO

        best_track, best_dist = None, float('inf')
        for track in self.hand_owner_tracks:
            # 🌟 同一幀內一個 track 只能被一隻手認領，
            # 避免邊界處兩隻手（施測者+兒童）距離近時互相污染投票
            if track.get("used_this_frame", False):
                continue
            d = math.hypot(wx - track["pos"][0], wy - track["pos"][1])
            if d < best_dist:
                best_dist, best_track = d, track

        if best_track is not None and best_dist <= match_radius:
            best_track["pos"] = (wx, wy)
            best_track["age"] = 0
            best_track["used_this_frame"] = True
            best_track["votes"].append(raw_owner)
            votes = best_track["votes"]
            opposite = "Child" if best_track["owner"] == "Tester" else "Tester"
            opp_ratio = sum(1 for v in votes if v == opposite) / len(votes)
            # 遲滯：相反證據要夠壓倒性（且視窗已累積夠多幀）才換主人
            if len(votes) >= 5 and opp_ratio >= self.HAND_OWNER_SWITCH_RATIO:
                old_owner = best_track["owner"]
                best_track["owner"] = opposite
                best_track["votes"] = deque(votes, maxlen=self.HAND_OWNER_VOTE_WIN)
                # 🌟 新增：歸屬翻轉時，立即清除舊主人的殘留射線
                # 此手先前以錯誤身分貢獻的射線方向已不可信；
                # 不清除的話，防閃爍的 hold 緩衝會讓錯色射線再拖 12 幀
                # （畫面上就是「標籤已改對、射線顏色還是錯的」殘影）。
                self.ray_history[old_owner]["origin"].clear()
                self.ray_history[old_owner]["vector"].clear()
                self.ray_angle_history[old_owner].clear()
                self.dwell_counters[old_owner] = 0
                self.hold_counters[old_owner] = 0
            return best_track["owner"]

        # 沒有匹配到 → 新的手，建立 track，首幀直接用 raw_owner
        new_track = {
            "pos": (wx, wy),
            "owner": raw_owner,
            "votes": deque([raw_owner], maxlen=self.HAND_OWNER_VOTE_WIN),
            "age": 0,
            "used_this_frame": True
        }
        self.hand_owner_tracks.append(new_track)
        return raw_owner

    def _age_hand_tracks(self) -> None:
        """🌟 新增（問題1）：每幀開始時老化 track 並重置本幀認領旗標，太久沒匹配到的刪除。"""
        for track in self.hand_owner_tracks:
            track["age"] += 1
            track["used_this_frame"] = False
        self.hand_owner_tracks = [t for t in self.hand_owner_tracks if t["age"] <= self.HAND_TRACK_MAX_AGE]

    def _validate_hand_geometry(self, hand_landmarks_list, FRAME_W: int, FRAME_H: int) -> bool:
        """
        幾何完整性驗證：排除以下三種情況
        1. 僅手腕可見、手指被遮擋（手腕到食指尖距離過短）
        2. 左手腕連接右手手指（跨手汙染）
        3. 受測者手腕連接施測者手指（跨人汙染）
        上述兩種跨汙染情況在畫面中表現為：手腕到指尖距離
        異常地長（超過幀寬 55%），因為實際跨越了兩個不同的手/人。
        """
        wri_x = hand_landmarks_list[self.WRIST].x * FRAME_W
        wri_y = hand_landmarks_list[self.WRIST].y * FRAME_H
        wri_pos = np.array([wri_x, wri_y])
        max_span = FRAME_W * self.MAX_HAND_SPAN_RATIO
        min_extend = FRAME_W * self.MIN_FINGER_EXTEND_RATIO

        # 🌟 修改：只用指尖做 span 檢查（MCP/PIP 距離本來就短，不需驗）
        # 跨人汙染的特徵是「手腕在一側，指尖跨到另一側」，span 極大
        for lm_idx in [self.IDX_TIP, self.MID_TIP]:
            lm_x = hand_landmarks_list[lm_idx].x * FRAME_W
            lm_y = hand_landmarks_list[lm_idx].y * FRAME_H
            lm_pos = np.array([lm_x, lm_y])
            if np.linalg.norm(lm_pos - wri_pos) > max_span:
                # 🌟 跨手/跨人汙染：指尖距離手腕過遠（> 55% 幀寬），拒絕此偵測
                return False

        # 檢查食指尖距手腕是否夠遠（手指有伸出來才顯示）
        idx_tip_x = hand_landmarks_list[self.IDX_TIP].x * FRAME_W
        idx_tip_y = hand_landmarks_list[self.IDX_TIP].y * FRAME_H
        idx_tip_pos = np.array([idx_tip_x, idx_tip_y])
        if np.linalg.norm(idx_tip_pos - wri_pos) < min_extend:
            # 🌟 手指被遮擋，只偵測到手腕：不顯示、不射線
            return False

        return True

    def reset_tracking(self):
        """🌟 批次模式：重置所有影片間狀態（保留 YOLO / MediaPipe 模型）"""
        # 🌟 修正：不歸零，改為加大偏移（10,000 秒）
        #    MediaPipe VIDEO 模式要求 timestamp 單調遞增；若歸零而 mp_hands 物件不變，
        #    下一支影片的第一個 timestamp（33ms）< 上一支影片末尾 → 拋出 monotonic 錯誤。
        #    加 10,000,000ms 的跳躍確保跨影片永遠遞增，遠超任何影片實際長度。
        self.timestamp_ms += 10_000_000
        self._ts_base = self.timestamp_ms  # 🌟 新增：記錄此影片的 timestamp 起點，供 elapsed_ms 模式使用
        self.dwell_counters = {"Child": 0, "Tester": 0}
        self.hold_counters  = {"Child": 0, "Tester": 0}
        self.ray_history = {
            "Child":  {"origin": deque(maxlen=self.ray_history["Child"]["origin"].maxlen),
                       "vector": deque(maxlen=self.ray_history["Child"]["vector"].maxlen)},
            "Tester": {"origin": deque(maxlen=self.ray_history["Tester"]["origin"].maxlen),
                       "vector": deque(maxlen=self.ray_history["Tester"]["vector"].maxlen)},
        }
        self.track_owner_votes  = {}
        self.ray_angle_history  = {
            "Child":  deque(maxlen=self.ray_angle_history["Child"].maxlen),
            "Tester": deque(maxlen=self.ray_angle_history["Tester"].maxlen),
        }
        self.hand_owner_tracks = []  # 🌟 新增：跨影片清空手部歸屬追蹤器
        self.last_child_pointing_active = False  # 🌟 新增：跨影片重置
        print(">>> [InteractionEngine] tracking 狀態已重置")

    def analyze_interaction(self, frame: np.ndarray, yolo_boxes: List[Tuple[float, float, float, float]], elapsed_ms: Optional[int] = None) -> bool:
        FRAME_H, FRAME_W = frame.shape[:2]
        DIVIDER_X = int(FRAME_W * self.divider_ratio)
        child_hit_target = False

        # 🌟 新增（問題1）：每幀老化手部歸屬 track
        self._age_hand_tracks()

        cv2.line(frame, (DIVIDER_X, 0), (DIVIDER_X, FRAME_H), (255, 255, 255), 2)
        cv2.putText(frame, "TESTER ZONE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.C_TESTER, 2)
        cv2.putText(frame, "CHILD ZONE", (DIVIDER_X + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.C_CHILD, 2)

        for (yx1, yy1, yx2, yy2) in yolo_boxes:
            cv2.rectangle(frame, (int(yx1), int(yy1)), (int(yx2), int(yy2)), (255, 255, 255), 2)
            self.draw_dashed_rectangle(frame, (int(yx1)-20, int(yy1)-20), (int(yx2)+20, int(yy2)+20), (150, 150, 150), 2)

        yolo_people = []
        # 🌟 修改：conf 0.5 → 0.3，施測者背對鏡頭或兒童被桌子遮擋時信心度不足 0.5 會整人消失
        pose_results = self.model_human.track(frame, persist=True, imgsz=640, conf=0.3, verbose=False)

        tester_present = False
        child_present = False

        if pose_results and len(pose_results) > 0 and pose_results[0].keypoints is not None:
            kpts_all = pose_results[0].keypoints.xy.cpu().numpy()
            confs_all = pose_results[0].keypoints.conf.cpu().numpy()
            boxes_all = pose_results[0].boxes.xyxy.cpu().numpy()

            track_ids = None
            if pose_results[0].boxes.id is not None:
                track_ids = pose_results[0].boxes.id.cpu().numpy()

            for i in range(len(boxes_all)):
                kpts, confs = kpts_all[i], confs_all[i]

                if confs[5] > 0.4 and confs[6] > 0.4:
                    true_body_x = (kpts[5][0] + kpts[6][0]) / 2
                elif confs[5] > 0.4:
                    true_body_x = kpts[5][0]
                elif confs[6] > 0.4:
                    true_body_x = kpts[6][0]
                else:
                    true_body_x = (boxes_all[i][0] + boxes_all[i][2]) / 2

                raw_owner = "Tester" if true_body_x < DIVIDER_X else "Child"

                track_id = int(track_ids[i]) if track_ids is not None else None
                if track_id is not None:
                    if track_id not in self.track_owner_votes:
                        self.track_owner_votes[track_id] = deque(maxlen=self.owner_vote_window)
                    self.track_owner_votes[track_id].append(raw_owner)
                    votes = self.track_owner_votes[track_id]
                    tester_votes = sum(1 for v in votes if v == "Tester")
                    owner = "Tester" if tester_votes * 2 >= len(votes) else "Child"
                else:
                    owner = raw_owner

                body_scale = self.estimate_body_scale(kpts, confs, boxes_all[i])
                yolo_people.append({"owner": owner, "box": boxes_all[i], "kpts": kpts, "confs": confs, "scale": body_scale})

            # ============================================================
            # 🌟 新增（防止大人的手被判成小朋友的手）：幽靈人物去重
            # YOLO-Pose 有時會把「伸長的手臂」誤偵測成第二個人，
            # 其中心點落在兒童區 → 被標成第二個 Child →
            # 施測者的手與這個幽靈 Child 連動最強 → 誤判成小朋友的手。
            # 真人與幽靈的差異：真人有清晰的頭部關鍵點（鼻/眼/耳）。
            # 場景中最多一位施測者＋一位兒童，每個 owner 只保留
            # 「頭部關鍵點信心總和最高（同分取框面積大者）」的一位，
            # 其餘視為幽靈偵測，不畫框、不參與手部歸屬。
            # ============================================================
            if len(yolo_people) > 1:
                best_by_owner = {}
                for person in yolo_people:
                    head_conf = float(np.sum(person["confs"][0:5]))
                    bx = person["box"]
                    area = float((bx[2] - bx[0]) * (bx[3] - bx[1]))
                    rank = (head_conf, area)
                    key = person["owner"]
                    if key not in best_by_owner or rank > best_by_owner[key][0]:
                        best_by_owner[key] = (rank, person)
                yolo_people = [v[1] for v in best_by_owner.values()]

            # 存在旗標與繪製（只針對去重後的真人）
            for person in yolo_people:
                owner = person["owner"]
                if owner == "Tester": tester_present = True
                if owner == "Child": child_present = True

                color = self.C_TESTER if owner == "Tester" else self.C_CHILD
                bx1, by1, bx2, by2 = map(int, person["box"])
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)
                cv2.putText(frame, f"Body: {owner}", (bx1, by1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        if not tester_present and not child_present:
            return child_hit_target

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        if elapsed_ms is not None:
            # 🌟 修正：elapsed_ms 為當前影片內部時間（從 0 起算），
            #          加上 _ts_base 確保跨影片 timestamp 單調遞增
            self.timestamp_ms = self._ts_base + elapsed_ms
        else:
            self.timestamp_ms += 33

        hand_result = self.mp_hands.detect_for_video(mp_image, self.timestamp_ms)

        current_frame_pointing = {"Child": False, "Tester": False}
        candidate_rays = {}
        pointing_owners = set()

        if hand_result.hand_landmarks:
            handedness_list = hand_result.handedness if hand_result.handedness else []

            for hand_idx, hand_landmarks_list in enumerate(hand_result.hand_landmarks):
                p_wri = (int(hand_landmarks_list[self.WRIST].x * FRAME_W), int(hand_landmarks_list[self.WRIST].y * FRAME_H))
                p_mcp = (int(hand_landmarks_list[self.IDX_MCP].x * FRAME_W), int(hand_landmarks_list[self.IDX_MCP].y * FRAME_H))
                p_idx = (int(hand_landmarks_list[self.IDX_TIP].x * FRAME_W), int(hand_landmarks_list[self.IDX_TIP].y * FRAME_H))

                # ============================================================
                # 🌟 修改 3：臉部關鍵點排斥
                # 若「手腕」位置在任何人的鼻/眼/耳附近 90px 內，
                # 代表 MediaPipe 把臉誤判為手掌，直接跳過此偵測結果。
                # ============================================================
                if self._is_near_face(p_wri, yolo_people):
                    continue

                hand_side: Optional[str] = None
                if hand_idx < len(handedness_list) and len(handedness_list[hand_idx]) > 0:
                    label = handedness_list[hand_idx][0].category_name
                    if label == "Left":
                        hand_side = "left"
                    elif label == "Right":
                        hand_side = "right"

                # ============================================================
                # 🌟 修改 4：跨區歸屬需有可見手臂關鍵點（手腕或手肘）
                #
                # 問題根因分析：
                # - 舊版 Zone Penalty (+0.8)：懲罰了施測者真正跨區指物的情況，
                #   導致「施測者手入兒童區 → 被誤歸兒童」。
                # - 本次改回純 arm link score 判斷，但加入跨區可見性防線：
                #   若某人要跨越分割線認領一隻手，其 YOLO 手腕或手肘關鍵點
                #   必須有足夠信心度（> 0.35）。
                #
                # 效果：
                # ✓ 施測者手入兒童區：施測者手臂伸出，YOLO 能看到手腕/手肘 → 允許跨區
                # ✓ 兒童手在兒童區：施測者手腕在施測者區，距離大 → arm link score 高 → 兒童贏
                # ✓ 施測者體型大造成 arm link 誤勝：若施測者手腕不可見且跨區，直接拒絕
                # ============================================================
                wrist_in_tester_zone = (p_wri[0] < DIVIDER_X)

                best_owner, min_score = None, float('inf')
                owner_scores: Dict[str, float] = {}  # 🌟 新增：各 owner 的最佳手臂連動分數（供 Child 證據檢查）
                for person in yolo_people:
                    person_in_tester_zone = (person["owner"] == "Tester")
                    is_cross_zone = (wrist_in_tester_zone != person_in_tester_zone)

                    # 🌟 修改：手臂歸屬不再信任 MediaPipe handedness
                    # 第三人稱（非自拍）攝影機下，MediaPipe 的 Left/Right 常被標反，
                    # 會選錯手臂側(5/7/9 vs 6/8/10) → arm link score 算錯 → 歸錯人。
                    # 改為「一律計算左右兩側取最小」，對 handedness 翻轉完全免疫。
                    score_r = self.calculate_arm_link_score(p_wri, person["kpts"], person["confs"], "right", person["scale"])
                    score_l = self.calculate_arm_link_score(p_wri, person["kpts"], person["confs"], "left", person["scale"])
                    score = min(score_r, score_l)
                    # 跨區可見性檢查：任一手腕或手肘可見即可
                    if is_cross_zone:
                        arm_visible = (person["confs"][10] > 0.35 or person["confs"][8] > 0.35 or
                                       person["confs"][9] > 0.35 or person["confs"][7] > 0.35)
                        if not arm_visible:
                            score = float('inf')  # 手臂不可見，拒絕跨區

                    if score < owner_scores.get(person["owner"], float('inf')):
                        owner_scores[person["owner"]] = score  # 🌟 記錄各 owner 最佳分數

                    if score < min_score:
                        min_score = score
                        best_owner = person["owner"]

                # ============================================================
                # 🌟 修改（防止大人的指向被誤判成小朋友的指向）：
                # 舊防呆邏輯只看手腕在分界線哪一邊——手腕在兒童區就判 Child。
                # 施測者側身跨區指物、手臂關鍵點被自己身體遮擋時：
                #   跨區可見性檢查 → 施測者分數 = inf
                #   兒童以爛分數勝出或 arm link 全失敗 → 手腕在兒童區 → 誤判 Child
                #   → 產生假的 Child HIT！
                # 新邏輯：arm link 不可信時，改看手腕落在誰的「外擴身體框」內：
                #   - 只落在一人框內 → 歸那個人（施測者前傾時她的大框會涵蓋跨區手）
                #   - 落在多人框內 → 取身體框中心較近者
                #   - 誰的框都不包含 → 孤兒手，無法可靠歸屬 → 直接跳過（不標記、不射線）
                # ============================================================
                if min_score > self.ARM_LINK_ACCEPT_TH or best_owner is None:
                    fallback_owner, fallback_dist = None, float('inf')
                    for person in yolo_people:
                        bx1, by1, bx2, by2 = person["box"]
                        m = self.FALLBACK_BOX_EXPAND * max(bx2 - bx1, by2 - by1)
                        if (bx1 - m) <= p_wri[0] <= (bx2 + m) and (by1 - m) <= p_wri[1] <= (by2 + m):
                            cx, cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
                            d = math.hypot(p_wri[0] - cx, p_wri[1] - cy)
                            if d < fallback_dist:
                                fallback_dist, fallback_owner = d, person["owner"]
                    if fallback_owner is not None:
                        best_owner = fallback_owner
                    elif (not child_present) and tester_present and p_wri[0] >= DIVIDER_X:
                        # 🌟 兒童身體漏偵測補償：單一施測者場景、手在兒童區、
                        # 與施測者無連動、也不在施測者身體框內 → 極可能是
                        # YOLO 漏偵測的兒童的手，歸 Child（原補償邏輯移到此處，
                        # 不再覆蓋 fallback 依身體框做出的 Tester 判定）
                        best_owner = "Child"
                    else:
                        continue  # 孤兒手：不指派 owner、不畫骨架、不形成射線

                # 單人場景防呆
                # 🌟 修改：移除舊的「單一施測者 + 手在兒童區 + 連動弱 → 強制翻 Child」邏輯。
                # 該邏輯會覆蓋上方 fallback 依身體框做出的 Tester 判定，
                # 讓「施測者跨區伸手但手臂被遮擋」再度被誤判成 Child。
                # 兒童身體漏偵測的補償已移至 fallback 的孤兒手分支
                # （與施測者無連動、也不在其身體框內時，才歸 Child）。
                if not tester_present and child_present:
                    best_owner = "Child"

                # ============================================================
                # 🌟 新增（防止施測者高舉的手被誤判給小朋友）：
                # Child 認領需有「正向手臂證據」。
                # 情境：施測者指遠物時手高舉、剛好落在兒童身體框內
                # （兒童雙手放下、沒有在指）→ 手臂連動兩邊都弱 →
                # fallback 依框包含判給 Child → 假的 Child 射線。
                # 規則：兩人都在場、手被判給 Child、但 Child 自己的
                # 手臂連動分數 > CHILD_CLAIM_TH（無真實證據）時：
                #   - 施測者連動可信 → 改判 Tester
                #   - 兩邊都沒證據 → 整隻手跳過（不標記、不射線）
                # 兒童真的指物時手臂會舉起、腕/肘關鍵點可見 → 分數低 → 不受影響
                # ============================================================
                if (best_owner == "Child" and tester_present and child_present
                        and owner_scores.get("Child", float('inf')) > self.CHILD_CLAIM_TH):
                    if owner_scores.get("Tester", float('inf')) <= self.ARM_LINK_ACCEPT_TH:
                        best_owner = "Tester"
                    else:
                        continue

                # ============================================================
                # 🌟 新增（曖昧情況偏向施測者）：
                # Child 以「些微分差」贏過 Tester 時不予採信。
                # 施測者的手伸進兒童框、小朋友緊鄰在旁時，YOLO 對兒童
                # 手腕的估計常飄到施測者手附近，讓 Child 險勝幾幀 →
                # 假 Child 射線。要判給 Child，其連動分數必須比 Tester
                # 好至少 TESTER_PRIORITY_MARGIN；否則只要 Tester 連動
                # 可信（<= ARM_LINK_ACCEPT_TH）就判給 Tester。
                # 小朋友真的指物時分數通常 0.1~0.3、遠優於施測者，不受影響。
                # ============================================================
                if best_owner == "Child" and tester_present and child_present:
                    _cs = owner_scores.get("Child", float('inf'))
                    _ts = owner_scores.get("Tester", float('inf'))
                    if _ts <= self.ARM_LINK_ACCEPT_TH and _cs > _ts - self.TESTER_PRIORITY_MARGIN:
                        best_owner = "Tester"

                # ============================================================
                # 🌟 新增（問題1）：歸屬時間平滑
                # 上面所有邏輯得到的是「本幀 raw owner」；
                # 交給追蹤器做跨幀投票 + 遲滯，邊界處不再逐幀翻轉。
                # ============================================================
                best_owner = self._smooth_hand_owner(p_wri, best_owner, FRAME_W)

                # 🌟 修改：幾何驗證在繪製之前
                # 只有通過驗證（手指清晰可見、無跨手/跨人汙染）才繪製標籤與關鍵點
                if not self._validate_hand_geometry(hand_landmarks_list, FRAME_W, FRAME_H):
                    continue

                hand_color = self.C_CHILD if best_owner == "Child" else self.C_TESTER
                cv2.putText(frame, f"[{best_owner}]", (p_wri[0], p_wri[1]+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, hand_color, 2)
                for pt in [4, 8, 12, 16, 20]:
                    cv2.circle(frame, (int(hand_landmarks_list[pt].x * FRAME_W), int(hand_landmarks_list[pt].y * FRAME_H)), 5, hand_color, -1)

                hand_lms = LegacyHandLms(hand_landmarks_list)

                idx_angle = self.is_valid_pointing(hand_lms, FRAME_W, FRAME_H, best_owner)
                if idx_angle is None:
                    continue

                current_frame_pointing[best_owner] = True
                # 🌟 修改（問題4）：射線一律「手腕 → 食指尖」
                # 方向 = 食指尖 - 手腕；起點也改為手腕（原為食指 MCP）
                raw_vec = np.array(p_idx) - np.array(p_wri)

                if best_owner in candidate_rays:
                    if idx_angle < candidate_rays[best_owner]['angle']:
                        candidate_rays[best_owner] = {'origin': p_wri, 'vec': raw_vec, 'angle': idx_angle}
                else:
                    candidate_rays[best_owner] = {'origin': p_wri, 'vec': raw_vec, 'angle': idx_angle}

        # ============================================================
        # 🌟 修改（問題2：射線/骨架閃爍 → 一次指向被算成多次）
        # 啟用 dwell / hold 機制（原本宣告了但沒接上）：
        # - Dwell：連續指向 >= DWELL_FRAMES 幀才開始顯示射線，
        #   濾掉單幀誤判的瞬間閃現。
        # - Hold：指向中斷後保留 HOLD_FRAMES 幀緩衝，
        #   MediaPipe 短暫掉偵測不會讓射線消失又出現，
        #   同一次指向動作維持連續、不會被切成多次事件。
        # ============================================================
        for owner in ["Tester", "Child"]:
            if current_frame_pointing[owner]:
                self.dwell_counters[owner] += 1
                self.hold_counters[owner] = self.HOLD_FRAMES
                if owner in candidate_rays:
                    self.ray_history[owner]["origin"].append(candidate_rays[owner]['origin'])
                    self.ray_history[owner]["vector"].append(candidate_rays[owner]['vec'])
            else:
                if self.hold_counters[owner] > 0:
                    # 短暫掉偵測：進入 hold 緩衝，保留射線歷史，不清空
                    self.hold_counters[owner] -= 1
                else:
                    # 緩衝耗盡 → 指向動作真正結束，重置
                    self.dwell_counters[owner] = 0
                    self.ray_history[owner]["origin"].clear()
                    self.ray_history[owner]["vector"].clear()
                    self.ray_angle_history[owner].clear()

            if self.dwell_counters[owner] >= self.DWELL_FRAMES and self.hold_counters[owner] > 0:
                pointing_owners.add(owner)

        # 🌟 新增：記錄「小朋友指向射線存在」訊號（與是否指中物品無關）
        # 供 Stage 8 等「只要有指的動作就計分」的關卡使用
        self.last_child_pointing_active = (
            "Child" in pointing_owners and len(self.ray_history["Child"]["vector"]) > 0
        )

        for owner in ["Tester", "Child"]:
            if owner in pointing_owners and len(self.ray_history[owner]["vector"]) > 0:
                avg_origin = np.mean(self.ray_history[owner]["origin"], axis=0)
                avg_vector = self._robust_average_vector(list(self.ray_history[owner]["vector"]))
                avg_ox, avg_oy = int(avg_origin[0]), int(avg_origin[1])
                hand_color = self.C_CHILD if owner == "Child" else self.C_TESTER

                if np.linalg.norm(avg_vector) > 0:
                    unit_vec = avg_vector / np.linalg.norm(avg_vector)
                    # 🌟 修復：射線長度 150 -> 1500（原截斷導致數字遺失）
                    end_point = np.array([avg_ox, avg_oy]) + unit_vec * 1500

                    self.draw_25d_laser(frame, (avg_ox, avg_oy), (int(end_point[0]), int(end_point[1])), hand_color)

                    for (yx1, yy1, yx2, yy2) in yolo_boxes:
                        exact_box = (yx1, yy1, yx2, yy2)
                        if self.ray_intersects_box((avg_ox, avg_oy), unit_vec, exact_box):
                            if owner == "Tester":
                                cv2.rectangle(frame, (int(yx1), int(yy1)), (int(yx2), int(yy2)), hand_color, 5)
                            elif owner == "Child":
                                cv2.putText(frame, f"Child HIT!", (avg_ox, avg_oy - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, hand_color, 3)
                                cv2.rectangle(frame, (int(yx1), int(yy1)), (int(yx2), int(yy2)), hand_color, 5)
                                child_hit_target = True
                            break

        return child_hit_target
