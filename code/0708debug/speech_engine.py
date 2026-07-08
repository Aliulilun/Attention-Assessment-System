import argparse
import difflib
import json
import os
import re
import subprocess
import sys
import tempfile
import wave
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np


WHISPER_INITIAL_PROMPT = (
    "這是一段早療評估影片。請精確逐字記錄治療師的獨立指令，包含："
    "你看、看這裡、準備、開始、321。請勿忽略短促的聲音，不要與雜音合併。"
)

# Whisper 在無語音／安靜片段常會把 initial_prompt 逐字「複誦」回來當作辨識結果。
# 用來偵測並過濾這類幻覺片段的門檻：與 prompt 連續重疊字數、佔片段本身長度比例。
PROMPT_ECHO_MIN_MATCH_LEN = 6
PROMPT_ECHO_MIN_RATIO = 0.6

DEFAULT_KEYWORDS = [
    "開始",
    "321",
    "三二一",
    "3 2 1",
    "準備",
    "你看",
    "看這裡",
    "準備囉",
]

DEFAULT_NOISE_SAMPLE_FILENAME = "noise_reference_2m23_2m33.wav"
DEFAULT_NOISE_TEMPLATE_THRESHOLD = 0.85
DEFAULT_AUTO_NOISE_TEMPLATE_THRESHOLD = 0.85

TEXT_CANONICAL_MAP = str.maketrans(
    {
        "這": "这",
        "裡": "里",
        "準": "准",
        "囉": "啰",
    }
)

DIGIT_CANONICAL_MAP = str.maketrans(
    {
        "零": "0",
        "〇": "0",
        "一": "1",
        "二": "2",
        "兩": "2",
        "三": "3",
        "四": "4",
        "五": "5",
        "六": "6",
        "七": "7",
        "八": "8",
        "九": "9",
    }
)

COUNTDOWN_NORMALIZED_KEYWORDS = {"321"}
COUNTDOWN_FRAGMENT_KEYWORDS = {"1", "2", "3"}
SHORT_KEYWORD_MIN_CONTEXT_LEN = 2
RECOVERED_NIKAN_ALIASES = {
    "腰前向",
    "腰前像",
    "有前向",
    "有前像",
    "右前向",
    "右前像",
    "宥前向",
    "宥前像",
}
MUSIC_CONTEXT_KEYWORDS = {
    "歌",
    "音樂",
    "音乐",
    "畫畫",
    "画画",
    "畫好了",
    "画好了",
}

# 【第 15 版：合併組員 noise.wav 樣本比對修正 + 幻覺片段過濾】
MATCHING_ALGORITHM_VERSION = 15


@dataclass
class SpeechConfig:
    base_dir: str
    video_path: str
    output_dir: str
    cache_path: str
    report_path: str
    extracted_audio_path: str
    model_name: str = "large-v3"
    language: str = "zh"
    response_window_sec: float = 3.0
    time_offset_sec: float = 0.0
    keywords: Tuple[str, ...] = tuple(DEFAULT_KEYWORDS)
    speech_context_gap_sec: float = 0.8
    recovered_keyword_max_duration_sec: float = 1.6
    recovered_keyword_max_avg_probability: float = 0.72
    repeated_nikan_min_duration_sec: float = 1.15
    repeated_nikan_interval_sec: float = 0.72
    repeated_nikan_tail_margin_sec: float = 0.18

    # 強制 0 容錯 (確保不會誤判你好)
    keyword_max_edit_distance: int = 0

    skip_whisper: bool = False
    noise_trigger_enabled: bool = True
    noise_sample_rate: int = 16000
    noise_frame_sec: float = 0.10
    noise_hop_sec: float = 0.05
    noise_response_window_sec: float = 10.0
    noise_music_context_sec: float = 12.0
    noise_sample_path: Optional[str] = None
    noise_reference_start_sec: Optional[float] = None
    noise_reference_duration_sec: float = 2.5
    noise_template_padding_sec: float = 0.25
    noise_template_similarity_min: float = 0.58

    # 怪聲偵測參數
    noise_min_duration_sec: float = 0.4
    noise_max_duration_sec: float = 4.0
    noise_merge_gap_sec: float = 0.5
    noise_onset_backtrack_sec: float = 1.5
    noise_low_band_hz: float = 1000.0
    noise_high_band_hz: float = 5000.0
    noise_flatness_max: float = 0.45
    noise_dominant_jump_hz: float = 500.0
    noise_dominant_std_max: float = 400.0
    force_rebuild: bool = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Whisper 語音辨識與怪聲觸發時間窗產生器"
    )
    parser.add_argument(
        "--video",
        default=None,
        help="影片路徑。預設為 ./video/8.mp4",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="輸出資料夾。預設為 ./output",
    )
    parser.add_argument(
        "--model",
        default="large-v3",
        help="Whisper 模型名稱。預設為 large-v3",
    )
    parser.add_argument(
        "--window",
        type=float,
        default=3.0,
        help="觸發後的反應時間窗秒數。預設為 3.0",
    )
    parser.add_argument(
        "--time-offset",
        type=float,
        default=0.0,
        help="整體時間校正秒數。正數代表時間往後，負數代表時間往前",
    )
    parser.add_argument(
        "--keywords",
        nargs="*",
        default=None,
        help="自訂語音關鍵字清單。未提供時使用內建預設值",
    )
    parser.add_argument(
        "--skip-whisper",
        action="store_true",
        help="跳過 Whisper 語音辨識，只做怪聲偵測",
    )
    parser.add_argument(
        "--disable-noise-trigger",
        action="store_true",
        help="停用怪聲觸發偵測",
    )
    parser.add_argument(
        "--noise-window",
        type=float,
        default=10.0,
        help="怪聲觸發後的判定時間窗秒數。預設為 10.0",
    )
    parser.add_argument(
        "--noise-sample",
        default=None,
        help="警報聲參考樣本路徑。建議使用 1~3 秒 wav，提供後會用相似度挑選怪聲",
    )
    parser.add_argument(
        "--noise-reference-start",
        default=None,
        help="從目前影片取警報聲參考起點，可用秒數或 mm:ss，例如 2:23",
    )
    parser.add_argument(
        "--noise-reference-duration",
        type=float,
        default=2.5,
        help="從目前影片取警報聲參考的秒數。預設為 2.5",
    )
    parser.add_argument(
        "--noise-template-threshold",
        type=float,
        default=None,
        help="怪聲參考樣本最低相似度。越高越保守；自動範本預設為 0.65，其他情況預設為 0.58",
    )
    parser.add_argument(
        "--noise-max-duration",
        type=float,
        default=4.0,
        help="未使用樣本時，單一怪聲候選最長秒數。預設為 4.0",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="忽略快取，強制重新分析",
    )
    return parser.parse_args()


def resolve_noise_sample_path(
    output_dir: str,
    sample_arg: Optional[str],
) -> Tuple[Optional[str], bool]:
    if sample_arg:
        return os.path.abspath(sample_arg), False

    default_sample_path = os.path.join(output_dir, DEFAULT_NOISE_SAMPLE_FILENAME)
    if os.path.exists(default_sample_path):
        return os.path.abspath(default_sample_path), True

    return None, False


def build_config(args: argparse.Namespace) -> SpeechConfig:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = args.video or os.path.join(base_dir, "video", "8.mp4")
    output_dir = args.output_dir or os.path.join(base_dir, "output")
    output_dir = os.path.abspath(output_dir)
    noise_sample_path, uses_default_noise_sample = resolve_noise_sample_path(
        output_dir,
        args.noise_sample,
    )
    noise_template_threshold = (
        args.noise_template_threshold
        if args.noise_template_threshold is not None
        else DEFAULT_AUTO_NOISE_TEMPLATE_THRESHOLD
        if uses_default_noise_sample
        else DEFAULT_NOISE_TEMPLATE_THRESHOLD
    )

    return SpeechConfig(
        base_dir=base_dir,
        video_path=os.path.abspath(video_path),
        output_dir=output_dir,
        cache_path=os.path.join(output_dir, "speech_cache.json"),
        report_path=os.path.join(output_dir, "transcript_with_events.txt"),
        extracted_audio_path=os.path.join(output_dir, "analysis_audio.wav"),
        model_name=args.model,
        response_window_sec=args.window,
        time_offset_sec=args.time_offset,
        keywords=tuple(args.keywords or DEFAULT_KEYWORDS),
        skip_whisper=args.skip_whisper,
        noise_trigger_enabled=not args.disable_noise_trigger,
        noise_response_window_sec=args.noise_window,
        noise_sample_path=noise_sample_path,
        noise_reference_start_sec=parse_time_value(args.noise_reference_start),
        noise_reference_duration_sec=args.noise_reference_duration,
        noise_template_similarity_min=noise_template_threshold,
        noise_max_duration_sec=args.noise_max_duration,
        force_rebuild=True,
    )


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_video_signature(video_path: str) -> Dict[str, Any]:
    stat_info = os.stat(video_path)
    return {
        "path": os.path.abspath(video_path),
        "size": stat_info.st_size,
        "mtime": stat_info.st_mtime,
    }


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.strip().lower()
    text = text.translate(TEXT_CANONICAL_MAP)
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[^\w一-鿿]+", "", text)
    text = text.translate(DIGIT_CANONICAL_MAP)
    return text


_NORMALIZED_INITIAL_PROMPT = None


def is_prompt_echo(normalized_segment_text: str) -> bool:
    """判斷這段文字是否為 Whisper 對 initial_prompt 的複誦幻覺。

    Whisper 在片段內沒有清楚語音時，常會把 initial_prompt 的一段連續文字
    原封不動地「回聲」出來當作辨識結果（尤其是影片開頭的安靜段落）。
    這裡比對片段文字與 prompt 的最長連續重疊子字串，若重疊長度與比例都
    夠高，視為幻覺片段並予以捨棄，避免誤觸關鍵字（例如 prompt 裡本來就
    含有「聲音」二字，會誤觸發 Stage 8 怪聲判定）。
    """
    global _NORMALIZED_INITIAL_PROMPT
    if not normalized_segment_text:
        return False
    if _NORMALIZED_INITIAL_PROMPT is None:
        _NORMALIZED_INITIAL_PROMPT = normalize_text(WHISPER_INITIAL_PROMPT)

    matcher = difflib.SequenceMatcher(
        None, normalized_segment_text, _NORMALIZED_INITIAL_PROMPT
    )
    match = matcher.find_longest_match(
        0, len(normalized_segment_text), 0, len(_NORMALIZED_INITIAL_PROMPT)
    )
    if match.size < PROMPT_ECHO_MIN_MATCH_LEN:
        return False
    return (match.size / len(normalized_segment_text)) >= PROMPT_ECHO_MIN_RATIO


def format_mmss(seconds: float) -> str:
    whole_seconds = int(seconds)
    minutes = whole_seconds // 60
    remain_seconds = whole_seconds % 60
    return f"{minutes:02d}:{remain_seconds:02d}"


def parse_time_value(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    if ":" not in text:
        return float(text)

    parts = [float(part) for part in text.split(":")]
    if len(parts) == 2:
        minutes, seconds = parts
        return minutes * 60.0 + seconds
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return hours * 3600.0 + minutes * 60.0 + seconds

    raise ValueError(f"不支援的時間格式：{value}")


def apply_time_offset(seconds: float, config: SpeechConfig) -> float:
    return round(max(0.0, seconds + config.time_offset_sec), 3)


def is_countdown_keyword(normalized_keyword: str) -> bool:
    return normalized_keyword in COUNTDOWN_NORMALIZED_KEYWORDS


def is_countdown_fragment_keyword(normalized_keyword: str) -> bool:
    return normalized_keyword in COUNTDOWN_FRAGMENT_KEYWORDS


def is_digit_char(char: str) -> bool:
    return char.isdigit()


def is_valid_keyword_occurrence(
    normalized_text: str,
    normalized_keyword: str,
    start_index: int,
    end_index: int,
) -> bool:
    if is_countdown_keyword(normalized_keyword):
        before = normalized_text[start_index - 1] if start_index > 0 else ""
        after = normalized_text[end_index] if end_index < len(normalized_text) else ""
        return not is_digit_char(before) and not is_digit_char(after)

    if is_countdown_fragment_keyword(normalized_keyword):
        return False

    return True


def find_boundary_keywords(
    current_text: str,
    next_text: str,
    keywords: Sequence[str],
) -> List[str]:
    found: List[str] = []
    current_normalized = normalize_text(current_text)
    next_normalized = normalize_text(next_text)

    if not current_normalized or not next_normalized:
        return found

    for keyword in keywords:
        normalized_keyword = normalize_text(keyword)
        if len(normalized_keyword) < 2:
            continue
        if is_countdown_fragment_keyword(normalized_keyword):
            continue

        if normalized_keyword in current_normalized:
            continue
        if normalized_keyword in next_normalized:
            continue

        for split_index in range(1, len(normalized_keyword)):
            prefix = normalized_keyword[:split_index]
            suffix = normalized_keyword[split_index:]
            if current_normalized.endswith(prefix) and next_normalized.startswith(
                suffix
            ):
                if is_countdown_keyword(normalized_keyword):
                    prefix_start = len(current_normalized) - len(prefix)
                    before = (
                        current_normalized[prefix_start - 1]
                        if prefix_start > 0
                        else ""
                    )
                    after = (
                        next_normalized[len(suffix)]
                        if len(next_normalized) > len(suffix)
                        else ""
                    )
                    if is_digit_char(before) or is_digit_char(after):
                        continue
                found.append(keyword)
                break

    return found


def find_all_occurrences(text: str, pattern: str) -> List[int]:
    positions: List[int] = []
    start_index = 0
    while True:
        match_index = text.find(pattern, start_index)
        if match_index < 0:
            return positions
        positions.append(match_index)
        start_index = match_index + max(1, len(pattern))


def get_word_text(word: Dict[str, Any]) -> str:
    raw_word = word.get("word", word.get("text", ""))
    return str(raw_word).strip()


def build_word_timeline(
    record: Dict[str, Any],
) -> Tuple[str, List[Dict[str, float]]]:
    word_timeline: List[Dict[str, float]] = []
    normalized_words: List[str] = []

    for word in record.get("words", []) or []:
        word_text = get_word_text(word)
        normalized_word = normalize_text(word_text)
        if not normalized_word:
            continue
        if "start" not in word or "end" not in word:
            continue

        start_index = sum(len(item) for item in normalized_words)
        end_index = start_index + len(normalized_word)
        normalized_words.append(normalized_word)
        word_timeline.append(
            {
                "start_index": float(start_index),
                "end_index": float(end_index),
                "raw_start": float(word["start"]),
                "raw_end": float(word["end"]),
            }
        )

    return "".join(normalized_words), word_timeline


def count_keyword_occurrences(
    normalized_text: str,
    keywords: Sequence[str],
) -> int:
    total_count = 0
    for keyword in keywords:
        normalized_keyword = normalize_text(keyword)
        if not normalized_keyword:
            continue
        if is_countdown_fragment_keyword(normalized_keyword):
            continue
        for start_index in find_all_occurrences(normalized_text, normalized_keyword):
            end_index = start_index + len(normalized_keyword)
            if is_valid_keyword_occurrence(
                normalized_text,
                normalized_keyword,
                start_index,
                end_index,
            ):
                total_count += 1
    return total_count


def choose_keyword_detection_text(
    segment_text: str,
    word_text: str,
    keywords: Sequence[str],
) -> Tuple[str, str]:
    if not word_text:
        return segment_text, "segment"

    segment_count = count_keyword_occurrences(segment_text, keywords)
    word_count = count_keyword_occurrences(word_text, keywords)
    if word_count > segment_count:
        return word_text, "word"
    return segment_text, "segment"


def estimate_keyword_raw_start(
    record: Dict[str, Any],
    detection_text: str,
    detection_source: str,
    start_index: int,
) -> float:
    segment_start = float(record.get("raw_start", 0.0))
    segment_end = float(record.get("raw_end", segment_start))
    segment_duration = max(0.0, segment_end - segment_start)
    detection_length = max(1, len(detection_text))
    fallback_start = segment_start + (start_index / detection_length) * segment_duration

    word_text, word_timeline = build_word_timeline(record)
    if not word_timeline:
        return fallback_start

    aligned_start_index = start_index
    if detection_source != "word" and word_text != detection_text:
        aligned_start_index = int(
            round((start_index / detection_length) * max(1, len(word_text)))
        )

    for word in word_timeline:
        word_start_index = int(word["start_index"])
        word_end_index = int(word["end_index"])
        if word_start_index <= aligned_start_index < word_end_index:
            word_duration = max(0.0, word["raw_end"] - word["raw_start"])
            word_length = max(1, word_end_index - word_start_index)
            inside_word_ratio = (aligned_start_index - word_start_index) / word_length
            return word["raw_start"] + inside_word_ratio * word_duration
        if aligned_start_index < word_start_index:
            return word["raw_start"]

    return float(word_timeline[-1]["raw_start"])


def has_existing_event_near(
    events: Sequence[Dict[str, Any]],
    keyword: str,
    raw_start: float,
    tolerance_sec: float = 0.35,
) -> bool:
    for event in events:
        if keyword not in event.get("keywords", []):
            continue
        event_raw_start = float(event.get("raw_start", event.get("start", 0.0)))
        if abs(event_raw_start - raw_start) <= tolerance_sec:
            return True
    return False


def word_probability_values(record: Dict[str, Any]) -> List[float]:
    values: List[float] = []
    for word in record.get("words", []) or []:
        probability = word.get("probability")
        if probability is None:
            continue
        values.append(float(probability))
    return values


def average_word_probability(record: Dict[str, Any]) -> float:
    values = word_probability_values(record)
    if not values:
        return 1.0
    return sum(values) / len(values)


def is_recoverable_nikan_text(normalized_text: str) -> bool:
    if not normalized_text:
        return False
    if "你好" in normalized_text:
        return False
    if normalized_text in RECOVERED_NIKAN_ALIASES:
        return True

    # Whisper 對「宥辰你看」的第 4 次短句，常會把尾音錯成「前向／前像」。
    if len(normalized_text) <= 4 and (
        normalized_text.endswith("前向") or normalized_text.endswith("前像")
    ):
        return True

    return False


def estimate_recovered_nikan_start(record: Dict[str, Any]) -> float:
    words = record.get("words", []) or []
    for word in words:
        word_text = normalize_text(get_word_text(word))
        if (word_text == "前" or word_text.startswith("前")) and "start" in word:
            return float(word["start"])

    segment_start = float(record.get("raw_start", 0.0))
    segment_end = float(record.get("raw_end", segment_start))
    return segment_start + max(0.0, segment_end - segment_start) * 0.55


def build_recovered_nikan_event(
    record: Dict[str, Any],
    config: SpeechConfig,
) -> List[Dict[str, Any]]:
    if "你看" not in config.keywords:
        return []

    normalized_text = normalize_text(str(record.get("text", "")))
    if not is_recoverable_nikan_text(normalized_text):
        return []

    segment_start = float(record.get("raw_start", 0.0))
    segment_end = float(record.get("raw_end", segment_start))
    duration = max(0.0, segment_end - segment_start)
    if duration > config.recovered_keyword_max_duration_sec:
        return []

    avg_probability = average_word_probability(record)
    min_probability = min(word_probability_values(record) or [1.0])
    if (
        avg_probability > config.recovered_keyword_max_avg_probability
        and min_probability > 0.35
    ):
        return []

    estimated_raw_start = estimate_recovered_nikan_start(record)
    event_start = apply_time_offset(estimated_raw_start, config)
    event_end = apply_time_offset(
        estimated_raw_start + config.response_window_sec,
        config,
    )

    return [
        {
            "id": f"speech_{record.get('id', '000')}_recover_nikan",
            "event_type": "speech",
            "keywords": ["你看"],
            "match_source": "recovered_low_confidence",
            "recognized_text": str(record.get("text", "")),
            "avg_probability": round(avg_probability, 3),
            "raw_start": round(estimated_raw_start, 3),
            "start": event_start,
            "end": event_start,
            "trigger_window": (event_start, event_end),
        }
    ]


def build_repeated_nikan_events(
    record: Dict[str, Any],
    config: SpeechConfig,
    existing_events: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if "你看" not in config.keywords:
        return []

    normalized_text = normalize_text(str(record.get("text", "")))
    if normalized_text != "你看":
        return []

    segment_start = float(record.get("raw_start", 0.0))
    segment_end = float(record.get("raw_end", segment_start))
    duration = max(0.0, segment_end - segment_start)
    if duration < config.repeated_nikan_min_duration_sec:
        return []

    estimated_count = int(round(duration / config.repeated_nikan_interval_sec))
    estimated_count = max(2, min(estimated_count, 4))

    candidate_raw_starts = [
        min(
            segment_end - config.repeated_nikan_tail_margin_sec,
            segment_start + config.repeated_nikan_interval_sec * index,
        )
        for index in range(1, estimated_count)
    ]

    recovered_events: List[Dict[str, Any]] = []
    for index, raw_start in enumerate(candidate_raw_starts, start=1):
        if raw_start <= segment_start:
            continue
        if has_existing_event_near(existing_events, "你看", raw_start):
            continue

        event_start = apply_time_offset(raw_start, config)
        event_end = apply_time_offset(raw_start + config.response_window_sec, config)
        recovered_events.append(
            {
                "id": f"speech_{record.get('id', '000')}_repeat_nikan_{index:02d}",
                "event_type": "speech",
                "keywords": ["你看"],
                "match_source": "repeated_nikan_compensation",
                "recognized_text": str(record.get("text", "")),
                "raw_start": round(raw_start, 3),
                "start": event_start,
                "end": event_start,
                "trigger_window": (event_start, event_end),
            }
        )

    return recovered_events


def build_direct_keyword_events(
    record: Dict[str, Any],
    config: SpeechConfig,
) -> List[Dict[str, Any]]:
    # 【安全防護】：確保 record["text"] 不是 None
    raw_text = record.get("text", "")
    if not raw_text:
        return []

    segment_normalized_text = normalize_text(raw_text)
    word_normalized_text, _ = build_word_timeline(record)
    normalized_text, detection_source = choose_keyword_detection_text(
        segment_normalized_text,
        word_normalized_text,
        config.keywords,
    )

    if not normalized_text:
        return []

    if "你好" in segment_normalized_text and not any(k in normalized_text for k in ["你看", "准备", "開始", "看这里"]):
        return []

    candidates: List[Dict[str, Any]] = []
    for keyword in config.keywords:
        normalized_keyword = normalize_text(keyword)
        if not normalized_keyword:
            continue
        if is_countdown_fragment_keyword(normalized_keyword):
            continue

        for start_index in find_all_occurrences(normalized_text, normalized_keyword):
            end_index = start_index + len(normalized_keyword)
            if not is_valid_keyword_occurrence(
                normalized_text,
                normalized_keyword,
                start_index,
                end_index,
            ):
                continue
            candidates.append(
                {
                    "start_index": start_index,
                    "end_index": end_index,
                    "keyword": keyword,
                    "normalized_keyword": normalized_keyword,
                }
            )

    if not candidates:
        return build_recovered_nikan_event(record, config)

    candidates.sort(
        key=lambda item: (
            -(item["end_index"] - item["start_index"]),
            item["start_index"],
            item["keyword"],
        )
    )
    accepted: List[Dict[str, Any]] = []
    occupied_positions: Set[int] = set()
    for candidate in candidates:
        span = set(range(candidate["start_index"], candidate["end_index"]))
        if span & occupied_positions:
            same_span = [
                item
                for item in accepted
                if item["start_index"] == candidate["start_index"]
                and item["end_index"] == candidate["end_index"]
            ]
            if same_span:
                if candidate["keyword"] not in same_span[0]["keywords"]:
                    same_span[0]["keywords"].append(candidate["keyword"])
            continue

        accepted.append(
            {
                "start_index": candidate["start_index"],
                "end_index": candidate["end_index"],
                "keywords": [candidate["keyword"]],
                "normalized_keyword": candidate["normalized_keyword"]
            }
        )
        occupied_positions.update(span)

    trigger_events: List[Dict[str, Any]] = []
    for index, match in enumerate(sorted(accepted, key=lambda item: item["start_index"])):

        estimated_raw_start = estimate_keyword_raw_start(
            record,
            normalized_text,
            detection_source,
            int(match["start_index"]),
        )

        event_start = apply_time_offset(estimated_raw_start, config)
        event_end = apply_time_offset(
            estimated_raw_start + config.response_window_sec,
            config,
        )
        trigger_events.append(
            {
                "id": f"speech_{record.get('id', '000')}_{index + 1:02d}",
                "event_type": "speech",
                "keywords": sorted(set(match["keywords"])),
                "match_source": "direct_word" if detection_source == "word" else "direct",
                "raw_start": round(estimated_raw_start, 3),
                "start": event_start,
                "end": event_start,
                "trigger_window": (event_start, event_end),
            }
        )

    trigger_events.extend(
        build_repeated_nikan_events(record, config, trigger_events)
    )
    trigger_events.sort(key=lambda event: float(event.get("start", 0.0)))
    return trigger_events


def build_boundary_keyword_events(
    record: Dict[str, Any],
    next_record: Dict[str, Any],
    config: SpeechConfig,
) -> List[Dict[str, Any]]:
    # 安全防護
    r_end = float(record.get("raw_end", 0.0))
    nr_start = float(next_record.get("raw_start", 0.0))

    if nr_start - r_end > config.speech_context_gap_sec:
        return []

    r_text = record.get("text", "")
    nr_text = next_record.get("text", "")

    found_keywords = find_boundary_keywords(
        r_text,
        nr_text,
        config.keywords,
    )
    if not found_keywords:
        return []

    boundary_raw_start = r_end if nr_start <= r_end else (r_end + nr_start) / 2.0
    event_start = apply_time_offset(boundary_raw_start, config)
    event_end = apply_time_offset(
        boundary_raw_start + config.response_window_sec,
        config,
    )
    return [
        {
            "id": f"speech_{record.get('id', '000')}_context",
            "event_type": "speech",
            "keywords": sorted(set(found_keywords)),
            "match_source": "context",
            "raw_start": round(boundary_raw_start, 3),
            "start": event_start,
            "end": event_start,
            "trigger_window": (event_start, event_end),
        }
    ]


def merge_overlapping_windows(
    windows: Sequence[Tuple[float, float]]
) -> List[Tuple[float, float]]:
    if not windows:
        return []

    sorted_windows = sorted(windows, key=lambda item: item[0])
    merged: List[Tuple[float, float]] = [sorted_windows[0]]

    for current_start, current_end in sorted_windows[1:]:
        last_start, last_end = merged[-1]
        if current_start <= last_end:
            merged[-1] = (last_start, max(last_end, current_end))
        else:
            merged.append((current_start, current_end))

    return merged


def is_music_context_record(record: Dict[str, Any]) -> bool:
    normalized_text = normalize_text(str(record.get("text", "")))
    if not normalized_text:
        return False

    if any(keyword in normalized_text for keyword in MUSIC_CONTEXT_KEYWORDS):
        return True

    if re.search(r"([一-鿿])\1{2,}", normalized_text):
        return True

    return False


def is_noise_near_music_context(
    noise_event: Dict[str, Any],
    segment_records: Sequence[Dict[str, Any]],
    config: SpeechConfig,
) -> bool:
    event_start = float(noise_event.get("raw_start", noise_event.get("start", 0.0)))
    event_end = float(noise_event.get("raw_end", noise_event.get("end", event_start)))

    for record in segment_records:
        if not is_music_context_record(record):
            continue

        record_start = float(record.get("raw_start", record.get("start", 0.0)))
        record_end = float(record.get("raw_end", record.get("end", record_start)))
        gap = max(record_start - event_end, event_start - record_end, 0.0)
        if gap <= config.noise_music_context_sec:
            noise_event["rejected_reason"] = (
                f"near_music_context:{record.get('text', '')}"
            )
            return True

    return False


def select_alarm_noise_events(
    noise_events: Sequence[Dict[str, Any]],
    segment_records: Sequence[Dict[str, Any]],
    config: SpeechConfig,
) -> List[Dict[str, Any]]:
    if not noise_events:
        return []

    eligible_events = [
        event
        for event in noise_events
        if not is_noise_near_music_context(event, segment_records, config)
    ]

    template_events = [
        event
        for event in eligible_events
        if float(event.get("template_similarity", -1.0))
        >= config.noise_template_similarity_min
    ]
    if template_events:
        best_event = max(
            template_events,
            key=lambda event: (
                float(event.get("template_similarity", -1.0)),
                float(event.get("score", 0.0)),
                float(event.get("peak_score", 0.0)),
            ),
        )
        best_event["selection_reason"] = "best_reference_sample_similarity"
        return [best_event]

    fallback_template_events = [
        event
        for event in noise_events
        if float(event.get("template_similarity", -1.0))
        >= config.noise_template_similarity_min
    ]
    if fallback_template_events:
        best_event = max(
            fallback_template_events,
            key=lambda event: (
                float(event.get("template_similarity", -1.0)),
                float(event.get("score", 0.0)),
                float(event.get("peak_score", 0.0)),
            ),
        )
        best_event["selection_reason"] = (
            "fallback_reference_sample_similarity_all_rejected"
        )
        return [best_event]

    rejected_template_events = [
        event for event in noise_events if "template_similarity" in event
    ]
    if rejected_template_events and (
        config.noise_sample_path is not None
        or config.noise_reference_start_sec is not None
    ):
        best_event = max(
            rejected_template_events,
            key=lambda event: float(event.get("template_similarity", -1.0)),
        )
        best_event["rejected_reason"] = (
            f"template_similarity_below_threshold:"
            f"{float(best_event.get('template_similarity', -1.0)):.4f}"
            f"<{config.noise_template_similarity_min:.2f}"
        )
        return []

    if eligible_events:
        best_event = max(
            eligible_events,
            key=lambda event: (
                float(event.get("score", 0.0)),
                float(event.get("peak_score", 0.0)),
                -float(event.get("raw_start", event.get("start", 0.0))),
            ),
        )
        best_event["selection_reason"] = "best_non_music_alarm_candidate"
        return [best_event]

    best_event = max(
        noise_events,
        key=lambda event: (
            float(event.get("score", 0.0)),
            float(event.get("peak_score", 0.0)),
            -float(event.get("raw_start", event.get("start", 0.0))),
        ),
    )
    best_event["selection_reason"] = "fallback_best_candidate_all_rejected"
    return [best_event]


def is_cache_valid(cache_data: Dict[str, Any], config: SpeechConfig) -> bool:
    return False


def load_cache(config: SpeechConfig) -> Optional[Dict[str, Any]]:
    if config.force_rebuild or not os.path.exists(config.cache_path):
        return None

    try:
        with open(config.cache_path, "r", encoding="utf-8") as file:
            cache_data = json.load(file)
    except (OSError, json.JSONDecodeError):
        return None

    if not is_cache_valid(cache_data, config):
        return None

    return cache_data


def transcribe_with_whisper(config: SpeechConfig) -> Dict[str, Any]:
    try:
        import whisper
        import torch
        torch.set_num_threads(4)

    except ImportError as exc:
        raise RuntimeError(
            "找不到 whisper 套件。請先安裝 openai-whisper 與對應依賴。"
        ) from exc

    # 🌟 自動偵測 CUDA，若有可用 GPU 則用 GPU 執行 Whisper（沒有則自動退回 CPU）
    whisper_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f">> 載入 Whisper 模型：{config.model_name}（裝置：{whisper_device}）")
    model = whisper.load_model(config.model_name, device=whisper_device)

    print(">> 開始進行語音辨識...")

    result = model.transcribe(
        config.video_path,
        language=config.language,
        initial_prompt=WHISPER_INITIAL_PROMPT,
        temperature=0.0,
        condition_on_previous_text=False,
        no_speech_threshold=0.3,
        logprob_threshold=-2.0,
        word_timestamps=True,
        compression_ratio_threshold=2.4,
        fp16=False,
        verbose=False,
    )
    return result


def build_segment_records(
    segments: Sequence[Dict[str, Any]],
    config: SpeechConfig,
) -> Tuple[List[Dict[str, Any]], List[Tuple[float, float]]]:
    records: List[Dict[str, Any]] = []
    last_normalized_text = ""

    for segment in segments:
        # 安全防護
        start_time = float(segment.get("start", 0.0))
        end_time = float(segment.get("end", start_time))
        text = str(segment.get("text", "")).strip()

        if not text:
            continue

        normalized_text = normalize_text(text)

        if is_prompt_echo(normalized_text):
            print(
                f"   ⚠️  忽略疑似 Whisper 幻覺片段（複誦 initial_prompt）："
                f"[{format_mmss(start_time)}] {text}"
            )
            continue

        has_keyword = any(normalize_text(kw) in normalized_text for kw in config.keywords)

        if normalized_text and normalized_text == last_normalized_text and not has_keyword:
            continue

        last_normalized_text = normalized_text

        display_start_time = apply_time_offset(start_time, config)
        display_end_time = apply_time_offset(end_time, config)

        records.append(
            {
                "id": f"{len(records) + 1:04d}",
                "raw_start": round(start_time, 3),
                "raw_end": round(end_time, 3),
                "start": display_start_time,
                "end": display_end_time,
                "text": text,
                "words": segment.get("words", []), # 儲存字級時間戳供後續安全取用
                "keywords": [],
                "trigger_events": [],
                "event_type": "speech",
            }
        )

    raw_windows: List[Tuple[float, float]] = []
    for index, record in enumerate(records):
        trigger_events = build_direct_keyword_events(record, config)

        if not trigger_events and index + 1 < len(records):
            trigger_events = build_boundary_keyword_events(
                record,
                records[index + 1],
                config,
            )

        if not trigger_events:
            continue

        record["trigger_events"] = trigger_events
        record["keywords"] = sorted(
            {
                keyword
                for event in trigger_events
                for keyword in event["keywords"]
            }
        )
        raw_windows.extend(
            [tuple(event["trigger_window"]) for event in trigger_events]
        )

    return records, raw_windows


def extract_audio_track(config: SpeechConfig) -> str:
    if os.path.exists(config.extracted_audio_path) and not config.force_rebuild:
        return config.extracted_audio_path

    command = [
        "ffmpeg",
        "-y",
        "-i",
        config.video_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(config.noise_sample_rate),
        "-acodec",
        "pcm_s16le",
        config.extracted_audio_path,
    ]

    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg 音訊抽取失敗：{completed.stderr.strip()}")

    return config.extracted_audio_path


def load_wav_as_float32(wav_path: str) -> Tuple[np.ndarray, int]:
    with wave.open(wav_path, "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()
        channels = wav_file.getnchannels()
        raw_audio = wav_file.readframes(wav_file.getnframes())

    if sample_width != 2:
        raise RuntimeError("目前只支援 16-bit PCM wav 音訊格式。")

    audio = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    return audio, sample_rate


def convert_audio_sample_to_wav(
    sample_path: str,
    target_sample_rate: int,
    output_path: str,
) -> str:
    command = [
        "ffmpeg",
        "-y",
        "-i",
        sample_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(target_sample_rate),
        "-acodec",
        "pcm_s16le",
        output_path,
    ]
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"警報樣本轉檔失敗：{completed.stderr.strip()}")
    return output_path


def load_reference_sample_audio(
    sample_path: str,
    target_sample_rate: int,
) -> Tuple[np.ndarray, int]:
    # 🌟 若參考樣本本身已經是 wav，直接讀取即可，省去 ffmpeg 轉檔開銷；
    #    非 wav（例如 mp3）才需要先轉檔，且用系統暫存檔避免在 output_dir 留下垃圾檔。
    if sample_path.lower().endswith(".wav"):
        try:
            return load_wav_as_float32(sample_path)
        except Exception:
            pass

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
        converted_path = convert_audio_sample_to_wav(
            sample_path,
            target_sample_rate,
            temp_path,
        )
        return load_wav_as_float32(converted_path)
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def l2_normalize(vector: np.ndarray) -> Optional[np.ndarray]:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        return None
    return vector / norm


def build_spectral_profile(
    audio: np.ndarray,
    sample_rate: int,
    start_time: float,
    end_time: float,
    config: SpeechConfig,
) -> Optional[np.ndarray]:
    start_sample = max(0, int(start_time * sample_rate))
    end_sample = min(len(audio), int(end_time * sample_rate))
    if end_sample <= start_sample:
        return None

    clip = audio[start_sample:end_sample].astype(np.float32)
    if clip.size < int(sample_rate * 0.18):
        return None

    clip = clip - float(np.mean(clip))
    rms = float(np.sqrt(np.mean(clip * clip)))
    if rms <= 1e-5:
        return None
    clip = clip / rms

    frame_size = max(int(sample_rate * config.noise_frame_sec), 512)
    hop_size = max(int(sample_rate * config.noise_hop_sec), 128)
    if clip.size < frame_size:
        clip = np.pad(clip, (0, frame_size - clip.size))

    freqs = np.fft.rfftfreq(frame_size, d=1.0 / sample_rate)
    profile_mask = (freqs >= 500.0) & (freqs <= 6000.0)
    profile_indices = np.where(profile_mask)[0]
    if profile_indices.size < 16:
        return None

    bin_groups = np.array_split(profile_indices, 48)
    window_fn = np.hanning(frame_size).astype(np.float32)
    frame_profiles: List[np.ndarray] = []

    for start in range(0, clip.size - frame_size + 1, hop_size):
        frame = clip[start : start + frame_size] * window_fn
        magnitude = np.abs(np.fft.rfft(frame))
        band_values = [
            float(np.mean(np.log1p(magnitude[group])))
            for group in bin_groups
            if group.size > 0
        ]
        if band_values:
            frame_profiles.append(np.asarray(band_values, dtype=np.float32))

    if not frame_profiles:
        return None

    profile = np.mean(np.vstack(frame_profiles), axis=0)
    profile = profile - float(np.mean(profile))
    return l2_normalize(profile.astype(np.float32))


def build_reference_noise_profile(
    audio: np.ndarray,
    sample_rate: int,
    config: SpeechConfig,
) -> Tuple[Optional[np.ndarray], Optional[str], Optional[float]]:
    if config.noise_reference_start_sec is not None:
        reference_start = config.noise_reference_start_sec
        reference_end = reference_start + config.noise_reference_duration_sec
        profile = build_spectral_profile(
            audio,
            sample_rate,
            reference_start,
            reference_end,
            config,
        )
        return profile, f"video@{reference_start:.3f}s", config.noise_reference_duration_sec

    if not config.noise_sample_path:
        return None, None, None

    sample_audio, sample_rate = load_reference_sample_audio(
        config.noise_sample_path,
        sample_rate,
    )
    profile = build_spectral_profile(
        sample_audio,
        sample_rate,
        0.0,
        len(sample_audio) / sample_rate,
        config,
    )
    return profile, os.path.abspath(config.noise_sample_path), len(sample_audio) / sample_rate


def annotate_template_similarity(
    noise_events: Sequence[Dict[str, Any]],
    audio: np.ndarray,
    sample_rate: int,
    config: SpeechConfig,
) -> None:
    reference_profile, reference_label, _ = build_reference_noise_profile(
        audio,
        sample_rate,
        config,
    )
    if reference_profile is None:
        return

    for event in noise_events:
        event_start = float(event["raw_start"]) - config.noise_template_padding_sec
        event_end = float(event["raw_end"]) + config.noise_template_padding_sec
        event_profile = build_spectral_profile(
            audio,
            sample_rate,
            event_start,
            event_end,
            config,
        )
        if event_profile is None:
            continue
        similarity = float(np.dot(reference_profile, event_profile))
        event["template_similarity"] = round(similarity, 4)
        event["template_source"] = reference_label


def summarize_noise_interval(
    audio: np.ndarray,
    sample_rate: int,
    start_time: float,
    end_time: float,
    config: SpeechConfig,
) -> Dict[str, float]:
    start_sample = max(0, int(start_time * sample_rate))
    end_sample = min(len(audio), int(end_time * sample_rate))
    clip = audio[start_sample:end_sample].astype(np.float32)
    if clip.size < 256:
        return {
            "dominant_freq_mean": 0.0,
            "dominant_freq_std": 0.0,
            "spectral_centroid": 0.0,
            "spectral_flatness": 1.0,
        }

    frame_size = max(int(sample_rate * config.noise_frame_sec), 512)
    hop_size = max(int(sample_rate * config.noise_hop_sec), 128)
    if clip.size < frame_size:
        clip = np.pad(clip, (0, frame_size - clip.size))

    freqs = np.fft.rfftfreq(frame_size, d=1.0 / sample_rate)
    target_mask = (freqs >= config.noise_low_band_hz) & (
        freqs <= config.noise_high_band_hz
    )
    window_fn = np.hanning(frame_size).astype(np.float32)
    dominant_values: List[float] = []
    centroid_values: List[float] = []
    flatness_values: List[float] = []

    for start in range(0, clip.size - frame_size + 1, hop_size):
        frame = clip[start : start + frame_size] * window_fn
        magnitude = np.abs(np.fft.rfft(frame))
        target_bins = magnitude[target_mask]
        target_mean = float(np.mean(target_bins) + 1e-8) if target_bins.size else 1e-8
        flatness = (
            float(np.exp(np.mean(np.log(target_bins + 1e-8))) / target_mean)
            if target_bins.size
            else 1.0
        )
        dominant_values.append(float(freqs[int(np.argmax(magnitude))]))
        centroid_values.append(
            float(np.sum(freqs * magnitude) / (np.sum(magnitude) + 1e-8))
        )
        flatness_values.append(flatness)

    return {
        "dominant_freq_mean": round(float(np.mean(dominant_values)), 1),
        "dominant_freq_std": round(float(np.std(dominant_values)), 1),
        "spectral_centroid": round(float(np.mean(centroid_values)), 1),
        "spectral_flatness": round(float(np.mean(flatness_values)), 3),
    }


def detect_template_noise_event(
    audio: np.ndarray,
    sample_rate: int,
    config: SpeechConfig,
) -> Optional[Dict[str, Any]]:
    reference_profile, reference_label, reference_duration = build_reference_noise_profile(
        audio,
        sample_rate,
        config,
    )
    if reference_profile is None or reference_duration is None:
        return None

    window_duration = min(max(reference_duration, 0.8), 10.0)
    hop_sec = 0.25
    max_start = max(0.0, len(audio) / sample_rate - window_duration)
    best_start: Optional[float] = None
    best_similarity = -1.0

    current_start = 0.0
    while current_start <= max_start:
        profile = build_spectral_profile(
            audio,
            sample_rate,
            current_start,
            current_start + window_duration,
            config,
        )
        if profile is not None:
            similarity = float(np.dot(reference_profile, profile))
            if similarity > best_similarity:
                best_similarity = similarity
                best_start = current_start
        current_start += hop_sec

    if best_start is None:
        return None

    # 🌟 修正：滑動視窗掃描全片，一定會找到「相對最相似」的一段，即使該段相似度
    #    其實很低（片中根本沒有真正的怪聲）。這裡補上門檻檢查，未達標直接視為
    #    沒有偵測到怪聲，避免每支影片都硬生出一個假的 Stage 8 候選事件。
    if best_similarity < config.noise_template_similarity_min:
        return None

    raw_start_time = refine_noise_onset_time(
        best_start,
        [
            {
                "start": best_start,
                "rms": 1.0,
                "band_ratio": 1.0,
                "high_low_ratio": 1.0,
                "spectral_flatness": 0.0,
                "spectral_centroid": config.noise_low_band_hz,
            }
        ],
        0.0,
        0.0,
        config.noise_flatness_max,
        config,
    )
    raw_end_time = best_start + window_duration
    start_time = apply_time_offset(raw_start_time, config)
    end_time = apply_time_offset(raw_end_time, config)
    summary = summarize_noise_interval(
        audio,
        sample_rate,
        raw_start_time,
        raw_end_time,
        config,
    )

    return {
        "id": "noise_template_001",
        "raw_start": round(raw_start_time, 3),
        "raw_end": round(raw_end_time, 3),
        "start": start_time,
        "end": end_time,
        "event_type": "noise",
        "label": "alarm_like_noise",
        "score": round(best_similarity * 100.0, 3),
        "peak_score": round(best_similarity * 100.0, 3),
        "template_similarity": round(best_similarity, 4),
        "template_source": reference_label,
        "dominant_freq_mean": summary["dominant_freq_mean"],
        "dominant_freq_std": summary["dominant_freq_std"],
        "spectral_centroid": summary["spectral_centroid"],
        "spectral_flatness": summary["spectral_flatness"],
        "trigger_window": (
            start_time,
            apply_time_offset(
                raw_start_time + config.noise_response_window_sec,
                config,
            ),
        ),
    }


def percentile_value(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float32), q))


def group_candidate_intervals(
    candidates: Sequence[Dict[str, float]],
    config: SpeechConfig,
) -> List[Dict[str, float]]:
    if not candidates:
        return []

    grouped: List[Dict[str, float]] = []
    current_group: List[Dict[str, float]] = [candidates[0]]

    def finalize_group(group: Sequence[Dict[str, float]]) -> None:
        if not group:
            return

        start_time = float(group[0]["start"])
        end_time = float(group[-1]["end"])
        duration = end_time - start_time
        if duration < config.noise_min_duration_sec:
            return
        if duration > config.noise_max_duration_sec:
            return

        dominant_frequencies = np.asarray(
            [item["dominant_freq"] for item in group], dtype=np.float32
        )
        centroid_values = np.asarray(
            [item["spectral_centroid"] for item in group], dtype=np.float32
        )
        flatness_values = np.asarray(
            [item["spectral_flatness"] for item in group], dtype=np.float32
        )
        scores = np.asarray([item["score"] for item in group], dtype=np.float32)

        dominant_std = float(np.std(dominant_frequencies))
        flatness_mean = float(np.mean(flatness_values))
        centroid_mean = float(np.mean(centroid_values))

        if dominant_std > config.noise_dominant_std_max:
            return
        if flatness_mean > config.noise_flatness_max:
            return
        if centroid_mean < config.noise_low_band_hz:
            return
        if centroid_mean > config.noise_high_band_hz + 600.0:
            return

        grouped.append(
            {
                "start": round(start_time, 3),
                "end": round(end_time, 3),
                "score": round(float(np.mean(scores)), 3),
                "peak_score": round(float(np.max(scores)), 3),
                "dominant_freq_mean": round(float(np.mean(dominant_frequencies)), 1),
                "dominant_freq_std": round(dominant_std, 1),
                "spectral_centroid": round(centroid_mean, 1),
                "spectral_flatness": round(flatness_mean, 3),
            }
        )

    for candidate in candidates[1:]:
        last_item = current_group[-1]
        within_gap = candidate["start"] - last_item["end"] <= config.noise_merge_gap_sec
        freq_consistent = (
            abs(candidate["dominant_freq"] - last_item["dominant_freq"])
            <= config.noise_dominant_jump_hz
        )

        if within_gap and freq_consistent:
            current_group.append(candidate)
            continue

        finalize_group(current_group)
        current_group = [candidate]

    finalize_group(current_group)

    return grouped


def refine_noise_onset_time(
    detected_start: float,
    frame_metrics: Sequence[Dict[str, float]],
    rms_threshold: float,
    band_ratio_threshold: float,
    flatness_threshold: float,
    config: SpeechConfig,
) -> float:
    search_start = max(0.0, detected_start - config.noise_onset_backtrack_sec)
    relaxed_rms_threshold = max(0.008, rms_threshold * 0.65)
    relaxed_band_threshold = max(0.18, band_ratio_threshold * 0.60)
    relaxed_flatness_threshold = min(config.noise_flatness_max + 0.15, flatness_threshold + 0.15)

    onset_candidates = [
        metric
        for metric in frame_metrics
        if search_start <= metric["start"] <= detected_start
        and metric["rms"] >= relaxed_rms_threshold
        and metric["band_ratio"] >= relaxed_band_threshold
        and metric["high_low_ratio"] >= 0.45
        and metric["spectral_flatness"] <= relaxed_flatness_threshold
        and config.noise_low_band_hz <= metric["spectral_centroid"] <= config.noise_high_band_hz + 800.0
    ]

    if not onset_candidates:
        return detected_start

    return float(onset_candidates[0]["start"])


def detect_noise_events(config: SpeechConfig) -> List[Dict[str, Any]]:
    if not config.noise_trigger_enabled:
        return []

    print(">> 開始進行怪聲偵測...")
    wav_path = extract_audio_track(config)
    audio, sample_rate = load_wav_as_float32(wav_path)

    frame_size = max(int(sample_rate * config.noise_frame_sec), 256)
    hop_size = max(int(sample_rate * config.noise_hop_sec), 128)
    if len(audio) < frame_size:
        return []

    window_fn = np.hanning(frame_size).astype(np.float32)
    freqs = np.fft.rfftfreq(frame_size, d=1.0 / sample_rate)
    target_mask = (freqs >= config.noise_low_band_hz) & (
        freqs <= config.noise_high_band_hz
    )
    low_mask = freqs < 1000.0

    frame_metrics: List[Dict[str, float]] = []
    rms_values: List[float] = []
    band_ratios: List[float] = []
    peak_ratios: List[float] = []
    flatness_values: List[float] = []

    for start in range(0, len(audio) - frame_size + 1, hop_size):
        frame = audio[start : start + frame_size]
        rms = float(np.sqrt(np.mean(frame * frame)))
        windowed = frame * window_fn
        magnitude = np.abs(np.fft.rfft(windowed))

        total_energy = float(np.sum(magnitude) + 1e-8)
        target_energy = float(np.sum(magnitude[target_mask]) + 1e-8)
        low_energy = float(np.sum(magnitude[low_mask]) + 1e-8)
        target_bins = magnitude[target_mask]
        target_mean = float(np.mean(target_bins) + 1e-8)
        target_peak = float(np.max(target_bins) + 1e-8) if target_bins.size else 0.0
        target_flatness = (
            float(np.exp(np.mean(np.log(target_bins + 1e-8))) / target_mean)
            if target_bins.size
            else 1.0
        )
        dominant_index = int(np.argmax(magnitude))
        dominant_freq = float(freqs[dominant_index])
        spectral_centroid = float(
            np.sum(freqs * magnitude) / (np.sum(magnitude) + 1e-8)
        )

        band_ratio = target_energy / total_energy
        peak_ratio = target_peak / target_mean if target_bins.size else 0.0
        high_low_ratio = target_energy / low_energy

        frame_metrics.append(
            {
                "start": start / sample_rate,
                "end": (start + frame_size) / sample_rate,
                "rms": rms,
                "band_ratio": band_ratio,
                "peak_ratio": peak_ratio,
                "high_low_ratio": high_low_ratio,
                "spectral_flatness": target_flatness,
                "dominant_freq": dominant_freq,
                "spectral_centroid": spectral_centroid,
            }
        )
        rms_values.append(rms)
        band_ratios.append(band_ratio)
        peak_ratios.append(peak_ratio)
        flatness_values.append(target_flatness)

    rms_threshold = max(0.015, percentile_value(rms_values, 50) * 1.5)
    band_ratio_threshold = max(0.3, percentile_value(band_ratios, 50))
    peak_ratio_threshold = max(2.0, percentile_value(peak_ratios, 50))
    flatness_threshold = min(config.noise_flatness_max, percentile_value(flatness_values, 60))

    candidates: List[Dict[str, float]] = []
    for metric in frame_metrics:
        if metric["rms"] < rms_threshold:
            continue
        if metric["band_ratio"] < band_ratio_threshold:
            continue
        if metric["peak_ratio"] < peak_ratio_threshold:
            continue
        if metric["high_low_ratio"] < 0.75:
            continue
        if metric["spectral_flatness"] > flatness_threshold:
            continue
        if metric["dominant_freq"] < config.noise_low_band_hz:
            continue
        if metric["dominant_freq"] > config.noise_high_band_hz:
            continue

        score = (
            metric["band_ratio"] * 2.5
            + metric["peak_ratio"] * 0.4
            + metric["high_low_ratio"] * 0.6
            + max(0.0, 0.35 - metric["spectral_flatness"]) * 2.0
        )
        candidates.append(
            {
                "start": metric["start"],
                "end": metric["end"],
                "score": score,
                "dominant_freq": metric["dominant_freq"],
                "spectral_centroid": metric["spectral_centroid"],
                "spectral_flatness": metric["spectral_flatness"],
            }
        )

    grouped = group_candidate_intervals(candidates, config)

    noise_events: List[Dict[str, Any]] = []
    for index, event in enumerate(grouped, start=1):
        raw_start_time = refine_noise_onset_time(
            float(event["start"]),
            frame_metrics,
            rms_threshold,
            band_ratio_threshold,
            flatness_threshold,
            config,
        )
        raw_end_time = float(event["end"])
        start_time = apply_time_offset(raw_start_time, config)
        end_time = apply_time_offset(raw_end_time, config)

        trigger_start = start_time
        trigger_end = apply_time_offset(
            raw_start_time + config.noise_response_window_sec,
            config,
        )

        noise_events.append(
            {
                "id": f"noise_{index:03d}",
                "raw_start": round(raw_start_time, 3),
                "raw_end": round(raw_end_time, 3),
                "start": start_time,
                "end": end_time,
                "event_type": "noise",
                "label": "alarm_like_noise",
                "score": event["score"],
                "peak_score": event["peak_score"],
                "dominant_freq_mean": event["dominant_freq_mean"],
                "dominant_freq_std": event["dominant_freq_std"],
                "spectral_centroid": event["spectral_centroid"],
                "spectral_flatness": event["spectral_flatness"],
                "trigger_window": (
                    trigger_start,
                    trigger_end,
                ),
            }
        )

    template_event = detect_template_noise_event(audio, sample_rate, config)
    if template_event is not None:
        noise_events.append(template_event)

    annotate_template_similarity(noise_events, audio, sample_rate, config)
    return noise_events


def save_cache(
    config: SpeechConfig,
    segment_records: Sequence[Dict[str, Any]],
    noise_events: Sequence[Dict[str, Any]],
    trigger_windows: Sequence[Tuple[float, float]],
) -> None:
    payload = {
        "video_signature": get_video_signature(config.video_path),
        "config": {
            "model_name": config.model_name,
            "language": config.language,
            "response_window_sec": config.response_window_sec,
            "time_offset_sec": config.time_offset_sec,
            "keywords": list(config.keywords),
            "speech_context_gap_sec": config.speech_context_gap_sec,
            "recovered_keyword_max_duration_sec": config.recovered_keyword_max_duration_sec,
            "recovered_keyword_max_avg_probability": config.recovered_keyword_max_avg_probability,
            "repeated_nikan_min_duration_sec": config.repeated_nikan_min_duration_sec,
            "repeated_nikan_interval_sec": config.repeated_nikan_interval_sec,
            "repeated_nikan_tail_margin_sec": config.repeated_nikan_tail_margin_sec,
            "keyword_max_edit_distance": config.keyword_max_edit_distance,
            "skip_whisper": config.skip_whisper,
            "noise_trigger_enabled": config.noise_trigger_enabled,
            "noise_sample_rate": config.noise_sample_rate,
            "noise_frame_sec": config.noise_frame_sec,
            "noise_hop_sec": config.noise_hop_sec,
            "noise_response_window_sec": config.noise_response_window_sec,
            "noise_music_context_sec": config.noise_music_context_sec,
            "noise_sample_path": config.noise_sample_path,
            "noise_reference_start_sec": config.noise_reference_start_sec,
            "noise_reference_duration_sec": config.noise_reference_duration_sec,
            "noise_template_padding_sec": config.noise_template_padding_sec,
            "noise_template_similarity_min": config.noise_template_similarity_min,
            "noise_min_duration_sec": config.noise_min_duration_sec,
            "noise_max_duration_sec": config.noise_max_duration_sec,
            "noise_merge_gap_sec": config.noise_merge_gap_sec,
            "noise_onset_backtrack_sec": config.noise_onset_backtrack_sec,
            "noise_low_band_hz": config.noise_low_band_hz,
            "noise_high_band_hz": config.noise_high_band_hz,
            "noise_flatness_max": config.noise_flatness_max,
            "noise_dominant_jump_hz": config.noise_dominant_jump_hz,
            "noise_dominant_std_max": config.noise_dominant_std_max,
            "matching_algorithm_version": MATCHING_ALGORITHM_VERSION,
        },
        "segment_records": list(segment_records),
        "noise_events": list(noise_events),
        "trigger_windows": [list(window) for window in trigger_windows],
    }

    with open(config.cache_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def save_report(
    config: SpeechConfig,
    segment_records: Sequence[Dict[str, Any]],
    noise_events: Sequence[Dict[str, Any]],
    trigger_windows: Sequence[Tuple[float, float]],
) -> None:
    with open(config.report_path, "w", encoding="utf-8") as file:
        file.write("=== 影片語音逐字稿與觸發事件紀錄 ===\n")
        file.write(f"影片：{config.video_path}\n")
        file.write(f"模型：{config.model_name}\n")
        file.write(f"反應時間窗：{config.response_window_sec:.1f} 秒\n")
        file.write(f"怪聲判定時間窗：{config.noise_response_window_sec:.1f} 秒\n")
        file.write(
            f"怪聲樣本相似度門檻：{config.noise_template_similarity_min:.2f}\n"
        )
        file.write(f"怪聲候選最長長度：{config.noise_max_duration_sec:.1f} 秒\n")
        file.write(f"時間校正：{config.time_offset_sec:+.3f} 秒\n")
        if config.noise_reference_start_sec is not None:
            file.write(
                f"怪聲參考來源：目前影片 {config.noise_reference_start_sec:.3f}s "
                f"起 {config.noise_reference_duration_sec:.1f} 秒\n"
            )
        elif config.noise_sample_path:
            file.write(f"怪聲參考來源：{config.noise_sample_path}\n")
        file.write(f"語音關鍵字：{'、'.join(config.keywords)}\n")
        file.write(f"是否跳過 Whisper：{'是' if config.skip_whisper else '否'}\n")
        file.write(
            f"是否啟用怪聲觸發：{'是' if config.noise_trigger_enabled else '否'}\n"
        )
        file.write(f"總觸發視窗數：{len(trigger_windows)}\n\n")

        file.write("=== 語音逐字稿 ===\n")
        if segment_records:
            for record in segment_records:
                time_str = (
                    f"[{format_mmss(record['start'])} - {format_mmss(record['end'])}]"
                )
                if record["trigger_events"]:
                    file.write(
                        f"⭐ {time_str} {record['text']}\n"
                    )
                    for event in record["trigger_events"]:
                        start_time, end_time = event["trigger_window"]
                        match_label = (
                            "跨片段觸發"
                            if event.get("match_source") == "context"
                            else "低信心修復觸發"
                            if event.get("match_source") == "recovered_low_confidence"
                            else "連續你看補償觸發"
                            if event.get("match_source") == "repeated_nikan_compensation"
                            else "語音觸發"
                        )
                        file.write(
                            f"   -> [{match_label}：{' / '.join(event['keywords'])}] "
                            f"{event['start']:.3f}s，判定視窗："
                            f"{start_time:.3f}s ~ {end_time:.3f}s\n"
                        )
                else:
                    file.write(f"   {time_str} {record['text']}\n")
        else:
            file.write("（本次未執行 Whisper 語音辨識）\n")

        file.write("\n=== 怪聲事件 ===\n")
        if noise_events:
            for event in noise_events:
                start_time, end_time = event["trigger_window"]
                file.write(
                    f"🔊 [{format_mmss(event['start'])} - {format_mmss(event['end'])}] "
                    f"{event['label']} score={event['score']:.3f} "
                    f"peak={event['peak_score']:.3f} "
                    f"freq={event['dominant_freq_mean']:.1f}Hz "
                    f"flatness={event['spectral_flatness']:.3f}\n"
                )
                if event.get("template_similarity") is not None:
                    file.write(
                        f"   -> 樣本相似度：{event['template_similarity']:.4f} "
                        f"source={event.get('template_source', '')}\n"
                    )
                if event.get("selection_reason"):
                    file.write(
                        f"   -> 選擇原因：{event['selection_reason']}\n"
                    )
                file.write(
                    f"   -> 判定視窗：{start_time:.3f}s ~ {end_time:.3f}s\n"
                )
        else:
            file.write("（未偵測到符合規則的警報型怪聲）\n")

        file.write("\n=== 合併後觸發時間窗 ===\n")
        for index, (start_time, end_time) in enumerate(trigger_windows, start=1):
            file.write(f"{index:02d}. {start_time:.3f}s ~ {end_time:.3f}s\n")


def load_or_build_analysis(
    config: SpeechConfig,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Tuple[float, float]], bool]:
    cached_data = load_cache(config)
    if cached_data is not None:
        segment_records = cached_data.get("segment_records", [])
        noise_events = cached_data.get("noise_events", [])
        trigger_windows = [
            (float(window[0]), float(window[1]))
            for window in cached_data.get("trigger_windows", [])
        ]
        return segment_records, noise_events, trigger_windows, True

    segment_records: List[Dict[str, Any]] = []
    speech_windows: List[Tuple[float, float]] = []

    if not config.skip_whisper:
        whisper_result = transcribe_with_whisper(config)
        segment_records, speech_windows = build_segment_records(
            whisper_result.get("segments", []),
            config,
        )

    noise_events = select_alarm_noise_events(
        detect_noise_events(config),
        segment_records,
        config,
    )

    noise_windows = merge_overlapping_windows(
        [
            tuple(event["trigger_window"])
            for event in noise_events
            if event["trigger_window"]
        ]
    )

    trigger_windows = sorted(
        speech_windows + noise_windows,
        key=lambda window: window[0],
    )
    save_cache(config, segment_records, noise_events, trigger_windows)
    return segment_records, noise_events, trigger_windows, False


def print_summary(
    segment_records: Sequence[Dict[str, Any]],
    noise_events: Sequence[Dict[str, Any]],
    trigger_windows: Sequence[Tuple[float, float]],
    loaded_from_cache: bool,
    config: SpeechConfig,
) -> None:
    source_label = "快取" if loaded_from_cache else "重新分析"
    speech_trigger_count = sum(
        len(record.get("trigger_events", [])) for record in segment_records
    )

    print("\n✅ 語音 / 怪聲解析完成")
    print(f"來源：{source_label}")
    print(f"影片：{config.video_path}")
    print(f"模型：{config.model_name}")
    print(f"時間校正：{config.time_offset_sec:+.3f}s")
    print(f"語音觸發次數：{speech_trigger_count}")
    print(f"怪聲事件次數：{len(noise_events)}")
    print(f"合併後判定視窗：{len(trigger_windows)} 段")
    print(f"文字報告：{config.report_path}")
    print(f"快取檔：{config.cache_path}")
    if noise_events:
        print("怪聲事件：")
        for event in noise_events:
            print(
                f"  - {event['start']:.3f}s ~ {event['end']:.3f}s "
                f"score={event['score']:.3f} "
                f"freq={event['dominant_freq_mean']:.1f}Hz"
            )
    print("觸發時間窗：")
    for start_time, end_time in trigger_windows:
        print(f"  - {start_time:.3f}s ~ {end_time:.3f}s")


def main() -> int:
    args = parse_args()
    config = build_config(args)

    print("--- 系統啟動中（語音辨識模組） ---")

    if not os.path.exists(config.video_path):
        print(f"❌ 找不到影片檔案：{config.video_path}")
        return 1

    ensure_output_dir(config.output_dir)

    try:
        (
            segment_records,
            noise_events,
            trigger_windows,
            loaded_from_cache,
        ) = load_or_build_analysis(config)
        save_report(config, segment_records, noise_events, trigger_windows)
        print_summary(
            segment_records,
            noise_events,
            trigger_windows,
            loaded_from_cache,
            config,
        )
        return 0
    except KeyboardInterrupt:
        print("\n⚠️ 分析已被手動中止。")
        return 130
    except Exception as exc:
        print(f"❌ 語音解析失敗：{exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())