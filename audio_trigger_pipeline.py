import argparse
import json
import os
import re
import subprocess
import sys
import wave
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


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
    keywords: Tuple[str, ...] = tuple(DEFAULT_KEYWORDS)
    speech_context_gap_sec: float = 0.8
    keyword_max_edit_distance: int = 1
    skip_whisper: bool = False
    noise_trigger_enabled: bool = True
    noise_sample_rate: int = 16000
    noise_frame_sec: float = 0.10
    noise_hop_sec: float = 0.05
    noise_min_duration_sec: float = 0.35
    noise_merge_gap_sec: float = 0.20
    noise_low_band_hz: float = 1500.0
    noise_high_band_hz: float = 4500.0
    noise_flatness_max: float = 0.42
    noise_dominant_jump_hz: float = 280.0
    noise_dominant_std_max: float = 220.0
    force_rebuild: bool = False


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
        "--force",
        action="store_true",
        help="忽略快取，強制重新分析",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> SpeechConfig:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = args.video or os.path.join(base_dir, "video", "8.mp4")
    output_dir = args.output_dir or os.path.join(base_dir, "output")
    output_dir = os.path.abspath(output_dir)

    return SpeechConfig(
        base_dir=base_dir,
        video_path=os.path.abspath(video_path),
        output_dir=output_dir,
        cache_path=os.path.join(output_dir, "speech_cache.json"),
        report_path=os.path.join(output_dir, "transcript_with_events.txt"),
        extracted_audio_path=os.path.join(output_dir, "analysis_audio.wav"),
        model_name=args.model,
        response_window_sec=args.window,
        keywords=tuple(args.keywords or DEFAULT_KEYWORDS),
        skip_whisper=args.skip_whisper,
        noise_trigger_enabled=not args.disable_noise_trigger,
        force_rebuild=args.force,
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
    text = text.strip().lower()
    text = text.translate(TEXT_CANONICAL_MAP)
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[^\w\u4e00-\u9fff]+", "", text)
    text = text.translate(DIGIT_CANONICAL_MAP)
    return text


def format_mmss(seconds: float) -> str:
    whole_seconds = int(seconds)
    minutes = whole_seconds // 60
    remain_seconds = whole_seconds % 60
    return f"{minutes:02d}:{remain_seconds:02d}"


def levenshtein_distance(text_a: str, text_b: str) -> int:
    if text_a == text_b:
        return 0
    if not text_a:
        return len(text_b)
    if not text_b:
        return len(text_a)

    previous_row = list(range(len(text_b) + 1))
    for index_a, char_a in enumerate(text_a, start=1):
        current_row = [index_a]
        for index_b, char_b in enumerate(text_b, start=1):
            insert_cost = current_row[index_b - 1] + 1
            delete_cost = previous_row[index_b] + 1
            replace_cost = previous_row[index_b - 1] + (char_a != char_b)
            current_row.append(min(insert_cost, delete_cost, replace_cost))
        previous_row = current_row
    return previous_row[-1]


def has_near_match(
    normalized_text: str,
    normalized_keyword: str,
    max_edit_distance: int,
) -> bool:
    if normalized_keyword in normalized_text:
        return True

    if max_edit_distance <= 0 or not normalized_text or not normalized_keyword:
        return False

    if min(len(normalized_text), len(normalized_keyword)) < 3:
        return False

    if abs(len(normalized_text) - len(normalized_keyword)) > max_edit_distance + 1:
        return False

    return (
        levenshtein_distance(normalized_text, normalized_keyword)
        <= max_edit_distance
    )


def find_trigger_keywords(
    text_candidates: Sequence[str],
    keywords: Sequence[str],
    max_edit_distance: int,
) -> List[str]:
    found: List[str] = []
    normalized_candidates = [
        normalize_text(candidate) for candidate in text_candidates if candidate
    ]

    for keyword in keywords:
        normalized_keyword = normalize_text(keyword)
        if not normalized_keyword:
            continue

        if any(
            has_near_match(candidate, normalized_keyword, max_edit_distance)
            for candidate in normalized_candidates
        ):
            found.append(keyword)

    return found


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
                found.append(keyword)
                break

    return found


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


def is_cache_valid(cache_data: Dict[str, Any], config: SpeechConfig) -> bool:
    expected_signature = get_video_signature(config.video_path)
    cached_signature = cache_data.get("video_signature")
    cached_config = cache_data.get("config", {})

    if cached_signature != expected_signature:
        return False

    checks = {
        "model_name": config.model_name,
        "language": config.language,
        "speech_context_gap_sec": config.speech_context_gap_sec,
        "keyword_max_edit_distance": config.keyword_max_edit_distance,
        "skip_whisper": config.skip_whisper,
        "noise_trigger_enabled": config.noise_trigger_enabled,
        "noise_sample_rate": config.noise_sample_rate,
        "noise_frame_sec": config.noise_frame_sec,
        "noise_hop_sec": config.noise_hop_sec,
        "noise_min_duration_sec": config.noise_min_duration_sec,
        "noise_merge_gap_sec": config.noise_merge_gap_sec,
        "noise_low_band_hz": config.noise_low_band_hz,
        "noise_high_band_hz": config.noise_high_band_hz,
        "noise_flatness_max": config.noise_flatness_max,
        "noise_dominant_jump_hz": config.noise_dominant_jump_hz,
        "noise_dominant_std_max": config.noise_dominant_std_max,
    }

    for key, expected_value in checks.items():
        cached_value = cached_config.get(key)
        if isinstance(expected_value, float):
            if float(cached_value) != float(expected_value):
                return False
        elif cached_value != expected_value:
            return False

    if float(cached_config.get("response_window_sec", -1)) != float(
        config.response_window_sec
    ):
        return False

    if tuple(cached_config.get("keywords", [])) != tuple(config.keywords):
        return False

    return True


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
    except ImportError as exc:
        raise RuntimeError(
            "找不到 whisper 套件。請先安裝 openai-whisper 與對應依賴。"
        ) from exc

    print(f">> 載入 Whisper 模型：{config.model_name}")
    model = whisper.load_model(config.model_name)

    print(">> 開始進行語音辨識...")
    result = model.transcribe(
        config.video_path,
        language=config.language,
        condition_on_previous_text=False,
        no_speech_threshold=0.4,
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
        start_time = float(segment["start"])
        end_time = float(segment["end"])
        text = str(segment["text"]).strip()

        if not text:
            continue

        normalized_text = normalize_text(text)
        if normalized_text and normalized_text == last_normalized_text:
            continue
        last_normalized_text = normalized_text

        records.append(
            {
                "start": round(start_time, 3),
                "end": round(end_time, 3),
                "text": text,
                "keywords": [],
                "trigger_window": None,
                "match_source": None,
                "event_type": "speech",
            }
        )

    raw_windows: List[Tuple[float, float]] = []
    for index, record in enumerate(records):
        direct_keywords = find_trigger_keywords(
            [record["text"]],
            config.keywords,
            config.keyword_max_edit_distance,
        )
        found_keywords = list(direct_keywords)
        match_source = "direct"

        if not found_keywords and index + 1 < len(records):
            next_record = records[index + 1]
            if next_record["start"] - record["end"] <= config.speech_context_gap_sec:
                found_keywords = find_boundary_keywords(
                    record["text"],
                    next_record["text"],
                    config.keywords,
                )
                if found_keywords:
                    match_source = "context"

        if not found_keywords:
            continue

        trigger_window = (
            round(record["start"], 3),
            round(record["start"] + config.response_window_sec, 3),
        )
        record["keywords"] = found_keywords
        record["trigger_window"] = trigger_window
        record["match_source"] = match_source
        raw_windows.append(trigger_window)

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

    rms_threshold = max(0.01, percentile_value(rms_values, 75) * 1.10)
    band_ratio_threshold = max(0.24, percentile_value(band_ratios, 82))
    peak_ratio_threshold = max(3.5, percentile_value(peak_ratios, 82))
    flatness_threshold = min(config.noise_flatness_max, percentile_value(flatness_values, 45))

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
        start_time = float(event["start"])
        end_time = float(event["end"])
        trigger_end = max(end_time, start_time + config.response_window_sec)
        noise_events.append(
            {
                "id": f"noise_{index:03d}",
                "start": round(start_time, 3),
                "end": round(end_time, 3),
                "event_type": "noise",
                "label": "alarm_like_noise",
                "score": event["score"],
                "peak_score": event["peak_score"],
                "dominant_freq_mean": event["dominant_freq_mean"],
                "dominant_freq_std": event["dominant_freq_std"],
                "spectral_centroid": event["spectral_centroid"],
                "spectral_flatness": event["spectral_flatness"],
                "trigger_window": (
                    round(start_time, 3),
                    round(trigger_end, 3),
                ),
            }
        )

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
            "keywords": list(config.keywords),
            "speech_context_gap_sec": config.speech_context_gap_sec,
            "keyword_max_edit_distance": config.keyword_max_edit_distance,
            "skip_whisper": config.skip_whisper,
            "noise_trigger_enabled": config.noise_trigger_enabled,
            "noise_sample_rate": config.noise_sample_rate,
            "noise_frame_sec": config.noise_frame_sec,
            "noise_hop_sec": config.noise_hop_sec,
            "noise_min_duration_sec": config.noise_min_duration_sec,
            "noise_merge_gap_sec": config.noise_merge_gap_sec,
            "noise_low_band_hz": config.noise_low_band_hz,
            "noise_high_band_hz": config.noise_high_band_hz,
            "noise_flatness_max": config.noise_flatness_max,
            "noise_dominant_jump_hz": config.noise_dominant_jump_hz,
            "noise_dominant_std_max": config.noise_dominant_std_max,
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
                if record["keywords"]:
                    start_time, end_time = record["trigger_window"]
                    match_label = (
                        "跨片段觸發"
                        if record.get("match_source") == "context"
                        else "語音觸發"
                    )
                    file.write(
                        f"⭐ {time_str} [{match_label}：{' / '.join(record['keywords'])}] "
                        f"{record['text']}\n"
                    )
                    file.write(
                        f"   -> 判定視窗：{start_time:.3f}s ~ {end_time:.3f}s\n"
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
    raw_windows: List[Tuple[float, float]] = []

    if not config.skip_whisper:
        whisper_result = transcribe_with_whisper(config)
        segment_records, speech_windows = build_segment_records(
            whisper_result.get("segments", []),
            config,
        )
        raw_windows.extend(speech_windows)

    noise_events = detect_noise_events(config)
    raw_windows.extend(
        [tuple(event["trigger_window"]) for event in noise_events if event["trigger_window"]]
    )

    trigger_windows = merge_overlapping_windows(raw_windows)
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
    speech_trigger_count = sum(1 for record in segment_records if record["keywords"])

    print("\n✅ 語音 / 怪聲解析完成")
    print(f"來源：{source_label}")
    print(f"影片：{config.video_path}")
    print(f"模型：{config.model_name}")
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
