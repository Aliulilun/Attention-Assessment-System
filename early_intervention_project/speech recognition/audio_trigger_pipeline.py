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
    skip_whisper: bool = False
    noise_trigger_enabled: bool = True
    noise_sample_rate: int = 16000
    noise_frame_sec: float = 0.10
    noise_hop_sec: float = 0.05
    noise_min_duration_sec: float = 0.35
    noise_merge_gap_sec: float = 0.20
    noise_low_band_hz: float = 1500.0
    noise_high_band_hz: float = 4500.0
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
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[^\w\u4e00-\u9fff]+", "", text)
    return text


def format_mmss(seconds: float) -> str:
    whole_seconds = int(seconds)
    minutes = whole_seconds // 60
    remain_seconds = whole_seconds % 60
    return f"{minutes:02d}:{remain_seconds:02d}"


def find_trigger_keywords(text: str, keywords: Sequence[str]) -> List[str]:
    found: List[str] = []
    normalized_text = normalize_text(text)

    for keyword in keywords:
        normalized_keyword = normalize_text(keyword)
        if not normalized_keyword:
            continue

        if keyword in text or normalized_keyword in normalized_text:
            found.append(keyword)

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
        "skip_whisper": config.skip_whisper,
        "noise_trigger_enabled": config.noise_trigger_enabled,
        "noise_sample_rate": config.noise_sample_rate,
        "noise_frame_sec": config.noise_frame_sec,
        "noise_hop_sec": config.noise_hop_sec,
        "noise_min_duration_sec": config.noise_min_duration_sec,
        "noise_merge_gap_sec": config.noise_merge_gap_sec,
        "noise_low_band_hz": config.noise_low_band_hz,
        "noise_high_band_hz": config.noise_high_band_hz,
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
    raw_windows: List[Tuple[float, float]] = []
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

        found_keywords = find_trigger_keywords(text, config.keywords)
        trigger_window = None

        if found_keywords:
            trigger_window = (
                round(start_time, 3),
                round(start_time + config.response_window_sec, 3),
            )
            raw_windows.append(trigger_window)

        records.append(
            {
                "start": round(start_time, 3),
                "end": round(end_time, 3),
                "text": text,
                "keywords": found_keywords,
                "trigger_window": trigger_window,
                "event_type": "speech",
            }
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


def percentile_value(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float32), q))


def group_candidate_intervals(
    candidates: Sequence[Tuple[float, float, float]],
    min_duration_sec: float,
    merge_gap_sec: float,
) -> List[Dict[str, float]]:
    if not candidates:
        return []

    grouped: List[Dict[str, float]] = []
    current_start, current_end, current_score = candidates[0]
    current_peak = current_score
    current_count = 1

    for start_time, end_time, score in candidates[1:]:
        if start_time - current_end <= merge_gap_sec:
            current_end = end_time
            current_peak = max(current_peak, score)
            current_score += score
            current_count += 1
            continue

        if current_end - current_start >= min_duration_sec:
            grouped.append(
                {
                    "start": round(current_start, 3),
                    "end": round(current_end, 3),
                    "score": round(current_score / current_count, 3),
                    "peak_score": round(current_peak, 3),
                }
            )

        current_start, current_end, current_score = start_time, end_time, score
        current_peak = score
        current_count = 1

    if current_end - current_start >= min_duration_sec:
        grouped.append(
            {
                "start": round(current_start, 3),
                "end": round(current_end, 3),
                "score": round(current_score / current_count, 3),
                "peak_score": round(current_peak, 3),
            }
        )

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
            }
        )
        rms_values.append(rms)
        band_ratios.append(band_ratio)
        peak_ratios.append(peak_ratio)

    rms_threshold = max(0.01, percentile_value(rms_values, 75) * 1.10)
    band_ratio_threshold = max(0.28, percentile_value(band_ratios, 85))
    peak_ratio_threshold = max(3.2, percentile_value(peak_ratios, 80))

    candidates: List[Tuple[float, float, float]] = []
    for metric in frame_metrics:
        if metric["rms"] < rms_threshold:
            continue
        if metric["band_ratio"] < band_ratio_threshold:
            continue
        if metric["peak_ratio"] < peak_ratio_threshold:
            continue
        if metric["high_low_ratio"] < 0.75:
            continue

        score = (
            metric["band_ratio"] * 2.5
            + metric["peak_ratio"] * 0.4
            + metric["high_low_ratio"] * 0.6
        )
        candidates.append((metric["start"], metric["end"], score))

    grouped = group_candidate_intervals(
        candidates,
        min_duration_sec=config.noise_min_duration_sec,
        merge_gap_sec=config.noise_merge_gap_sec,
    )

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
            "skip_whisper": config.skip_whisper,
            "noise_trigger_enabled": config.noise_trigger_enabled,
            "noise_sample_rate": config.noise_sample_rate,
            "noise_frame_sec": config.noise_frame_sec,
            "noise_hop_sec": config.noise_hop_sec,
            "noise_min_duration_sec": config.noise_min_duration_sec,
            "noise_merge_gap_sec": config.noise_merge_gap_sec,
            "noise_low_band_hz": config.noise_low_band_hz,
            "noise_high_band_hz": config.noise_high_band_hz,
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
                    file.write(
                        f"⭐ {time_str} [語音觸發：{' / '.join(record['keywords'])}] "
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
                    f"peak={event['peak_score']:.3f}\n"
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
                f"score={event['score']:.3f}"
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
