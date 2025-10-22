"""
Helper utilities with ACCURATE timestamp handling
"""
import json
from pathlib import Path
from typing import Dict, Any
from datetime import timedelta


def seconds_to_timestamp(seconds: float, include_ms: bool = True) -> str:
    """
    Convert seconds to HH:MM:SS or HH:MM:SS.mmm format with ACCURACY

    Args:
        seconds: Time in seconds (can be float)
        include_ms: Include milliseconds

    Returns:
        Formatted timestamp string

    Examples:
        >>> seconds_to_timestamp(125.456)
        '00:02:05.456'
        >>> seconds_to_timestamp(3661.5, include_ms=False)
        '01:01:01'
    """
    if seconds < 0:
        seconds = 0

    td = timedelta(seconds=seconds)
    hours = td.seconds // 3600
    minutes = (td.seconds % 3600) // 60
    secs = td.seconds % 60

    if include_ms:
        milliseconds = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"
    else:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def timestamp_to_seconds(timestamp: str) -> float:
    """
    Convert HH:MM:SS or HH:MM:SS.mmm to seconds with ACCURACY

    Args:
        timestamp: Time string in format HH:MM:SS or HH:MM:SS.mmm

    Returns:
        Time in seconds (float)

    Examples:
        >>> timestamp_to_seconds('00:02:05.456')
        125.456
        >>> timestamp_to_seconds('01:01:01')
        3661.0
    """
    try:
        # Handle formats: HH:MM:SS or HH:MM:SS.mmm
        if '.' in timestamp:
            time_part, ms_part = timestamp.split('.')
            milliseconds = float(f"0.{ms_part}")
        else:
            time_part = timestamp
            milliseconds = 0.0

        parts = time_part.split(':')

        if len(parts) == 3:
            hours, minutes, seconds = map(int, parts)
        elif len(parts) == 2:
            hours = 0
            minutes, seconds = map(int, parts)
        else:
            raise ValueError(f"Invalid timestamp format: {timestamp}")

        total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds
        return round(total_seconds, 3)

    except Exception as e:
        raise ValueError(f"Invalid timestamp format '{timestamp}': {str(e)}")


def get_episode_name(file_path: Path) -> str:
    """
    Extract episode name from file path

    Args:
        file_path: Path to audio/video file

    Returns:
        Episode name without extension
    """
    return Path(file_path).stem


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted size string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def save_json(data: Dict[str, Any], file_path: Path, indent: int = 2) -> None:
    """
    Save dictionary to JSON file

    Args:
        data: Dictionary to save
        file_path: Output file path
        indent: JSON indentation level
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(file_path: Path) -> Dict[str, Any]:
    """
    Load JSON file to dictionary

    Args:
        file_path: Input file path

    Returns:
        Dictionary with JSON data
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
