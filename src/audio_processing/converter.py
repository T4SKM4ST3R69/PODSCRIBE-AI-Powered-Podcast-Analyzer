"""
Audio/Video conversion module using MoviePy and Pydub
"""
from pathlib import Path
from typing import List
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.config import Config
from src.utils.logger import setup_logger
from src.utils.helpers import get_episode_name, format_file_size

logger = setup_logger()


def convert_to_mp3(input_path: Path, output_path: Path = None,
                   bitrate: str = "192k") -> Path:
    """
    Convert MP4 video or audio file to MP3

    Args:
        input_path: Path to input file (MP4 or other audio format)
        output_path: Optional output path (auto-generated if None)
        bitrate: Output MP3 bitrate (default: 192k)

    Returns:
        Path to converted MP3 file
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Auto-generate output path if not provided
    if output_path is None:
        episode_name = get_episode_name(input_path)
        output_path = Config.CONVERTED_AUDIO_DIR / f"{episode_name}.mp3"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Converting {input_path.name} to MP3...")

    try:
        # Check if input is video (MP4, AVI, MOV, etc.)
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']

        if input_path.suffix.lower() in video_extensions:
            # Use MoviePy for video files
            logger.debug(f"Detected video file, extracting audio with MoviePy")
            video = VideoFileClip(str(input_path))
            video.audio.write_audiofile(
                str(output_path),
                codec='libmp3lame',
                bitrate=bitrate,
                logger=None  # Suppress MoviePy's verbose output
            )
            video.close()
        else:
            # Use Pydub for audio files
            logger.debug(f"Detected audio file, converting with Pydub")
            audio = AudioSegment.from_file(str(input_path))
            audio.export(
                str(output_path),
                format='mp3',
                bitrate=bitrate
            )

        output_size = format_file_size(output_path.stat().st_size)
        logger.info(f"✓ Conversion complete: {output_path.name} ({output_size})")
        return output_path

    except Exception as e:
        logger.error(f"Conversion failed for {input_path.name}: {str(e)}")
        raise


def batch_convert(input_dir: Path = None, output_dir: Path = None,
                  extensions: List[str] = None) -> List[Path]:
    """
    Batch convert all media files in a directory to MP3

    Args:
        input_dir: Directory containing input files (default: Config.RAW_AUDIO_DIR)
        output_dir: Directory for output MP3s (default: Config.CONVERTED_AUDIO_DIR)
        extensions: List of file extensions to process (default: ['.mp4', '.avi', '.mov'])

    Returns:
        List of converted MP3 file paths
    """
    input_dir = Path(input_dir) if input_dir else Config.RAW_AUDIO_DIR
    output_dir = Path(output_dir) if output_dir else Config.CONVERTED_AUDIO_DIR

    if extensions is None:
        extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wav', '.m4a', '.flac']

    # Find all matching files
    input_files = []
    for ext in extensions:
        input_files.extend(input_dir.glob(f"*{ext}"))

    if not input_files:
        logger.warning(f"No files found in {input_dir} with extensions {extensions}")
        return []

    logger.info(f"Found {len(input_files)} files to convert")

    converted_files = []
    for i, input_file in enumerate(input_files, 1):
        logger.info(f"[{i}/{len(input_files)}] Processing {input_file.name}")
        try:
            output_path = convert_to_mp3(input_file, output_dir / f"{input_file.stem}.mp3")
            converted_files.append(output_path)
        except Exception as e:
            logger.error(f"Failed to convert {input_file.name}: {e}")
            continue

    logger.info(f"✓ Batch conversion complete: {len(converted_files)}/{len(input_files)} successful")
    return converted_files
