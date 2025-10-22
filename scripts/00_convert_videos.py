"""
Batch convert all MP4 video files to MP3 audio
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.audio_processing import batch_convert
from src.utils import Config, setup_logger

logger = setup_logger()


def main():
    """
    Convert all video files in raw_audio directory to MP3
    """
    logger.info("=" * 60)
    logger.info("STEP 0: Video to MP3 Conversion")
    logger.info("=" * 60)

    try:
        # Perform batch conversion
        converted_files = batch_convert(
            input_dir=Config.RAW_AUDIO_DIR,
            output_dir=Config.CONVERTED_AUDIO_DIR,
            extensions=['.mp4', '.avi', '.mov', '.mkv', '.m4a', '.flv']
        )

        if converted_files:
            logger.info(f"\n✓ Successfully converted {len(converted_files)} files")
            logger.info(f"Output directory: {Config.CONVERTED_AUDIO_DIR}")
        else:
            logger.warning("No files were converted. Check your input directory.")

    except Exception as e:
        logger.error(f"Conversion failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
