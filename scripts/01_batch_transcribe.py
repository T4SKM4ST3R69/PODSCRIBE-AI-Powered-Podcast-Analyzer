"""
Batch transcribe all audio files with WhisperX and Pyannote diarization
"""
import sys
from pathlib import Path
import time


# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.audio_processing import transcribe_audio, diarize_audio, merge_transcription_and_diarization
from src.utils import Config, setup_logger, get_episode_name, save_json

logger = setup_logger()


def process_single_audio(audio_path: Path) -> bool:
    """
    Process a single audio file: transcribe, diarize, and merge

    Args:
        audio_path: Path to audio file

    Returns:
        True if successful, False otherwise
    """
    episode_name = get_episode_name(audio_path)
    output_path = Config.TRANSCRIPTS_DIR / f"{episode_name}.json"

    # Skip if already processed
    if output_path.exists():
        logger.info(f"[SKIP] Skipping...")
        return True

    try:
        logger.info(f" Processing: {audio_path.name}")
        start_time = time.time()

        # Step 1: Transcription with WhisperX
        logger.info("  1/3 Transcribing with WhisperX...")
        transcription = transcribe_audio(audio_path)

        # Step 2: Speaker diarization with Pyannote
        logger.info("  2/3 Performing speaker diarization...")
        diarization = diarize_audio(audio_path)

        # Step 3: Merge results
        logger.info("  3/3 Merging transcription and diarization...")
        merged = merge_transcription_and_diarization(
            transcription,
            diarization,
            output_path=output_path
        )

        elapsed = time.time() - start_time
        logger.info(f"✓ Completed {episode_name} in {elapsed:.1f}s")
        logger.info(f"  - Language: {merged['language']}")
        logger.info(f"  - Speakers: {len(merged['speakers'])}")
        logger.info(f"  - Segments: {len(merged['segments'])}")

        return True

    except Exception as e:
        logger.error(f" Failed to process {episode_name}: {str(e)}")
        return False


def main():
    """
    Batch transcribe all audio files
    """

    logger.info("=" * 60)
    logger.info("STEP 1: Batch Transcription with WhisperX + Pyannote")
    logger.info("=" * 60)

    # Gather all audio files from both directories
    audio_files = []

    # MP3 files from raw_audio
    audio_files.extend(Config.RAW_AUDIO_DIR.glob("*.mp3"))
    audio_files.extend(Config.RAW_AUDIO_DIR.glob("*.wav"))

    # MP3 files from converted_audio
    audio_files.extend(Config.CONVERTED_AUDIO_DIR.glob("*.mp3"))

    # Remove duplicates (same filename from different dirs)
    audio_files = list(set(audio_files))

    if not audio_files:
        logger.error(" No audio files found in:")
        logger.error(f"   - {Config.RAW_AUDIO_DIR}")
        logger.error(f"   - {Config.CONVERTED_AUDIO_DIR}")
        sys.exit(1)

    logger.info(f"Found {len(audio_files)} audio files to process")
    logger.info(f"Using device: {Config.DEVICE}")
    logger.info(f"WhisperX model: {Config.WHISPER_MODEL}")
    logger.info(f"Batch size: {Config.WHISPER_BATCH_SIZE}")
    logger.info("")

    # Process each file
    success_count = 0
    failed_files = []
    total_start = time.time()

    for i, audio_file in enumerate(audio_files, 1):
        logger.info(f"\n[{i}/{len(audio_files)}] " + "=" * 40)

        if process_single_audio(audio_file):
            success_count += 1
        else:
            failed_files.append(audio_file.name)

    # Summary
    total_elapsed = time.time() - total_start
    logger.info("\n" + "=" * 60)
    logger.info("TRANSCRIPTION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total files: {len(audio_files)}")
    logger.info(f" Success: {success_count}")
    logger.info(f"Failed: {len(failed_files)}")
    logger.info(f"  Total time: {total_elapsed / 60:.1f} minutes")

    if failed_files:
        logger.warning("\nFailed files:")
        for filename in failed_files:
            logger.warning(f"  - {filename}")

    logger.info(f"\n✓ Transcripts saved to: {Config.TRANSCRIPTS_DIR}")


if __name__ == "__main__":
    main()
