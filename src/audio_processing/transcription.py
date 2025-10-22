"""
WhisperX transcription module with word-level timestamps
"""
import whisperx
import torch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.config import Config
from src.utils.logger import setup_logger

logger = setup_logger()


def transcribe_audio(audio_path: Path,
                     model_name: str = None,
                     device: str = None,
                     batch_size: int = None,
                     compute_type: str = None,
                     language: str = None) -> dict:
    """
    Transcribe audio file using WhisperX with word-level timestamps

    Args:
        audio_path: Path to audio file (MP3, WAV, etc.)
        model_name: Whisper model name (default: from Config)
        device: Device to use (default: from Config)
        batch_size: Batch size for processing (default: from Config)
        compute_type: Compute precision (default: from Config)
        language: Language code or None for auto-detect (default: from Config)

    Returns:
        Dictionary containing transcription with timestamps and metadata
    """
    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Use config defaults if not specified
    model_name = model_name or Config.WHISPER_MODEL
    device = device or Config.DEVICE
    batch_size = batch_size or Config.WHISPER_BATCH_SIZE
    compute_type = compute_type or Config.COMPUTE_TYPE
    language = language or Config.WHISPER_LANGUAGE

    logger.info(f"Transcribing {audio_path.name} with WhisperX")
    logger.debug(f"Model: {model_name}, Device: {device}, Batch: {batch_size}, Type: {compute_type}")

    try:
        # Load WhisperX model
        logger.info("Loading WhisperX model...")
        model = whisperx.load_model(
            model_name,
            device=device,
            compute_type=compute_type
        )

        # Load audio
        logger.info("Loading audio file...")
        audio = whisperx.load_audio(str(audio_path))

        # Transcribe
        logger.info("Transcribing audio...")
        result = model.transcribe(audio, batch_size=batch_size, language=language)

        detected_language = result.get("language", "unknown")
        logger.info(f"Detected language: {detected_language}")

        # Align for accurate word-level timestamps
        logger.info("Aligning timestamps...")
        model_a, metadata = whisperx.load_align_model(
            language_code=detected_language,
            device=device
        )

        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            device,
            return_char_alignments=False
        )

        # Cleanup GPU memory
        del model, model_a
        if device == "cuda":
            torch.cuda.empty_cache()

        logger.info(f"✓ Transcription complete: {len(result['segments'])} segments")

        # Format output
        output = {
            "file": audio_path.name,
            "language": detected_language,
            "segments": result["segments"],
            "word_segments": result.get("word_segments", [])
        }

        return output

    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}")
        raise
