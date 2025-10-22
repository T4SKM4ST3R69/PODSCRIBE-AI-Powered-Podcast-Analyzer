"""
Pyannote speaker diarization module
"""
from pyannote.audio import Pipeline
import torch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.config import Config
from src.utils.logger import setup_logger

logger = setup_logger()


def diarize_audio(audio_path: Path,
                  hf_token: str = None,
                  model_name: str = None,
                  min_speakers: int = None,
                  max_speakers: int = None,
                  device: str = None) -> dict:
    """
    Perform speaker diarization using Pyannote

    Args:
        audio_path: Path to audio file
        hf_token: Hugging Face authentication token (default: from Config)
        model_name: Diarization model name (default: from Config)
        min_speakers: Minimum number of speakers (None for auto-detect)
        max_speakers: Maximum number of speakers (None for auto-detect)
        device: Device to use (default: from Config)

    Returns:
        Dictionary containing speaker segments with timestamps
    """
    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Use config defaults
    hf_token = hf_token or Config.HF_TOKEN
    model_name = model_name or Config.DIARIZATION_MODEL
    device = device or Config.DEVICE
    min_speakers = min_speakers or Config.MIN_SPEAKERS
    max_speakers = max_speakers or Config.MAX_SPEAKERS

    if not hf_token:
        raise ValueError("HF_TOKEN required for speaker diarization. Set in .env file.")

    logger.info(f"Performing speaker diarization on {audio_path.name}")
    logger.debug(f"Model: {model_name}, Device: {device}")

    try:
        # Load diarization pipeline
        logger.info("Loading Pyannote pipeline...")
        pipeline = Pipeline.from_pretrained(
            model_name,
            use_auth_token=hf_token
        )

        # Move to GPU if available
        if device == "cuda":
            pipeline = pipeline.to(torch.device("cuda"))

        # Perform diarization
        logger.info("Running diarization...")
        diarization_params = {}
        if min_speakers is not None:
            diarization_params['min_speakers'] = min_speakers
        if max_speakers is not None:
            diarization_params['max_speakers'] = max_speakers

        diarization = pipeline(str(audio_path), **diarization_params)

        # Extract speaker segments
        speaker_segments = []
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            speaker_segments.append({
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "speaker": speaker
            })

        # Count unique speakers
        unique_speakers = set(seg["speaker"] for seg in speaker_segments)

        logger.info(f"✓ Diarization complete: {len(unique_speakers)} speakers, {len(speaker_segments)} segments")

        # Cleanup
        if device == "cuda":
            torch.cuda.empty_cache()

        output = {
            "file": audio_path.name,
            "num_speakers": len(unique_speakers),
            "speakers": sorted(list(unique_speakers)),
            "segments": speaker_segments
        }

        return output

    except Exception as e:
        logger.error(f"Diarization failed: {str(e)}")
        raise
