"""
Merge transcription and diarization results with ACCURATE timestamps
"""
from typing import Dict, List
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import setup_logger
from src.utils.helpers import save_json

logger = setup_logger()


def find_speaker_for_segment(segment_start: float, segment_end: float,
                              speaker_segments: List[Dict]) -> str:
    """
    Find speaker for a segment using OVERLAP-BASED matching

    Args:
        segment_start: Start time of transcript segment
        segment_end: End time of transcript segment
        speaker_segments: List of speaker segments from diarization

    Returns:
        Speaker label with highest overlap
    """
    max_overlap = 0
    best_speaker = "UNKNOWN"

    for seg in speaker_segments:
        # Calculate overlap between transcript segment and speaker segment
        overlap_start = max(segment_start, seg["start"])
        overlap_end = min(segment_end, seg["end"])
        overlap = max(0, overlap_end - overlap_start)

        if overlap > max_overlap:
            max_overlap = overlap
            best_speaker = seg["speaker"]

    return best_speaker if max_overlap > 0 else "UNKNOWN"


def merge_transcription_and_diarization(transcription: Dict,
                                        diarization: Dict,
                                        output_path: Path = None) -> Dict:
    """
    Merge transcription with speaker diarization using ACCURATE alignment

    Args:
        transcription: Output from transcribe_audio()
        diarization: Output from diarize_audio()
        output_path: Optional path to save merged JSON

    Returns:
        Dictionary with merged transcript and accurate speaker labels
    """
    logger.info("Merging transcription and diarization with accurate timestamps...")

    speaker_segments = diarization["segments"]
    transcript_segments = transcription["segments"]

    # Merge with overlap-based speaker assignment
    merged_segments = []

    for segment in transcript_segments:
        # Use overlap-based matching instead of point-in-time
        start_time = segment["start"]
        end_time = segment["end"]

        speaker = find_speaker_for_segment(start_time, end_time, speaker_segments)

        merged_segment = {
            "start": round(start_time, 3),  # Keep 3 decimal places for accuracy
            "end": round(end_time, 3),
            "text": segment["text"].strip(),
            "speaker": speaker
        }

        # Preserve word-level timestamps if available
        if "words" in segment:
            words_with_speakers = []
            for word in segment["words"]:
                word_start = word.get("start", start_time)
                word_end = word.get("end", end_time)
                word_speaker = find_speaker_for_segment(word_start, word_end, speaker_segments)

                words_with_speakers.append({
                    "word": word["word"],
                    "start": round(word_start, 3),
                    "end": round(word_end, 3),
                    "speaker": word_speaker
                })

            merged_segment["words"] = words_with_speakers

        merged_segments.append(merged_segment)

    # Create final output
    output = {
        "file": transcription["file"],
        "language": transcription["language"],
        "num_speakers": diarization["num_speakers"],
        "speakers": diarization["speakers"],
        "segments": merged_segments
    }

    logger.info(f"✓ Merge complete: {len(merged_segments)} segments with accurate timestamps")

    # Save if output path provided
    if output_path:
        save_json(output, Path(output_path))
        logger.info(f"Saved merged transcript to {output_path}")

    return output
