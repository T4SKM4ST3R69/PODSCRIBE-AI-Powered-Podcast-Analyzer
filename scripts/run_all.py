"""
Master script to run all batch processing steps in sequence
"""
import sys
import os
from pathlib import Path
import time
import subprocess

# Add project root to path
SCRIPT_DIR = Path(__file__).parent if '__file__' in globals() else Path.cwd()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import setup_logger

logger = setup_logger()

def run_script(script_path: Path, description: str) -> bool:
    """
    Run a script as a subprocess and return success status

    Args:
        script_path: Path to script file
        description: Human-readable description

    Returns:
        True if successful, False otherwise
    """
    logger.info("\n" + "=" * 70)
    logger.info(f"RUNNING: {description}")
    logger.info("=" * 70)

    try:
        # Run script as subprocess (fixes __file__ issue)
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(PROJECT_ROOT),
            capture_output=False,
            text=True
        )

        if result.returncode == 0:
            logger.info(f"[SUCCESS] {description}")
            return True
        else:
            logger.error(f"[FAILED] {description} (exit code: {result.returncode})")
            return False

    except Exception as e:
        logger.error(f"[ERROR] Script execution failed: {str(e)}")
        return False

def main():
    """
    Run all processing scripts in sequence
    """
    logger.info("=" * 70)
    logger.info("PODSCRIBE - FULL PIPELINE EXECUTION")
    logger.info("=" * 70)

    print("\nThis will run all processing steps:")
    print("  0. Convert videos to MP3")
    print("  1. Transcribe audio with WhisperX + Pyannote")
    print("  2. Build vector database with ChromaDB")
    print("  3. Generate episode summaries with Groq")
    print("\n[WARNING] This may take several hours depending on your data!")

    confirm = input("\nContinue with full pipeline? (yes/no): ").lower().strip()

    if confirm not in ['yes', 'y']:
        logger.info("Cancelled by user")
        sys.exit(0)

    pipeline_start = time.time()

    # Define pipeline steps
    steps = [
        ("00_convert_videos.py", "Step 0: Video to MP3 Conversion"),
        ("01_batch_transcribe.py", "Step 1: Audio Transcription"),
        ("02_build_vector_db.py", "Step 2: Vector Database Building"),
        ("03_generate_summaries.py", "Step 3: Summary Generation")
    ]

    results = {}

    # Run each step
    for script_name, description in steps:
        script_path = SCRIPT_DIR / script_name

        if not script_path.exists():
            logger.error(f"[ERROR] Script not found: {script_path}")
            results[description] = False
            break

        success = run_script(script_path, description)
        results[description] = success

        if not success:
            logger.error(f"\n[FAILED] Pipeline failed at: {description}")
            logger.error("Fix the errors and run again, or run individual scripts")
            break

    # Final summary
    pipeline_elapsed = time.time() - pipeline_start

    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 70)

    for step, success in results.items():
        status = "[SUCCESS]" if success else "[FAILED]"
        logger.info(f"{status}: {step}")

    logger.info(f"\n[TIME] Total pipeline time: {pipeline_elapsed/60:.1f} minutes")

    if all(results.values()):
        logger.info("\n[COMPLETE] PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("\nYour PodScribe system is ready!")
        logger.info("Next step: Run the Streamlit app with: streamlit run streamlit_app.py")
    else:
        logger.error("\n[FAILED] PIPELINE FAILED")
        logger.error("Please check the logs and fix errors before proceeding")
        sys.exit(1)

if __name__ == "__main__":
    main()
