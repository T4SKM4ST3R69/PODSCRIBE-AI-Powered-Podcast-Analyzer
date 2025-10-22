"""
Centralized configuration management for PodScribe
"""
import os
from pathlib import Path
from dotenv import load_dotenv
import torch

# Load environment variables
load_dotenv()

class Config:
    """Application configuration"""

    # Project paths
    BASE_DIR = Path(__file__).parent.parent.parent
    DATA_DIR = BASE_DIR / "data"
    RAW_AUDIO_DIR = DATA_DIR / "raw_audio"
    CONVERTED_AUDIO_DIR = DATA_DIR / "converted_audio"
    TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
    SUMMARIES_DIR = DATA_DIR / "summaries"
    CHROMA_DB_DIR = BASE_DIR / "chroma_db"
    LOGS_DIR = BASE_DIR / "logs"

    # Ensure directories exist
    for directory in [RAW_AUDIO_DIR, CONVERTED_AUDIO_DIR, TRANSCRIPTS_DIR,
                      SUMMARIES_DIR, CHROMA_DB_DIR, LOGS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    # API Keys
    HF_TOKEN = os.getenv("HF_TOKEN")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Changed from OPENAI_API_KEY

    # Device configuration (CUDA for RTX 3050)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"

    # WhisperX configuration
    WHISPER_MODEL = "medium"
    WHISPER_BATCH_SIZE = 32  # Optimized for RTX 3050 6GB
    WHISPER_LANGUAGE = None  # Auto-detect, or set to "en" for English

    # Pyannote configuration
    DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
    MIN_SPEAKERS = None  # Auto-detect
    MAX_SPEAKERS = None  # Auto-detect

    # ChromaDB configuration
    COLLECTION_NAME = "podcast_transcripts"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    CHUNK_DURATION = 45  # seconds per chunk
    CHUNK_OVERLAP = 5    # seconds overlap between chunks

    # RAG configuration
    TOP_K_RESULTS = 5
    SIMILARITY_THRESHOLD = 0.0

    # LLM configuration (Groq)
    # Available models: llama-3.3-70b-versatile, llama-3.1-70b-versatile,
    # mixtral-8x7b-32768, gemma2-9b-it, llama3-70b-8192, llama3-8b-8192
    LLM_MODEL = "llama-3.3-70b-versatile"  # Changed from gpt-4o-mini
    LLM_TEMPERATURE = 0.3
    MAX_TOKENS = 8000  # Groq supports larger context windows

    @classmethod
    def validate(cls):
        """Validate configuration"""
        if not cls.HF_TOKEN:
            raise ValueError("HF_TOKEN not found in .env file")
        if not cls.GROQ_API_KEY:  # Changed from OPENAI_API_KEY
            print("Warning: GROQ_API_KEY not found. LLM features will be disabled.")
        if cls.DEVICE == "cpu":
            print("Warning: CUDA not available. Using CPU (slower performance).")
        return True

# Validate on import
Config.validate()
