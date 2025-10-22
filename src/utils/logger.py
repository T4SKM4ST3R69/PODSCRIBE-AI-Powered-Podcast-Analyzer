"""
Logging configuration for PodScribe (Windows UTF-8 compatible)
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from .config import Config

def setup_logger(name: str = "podscribe", level: int = logging.INFO):
    """
    Setup application logger with file and console handlers
    (Windows UTF-8 compatible)

    Args:
        name: Logger name
        level: Logging level

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Create formatters (without emojis for Windows compatibility)
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    simple_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )

    # File handler (detailed logs) - UTF-8 encoding
    log_file = Config.LOGS_DIR / f"podscribe_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)

    # Console handler (simple logs) - UTF-8 encoding for Windows
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)

    # Force UTF-8 encoding on Windows console
    if sys.platform == 'win32':
        import codecs
        if hasattr(sys.stdout, 'buffer'):
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'replace')
        if hasattr(sys.stderr, 'buffer'):
            sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'replace')

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
