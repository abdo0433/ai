"""
Audio Subsystem Configuration
==============================
Central configuration for STT and TTS engines.
"""

import os

# Whisper STT Configuration
# ==========================
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
"""
Available Whisper models (in order of size/accuracy):
- tiny: Fastest, least accurate (~40MB)
- base: Good balance (~75MB) - RECOMMENDED
- small: Better accuracy (~250MB)
- medium: High accuracy (~770MB)
- large: Best accuracy (~1.5GB)
"""

WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "en")
"""Fixed language for transcription (English by default)"""

WHISPER_DEVICE = "cpu"
"""Force CPU usage (GPU not required for base model)"""

# Recording Configuration
# =======================
AUDIO_SAMPLE_RATE = 16000
"""Sample rate in Hz (Whisper expects 16kHz)"""

MAX_RECORDING_DURATION = int(os.getenv("MAX_RECORDING_DURATION", "60"))
"""Maximum recording duration in seconds (1 minute for interview answers)"""

SILENCE_THRESHOLD = float(os.getenv("SILENCE_THRESHOLD", "0.015"))
"""Volume threshold for silence detection (0.0 - 1.0). Higher = less sensitive."""

SILENCE_DURATION = float(os.getenv("SILENCE_DURATION", "3.5"))
"""Seconds of silence before stopping recording (3.5s to allow thinking pauses)"""

# TTS Configuration
# =================
TTS_RATE = int(os.getenv("TTS_RATE", "150"))
"""Speech rate in words per minute (default: 150)"""

TTS_VOLUME = float(os.getenv("TTS_VOLUME", "0.9"))
"""Volume level (0.0 to 1.0)"""

TTS_VOICE_ID = os.getenv("TTS_VOICE_ID", None)
"""Voice ID (None for default system voice)"""

# Audio Processing
# ================
MIN_AUDIO_LENGTH = 0.5
"""Minimum audio length in seconds to process"""

ENABLE_NOISE_REDUCTION = os.getenv("ENABLE_NOISE_REDUCTION", "false").lower() == "true"
"""Enable basic noise reduction (experimental)"""

# Debug/Logging
# =============
AUDIO_DEBUG_MODE = os.getenv("AUDIO_DEBUG_MODE", "false").lower() == "true"
"""Enable detailed audio debug logging"""

SAVE_RECORDINGS = os.getenv("SAVE_RECORDINGS", "false").lower() == "true"
"""Save recordings to disk for debugging"""

RECORDINGS_DIR = os.getenv("RECORDINGS_DIR", "./recordings")
"""Directory to save recordings (if SAVE_RECORDINGS is True)"""