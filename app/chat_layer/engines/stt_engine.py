"""
Speech-to-Text Engine using OpenAI Whisper (Local Model)
========================================================
Responsibilities:
- Record audio from microphone (Local testing)
- Convert audio to text using Whisper
- Transcribe audio files directly (Web API ready)
- Return clean transcribed text ONLY
"""

import whisper
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import tempfile
import os
from typing import Optional


class STTEngine:
    """
    Stateless Speech-to-Text engine using local Whisper model.
    """
    
    def __init__(self, model_path: str = "base", language: str = "en"):
        """
        Initialize Whisper model.
        
        Args:
            model_path: Whisper model size or path to local model
                       Options: "tiny", "base", "small", "medium", "large"
            language: Fixed language code (default: "en" for English)
        """
        print(f"Loading Whisper model: {model_path}")
        # Disable fp16 for CPU compatibility
        self.model = whisper.load_model(model_path, device="cpu")
        self.language = language
        self.sample_rate = 16000  # Whisper expects 16kHz audio
        
    def record_audio(
        self,
        max_duration: int = 60,
        silence_threshold: float = 0.015,
        silence_duration: float = 3.5,
        min_speech_duration: float = 1.0
    ) -> Optional[np.ndarray]:
        """
        Record audio from microphone with automatic silence detection.
        """
        print("🎤 Recording... (speak now - you have up to 1 minute)")
        
        audio_chunks = []
        silence_counter = 0
        silence_frames = int(silence_duration * self.sample_rate / 1024)
        min_speech_frames = int(min_speech_duration * self.sample_rate / 1024)
        frames_recorded = 0
        has_speech = False
        
        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32',
                blocksize=1024
            ) as stream:
                
                for _ in range(int(max_duration * self.sample_rate / 1024)):
                    audio_chunk, _ = stream.read(1024)
                    audio_chunks.append(audio_chunk)
                    frames_recorded += 1
                    
                    # Check if current chunk is silence
                    volume = np.abs(audio_chunk).mean()
                    
                    # Detect if there's speech
                    if volume >= silence_threshold:
                        has_speech = True
                        silence_counter = 0
                    else:
                        silence_counter += 1
                    
                    # Only stop on silence if we've recorded minimum speech
                    if (silence_counter >= silence_frames and 
                        frames_recorded >= min_speech_frames and 
                        has_speech):
                        print("✓ Silence detected, stopping recording")
                        break
                        
        except Exception as e:
            print(f"❌ Recording error: {e}")
            return None
        
        # Combine all chunks
        audio_data = np.concatenate(audio_chunks, axis=0)
        
        # Check if audio is too short or empty
        if len(audio_data) < self.sample_rate * 0.5:  # Less than 0.5 seconds
            print("⚠️  Recording too short or empty")
            return None
            
        print(f"✓ Recording complete ({len(audio_data) / self.sample_rate:.1f}s)")
        return audio_data
    
    def transcribe(self, audio_data: np.ndarray) -> str:
        """
        Transcribe audio data to text using Whisper.
        """
        # Create temporary WAV file for Whisper
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
            
        try:
            # Save audio to temporary file
            wav.write(temp_path, self.sample_rate, audio_data)
            
            # Transcribe with Whisper
            print("🔄 Transcribing...")
            result = self.model.transcribe(
                temp_path,
                language=self.language,  # Fixed language
                fp16=False,  # Disable fp16 for CPU
                verbose=False
            )
            
            # Extract and clean text
            text = result["text"].strip()
            
            # Filter out hallucinated/empty results
            if not text or len(text) < 2:
                return ""
                
            print(f"✓ Transcription: {text}")
            return text
            
        except Exception as e:
            print(f"❌ Transcription error: {e}")
            return ""
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    # =================================================================
    # 🔥 الإضافة الجديدة الخاصة بـ FastAPI (لتحويل الملفات الجاهزة)
    # =================================================================
    def transcribe_file(self, file_path: str) -> str:
        """
        Transcribe an existing audio file directly.
        (Perfect for FastAPI web uploads from the frontend)

        IMPORTANT: Flutter Web's AudioRecorder with path='' records a browser
        blob that is actually WebM/Opus regardless of the encoder hint (wav).
        We detect that and convert to proper 16 kHz mono WAV before Whisper.
        """
        converted_path = None
        try:
            print(f"🔄 Transcribing uploaded file: {file_path}")

            # ── Detect real format from file magic bytes ──────────────────
            with open(file_path, "rb") as f:
                header = f.read(16)

            is_webm_or_ogg = (
                header[:4] == b"\x1a\x45\xdf\xa3"   # EBML magic → WebM/MKV
                or header[:4] == b"OggS"              # Ogg container
            )
            is_proper_wav  = header[:4] == b"RIFF" and header[8:12] == b"WAVE"

            needs_conversion = is_webm_or_ogg or (
                not is_proper_wav and not header[:3] == b"ID3"  # not MP3 either
            )

            target_path = file_path  # default: pass original to Whisper

            if needs_conversion:
                print("⚠️  Non-WAV audio detected (likely WebM/Opus from browser). Converting…")
                converted_path = file_path + "_converted.wav"
                import subprocess, shutil

                if shutil.which("ffmpeg"):
                    # ffmpeg: decode anything → 16 kHz mono PCM WAV
                    ret = subprocess.run(
                        [
                            "ffmpeg", "-y",
                            "-i", file_path,
                            "-ar", "16000",   # 16 kHz — Whisper's native rate
                            "-ac", "1",       # mono
                            "-c:a", "pcm_s16le",
                            converted_path,
                        ],
                        capture_output=True, timeout=30,
                    )
                    if ret.returncode == 0:
                        target_path = converted_path
                        print("✅ Converted via ffmpeg")
                    else:
                        print(f"⚠️  ffmpeg failed: {ret.stderr.decode()[:200]}")
                        # Fall through — try librosa as backup
                else:
                    print("⚠️  ffmpeg not found — trying librosa fallback")

                # librosa fallback (handles WebM/Ogg via soundfile/audioread)
                if target_path == file_path:
                    try:
                        import librosa
                        import soundfile as sf
                        audio_data, _ = librosa.load(file_path, sr=16000, mono=True)
                        sf.write(converted_path, audio_data, 16000, subtype="PCM_16")
                        target_path = converted_path
                        print("✅ Converted via librosa")
                    except Exception as lb_err:
                        print(f"⚠️  librosa fallback failed: {lb_err}")
                        # Last resort: let Whisper try anyway

            result = self.model.transcribe(
                target_path,
                language=self.language,
                fp16=False,
                verbose=False,
            )
            text = result["text"].strip()

            # Filter out hallucinated/empty results
            if not text or len(text) < 2:
                return ""

            print(f"✓ Transcription: {text}")
            return text

        except Exception as e:
            print(f"❌ Transcription error: {e}")
            return ""

        finally:
            # Clean up converted temp file (original is managed by the caller)
            if converted_path and os.path.exists(converted_path):
                os.remove(converted_path)
    
    def listen_and_transcribe(
        self,
        max_duration: int = 60,
        silence_threshold: float = 0.015,
        silence_duration: float = 3.5,
        return_audio: bool = False
    ) -> tuple:
        """
        Combined method: Record audio and transcribe in one call.
        """
        audio_data = self.record_audio(
            max_duration=max_duration,
            silence_threshold=silence_threshold,
            silence_duration=silence_duration
        )
        
        if audio_data is None:
            return ("", None) if return_audio else ""
        
        text = self.transcribe(audio_data)
        
        if return_audio:
            return text, audio_data
        else:
            return text


# Singleton instance (optional, for convenience)
_stt_instance: Optional[STTEngine] = None

def get_stt_engine(model_path: str = "small", language: str = "en") -> STTEngine:
    """
    Get or create STT engine instance.
    """
    global _stt_instance
    if _stt_instance is None:
        _stt_instance = STTEngine(model_path=model_path, language=language)
    return _stt_instance


# Simple function interface
def transcribe_from_microphone(
    model_path: str = "small",
    language: str = "en",
    max_duration: int = 60
) -> str:
    """
    Simple function to record and transcribe user speech.
    """
    engine = get_stt_engine(model_path=model_path, language=language)
    return engine.listen_and_transcribe(max_duration=max_duration)