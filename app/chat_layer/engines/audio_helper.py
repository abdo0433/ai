"""
Safe Audio Validation Helper
=============================
Helper functions for safely handling audio data in Streamlit.
"""

import numpy as np
from typing import Optional, Tuple, Any


def is_valid_audio_data(audio_data: Any) -> bool:
    """
    Safely check if audio data is valid for processing.
    
    Args:
        audio_data: Audio data to validate (can be None, array, etc.)
        
    Returns:
        True if valid numpy array with data, False otherwise
    """
    # Check if None
    if audio_data is None:
        return False
    
    # Check if numpy array
    if not isinstance(audio_data, np.ndarray):
        return False
    
    # Check if empty
    if audio_data.size == 0:
        return False
    
    # Check if all zeros (silence)
    if np.all(audio_data == 0):
        return False
    
    return True


def safe_extract_audio_features(audio_extractor, audio_data: Any) -> Optional[dict]:
    """
    Safely extract audio features with validation.
    
    Args:
        audio_extractor: AudioFeatureExtractor instance
        audio_data: Audio data to process
        
    Returns:
        Features dict if successful, None otherwise
    """
    if not is_valid_audio_data(audio_data):
        return None
    
    try:
        features = audio_extractor.extract_features(audio_data)
        return features if features else None
    except Exception as e:
        print(f"⚠️  Audio feature extraction failed: {e}")
        return None


def unpack_stt_result(stt_result: Any) -> Tuple[Optional[str], Optional[np.ndarray]]:
    """
    Safely unpack STT engine result.
    
    Args:
        stt_result: Result from STT engine (can be str or tuple)
        
    Returns:
        Tuple of (transcribed_text, audio_data)
    """
    # Handle tuple result (text, audio)
    if isinstance(stt_result, tuple):
        if len(stt_result) == 2:
            text, audio = stt_result
            return (text, audio)
        elif len(stt_result) > 0:
            return (stt_result[0], None)
        else:
            return (None, None)
    
    # Handle string result (text only)
    if isinstance(stt_result, str):
        return (stt_result, None)
    
    # Unknown format
    return (None, None)


# Example usage in Streamlit:
"""
from utils.audio_helpers import safe_extract_audio_features, unpack_stt_result

# In your Streamlit code:
stt_result = stt_engine.listen_and_transcribe(return_audio=True)
transcribed_text, audio_data = unpack_stt_result(stt_result)

if transcribed_text:
    st.success(f"Recorded: {transcribed_text}")
    
    # Safely extract features
    audio_features = safe_extract_audio_features(audio_extractor, audio_data)
    
    if audio_features:
        st.info(f"Audio quality: {audio_features['overall_audio_quality']}/100")
    else:
        st.warning("Could not analyze audio")
"""