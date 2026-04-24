"""
Audio Feature Extraction Engine
=================================
Extracts audio features for comprehensive interview evaluation:
- Tone & Pitch (حدة الصوت)
- Energy (قوة الصوت)
- Speaking Rate (سرعة الكلام)
- Silence Ratio (نسبة السكتات)
- MFCCs (Mel-frequency cepstral coefficients)
- Pitch Variability (تغير النبرة)
"""

import numpy as np
import librosa
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class AudioFeatureExtractor:
    """
    Extract audio features for interview performance analysis.
    """
    
    def __init__(self, sample_rate: int = 16000):
        """
        Initialize feature extractor.
        
        Args:
            sample_rate: Audio sample rate (default: 16000 Hz for Whisper)
        """
        self.sample_rate = sample_rate
    
    def extract_features(self, audio_data: np.ndarray) -> Dict:
        """
        Extract comprehensive audio features from audio data.
        
        Args:
            audio_data: Audio samples as numpy array
            
        Returns:
            Dictionary containing all extracted features
        """
        # Proper numpy array validation
        if audio_data is None:
            return self._empty_features()
        
        if not isinstance(audio_data, np.ndarray):
            return self._empty_features()
        
        if audio_data.size == 0:
            return self._empty_features()
        
        # Normalize audio
        audio = audio_data.flatten()
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        
        features = {}
        
        try:
            # 1. Pitch Features (حدة الصوت)
            features.update(self._extract_pitch_features(audio))
            
            # 2. Energy Features (قوة الصوت)
            features.update(self._extract_energy_features(audio))
            
            # 3. Speaking Rate Features (سرعة الكلام)
            features.update(self._extract_speaking_rate_features(audio))
            
            # 4. Silence Features (نسبة السكتات)
            features.update(self._extract_silence_features(audio))
            
            # 5. MFCCs (خصائص طيفية)
            features.update(self._extract_mfcc_features(audio))
            
            # 6. Additional Spectral Features
            features.update(self._extract_spectral_features(audio))
            
            # 7. Overall Quality Score
            features['overall_audio_quality'] = self._calculate_quality_score(features)
            
        except Exception as e:
            print(f"⚠️  Feature extraction error: {e}")
            features = self._empty_features()
        
        return features
    
    def _extract_pitch_features(self, audio: np.ndarray) -> Dict:
        """
        Extract pitch-related features.
        
        Returns:
            pitch_mean: متوسط حدة الصوت
            pitch_std: تغير حدة الصوت (Variability)
            pitch_min: أقل حدة
            pitch_max: أعلى حدة
            pitch_range: المدى الصوتي
        """
        try:
            # Extract pitch using librosa's pyin (probabilistic YIN)
            f0, voiced_flag, _ = librosa.pyin(
                audio,
                fmin=librosa.note_to_hz('C2'),  # ~65 Hz (male voice)
                fmax=librosa.note_to_hz('C7'),  # ~2093 Hz (female voice)
                sr=self.sample_rate
            )
            
            # Filter out unvoiced frames (NaN values)
            f0_voiced = f0[~np.isnan(f0)]
            
            if len(f0_voiced) > 0:
                return {
                    'pitch_mean': float(np.mean(f0_voiced)),
                    'pitch_std': float(np.std(f0_voiced)),
                    'pitch_min': float(np.min(f0_voiced)),
                    'pitch_max': float(np.max(f0_voiced)),
                    'pitch_range': float(np.max(f0_voiced) - np.min(f0_voiced)),
                    'pitch_variability': float(np.std(f0_voiced) / (np.mean(f0_voiced) + 1e-6))
                }
            else:
                return self._empty_pitch_features()
                
        except Exception as e:
            print(f"⚠️  Pitch extraction error: {e}")
            return self._empty_pitch_features()
    
    def _extract_energy_features(self, audio: np.ndarray) -> Dict:
        """
        Extract energy-related features.
        
        Returns:
            energy_mean: متوسط قوة الصوت
            energy_std: تغير قوة الصوت
            energy_max: أقصى طاقة
        """
        try:
            # RMS Energy (Root Mean Square)
            rms = librosa.feature.rms(y=audio)[0]
            
            # Convert to dB scale
            rms_db = librosa.amplitude_to_db(rms, ref=np.max)
            
            return {
                'energy_mean': float(np.mean(rms)),
                'energy_std': float(np.std(rms)),
                'energy_max': float(np.max(rms)),
                'energy_db_mean': float(np.mean(rms_db)),
                'energy_db_range': float(np.max(rms_db) - np.min(rms_db))
            }
        except Exception as e:
            print(f"⚠️  Energy extraction error: {e}")
            return {
                'energy_mean': 0.0,
                'energy_std': 0.0,
                'energy_max': 0.0,
                'energy_db_mean': 0.0,
                'energy_db_range': 0.0
            }
    
    def _extract_speaking_rate_features(self, audio: np.ndarray) -> Dict:
        """
        Extract speaking rate features.
        
        Returns:
            speaking_rate: عدد الكلمات/المقاطع في الدقيقة (تقديري)
            tempo: السرعة الإيقاعية
        """
        try:
            # Onset detection (بداية الأصوات)
            onset_env = librosa.onset.onset_strength(y=audio, sr=self.sample_rate)
            tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=self.sample_rate)
            
            # Estimate syllables per second
            onset_frames = librosa.onset.onset_detect(
                onset_envelope=onset_env,
                sr=self.sample_rate
            )
            
            duration = len(audio) / self.sample_rate
            syllables_per_second = len(onset_frames) / duration if duration > 0 else 0
            
            return {
                'tempo': float(tempo),
                'syllables_per_second': float(syllables_per_second),
                'speaking_rate_estimate': float(syllables_per_second * 60),  # per minute
                'total_onsets': int(len(onset_frames))
            }
        except Exception as e:
            print(f"⚠️  Speaking rate error: {e}")
            return {
                'tempo': 0.0,
                'syllables_per_second': 0.0,
                'speaking_rate_estimate': 0.0,
                'total_onsets': 0
            }
    
    def _extract_silence_features(self, audio: np.ndarray) -> Dict:
        """
        Extract silence-related features.
        
        Returns:
            silence_ratio: نسبة السكتات من الإجمالي
            num_pauses: عدد الوقفات
            avg_pause_duration: متوسط طول الوقفة
        """
        try:
            # Detect silent intervals
            intervals = librosa.effects.split(
                audio,
                top_db=30,  # Threshold for silence detection
                frame_length=2048,
                hop_length=512
            )
            
            total_duration = len(audio) / self.sample_rate
            
            # Calculate voiced duration
            voiced_duration = 0
            for start, end in intervals:
                voiced_duration += (end - start) / self.sample_rate
            
            silence_duration = total_duration - voiced_duration
            silence_ratio = silence_duration / total_duration if total_duration > 0 else 0
            
            # Calculate pauses (gaps between voiced segments)
            num_pauses = len(intervals) - 1 if len(intervals) > 1 else 0
            
            pause_durations = []
            for i in range(len(intervals) - 1):
                pause_start = intervals[i][1]
                pause_end = intervals[i + 1][0]
                pause_duration = (pause_end - pause_start) / self.sample_rate
                if pause_duration > 0.1:  # Only count pauses > 100ms
                    pause_durations.append(pause_duration)
            
            avg_pause_duration = np.mean(pause_durations) if pause_durations else 0
            
            return {
                'silence_ratio': float(silence_ratio),
                'voiced_ratio': float(1 - silence_ratio),
                'num_pauses': int(len(pause_durations)),
                'avg_pause_duration': float(avg_pause_duration),
                'max_pause_duration': float(max(pause_durations)) if pause_durations else 0,
                'total_voiced_duration': float(voiced_duration)
            }
        except Exception as e:
            print(f"⚠️  Silence detection error: {e}")
            return {
                'silence_ratio': 0.0,
                'voiced_ratio': 1.0,
                'num_pauses': 0,
                'avg_pause_duration': 0.0,
                'max_pause_duration': 0.0,
                'total_voiced_duration': 0.0
            }
    
    def _extract_mfcc_features(self, audio: np.ndarray) -> Dict:
        """
        Extract MFCC features (Mel-frequency cepstral coefficients).
        خصائص طيفية مهمة جداً للتعرف على الكلام.
        
        Returns:
            mfcc_mean: متوسط MFCCs
            mfcc_std: تغير MFCCs
        """
        try:
            # Extract MFCCs (13 coefficients is standard)
            mfccs = librosa.feature.mfcc(
                y=audio,
                sr=self.sample_rate,
                n_mfcc=13
            )
            
            # Calculate statistics for each coefficient
            mfcc_mean = np.mean(mfccs, axis=1)
            mfcc_std = np.std(mfccs, axis=1)
            
            return {
                'mfcc_mean': mfcc_mean.tolist(),
                'mfcc_std': mfcc_std.tolist(),
                'mfcc_overall_mean': float(np.mean(mfcc_mean)),
                'mfcc_overall_std': float(np.mean(mfcc_std))
            }
        except Exception as e:
            print(f"⚠️  MFCC extraction error: {e}")
            return {
                'mfcc_mean': [0.0] * 13,
                'mfcc_std': [0.0] * 13,
                'mfcc_overall_mean': 0.0,
                'mfcc_overall_std': 0.0
            }
    
    def _extract_spectral_features(self, audio: np.ndarray) -> Dict:
        """
        Extract additional spectral features.
        
        Returns:
            spectral_centroid: مركز الطيف الترددي
            spectral_bandwidth: عرض النطاق الترددي
            spectral_rolloff: نقطة الانقطاع الطيفي
            zero_crossing_rate: معدل تقاطع الصفر
        """
        try:
            # Spectral centroid (brightness)
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
            
            # Spectral bandwidth
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate)[0]
            
            # Spectral rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)[0]
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            
            return {
                'spectral_centroid_mean': float(np.mean(spectral_centroid)),
                'spectral_centroid_std': float(np.std(spectral_centroid)),
                'spectral_bandwidth_mean': float(np.mean(spectral_bandwidth)),
                'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
                'zero_crossing_rate_mean': float(np.mean(zcr)),
                'zero_crossing_rate_std': float(np.std(zcr))
            }
        except Exception as e:
            print(f"⚠️  Spectral features error: {e}")
            return {
                'spectral_centroid_mean': 0.0,
                'spectral_centroid_std': 0.0,
                'spectral_bandwidth_mean': 0.0,
                'spectral_rolloff_mean': 0.0,
                'zero_crossing_rate_mean': 0.0,
                'zero_crossing_rate_std': 0.0
            }
    
    def _calculate_quality_score(self, features: Dict) -> float:
        """
        Calculate overall audio quality score based on features.
        
        Returns:
            Score from 0 to 100
        """
        try:
            score = 100.0
            
            # 1. Pitch variability (good: moderate variation)
            pitch_var = features.get('pitch_variability', 0)
            if pitch_var < 0.05:  # Too monotone
                score -= 10
            elif pitch_var > 0.3:  # Too variable
                score -= 5
            
            # 2. Energy (good: consistent, not too quiet)
            energy = features.get('energy_mean', 0)
            if energy < 0.05:  # Too quiet
                score -= 15
            elif energy > 0.5:  # Too loud
                score -= 5
            
            # 3. Speaking rate (good: 120-180 syllables/min)
            speaking_rate = features.get('speaking_rate_estimate', 0)
            if speaking_rate < 60:  # Too slow
                score -= 10
            elif speaking_rate > 240:  # Too fast
                score -= 10
            
            # 4. Silence ratio (good: 10-30%)
            silence_ratio = features.get('silence_ratio', 0)
            if silence_ratio < 0.1:  # Too little pause
                score -= 5
            elif silence_ratio > 0.5:  # Too much silence
                score -= 15
            
            # 5. Pause management (good: short, natural pauses)
            avg_pause = features.get('avg_pause_duration', 0)
            if avg_pause > 2.0:  # Long pauses
                score -= 10
            
            return max(0.0, min(100.0, score))
            
        except Exception as e:
            print(f"⚠️  Quality score error: {e}")
            return 50.0
    
    def _empty_features(self) -> Dict:
        """Return empty feature dict"""
        return {
            **self._empty_pitch_features(),
            'energy_mean': 0.0,
            'energy_std': 0.0,
            'energy_max': 0.0,
            'tempo': 0.0,
            'syllables_per_second': 0.0,
            'speaking_rate_estimate': 0.0,
            'silence_ratio': 0.0,
            'num_pauses': 0,
            'mfcc_mean': [0.0] * 13,
            'mfcc_std': [0.0] * 13,
            'overall_audio_quality': 0.0
        }
    
    def _empty_pitch_features(self) -> Dict:
        """Return empty pitch features"""
        return {
            'pitch_mean': 0.0,
            'pitch_std': 0.0,
            'pitch_min': 0.0,
            'pitch_max': 0.0,
            'pitch_range': 0.0,
            'pitch_variability': 0.0
        }
    
    def get_human_readable_analysis(self, features: Dict) -> Dict:
        """
        Convert features to human-readable analysis.
        
        Returns:
            Dictionary with interpretations in Arabic and English
        """
        analysis = {}
        
        # Pitch Analysis
        pitch_mean = features.get('pitch_mean', 0)
        pitch_var = features.get('pitch_variability', 0)
        
        if pitch_mean > 0:
            if pitch_mean < 120:
                analysis['pitch_level'] = "Deep voice / صوت غليظ"
            elif pitch_mean < 200:
                analysis['pitch_level'] = "Normal voice / صوت طبيعي"
            else:
                analysis['pitch_level'] = "High voice / صوت حاد"
        
        if pitch_var < 0.05:
            analysis['pitch_variation'] = "Monotone / رتيب"
        elif pitch_var < 0.15:
            analysis['pitch_variation'] = "Good variation / تنوع جيد"
        else:
            analysis['pitch_variation'] = "Very expressive / تعبيري جداً"
        
        # Energy Analysis
        energy = features.get('energy_mean', 0)
        if energy < 0.1:
            analysis['volume_level'] = "Too quiet / هادئ جداً"
        elif energy < 0.3:
            analysis['volume_level'] = "Good volume / صوت جيد"
        else:
            analysis['volume_level'] = "Loud / عالي"
        
        # Speaking Rate Analysis
        rate = features.get('speaking_rate_estimate', 0)
        if rate < 90:
            analysis['speaking_speed'] = "Slow / بطيء"
        elif rate < 150:
            analysis['speaking_speed'] = "Normal / طبيعي"
        elif rate < 200:
            analysis['speaking_speed'] = "Fast / سريع"
        else:
            analysis['speaking_speed'] = "Very fast / سريع جداً"
        
        # Silence Analysis
        silence = features.get('silence_ratio', 0)
        if silence < 0.1:
            analysis['pause_usage'] = "Few pauses / وقفات قليلة"
        elif silence < 0.3:
            analysis['pause_usage'] = "Good pauses / وقفات جيدة"
        else:
            analysis['pause_usage'] = "Many pauses / وقفات كثيرة"
        
        # Overall Quality
        quality = features.get('overall_audio_quality', 0)
        if quality >= 80:
            analysis['overall'] = "Excellent / ممتاز"
        elif quality >= 60:
            analysis['overall'] = "Good / جيد"
        elif quality >= 40:
            analysis['overall'] = "Fair / مقبول"
        else:
            analysis['overall'] = "Needs improvement / يحتاج تحسين"
        
        return analysis


# Singleton instance
_extractor_instance: Optional[AudioFeatureExtractor] = None

def get_audio_extractor(sample_rate: int = 16000) -> AudioFeatureExtractor:
    """
    Get or create audio feature extractor instance.
    
    Args:
        sample_rate: Audio sample rate
        
    Returns:
        AudioFeatureExtractor instance
    """
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = AudioFeatureExtractor(sample_rate=sample_rate)
    return _extractor_instance


# Simple function interface
def extract_audio_features(audio_data: np.ndarray, sample_rate: int = 16000) -> Dict:
    """
    Simple function to extract audio features.
    
    Args:
        audio_data: Audio samples as numpy array
        sample_rate: Sample rate
        
    Returns:
        Dictionary of features
    """
    extractor = get_audio_extractor(sample_rate=sample_rate)
    return extractor.extract_features(audio_data)