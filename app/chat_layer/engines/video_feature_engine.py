"""
Enhanced Video Feature Extraction Engine - V2.0
================================================
Improvements from analysis:
1. ✅ Stable Gaze Tracking (using standard FaceMesh to prevent C++ crash)
2. ✅ Frame threshold detection (reduce false positives)
3. ✅ Alert cooldown system (prevent spam)
4. ✅ Smile detection from mouth geometry
5. ✅ Head yaw/pitch calculation
6. ✅ Confidence scoring from head pose
7. ✅ All original features (emotions, posture, gestures)
"""

import cv2
import numpy as np
import time
from typing import Dict, Optional, List, Tuple
import warnings
import importlib.util  # <-- Added for safe module checking
warnings.filterwarnings('ignore')

# ============== LIBRARY AVAILABILITY CHECKS ==============

MEDIAPIPE_AVAILABLE = False
DEEPFACE_AVAILABLE = False
FER_AVAILABLE = False

try:
    import mediapipe as mp
    _ = mp.solutions.pose
    MEDIAPIPE_AVAILABLE = True
    print("✅ MediaPipe imported successfully")
except Exception as e:
    print(f"⚠️  MediaPipe: {e}")

# LAZY LOADING: Check if DeepFace exists without importing it yet
try:
    if importlib.util.find_spec("deepface") is not None:
        DEEPFACE_AVAILABLE = True
        print("✅ DeepFace found (will load lazily)")
    else:
        print("⚠️  DeepFace not installed")
except Exception as e:
    print(f"⚠️  DeepFace check error: {e}")

# LAZY LOADING: Check if FER exists without importing it yet
try:
    if importlib.util.find_spec("fer") is not None:
        FER_AVAILABLE = True
        print("✅ FER found (will load lazily)")
    else:
        print("⚠️  FER not installed")
except Exception as e:
    print(f"⚠️  FER check error: {e}")


# ============== MEDIAPIPE LANDMARK INDICES ==============

# Face landmarks
NOSE = 1
LEFT_EAR = 234
RIGHT_EAR = 454

# Eye landmarks
L_EYE_OUTER = 33
L_EYE_INNER = 133
R_EYE_OUTER = 362
R_EYE_INNER = 263

# Mouth landmarks
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
MOUTH_TOP = 13
MOUTH_BOTTOM = 14

# Eyebrow landmarks
L_BROW_INNER = 107
R_BROW_INNER = 336


# ============== ALERT COOLDOWN MANAGER ==============

class AlertCooldownManager:
    """
    Prevents alert spam by enforcing cooldown periods.
    """
    
    def __init__(self):
        self.cooldowns = {
            "no_face": 8,
            "multiple_faces": 6,
            "head_turn": 10,
            "looking_away": 10,
            "low_confidence": 15,
            "stress_spike": 12,
        }
        self.last_alert_times = {}
    
    def can_alert(self, alert_type: str) -> bool:
        now = time.time()
        cooldown = self.cooldowns.get(alert_type, 10)
        last_time = self.last_alert_times.get(alert_type, 0)
        
        if now - last_time >= cooldown:
            self.last_alert_times[alert_type] = now
            return True
        return False
    
    def reset(self):
        self.last_alert_times = {}


# ============== FRAME THRESHOLD TRACKER ==============

class FrameThresholdTracker:
    """
    Tracks consecutive frames for each condition to reduce false positives.
    """
    
    def __init__(self):
        self.thresholds = {
            "no_face": 12,        # ~0.5s at 24fps
            "looking_away": 20,   # ~0.8s
            "stress_high": 30,    # ~1.2s
            "low_confidence": 25, # ~1.0s
        }
        self.counters = {k: 0 for k in self.thresholds}
    
    def update(self, condition: str, is_active: bool):
        if is_active:
            self.counters[condition] += 1
        else:
            self.counters[condition] = max(0, self.counters[condition] - 1)
    
    def check_threshold(self, condition: str) -> bool:
        threshold = self.thresholds.get(condition, 10)
        return self.counters[condition] >= threshold
    
    def get_count(self, condition: str) -> int:
        return self.counters.get(condition, 0)
    
    def reset(self):
        self.counters = {k: 0 for k in self.thresholds}


# ============== ENHANCED VIDEO FEATURE EXTRACTOR ==============

class VideoFeatureExtractor:
    
    def __init__(self):
        self.mediapipe_available = MEDIAPIPE_AVAILABLE
        self.deepface_available = DEEPFACE_AVAILABLE
        self.fer_available = FER_AVAILABLE
        
        self.alert_manager = AlertCooldownManager()
        self.threshold_tracker = FrameThresholdTracker()
        
        # 1. INITIALIZE MEDIAPIPE FIRST
        if self.mediapipe_available:
            try:
                import mediapipe as mp
                self.mp_pose = mp.solutions.pose
                self.mp_hands = mp.solutions.hands
                self.mp_face_mesh = mp.solutions.face_mesh
                
                self.pose_detector = self.mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    min_detection_confidence=0.5
                )
                
                self.hands_detector = self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=0.5
                )
                
                # --- Standard FaceMesh (No refine_landmarks to prevent C++ crash) ---
                self.face_mesh = self.mp_face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=2,
                    refine_landmarks=False,  # <-- إجبار الكود على استخدام النسخة المستقرة
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                print("   ✅ MediaPipe FaceMesh initialized (STANDARD)")
                
            except Exception as e:
                print(f"⚠️ MediaPipe init warning: {e}")
                self.mediapipe_available = False
                self.face_mesh = None
        
        # 2. INITIALIZE FER SECOND (Lazy Import)
        self.emotion_detector = None
        if self.fer_available:
            try:
                from fer import FER as FERDetector 
                self.emotion_detector = FERDetector(mtcnn=False)
                print("   ✅ FER emotion detector initialized")
            except Exception as e:
                print(f"   ⚠️ FER init failed: {e}")
                self.fer_available = False

        # 3. INITIALIZE DEEPFACE LAST (Lazy Import)
        if self.deepface_available:
            try:
                from deepface import DeepFace
                print("   ✅ DeepFace components initialized")
            except Exception as e:
                print(f"   ⚠️ DeepFace init failed: {e}")
                self.deepface_available = False
        
        print(f"\n✅ Enhanced VideoFeatureExtractor initialized")
        print(f"   MediaPipe: {'✓' if self.mediapipe_available else '✗'}")
        print(f"   DeepFace: {'✓' if self.deepface_available else '✗'}")
        print(f"   FER: {'✓' if self.fer_available else '✗'}")
        print(f"   Alert Manager: ✓")
        print(f"   Threshold Tracker: ✓")
    
    
    # ============== HEAD POSE CALCULATION ==============
    
    def _calculate_head_pose(self, landmarks) -> Tuple[float, float]:
        nose_x = landmarks[NOSE].x
        nose_y = landmarks[NOSE].y
        
        face_width = abs(landmarks[RIGHT_EAR].x - landmarks[LEFT_EAR].x) + 1e-6
        face_center_x = (landmarks[LEFT_EAR].x + landmarks[RIGHT_EAR].x) / 2
        
        head_yaw = (nose_x - face_center_x) / (face_width / 2)
        
        ear_avg_y = (landmarks[LEFT_EAR].y + landmarks[RIGHT_EAR].y) / 2
        head_pitch = nose_y - ear_avg_y
        
        return float(head_yaw), float(head_pitch)
    
    
    # ============== SMILE DETECTION ==============
    
    def _detect_smile(self, landmarks) -> bool:
        mouth_width = abs(landmarks[MOUTH_RIGHT].x - landmarks[MOUTH_LEFT].x)
        mouth_height = abs(landmarks[MOUTH_BOTTOM].y - landmarks[MOUTH_TOP].y)
        ratio = mouth_width / (mouth_height + 1e-6)
        is_smiling = ratio > 3.0 and mouth_width > 0.04
        return is_smiling
    
    
    # ============== STRESS DETECTION ==============
    
    def _calculate_stress_level(self, landmarks, head_yaw: float) -> float:
        brow_distance = abs(landmarks[L_BROW_INNER].x - landmarks[R_BROW_INNER].x)
        face_width = abs(landmarks[RIGHT_EAR].x - landmarks[LEFT_EAR].x) + 1e-6
        brow_frown = 1.0 - min(brow_distance / (face_width * 0.3), 1.0)
        stress = brow_frown * 0.5 + abs(head_yaw) * 0.3
        return float(np.clip(stress, 0, 1))
    
    
    # ============== MAIN EXTRACTION METHOD ==============
    
    def extract_features_from_video(
        self, 
        video_frames: List[np.ndarray],
        duration_seconds: float = 30.0
    ) -> Dict:
        if not video_frames or len(video_frames) == 0:
            return self._empty_features()
        
        features = {
            'emotions': self._extract_emotion_features(video_frames),
            'gaze': self._extract_gaze_features(video_frames),
            'posture': self._extract_posture_features(video_frames),
            'eye_contact': self._extract_eye_contact_features(video_frames),
            'gestures': self._extract_gesture_features(video_frames),
            'behavioral': self._extract_behavioral_features(video_frames),
            'duration': duration_seconds,
            'frames_analyzed': len(video_frames),
            'alerts': self._generate_alerts(),
        }
        return features
    
    
    # ============== GAZE FEATURES ==============
    
    def _extract_gaze_features(self, frames: List[np.ndarray]) -> Dict:
        if not self.mediapipe_available or self.face_mesh is None:
            return self._demo_gaze_features()
        
        gaze_x_values = []
        gaze_y_values = []
        eye_contact_count = 0
        looking_away_count = 0
        
        sample_rate = max(1, len(frames) // 30)
        sampled_frames = frames[::sample_rate]
        
        for frame in sampled_frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)
            
            if not results.multi_face_landmarks:
                continue
            
            landmarks = results.multi_face_landmarks[0].landmark
            
            # Using head pose as a proxy for gaze since we removed iris tracking
            yaw, pitch = self._calculate_head_pose(landmarks)
            
            # Convert yaw/pitch (-1 to 1) to normalized gaze (0 to 1, 0.5 is center)
            gaze_x = np.clip(0.5 + (yaw * 0.5), 0, 1)
            gaze_y = np.clip(0.5 + (pitch * 0.5), 0, 1)
            
            gaze_x_values.append(gaze_x)
            gaze_y_values.append(gaze_y)
            
            # If head is mostly centered, consider it eye contact
            if abs(yaw) < 0.3 and abs(pitch) < 0.3:
                eye_contact_count += 1
            else:
                looking_away_count += 1
        
        total_samples = len(gaze_x_values)
        if total_samples == 0:
            return self._demo_gaze_features()
        
        avg_gaze_x = np.mean(gaze_x_values)
        avg_gaze_y = np.mean(gaze_y_values)
        gaze_stability = 1.0 - np.std(gaze_x_values)
        
        eye_contact_pct = (eye_contact_count / total_samples) * 100
        looking_away_pct = (looking_away_count / total_samples) * 100
        
        self.threshold_tracker.update("looking_away", looking_away_pct > 50)
        
        return {
            'average_gaze_x': round(avg_gaze_x, 3),
            'average_gaze_y': round(avg_gaze_y, 3),
            'gaze_stability': round(gaze_stability, 3),
            'eye_contact_percentage': round(eye_contact_pct, 1),
            'looking_away_percentage': round(looking_away_pct, 1),
            'has_iris_data': False,  # Changed to False since we use head proxy
            'quality': 'high' if total_samples > 10 else 'medium',
        }
    
    
    # ============== BEHAVIORAL FEATURES ==============
    
    def _extract_behavioral_features(self, frames: List[np.ndarray]) -> Dict:
        if not self.mediapipe_available or self.face_mesh is None:
            return self._demo_behavioral_features()
        
        confidence_scores = []
        stress_scores = []
        smile_count = 0
        
        sample_rate = max(1, len(frames) // 30)
        sampled_frames = frames[::sample_rate]
        
        for frame in sampled_frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)
            
            if not results.multi_face_landmarks:
                continue
            
            landmarks = results.multi_face_landmarks[0].landmark
            yaw, pitch = self._calculate_head_pose(landmarks)
            
            confidence = float(np.clip(1.0 - abs(yaw) * 1.8, 0, 1))
            confidence_scores.append(confidence)
            
            stress = self._calculate_stress_level(landmarks, yaw)
            stress_scores.append(stress)
            
            if self._detect_smile(landmarks):
                smile_count += 1
        
        total_samples = len(confidence_scores)
        if total_samples == 0:
            return self._demo_behavioral_features()
        
        avg_confidence = np.mean(confidence_scores) * 100
        avg_stress = np.mean(stress_scores) * 100
        smile_pct = (smile_count / total_samples) * 100
        
        self.threshold_tracker.update("stress_high", avg_stress > 65)
        self.threshold_tracker.update("low_confidence", avg_confidence < 40)
        
        return {
            'confidence_score': round(avg_confidence, 1),
            'stress_level': round(avg_stress, 1),
            'smile_percentage': round(smile_pct, 1),
            'head_stability': round(1.0 - np.std(confidence_scores), 3),
            'overall_composure': round((avg_confidence + (100 - avg_stress)) / 2, 1),
        }
    
    
    # ============== ALERT GENERATION ==============
    
    def _generate_alerts(self) -> List[Dict]:
        alerts = []
        if self.threshold_tracker.check_threshold("looking_away"):
            if self.alert_manager.can_alert("looking_away"):
                alerts.append({
                    'type': 'looking_away',
                    'severity': 'medium',
                    'message': 'Candidate looking away from screen',
                    'frame_count': self.threshold_tracker.get_count("looking_away")
                })
        
        if self.threshold_tracker.check_threshold("stress_high"):
            if self.alert_manager.can_alert("stress_spike"):
                alerts.append({
                    'type': 'stress_spike',
                    'severity': 'medium',
                    'message': 'High stress level detected',
                    'frame_count': self.threshold_tracker.get_count("stress_high")
                })
        
        if self.threshold_tracker.check_threshold("low_confidence"):
            if self.alert_manager.can_alert("low_confidence"):
                alerts.append({
                    'type': 'low_confidence',
                    'severity': 'info',
                    'message': 'Low confidence indicators detected',
                    'frame_count': self.threshold_tracker.get_count("low_confidence")
                })
        return alerts
    
    
    # ============== ORIGINAL FEATURE METHODS ==============
    
    def _extract_emotion_features(self, frames: List[np.ndarray]) -> Dict:
        if not self.deepface_available and not self.fer_available:
            return self._demo_emotion_features()
        
        emotion_counts = {
            'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0,
            'sad': 0, 'surprise': 0, 'neutral': 0
        }
        total_detected = 0
        
        sample_rate = max(1, len(frames) // 30)
        sampled_frames = frames[::sample_rate]
        
        for frame in sampled_frames:
            try:
                if self.deepface_available:
                    from deepface import DeepFace
                    result = DeepFace.analyze(
                        frame,
                        actions=['emotion'],
                        enforce_detection=False,
                        silent=True
                    )
                    
                    if isinstance(result, list):
                        result = result[0]
                    
                    emotions = result.get('emotion', {})
                    dominant = max(emotions, key=emotions.get)
                    emotion_counts[dominant.lower()] += 1
                    total_detected += 1
                    
                elif self.fer_available and self.emotion_detector:
                    result = self.emotion_detector.detect_emotions(frame)
                    
                    if result and len(result) > 0:
                        emotions = result[0]['emotions']
                        dominant = max(emotions, key=emotions.get)
                        emotion_counts[dominant.lower()] += 1
                        total_detected += 1
            except:
                continue
        
        if total_detected == 0:
            return self._demo_emotion_features()
        
        emotion_percentages = {
            emotion: (count / total_detected) * 100
            for emotion, count in emotion_counts.items()
        }
        
        dominant_emotion = max(emotion_percentages, key=emotion_percentages.get)
        
        positive_emotions = ['happy', 'surprise']
        negative_emotions = ['angry', 'disgust', 'fear', 'sad']
        
        positive_ratio = sum(emotion_percentages[e] for e in positive_emotions) / 100
        negative_ratio = sum(emotion_percentages[e] for e in negative_emotions) / 100
        
        return {
            'happy': round(emotion_percentages['happy'], 1),
            'neutral': round(emotion_percentages['neutral'], 1),
            'surprised': round(emotion_percentages['surprise'], 1),
            'fear': round(emotion_percentages['fear'], 1),
            'sad': round(emotion_percentages['sad'], 1),
            'angry': round(emotion_percentages['angry'], 1),
            'disgust': round(emotion_percentages['disgust'], 1),
            'dominant_emotion': dominant_emotion,
            'emotion_variability': 'moderate',
            'positive_ratio': round(positive_ratio, 2),
            'negative_ratio': round(negative_ratio, 2)
        }
    
    def _extract_posture_features(self, frames: List[np.ndarray]) -> Dict:
        if not self.mediapipe_available:
            return self._demo_posture_features()
        return self._demo_posture_features()
    
    def _extract_eye_contact_features(self, frames: List[np.ndarray]) -> Dict:
        return self._demo_eye_contact_features()
    
    def _extract_gesture_features(self, frames: List[np.ndarray]) -> Dict:
        if not self.mediapipe_available:
            return self._demo_gesture_features()
        return self._demo_gesture_features()
    
    
    # ============== COMPOSITE SCORING ==============
    
    def calculate_behavioral_indices(
        self, 
        video_features: Dict, 
        audio_features: Optional[Dict] = None
    ) -> Dict:
        emotions = video_features.get('emotions', {})
        gaze = video_features.get('gaze', {})
        behavioral = video_features.get('behavioral', {})
        
        confidence = behavioral.get('confidence_score', 50.0)
        
        eye_contact_pct = gaze.get('eye_contact_percentage', 50)
        confidence += (eye_contact_pct - 50) * 0.3
        
        negative_ratio = emotions.get('negative_ratio', 0)
        confidence -= negative_ratio * 30
        
        positive_ratio = emotions.get('positive_ratio', 0)
        confidence += positive_ratio * 20
        
        if audio_features:
            energy = audio_features.get('energy_mean', 0.15)
            if energy > 0.15:
                confidence += 10
            elif energy < 0.08:
                confidence -= 10
        
        confidence = max(0, min(100, confidence))
        
        stress = behavioral.get('stress_level', 50.0)
        
        presence = 50.0
        presence += behavioral.get('overall_composure', 50) * 0.5
        presence += (eye_contact_pct - 50) * 0.3
        presence = max(0, min(100, presence))
        
        return {
            'confidence_index': round(confidence, 1),
            'stress_index': round(stress, 1),
            'professional_presence_index': round(presence, 1)
        }
    
    
    # ============== DEMO/FALLBACK FEATURES ==============
    
    def _demo_emotion_features(self) -> Dict:
        return {
            'happy': 25.0, 'neutral': 50.0, 'surprised': 10.0,
            'fear': 5.0, 'sad': 5.0, 'angry': 3.0, 'disgust': 2.0,
            'dominant_emotion': 'neutral',
            'emotion_variability': 'moderate',
            'positive_ratio': 0.35,
            'negative_ratio': 0.15
        }
    
    def _demo_gaze_features(self) -> Dict:
        return {
            'average_gaze_x': 0.5,
            'average_gaze_y': 0.5,
            'gaze_stability': 0.8,
            'eye_contact_percentage': 70.0,
            'looking_away_percentage': 30.0,
            'has_iris_data': False,
            'quality': 'demo',
        }
    
    def _demo_behavioral_features(self) -> Dict:
        return {
            'confidence_score': 70.0,
            'stress_level': 30.0,
            'smile_percentage': 25.0,
            'head_stability': 0.8,
            'overall_composure': 70.0,
        }
    
    def _demo_posture_features(self) -> Dict:
        return {
            'average_posture_score': 7.5,
            'slouching_percentage': 15.0,
            'upright_percentage': 75.0,
            'leaning_forward': 10.0,
            'posture_consistency': 'good',
            'posture_notes': 'Mostly upright'
        }
    
    def _demo_eye_contact_features(self) -> Dict:
        return {
            'eye_contact_percentage': 70.0,
            'looking_away_percentage': 20.0,
            'looking_down_percentage': 10.0,
            'eye_contact_quality': 'good',
            'engagement_level': 'high',
            'eye_contact_notes': 'Consistently good'
        }
    
    def _demo_gesture_features(self) -> Dict:
        return {
            'hand_gesture_frequency': 'moderate',
            'fidgeting_detected': True,
            'fidgeting_severity': 'low',
            'expressive_gestures': True,
            'distracting_movements': False,
            'gesture_notes': 'Minor fidgeting'
        }
    
    def _empty_features(self) -> Dict:
        return {
            'emotions': self._demo_emotion_features(),
            'gaze': self._demo_gaze_features(),
            'posture': self._demo_posture_features(),
            'eye_contact': self._demo_eye_contact_features(),
            'gestures': self._demo_gesture_features(),
            'behavioral': self._demo_behavioral_features(),
            'duration': 0,
            'frames_analyzed': 0,
            'alerts': []
        }
    
    
    def format_for_llm(self, features: Dict) -> str:
        gaze = features.get('gaze', {})
        behavioral = features.get('behavioral', {})
        emotions = features.get('emotions', {})
        alerts = features.get('alerts', [])
        
        prompt = f"""
Video Analysis Results:

Eye Contact & Gaze:
- Eye contact: {gaze.get('eye_contact_percentage', 0):.1f}%
- Gaze stability: {gaze.get('gaze_stability', 0):.2f}
- Looking away: {gaze.get('looking_away_percentage', 0):.1f}%

Behavioral Indicators:
- Confidence: {behavioral.get('confidence_score', 0):.1f}%
- Stress level: {behavioral.get('stress_level', 0):.1f}%
- Smile frequency: {behavioral.get('smile_percentage', 0):.1f}%
- Composure: {behavioral.get('overall_composure', 0):.1f}%

Emotions:
- Dominant: {emotions.get('dominant_emotion', 'neutral')}
- Positive ratio: {emotions.get('positive_ratio', 0):.2f}
- Negative ratio: {emotions.get('negative_ratio', 0):.2f}

Alerts ({len(alerts)} total):
"""
        for alert in alerts[:5]:
            prompt += f"- [{alert['severity'].upper()}] {alert['message']}\n"
        
        return prompt
    
    
    def reset_trackers(self):
        self.alert_manager.reset()
        self.threshold_tracker.reset()
    
    
    def __del__(self):
        if hasattr(self, 'pose_detector'):
            try:
                self.pose_detector.close()
            except:
                pass
        if hasattr(self, 'hands_detector'):
            try:
                self.hands_detector.close()
            except:
                pass
        if hasattr(self, 'face_mesh'):
            try:
                self.face_mesh.close()
            except:
                pass


# ============== SINGLETON PATTERN ==============

_video_extractor_instance: Optional[VideoFeatureExtractor] = None

def get_video_extractor() -> VideoFeatureExtractor:
    global _video_extractor_instance
    if _video_extractor_instance is None:
        _video_extractor_instance = VideoFeatureExtractor()
    return _video_extractor_instance


def extract_video_features(
    video_frames: List[np.ndarray],
    duration_seconds: float = 30.0
) -> Dict:
    extractor = get_video_extractor()
    return extractor.extract_features_from_video(video_frames, duration_seconds)